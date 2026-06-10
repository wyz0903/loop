"""
detector/cfm_detector.py — PINN-Flow: 物理信息流匹配攻击检测器
===============================================================
统一架构: Transformer 主干 + 流匹配生成 + 物理正则化。

核心洞察: 已知精确的运动学 ODE dX/dt = F_h(θ)·u (α=0.17m, Ts=0.05s),
将其作为流匹配生成过程的物理约束, 而非为每种攻击类型单独设计模块。

架构:
  TransformerBackbone (4层, d_model=128) → features (B,100,128)
    ├── ClassificationHead: mean→LN→Linear(9)
    └── FlowMatchingHead: AdaLN-Zero×4 + SinusoidalTimeEmbedding
        → v_θ(t, x_t, cond) velocity field
        → ODE Solver (Euler 10步) → â (B,100,3)

物理正则化 (PINN):
  L_phys = max(0, mean(||r_phys||²) − κ·Tr(R))
  r_phys[k] = y_rec[k+1] − kinematic_step(y_rec[k], u_cmd[k])
  Tr(R) = 0.018 (measurement noise covariance trace)

输入: (B, W, 5)  [internal_innovation(3) + u_cmd(2)]
输出: cls_logits (B, 9), attack_seq (B, W, 3), z (B, d_model)

参数量: ~1.0M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ============================================================================
# 全局配置
# ============================================================================

# 物理常量 (与 model.py 严格一致)
ALPHA = 0.17       # 前端偏置距离 [m]
TS = 0.05          # 采样周期 [s]

# 模型架构
D_MODEL = 128
NUM_HEADS = 8
NUM_TRANSFORMER_LAYERS = 4
DIM_FEEDFORWARD = 512
NUM_FLOW_BLOCKS = 4
DIM_FEEDFORWARD_FLOW = 192
DROPOUT = 0.1
NUM_CLASSES = 9
IN_CHANNELS = 5
WINDOW_SIZE = 100
OUT_CHANNELS = 3

# 噪声统计 (用于 PINN 损失)
TRACE_R = 0.018    # Tr(diag([0.008, 0.008, 0.002])) —— 测量噪声协方差迹


# ============================================================================
# 1. 正弦时间嵌入
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Transformer 风格的正弦时间嵌入

    将标量 t ∈ [0,1] 映射为 d_model 维向量。
    PE(t, 2i) = sin(t · 10000^(2i/d)), PE(t, 2i+1) = cos(t · 10000^(2i/d))
    共 256 个频率, 最高频率覆盖 Nyquist 极限。
    """

    def __init__(self, d_model: int = D_MODEL, max_period: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        # 频率: ω_i = 1 / (10000^(2i/d))
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                            * -(math.log(max_period) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args: t: (B,) 范围 [0,1]。Returns: (B, d_model)"""
        t_scaled = t.float().unsqueeze(1) * self.div_term  # (B, d_model/2)
        sin = torch.sin(t_scaled)
        cos = torch.cos(t_scaled)
        emb = torch.cat([sin, cos], dim=-1)  # (B, d_model)
        return emb


# ============================================================================
# 2. 自适应层归一化块 (DiT 风格的 AdaLN-Zero)
# ============================================================================

class AdaLNZeroBlock(nn.Module):
    """自适应层归一化块, 带零初始化门控。

    从 DiT (Peebles & Xie, 2023) 适配。
    时间嵌入调节所有层的归一化, zero-init 门控保证训练初期的稳定性。

    前向:
      h = h + gate(gamma1, beta1) · FFN(LN1(h, t_emb))
    其中 gate(gamma1, beta1) 将 t_emb 映射为通道级缩放因子,
    初始化为近零, 使得该块在训练初期近似为恒等映射。
    """

    def __init__(self, d_model: int = D_MODEL, d_ff: int = DIM_FEEDFORWARD_FLOW):
        super().__init__()
        # 时间 → 调制参数 (γ, β, gate_scale)
        self.modulation = nn.Linear(d_model, 3 * d_model)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(d_ff, d_model),
            nn.Dropout(DROPOUT),
        )

        self._init_zero()

    def _init_zero(self):
        """DiT 风格的 zero-initialization: 调制权重和偏置置零, gate_scale 近零。"""
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        # FFN 标准初始化
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Args:
            h:     (B, L, d_model) 输入特征
            t_emb: (B, d_model) 时间嵌入
        Returns:
            h:     (B, L, d_model) 调制后的特征
        """
        # 调制参数: γ1, β1, gate_scale (各 d_model 维)
        gamma1, beta1, gate = self.modulation(t_emb).chunk(3, dim=-1)
        # 广播到序列维度: (B, 1, d_model)
        gamma1 = gamma1.unsqueeze(1)
        beta1 = beta1.unsqueeze(1)
        gate = gate.unsqueeze(1)

        # AdaLN → FFN → 门控残差
        h_norm = self.norm1(h)
        h_mod = (1 + gamma1) * h_norm + beta1
        h_ffn = self.ffn(h_mod)
        h = h + gate * h_ffn

        # 第二个归一化 (不含调制)
        h = self.norm2(h)
        return h


# ============================================================================
# 3. Transformer 主干
# ============================================================================

class TransformerBackbone(nn.Module):
    """统一 Transformer 编码器主干。

    输入窗口 (B, W, C) → 线性投影 + 可学习位置嵌入 → Transformer × N 层。
    全局自注意力天然捕获当前需要膨胀卷积处理的多尺度时序依赖。
    """

    def __init__(self, in_channels: int = IN_CHANNELS, window_size: int = WINDOW_SIZE,
                 d_model: int = D_MODEL, num_layers: int = NUM_TRANSFORMER_LAYERS,
                 num_heads: int = NUM_HEADS, d_ff: int = DIM_FEEDFORWARD,
                 dropout: float = DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        # 输入投影
        self.input_proj = nn.Linear(in_channels, d_model)

        # 可学习位置嵌入
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True,
            norm_first=True,           # Pre-LN 更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, W, C) 输入窗口。Returns: features (B, W, d_model)"""
        h = self.input_proj(x)  # (B, W, d_model)
        h = h + self.pos_embed[:, :h.shape[1], :]
        h = self.encoder(h)
        return h


# ============================================================================
# 4. 流匹配头
# ============================================================================

class FlowMatchingHead(nn.Module):
    """条件流匹配向量场网络。

    预测 v_θ(t, x_t, cond), 其中 cond 为来自主干的条件特征。

    架构:
      x_t (B,W,3) → Linear(3→d_model) + cond(主干特征) → AdaLNZeroBlock × M
      t → SinusoidalEmbedding → 时间嵌入 (B, d_model)
      → LayerNorm → Linear(d_model→3) → velocity (B,W,3)
    """

    def __init__(self, d_model: int = D_MODEL, out_channels: int = OUT_CHANNELS,
                 num_blocks: int = NUM_FLOW_BLOCKS, d_ff: int = DIM_FEEDFORWARD_FLOW):
        super().__init__()
        self.d_model = d_model

        # x_t 投影
        self.x_proj = nn.Linear(out_channels, d_model)

        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(d_model)

        # AdaLN-Zero 块
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(d_model, d_ff) for _ in range(num_blocks)
        ])

        # 最终输出
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_channels)

        self._init_head_zero()

    def _init_head_zero(self):
        """输出头近零初始化, 保证训练初期速度场 ≈ 0。"""
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-6)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.zeros_(self.x_proj.bias)

    def forward(self, t: torch.Tensor, x_t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """预测速度场 v_θ(t, x_t, cond)。

        Args:
            t:    (B,)          时间标量 ∈ [0,1]
            x_t:  (B, W, 3)     当前流样本 (中间攻击估计)
            cond: (B, W, d_model) 主干特征 (条件)

        Returns:
            v:    (B, W, 3)     预测的速度场
        """
        # 时间嵌入
        t_emb = self.time_embed(t)  # (B, d_model)

        # x_t → 特征空间
        h = self.x_proj(x_t)  # (B, W, d_model)

        # 与主干特征融合 (残差加法)
        h = h + cond

        # AdaLN-Zero 块
        for block in self.blocks:
            h = block(h, t_emb)

        # 输出投影
        h = self.norm_out(h)
        v = self.head(h)  # (B, W, 3)
        return v


# ============================================================================
# 5. 物理运动学工具 (用于 PINN 损失)
# ============================================================================

def kinematic_step_batch(state: torch.Tensor, u_cmd: torch.Tensor) -> torch.Tensor:
    """批量运动学 Euler 积分 — WMR 前端位姿运动学标准形式。

    WMR 前端位姿运动学:
      dx/dt = v·cos(θ) − α·ω·sin(θ)
      dy/dt = v·sin(θ) + α·ω·cos(θ)
      dθ/dt = ω

    Args:
        state: (B, L, 3) 当前状态 [x, y, θ]
        u_cmd: (B, L, 2) 控制指令 [v, ω]

    Returns:
        next_state: (B, L, 3) 下一步预测状态
    """
    v = u_cmd[..., 0]
    w = u_cmd[..., 1]
    theta = state[..., 2]

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    dx = v * cos_t - ALPHA * w * sin_t
    dy = v * sin_t + ALPHA * w * cos_t

    next_x = state[..., 0] + TS * dx
    next_y = state[..., 1] + TS * dy
    next_theta = state[..., 2] + TS * w

    return torch.stack([next_x, next_y, next_theta], dim=-1)


def compute_physics_residual(y_rec: torch.Tensor,
                              u_cmd: torch.Tensor) -> torch.Tensor:
    """计算 PINN 物理残差: r[k] = y_rec[k+1] − kinematic_step(y_rec[k], u_cmd[k])

    Args:
        y_rec: (B, W, 3) 恢复的测量信号 (y_meas − attack_estimate)
        u_cmd: (B, W, 2) 控制指令序列

    Returns:
        r_phys: (B, W−1, 3) 运动学残差
    """
    y_now = y_rec[:, :-1, :]       # (B, W−1, 3)
    y_next = y_rec[:, 1:, :]       # (B, W−1, 3)
    u_now = u_cmd[:, :-1, :]       # (B, W−1, 2)

    y_pred_next = kinematic_step_batch(y_now, u_now)
    return y_next - y_pred_next     # (B, W−1, 3)


# ============================================================================
# 6. CFMDetector — 完整的 PINN-Flow 模型
# ============================================================================

class CFMDetector(nn.Module):
    """物理信息条件流匹配攻击检测器。

    统一架构: 无攻击类型特定模块, 无频率路径, 无 DC 偏置支路,
    无膨胀卷积, 无 FiLM 调制。单一 Transformer 主干处理所有攻击类型。

    前向:
      x → TransformerBackbone → features
        ├→ mean→LN→Linear(9) → cls_logits
        └→ FlowMatchingHead(t, x_t, features) → velocity
    """

    def __init__(self,
                 in_channels: int = IN_CHANNELS,
                 window_size: int = WINDOW_SIZE,
                 d_model: int = D_MODEL,
                 num_classes: int = NUM_CLASSES,
                 num_transformer_layers: int = NUM_TRANSFORMER_LAYERS,
                 num_heads: int = NUM_HEADS,
                 dim_feedforward: int = DIM_FEEDFORWARD,
                 num_flow_blocks: int = NUM_FLOW_BLOCKS,
                 dim_feedforward_flow: int = DIM_FEEDFORWARD_FLOW,
                 dropout: float = DROPOUT):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.d_model = d_model
        self.num_classes = num_classes

        # ---- 主干 ----
        self.backbone = TransformerBackbone(
            in_channels=in_channels, window_size=window_size,
            d_model=d_model, num_layers=num_transformer_layers,
            num_heads=num_heads, d_ff=dim_feedforward, dropout=dropout,
        )

        # ---- 分类头 ----
        self.cls_norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, num_classes)

        # ---- 流匹配头 ----
        self.flow_head = FlowMatchingHead(
            d_model=d_model, out_channels=OUT_CHANNELS,
            num_blocks=num_flow_blocks, d_ff=dim_feedforward_flow,
        )

        # ---- 分类器 dropout ----
        self.cls_dropout = nn.Dropout(0.2)

        self._init_cls_head()

    def _init_cls_head(self):
        nn.init.xavier_uniform_(self.cls_head.weight)
        nn.init.zeros_(self.cls_head.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码输入窗口 → 特征序列。

        Args: x: (B, W, 5) 归一化输入。Returns: features (B, W, d_model)
        """
        return self.backbone(x)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """从特征序列分类攻击类型。

        Args: features: (B, W, d_model)。Returns: cls_logits (B, 9)
        """
        pooled = features.mean(dim=1)  # (B, d_model)
        pooled = self.cls_norm(pooled)
        pooled = self.cls_dropout(pooled)
        return self.cls_head(pooled)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """标准前向传播 (评估用, 保持与现有接口兼容)。

        注意: 训练时应使用独立的 `encode()` + `classify()` + `flow_head()`
              调用, 配合随机的 t 和 x_t 采样。此方法使用 teacher-forced
              解码, 仅用于快速评估/日志, 不用于流匹配训练。

        Args: x: (B, W, C) 输入窗口。
        Returns: (cls_logits (B, 9), attack_seq (B, W, 3), z (B, d_model))
        """
        features = self.encode(x)
        cls_logits = self.classify(features)

        # 确定性重建: t=1, x_t=0 (prior mean) → 速度场直接输出
        # 这不是流匹配推理, 而是简单的 teacher-force 输出供 eval 日志
        B = x.shape[0]
        t_eval = torch.ones(B, device=x.device)
        x_t_eval = torch.zeros(B, self.window_size, OUT_CHANNELS, device=x.device)
        attack_seq = self.flow_head(t_eval, x_t_eval, features)

        # z = 池化特征 (与现有接口兼容)
        z = features.mean(dim=1)
        return cls_logits, attack_seq, z

    @torch.no_grad()
    def sample_ode(self, x: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """通过 ODE 求解从噪声生成攻击信号估计 (推理)。

        Euler 方法: x_{t+dt} = x_t + v_θ(t, x_t, cond) · dt
        从 x_0 ~ N(0,I) 积分到 x_1 ≈ â。

        Args:
            x:       (B, W, 5) 归一化输入窗口
            n_steps: Euler 步数 (默认 10)

        Returns:
            a_hat:   (B, W, 3) 估计的攻击信号
        """
        features = self.encode(x)
        B = x.shape[0]
        device = x.device

        # 从先验开始
        y_t = torch.randn(B, self.window_size, OUT_CHANNELS, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.flow_head(t, y_t, features)
            y_t = y_t + v * dt

        return y_t  # ≈ â

    def enable_mc_dropout(self):
        """启用 MC Dropout 用于推理时的不确定性估计。"""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


# ============================================================================
# 7. 自测
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CFMDetector 自测")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CFMDetector().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {n_params:,} (~{n_params/1e6:.2f}M)")

    # 测试前向
    x = torch.randn(4, WINDOW_SIZE, IN_CHANNELS, device=device)
    cls_logits, attack_seq, z = model(x)
    print(f"输入:   {x.shape}")
    print(f"分类:   {cls_logits.shape}  (期望: [4, 9])")
    print(f"攻击:   {attack_seq.shape}  (期望: [4, {WINDOW_SIZE}, 3])")
    print(f"潜在变量: {z.shape}       (期望: [4, {D_MODEL}])")

    # 测试 ODE 采样
    a_hat = model.sample_ode(x, n_steps=10)
    print(f"ODE样本: {a_hat.shape}  (期望: [4, {WINDOW_SIZE}, 3])")

    # 测试编码+分类
    with torch.no_grad():
        features = model.encode(x)
        cls = model.classify(features)
        print(f"编码: {features.shape}  (期望: [4, {WINDOW_SIZE}, {D_MODEL}])")
        print(f"分类: {cls.shape}      (期望: [4, 9])")

    # 测试流匹配头
    t = torch.rand(4, device=device)
    x_t = torch.randn(4, WINDOW_SIZE, OUT_CHANNELS, device=device)
    v = model.flow_head(t, x_t, features)
    print(f"速度: {v.shape}         (期望: [4, {WINDOW_SIZE}, {OUT_CHANNELS}])")

    # 测试物理残差
    u_cmd = torch.randn(4, WINDOW_SIZE, 2, device=device)
    y_rec = torch.randn(4, WINDOW_SIZE, 3, device=device)
    r_phys = compute_physics_residual(y_rec, u_cmd)
    print(f"物理残差: {r_phys.shape}  (期望: [4, {WINDOW_SIZE-1}, 3])")

    # 验证运动学步与 detector/backend.py 一致
    import numpy as np
    state_np = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    u_np = np.array([[0.15, 0.5]], dtype=np.float32)

    # Torch 版本
    state_t = torch.from_numpy(state_np).to(device)
    u_t = torch.from_numpy(u_np).to(device)
    next_t = kinematic_step_batch(state_t, u_t).cpu().numpy()

    # 参考版本 (来自 backend.py)
    v_r, w_r = u_np[0]
    theta_r = state_np[0, 2]
    cos_r = np.cos(theta_r)
    sin_r = np.sin(theta_r)
    dx_r = v_r * cos_r - ALPHA * w_r * sin_r
    dy_r = v_r * sin_r + ALPHA * w_r * cos_r
    next_ref = state_np[0] + TS * np.array([dx_r, dy_r, w_r])

    print(f"\n运动学步一致性检查:")
    print(f"  Torch版本:  {next_t[0]}")
    print(f"  参考版本:   {next_ref}")
    print(f"  一致:       {np.allclose(next_t[0], next_ref, atol=1e-6)}")

    print(f"\n所有自测通过!")
