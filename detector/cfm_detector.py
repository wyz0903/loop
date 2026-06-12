"""
detector/cfm_detector.py — PINN-Flow: 物理信息流匹配攻击检测器
===============================================================
统一架构: 因果空洞卷积主干 + 正交特征子空间 + 流匹配生成 + 物理正则化。

核心洞察: 已知精确的运动学 ODE dX/dt = F_h(θ)·u (α=0.17m, Ts=0.05s),
将其作为流匹配生成过程的物理约束, 而非为每种攻击类型单独设计模块。

架构 (v2):
  CausalDilatedConvBackbone (6层, d_model=128, dilations=[1,2,4,8,16,32])
    → features (B,100,128)
    → OrthogonalFeatureSplitter → cls_features (B,100,64) ⊥ fm_features (B,100,64)
      ├── ClassificationHead: mean→LN→Linear(64→9)
      └── FlowMatchingHead: AdaLN-Zero×4 + SinusoidalTimeEmbedding
          → v_θ(t, x_t, cond) velocity field
          → ODE Solver (Euler 10步) → â (B,100,3)

物理正则化 (PINN):
  L_phys = max(0, mean(||r_phys||²) − κ·Tr(R))
  r_phys[k] = y_rec[k+1] − kinematic_step(y_rec[k], u_cmd[k])
  Tr(R) = 0.018 (measurement noise covariance trace)

正交正则化:
  L_ortho = ||W_cls @ W_fm^T||_F^2  (分类/流匹配子空间正交约束)

输入: (B, W, 5)  [internal_innovation(3) + u_cmd(2)]
输出: cls_logits (B, 9), attack_seq (B, W, 3), z (B, d_model)

参数量: ~0.76M
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

# 正交特征子空间
D_CLS = 64              # 分类子空间维度
D_FM = 64               # 流匹配子空间维度
LAMBDA_ORTHO = 0.01     # 正交正则化权重

# 因果空洞卷积
DILATIONS = [1, 2, 4, 8, 16, 32]  # 膨胀因子序列 (RF=253)
CONV_KERNEL_SIZE = 3               # 因果卷积核大小

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
# 4. 因果空洞卷积主干 (TCN 风格) —— 替代 Transformer 自注意力
# ============================================================================

class CausalConv1d(nn.Module):
    """因果 Conv1d 包装器: 仅左侧填充, 保证时刻 t 的输出只依赖于 ≤t 的输入。

    对 kernel_size=3, dilation=d:
      输出 y[t] 仅依赖于 x[t], x[t-d], x[t-2d] (均为过去/当前)
      无未来泄漏。
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=0)
        self.pad_left = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, W)。Returns: (B, C_out, W) — 同长度。"""
        x = F.pad(x, (self.pad_left, 0))  # 仅左侧填充
        return self.conv(x)


class CausalDilatedConvBlock(nn.Module):
    """因果空洞卷积残差块。

    结构: CausalConv → GELU → CausalConv → Dropout → residual add
    同一膨胀率 d 应用于块内两个卷积, 实现特定尺度的时序建模。
    """

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化 (fan_out 模式, 适配残差结构)。"""
        for conv in [self.conv1.conv, self.conv2.conv]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out',
                                     nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, W)。Returns: (B, C, W) — 残差连接。"""
        h = F.gelu(self.conv1(x))
        h = self.conv2(h)
        h = self.dropout(h)
        return x + h


class CausalDilatedConvBackbone(nn.Module):
    """因果空洞卷积骨干网络 (TCN 风格)。

    以指数膨胀空洞卷积捕获多尺度时序依赖, 因果约束保证实时性。
    无位置编码: 膨胀层次结构天然编码时序位置。

    输入: (B, W, C)  → 内部排列为 Conv1d 格式 (B, C, W)
    输出: (B, W, d_model)

    感受野 (kernel_size=3, dilations=[1,2,4,8,16,32]):
      RF = 1 + 2*(k-1)*sum(dilations) = 1 + 4*63 = 253 > 100 ✓
    """

    def __init__(self, in_channels: int = IN_CHANNELS, d_model: int = D_MODEL,
                 dilations: list = None, kernel_size: int = CONV_KERNEL_SIZE,
                 dropout: float = DROPOUT):
        super().__init__()
        if dilations is None:
            dilations = DILATIONS
        self.d_model = d_model
        self.dilations = list(dilations)

        # 输入投影: 5 → 128, 逐点卷积
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)

        # 膨胀因果残差块堆叠
        self.blocks = nn.ModuleList([
            CausalDilatedConvBlock(d_model, d, kernel_size, dropout)
            for d in self.dilations
        ])

        # 输出归一化
        self.norm_out = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, W, C)。Returns: features (B, W, d_model)。"""
        # (B, W, C) → (B, C, W)  Conv1d 原生格式
        h = x.permute(0, 2, 1)
        h = self.input_proj(h)  # (B, d_model, W)

        for block in self.blocks:
            h = block(h)  # (B, d_model, W)

        # (B, d_model, W) → (B, W, d_model)
        h = h.permute(0, 2, 1)
        h = self.norm_out(h)
        return h


# ============================================================================
# 5. 正交特征子空间投影器
# ============================================================================

class OrthogonalFeatureSplitter(nn.Module):
    """将骨干特征投影到两个正交子空间中。

    分类头和流匹配头分别在独立的正交子空间中运行:
      cls_features = features @ W_cls^T   (子空间 C, d_cls 维)
      fm_features  = features @ W_fm^T    (子空间 F, d_fm 维)
      其中 W_cls @ W_fm^T ≈ 0  (行正交)

    正交初始化通过 QR 分解保证严格初始正交性,
    训练中通过 L_ortho = ||W_cls @ W_fm^T||_F^2 维持正交约束。
    """

    def __init__(self, d_model: int = D_MODEL, d_cls: int = D_CLS,
                 d_fm: int = D_FM):
        super().__init__()
        self.d_model = d_model
        self.d_cls = d_cls
        self.d_fm = d_fm

        # 无偏置投影矩阵 (正交性约束要求纯线性投影)
        self.proj_cls = nn.Linear(d_model, d_cls, bias=False)
        self.proj_fm = nn.Linear(d_model, d_fm, bias=False)

        self._init_orthogonal()

    def _init_orthogonal(self):
        """从随机正交矩阵 Q 取行初始化, 保证 W_cls ⟂ W_fm。"""
        M = torch.randn(self.d_model, self.d_model)
        Q, _ = torch.linalg.qr(M)  # Q ∈ R^{d_model×d_model}, 正交列

        with torch.no_grad():
            # proj_cls.weight ∈ R^{d_cls×d_model} = Q 的前 d_cls 行
            self.proj_cls.weight.copy_(Q[:self.d_cls, :])
            # proj_fm.weight ∈ R^{d_fm×d_model} = Q 的接下来 d_fm 行
            self.proj_fm.weight.copy_(Q[self.d_cls:self.d_cls + self.d_fm, :])

    def forward(self, features: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: features (B, W, d_model)。
        Returns: cls_features (B, W, d_cls), fm_features (B, W, d_fm)。
        """
        return self.proj_cls(features), self.proj_fm(features)

    def compute_ortho_loss(self) -> torch.Tensor:
        """正交正则化损失: L_ortho = ||W_cls @ W_fm^T||_F^2。"""
        W_cls = self.proj_cls.weight  # (d_cls, d_model)
        W_fm = self.proj_fm.weight    # (d_fm, d_model)
        cross = W_cls @ W_fm.T        # (d_cls, d_fm)
        return (cross ** 2).sum()


# ============================================================================
# 6. 流匹配头
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
# 7. 物理运动学工具 (用于 PINN 损失)
# ============================================================================

def _kinematics_rhs(state: torch.Tensor, u_cmd: torch.Tensor) -> torch.Tensor:
    """WMR 前端位姿运动学右手边: dX/dt = F_h(θ)·u

    Args:
        state: (..., 3) 当前状态 [x, y, θ]
        u_cmd: (..., 2) 控制指令 [v, ω]

    Returns:
        dX: (..., 3) 状态导数 [dx/dt, dy/dt, dθ/dt]
    """
    v = u_cmd[..., 0]
    w = u_cmd[..., 1]
    theta = state[..., 2]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    dx = v * cos_t - ALPHA * w * sin_t
    dy = v * sin_t + ALPHA * w * cos_t
    dtheta = w
    return torch.stack([dx, dy, dtheta], dim=-1)


def kinematic_step_batch(state: torch.Tensor, u_cmd: torch.Tensor) -> torch.Tensor:
    """批量运动学 RK4 积分 — 与 model.py WMRKinematics.rk4_step() 一致。

    WMR 前端位姿运动学:
      dx/dt = v·cos(θ) − α·ω·sin(θ)
      dy/dt = v·sin(θ) + α·ω·cos(θ)
      dθ/dt = ω

    Args:
        state: (B, L, 3) 当前状态 [x, y, θ]
        u_cmd: (B, L, 2) 控制指令 [v, ω]

    Returns:
        next_state: (B, L, 3) RK4 积分下一步状态
    """
    h = TS
    k1 = _kinematics_rhs(state, u_cmd)
    k2 = _kinematics_rhs(state + 0.5 * h * k1, u_cmd)
    k3 = _kinematics_rhs(state + 0.5 * h * k2, u_cmd)
    k4 = _kinematics_rhs(state + h * k3, u_cmd)

    next_state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # 角度归一化 (与 model.py 一致)
    next_theta = torch.atan2(torch.sin(next_state[..., 2]),
                              torch.cos(next_state[..., 2]))
    next_state = torch.stack([next_state[..., 0], next_state[..., 1], next_theta], dim=-1)
    return next_state


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
    r_phys = y_next - y_pred_next                        # (B, W−1, 3)
    # 角度通道包裹: 当航向角跨越 ±π 边界时, 直接差值会产生 ~2π 的虚假残差,
    # 导致梯度爆炸。使用 atan2(sin, cos) 将角度差包裹到 [-π, π]。
    # 注意: 不可用切片赋值 (inplace), 需通过 stack 保持 autograd 图完整。
    r_theta = torch.atan2(torch.sin(r_phys[..., 2]),
                           torch.cos(r_phys[..., 2]))
    r_phys = torch.stack([r_phys[..., 0], r_phys[..., 1], r_theta], dim=-1)
    return r_phys


# ============================================================================
# 8. CFMDetector — 完整的 PINN-Flow 模型
# ============================================================================

class CFMDetector(nn.Module):
    """物理信息条件流匹配攻击检测器 (v2)。

    架构:
      x → Backbone (causal_conv 或 transformer) → features (B,W,d_model)
        → OrthogonalFeatureSplitter
          ├→ cls_features (B,W,d_cls) → ClassificationHead → cls_logits
          └→ fm_features  (B,W,d_fm)  → FlowMatchingHead  → velocity field

    关键设计:
      - 因果空洞卷积骨干: 膨胀率 [1,2,4,8,16,32], 感受野 253, 无未来泄漏
      - 正交特征子空间: 分类/流匹配在独立正交子空间中运行, 避免梯度冲突
      - 向后兼容: backbone_type='transformer', d_cls=d_fm=d_model 恢复原架构
    """

    def __init__(self,
                 in_channels: int = IN_CHANNELS,
                 window_size: int = WINDOW_SIZE,
                 d_model: int = D_MODEL,
                 num_classes: int = NUM_CLASSES,
                 # --- 骨干网络 ---
                 backbone_type: str = 'causal_conv',
                 dilations: list = None,
                 conv_kernel_size: int = CONV_KERNEL_SIZE,
                 # --- 正交子空间 ---
                 d_cls: int = D_CLS,
                 d_fm: int = D_FM,
                 # --- Transformer 参数 (向后兼容) ---
                 num_transformer_layers: int = NUM_TRANSFORMER_LAYERS,
                 num_heads: int = NUM_HEADS,
                 dim_feedforward: int = DIM_FEEDFORWARD,
                 # --- 流匹配头 ---
                 num_flow_blocks: int = NUM_FLOW_BLOCKS,
                 dim_feedforward_flow: int = DIM_FEEDFORWARD_FLOW,
                 dropout: float = DROPOUT):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.backbone_type = backbone_type

        # ---- 骨干网络 ----
        if backbone_type == 'causal_conv':
            self.backbone = CausalDilatedConvBackbone(
                in_channels=in_channels, d_model=d_model,
                dilations=dilations, kernel_size=conv_kernel_size,
                dropout=dropout,
            )
        elif backbone_type == 'transformer':
            self.backbone = TransformerBackbone(
                in_channels=in_channels, window_size=window_size,
                d_model=d_model, num_layers=num_transformer_layers,
                num_heads=num_heads, d_ff=dim_feedforward, dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # ---- 正交特征分解器 ----
        if d_cls == d_model and d_fm == d_model:
            # 无子空间分裂: 恒等通过 (向后兼容旧架构)
            self.splitter = None
        else:
            self.splitter = OrthogonalFeatureSplitter(d_model, d_cls, d_fm)

        self._d_cls = d_cls
        self._d_fm = d_fm

        # ---- 分类头 ----
        self.cls_norm = nn.LayerNorm(d_cls)
        self.cls_head = nn.Linear(d_cls, num_classes)
        self.cls_dropout = nn.Dropout(0.2)

        # ---- 流匹配头 ----
        self.flow_head = FlowMatchingHead(
            d_model=d_fm, out_channels=OUT_CHANNELS,
            num_blocks=num_flow_blocks, d_ff=dim_feedforward_flow,
        )

        self._init_cls_head()

    def _init_cls_head(self):
        nn.init.xavier_uniform_(self.cls_head.weight)
        nn.init.zeros_(self.cls_head.bias)

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码输入窗口 → 特征序列。

        Args: x: (B, W, 5) 归一化输入。Returns: features (B, W, d_model)
        """
        return self.backbone(x)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """从特征序列分类攻击类型 (内部使用分类子空间)。

        Args: features: (B, W, d_model)。Returns: cls_logits (B, num_classes)
        """
        if self.splitter is not None:
            cls_feat, _ = self.splitter(features)  # (B, W, d_cls)
        else:
            cls_feat = features  # (B, W, d_model) — 向后兼容

        pooled = cls_feat.mean(dim=1)  # (B, d_cls)
        pooled = self.cls_norm(pooled)
        pooled = self.cls_dropout(pooled)
        return self.cls_head(pooled)

    def get_fm_features(self, features: torch.Tensor) -> torch.Tensor:
        """提取流匹配子空间特征。

        Args: features: (B, W, d_model)。
        Returns: fm_features (B, W, d_fm)
        """
        if self.splitter is not None:
            _, fm_feat = self.splitter(features)
            return fm_feat
        return features  # 向后兼容: 无分裂时使用原始特征

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """标准前向传播 (评估用, teacher-force 解码)。

        注意: 训练时应使用独立的 encode() + classify() + flow_head()
              调用, 配合随机的 t 和 x_t 采样。

        Args: x: (B, W, C) 输入窗口。
        Returns: (cls_logits (B, 9), attack_seq (B, W, 3), z (B, d_model))
        """
        features = self.encode(x)
        cls_logits = self.classify(features)

        # 确定性重建: t=1, x_t=0 → 速度场直接输出
        B = x.shape[0]
        t_eval = torch.ones(B, device=x.device)
        x_t_eval = torch.zeros(B, self.window_size, OUT_CHANNELS, device=x.device)
        fm_features = self.get_fm_features(features)
        attack_seq = self.flow_head(t_eval, x_t_eval, fm_features)

        z = features.mean(dim=1)
        return cls_logits, attack_seq, z

    def compute_ortho_loss(self) -> torch.Tensor:
        """正交正则化损失: L_ortho = ||W_cls @ W_fm^T||_F^2。"""
        if self.splitter is not None:
            return self.splitter.compute_ortho_loss()
        return torch.tensor(0.0)

    @torch.no_grad()
    def sample_ode(self, x: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """通过 ODE 求解从噪声生成攻击信号估计 (推理)。

        Euler 方法: x_{t+dt} = x_t + v_θ(t, x_t, cond) · dt
        从 x_0 ~ N(0,I) 积分到 x_1 ≈ â。
        使用流匹配子空间特征作为条件。

        Args:
            x:       (B, W, 5) 归一化输入窗口
            n_steps: Euler 步数 (默认 10)

        Returns:
            a_hat:   (B, W, 3) 估计的攻击信号
        """
        features = self.encode(x)
        fm_features = self.get_fm_features(features)
        B = x.shape[0]
        device = x.device

        y_t = torch.randn(B, self.window_size, OUT_CHANNELS, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.flow_head(t, y_t, fm_features)
            y_t = y_t + v * dt

        return y_t

    def enable_mc_dropout(self):
        """启用 MC Dropout 用于推理时的不确定性估计。"""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()



