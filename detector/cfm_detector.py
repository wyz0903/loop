"""
detector/cfm_detector.py — KAD: Kinematics-Aware Detector
==========================================================
轻量级多尺度运动学感知攻击检测器 + 物理引导解码器。

输入: [y_meas(3) + innov_anchored(3) + u_cmd(2)] = 8 通道, 归一化后
  - y_meas (3):          传感器测量值 (含攻击)
  - innov_anchored (3):  窗口锚定运动学残差 = y_meas - rollout(y_meas[0], u_cmd)
                         (统一替代 1-step innov + kin_res, 打破非加性攻击自洽性)
  - u_cmd (2):           控制指令
输出:
  - cls_logits (B, 8)            攻击类别 A0-A7
  - y_pred (B, 100, 3)           重建的干净传感器信号 (物理单位)

架构:
  Encoder:  MultiScaleDSConvBackbone (3块膨胀深度可分离卷积, 自适应K)
            8→32→64→96 通道, 100→50→25→12 时序
            → features (B, 12, 96)
  KinematicConsistencyBias: 零参数物理先验, 注入注意力
  Classifier: 运动学引导注意力池化 → LN → Dropout → Linear(96→8)
  Decoder:  features → ConvTranspose1d 上采样 → delta_pred (B, 100, 3)
            y_kin ← batch_kinematic_rollout(y0_phys, u_cmd_phys)  [冻结梯度]
            y_pred = y_kin + delta_pred

设计亮点:
  - 多尺度膨胀卷积: dilation=1,3,9 自适应最优运动学时间尺度 (28K 参数)
  - 运动学一致性偏置: 零参数物理先验引导注意力关注可信时间步
  - 物理引导解码: 只学习攻击修正量, 运动学由已知模型提供
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ============================================================================
# 物理常量 (TurtleBot4)
# ============================================================================

TS = 0.05          # 采样周期 [s]
ALPHA = 0.17       # 前端偏移量 [m]

# ============================================================================
# 全局模型参数
# ============================================================================

# 模型架构
D_MODEL = 96
NUM_CLASSES = 8
IN_CHANNELS = 8          # [y_meas(3) + innov_anchored(3) + u_cmd(2)]
WINDOW_SIZE = 100

# 多尺度深度可分离卷积骨干
CONV_CHANNELS = [32, 64, 96]       # 逐块输出通道 (3块)
CONV_KERNEL_SIZES = [7, 5, 3]      # 逐块卷积核大小
CONV_DILATIONS = [1, 3, 9]         # 3 个并行膨胀率 (自适应 K)
POOL_SIZE = 2                      # 池化核大小 (时序降采样)

# Transformer 骨干参数 (向后兼容)
NUM_HEADS = 8
NUM_TRANSFORMER_LAYERS = 4
DIM_FEEDFORWARD = 512

# 物理引导解码器
DECODER_CHANNELS = [64, 32, 16]    # 上采样逐层通道
USE_DECODER = True

# 运动学一致性偏置
KIN_BIAS_SIGMA = 0.5               # 创新异常阈值 [m]


# ============================================================================
# 工具函数: 批量运动学 rollout
# ============================================================================

def batch_kinematic_rollout(y0: torch.Tensor, u_seq: torch.Tensor,
                            Ts: float = TS, alpha: float = ALPHA
                            ) -> torch.Tensor:
    """欧拉积分运动学递推 (与 WMRKinematics.kinematic_predict 一致)。

    从初始状态 y0 出发, 沿控制序列 u_seq 递推。
    使用欧拉积分以匹配内部运动学模型, 截断误差 O(Ts²) ≈ 0.0025/步。

    Args:
        y0:    (B, 3)   初始状态 [x, y, theta], 物理单位
        u_seq: (B, W, 2) 控制序列 [v, w], 物理单位
        Ts:    采样周期 [s]
        alpha: 前端偏移量 [m]

    Returns:
        y_kin: (B, W, 3) 运动学递推轨迹, 物理单位
    """
    B, W, _ = u_seq.shape
    device = u_seq.device
    dtype = u_seq.dtype
    y_kin = torch.zeros(B, W, 3, device=device, dtype=dtype)
    y = y0.to(device=device, dtype=dtype)
    y_kin[:, 0, :] = y

    for k in range(W - 1):
        v = u_seq[:, k, 0]
        w = u_seq[:, k, 1]
        cos_t = torch.cos(y[:, 2])
        sin_t = torch.sin(y[:, 2])
        dx = v * cos_t - alpha * w * sin_t
        dy = v * sin_t + alpha * w * cos_t
        dtheta = w
        y = y + Ts * torch.stack([dx, dy, dtheta], dim=-1)
        y_kin[:, k + 1, :] = y

    return y_kin


# ============================================================================
# 1. Transformer 骨干 (向后兼容)
# ============================================================================

class TransformerBackbone(nn.Module):
    """统一 Transformer 编码器主干 (向后兼容旧版配置)。"""

    def __init__(self, in_channels: int = IN_CHANNELS, window_size: int = WINDOW_SIZE,
                 d_model: int = D_MODEL, num_layers: int = NUM_TRANSFORMER_LAYERS,
                 num_heads: int = NUM_HEADS, d_ff: int = DIM_FEEDFORWARD,
                 dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.pos_embed[:, :h.shape[1], :]
        return self.encoder(h)


# ============================================================================
# 2. 多尺度膨胀深度可分离卷积骨干 (KAD 核心)
# ============================================================================

class MultiScaleDSConvBlock(nn.Module):
    """多尺度膨胀深度可分离卷积块。

    3 个并行的膨胀深度卷积 (dilation=1,3,9) 捕获不同时间尺度的特征:
      - d=1 (感受野 7 步 = 0.35s): 快速变化 (类 1-step innov)
      - d=3 (感受野 19 步 = 0.95s): 中等尺度动态
      - d=9 (感受野 55 步 = 2.75s): 慢速累积 (类 window-anchored kin_res)

    Pointwise Conv 学习三尺度间的最优加权 — 自适应 K。
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 7,
                 dilations: list = None,
                 pool_size: int = 2):
        super().__init__()
        if dilations is None:
            dilations = CONV_DILATIONS

        # 3 个并行膨胀深度卷积 (groups=in_channels: 逐通道独立滤波)
        self.depthwise_convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size,
                      padding=(kernel_size // 2) * d,
                      dilation=d, groups=in_channels, bias=False)
            for d in dilations
        ])

        # Pointwise 卷积: 混合多尺度特征 (3*C_in → C_out)
        self.pointwise = nn.Conv1d(in_channels * len(dilations), out_channels,
                                   kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, W)
        # 并行多尺度深度卷积
        multi_scale = [dw(x) for dw in self.depthwise_convs]  # [(B, C_in, W)] × 3
        h = torch.cat(multi_scale, dim=1)                      # (B, 3*C_in, W)

        # Pointwise 混合 + BN + GELU + Pool
        h = self.pointwise(h)                                   # (B, C_out, W)
        h = F.gelu(self.bn(h))
        h = self.pool(h)                                        # (B, C_out, W//2)
        return h


class MultiScaleDSConvBackbone(nn.Module):
    """多尺度深度可分离卷积骨干网络。

    3 个 MultiScaleDSConvBlock, 逐块通道扩增 + 时序降采样。

    通道变化: 8 → 32 → 64 → 96
    时序变化: 100 → 50 → 25 → 12

    输入: (B, W, C)
    输出: (B, W', d_model)  其中 W' = W // 8, d_model = 96
    """

    def __init__(self, in_channels: int = IN_CHANNELS,
                 channels: list = None,
                 kernel_sizes: list = None,
                 dilations: list = None,
                 pool_size: int = POOL_SIZE):
        super().__init__()
        if channels is None:
            channels = CONV_CHANNELS
        if kernel_sizes is None:
            kernel_sizes = CONV_KERNEL_SIZES
        if dilations is None:
            dilations = CONV_DILATIONS
        self.d_model = channels[-1]

        blocks = []
        ci = in_channels
        for i, co in enumerate(channels):
            blocks.append(MultiScaleDSConvBlock(
                in_channels=ci, out_channels=co,
                kernel_size=kernel_sizes[i],
                dilations=dilations,
                pool_size=pool_size,
            ))
            ci = co
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, C) → (B, C, W) for Conv1d
        h = x.permute(0, 2, 1)
        h = self.blocks(h)          # (B, C, W) → (B, 96, 12)
        h = h.permute(0, 2, 1)     # (B, 12, 96)
        h = self.norm_out(h)
        return h


# ============================================================================
# 3. 运动学一致性偏置 (零参数物理先验)
# ============================================================================

class KinematicConsistencyBias(nn.Module):
    """零参数物理先验: 从窗口锚定创新计算运动学一致性偏置。

    不参与梯度。将 innov_anchored 的 L2 范数映射为注意力偏置:
      bias[j] = -‖innov_anchored[t_j]‖ / σ

    物理解释:
      innov_anchored[t] = y_meas[t] - rollout(y_meas[0], u_cmd[0:t])[t]
      若测量与运动学预测一致 → innov ≈ 0 → bias ≈ 0 (中性)
      若测量偏离运动学预测 → innov 大 → bias < 0 (抑制该步注意力)

    攻击类型偏置特征 (可视化素材):
      A0 正常:  所有步 bias ≈ 0 (运动学一致)
      A4 重放:  重放起始处 bias 突降 (轨迹跳变)
      A5 丢包:  丢包步 bias 大幅为负 (零测量 vs 递推预测)
      A7 冻结:  冻结后 bias 单调下降 (冻结值无法预测后续运动)
    """

    def __init__(self, sigma: float = KIN_BIAS_SIGMA):
        super().__init__()
        self.sigma = sigma

    def forward(self, x_norm: torch.Tensor,
                ymeas_scale: torch.Tensor) -> torch.Tensor:
        """计算运动学一致性偏置。

        Args:
            x_norm:      (B, W, 8) 归一化输入, 通道 3:5 = innov_anchored
            ymeas_scale: (3,) y_meas 物理锚点尺度

        Returns:
            bias: (B, 12) 每输出时间步的运动学一致性偏置
        """
        with torch.no_grad():
            # innov_anchored 在预处理中用 ymeas_scale 归一化
            innov_norm = x_norm[:, :, 3:6]                      # (B, W, 3)
            innov_phys = innov_norm * ymeas_scale.view(1, 1, 3)  # 反归一化到物理单位

            # 下采样到 12 个输出时间步 (每块中心)
            t_idx = [4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92]
            innov_at_t = innov_phys[:, t_idx, :]                 # (B, 12, 3)

            # L2 范数 → 偏置 (高创新 → 低一致性 → 负偏置)
            innov_l2 = torch.norm(innov_at_t, dim=-1)            # (B, 12)
            bias = -innov_l2 / self.sigma

            # 去中心 (保持注意力 softmax 的数值稳定性)
            bias = bias - bias.mean(dim=-1, keepdim=True)

        return bias


# ============================================================================
# 4. 物理引导解码器
# ============================================================================

class PhysicsGuidedDecoder(nn.Module):
    """物理引导解码器: 从编码器特征预测攻击修正量, 重建干净传感器信号。

    设计原则:
      - 不学习运动学 (由 batch_kinematic_rollout 提供, 冻结梯度)
      - 只预测 delta = y_clean - y_kin (攻击导致的偏差)
      - 轻量 ConvTranspose1d 上采样: 12→24→48→96→100

    输入: features (B, 12, d_model)  编码器输出
    输出: delta_pred (B, 100, 3)     攻击修正量 (物理单位)
    """

    def __init__(self, d_model: int = D_MODEL,
                 dec_channels: list = None,
                 out_channels: int = 3):
        super().__init__()
        if dec_channels is None:
            dec_channels = DECODER_CHANNELS

        layers = []
        ci = d_model
        for co in dec_channels:
            layers.extend([
                nn.ConvTranspose1d(ci, co, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(co),
                nn.GELU(),
            ])
            ci = co
        layers.append(nn.Conv1d(ci, out_channels, kernel_size=5, padding=2))
        self.upsample = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = features.permute(0, 2, 1)           # (B, d_model, 12)
        h = self.upsample(h)                     # (B, 3, 96)
        h = F.interpolate(h, size=100, mode='linear', align_corners=False)
        h = h.permute(0, 2, 1)                  # (B, 100, 3)
        return h


# ============================================================================
# 5. KAD 分类检测器 (编码器 + 运动学偏置注意力 + 解码器)
# ============================================================================

class CFMDetector(nn.Module):
    """KAD: Kinematics-Aware Detector。

    架构:
      Encoder:   x (B,100,8)
                 → MultiScaleDSConvBackbone (11→32→64→96)
                 → features (B, 12, 96)
      Bias:      KinematicConsistencyBias (零参数物理先验)
      Classifier: 运动学引导注意力池化 → LN → Dropout → Linear(96,8)
                 scores = features @ query / √d + α · bias
      Decoder:    features → PhysicsGuidedDecoder → delta_pred (B,100,3)
                 y_kin ← batch_kinematic_rollout(y0_phys, u_cmd_phys)
                 y_pred = y_kin + delta_pred

    输入:  [y_meas(3) + innov_anchored(3) + u_cmd(2)] = 8 通道, 归一化后
    输出:  cls_logits (B, 8), y_pred (B, 100, 3), delta_pred (B, 100, 3)
    """

    def __init__(self,
                 in_channels: int = IN_CHANNELS,
                 window_size: int = WINDOW_SIZE,
                 d_model: int = D_MODEL,
                 num_classes: int = NUM_CLASSES,
                 backbone_type: str = 'simple_conv',
                 conv_channels: list = None,
                 conv_kernel_size: int = 3,
                 pool_size: int = POOL_SIZE,
                 use_channel_attn: bool = False,
                 channel_attn_heads: int = 4,
                 channel_attn_dim: int = 64,
                 num_transformer_layers: int = NUM_TRANSFORMER_LAYERS,
                 num_heads: int = NUM_HEADS,
                 dim_feedforward: int = DIM_FEEDFORWARD,
                 dropout: float = 0.2,
                 use_decoder: bool = USE_DECODER,
                 dec_channels: list = None,
                 # 归一化参数 (注册为 buffer, 不参与训练)
                 ymeas_scale: list = None,
                 ymeas_median: list = None,
                 cmd_max: list = None,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.backbone_type = backbone_type
        self.use_decoder = use_decoder

        # ---- 归一化参数 (buffer: 持久保存但不参与梯度) ----
        _ymeas_scale = torch.tensor([2.5, 2.5, 3.141592653589793],
                                     dtype=torch.float32)
        _ymeas_median = torch.zeros(3, dtype=torch.float32)
        _cmd_max = torch.tensor([0.3, 1.76], dtype=torch.float32)
        if ymeas_scale is not None:
            _ymeas_scale = torch.tensor(ymeas_scale, dtype=torch.float32)
        if ymeas_median is not None:
            _ymeas_median = torch.tensor(ymeas_median, dtype=torch.float32)
        if cmd_max is not None:
            _cmd_max = torch.tensor(cmd_max, dtype=torch.float32)
        self.register_buffer('ymeas_scale', _ymeas_scale)
        self.register_buffer('ymeas_median', _ymeas_median)
        self.register_buffer('cmd_max', _cmd_max)

        # ---- 骨干网络 ----
        if backbone_type == 'simple_conv':
            # KAD 骨干: 多尺度膨胀深度可分离卷积
            self.backbone = MultiScaleDSConvBackbone(
                in_channels=in_channels,
                channels=conv_channels if conv_channels else CONV_CHANNELS,
                kernel_sizes=CONV_KERNEL_SIZES,
                dilations=CONV_DILATIONS,
                pool_size=pool_size,
            )
        elif backbone_type == 'transformer':
            self.backbone = TransformerBackbone(
                in_channels=in_channels, window_size=window_size,
                d_model=d_model, num_layers=num_transformer_layers,
                num_heads=num_heads, d_ff=dim_feedforward, dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # ---- 运动学一致性偏置 (零参数物理先验) ----
        self.kin_bias = KinematicConsistencyBias(sigma=KIN_BIAS_SIGMA)

        # ---- 可学习偏置强度 ----
        self.bias_alpha = nn.Parameter(torch.tensor(0.1))

        # ---- 分类头 (运动学引导注意力池化) ----
        self.cls_norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, num_classes)
        self.cls_dropout = nn.Dropout(0.3)
        self.attn_query = nn.Parameter(torch.randn(d_model) * 0.02)

        # ---- 物理引导解码器 ----
        if use_decoder:
            self.decoder = PhysicsGuidedDecoder(
                d_model=d_model,
                dec_channels=dec_channels,
                out_channels=3,
            )
        else:
            self.decoder = None

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.cls_head.weight)
        nn.init.zeros_(self.cls_head.bias)

    # ------------------------------------------------------------------
    # 归一化参数设置
    # ------------------------------------------------------------------

    def set_norm_params(self, ymeas_scale, ymeas_median, cmd_max):
        """设置归一化参数 buffer (训练/推理时从 normalizer 加载后调用)。"""
        self.ymeas_scale.copy_(torch.as_tensor(ymeas_scale, dtype=torch.float32))
        self.ymeas_median.copy_(torch.as_tensor(ymeas_median, dtype=torch.float32))
        self.cmd_max.copy_(torch.as_tensor(cmd_max, dtype=torch.float32))

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码输入窗口 → 特征序列。

        Args: x: (B, W, 8) 归一化输入。Returns: features (B, 12, d_model)
        """
        return self.backbone(x)

    def classify(self, features: torch.Tensor,
                 x_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """运动学引导注意力池化分类。

        在标准注意力分数上叠加运动学一致性偏置:
          scores[i] = features[i] · query / √d  +  α · kin_bias[i]
        测量与运动学一致的步获得更高偏置 → 更多注意力。

        Args:
            features: (B, 12, d_model) 编码器特征
            x_norm:   (B, 100, 8) 归一化输入 (用于计算运动学偏置)

        Returns:
            cls_logits: (B, num_classes)
        """
        d = features.shape[-1]

        # 标准注意力分数
        scores = torch.matmul(features, self.attn_query) / (d ** 0.5)  # (B, 12)

        # 注入运动学一致性偏置 (零参数物理先验)
        if x_norm is not None:
            kin_bias = self.kin_bias(x_norm, self.ymeas_scale)          # (B, 12)
            scores = scores + self.bias_alpha * kin_bias

        attn_weights = torch.softmax(scores, dim=1)                     # (B, 12)
        pooled = (features * attn_weights.unsqueeze(-1)).sum(dim=1)     # (B, d_model)
        pooled = self.cls_norm(pooled)
        pooled = self.cls_dropout(pooled)
        return self.cls_head(pooled)

    def decode(self, features: torch.Tensor, x_norm: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """物理引导解码: 运动学递推 + 学习修正 → 重建干净信号。

        Args:
            features: (B, 12, d_model) 编码器特征
            x_norm:   (B, 100, 8) 归一化输入 (用于提取 y0 和 u_cmd)

        Returns:
            y_pred:     (B, 100, 3) 重建的干净传感器信号 (物理单位)
            delta_pred: (B, 100, 3) 攻击修正量 (物理单位)
        """
        if self.decoder is None:
            raise RuntimeError("Decoder not enabled. Set use_decoder=True.")

        # 1. 从归一化输入反归一化得到物理量
        # 通道 0:2 = y_meas, 通道 6:7 = u_cmd
        y0_norm = x_norm[:, 0, :3]                     # (B, 3) 首步 y_meas
        u_norm = x_norm[:, :, -2:]                      # (B, W, 2) u_cmd
        y0_phys = y0_norm * self.ymeas_scale + self.ymeas_median
        u_phys = u_norm * self.cmd_max

        # 2. 运动学递推 (冻结梯度 — 已知物理, 不学习)
        with torch.no_grad():
            y_kin = batch_kinematic_rollout(y0_phys, u_phys)  # (B, W, 3)

        # 3. 解码器预测攻击修正量
        delta_pred = self.decoder(features)                  # (B, W, 3)

        # 4. 重建干净信号
        y_pred = y_kin + delta_pred                          # (B, W, 3)

        return y_pred, delta_pred

    def forward(self, x: torch.Tensor,
                return_recon: bool = False
                ) -> Tuple[torch.Tensor, ...]:
        """单次前向: 编码 → 分类 [+ 解码]。

        Args:
            x:            (B, W, 8) 归一化输入窗口
            return_recon: 是否返回重建信号

        Returns:
            return_recon=False: (cls_logits, features)
            return_recon=True:  (cls_logits, features, y_pred, delta_pred)
        """
        features = self.encode(x)
        cls_logits = self.classify(features, x)

        if return_recon and self.decoder is not None:
            y_pred, delta_pred = self.decode(features, x)
            return cls_logits, features, y_pred, delta_pred

        return cls_logits, features
