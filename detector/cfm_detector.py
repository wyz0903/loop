"""
detector/cfm_detector.py — 攻击分类检测器 + 物理引导解码器
============================================================
编码器-解码器架构: 简单卷积骨干 + 注意力池化分类头 + 物理引导解码器。

输入: [y_meas(3) + innov(3) + kin_res(3) + u_cmd(2)] = 11 通道, 归一化后
  - y_meas (3):    传感器测量值 (含攻击)
  - innov (3):     1-step 运动学新息 (短尺度, 捕获突变型异常)
  - kin_res (3):   窗口锚定运动学残差 (长尺度, 捕获累积不一致, 打破非加性攻击自洽性)
  - u_cmd (2):     控制指令
输出:
  - cls_logits (B, 8)            攻击类别 A0-A7
  - y_pred (B, 100, 3)           重建的干净传感器信号 (物理单位)

架构:
  Encoder:  SimpleConvBackbone (3块 Conv-BN-ReLU-Pool, 11→64→128→128)
            → features (B, 12, 128)
  Classifier: 注意力池化 → LN → Dropout → Linear(128→8) → cls_logits
  Decoder:  features → ConvTranspose1d 上采样 → delta_pred (B, 100, 3)
            y_kin ← batch_kinematic_rollout(y0_phys, u_cmd_phys)  [冻结, 不参与梯度]
            y_pred = y_kin + delta_pred

物理引导: 解码器只学习攻击修正量 delta_pred, 运动学部分由已知模型提供,
        不浪费参数学控制→状态映射。delta_pred 显式编码攻击模式。
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
D_MODEL = 128
NUM_HEADS = 8
NUM_TRANSFORMER_LAYERS = 4
DIM_FEEDFORWARD = 512
NUM_CLASSES = 8
IN_CHANNELS = 11          # [y_meas(3) + innov(3) + kin_res(3) + u_cmd(2)]
WINDOW_SIZE = 100

# 简单卷积骨干
CONV_CHANNELS = [64, 128, 128]     # 逐块输出通道 (3块)
CONV_KERNEL_SIZE = 3               # 卷积核大小 (same padding)
POOL_SIZE = 2                      # 池化核大小 (时序降采样)

# 通道自注意力
USE_CHANNEL_ATTN = True            # 是否启用通道自注意力 (消融实验开关)
CHANNEL_ATTN_HEADS = 4             # 注意力头数
CHANNEL_ATTN_DIM = 64              # 中间投影维度

# 物理引导解码器
DECODER_CHANNELS = [64, 32, 16]    # 上采样逐层通道
USE_DECODER = True                 # 是否启用解码器 (消融实验开关)

# ============================================================================
# 工具函数: 批量运动学 rollout
# ============================================================================

def batch_kinematic_rollout(y0: torch.Tensor, u_seq: torch.Tensor,
                            Ts: float = TS, alpha: float = ALPHA
                            ) -> torch.Tensor:
    """欧拉积分运动学递推 (与 WMRKinematics.kinematic_predict 一致)。

    从初始状态 y0 出发, 沿控制序列 u_seq 递推 100 步。
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
    """统一 Transformer 编码器主干。"""

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
# 2. 简单卷积骨干 (Conv-BN-ReLU-Pool)
# ============================================================================

class SimpleConvBlock(nn.Module):
    """标准卷积块: Conv1d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, pool_size: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class ChannelSelfAttention(nn.Module):
    """通道自注意力: 原始输入通道作为 token 互相 attend, 学习物理耦合。

    每个通道的时序信号作为其 embedding。注意力矩阵 (B, h, C, C) 揭示
    通道间物理依赖关系 (如 innov_x ↔ y_meas_x ↔ u_cmd_v)。

    输入/输出: (B, C, W) — 同形状, 同维度
    """

    def __init__(self, num_channels: int = 11, time_steps: int = 100,
                 proj_dim: int = CHANNEL_ATTN_DIM,
                 num_heads: int = CHANNEL_ATTN_HEADS, dropout: float = 0.2):
        super().__init__()
        assert time_steps % num_heads == 0 or num_heads == 1, \
            f"time_steps ({time_steps}) 必须整除 num_heads ({num_heads})"
        self.num_channels = num_channels
        self.time_steps = time_steps
        self.proj_dim = proj_dim

        self.input_proj = nn.Linear(time_steps, proj_dim)
        self.mha = nn.MultiheadAttention(proj_dim, num_heads,
                                         dropout=dropout, batch_first=True)
        self.output_proj = nn.Linear(proj_dim, time_steps)
        self.norm = nn.LayerNorm(time_steps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, W)
        residual = x
        h = self.input_proj(x)                     # (B, C, proj_dim)
        h, _attn = self.mha(h, h, h)               # (B, C, proj_dim)
        h = self.output_proj(h)                     # (B, C, W)
        return self.norm(residual + self.dropout(h))


class SimpleConvBackbone(nn.Module):
    """简单卷积骨干网络。

    3 个 Conv-BN-ReLU-Pool 块, 逐块通道扩增 + 时序降采样。

    输入: (B, W, C) → 内部排列为 Conv1d 格式 (B, C, W)
    输出: (B, W', d_model)  其中 W' = W // 8

    通道变化: 8→64→128→128
    时序变化: 100→50→25→12
    """

    def __init__(self, in_channels: int = IN_CHANNELS,
                 channels: list = None,
                 kernel_size: int = CONV_KERNEL_SIZE,
                 pool_size: int = POOL_SIZE,
                 use_channel_attn: bool = USE_CHANNEL_ATTN,
                 channel_attn_heads: int = CHANNEL_ATTN_HEADS,
                 channel_attn_dim: int = CHANNEL_ATTN_DIM,
                 time_steps: int = None,
                 dropout: float = 0.2):
        super().__init__()
        if channels is None:
            channels = CONV_CHANNELS
        self.d_model = channels[-1]
        self.use_channel_attn = use_channel_attn

        if use_channel_attn:
            self.channel_attn = ChannelSelfAttention(
                num_channels=in_channels,
                time_steps=time_steps if time_steps is not None else WINDOW_SIZE,
                proj_dim=channel_attn_dim,
                num_heads=channel_attn_heads,
                dropout=dropout,
            )
        else:
            self.channel_attn = None

        layers = []
        ci = in_channels
        for co in channels:
            layers.append(SimpleConvBlock(ci, co, kernel_size, pool_size))
            ci = co
        self.blocks = nn.Sequential(*layers)
        self.norm_out = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 2, 1)     # (B,W,C) -> (B,C,W)
        if self.channel_attn is not None:
            h = self.channel_attn(h)  # channel mixing
        h = self.blocks(h)          # (B,C,W) -> (B,C,W//8)
        h = h.permute(0, 2, 1)     # (B,C,W//8) -> (B,W//8,C)
        h = self.norm_out(h)
        return h


# ============================================================================
# 3. 物理引导解码器
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

        # ConvTranspose1d 上采样: 12 → 24 → 48 → 96
        layers = []
        ci = d_model
        for co in dec_channels:
            layers.extend([
                nn.ConvTranspose1d(ci, co, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(co),
                nn.ReLU(inplace=True),
            ])
            ci = co
        # 最终层: 映射到 3 通道 + 插值到精确 100 步
        layers.append(nn.Conv1d(ci, out_channels, kernel_size=5, padding=2))
        self.upsample = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """从编码器特征预测攻击修正量。

        Args:
            features: (B, 12, d_model) 编码器降采样特征序列

        Returns:
            delta_pred: (B, 100, 3) 攻击修正量, 物理单位
        """
        h = features.permute(0, 2, 1)           # (B, d_model, 12)
        h = self.upsample(h)                     # (B, 3, 96)
        h = F.interpolate(h, size=100, mode='linear', align_corners=False)
        h = h.permute(0, 2, 1)                  # (B, 100, 3)
        return h


# ============================================================================
# 4. 分类检测器 (编码器 + 分类头 + 解码器)
# ============================================================================

class CFMDetector(nn.Module):
    """攻击分类检测器 + 物理引导解码器。

    架构:
      Encoder:   x (B,100,8)
                 → ChannelSelfAttention → SimpleConvBackbone
                 → features (B, 12, 128)
      Classifier: 注意力池化 → LN → Dropout → Linear(128,8) → cls_logits
      Decoder:    features → PhysicsGuidedDecoder → delta_pred (B,100,3)
                 y_kin ← batch_kinematic_rollout(y0_phys, u_cmd_phys)
                 y_pred = y_kin + delta_pred

    输入:  [y_meas(3) + innov(3) + u_cmd(2)] = 8 通道, 归一化后
    输出:  cls_logits (B, 8)    攻击类别 A0-A7
           y_pred (B, 100, 3)    重建的干净传感器信号 (物理单位)
           delta_pred (B, 100, 3) 攻击修正量 (物理单位)
    """

    def __init__(self,
                 in_channels: int = IN_CHANNELS,
                 window_size: int = WINDOW_SIZE,
                 d_model: int = D_MODEL,
                 num_classes: int = NUM_CLASSES,
                 backbone_type: str = 'simple_conv',
                 conv_channels: list = None,
                 conv_kernel_size: int = CONV_KERNEL_SIZE,
                 pool_size: int = POOL_SIZE,
                 use_channel_attn: bool = USE_CHANNEL_ATTN,
                 channel_attn_heads: int = CHANNEL_ATTN_HEADS,
                 channel_attn_dim: int = CHANNEL_ATTN_DIM,
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
            self.backbone = SimpleConvBackbone(
                in_channels=in_channels,
                channels=conv_channels,
                kernel_size=conv_kernel_size,
                pool_size=pool_size,
                use_channel_attn=use_channel_attn,
                channel_attn_heads=channel_attn_heads,
                channel_attn_dim=channel_attn_dim,
                time_steps=window_size,
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

        # ---- 分类头 (注意力池化) ----
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
    # 归一化参数设置 (训练/推理时从 normalizer 加载后调用)
    # ------------------------------------------------------------------

    def set_norm_params(self, ymeas_scale, ymeas_median, cmd_max):
        """设置归一化参数 buffer。"""
        self.ymeas_scale.copy_(torch.as_tensor(ymeas_scale, dtype=torch.float32))
        self.ymeas_median.copy_(torch.as_tensor(ymeas_median, dtype=torch.float32))
        self.cmd_max.copy_(torch.as_tensor(cmd_max, dtype=torch.float32))

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码输入窗口 → 特征序列。

        Args: x: (B, W, 8) 归一化输入。Returns: features (B, W//8, d_model)
        """
        return self.backbone(x)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """从特征序列分类攻击类型 (注意力池化)。

        Args: features: (B, W', d_model)。Returns: cls_logits (B, num_classes)
        """
        d = features.shape[-1]
        scores = torch.matmul(features, self.attn_query) / (d ** 0.5)
        attn_weights = torch.softmax(scores, dim=1)          # (B, W')
        pooled = (features * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        pooled = self.cls_norm(pooled)
        pooled = self.cls_dropout(pooled)
        return self.cls_head(pooled)

    def decode(self, features: torch.Tensor, x_norm: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """物理引导解码: 运动学递推 + 学习修正 → 重建干净信号。

        Args:
            features: (B, W', d_model) 编码器特征
            x_norm:   (B, W, 8) 归一化输入窗口 (用于提取 y0 和 u_cmd)

        Returns:
            y_pred:     (B, W, 3) 重建的干净传感器信号 (物理单位)
            delta_pred: (B, W, 3) 攻击修正量 (物理单位)
        """
        if self.decoder is None:
            raise RuntimeError("Decoder not enabled. Set use_decoder=True.")

        # 1. 从归一化输入反归一化得到物理量
        y0_norm = x_norm[:, 0, :3]                     # (B, 3) 首步 y_meas
        u_norm = x_norm[:, :, -2:]                      # (B, W, 2) u_cmd
        y0_phys = y0_norm * self.ymeas_scale + self.ymeas_median
        u_phys = u_norm * self.cmd_max

        # 2. 运动学递推 (冻结梯度 — 这是已知物理, 不学习)
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
            return_recon: 是否返回重建信号 (训练/完整推理时为 True)

        Returns:
            return_recon=False: (cls_logits, features)
            return_recon=True:  (cls_logits, features, y_pred, delta_pred)
        """
        features = self.encode(x)
        cls_logits = self.classify(features)

        if return_recon and self.decoder is not None:
            y_pred, delta_pred = self.decode(features, x)
            return cls_logits, features, y_pred, delta_pred

        return cls_logits, features
