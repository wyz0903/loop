"""
detector/cfm_detector.py — 攻击分类检测器 (cls-only 分支)
============================================================
精简架构: 简单卷积骨干 + 注意力池化分类头。仅做攻击类型识别。

输入: [y_meas(3) + innov(3) + u_cmd(2)] = 8 通道
输出: 攻击类别 A0-A8 (9 类 softmax)

架构:
  SimpleConvBackbone (3块 Conv-BN-ReLU-Pool, 通道 8→64→128→128)
    → features (B,W//8,d_model)
    → 注意力池化 → LN → Linear(d_model→9) → cls_logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ============================================================================
# 全局模型参数
# ============================================================================

TS = 0.05          # 采样周期 [s]

# 模型架构
D_MODEL = 128
NUM_HEADS = 8
NUM_TRANSFORMER_LAYERS = 4
DIM_FEEDFORWARD = 512
NUM_CLASSES = 9
IN_CHANNELS = 8          # [y_meas(3) + innov(3) + u_cmd(2)]
WINDOW_SIZE = 100

# 简单卷积骨干
CONV_CHANNELS = [64, 128, 128]     # 逐块输出通道 (3块)
CONV_KERNEL_SIZE = 3               # 卷积核大小 (same padding)
POOL_SIZE = 2                      # 池化核大小 (时序降采样)

# 通道自注意力
USE_CHANNEL_ATTN = True            # 是否启用通道自注意力 (消融实验开关)
CHANNEL_ATTN_HEADS = 4             # 注意力头数
CHANNEL_ATTN_DIM = 64              # 中间投影维度


# ============================================================================
# 1. Transformer 骨干 (向后兼容)
# ============================================================================

class TransformerBackbone(nn.Module):
    """统一 Transformer 编码器主干。"""

    def __init__(self, in_channels: int = IN_CHANNELS, window_size: int = WINDOW_SIZE,
                 d_model: int = D_MODEL, num_layers: int = NUM_TRANSFORMER_LAYERS,
                 num_heads: int = NUM_HEADS, d_ff: int = DIM_FEEDFORWARD,
                 dropout: float = 0.1):
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

    def __init__(self, num_channels: int = 8, time_steps: int = 100,
                 proj_dim: int = CHANNEL_ATTN_DIM,
                 num_heads: int = CHANNEL_ATTN_HEADS, dropout: float = 0.1):
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
                 time_steps: int = None):
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
# 3. 分类检测器 (cls-only)
# ============================================================================

class CFMDetector(nn.Module):
    """攻击分类检测器 (cls-only 分支)。

    架构:
      x (B,W,8) -> [ChannelSelfAttention] -> SimpleConvBackbone (3 block)
        -> features (B,W//8,d_model)
        -> 注意力池化 -> LN -> Dropout -> cls_head -> cls_logits (B,9)

    输入:  [y_meas(3) + innov(3) + u_cmd(2)] = 8 通道
    输出:  cls_logits (B, 9)  攻击类别 A0-A8
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
                 dropout: float = 0.1,
                 # 以下参数保持签名兼容 (cls-only 分支忽略)
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.backbone_type = backbone_type

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
        self.cls_dropout = nn.Dropout(0.2)
        self.attn_query = nn.Parameter(torch.randn(d_model))  # 注意力池化 query

        self._init_cls_head()

    def _init_cls_head(self):
        nn.init.xavier_uniform_(self.cls_head.weight)
        nn.init.zeros_(self.cls_head.bias)

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码输入窗口 → 特征序列。

        Args: x: (B, W, 8) 归一化输入。Returns: features (B, W, d_model)
        """
        return self.backbone(x)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """从特征序列分类攻击类型 (注意力池化)。

        Args: features: (B, W, d_model)。Returns: cls_logits (B, num_classes)
        """
        d = features.shape[-1]
        scores = torch.matmul(features, self.attn_query) / (d ** 0.5)
        attn_weights = torch.softmax(scores, dim=1)          # (B, W)
        pooled = (features * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        pooled = self.cls_norm(pooled)
        pooled = self.cls_dropout(pooled)
        return self.cls_head(pooled)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """单次前向: 编码 + 分类。

        Args: x: (B, W, 8) 归一化输入窗口。
        Returns: (cls_logits (B, 9), features (B, W, d_model))
        """
        features = self.encode(x)
        cls_logits = self.classify(features)
        return cls_logits, features
