"""
detector/cfm_detector.py — 攻击分类检测器 (cls-only 分支)
============================================================
精简架构: 因果空洞卷积主干 + 注意力池化分类头。仅做攻击类型识别。

输入: [y_meas(3) + innov(3) + u_cmd(2)] = 8 通道
输出: 攻击类别 A0-A8 (9 类 softmax)

架构:
  CausalDilatedConvBackbone (6层, d_model=128, dilations=[1,2,4,8,16,32])
    → features (B,100,128)
    → 注意力池化 → LN → Linear(128→9) → cls_logits
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

# 因果空洞卷积
DILATIONS = [1, 2, 4, 8, 16, 32]  # 膨胀因子序列 (RF=253)
CONV_KERNEL_SIZE = 3               # 因果卷积核大小


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
# 2. 因果空洞卷积骨干 (TCN 风格)
# ============================================================================

class CausalConv1d(nn.Module):
    """因果 Conv1d: 仅左侧填充, 保证时刻 t 的输出只依赖于 ≤t 的输入。"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=0)
        self.pad_left = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad_left, 0))
        return self.conv(x)


class CausalDilatedConvBlock(nn.Module):
    """因果空洞卷积残差块: CausalConv → GELU → CausalConv → Dropout → +residual."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for conv in [self.conv1.conv, self.conv2.conv]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.conv1(x))
        h = self.conv2(h)
        h = self.dropout(h)
        return x + h


class CausalDilatedConvBackbone(nn.Module):
    """因果空洞卷积骨干网络 (TCN 风格)。

    输入: (B, W, C) → 内部排列为 Conv1d 格式 (B, C, W)
    输出: (B, W, d_model)
    感受野: RF = 1 + 2*(k-1)*sum(dilations) = 253 > 100 ✓
    """

    def __init__(self, in_channels: int = IN_CHANNELS, d_model: int = D_MODEL,
                 dilations: list = None, kernel_size: int = CONV_KERNEL_SIZE,
                 dropout: float = 0.1):
        super().__init__()
        if dilations is None:
            dilations = DILATIONS
        self.d_model = d_model
        self.dilations = list(dilations)
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.blocks = nn.ModuleList([
            CausalDilatedConvBlock(d_model, d, kernel_size, dropout)
            for d in self.dilations
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 2, 1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        h = h.permute(0, 2, 1)
        h = self.norm_out(h)
        return h


# ============================================================================
# 3. 分类检测器 (cls-only)
# ============================================================================

class CFMDetector(nn.Module):
    """攻击分类检测器 (cls-only 分支)。

    架构:
      x (B,W,8) → Backbone → features (B,W,d_model)
        → 注意力池化 → LN → Dropout → cls_head → cls_logits (B,9)

    输入:  [y_meas(3) + innov(3) + u_cmd(2)] = 8 通道
    输出:  cls_logits (B, 9)  攻击类别 A0-A8
    """

    def __init__(self,
                 in_channels: int = IN_CHANNELS,
                 window_size: int = WINDOW_SIZE,
                 d_model: int = D_MODEL,
                 num_classes: int = NUM_CLASSES,
                 backbone_type: str = 'causal_conv',
                 dilations: list = None,
                 conv_kernel_size: int = CONV_KERNEL_SIZE,
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
