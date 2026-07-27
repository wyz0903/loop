"""
detector/classifier.py — 攻击检测模型 (基线: 多尺度膨胀深度可分离卷积 + 注意力池化)
============================================================================
5 通道输入: [y_meas(3) + u_cmd(2)]
输出: cls_logits (B,8)

架构: MultiScaleDSConvBackbone → 注意力池化 → 分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

D_MODEL = 96
NUM_CLASSES = 8
IN_CHANNELS = 5
WINDOW_SIZE = 128
CONV_CHANNELS = [32, 64, 96]
CONV_KERNEL_SIZES = [7, 5, 3]
CONV_DILATIONS = [1, 3, 9]
POOL_SIZE = 2


# ============================================================================
# 多尺度膨胀深度可分离卷积骨干
# ============================================================================

class MultiScaleDSConvBlock(nn.Module):
    """3 并行膨胀深度卷积 (d=1,3,9) → Pointwise 混合 → BN+GELU+Pool"""

    def __init__(self, in_channels, out_channels, kernel_size=7,
                 dilations=None, pool_size=POOL_SIZE):
        super().__init__()
        dilations = dilations or CONV_DILATIONS
        self.depthwise_convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size,
                      padding=(kernel_size // 2) * d,
                      dilation=d, groups=in_channels, bias=False)
            for d in dilations
        ])
        self.pointwise = nn.Conv1d(in_channels * 3, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        h = torch.cat([dw(x) for dw in self.depthwise_convs], dim=1)
        return self.pool(F.gelu(self.bn(self.pointwise(h))))


class MultiScaleDSConvBackbone(nn.Module):
    """3 块逐通道扩增 + 降采样: 5→32→64→96, 128→64→32→16"""

    def __init__(self, in_channels=IN_CHANNELS):
        super().__init__()
        self.d_model = CONV_CHANNELS[-1]
        blocks = []
        ci = in_channels
        for i, co in enumerate(CONV_CHANNELS):
            blocks.append(MultiScaleDSConvBlock(ci, co, CONV_KERNEL_SIZES[i]))
            ci = co
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = nn.LayerNorm(self.d_model)

    def forward(self, x):
        h = x.permute(0, 2, 1)
        h = self.blocks(h)
        return self.norm_out(h.permute(0, 2, 1))


# ============================================================================
# 检测器
# ============================================================================

class Detector(nn.Module):
    """攻击检测模型: 膨胀深度可分离卷积骨干 + 注意力池化分类"""

    def __init__(self, ymeas_scale=None, ymeas_median=None, cmd_max=None):
        super().__init__()

        # 归一化参数 (buffer, 不参与梯度)
        self.register_buffer('ymeas_scale', torch.tensor(
            ymeas_scale or [2.5, 2.5, 3.141592653589793], dtype=torch.float32))
        self.register_buffer('ymeas_median', torch.tensor(
            ymeas_median or [0., 0., 0.], dtype=torch.float32))
        self.register_buffer('cmd_max', torch.tensor(
            cmd_max or [0.3, 1.76], dtype=torch.float32))

        self.backbone = MultiScaleDSConvBackbone()

        self.cls_norm = nn.LayerNorm(D_MODEL)
        self.cls_head = nn.Linear(D_MODEL, NUM_CLASSES)
        self.cls_dropout = nn.Dropout(0.3)
        self.attn_query = nn.Parameter(torch.randn(D_MODEL) * 0.02)

        nn.init.xavier_uniform_(self.cls_head.weight)
        nn.init.zeros_(self.cls_head.bias)

    def set_norm_params(self, ymeas_scale, ymeas_median, cmd_max):
        """从 normalizer 加载后更新归一化参数"""
        for name, val in [('ymeas_scale', ymeas_scale), ('ymeas_median', ymeas_median),
                           ('cmd_max', cmd_max)]:
            getattr(self, name).copy_(torch.as_tensor(val, dtype=torch.float32))

    def encode(self, x):
        return self.backbone(x)

    def classify(self, features):
        d = features.shape[-1]
        scores = torch.matmul(features, self.attn_query) / (d ** 0.5)
        attn = torch.softmax(scores, dim=1)
        pooled = (features * attn.unsqueeze(-1)).sum(dim=1)
        return self.cls_head(self.cls_dropout(self.cls_norm(pooled)))

    def forward(self, x):
        features = self.encode(x)
        cls_logits = self.classify(features)
        return cls_logits, features
