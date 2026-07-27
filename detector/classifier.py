"""
detector/classifier.py — 攻击检测模型 (U-Net 掩码重建 + 双尺度注意力分类)
====================================================================
5 通道输入: [y_meas(3) + u_cmd(2)]
训练时: 随机掩码窗口最后 k% 时间步 → 膨胀DSConv U-Net 重建 + 双尺度注意力分类
推理时: 完整序列 → 分类

架构: 膨胀深度可分离卷积 U-Net (Encoder-Bottleneck-Decoder + skip connections)
      瓶颈(8步) + enc3(16步) 双尺度 → 注意力池化 → 分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

D_MODEL = 128
NUM_CLASSES = 8
IN_CHANNELS = 5
WINDOW_SIZE = 128
CLS_DIM = 96

# 编码器通道: 5 → 16 → 32 → 64 → 128
ENC_CHANNELS = [16, 32, 64, 128]
ENC_KERNELS = [7, 5, 5, 3]
DS_DILATIONS = [1, 3, 9]

# 解码器通道 (反向): 128 → 64 → 32 → 16 → 8
DEC_CHANNELS = [64, 32, 16, 8]
DEC_KERNELS = [3, 5, 5, 7]


# ============================================================================
# 多尺度膨胀深度可分离卷积编码器块
# ============================================================================

class MultiScaleEncoderBlock(nn.Module):
    """3 并行膨胀深度卷积 (d=1,3,9) → Pointwise 混合 → BN → GELU → MaxPool(2)

    不同膨胀率覆盖不同时间尺度:
      d=1:  0.35s — 瞬时突变 (A5 dropout, A6 scaling)
      d=3:  0.95s — 短时异常 (A4 replay)
      d=9:  2.75s — 慢变漂移 (A3 drift, A2 sinusoidal)
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilations=None):
        super().__init__()
        dilations = dilations or DS_DILATIONS
        self.depthwise = nn.ModuleList([
            nn.Conv1d(in_ch, in_ch, kernel_size,
                      padding=(kernel_size // 2) * d,
                      dilation=d, groups=in_ch, bias=False)
            for d in dilations
        ])
        self.pointwise = nn.Conv1d(in_ch * len(dilations), out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        h = torch.cat([dw(x) for dw in self.depthwise], dim=1)
        return self.pool(F.gelu(self.bn(self.pointwise(h))))


# ============================================================================
# 解码器块
# ============================================================================

class DecoderBlock(nn.Module):
    """单级解码器: Upsample(×2) + Concat(skip) → Conv1d → BN → GELU"""

    def __init__(self, in_ch, skip_ch, out_ch, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
        return F.gelu(self.bn(self.conv(torch.cat([x, skip], dim=1))))


# ============================================================================
# 双尺度注意力分类头
# ============================================================================

class DualScaleAttentionClassifier(nn.Module):
    """瓶颈 (8 时间步) + enc3 (16 时间步, 池化到 8) → 注意力池化 → 分类

    双尺度融合:
      - 瓶颈: 最深层的抽象语义特征
      - enc3: 较高时间分辨率的细节特征 (攻击边界、短时突变)
    """

    def __init__(self, bn_ch=ENC_CHANNELS[-1], enc3_ch=ENC_CHANNELS[-2],
                 cls_dim=CLS_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.proj_bn = nn.Conv1d(bn_ch, cls_dim, 1)       # 瓶颈 → cls_dim
        self.proj_enc3 = nn.Conv1d(enc3_ch, cls_dim, 1)   # enc3 → cls_dim (pool to 8)
        self.norm = nn.LayerNorm(cls_dim)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(cls_dim, num_classes)

        # 可学习注意力查询向量
        self.attn_query = nn.Parameter(torch.randn(cls_dim) * 0.02)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, bottleneck, enc3):
        """
        Args:
            bottleneck: (B, bn_ch, 8)  瓶颈特征
            enc3:       (B, enc3_ch, 16)  enc3 输出
        Returns:
            cls_logits: (B, num_classes)
        """
        f_bn = self.proj_bn(bottleneck)                       # (B, cls_dim, 8)
        f_enc3 = F.adaptive_avg_pool1d(self.proj_enc3(enc3), 8)  # (B, cls_dim, 8)
        f = f_bn + f_enc3                                     # 双尺度逐元素融合

        f = f.permute(0, 2, 1)                                # (B, 8, cls_dim)
        f = self.norm(f)

        # 注意力池化
        d = f.shape[-1]
        scores = torch.matmul(f, self.attn_query) / (d ** 0.5)
        attn = torch.softmax(scores, dim=1)
        pooled = (f * attn.unsqueeze(-1)).sum(dim=1)          # (B, cls_dim)

        return self.head(self.dropout(pooled))


# ============================================================================
# U-Net 骨干
# ============================================================================

class UNet1D(nn.Module):
    """1D U-Net: 4 级膨胀 DSConv 编码器 → 瓶颈 → 4 级解码器 + skip connections

    编码器使用多尺度膨胀深度可分离卷积 (d=1,3,9) 覆盖多时间尺度。
    返回重建 + 瓶颈 + enc3 特征供双尺度分类头使用。
    """

    def __init__(self, in_channels=IN_CHANNELS):
        super().__init__()
        ci = in_channels
        self.encoders = nn.ModuleList()
        for i, co in enumerate(ENC_CHANNELS):
            self.encoders.append(
                MultiScaleEncoderBlock(ci, co, ENC_KERNELS[i]))
            ci = co

        # 瓶颈: 标准卷积 (8 时间步分辨率下膨胀收益有限)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(ENC_CHANNELS[-1], ENC_CHANNELS[-1], 3, padding=1, bias=False),
            nn.BatchNorm1d(ENC_CHANNELS[-1]),
            nn.GELU(),
        )

        # 解码器 (skip 来自对应编码器层, 逆向)
        self.decoders = nn.ModuleList()
        skip_channels = list(reversed(ENC_CHANNELS[:-1])) + [in_channels]
        in_ch = ENC_CHANNELS[-1]
        for out_ch, skip_ch in zip(DEC_CHANNELS, skip_channels):
            self.decoders.append(DecoderBlock(in_ch, skip_ch, out_ch, DEC_KERNELS[len(self.decoders)]))
            in_ch = out_ch

        self.out_conv = nn.Conv1d(DEC_CHANNELS[-1], in_channels, 1)

    @property
    def d_model(self):
        return ENC_CHANNELS[-1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, C, T) → (x_recon, bottleneck, enc3_features)"""
        skips = [x]
        h = x
        enc3_out = None
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            if i == 2:  # enc3 (索引: 0,1,2,3)
                enc3_out = h
            skips.append(h)

        b = self.bottleneck(h)

        h = b
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 2)]
            h = dec(h, skip)

        x_recon = self.out_conv(h)
        return x_recon, b, enc3_out


# ============================================================================
# 检测器
# ============================================================================

class Detector(nn.Module):
    """掩码重建 + 双尺度注意力分类检测器

    训练: 随机掩码窗口最后 k% 步 → U-Net 重建 + 双尺度注意力分类
    推理: 完整序列 → 分类
    """

    def __init__(self, ymeas_scale=None, ymeas_median=None, cmd_max=None,
                 mask_min: float = 0.1, mask_max: float = 0.5):
        super().__init__()
        self.mask_min = mask_min
        self.mask_max = mask_max

        # 归一化参数 (buffer)
        self.register_buffer('ymeas_scale', torch.tensor(
            ymeas_scale or [2.5, 2.5, 3.141592653589793], dtype=torch.float32))
        self.register_buffer('ymeas_median', torch.tensor(
            ymeas_median or [0., 0., 0.], dtype=torch.float32))
        self.register_buffer('cmd_max', torch.tensor(
            cmd_max or [0.3, 1.76], dtype=torch.float32))

        self.unet = UNet1D(in_channels=IN_CHANNELS)
        self.classifier = DualScaleAttentionClassifier()

    def set_norm_params(self, ymeas_scale, ymeas_median, cmd_max):
        for name, val in [('ymeas_scale', ymeas_scale), ('ymeas_median', ymeas_median),
                           ('cmd_max', cmd_max)]:
            getattr(self, name).copy_(torch.as_tensor(val, dtype=torch.float32))

    def _apply_mask(self, x: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """掩码序列最后 ratio 比例的时间步 (置零)"""
        B, C, T = x.shape
        mask_len = max(1, int(T * ratio))
        mask = torch.ones(B, 1, T, device=x.device, dtype=x.dtype)
        mask[:, :, -mask_len:] = 0.0
        return x * mask, mask

    def forward(self, x: torch.Tensor,
                mask_ratio: Optional[float] = None,
                return_recon: bool = False):
        """
        Args:
            x: (B, T, C) 归一化后的输入
            mask_ratio: 掩码比例 (None=不掩码/推理模式)
            return_recon: 是否返回重建结果

        Returns:
            训练模式: cls_logits, x_recon, mask
            推理模式: cls_logits, features
        """
        # (B, T, C) → (B, C, T)
        x = x.permute(0, 2, 1)

        mask = None
        if mask_ratio is not None and mask_ratio > 0:
            x_input, mask = self._apply_mask(x, mask_ratio)
        else:
            x_input = x

        x_recon, bottleneck, enc3 = self.unet(x_input)
        cls_logits = self.classifier(bottleneck, enc3)

        if return_recon and mask is not None:
            return cls_logits, x_recon.permute(0, 2, 1), mask.squeeze(1)
        return cls_logits, bottleneck.permute(0, 2, 1)
