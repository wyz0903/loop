"""
detector/detector.py — 攻击检测模型
====================================
8 通道输入: [y_meas(3) + innov_anchored(3) + u_cmd(2)]
输出: cls_logits (B,8), y_pred (B,100,3)

架构: MultiScaleDSConvBackbone → 运动学一致性偏置注意力池化 → 分类头
      features → PhysicsGuidedDecoder → y_kin + delta_pred = y_pred
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

TS = 0.05
ALPHA = 0.17

D_MODEL = 96
NUM_CLASSES = 8
IN_CHANNELS = 8
WINDOW_SIZE = 100
CONV_CHANNELS = [32, 64, 96]
CONV_KERNEL_SIZES = [7, 5, 3]
CONV_DILATIONS = [1, 3, 9]
POOL_SIZE = 2
DECODER_CHANNELS = [64, 32, 16]
KIN_BIAS_SIGMA = 0.5


# ============================================================================
# 工具: 批量运动学 rollout
# ============================================================================

def batch_kinematic_rollout(y0, u_seq, Ts=TS, alpha=ALPHA):
    """欧拉积分运动学递推。y0:(B,3), u_seq:(B,W,2) → y_kin:(B,W,3)"""
    B, W, _ = u_seq.shape
    y_kin = torch.zeros(B, W, 3, device=u_seq.device, dtype=u_seq.dtype)
    y = y0.to(device=u_seq.device, dtype=u_seq.dtype)
    y_kin[:, 0, :] = y
    for k in range(W - 1):
        v, w = u_seq[:, k, 0], u_seq[:, k, 1]
        cos_t, sin_t = torch.cos(y[:, 2]), torch.sin(y[:, 2])
        y = y + Ts * torch.stack([v * cos_t - alpha * w * sin_t,
                                   v * sin_t + alpha * w * cos_t, w], dim=-1)
        y_kin[:, k + 1, :] = y
    return y_kin


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
    """3 块逐通道扩增 + 降采样: 8→32→64→96, 100→50→25→12"""

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
# 运动学一致性偏置 (零参数物理先验)
# ============================================================================

class KinematicConsistencyBias(nn.Module):
    """从 innov_anchored 的 L2 范数计算注意力偏置: bias = -‖innov‖/σ"""

    def __init__(self, sigma=KIN_BIAS_SIGMA):
        super().__init__()
        self.sigma = sigma

    def forward(self, x_norm, feat_scale, feat_median):
        with torch.no_grad():
            innov_phys = (x_norm[:, :, 3:6] * feat_scale.view(1, 1, 3)
                          + feat_median.view(1, 1, 3))
            t_idx = list(range(4, 100, 8))
            innov_l2 = torch.norm(innov_phys[:, t_idx, :], dim=-1)
            bias = -innov_l2 / self.sigma
            return bias - bias.mean(dim=-1, keepdim=True)


# ============================================================================
# 物理引导解码器
# ============================================================================

class PhysicsGuidedDecoder(nn.Module):
    """ConvTranspose1d 上采样: 12→24→48→96→100, 预测 delta = y_clean - y_kin"""

    def __init__(self, d_model=D_MODEL):
        super().__init__()
        layers = []
        ci = d_model
        for co in DECODER_CHANNELS:
            layers.extend([nn.ConvTranspose1d(ci, co, 4, 2, 1),
                           nn.BatchNorm1d(co), nn.GELU()])
            ci = co
        layers.append(nn.Conv1d(ci, 3, 5, padding=2))
        self.upsample = nn.Sequential(*layers)

    def forward(self, features):
        h = features.permute(0, 2, 1)
        h = self.upsample(h)
        return F.interpolate(h, size=100, mode='linear', align_corners=False).permute(0, 2, 1)


# ============================================================================
# 检测器
# ============================================================================

class Detector(nn.Module):
    """攻击检测模型: 编码器 + 运动学偏置注意力分类 + 物理引导解码"""

    def __init__(self, use_decoder=True,
                 ymeas_scale=None, ymeas_median=None, cmd_max=None,
                 feat_scale=None, feat_median=None):
        super().__init__()
        self.use_decoder = use_decoder

        # 归一化参数 (buffer, 不参与梯度)
        self.register_buffer('ymeas_scale', torch.tensor(
            ymeas_scale or [2.5, 2.5, 3.141592653589793], dtype=torch.float32))
        self.register_buffer('ymeas_median', torch.tensor(
            ymeas_median or [0., 0., 0.], dtype=torch.float32))
        self.register_buffer('cmd_max', torch.tensor(
            cmd_max or [0.3, 1.76], dtype=torch.float32))
        self.register_buffer('feat_scale', torch.tensor(
            feat_scale or [2.5, 2.5, 3.141592653589793], dtype=torch.float32))
        self.register_buffer('feat_median', torch.tensor(
            feat_median or [0., 0., 0.], dtype=torch.float32))

        self.backbone = MultiScaleDSConvBackbone()
        self.kin_bias = KinematicConsistencyBias()
        self.bias_alpha = nn.Parameter(torch.tensor(0.1))

        self.cls_norm = nn.LayerNorm(D_MODEL)
        self.cls_head = nn.Linear(D_MODEL, NUM_CLASSES)
        self.cls_dropout = nn.Dropout(0.3)
        self.attn_query = nn.Parameter(torch.randn(D_MODEL) * 0.02)

        self.decoder = PhysicsGuidedDecoder() if use_decoder else None
        nn.init.xavier_uniform_(self.cls_head.weight)
        nn.init.zeros_(self.cls_head.bias)

    def set_norm_params(self, ymeas_scale, ymeas_median, cmd_max, feat_median, feat_scale):
        """从 normalizer 加载后更新归一化参数"""
        for name, val in [('ymeas_scale', ymeas_scale), ('ymeas_median', ymeas_median),
                           ('cmd_max', cmd_max), ('feat_median', feat_median),
                           ('feat_scale', feat_scale)]:
            getattr(self, name).copy_(torch.as_tensor(val, dtype=torch.float32))

    def encode(self, x):
        return self.backbone(x)

    def classify(self, features, x_norm=None):
        d = features.shape[-1]
        scores = torch.matmul(features, self.attn_query) / (d ** 0.5)
        if x_norm is not None:
            scores = scores + self.bias_alpha * self.kin_bias(
                x_norm, self.feat_scale, self.feat_median)
        attn = torch.softmax(scores, dim=1)
        pooled = (features * attn.unsqueeze(-1)).sum(dim=1)
        return self.cls_head(self.cls_dropout(self.cls_norm(pooled)))

    def decode(self, features, x_norm):
        """物理引导解码: y_pred = y_kin + delta_pred"""
        y0_phys = x_norm[:, 0, :3] * self.ymeas_scale + self.ymeas_median
        u_phys = x_norm[:, :, -2:] * self.cmd_max
        with torch.no_grad():
            y_kin = batch_kinematic_rollout(y0_phys, u_phys)
        delta_pred = self.decoder(features)
        return y_kin + delta_pred, delta_pred

    def forward(self, x, return_recon=False):
        features = self.encode(x)
        cls_logits = self.classify(features, x)
        if return_recon and self.decoder is not None:
            y_pred, delta_pred = self.decode(features, x)
            return cls_logits, features, y_pred, delta_pred
        return cls_logits, features
