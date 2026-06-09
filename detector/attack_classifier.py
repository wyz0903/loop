"""
detector/attack_classifier.py — 基线攻击分类与重建网络
========================================================
AttackClassifier: 分类器 + 解码器架构。

编码器: ResDownBlock × 4 → GlobalAvgPool → FC → latent(256)
分类器: latent → FC(128) → FC(64) → 9 (主任务)
解码器: latent → FC → ResUpBlock × 4 → 攻击信号 (辅助, 无跳连)

输入: (B, W, 5) 归一化特征窗口 [internal_innovation(3) + u_cmd(2)]
输出: cls_logits (B, 9), attack_seq (B, W, 3), z (B, latent_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from detector.config import ENC_CHANNELS, LATENT_DIM
from detector.nn_blocks import ResDownBlock, ResUpBlock


class AttackClassifier(nn.Module):
    """分类器 + 解码器 (无编码器→解码器跳跃连接, 解码器内部残差)

    编码器: ResDownBlock × 4 → GlobalAvgPool → FC → latent(256)
    分类器: latent → FC(128) → FC(64) → 9 (主任务)
    解码器: latent → FC → ResUpBlock × 4 → 攻击信号 (辅助, 无跳连)
    """

    def __init__(self, in_channels: int = 5, window_size: int = 100,
                 latent_dim: int = LATENT_DIM, num_classes: int = 9,
                 enc_channels: list = None, dec_channels_override: list = None):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.latent_dim = latent_dim

        enc_ch = enc_channels or ENC_CHANNELS
        # 解码器通道: 优先使用显式覆盖 (兼容旧模型)
        if dec_channels_override is not None:
            dec_ch = list(dec_channels_override)
        else:
            dec_start = max(enc_ch[-1] * 2 // 3, 64)
            dec_ch = []
            cur = dec_start
            for _ in range(len(enc_ch)):
                dec_ch.append(cur)
                cur = max(cur * 2 // 3, 32)
        self.dec_channels = dec_ch

        # ---- 编码器 ----
        prev_ch = in_channels
        self.enc_blocks = nn.ModuleList()
        for i, ch in enumerate(enc_ch):
            ks = 7 if i == 0 else 5 if i < 3 else 3
            self.enc_blocks.append(ResDownBlock(prev_ch, ch, kernel_size=ks))
            prev_ch = ch

        # ---- 瓶颈 + 潜在空间 ----
        self.bottleneck_conv = nn.Conv1d(enc_ch[-1], enc_ch[-1], kernel_size=3,
                                          padding=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm1d(enc_ch[-1])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.enc_fc = nn.Linear(enc_ch[-1], latent_dim)

        # ---- 分类器 ----
        self.cls_fc1 = nn.Linear(latent_dim, 128)
        self.cls_bn1 = nn.BatchNorm1d(128)
        self.cls_dropout1 = nn.Dropout(0.3)
        self.cls_fc2 = nn.Linear(128, 64)
        self.cls_bn2 = nn.BatchNorm1d(64)
        self.cls_dropout2 = nn.Dropout(0.2)
        self.cls_head = nn.Linear(64, num_classes)

        # ---- 解码器 ----
        self.dec_init_size = self._calc_feat_size(window_size, len(enc_ch))
        self.dec_fc = nn.Linear(latent_dim, enc_ch[-1] * self.dec_init_size)

        dec_in = enc_ch[-1]
        self.dec_blocks = nn.ModuleList()
        for i, out_ch in enumerate(dec_ch):
            ks = 5 if i < 3 else 7
            self.dec_blocks.append(ResUpBlock(dec_in, out_ch, kernel_size=ks))
            dec_in = out_ch

        self.output_conv = nn.Conv1d(dec_ch[-1], 3, kernel_size=3, padding=1)

        self._init_weights()

    def _calc_feat_size(self, w: int, n_blocks: int) -> int:
        for _ in range(n_blocks):
            w = (w + 1) // 2
        return w

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x_conv = x.permute(0, 2, 1)  # (B, C, W)

        # 编码
        h = x_conv
        for block in self.enc_blocks:
            h = block(h)

        # 瓶颈 → latent
        h = F.relu(self.bottleneck_bn(self.bottleneck_conv(h)))
        h = self.global_pool(h).squeeze(-1)
        z = F.relu(self.enc_fc(h))  # (B, latent_dim)

        # === 分类 (主任务) ===
        c = F.relu(self.cls_bn1(self.cls_fc1(z)))
        c = self.cls_dropout1(c)
        c = F.relu(self.cls_bn2(self.cls_fc2(c)))
        c = self.cls_dropout2(c)
        cls_logits = self.cls_head(c)

        # === 解码 (辅助任务, 无跳连, 输出细化) ===
        d = F.relu(self.dec_fc(z))
        d = d.view(B, -1, self.dec_init_size)
        for block in self.dec_blocks:
            d = block(d)
        if d.shape[-1] != self.window_size:
            d = F.interpolate(d, size=self.window_size, mode='linear', align_corners=False)
        attack_seq = self.output_conv(d).permute(0, 2, 1)  # (B, W, 3)

        return cls_logits, attack_seq, z
