"""
detector/nn_blocks.py — 共享神经网络构建块
=============================================
ResDownBlock / ResUpBlock — 残差下采样/上采样块，
被 AttackClassifier 和 FreqAwareClassifier 共用。
"""

import torch.nn as nn


class ResDownBlock(nn.Module):
    """残差下采样块: 双卷积 + 残差投影 + stride=2 池化"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=2, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        # 残差投影: 匹配维度 (stride=2 + channel change)
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1,
                              stride=2, bias=False)
        self.proj_bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.proj_bn(self.proj(x))
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


class ResUpBlock(nn.Module):
    """残差上采样块: ConvTranspose + 残差投影 + BN + ReLU

    残差连接保证上采样时不丢失低频轮廓信息。
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel_size,
                                      stride=2, padding=kernel_size // 2,
                                      output_padding=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.proj = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=1,
                                        stride=2, output_padding=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.proj_bn(self.proj(x))
        out = self.bn(self.up(x))
        return self.act(out + residual)
