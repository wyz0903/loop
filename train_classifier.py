"""
train_classifier.py — 攻击检测分类器 + 多尺度频率感知解码器
============================================================
双路径架构: 分类编码器 + 频率保持路径(50步分辨率 Nyquist=5Hz)。

架构:
  Encoder: ResDownBlock×4 [48,96,192,256] → latent(256)
  Classifier: latent → FC(128)→FC(64)→9
  FreqPath: Conv(k=7,s=2)→Conv(k=5,s=1) → 50步分辨率(Nyq=5Hz)
  Position Encoding: 正弦位置编码 → 解码器感知绝对时间位置
  DilatedConv: rate=1,2,4 多尺度时间上下文
  Decoder: FiLM + ConvTranspose(50→100) + 2×Conv1d → 攻击信号(3,100)
  DC Offset: latent → FC→3 直流分量

输入: internal_innovation(3) + u_cmd(2) = 5 通道
参数量: ~900K

用法:
  python preprocess_data.py                     # 先运行预处理
  python train_classifier.py                    # 训练
  python train_classifier.py --eval-only models/cls_best.pt
"""

import os
import sys
import math
import argparse
import numpy as np
from collections import defaultdict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', message='Detected call of.*lr_scheduler.step.*before.*optimizer.step')
warnings.filterwarnings('ignore', message='.*non-writable tensor.*')  # mmap 只读无害

# ============================================================================
# 全局配置
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'dataset_win', 'config')  # 默认 config 划分
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

ALL_ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
ATTACK_NAMES_CN = {
    'A0': '正常', 'A1': '恒定偏置', 'A2': '正弦振荡',
    'A3': '斜坡漂移', 'A4': '阶跃', 'A5': '重放攻击',
    'A6': '脉冲串', 'A7': '扫频', 'A8': '多频叠加',
}

# 训练超参数
BATCH_SIZE = 2048
NUM_WORKERS = 2
PREFETCH_FACTOR = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 150
RECON_LAMBDA = 3.0             # 重建损失总权重
LATENT_DIM = 256
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 50
GRAD_CLIP = 1.0
USE_AMP = True
USE_COMPILE = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 编码器通道配置 (加宽以匹配/超越旧 baseline 分类精度)
ENC_CHANNELS = [48, 96, 192, 256]

# FreqAware 专用超参数
FREQ_CHANNELS = [24, 32]      # 频率路径: 50步分辨率 (Nyquist=5Hz, 覆盖A7 4Hz扫频)
CLASS_EMBED_DIM = 48           # 分类嵌入维度 (用于 FiLM 调制)
A0_RECON_WEIGHT = 0.15         # A0 窗口重建权重 (MSE only, 降低零先验)

# 复合重建损失权重
PEARSON_WEIGHT = 1.0           # Pearson 相关损失 (形状保持, 尺度无关)
AMPLITUDE_WEIGHT = 0.3         # 幅度比损失 (降低以优先分类精度)
MSE_WEIGHT = 0.5               # MSE 基线稳定性 (提高)

# Focal Loss 超参数
FOCAL_GAMMA = 2.0              # Focal Loss gamma (聚焦难例程度)
FOCAL_ALPHA = None             # Focal Loss alpha (类别权重, None=均匀)

# Flow Matching (FM) 超参数
FM_LAMBDA = 0.5                # FM 损失权重 (相对分类损失)
FM_LATENT_DIM = 128            # FM 压缩潜在维度
FM_K_STEPS = 4                 # ODE 积分步数 (推理时)
FM_N_SAMPLES = 3               # 多采样次数 (不确定性估计)
FM_UNCERTAINTY_THRESH = 0.3    # FM 不确定性阈值 (超过此值 → 保守恢复)
FM_TIME_EMBED_DIM = 64         # 正弦时间嵌入维度
SPECTRAL_WEIGHT = 0.05         # 频谱损失 (降低以避免干扰分类)


# ============================================================================
# 1. 数据集 (直接从 .npy 加载)
# ============================================================================

class PreprocessedDataset(Dataset):
    """加载预处理的 .npy 窗口数据

    数据已在 preprocess_data.py 中完成:
      - 滑动窗口提取
      - RobustScaler + 物理归一化
      - 训练/验证划分

    A0 降采样: 训练时每个 epoch 随机丢弃一部分 A0 窗口，
    减少解码器"输出≈0"的先验偏差。
    """

    def __init__(self, data_dir: str = DATA_DIR, split: str = 'train',
                 downsample_a0: float = 0.0):
        """
        Args:
            downsample_a0: A0 降采样比例 (0.0=不降采样, 0.5=丢弃一半A0, 仅训练集生效)
        """
        self.split = split
        # mmap 模式: 多进程共享 OS 文件缓存
        self.X = np.load(os.path.join(data_dir, f'X_{split}.npy'), mmap_mode='r')
        self.cls_labels = np.load(os.path.join(data_dir, f'Y_{split}_cls.npy'), mmap_mode='r')
        self.atk_labels = np.load(os.path.join(data_dir, f'Y_{split}_atk.npy'), mmap_mode='r')

        # A0 降采样: 仅训练集
        self._active_indices = np.arange(len(self.cls_labels))
        if downsample_a0 > 0 and split == 'train':
            a0_idx = np.where(self.cls_labels == 0)[0]
            n_keep = int(len(a0_idx) * (1.0 - downsample_a0))
            if n_keep < len(a0_idx):
                rng = np.random.RandomState(42)
                a0_keep = rng.choice(a0_idx, size=max(n_keep, 1000), replace=False)
                non_a0_idx = np.where(self.cls_labels != 0)[0]
                self._active_indices = np.sort(np.concatenate([a0_keep, non_a0_idx]))
                print(f"[Dataset] A0 降采样 {downsample_a0:.0%}: "
                      f"{len(a0_idx)}→{len(a0_keep)} A0 窗口")

        self._compute_class_weights()
        print(f"[Dataset] {split}: {len(self):,} 窗口, X={self.X.shape}")

    def _compute_class_weights(self):
        class_counts = defaultdict(int)
        for idx in self._active_indices:
            lbl = self.cls_labels[idx]
            class_counts[ALL_ATTACK_TYPES[lbl]] += 1
        total = len(self._active_indices)
        self.class_weights = torch.zeros(len(ALL_ATTACK_TYPES))
        for i, atk in enumerate(ALL_ATTACK_TYPES):
            count = class_counts.get(atk, 0)
            self.class_weights[i] = total / max(count, 1) / len(ALL_ATTACK_TYPES)

        self.sample_weights = np.array([
            self.class_weights[self.cls_labels[idx]].item()
            for idx in self._active_indices
        ])

    def __len__(self) -> int:
        return len(self._active_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_idx = self._active_indices[idx]
        return (torch.from_numpy(self.X[real_idx]),
                torch.tensor(self.cls_labels[real_idx], dtype=torch.long),
                torch.from_numpy(self.atk_labels[real_idx]))


# ============================================================================
# 2. U-Net 构建块 (参考 TCE_SS SK 残差块设计)
# ============================================================================

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


# ============================================================================
# 3. 分类器-解码器
# ============================================================================

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


# ============================================================================
# 3b. FreqAware 分类器 — 多尺度频率感知双路径架构 (v3)
# ============================================================================

class FreqAwareClassifier(nn.Module):
    """多尺度频率感知攻击分类与重建网络 (v3)

    核心创新:
      1. 频率路径保持 50 步分辨率 (Nyquist=5Hz) — 覆盖 A7 4Hz 扫频
      2. 正弦位置编码 — 解码器感知绝对时间位置，可生成时变频率
      3. 膨胀卷积块 (rate 1,2,4) — 多尺度时间上下文
      4. FiLM 调制 — 分类嵌入调制解码器，类别条件生成
      5. DC 偏置支路 — latent→FC→3，提供恒定/阶跃直流分量
      6. 3 层解码器 — 比旧版(2层)更深，更好表达复杂波形

    参数量: ~900K (旧 AttackClassifier 1.96M 的 46%)
    推理速度: 与旧版相当 (频率路径 50 步而非 25 步但通道更少)
    """

    def __init__(self, in_channels: int = 5, window_size: int = 100,
                 latent_dim: int = LATENT_DIM, num_classes: int = 9,
                 enc_channels: list = None,
                 freq_channels: list = None,
                 class_embed_dim: int = CLASS_EMBED_DIM,
                 use_fm: bool = False,
                 fm_latent_dim: int = FM_LATENT_DIM):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.use_fm = use_fm
        self.fm_latent_dim = fm_latent_dim

        enc_ch = enc_channels or ENC_CHANNELS
        freq_ch = freq_channels or FREQ_CHANNELS

        # ================================================================
        # 分类编码器 (ResDownBlock×4, 加宽通道)
        # 5→48→96→192→256, 100→50→25→13→7 步
        # ================================================================
        prev_ch = in_channels
        self.enc_blocks = nn.ModuleList()
        for i, ch in enumerate(enc_ch):
            ks = 7 if i == 0 else 5 if i < 3 else 3
            self.enc_blocks.append(ResDownBlock(prev_ch, ch, kernel_size=ks))
            prev_ch = ch

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

        # ---- 分类嵌入 (用于 FiLM 调制) ----
        self.class_embedding = nn.Parameter(
            torch.randn(num_classes, class_embed_dim) * 0.02)

        # ---- DC 偏置支路: latent → 3 通道直流分量 ----
        self.dc_fc = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
        )

        # ================================================================
        # 频率保持路径: 保持 50 步分辨率 (Nyquist = 50/(2*5s) = 5Hz)
        # 5→freq_ch[0]→freq_ch[1], 100→50→50 步 (仅一次 stride-2)
        # ================================================================
        self.freq_conv1 = nn.Conv1d(in_channels, freq_ch[0],
                                     kernel_size=7, stride=2, padding=3, bias=False)
        self.freq_bn1 = nn.BatchNorm1d(freq_ch[0])
        self.freq_conv2 = nn.Conv1d(freq_ch[0], freq_ch[1],
                                     kernel_size=5, stride=1, padding=2, bias=False)
        self.freq_bn2 = nn.BatchNorm1d(freq_ch[1])
        self.freq_feat_size = (window_size + 1) // 2  # 50

        # ================================================================
        # 正弦位置编码: 为解码器提供绝对时间位置信息
        # 使解码器能生成时变频率模式 (A7 扫频关键)
        # ================================================================
        self.register_buffer('_pos_enc',
            self._build_sinusoidal_encoding(self.freq_feat_size, freq_ch[1]))

        # ================================================================
        # 多尺度膨胀卷积块: 同时捕获不同时间尺度的上下文
        # rate=1 (局部高频), rate=2 (中频), rate=4 (低频包络)
        # ================================================================
        self.dil_conv1 = nn.Conv1d(freq_ch[1], freq_ch[1], kernel_size=3,
                                    dilation=1, padding=1, bias=False)
        self.dil_conv2 = nn.Conv1d(freq_ch[1], freq_ch[1], kernel_size=3,
                                    dilation=2, padding=2, bias=False)
        self.dil_conv4 = nn.Conv1d(freq_ch[1], freq_ch[1], kernel_size=3,
                                    dilation=4, padding=4, bias=False)
        self.dil_bn = nn.BatchNorm1d(freq_ch[1])

        # ================================================================
        # FiLM 调制网络: class_embed → (scale, shift) 每层解码器
        # ================================================================
        self.film_net1 = nn.Sequential(
            nn.Linear(class_embed_dim, freq_ch[1] * 2),
        )  # 调制膨胀卷积输出
        self.film_net2 = nn.Sequential(
            nn.Linear(class_embed_dim, 24 * 2),
        )  # 调制 dec_up1 输出
        self.film_net3 = nn.Sequential(
            nn.Linear(class_embed_dim, 16 * 2),
        )  # 调制 dec_conv2 输出

        # ================================================================
        # [FM] Conditional Flow Matching — 生成式解码器潜在桥梁
        # 仅在 use_fm=True 时构建。参数量: ~83K
        # ================================================================
        if use_fm:
            # 编码器 latent → FM 目标空间投影 (Linear, 无激活)
            self.dec_latent_proj = nn.Linear(latent_dim, fm_latent_dim)

            # FM 速度场 MLP: z_t(128) + t_emb(64) + class_embed(48) = 240 → 128 → 128
            fm_input_dim = fm_latent_dim + FM_TIME_EMBED_DIM + class_embed_dim
            self.fm_vecfield = nn.Sequential(
                nn.Linear(fm_input_dim, fm_latent_dim),
                nn.SiLU(),
                nn.Linear(fm_latent_dim, fm_latent_dim),
            )

            # FM latent → FiLM 调制参数注入频率特征 f(32, 50)
            self.dec_latent_adapter = nn.Sequential(
                nn.Linear(fm_latent_dim, 64),
                nn.SiLU(),
                nn.Linear(64, freq_ch[1] * 2),  # gamma(32) + beta(32)
            )

        # ================================================================
        # 解码器: 3 层 (50→100→100→100)
        # 比旧版(25→50→100)更深，每层 100 步分辨率处理更多细节
        # ================================================================
        self.dec_up1 = nn.ConvTranspose1d(freq_ch[1], 24,
                                           kernel_size=5, stride=2,
                                           padding=2, output_padding=1, bias=False)
        self.dec_bn1 = nn.BatchNorm1d(24)
        self.dec_conv2 = nn.Conv1d(24, 16, kernel_size=5, padding=2, bias=False)
        self.dec_bn2 = nn.BatchNorm1d(16)
        self.output_conv = nn.Conv1d(16, 3, kernel_size=5, padding=2)

        self._init_weights()

        # FM 向量场近零初始化: 初始速度场 ≈ 0 → 初始流 ≈ 恒等映射
        if use_fm:
            last_linear = self.fm_vecfield[-1]
            nn.init.normal_(last_linear.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(last_linear.bias)
            adapter_last = self.dec_latent_adapter[-1]
            nn.init.normal_(adapter_last.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(adapter_last.bias)

    @staticmethod
    def _build_sinusoidal_encoding(steps: int, channels: int) -> torch.Tensor:
        """构建正弦位置编码 (Transformer 风格)"""
        position = torch.arange(steps).float().unsqueeze(1)  # (steps, 1)
        div_term = torch.exp(torch.arange(0, channels, 2).float()
                           * -(math.log(10000.0) / channels))
        pe = torch.zeros(1, channels, steps)
        pe[0, 0::2, :] = torch.sin(position * div_term).T
        pe[0, 1::2, :] = torch.cos(position * div_term).T
        return pe  # (1, channels, steps)

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

    def forward(self, x: torch.Tensor, z_dec_override: torch.Tensor = None):
        """前向传播

        Args:
            x: (B, W, C) 输入窗口 [internal_innovation(3) + u_cmd(2)]
            z_dec_override: (B, fm_latent_dim) FM 集成潜在向量, 仅推理时使用.
                           为 None 时使用 teacher forcing (encoder 投影目标).

        Returns:
            cls_logits:  (B, 9) 分类 logits
            attack_seq:  (B, W, 3) 攻击信号重建
            z:           (B, latent_dim) 潜在表示
        """
        B = x.shape[0]
        x_conv = x.permute(0, 2, 1)  # (B, C, W)

        # ================================================================
        # 1. 分类编码器 → latent → classifier
        # ================================================================
        h = x_conv
        for block in self.enc_blocks:
            h = block(h)

        h = F.relu(self.bottleneck_bn(self.bottleneck_conv(h)))
        h = self.global_pool(h).squeeze(-1)
        z = F.relu(self.enc_fc(h))  # (B, latent_dim)

        # 分类
        c = F.relu(self.cls_bn1(self.cls_fc1(z)))
        c = self.cls_dropout1(c)
        c = F.relu(self.cls_bn2(self.cls_fc2(c)))
        c = self.cls_dropout2(c)
        cls_logits = self.cls_head(c)

        # ================================================================
        # 2. 分类软嵌入 (可微分 FiLM 条件)
        # ================================================================
        cls_probs = F.softmax(cls_logits, dim=1)
        class_embed = cls_probs @ self.class_embedding.to(cls_probs.dtype)  # (B, class_embed_dim)

        # DC 偏置: latent → 3 通道直流分量
        dc_offset = self.dc_fc(z)  # (B, 3)

        # [FM] 解码器潜在变量: teacher forcing 目标 或 ODE 集成结果
        if self.use_fm:
            if z_dec_override is not None:
                z_dec = z_dec_override
            else:
                z_dec = self.dec_latent_proj(z)  # (B, fm_latent_dim), teacher forcing

        # ================================================================
        # 3. 频率保持路径: 50 步分辨率 (Nyquist=5Hz)
        # ================================================================
        f = F.leaky_relu(self.freq_bn1(self.freq_conv1(x_conv)), 0.2)  # (B, 24, 50)
        f = F.leaky_relu(self.freq_bn2(self.freq_conv2(f)), 0.2)        # (B, 32, 50)

        # 正弦位置编码: 解码器知道绝对时间位置 → 可生成时变频率
        f = f + self._pos_enc  # (B, 32, 50)

        # ================================================================
        # 4. 多尺度膨胀卷积: 同时捕获不同时间尺度
        # ================================================================
        f_d1 = self.dil_conv1(f)  # rate=1: 局部高频细节
        f_d2 = self.dil_conv2(f)  # rate=2: 中频模式
        f_d4 = self.dil_conv4(f)  # rate=4: 低频包络
        f = f + F.leaky_relu(self.dil_bn(f_d1 + f_d2 + f_d4), 0.2)  # 残差融合

        # ================================================================
        # [FM] 解码器潜在变量 → FiLM 调制注入频率特征
        # 将 encoder 的知识通过生成式桥梁注入 decoder 输入
        # ================================================================
        if self.use_fm:
            fm_film = self.dec_latent_adapter(z_dec)  # (B, 64)
            gamma_fm, beta_fm = fm_film.chunk(2, dim=1)  # (B, 32) each
            f = gamma_fm.unsqueeze(-1) * f + beta_fm.unsqueeze(-1)  # channel-wise modulation

        # ================================================================
        # 5. FiLM 调制 + 解码 (50→100→100→100)
        # ================================================================
        # FiLM 1: 调制上采样输入
        film1 = self.film_net1(class_embed)
        gamma1, beta1 = film1.chunk(2, dim=1)
        f = gamma1.unsqueeze(-1) * f + beta1.unsqueeze(-1)
        f = F.leaky_relu(self.dec_bn1(self.dec_up1(f)), 0.2)  # (B, 24, 100)

        # FiLM 2
        film2 = self.film_net2(class_embed)
        gamma2, beta2 = film2.chunk(2, dim=1)
        f = gamma2.unsqueeze(-1) * f + beta2.unsqueeze(-1)
        f = F.leaky_relu(self.dec_bn2(self.dec_conv2(f)), 0.2)  # (B, 16, 100)

        # FiLM 3
        film3 = self.film_net3(class_embed)
        gamma3, beta3 = film3.chunk(2, dim=1)
        f = gamma3.unsqueeze(-1) * f + beta3.unsqueeze(-1)
        attack_seq = self.output_conv(f)  # (B, 3, 100)

        # 加直流分量
        attack_seq = attack_seq + dc_offset.unsqueeze(-1)
        attack_seq = attack_seq.permute(0, 2, 1)  # (B, W, 3)

        return cls_logits, attack_seq, z

    # ------------------------------------------------------------------
    # FM 辅助方法
    # ------------------------------------------------------------------

    def fm_sample_and_predict(self, z: torch.Tensor,
                              class_embed: torch.Tensor) -> dict:
        """FM 训练采样: 噪声 → 随机时间点 → 预测速度场

        用于训练时计算 FM loss。对 encoder latent z 投影得到目标 z_dec_target，
        在随机时间 t 采样噪声，预测速度场 v_θ(t, z_t, c)。

        Args:
            z:           (B, latent_dim) encoder 输出, 用于生成 FM 目标
            class_embed: (B, class_embed_dim) 分类软嵌入, 作为 FM 条件

        Returns:
            dict: {
                'z_dec_target': (B, fm_latent_dim) FM 回归目标,
                'z0':          (B, fm_latent_dim) 初始噪声,
                't':           (B, 1) 随机时间,
                'z_t':         (B, fm_latent_dim) 插值潜在变量,
                'v_pred':      (B, fm_latent_dim) 预测速度,
                'fm_loss':     scalar, MSE(v_pred, z_dec_target - z0)
            }
        """
        B = z.shape[0]
        device = z.device
        z_dec_target = self.dec_latent_proj(z)
        z0 = torch.randn(B, self.fm_latent_dim, device=device)
        t = torch.rand(B, 1, device=device)

        # 线性概率路径: z_t = (1-t)·z0 + t·z_dec_target
        z_t = (1 - t) * z0 + t * z_dec_target

        # 正弦时间嵌入
        t_emb = self._fm_time_embedding(t)  # (B, FM_TIME_EMBED_DIM)

        # 条件拼接: [z_t, t_emb, class_embed]
        cond = torch.cat([z_t, t_emb, class_embed], dim=1)
        v_pred = self.fm_vecfield(cond)

        # 目标速度: z_dec_target - z0 (线性路径上的常数速度)
        fm_loss = F.mse_loss(v_pred, z_dec_target - z0)

        return {
            'z_dec_target': z_dec_target,
            'z0': z0, 't': t, 'z_t': z_t,
            'v_pred': v_pred, 'fm_loss': fm_loss,
        }

    def _fm_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """正弦时间嵌入 (Transformer 风格)

        Args:
            t: (B, 1) 标量时间 ∈ [0, 1]

        Returns:
            (B, FM_TIME_EMBED_DIM) 正弦时间嵌入
        """
        B = t.shape[0]
        device = t.device
        half_dim = FM_TIME_EMBED_DIM // 2
        # 频率: exp(-log(10000) * arange(0, half_dim) / half_dim)
        freq = torch.exp(-math.log(10000.0) *
                        torch.arange(0, half_dim, device=device).float() / half_dim)
        arg = t * freq.unsqueeze(0)  # (B, half_dim)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=1)  # (B, FM_TIME_EMBED_DIM)

    @torch.no_grad()
    def fm_integrate(self, class_embed: torch.Tensor,
                     k_steps: int = FM_K_STEPS) -> torch.Tensor:
        """K 步 Euler ODE 积分 — 推理时从噪声生成 decoder latent

        Args:
            class_embed: (B, class_embed_dim) 分类软嵌入 (FM 条件)
            k_steps:     ODE 积分步数 (默认 4)

        Returns:
            z_dec_hat: (B, fm_latent_dim) 积分后的潜在向量
        """
        B = class_embed.shape[0]
        device = class_embed.device
        z = torch.randn(B, self.fm_latent_dim, device=device)
        dt = 1.0 / k_steps

        for i in range(k_steps):
            t_val = i * dt
            t = torch.full((B, 1), t_val, device=device)
            t_emb = self._fm_time_embedding(t)
            cond = torch.cat([z, t_emb, class_embed], dim=1)
            v = self.fm_vecfield(cond)
            z = z + v * dt

        return z

    @torch.no_grad()
    def fm_integrate_multi(self, class_embed: torch.Tensor,
                           k_steps: int = FM_K_STEPS,
                           n_samples: int = FM_N_SAMPLES) -> torch.Tensor:
        """多次 ODE 积分 — 用于不确定性估计

        Args:
            class_embed: (B, class_embed_dim)
            k_steps:     ODE 积分步数
            n_samples:   采样次数

        Returns:
            z_samples: (N, B, fm_latent_dim) 多个积分结果
        """
        samples = []
        for _ in range(n_samples):
            z_hat = self.fm_integrate(class_embed, k_steps)
            samples.append(z_hat)
        return torch.stack(samples, dim=0)


# ============================================================================
# 4. Focal Loss 和复合重建损失
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    自动聚焦于难分类样本, 降低易分类样本的损失贡献。
    对 A1(偏置)/A3(斜坡) 等难检测攻击类型特别有效。
    """
    def __init__(self, gamma: float = FOCAL_GAMMA, alpha=None,
                 label_smoothing: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Tensor of shape (num_classes,) or None
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none',
                                   label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            at = self.alpha.gather(0, targets)
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def composite_recon_loss(atk_pred, atk_seq, cls_label,
                          pearson_w: float = PEARSON_WEIGHT,
                          amplitude_w: float = AMPLITUDE_WEIGHT,
                          mse_w: float = MSE_WEIGHT,
                          spectral_w: float = SPECTRAL_WEIGHT,
                          a0_weight: float = A0_RECON_WEIGHT):
    """复合重建损失: 分离形状、幅度、频域监督

    针对旧版"输出≈0"先验问题的根本修复:
      - Pearson 相关损失: 尺度无关的形状保持 (解决幅度低估)
      - 幅度比损失: 对数尺度惩罚幅度不匹配
      - MSE: 基线平滑监督
      - 频谱损失: 频域精度 (A7 扫频关键)

    A0 窗口(target≈0): Pearson/幅度/频谱无定义, 退化为纯 MSE。
    """
    B = atk_pred.shape[0]
    device = atk_pred.device

    # 展平用于相关计算
    pred_f = atk_pred.reshape(B, -1)   # (B, W*3)
    target_f = atk_seq.reshape(B, -1)

    # ---- 1. Pearson 相关损失 (1 - |r|): 尺度无关形状保持 ----
    pred_m = pred_f.mean(dim=1, keepdim=True)
    target_m = target_f.mean(dim=1, keepdim=True)
    pred_c = pred_f - pred_m
    target_c = target_f - target_m

    pred_std = torch.sqrt((pred_c ** 2).sum(dim=1) + 1e-8)
    target_std = torch.sqrt((target_c ** 2).sum(dim=1) + 1e-8)

    correlation = (pred_c * target_c).sum(dim=1) / (pred_std * target_std)
    pearson_loss = 1.0 - correlation.abs()  # (B,)

    # ---- 2. 幅度比损失: log(|pred_std / target_std|) ----
    amp_ratio = pred_std / target_std.clamp(min=1e-8)
    amplitude_loss = torch.abs(torch.log(amp_ratio.clamp(min=1e-4, max=1e4)))

    # ---- 3. MSE 损失 ----
    mse_loss = ((atk_pred - atk_seq) ** 2).mean(dim=[1, 2])  # (B,)

    # ---- 4. 频谱损失: FFT 幅度差 (零填充到 128 以兼容 FP16 cuFFT) ----
    # cuFFT 在半精度下要求信号长度为 2 的幂
    n_fft = 128  # 下一个 2 的幂 > 100
    pred_pad = F.pad(atk_pred.permute(0, 2, 1), (0, n_fft - atk_pred.shape[1]))  # (B, 3, 128)
    target_pad = F.pad(atk_seq.permute(0, 2, 1), (0, n_fft - atk_seq.shape[1]))
    pred_fft = torch.fft.rfft(pred_pad.float(), dim=2, norm='ortho').abs()  # (B, 3, 65)
    target_fft = torch.fft.rfft(target_pad.float(), dim=2, norm='ortho').abs()
    spectral_loss = ((pred_fft - target_fft) ** 2).mean(dim=[1, 2])  # (B,)

    # ---- 逐样本加权: A0 只用 MSE, 非 A0 用全部组件 ----
    is_a0 = (cls_label == 0)
    # A0: 纯 MSE * a0_weight (降低零先验)
    loss_a0 = mse_loss * a0_weight
    # 非 A0: 全部组件
    loss_attack = (pearson_w * pearson_loss +
                   amplitude_w * amplitude_loss +
                   mse_w * mse_loss +
                   spectral_w * spectral_loss)

    loss = torch.where(is_a0, loss_a0, loss_attack)
    return loss.mean()


def _per_sample_recon_loss(atk_pred, atk_seq, cls_label, *args, **kwargs):
    """向后兼容包装: 旧版 per-class weighted MSE"""
    # 当 use_per_class_weight=False 时回退用此函数
    per_sample_mse = ((atk_pred - atk_seq) ** 2).mean(dim=[1, 2])
    a0_weight = kwargs.get('a0_weight', A0_RECON_WEIGHT)
    weights = torch.where(cls_label == 0,
                          torch.tensor(a0_weight, device=cls_label.device),
                          torch.tensor(1.0, device=cls_label.device))
    return (per_sample_mse * weights).mean()


def train_epoch(model, dataloader, optimizer, scheduler, criterion_cls,
                device, recon_lambda, scaler=None, use_composite_loss=True,
                fm_lambda=FM_LAMBDA):
    model.train()
    total_loss = total_cls = total_recon = total_fm = 0.0
    correct = total = 0

    for x, cls_label, atk_seq in dataloader:
        x, cls_label = x.to(device, non_blocking=True), cls_label.to(device, non_blocking=True)
        atk_seq = atk_seq.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                cls_logits, atk_pred, z = model(x)
                loss_cls = criterion_cls(cls_logits, cls_label)
                if use_composite_loss:
                    loss_recon = composite_recon_loss(atk_pred, atk_seq, cls_label)
                else:
                    loss_recon = F.mse_loss(atk_pred, atk_seq)
                if model.use_fm:
                    cls_probs = F.softmax(cls_logits, dim=1)
                    class_embed = cls_probs @ model.class_embedding.to(cls_probs.dtype)
                    fm_result = model.fm_sample_and_predict(z, class_embed)
                    loss_fm = fm_result['fm_loss']
                    loss = loss_cls + fm_lambda * loss_fm + recon_lambda * loss_recon
                else:
                    loss_fm = torch.tensor(0.0, device=device)
                    loss = loss_cls + recon_lambda * loss_recon
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            cls_logits, atk_pred, z = model(x)
            loss_cls = criterion_cls(cls_logits, cls_label)
            if use_composite_loss:
                loss_recon = composite_recon_loss(atk_pred, atk_seq, cls_label)
            else:
                loss_recon = F.mse_loss(atk_pred, atk_seq)
            if model.use_fm:
                cls_probs = F.softmax(cls_logits, dim=1)
                class_embed = cls_probs @ model.class_embedding.to(cls_probs.dtype)
                fm_result = model.fm_sample_and_predict(z, class_embed)
                loss_fm = fm_result['fm_loss']
                loss = loss_cls + fm_lambda * loss_fm + recon_lambda * loss_recon
            else:
                loss_fm = torch.tensor(0.0, device=device)
                loss = loss_cls + recon_lambda * loss_recon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_cls += loss_cls.item() * bs
        total_recon += loss_recon.item() * bs
        total_fm += loss_fm.item() * bs
        correct += (cls_logits.argmax(dim=1) == cls_label).sum().item()
        total += bs

    n = max(total, 1)
    return total_loss / n, total_cls / n, total_recon / n, total_fm / n, correct / n


@torch.no_grad()
def evaluate(model, dataloader, criterion_cls,
             device, recon_lambda, use_composite_loss=True):
    model.eval()
    total_loss = total_cls = total_recon = total_fm_recon = 0.0
    correct = total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    recon_by_class = defaultdict(list)

    for x, cls_label, atk_seq in dataloader:
        x, cls_label = x.to(device, non_blocking=True), cls_label.to(device, non_blocking=True)
        atk_seq = atk_seq.to(device, non_blocking=True)

        if USE_AMP:
            with torch.amp.autocast('cuda'):
                cls_logits, atk_pred, _ = model(x)
        else:
            cls_logits, atk_pred, _ = model(x)

        loss_cls = criterion_cls(cls_logits, cls_label)
        if use_composite_loss:
            loss_recon = composite_recon_loss(atk_pred, atk_seq, cls_label)
        else:
            loss_recon = F.mse_loss(atk_pred, atk_seq)
        loss = loss_cls + recon_lambda * loss_recon

        # FM 集成重建质量监控 (仅 FM 模型, 第一个 batch 以避免 OOM)
        if model.use_fm and total == 0:
            cls_probs = F.softmax(cls_logits, dim=1)
            class_embed = cls_probs @ model.class_embedding.to(cls_probs.dtype)
            z_dec_int = model.fm_integrate(class_embed.float(), k_steps=FM_K_STEPS)
            _, atk_pred_fm, _ = model(x, z_dec_override=z_dec_int)
            if use_composite_loss:
                loss_fm_recon = composite_recon_loss(atk_pred_fm, atk_seq, cls_label)
            else:
                loss_fm_recon = F.mse_loss(atk_pred_fm, atk_seq)
            total_fm_recon += loss_fm_recon.item() * x.size(0)
        elif model.use_fm:
            total_fm_recon += loss_recon.item() * x.size(0)  # 占位, 非精确

        total_loss += loss.item() * x.size(0)
        total_cls += loss_cls.item() * x.size(0)
        total_recon += loss_recon.item() * x.size(0)

        pred = cls_logits.argmax(dim=1)
        correct += (pred == cls_label).sum().item()
        total += x.size(0)

        for i in range(len(cls_label)):
            gt = cls_label[i].item()
            class_total[gt] += 1
            if pred[i].item() == gt:
                class_correct[gt] += 1
            # MAE 用于报告
            err = torch.norm(atk_pred[i] - atk_seq[i], dim=-1).mean().item()
            recon_by_class[gt].append(err)

    n = max(total, 1)
    per_class_acc = {}
    for cls_idx in range(len(ALL_ATTACK_TYPES)):
        if class_total[cls_idx] > 0:
            per_class_acc[ALL_ATTACK_TYPES[cls_idx]] = (
                class_correct[cls_idx] / class_total[cls_idx])

    per_class_recon = {}
    for cls_idx in range(len(ALL_ATTACK_TYPES)):
        if recon_by_class[cls_idx]:
            per_class_recon[ALL_ATTACK_TYPES[cls_idx]] = np.mean(recon_by_class[cls_idx])

    return (total_loss / n, total_cls / n, total_recon / n,
            correct / n, per_class_acc, per_class_recon,
            total_fm_recon / max(total, 1))


def plot_curves(history: dict, save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('U-Net Attack AE Training', fontsize=14, fontweight='bold')
    epochs = range(1, len(history['train_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', alpha=0.7, label='Train')
    ax.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', alpha=0.7, label='Train')
    ax.plot(epochs, history['val_acc'], 'r-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history['val_recon'], 'r-', label='Val Recon MSE')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    ax.set_title('Attack Reconstruction MSE'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if history.get('val_per_class'):
        per_class = history['val_per_class'][-1]
        classes = list(per_class.keys())
        accs = [per_class[c] * 100 for c in classes]
        colors = ['#2ca02c' if a > 50 else '#d62728' for a in accs]
        ax.bar(classes, accs, color=colors, alpha=0.8)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Accuracy [%]')
        ax.set_title('Per-Class Accuracy (Final)')
        ax.set_ylim(0, 105); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot] {save_path}")


# ============================================================================
# 主训练流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='攻击分类 + 多尺度频率感知重建网络训练')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--recon-lambda', type=float, default=RECON_LAMBDA)
    parser.add_argument('--latent-dim', type=int, default=LATENT_DIM)
    parser.add_argument('--eval-only', type=str, default=None)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--model-type', type=str, default='freqaware',
                       choices=['baseline', 'freqaware'],
                       help='baseline=AttackClassifier, freqaware=FreqAwareClassifier')
    parser.add_argument('--downsample-a0', type=float, default=0.0,
                       help='A0 降采样比例 (0.0=不降采样, 默认0.0, 依赖WeightedSampler平衡)')
    parser.add_argument('--simple-loss', action='store_true',
                       help='使用简单 MSE 损失 (非复合损失), 更稳定训练')
    parser.add_argument('--focal-loss', action='store_true',
                       help='使用 Focal Loss (替代标准 CrossEntropy), 聚焦难分类样本')
    parser.add_argument('--focal-gamma', type=float, default=FOCAL_GAMMA,
                       help='Focal Loss gamma 参数 (默认 2.0)')
    # Flow Matching 参数
    parser.add_argument('--use-fm', action='store_true', default=False,
                       help='启用 Conditional Flow Matching 生成式解码器')
    parser.add_argument('--no-fm', dest='use_fm', action='store_false',
                       help='禁用 Flow Matching (默认)')
    parser.add_argument('--fm-lambda', type=float, default=FM_LAMBDA,
                       help='FM 损失权重 (默认 0.5)')
    parser.add_argument('--fm-latent-dim', type=int, default=FM_LATENT_DIM,
                       help='FM 压缩潜在维度 (默认 128)')
    parser.add_argument('--fm-steps', type=int, default=FM_K_STEPS,
                       help='ODE 积分步数 (默认 4)')
    args = parser.parse_args()

    # 验证预处理数据
    if not os.path.exists(os.path.join(args.data_dir, 'X_train.npy')):
        print(f"[ERROR] 预处理数据未找到: {args.data_dir}/X_train.npy")
        print(f"请先运行: python preprocess_data.py")
        sys.exit(1)

    model_type = args.model_type
    use_fm = args.use_fm and model_type == 'freqaware'  # FM 仅对 FreqAware 有效
    if args.use_fm and model_type != 'freqaware':
        print("[WARN] --use-fm 仅对 freqaware 模型有效, 已忽略")
    model_name = 'FreqAwareClassifier-v3' if model_type == 'freqaware' else 'AttackClassifier'
    if use_fm:
        model_name += '+FM'
    use_composite_loss = (model_type == 'freqaware' and not args.simple_loss)

    print("=" * 60)
    print(f"{model_name} (AMP+OneCycle)")
    print("=" * 60)
    print(f"  模型类型:    {model_name}")
    print(f"  设备:        {DEVICE}")
    if DEVICE.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU:         {gpu_name} ({gpu_mem:.0f}GB)")
    print(f"  AMP:         {USE_AMP}")
    print(f"  编码器通道:  {ENC_CHANNELS}")
    print(f"  频率通道:    {FREQ_CHANNELS} (50步分辨率, Nyquist=5Hz)")
    print(f"  潜在维度:    {args.latent_dim}")
    if use_fm:
        print(f"  FM:          启用 (latent={args.fm_latent_dim}, "
              f"lambda={args.fm_lambda}, K={args.fm_steps})")
    print(f"  Batch:       {args.batch_size} (workers={NUM_WORKERS})")
    print(f"  Epochs:      {args.epochs} (早停={EARLY_STOP_PATIENCE})")
    print(f"  LR 峰值:     {args.lr}")
    print(f"  标签平滑:    {LABEL_SMOOTHING}")
    print(f"  Recon lambda: {args.recon_lambda}")
    if use_composite_loss:
        print(f"  复合损失:    Pearson={PEARSON_WEIGHT} Amp={AMPLITUDE_WEIGHT} "
              f"MSE={MSE_WEIGHT} Spectral={SPECTRAL_WEIGHT}")
    else:
        print(f"  重建损失:    MSE (简单, 逐类加权)")
    print(f"  A0 降采样:   {args.downsample_a0:.0%}")
    print("=" * 60)

    # ---- 加载数据 ----
    train_dataset = PreprocessedDataset(args.data_dir, 'train',
                                         downsample_a0=args.downsample_a0)
    val_dataset = PreprocessedDataset(args.data_dir, 'val')

    sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=NUM_WORKERS,
                              pin_memory=True,
                              persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True,
                            persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)

    # ---- 模型 ----
    sample_x, _, _ = train_dataset[0]
    in_channels = sample_x.shape[1]
    window_size = sample_x.shape[0]

    if model_type == 'freqaware':
        model = FreqAwareClassifier(
            in_channels=in_channels, window_size=window_size,
            latent_dim=args.latent_dim, num_classes=len(ALL_ATTACK_TYPES),
            use_fm=use_fm, fm_latent_dim=args.fm_latent_dim,
        ).to(DEVICE)
    else:
        model = AttackClassifier(
            in_channels=in_channels, window_size=window_size,
            latent_dim=args.latent_dim, num_classes=len(ALL_ATTACK_TYPES)
        ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {n_params:,} (~{n_params/1e6:.2f}M)")

    # 仅评估
    if args.eval_only:
        print(f"\n加载模型: {args.eval_only}")
        state = torch.load(args.eval_only, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state, strict=False)
        criterion_cls = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        _, _, _, val_acc, per_class, per_class_recon, _ = evaluate(
            model, val_loader, criterion_cls, DEVICE,
            args.recon_lambda, use_composite_loss=False)
        print(f"\n验证准确率: {val_acc:.4f}")
        print(f"\n每类准确率:")
        for cls_name in ALL_ATTACK_TYPES:
            acc = per_class.get(cls_name, 0.0)
            print(f"  {cls_name} ({ATTACK_NAMES_CN[cls_name]}): {acc*100:5.1f}%")
        print(f"\n每类重建误差 (MAE):")
        for cls_name in ALL_ATTACK_TYPES:
            err = per_class_recon.get(cls_name, float('nan'))
            print(f"  {cls_name} ({ATTACK_NAMES_CN[cls_name]}): {err:.4f}")
        return

    # ---- 优化器 & 调度器 ----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.12,           # 12% 步数用于热身 (更长, 更稳定)
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos')
    print(f"  Scheduler:    OneCycleLR (warmup={total_steps*0.12:.0f} steps, "
          f"total={total_steps})")

    # 损失函数
    if args.focal_loss:
        criterion_cls = FocalLoss(gamma=args.focal_gamma,
                                   label_smoothing=LABEL_SMOOTHING)
        print(f"  Criterion:    FocalLoss (gamma={args.focal_gamma})")
    else:
        criterion_cls = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        print(f"  Criterion:    CrossEntropyLoss (label_smoothing={LABEL_SMOOTHING})")

    # 混合精度梯度缩放器
    scaler = torch.amp.GradScaler('cuda') if USE_AMP and DEVICE.type == 'cuda' else None
    if scaler:
        print("  AMP scaler:   启用 (FP16 混合精度)")

    # ---- 训练 ----
    history = defaultdict(list)
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_path = os.path.join(MODEL_DIR, 'cls_best.pt')

    print(f"\n{'='*60}")
    print(f"开始训练 ({steps_per_epoch} steps/epoch)")
    print(f"{'='*60}")

    import time as _time
    for epoch in range(1, args.epochs + 1):
        t0 = _time.time()

        train_loss, train_cls, train_recon, train_fm, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion_cls,
            DEVICE, args.recon_lambda, scaler=scaler,
            use_composite_loss=use_composite_loss,
            fm_lambda=args.fm_lambda)

        val_loss, val_cls, val_recon, val_acc, per_class, per_class_recon, val_fm_recon = evaluate(
            model, val_loader, criterion_cls,
            DEVICE, args.recon_lambda, use_composite_loss=use_composite_loss)

        elapsed = _time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_recon'].append(val_recon)
        history['val_per_class'].append(per_class)
        history['val_per_class_recon'].append(per_class_recon)
        if use_fm:
            history['train_fm'].append(train_fm)
            history['val_fm_recon'].append(val_fm_recon)

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            marker = f' * (best, epoch {epoch})'
        else:
            epochs_no_improve += 1
            marker = ''

        lr_now = optimizer.param_groups[0]['lr']
        fm_str = f" | FM={train_fm:.4f}" if use_fm else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"LR={lr_now:.1e} | {elapsed:.0f}s | "
              f"Acc={val_acc:.3f} (best={best_val_acc:.3f}) | "
              f"Recon={val_recon:.4f}{fm_str} | "
              f"未改善={epochs_no_improve}{marker}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n早停: {EARLY_STOP_PATIENCE} epochs 未改善, 在 epoch {epoch} 停止")
            break

    # ---- 保存 ----
    final_path = os.path.join(MODEL_DIR, 'cls_final.pt')
    torch.save(model.state_dict(), final_path)

    config = {
        'in_channels': in_channels, 'window_size': window_size,
        'latent_dim': args.latent_dim, 'enc_channels': ENC_CHANNELS,
        'num_classes': len(ALL_ATTACK_TYPES),
        'model_type': model_type,
        'use_fm': use_fm,
        'fm_latent_dim': args.fm_latent_dim,
        'fm_k_steps': args.fm_steps,
    }
    if model_type == 'baseline':
        config['dec_channels'] = model.dec_channels
    else:
        config['freq_channels'] = FREQ_CHANNELS
        config['class_embed_dim'] = CLASS_EMBED_DIM
    np.savez(os.path.join(MODEL_DIR, 'cls_config.npz'),
             **{k: np.array(v) if isinstance(v, list) else v
                for k, v in config.items()})

    # ---- 最终评估 ----
    print(f"\n{'='*60}")
    print("最终评估 (最佳模型)")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    _, _, val_recon, val_acc, per_class, per_class_recon, val_fm_recon = evaluate(
        model, val_loader, criterion_cls, DEVICE,
        args.recon_lambda, use_composite_loss=False)

    print(f"\n最佳验证准确率: {val_acc:.4f} (总体)")
    print(f"攻击信号重建 MAE (teacher forcing): {val_recon:.6f}")
    if use_fm:
        print(f"攻击信号重建 MAE (FM集成 K={args.fm_steps}): {val_fm_recon:.6f}")
    print(f"\n每类准确率:")
    for cls_name in ALL_ATTACK_TYPES:
        acc = per_class.get(cls_name, 0.0)
        bar = '#' * int(acc * 50) + '.' * (50 - int(acc * 50))
        print(f"  {cls_name} ({ATTACK_NAMES_CN[cls_name]}): {bar} {acc*100:5.1f}%")

    print(f"\n每类重建误差 (MAE):")
    for cls_name in ALL_ATTACK_TYPES:
        err = per_class_recon.get(cls_name, float('nan'))
        bar = '#' * min(int(err * 200), 50) + '.' * max(50 - int(err * 200), 0)
        print(f"  {cls_name} ({ATTACK_NAMES_CN[cls_name]}): {bar} {err:.4f}")

    print(f"\n最佳 epoch: {best_epoch}/{epoch} | 最佳 Acc: {best_val_acc:.4f}")
    print(f"模型: {best_path} | {final_path}")

    if not args.no_plot:
        plot_curves(history, os.path.join(MODEL_DIR, 'cls_curves.png'))

    print(f"\n训练完成!")


if __name__ == "__main__":
    main()
