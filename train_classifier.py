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

# 从 detector 包导入模型和损失函数
from detector.attack_classifier import AttackClassifier
from detector.freq_aware_classifier import FreqAwareClassifier
from detector.losses import FocalLoss, composite_recon_loss, _per_sample_recon_loss
from detector.config import (ALL_ATTACK_TYPES as _ALL_ATK_TYPES,
                              ENC_CHANNELS, LATENT_DIM,
                              FREQ_CHANNELS, CLASS_EMBED_DIM)

# ============================================================================
# 全局配置
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'dataset_win', 'config')  # 默认 config 划分
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

ALL_ATTACK_TYPES = _ALL_ATK_TYPES
ATTACK_NAMES_CN = {
    'A0': '正常', 'A1': '恒定偏移', 'A2': '正弦注入',
    'A3': '斜坡漂移', 'A4': '阶跃', 'A5': '重放攻击',
    'A6': '信号丢失', 'A7': '缩放攻击', 'A8': '传感器冻结',
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

# ========================================================================
# 2. 模型和损失定义已移至 detector/ 包
#    - ResDownBlock, ResUpBlock → detector/nn_blocks.py
#    - AttackClassifier          → detector/attack_classifier.py
#    - FreqAwareClassifier       → detector/freq_aware_classifier.py
#    - FocalLoss, composite_recon_loss → detector/losses.py
# ========================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion_cls,
                device, recon_lambda, scaler=None, use_composite_loss=True):
    model.train()
    total_loss = total_cls = total_recon = 0.0
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
            loss = loss_cls + recon_lambda * loss_recon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_cls += loss_cls.item() * bs
        total_recon += loss_recon.item() * bs
        correct += (cls_logits.argmax(dim=1) == cls_label).sum().item()
        total += bs

    n = max(total, 1)
    return total_loss / n, total_cls / n, total_recon / n, correct / n


@torch.no_grad()
def evaluate(model, dataloader, criterion_cls,
             device, recon_lambda, use_composite_loss=True):
    model.eval()
    total_loss = total_cls = total_recon = 0.0
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
            correct / n, per_class_acc, per_class_recon)


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
    args = parser.parse_args()

    # 验证预处理数据
    if not os.path.exists(os.path.join(args.data_dir, 'X_train.npy')):
        print(f"[ERROR] 预处理数据未找到: {args.data_dir}/X_train.npy")
        print(f"请先运行: python preprocess_data.py")
        sys.exit(1)

    model_type = args.model_type
    model_name = 'FreqAwareClassifier-v4' if model_type == 'freqaware' else 'AttackClassifier'
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
    print(f"  Batch:       {args.batch_size} (workers={NUM_WORKERS})")
    print(f"  Epochs:      {args.epochs} (早停={EARLY_STOP_PATIENCE})")
    print(f"  LR 峰值:     {args.lr}")
    print(f"  标签平滑:    {LABEL_SMOOTHING}")
    print(f"  Recon lambda: {args.recon_lambda}")
    if use_composite_loss:
        print(f"  复合损失:    Pearson={PEARSON_WEIGHT} Amp={AMPLITUDE_WEIGHT} "
              f"MSE={MSE_WEIGHT} Spectral=0.05")
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
        _, _, _, val_acc, per_class, per_class_recon = evaluate(
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

        train_loss, train_cls, train_recon, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion_cls,
            DEVICE, args.recon_lambda, scaler=scaler,
            use_composite_loss=use_composite_loss)

        val_loss, val_cls, val_recon, val_acc, per_class, per_class_recon = evaluate(
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
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"LR={lr_now:.1e} | {elapsed:.0f}s | "
              f"Acc={val_acc:.3f} (best={best_val_acc:.3f}) | "
              f"Recon={val_recon:.4f} | "
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
    _, _, val_recon, val_acc, per_class, per_class_recon = evaluate(
        model, val_loader, criterion_cls, DEVICE,
        args.recon_lambda, use_composite_loss=False)

    print(f"\n最佳验证准确率: {val_acc:.4f} (总体)")
    print(f"攻击信号重建 MAE (teacher forcing): {val_recon:.6f}")
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
