"""
train_cfm.py — 攻击分类检测器训练脚本 (cls-only 分支)
=========================================================
精简训练: 仅交叉熵分类损失。无流匹配、物理正则化、正交约束。

用法:
  python train_cfm.py                          # 默认训练 (因果卷积)
  python train_cfm.py --backbone transformer   # Transformer 骨干
  python train_cfm.py --eval-only models/cfm_cls_best.pt
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from collections import defaultdict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', message='Detected call of.*lr_scheduler.step.*before.*optimizer.step')
warnings.filterwarnings('ignore', message='.*non-writable tensor.*')

from detector.cfm_detector import (CFMDetector,
                                    D_MODEL, NUM_TRANSFORMER_LAYERS, NUM_HEADS,
                                    DIM_FEEDFORWARD,
                                    DILATIONS, CONV_KERNEL_SIZE)

# ============================================================================
# 攻击类型常量
# ============================================================================

ALL_ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
ATTACK_NAMES_CN = {
    'A0': '正常', 'A1': '恒定偏移', 'A2': '正弦注入',
    'A3': '斜坡漂移', 'A4': '阶跃', 'A5': '重放攻击',
    'A6': '信号丢失', 'A7': '缩放攻击', 'A8': '传感器冻结',
}

# ============================================================================
# 全局路径配置
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'dataset_win')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# 数据集类 (cls-only: 仅需 X + cls_label)
# ============================================================================

class PreprocessedDataset(Dataset):
    """加载预处理的 .npy 窗口数据 (分类任务)。"""

    def __init__(self, data_dir: str = DATA_DIR, split: str = 'train',
                 downsample_a0: float = 0.0):
        self.split = split
        self.X = np.load(os.path.join(data_dir, f'X_{split}.npy'), mmap_mode='r')
        self.cls_labels = np.load(os.path.join(data_dir, f'Y_{split}_cls.npy'), mmap_mode='r')

        # A0 降采样: 每 epoch 调用 resample_a0() 重新随机
        self._all_indices = np.arange(len(self.cls_labels))
        self._downsample_rate = downsample_a0
        self._active_indices = self._all_indices.copy()
        if downsample_a0 > 0 and split == 'train':
            a0_before = int(np.sum(self.cls_labels == 0))
            self.resample_a0()
            a0_after = int(np.sum(self.cls_labels[self._active_indices] == 0))
            print(f"[Dataset] A0 降采样 {downsample_a0:.0%}: "
                  f"{a0_before}→{a0_after} A0 窗口 (每 epoch 重新随机)")

        print(f"[Dataset] {split}: {len(self):,} 窗口, X={self.X.shape}")

    def resample_a0(self):
        """每 epoch 随机重新降采样 A0 窗口 (仅训练集, 无固定种子)。"""
        if self._downsample_rate <= 0 or self.split != 'train':
            return
        a0_idx = np.where(self.cls_labels == 0)[0]
        n_keep = int(len(a0_idx) * (1.0 - self._downsample_rate))
        if n_keep < len(a0_idx):
            a0_keep = np.random.choice(a0_idx, size=max(n_keep, 1000), replace=False)
            non_a0_idx = np.where(self.cls_labels != 0)[0]
            self._active_indices = np.sort(np.concatenate([a0_keep, non_a0_idx]))

    def __len__(self) -> int:
        return len(self._active_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self._active_indices[idx]
        return (torch.from_numpy(self.X[real_idx]),
                torch.tensor(self.cls_labels[real_idx], dtype=torch.long))


# ============================================================================
# 训练超参数
# ============================================================================
BATCH_SIZE = 1024
NUM_WORKERS = 2
PREFETCH_FACTOR = 4
LEARNING_RATE = 5e-4          # 固定学习率
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 150
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 50
GRAD_CLIP = 1.0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 1. 训练循环
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """单 epoch 训练 (纯 float32, 标准交叉熵)。"""
    model.train()
    total_loss = 0.0
    correct = total = 0

    for x, cls_label in dataloader:
        x = x.to(device, non_blocking=True)
        cls_label = cls_label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        cls_logits, _ = model(x)
        loss = F.cross_entropy(cls_logits, cls_label,
                               label_smoothing=LABEL_SMOOTHING)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (cls_logits.argmax(dim=1) == cls_label).sum().item()
        total += bs

    n = max(total, 1)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, dataloader, device):
    """评估: 分类准确率 (标准交叉熵, 无标签平滑, 无类别权重)"""
    model.eval()
    total_loss = 0.0
    correct = total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for x, cls_label in dataloader:
        x = x.to(device, non_blocking=True)
        cls_label = cls_label.to(device, non_blocking=True)

        cls_logits, _ = model(x)
        loss = F.cross_entropy(cls_logits, cls_label)

        B = x.size(0)
        total_loss += loss.item() * B
        pred = cls_logits.argmax(dim=1)
        correct += (pred == cls_label).sum().item()
        total += B

        for i in range(B):
            gt = cls_label[i].item()
            class_total[gt] += 1
            if pred[i].item() == gt:
                class_correct[gt] += 1

    n = max(total, 1)
    per_class_acc = {}
    for cls_idx in range(9):
        if class_total[cls_idx] > 0:
            per_class_acc[ALL_ATTACK_TYPES[cls_idx]] = (
                class_correct[cls_idx] / class_total[cls_idx])

    return total_loss / n, correct / n, per_class_acc


# ============================================================================
# 2. 主训练入口
# ============================================================================

def build_model(backbone_type='causal_conv'):
    """构建 CFMDetector (cls-only)。"""
    model = CFMDetector(
        backbone_type=backbone_type,
        dilations=DILATIONS,
        conv_kernel_size=CONV_KERNEL_SIZE,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 参数量: {n_params:,}")
    return model


def main():
    parser = argparse.ArgumentParser(description='CFM 分类检测器训练 (cls-only)')
    parser.add_argument('--backbone', type=str, default='causal_conv',
                        choices=['causal_conv', 'transformer'])
    parser.add_argument('--eval-only', type=str, default=None,
                        help='仅评估指定模型权重')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--downsample-a0', type=float, default=0.5,
                        help='A0 降采样比例')
    args = parser.parse_args()

    print(f"设备: {DEVICE}")
    print(f"骨干: {args.backbone}")
    print(f"精度: float32, 固定 LR={args.lr}")

    # ---- 数据集 ----
    train_dataset = PreprocessedDataset(split='train', downsample_a0=args.downsample_a0)
    val_dataset = PreprocessedDataset(split='val')
    test_dataset = PreprocessedDataset(split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=NUM_WORKERS,
                               pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
                               persistent_workers=(NUM_WORKERS > 0))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
                             persistent_workers=(NUM_WORKERS > 0))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
                              persistent_workers=(NUM_WORKERS > 0))

    # ---- 模型 ----
    model = build_model(backbone_type=args.backbone)
    model.to(DEVICE)

    # ---- 仅评估 ----
    if args.eval_only:
        print(f"\n加载权重: {args.eval_only}")
        state_dict = torch.load(args.eval_only, map_location=DEVICE, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  缺少键: {len(missing)}")
        if unexpected:
            print(f"  多余键: {len(unexpected)}")

        test_loss, test_acc, per_class_acc = evaluate(
        model, test_loader, DEVICE)
        print(f"\n测试集: Loss={test_loss:.4f}, Acc={test_acc:.4f}")
        print("各类别准确率:")
        for atk in ALL_ATTACK_TYPES:
            acc = per_class_acc.get(atk, 0)
            print(f"  {atk} ({ATTACK_NAMES_CN[atk]}): {acc:.4f}")
        return

    # ---- 优化器 ----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    print(f"  优化器: AdamW (lr={args.lr}, wd={WEIGHT_DECAY})")
    print(f"  调度器: ReduceLROnPlateau (factor=0.5, patience=10)")

    # ---- 训练循环 ----
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        train_dataset.resample_a0()  # 每 epoch 随机重新降采样 A0
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, DEVICE)
        val_loss, val_acc, _ = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印
        lr_now = optimizer.param_groups[0]['lr']
        print(f"E {epoch:3d} | LR={lr_now:.1e} | "
              f"T Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"V Loss={val_loss:.4f} Acc={val_acc:.4f}",
              end='')

        # 保存最佳
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, 'cfm_cls_best.pt'))
            print("  *", end='')
        else:
            patience_counter += 1

        print()

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"早停: {EARLY_STOP_PATIENCE} epoch 无提升 (最佳 epoch {best_epoch})")
            break

    # ---- 最终测试 ----
    print(f"\n加载最佳模型 (epoch {best_epoch}, Val Acc={best_val_acc:.4f})")
    state_dict = torch.load(os.path.join(MODEL_DIR, 'cfm_cls_best.pt'),
                            map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    test_loss, test_acc, per_class_acc = evaluate(
        model, test_loader, DEVICE)
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"  最佳 epoch: {best_epoch}")
    print(f"  测试 Loss:  {test_loss:.4f}")
    print(f"  测试 Acc:   {test_acc:.4f}")
    print(f"各类别准确率:")
    for atk in ALL_ATTACK_TYPES:
        acc = per_class_acc.get(atk, 0)
        print(f"    {atk} ({ATTACK_NAMES_CN[atk]}): {acc:.4f}")
    print(f"{'='*60}")

    # ---- 绘制训练曲线 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Classification Loss'); ax1.legend(); ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.axhline(y=test_acc, color='g', linestyle='--', label=f'Test={test_acc:.3f}')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Classification Accuracy'); ax2.legend(); ax2.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, 'training_curve.png'), dpi=150)
    plt.close(fig)
    print(f"训练曲线已保存至: {os.path.join(MODEL_DIR, 'training_curve.png')}")


if __name__ == "__main__":
    main()
