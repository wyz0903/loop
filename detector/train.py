"""
train.py — 攻击检测器训练脚本 (基线: 纯分类)
============================================
交叉熵分类训练, 多尺度膨胀深度可分离卷积骨干 + 注意力池化。

用法:
  python train.py                    # 默认训练
  python train.py --eval-only detector/models/nn_cls_best.pt
"""

import os, sys, argparse
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from detector.classifier import Detector
from attack import ALL_ATTACK_TYPES, ATTACK_NAMES
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'dataset_win')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def _latest_data_dir() -> str:
    """返回 dataset_win/ 下最新的时间戳子目录，若无则回退到根目录"""
    if not os.path.isdir(DATA_DIR):
        return DATA_DIR
    subdirs = sorted(
        [d for d in os.listdir(DATA_DIR)
         if os.path.isdir(os.path.join(DATA_DIR, d))],
        reverse=True)
    if subdirs:
        return os.path.join(DATA_DIR, subdirs[0])
    return DATA_DIR


# ============================================================================
# 超参数
# ============================================================================
BATCH_SIZE = 256
NUM_WORKERS = 2
PREFETCH_FACTOR = 4
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 150
LABEL_SMOOTHING = 0.0
EARLY_STOP_PATIENCE = 50
GRAD_CLIP = 1.0
LR_PATIENCE = 30
LR_FACTOR = 0.8
LR_MIN = 1e-6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 数据集
# ============================================================================

class PreprocessedDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, split='train', downsample_a0=0.0):
        self.split = split
        self.X = np.load(os.path.join(data_dir, f'X_{split}.npy'), mmap_mode='r')
        self.cls_labels = np.load(os.path.join(data_dir, f'Y_{split}_cls.npy'), mmap_mode='r')

        self._all_indices = np.arange(len(self.cls_labels))
        self._downsample_rate = downsample_a0
        self._active_indices = self._all_indices.copy()
        if downsample_a0 > 0 and split == 'train':
            a0_before = int(np.sum(self.cls_labels == 0))
            self.resample_a0()
            a0_after = int(np.sum(self.cls_labels[self._active_indices] == 0))
            print(f"[Dataset] A0 降采样 {downsample_a0:.0%}: {a0_before}→{a0_after} 窗口")

        print(f"[Dataset] {split}: {len(self):,} 窗口, X={self.X.shape}")

    def resample_a0(self):
        if self._downsample_rate <= 0 or self.split != 'train':
            return
        a0_idx = np.where(self.cls_labels == 0)[0]
        n_keep = int(len(a0_idx) * (1.0 - self._downsample_rate))
        if n_keep < len(a0_idx):
            a0_keep = np.random.choice(a0_idx, size=max(n_keep, 1000), replace=False)
            non_a0_idx = np.where(self.cls_labels != 0)[0]
            self._active_indices = np.sort(np.concatenate([a0_keep, non_a0_idx]))

    def __len__(self):
        return len(self._active_indices)

    def __getitem__(self, idx):
        real_idx = self._active_indices[idx]
        x = torch.from_numpy(self.X[real_idx])
        cls_label = torch.tensor(self.cls_labels[real_idx], dtype=torch.long)
        return x, cls_label


# ============================================================================
# 训练循环
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = total = 0

    for batch in dataloader:
        x, cls_label = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        cls_logits, _ = model(x)

        loss = F.cross_entropy(cls_logits, cls_label, label_smoothing=LABEL_SMOOTHING)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (cls_logits.argmax(1) == cls_label).sum().item()
        total += bs

    n = max(total, 1)
    return {'loss': total_loss / n, 'acc': correct / n}


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for batch in dataloader:
        x, cls_label = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

        cls_logits, _ = model(x)

        loss = F.cross_entropy(cls_logits, cls_label)

        B = x.size(0)
        total_loss += loss.item() * B
        pred = cls_logits.argmax(1)
        correct += (pred == cls_label).sum().item()
        total += B
        for i in range(B):
            gt = cls_label[i].item()
            class_total[gt] += 1
            if pred[i].item() == gt:
                class_correct[gt] += 1

    n = max(total, 1)
    per_class_acc = {ALL_ATTACK_TYPES[c]: class_correct[c] / max(class_total[c], 1)
                     for c in range(8) if class_total[c] > 0}
    return {'loss': total_loss / n, 'acc': correct / n, 'per_class_acc': per_class_acc}


# ============================================================================
# 主入口
# ============================================================================

def main():
    global LABEL_SMOOTHING
    parser = argparse.ArgumentParser(description='分类检测器训练')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='预处理后的数据目录 (默认: dataset_win/ 下最新子目录)')
    parser.add_argument('--eval-only', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--label-smoothing', type=float, default=LABEL_SMOOTHING)
    parser.add_argument('--downsample-a0', type=float, default=0.5)
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = _latest_data_dir()
        print(f"自动选择最新数据: {args.data_dir}")

    LABEL_SMOOTHING = args.label_smoothing

    print(f"设备: {DEVICE}, 骨干: MultiScaleDSConvBackbone + 注意力池化")
    print(f"A0 降采样: {args.downsample_a0:.0%}")

    # ---- 归一化参数 ----
    norm_data = np.load(os.path.join(args.data_dir, 'normalizer.npz'))
    norm_params = {k: norm_data[k] for k in ['ymeas_scale', 'ymeas_median', 'cmd_max']}

    # ---- 数据集 ----
    train_dataset = PreprocessedDataset(args.data_dir, split='train',
                                        downsample_a0=args.downsample_a0)
    val_dataset = PreprocessedDataset(args.data_dir, split='val')
    test_dataset = PreprocessedDataset(args.data_dir, split='test')

    def _make_loader(ds, bs, shuffle):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=NUM_WORKERS,
                         pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
                         persistent_workers=(NUM_WORKERS > 0))

    train_loader = _make_loader(train_dataset, args.batch_size, shuffle=True)
    val_loader = _make_loader(val_dataset, args.batch_size * 2, shuffle=False)
    test_loader = _make_loader(test_dataset, args.batch_size * 2, shuffle=False)

    # ---- 模型 ----
    model = Detector(
        ymeas_scale=norm_params['ymeas_scale'].tolist(),
        ymeas_median=norm_params['ymeas_median'].tolist(),
        cmd_max=norm_params['cmd_max'].tolist())
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 参数量: {n_params:,}")

    # ---- 仅评估 ----
    if args.eval_only:
        print(f"\n加载权重: {args.eval_only}")
        model.load_state_dict(torch.load(args.eval_only, map_location=DEVICE, weights_only=True), strict=False)
        result = evaluate(model, test_loader, DEVICE)
        print(f"\n测试集: Loss={result['loss']:.4f}, Acc={result['acc']:.4f}")
        for atk in ALL_ATTACK_TYPES:
            print(f"  {atk} ({ATTACK_NAMES[atk]}): {result['per_class_acc'].get(atk, 0):.4f}")
        return

    # ---- 训练 ----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        train_dataset.resample_a0()
        train_m = train_epoch(model, train_loader, optimizer, DEVICE)
        val_m = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_m['loss'])

        history['train_loss'].append(train_m['loss'])
        history['train_acc'].append(train_m['acc'])
        history['val_loss'].append(val_m['loss'])
        history['val_acc'].append(val_m['acc'])

        lr_now = optimizer.param_groups[0]['lr']
        print(f"E {epoch:3d} | LR={lr_now:.1e} | "
              f"T Loss={train_m['loss']:.4f} Acc={train_m['acc']:.4f} | "
              f"V Loss={val_m['loss']:.4f} Acc={val_m['acc']:.4f}",
              end='')

        if val_m['acc'] > best_val_acc:
            best_val_acc = val_m['acc']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'nn_cls_best.pt'))
            print("  *", end='')
        else:
            patience_counter += 1
        print()

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"早停: {EARLY_STOP_PATIENCE} epoch 无提升 (最佳 epoch {best_epoch})")
            break

    # ---- 最终测试 ----
    print(f"\n加载最佳模型 (epoch {best_epoch}, Val Acc={best_val_acc:.4f})")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'nn_cls_best.pt'),
                                     map_location=DEVICE, weights_only=True))
    test_m = evaluate(model, test_loader, DEVICE)
    print(f"\n{'='*60}")
    print(f"训练完成！ 最佳 epoch: {best_epoch}")
    print(f"  测试 Loss: {test_m['loss']:.4f}, Acc: {test_m['acc']:.4f}")
    for atk in ALL_ATTACK_TYPES:
        print(f"    {atk} ({ATTACK_NAMES[atk]}): {test_m['per_class_acc'].get(atk, 0):.4f}")
    print(f"{'='*60}")

    # ---- 训练曲线 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Cross-Entropy Loss'); ax1.legend(); ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.axhline(y=test_m['acc'], color='g', linestyle='--', label=f'Test={test_m["acc"]:.3f}')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Classification Accuracy'); ax2.legend(); ax2.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, 'training_curve.png'), dpi=150)
    plt.close(fig)
    print(f"训练曲线已保存至: {os.path.join(MODEL_DIR, 'training_curve.png')}")


if __name__ == "__main__":
    main()
