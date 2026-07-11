"""
train.py — 攻击检测器训练脚本
==============================
联合训练: 交叉熵分类 + 物理引导重建 (MSE on y_clean)。

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

from detector.detector import Detector
from attack import ALL_ATTACK_TYPES, ATTACK_NAMES
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'dataset_win')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

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
RECON_LAMBDA = 0.3
RECON_WARMUP = 20
LR_PATIENCE = 30
LR_FACTOR = 0.8
LR_MIN = 1e-6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 数据集
# ============================================================================

class PreprocessedDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, split='train',
                 downsample_a0=0.0, load_clean=True):
        self.split = split
        self.load_clean = load_clean
        self.X = np.load(os.path.join(data_dir, f'X_{split}.npy'), mmap_mode='r')
        self.cls_labels = np.load(os.path.join(data_dir, f'Y_{split}_cls.npy'), mmap_mode='r')
        if load_clean:
            clean_path = os.path.join(data_dir, f'Y_{split}_clean.npy')
            self.y_clean = np.load(clean_path, mmap_mode='r') if os.path.exists(clean_path) else None
        else:
            self.y_clean = None

        self._all_indices = np.arange(len(self.cls_labels))
        self._downsample_rate = downsample_a0
        self._active_indices = self._all_indices.copy()
        if downsample_a0 > 0 and split == 'train':
            a0_before = int(np.sum(self.cls_labels == 0))
            self.resample_a0()
            a0_after = int(np.sum(self.cls_labels[self._active_indices] == 0))
            print(f"[Dataset] A0 降采样 {downsample_a0:.0%}: {a0_before}→{a0_after} 窗口")

        yclean_info = f", Y_clean={self.y_clean.shape}" if self.y_clean is not None else ""
        print(f"[Dataset] {split}: {len(self):,} 窗口, X={self.X.shape}{yclean_info}")

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
        if self.y_clean is not None:
            return x, cls_label, torch.from_numpy(self.y_clean[real_idx])
        return x, cls_label


# ============================================================================
# 训练循环
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    use_recon = epoch > RECON_WARMUP
    total_loss = total_cls = total_recon = 0.0
    correct = total = 0

    for batch in dataloader:
        x, cls_label = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        y_clean = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None

        optimizer.zero_grad(set_to_none=True)
        if use_recon and y_clean is not None:
            cls_logits, _, y_pred, _ = model(x, return_recon=True)
        else:
            cls_logits, _ = model(x, return_recon=False)

        loss_cls = F.cross_entropy(cls_logits, cls_label, label_smoothing=LABEL_SMOOTHING)
        if use_recon and y_clean is not None:
            loss = loss_cls + RECON_LAMBDA * F.mse_loss(y_pred, y_clean)
            total_recon += (loss.item() - loss_cls.item()) * x.size(0) / RECON_LAMBDA
        else:
            loss = loss_cls

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_cls += loss_cls.item() * bs
        correct += (cls_logits.argmax(1) == cls_label).sum().item()
        total += bs

    n = max(total, 1)
    m = {'loss': total_loss / n, 'cls_loss': total_cls / n, 'acc': correct / n}
    if use_recon:
        m['recon_loss'] = total_recon / n
    return m


@torch.no_grad()
def evaluate(model, dataloader, device, epoch=999):
    model.eval()
    use_recon = epoch > RECON_WARMUP
    total_loss = total_cls = total_recon = 0.0
    correct = total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for batch in dataloader:
        x, cls_label = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        y_clean = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None

        if use_recon and y_clean is not None:
            cls_logits, _, y_pred, _ = model(x, return_recon=True)
        else:
            cls_logits, _ = model(x, return_recon=False)

        loss_cls = F.cross_entropy(cls_logits, cls_label)
        loss = loss_cls
        if use_recon and y_clean is not None:
            loss_recon = F.mse_loss(y_pred, y_clean)
            loss = loss_cls + RECON_LAMBDA * loss_recon
            total_recon += loss_recon.item() * x.size(0)

        B = x.size(0)
        total_loss += loss.item() * B
        total_cls += loss_cls.item() * B
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
    result = {'loss': total_loss / n, 'cls_loss': total_cls / n,
              'acc': correct / n, 'per_class_acc': per_class_acc}
    if use_recon:
        result['recon_loss'] = total_recon / n
    return result


# ============================================================================
# 主入口
# ============================================================================

def main():
    global RECON_LAMBDA, RECON_WARMUP, LABEL_SMOOTHING
    parser = argparse.ArgumentParser(description='分类检测器训练')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='预处理后的数据目录')
    parser.add_argument('--eval-only', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--label-smoothing', type=float, default=LABEL_SMOOTHING)
    parser.add_argument('--downsample-a0', type=float, default=0.5)
    parser.add_argument('--recon-lambda', type=float, default=RECON_LAMBDA)
    parser.add_argument('--recon-warmup', type=int, default=RECON_WARMUP)
    parser.add_argument('--no-decoder', action='store_true')
    args = parser.parse_args()

    RECON_LAMBDA = args.recon_lambda
    RECON_WARMUP = args.recon_warmup
    LABEL_SMOOTHING = args.label_smoothing

    print(f"设备: {DEVICE}, 骨干: MultiScaleDSConvBackbone")
    print(f"解码器: {'禁用' if args.no_decoder else '启用'} (λ={RECON_LAMBDA}, warmup={RECON_WARMUP})")

    # ---- 归一化参数 ----
    norm_data = np.load(os.path.join(args.data_dir, 'normalizer.npz'))
    norm_params = {k: norm_data[k] for k in
                   ['ymeas_scale', 'ymeas_median', 'cmd_max', 'feat_scale', 'feat_median']}

    # ---- 数据集 ----
    load_clean = not args.no_decoder
    train_dataset = PreprocessedDataset(args.data_dir, split='train', downsample_a0=args.downsample_a0, load_clean=load_clean)
    val_dataset = PreprocessedDataset(args.data_dir, split='val', load_clean=load_clean)
    test_dataset = PreprocessedDataset(args.data_dir, split='test', load_clean=load_clean)

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
        cmd_max=norm_params['cmd_max'].tolist(),
        feat_scale=norm_params['feat_scale'].tolist(),
        feat_median=norm_params['feat_median'].tolist(),
        use_decoder=not args.no_decoder)
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 参数量: {n_params:,}")

    # ---- 仅评估 ----
    if args.eval_only:
        print(f"\n加载权重: {args.eval_only}")
        model.load_state_dict(torch.load(args.eval_only, map_location=DEVICE, weights_only=True), strict=False)
        result = evaluate(model, test_loader, DEVICE, epoch=RECON_WARMUP + 1)
        print(f"\n测试集: Loss={result['loss']:.4f}, Acc={result['acc']:.4f}")
        if 'recon_loss' in result:
            print(f"  重建 Loss: {result['recon_loss']:.6f}")
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
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
               'train_recon': [], 'val_recon': []}

    for epoch in range(1, args.epochs + 1):
        train_dataset.resample_a0()
        train_m = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
        val_m = evaluate(model, val_loader, DEVICE, epoch)
        scheduler.step(val_m['loss'])

        history['train_loss'].append(train_m['loss'])
        history['train_acc'].append(train_m['acc'])
        history['val_loss'].append(val_m['loss'])
        history['val_acc'].append(val_m['acc'])
        history['train_recon'].append(train_m.get('recon_loss', 0.0))
        history['val_recon'].append(val_m.get('recon_loss', 0.0))

        lr_now = optimizer.param_groups[0]['lr']
        recon_str = f" | R T={train_m['recon_loss']:.4f} V={val_m['recon_loss']:.4f}" if 'recon_loss' in train_m else ""
        print(f"E {epoch:3d} | LR={lr_now:.1e} | "
              f"T Loss={train_m['loss']:.4f} Acc={train_m['acc']:.4f} | "
              f"V Loss={val_m['loss']:.4f} Acc={val_m['acc']:.4f}{recon_str}",
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
    test_m = evaluate(model, test_loader, DEVICE, epoch=RECON_WARMUP + 1)
    print(f"\n{'='*60}")
    print(f"训练完成！ 最佳 epoch: {best_epoch}")
    print(f"  测试 Loss: {test_m['loss']:.4f}, Acc: {test_m['acc']:.4f}")
    if 'recon_loss' in test_m:
        print(f"  重建 Loss: {test_m['recon_loss']:.6f}")
    for atk in ALL_ATTACK_TYPES:
        print(f"    {atk} ({ATTACK_NAMES[atk]}): {test_m['per_class_acc'].get(atk, 0):.4f}")
    print(f"{'='*60}")

    # ---- 训练曲线 ----
    n_plots = 3 if 'recon_loss' in test_m else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 2:
        ax1, ax2 = axes[0], axes[1]
    else:
        ax1, ax2, ax3 = axes

    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss'); ax1.legend(); ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.axhline(y=test_m['acc'], color='g', linestyle='--', label=f'Test={test_m["acc"]:.3f}')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Classification Accuracy'); ax2.legend(); ax2.grid(True)

    if n_plots >= 3:
        ax3.plot(history['train_recon'], label='Train Recon')
        ax3.plot(history['val_recon'], label='Val Recon')
        ax3.set_xlabel('Epoch'); ax3.set_ylabel('MSE')
        ax3.set_title('Reconstruction Loss'); ax3.legend(); ax3.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, 'training_curve.png'), dpi=150)
    plt.close(fig)
    print(f"训练曲线已保存至: {os.path.join(MODEL_DIR, 'training_curve.png')}")


if __name__ == "__main__":
    main()
