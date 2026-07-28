"""
train.py — 运动学约束状态估计网络训练脚本
==========================================
支持三种训练模式:
  - 联合训练 (默认): 状态估计 + 分类同步训练
  - 预训练: 纯状态估计 (自监督, L_recon + L_kin), 保存 nn_recon_pretrain.pt
  - 微调: 加载预训练权重 → 联合训练, 可选冻结编码器

用法:
  python train.py                                    # 联合训练
  python train.py --pretrain                         # 阶段1: 状态估计预训练
  python train.py --finetune models/nn_recon_pretrain.pt  # 阶段2: 分类微调
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
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 150
PRETRAIN_EPOCHS = 100
LABEL_SMOOTHING = 0.0
EARLY_STOP_PATIENCE = 50
GRAD_CLIP = 1.0
LR_PATIENCE = 30
LR_FACTOR = 0.8
LR_MIN = 1e-6
MASK_MIN = 0.10
MASK_MAX = 0.40
KIN_LAMBDA = 0.1             # 运动学一致性损失权重
CLS_LAMBDA = 0.5             # 分类损失权重
FREEZE_ENCODER_EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 数据集
# ============================================================================

class PreprocessedDataset(Dataset):
    """加载预处理数据, 返回 (x, cls_label, y_clean_norm)。

    y_clean 在加载时归一化, 匹配模型的归一化空间输出 q。
    """

    def __init__(self, data_dir=DATA_DIR, split='train', downsample_a0=0.0,
                 norm_params=None):
        self.split = split
        self.X = np.load(os.path.join(data_dir, f'X_{split}.npy'), mmap_mode='r')
        self.cls_labels = np.load(os.path.join(data_dir, f'Y_{split}_cls.npy'), mmap_mode='r')
        self.Y_clean = np.load(os.path.join(data_dir, f'Y_{split}_clean.npy'), mmap_mode='r')

        # y_clean 归一化 (与 X 的 y_meas 使用相同参数)
        if norm_params is None:
            norm_path = os.path.join(data_dir, 'normalizer.npz')
            norm_params = dict(np.load(norm_path))
        self._ymeas_median = norm_params['ymeas_median'].astype(np.float32)
        self._ymeas_scale = norm_params['ymeas_scale'].astype(np.float32)

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
        # 归一化 y_clean: (y_phys - median) / scale, 与模型的 q 在同一空间
        y_clean_raw = self.Y_clean[real_idx]
        y_clean_norm = (y_clean_raw - self._ymeas_median) / np.maximum(self._ymeas_scale, 1e-6)
        return x, cls_label, torch.from_numpy(y_clean_norm)


# ============================================================================
# 预训练: 纯状态估计 (L_recon + L_kin)
# ============================================================================

def pretrain_epoch(model, dataloader, optimizer, device, rng):
    model.train()
    total_loss = total_recon = total_kin = 0.0
    total_samples = 0

    for batch in dataloader:
        x = batch[0].to(device, non_blocking=True)
        y_clean = batch[2].to(device, non_blocking=True)
        mask_ratio = float(rng.uniform(MASK_MIN, MASK_MAX))

        optimizer.zero_grad(set_to_none=True)
        _, q, mask, _ = model(x, mask_ratio=mask_ratio, return_all=True)

        # L_recon: MSE(q, y_clean) 所有步
        loss_recon = F.mse_loss(q, y_clean)

        # L_kin: 运动学一致性 (q[t] ≈ Euler(q[t-1], u[t-1]))
        q_ct = q.permute(0, 2, 1)
        u_ct = x[:, :, 3:5].to(device, non_blocking=True).permute(0, 2, 1)
        loss_kin = model.compute_kin_loss(q_ct, u_ct) if KIN_LAMBDA > 0 else torch.tensor(0.0)

        loss = loss_recon + KIN_LAMBDA * loss_kin
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += loss_recon.item() * bs
        total_kin += loss_kin.item() * bs if isinstance(loss_kin, torch.Tensor) else 0.0
        total_samples += bs

    n = max(total_samples, 1)
    return {'loss': total_loss / n, 'recon_loss': total_recon / n, 'kin_loss': total_kin / n}


@torch.no_grad()
def evaluate_recon(model, dataloader, device, rng):
    """验证状态估计质量"""
    model.eval()
    total_loss = total_recon = total_kin = 0.0
    total_samples = 0

    for batch in dataloader:
        x = batch[0].to(device, non_blocking=True)
        y_clean = batch[2].to(device, non_blocking=True)
        mask_ratio = float(rng.uniform(MASK_MIN, MASK_MAX))

        _, q, _, _ = model(x, mask_ratio=mask_ratio, return_all=True)

        loss_recon = F.mse_loss(q, y_clean)
        q_ct = q.permute(0, 2, 1)
        u_ct = x[:, :, 3:5].to(device, non_blocking=True).permute(0, 2, 1)
        loss_kin = model.compute_kin_loss(q_ct, u_ct) if KIN_LAMBDA > 0 else torch.tensor(0.0)
        loss = loss_recon + KIN_LAMBDA * loss_kin

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += loss_recon.item() * bs
        total_kin += loss_kin.item() * bs if isinstance(loss_kin, torch.Tensor) else 0.0
        total_samples += bs

    n = max(total_samples, 1)
    return {'loss': total_loss / n, 'recon_loss': total_recon / n, 'kin_loss': total_kin / n}


# ============================================================================
# 联合训练 / 微调
# ============================================================================

def joint_train_epoch(model, dataloader, optimizer, device, rng, use_cls=True):
    model.train()
    total_loss = total_recon = total_kin = total_cls = 0.0
    correct = total = 0

    for batch in dataloader:
        x = batch[0].to(device, non_blocking=True)
        cls_label = batch[1].to(device, non_blocking=True)
        y_clean = batch[2].to(device, non_blocking=True)
        mask_ratio = float(rng.uniform(MASK_MIN, MASK_MAX))

        optimizer.zero_grad(set_to_none=True)
        cls_logits, q, mask, _ = model(x, mask_ratio=mask_ratio, return_all=True)

        # L_recon
        loss_recon = F.mse_loss(q, y_clean)

        # L_kin
        q_ct = q.permute(0, 2, 1)
        u_ct = x[:, :, 3:5].to(device, non_blocking=True).permute(0, 2, 1)
        loss_kin = model.compute_kin_loss(q_ct, u_ct) if KIN_LAMBDA > 0 else torch.tensor(0.0)

        if use_cls:
            loss_cls = F.cross_entropy(cls_logits, cls_label, label_smoothing=LABEL_SMOOTHING)
            loss = loss_recon + KIN_LAMBDA * loss_kin + CLS_LAMBDA * loss_cls
        else:
            loss_cls = torch.tensor(0.0)
            loss = loss_recon + KIN_LAMBDA * loss_kin

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += loss_recon.item() * bs
        total_kin += loss_kin.item() * bs if isinstance(loss_kin, torch.Tensor) else 0.0
        total_cls += loss_cls.item() * bs if isinstance(loss_cls, torch.Tensor) else 0.0
        if use_cls:
            correct += (cls_logits.argmax(1) == cls_label).sum().item()
        total += bs

    n = max(total, 1)
    result = {'loss': total_loss / n, 'recon_loss': total_recon / n, 'kin_loss': total_kin / n}
    if use_cls:
        result['cls_loss'] = total_cls / n
        result['acc'] = correct / n
    return result


@torch.no_grad()
def evaluate_cls(model, dataloader, device):
    """验证/测试: 不掩码, 仅分类"""
    model.eval()
    total_loss = 0.0
    correct = total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for batch in dataloader:
        x = batch[0].to(device, non_blocking=True)
        cls_label = batch[1].to(device, non_blocking=True)

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
# 工具
# ============================================================================

def _set_encoder_grad(model, requires_grad: bool):
    """冻结/解冻编码器 + state_head (不包括分类头)"""
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = requires_grad


def _print_cls_result(test_m):
    print(f"  测试 Loss: {test_m['loss']:.4f}, Acc: {test_m['acc']:.4f}")
    for atk in ALL_ATTACK_TYPES:
        print(f"    {atk} ({ATTACK_NAMES[atk]}): {test_m['per_class_acc'].get(atk, 0):.4f}")


# ============================================================================
# 主入口
# ============================================================================

def main():
    global LABEL_SMOOTHING, KIN_LAMBDA, CLS_LAMBDA
    parser = argparse.ArgumentParser(description='运动学约束状态估计 + 分类训练')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='预处理后的数据目录')
    parser.add_argument('--pretrain', action='store_true',
                        help='阶段1: 纯状态估计预训练')
    parser.add_argument('--finetune', type=str, default=None, metavar='PATH',
                        help='阶段2: 加载预训练权重进行联合训练')
    parser.add_argument('--eval-only', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--label-smoothing', type=float, default=LABEL_SMOOTHING)
    parser.add_argument('--downsample-a0', type=float, default=0.5)
    parser.add_argument('--mask-min', type=float, default=MASK_MIN)
    parser.add_argument('--mask-max', type=float, default=MASK_MAX)
    parser.add_argument('--kin-lambda', type=float, default=KIN_LAMBDA,
                        help='运动学一致性损失权重')
    parser.add_argument('--cls-lambda', type=float, default=CLS_LAMBDA,
                        help='分类损失权重')
    parser.add_argument('--freeze-encoder', type=int, default=FREEZE_ENCODER_EPOCHS,
                        help='微调时前 N epoch 冻结编码器 (默认10, 0=不冻结)')
    parser.add_argument('--no-cls', action='store_true',
                        help='微调时关闭分类损失 (仅状态估计)')
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = _latest_data_dir()
        print(f"自动选择最新数据: {args.data_dir}")

    KIN_LAMBDA = args.kin_lambda
    CLS_LAMBDA = args.cls_lambda
    LABEL_SMOOTHING = args.label_smoothing

    # ---- 归一化参数 ----
    norm_data = np.load(os.path.join(args.data_dir, 'normalizer.npz'))
    norm_params = {k: norm_data[k] for k in ['ymeas_scale', 'ymeas_median', 'cmd_max']}

    # ---- 数据集 ----
    train_dataset = PreprocessedDataset(
        args.data_dir, split='train', downsample_a0=args.downsample_a0, norm_params=norm_params)
    val_dataset = PreprocessedDataset(
        args.data_dir, split='val', norm_params=norm_params)
    test_dataset = PreprocessedDataset(
        args.data_dir, split='test', norm_params=norm_params)

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
        mask_min=args.mask_min, mask_max=args.mask_max)
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 参数量: {n_params:,}")

    # ---- 仅评估 ----
    if args.eval_only:
        print(f"\n加载权重: {args.eval_only}")
        model.load_state_dict(torch.load(args.eval_only, map_location=DEVICE, weights_only=True), strict=False)
        _print_cls_result(evaluate_cls(model, test_loader, DEVICE))
        return

    rng = np.random.RandomState()

    # ========================================================================
    # 阶段1: 预训练 (纯状态估计)
    # ========================================================================
    if args.pretrain:
        epochs = args.epochs or PRETRAIN_EPOCHS
        lr = args.lr or LEARNING_RATE
        print(f"\n{'='*60}")
        print(f"阶段1: 状态估计预训练 (L_recon + L_kin, 全部数据)")
        print(f"{'='*60}")
        print(f"设备: {DEVICE}, 骨干: TemporalEncoder (4×DilatedResidual), Epochs: {epochs}, LR: {lr:.1e}")
        print(f"掩码: [{args.mask_min:.0%}, {args.mask_max:.0%}], L_kin λ={KIN_LAMBDA}")

        pretrain_dataset = PreprocessedDataset(
            args.data_dir, split='train', downsample_a0=0.0, norm_params=norm_params)
        pretrain_loader = _make_loader(pretrain_dataset, args.batch_size, shuffle=True)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)

        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': []}

        for epoch in range(1, epochs + 1):
            train_m = pretrain_epoch(model, pretrain_loader, optimizer, DEVICE, rng)
            val_m = evaluate_recon(model, val_loader, DEVICE, rng)
            scheduler.step(val_m['loss'])

            history['train_loss'].append(train_m['loss'])
            history['val_loss'].append(val_m['loss'])
            history['train_recon'].append(train_m['recon_loss'])
            history['val_recon'].append(val_m['recon_loss'])

            lr_now = optimizer.param_groups[0]['lr']
            print(f"E {epoch:3d} | LR={lr_now:.1e} | "
                  f"T Loss={train_m['loss']:.4f} R={train_m['recon_loss']:.4f} K={train_m['kin_loss']:.4f} | "
                  f"V Loss={val_m['loss']:.4f} R={val_m['recon_loss']:.4f}",
                  end='')

            if val_m['loss'] < best_loss:
                best_loss = val_m['loss']
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'nn_recon_pretrain.pt'))
                print("  *", end='')
            else:
                patience_counter += 1
            print()

            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"早停: 最佳 epoch {best_epoch}, Val Loss={best_loss:.4f}")
                break

        print(f"\n预训练完成！最佳 epoch: {best_epoch}, Val Loss={best_loss:.4f}")
        print(f"模型已保存: {os.path.join(MODEL_DIR, 'nn_recon_pretrain.pt')}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(history['train_loss'], label='Train')
        ax1.plot(history['val_loss'], label='Val')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title('Total Loss (Recon + Kin)'); ax1.legend(); ax1.grid(True)
        ax2.plot(history['train_recon'], label='Train')
        ax2.plot(history['val_recon'], label='Val')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('MSE')
        ax2.set_title('Reconstruction MSE'); ax2.legend(); ax2.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(MODEL_DIR, 'pretrain_curve.png'), dpi=150)
        plt.close(fig)
        return

    # ========================================================================
    # 阶段2: 微调
    # ========================================================================
    if args.finetune:
        epochs = args.epochs or NUM_EPOCHS
        lr = args.lr or FINETUNE_LR
        freeze_epochs = args.freeze_encoder
        use_cls = not args.no_cls

        print(f"\n{'='*60}")
        print(f"阶段2: 联合微调 (状态估计 + 分类)")
        print(f"{'='*60}")
        print(f"预训练权重: {args.finetune}")
        print(f"设备: {DEVICE}, Epochs: {epochs}, LR: {lr:.1e}")
        print(f"冻结编码器: {freeze_epochs} epoch, 分类损失: {'开启' if use_cls else '关闭'}")
        print(f"L_kin λ={KIN_LAMBDA}, L_cls λ={CLS_LAMBDA}")

        state = torch.load(args.finetune, map_location=DEVICE, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            cls_missing = [k for k in missing if 'classifier' in k]
            if cls_missing:
                print(f"  分类头随机初始化: {cls_missing}")

        if freeze_epochs > 0:
            _set_encoder_grad(model, False)
            print(f"  编码器已冻结 (前 {freeze_epochs} epoch)")

        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)

        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
                   'train_recon': [], 'train_cls': []}

        for epoch in range(1, epochs + 1):
            if freeze_epochs > 0 and epoch == freeze_epochs + 1:
                _set_encoder_grad(model, True)
                optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)
                print(f"  E {epoch}: 编码器已解冻, 优化器重建 (LR={lr:.1e})")

            train_dataset.resample_a0()
            train_m = joint_train_epoch(model, train_loader, optimizer, DEVICE, rng, use_cls=use_cls)
            val_m = evaluate_cls(model, val_loader, DEVICE)
            scheduler.step(val_m['loss'])

            history['train_loss'].append(train_m['loss'])
            history['train_acc'].append(train_m.get('acc', 0))
            history['val_loss'].append(val_m['loss'])
            history['val_acc'].append(val_m['acc'])
            history['train_recon'].append(train_m['recon_loss'])
            history['train_cls'].append(train_m.get('cls_loss', 0))

            lr_now = optimizer.param_groups[0]['lr']
            print(f"E {epoch:3d} | LR={lr_now:.1e} | "
                  f"T Loss={train_m['loss']:.4f} Acc={train_m.get('acc', 0):.4f} "
                  f"(R={train_m['recon_loss']:.4f} C={train_m.get('cls_loss', 0):.4f}) | "
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
                print(f"早停: 最佳 epoch {best_epoch}")
                break

        print(f"\n加载最佳模型 (epoch {best_epoch}, Val Acc={best_val_acc:.4f})")
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'nn_cls_best.pt'),
                                         map_location=DEVICE, weights_only=True))
        test_m = evaluate_cls(model, test_loader, DEVICE)
        print(f"\n微调完成！最佳 epoch: {best_epoch}")
        _print_cls_result(test_m)
        return

    # ========================================================================
    # 默认: 联合训练 (从头开始)
    # ========================================================================
    epochs = args.epochs or NUM_EPOCHS
    lr = args.lr or LEARNING_RATE
    print(f"\n{'='*60}")
    print(f"联合训练 (状态估计 + 分类)")
    print(f"{'='*60}")
    print(f"设备: {DEVICE}, 骨干: TemporalEncoder (4×DilatedResidual), Epochs: {epochs}, LR: {lr:.1e}")
    print(f"掩码: [{args.mask_min:.0%}, {args.mask_max:.0%}]")
    print(f"L_kin λ={KIN_LAMBDA}, L_cls λ={CLS_LAMBDA}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
               'train_recon': [], 'train_cls': [], 'train_kin': []}

    for epoch in range(1, epochs + 1):
        train_dataset.resample_a0()
        train_m = joint_train_epoch(model, train_loader, optimizer, DEVICE, rng, use_cls=True)
        val_m = evaluate_cls(model, val_loader, DEVICE)
        scheduler.step(val_m['loss'])

        history['train_loss'].append(train_m['loss'])
        history['train_acc'].append(train_m['acc'])
        history['val_loss'].append(val_m['loss'])
        history['val_acc'].append(val_m['acc'])
        history['train_recon'].append(train_m['recon_loss'])
        history['train_cls'].append(train_m['cls_loss'])
        history['train_kin'].append(train_m['kin_loss'])

        lr_now = optimizer.param_groups[0]['lr']
        print(f"E {epoch:3d} | LR={lr_now:.1e} | "
              f"T Loss={train_m['loss']:.4f} Acc={train_m['acc']:.4f} "
              f"(R={train_m['recon_loss']:.4f} K={train_m['kin_loss']:.4f} C={train_m['cls_loss']:.4f}) | "
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
            print(f"早停: 最佳 epoch {best_epoch}")
            break

    print(f"\n加载最佳模型 (epoch {best_epoch}, Val Acc={best_val_acc:.4f})")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'nn_cls_best.pt'),
                                     map_location=DEVICE, weights_only=True))
    test_m = evaluate_cls(model, test_loader, DEVICE)
    print(f"\n联合训练完成！最佳 epoch: {best_epoch}")
    _print_cls_result(test_m)

    # ---- 训练曲线 ----
    n_plots = 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].axhline(y=test_m['acc'], color='g', linestyle='--', label=f'Test={test_m["acc"]:.3f}')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Classification Accuracy'); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(history['train_recon'], label='Recon MSE')
    axes[2].plot(history['train_cls'], label='Cls CE')
    axes[2].plot(history['train_kin'], label='Kin MSE', alpha=0.6)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Loss')
    axes[2].set_title('Recon / Kin / Cls Loss'); axes[2].legend(); axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, 'training_curve.png'), dpi=150)
    plt.close(fig)
    print(f"训练曲线已保存至: {os.path.join(MODEL_DIR, 'training_curve.png')}")


if __name__ == "__main__":
    main()
