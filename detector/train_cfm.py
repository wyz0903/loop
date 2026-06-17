"""
train_cfm.py — 攻击分类检测器训练脚本 (编码器-解码器)
=========================================================
联合训练: 交叉熵分类损失 + 物理引导重建损失 (MSE on y_clean)。

用法:
  python train_cfm.py                          # 默认训练
  python train_cfm.py --backbone transformer   # Transformer 骨干
  python train_cfm.py --eval-only detector/models/cfm_cls_best.pt
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
                                    CONV_CHANNELS, CONV_KERNEL_SIZES, CONV_DILATIONS,
                                    POOL_SIZE)

# ============================================================================
# 攻击类型常量 (来自 attack.py 唯一数据源)
# ============================================================================

from attack import ALL_ATTACK_TYPES, ATTACK_NAMES_CN

# ============================================================================
# 全局路径配置
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'dataset_win')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# 数据集类 (返回 X + cls_label + y_clean)
# ============================================================================

class PreprocessedDataset(Dataset):
    """加载预处理的 .npy 窗口数据 (分类 + 重建任务)。"""

    def __init__(self, data_dir: str = DATA_DIR, split: str = 'train',
                 downsample_a0: float = 0.0, load_clean: bool = True):
        self.split = split
        self.load_clean = load_clean
        self.X = np.load(os.path.join(data_dir, f'X_{split}.npy'), mmap_mode='r')
        self.cls_labels = np.load(os.path.join(data_dir, f'Y_{split}_cls.npy'), mmap_mode='r')
        if load_clean:
            clean_path = os.path.join(data_dir, f'Y_{split}_clean.npy')
            if os.path.exists(clean_path):
                self.y_clean = np.load(clean_path, mmap_mode='r')
            else:
                self.y_clean = None
        else:
            self.y_clean = None

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

        yclean_info = f", Y_clean={self.y_clean.shape}" if self.y_clean is not None else ""
        print(f"[Dataset] {split}: {len(self):,} 窗口, X={self.X.shape}{yclean_info}")

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

    def __getitem__(self, idx: int):
        real_idx = self._active_indices[idx]
        x = torch.from_numpy(self.X[real_idx])
        cls_label = torch.tensor(self.cls_labels[real_idx], dtype=torch.long)
        if self.y_clean is not None:
            y_clean = torch.from_numpy(self.y_clean[real_idx])
            return x, cls_label, y_clean
        return x, cls_label


# ============================================================================
# 训练超参数
# ============================================================================
BATCH_SIZE = 256
NUM_WORKERS = 2
PREFETCH_FACTOR = 4
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 150
LABEL_SMOOTHING = 0.0            # 标签平滑 (0=关闭)
EARLY_STOP_PATIENCE = 50
GRAD_CLIP = 1.0

# 重建损失
RECON_LAMBDA = 0.3               # 重建损失权重
RECON_WARMUP = 20                # 前 N epoch 仅分类, 之后加入重建损失

# ReduceLROnPlateau 调度器
LR_PATIENCE = 30
LR_FACTOR = 0.8
LR_MIN = 1e-6

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 1. 训练循环
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch: int):
    """单 epoch 训练 (分类 + 重建联合损失)。

    前 RECON_WARMUP epoch 仅使用分类损失, 之后加入重建损失。
    """
    model.train()
    use_recon = epoch > RECON_WARMUP

    total_loss = total_cls = total_recon = 0.0
    correct = total = 0

    for batch in dataloader:
        if len(batch) == 3:
            x, cls_label, y_clean = batch
            y_clean = y_clean.to(device, non_blocking=True)
        else:
            x, cls_label = batch
            y_clean = None

        x = x.to(device, non_blocking=True)
        cls_label = cls_label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_recon and y_clean is not None:
            cls_logits, _, y_pred, _ = model(x, return_recon=True)
        else:
            cls_logits, _ = model(x, return_recon=False)

        # 分类损失
        loss_cls = F.cross_entropy(cls_logits, cls_label,
                                   label_smoothing=LABEL_SMOOTHING)

        # 重建损失
        if use_recon and y_clean is not None:
            loss_recon = F.mse_loss(y_pred, y_clean)
            loss = loss_cls + RECON_LAMBDA * loss_recon
            total_recon += loss_recon.item() * x.size(0)
        else:
            loss = loss_cls

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_cls += loss_cls.item() * bs
        correct += (cls_logits.argmax(dim=1) == cls_label).sum().item()
        total += bs

    n = max(total, 1)
    metrics = {
        'loss': total_loss / n,
        'cls_loss': total_cls / n,
        'acc': correct / n,
    }
    if use_recon:
        metrics['recon_loss'] = total_recon / n
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, epoch: int = 999):
    """评估: 分类准确率 + 重建损失 (标准交叉熵, 无标签平滑)"""
    model.eval()
    use_recon = epoch > RECON_WARMUP

    total_loss = total_cls = total_recon = 0.0
    correct = total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for batch in dataloader:
        if len(batch) == 3:
            x, cls_label, y_clean = batch
            y_clean = y_clean.to(device, non_blocking=True)
        else:
            x, cls_label = batch
            y_clean = None

        x = x.to(device, non_blocking=True)
        cls_label = cls_label.to(device, non_blocking=True)

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
    for cls_idx in range(8):
        if class_total[cls_idx] > 0:
            per_class_acc[ALL_ATTACK_TYPES[cls_idx]] = (
                class_correct[cls_idx] / class_total[cls_idx])

    result = {
        'loss': total_loss / n,
        'cls_loss': total_cls / n,
        'acc': correct / n,
        'per_class_acc': per_class_acc,
    }
    if use_recon:
        result['recon_loss'] = total_recon / n
    return result


# ============================================================================
# 2. 主训练入口
# ============================================================================

def _save_model_config(model: CFMDetector):
    """保存模型配置到 cfm_cls_config.npz (供 cfm_backend 加载使用)。"""
    cfg = {
        'in_channels': int(model.in_channels),
        'window_size': int(model.window_size),
        'd_model': int(model.d_model),
        'num_classes': int(model.num_classes),
        'backbone_type': str(model.backbone_type),
        'use_decoder': bool(model.use_decoder),
        'ymeas_scale': model.ymeas_scale.cpu().numpy(),
        'ymeas_median': model.ymeas_median.cpu().numpy(),
        'cmd_max': model.cmd_max.cpu().numpy(),
    }
    if model.backbone_type == 'simple_conv':
        backbone = model.backbone
        cfg['use_channel_attn'] = False
        # KAD 骨干特定配置
        if hasattr(backbone, 'conv_channels'):
            cfg['conv_channels'] = backbone.conv_channels
        if hasattr(backbone, 'conv_kernel_sizes'):
            cfg['conv_kernel_sizes'] = backbone.conv_kernel_sizes
        if hasattr(backbone, 'conv_dilations'):
            cfg['conv_dilations'] = backbone.conv_dilations
        if hasattr(backbone, 'pool_size'):
            cfg['pool_size'] = int(backbone.pool_size)
    np.savez(os.path.join(MODEL_DIR, 'cfm_cls_config.npz'), **cfg)


def _load_norm_params(data_dir: str):
    """从 normalizer.npz 加载归一化参数。"""
    norm_path = os.path.join(data_dir, 'normalizer.npz')
    if os.path.exists(norm_path):
        data = np.load(norm_path)
        return {
            'ymeas_scale': data.get('ymeas_scale', np.array([2.5, 2.5, np.pi])),
            'ymeas_median': data.get('ymeas_median', np.zeros(3)),
            'cmd_max': data.get('cmd_max', np.array([0.3, 1.76])),
        }
    return {}


def build_model(backbone_type='simple_conv', norm_params: dict = None):
    """构建 CFMDetector (编码器-解码器)。"""
    kwargs = {'backbone_type': backbone_type,
              'conv_channels': CONV_CHANNELS,
              'conv_kernel_size': CONV_KERNEL_SIZES[0],
              'pool_size': POOL_SIZE}
    if norm_params:
        kwargs['ymeas_scale'] = norm_params['ymeas_scale'].tolist()
        kwargs['ymeas_median'] = norm_params['ymeas_median'].tolist()
        kwargs['cmd_max'] = norm_params['cmd_max'].tolist()

    model = CFMDetector(**kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 参数量: {n_params:,}")
    if model.use_decoder:
        dec_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
        print(f"  编码器+分类头: {n_params - dec_params:,}")
        print(f"  解码器:         {dec_params:,}")
    return model


def main():
    global RECON_LAMBDA, RECON_WARMUP, LABEL_SMOOTHING
    parser = argparse.ArgumentParser(description='CFM 分类检测器训练 (编码器-解码器)')
    parser.add_argument('--backbone', type=str, default='simple_conv',
                        choices=['simple_conv', 'transformer'])
    parser.add_argument('--eval-only', type=str, default=None,
                        help='仅评估指定模型权重')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--label-smoothing', type=float, default=LABEL_SMOOTHING,
                        help='标签平滑系数 (0=关闭)')
    parser.add_argument('--downsample-a0', type=float, default=0.5,
                        help='A0 降采样比例')
    parser.add_argument('--recon-lambda', type=float, default=RECON_LAMBDA,
                        help='重建损失权重')
    parser.add_argument('--recon-warmup', type=int, default=RECON_WARMUP,
                        help='重建损失预热 epoch 数')
    parser.add_argument('--no-decoder', action='store_true',
                        help='禁用解码器 (消融实验)')
    args = parser.parse_args()

    # 更新全局参数 (CLI 覆盖默认值)
    RECON_LAMBDA = args.recon_lambda
    RECON_WARMUP = args.recon_warmup
    LABEL_SMOOTHING = args.label_smoothing

    print(f"设备: {DEVICE}")
    print(f"骨干: {args.backbone}")
    print(f"解码器: {'禁用' if args.no_decoder else '启用'} "
          f"(λ={RECON_LAMBDA}, warmup={RECON_WARMUP})")
    print(f"正则化: wd={WEIGHT_DECAY}, label_smooth={LABEL_SMOOTHING}")
    print(f"精度: float32, 固定 LR={args.lr}")

    # ---- 加载归一化参数 ----
    norm_params = _load_norm_params(DATA_DIR)

    # ---- 数据集 ----
    load_clean = not args.no_decoder
    train_dataset = PreprocessedDataset(split='train',
                                         downsample_a0=args.downsample_a0,
                                         load_clean=load_clean)
    val_dataset = PreprocessedDataset(split='val', load_clean=load_clean)
    test_dataset = PreprocessedDataset(split='test', load_clean=load_clean)

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
    if args.no_decoder:
        norm_params_for_model = None
    else:
        norm_params_for_model = norm_params
    model = build_model(backbone_type=args.backbone, norm_params=norm_params_for_model)
    model.to(DEVICE)

    # ---- 仅评估 ----
    if args.eval_only:
        print(f"\n加载权重: {args.eval_only}")
        state_dict = torch.load(args.eval_only, map_location=DEVICE, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  缺少键: {len(missing)} (解码器相关: {sum(1 for k in missing if 'decoder' in k)})")
        if unexpected:
            print(f"  多余键: {len(unexpected)}")

        result = evaluate(model, test_loader, DEVICE, epoch=RECON_WARMUP + 1)
        print(f"\n测试集: Loss={result['loss']:.4f}, Acc={result['acc']:.4f}")
        if 'recon_loss' in result:
            print(f"  重建 Loss: {result['recon_loss']:.6f}")
        print("各类别准确率:")
        for atk in ALL_ATTACK_TYPES:
            acc = result['per_class_acc'].get(atk, 0)
            print(f"  {atk} ({ATTACK_NAMES_CN[atk]}): {acc:.4f}")
        return

    # ---- 优化器 ----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)
    print(f"  优化器: AdamW (lr={args.lr}, wd={WEIGHT_DECAY})")
    print(f"  调度器: ReduceLROnPlateau (factor={LR_FACTOR}, patience={LR_PATIENCE})")

    # ---- 训练循环 ----
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
        if 'recon_loss' in train_m:
            history['train_recon'].append(train_m['recon_loss'])
        if 'recon_loss' in val_m:
            history['val_recon'].append(val_m['recon_loss'])

        # 打印
        lr_now = optimizer.param_groups[0]['lr']
        recon_str = ""
        if 'recon_loss' in train_m:
            recon_str = (f" | R T={train_m['recon_loss']:.4f} "
                         f"V={val_m['recon_loss']:.4f}")
        print(f"E {epoch:3d} | LR={lr_now:.1e} | "
              f"T Loss={train_m['loss']:.4f} Acc={train_m['acc']:.4f} | "
              f"V Loss={val_m['loss']:.4f} Acc={val_m['acc']:.4f}"
              f"{recon_str}",
              end='')

        # 保存最佳
        if val_m['acc'] > best_val_acc:
            best_val_acc = val_m['acc']
            best_epoch = epoch
            patience_counter = 0
            model_path = os.path.join(MODEL_DIR, 'cfm_cls_best.pt')
            torch.save(model.state_dict(), model_path)
            _save_model_config(model)
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

    test_m = evaluate(model, test_loader, DEVICE, epoch=RECON_WARMUP + 1)
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"  最佳 epoch: {best_epoch}")
    print(f"  测试 Loss:  {test_m['loss']:.4f}")
    print(f"  测试 Acc:   {test_m['acc']:.4f}")
    if 'recon_loss' in test_m:
        print(f"  重建 Loss:  {test_m['recon_loss']:.6f}")
    print(f"各类别准确率:")
    for atk in ALL_ATTACK_TYPES:
        acc = test_m['per_class_acc'].get(atk, 0)
        print(f"    {atk} ({ATTACK_NAMES_CN[atk]}): {acc:.4f}")
    print(f"{'='*60}")

    # ---- 绘制训练曲线 ----
    n_plots = 3 if history['train_recon'] else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))

    if n_plots == 2:
        ax1, ax2 = axes[0], axes[1]
    else:
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]

    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss (Cls + Recon)'); ax1.legend(); ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.axhline(y=test_m['acc'], color='g', linestyle='--',
                label=f'Test={test_m["acc"]:.3f}')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Classification Accuracy'); ax2.legend(); ax2.grid(True)

    if n_plots >= 3:
        ax3.plot(history['train_recon'], label='Train Recon')
        ax3.plot(history['val_recon'], label='Val Recon')
        if 'recon_loss' in test_m:
            ax3.axhline(y=test_m['recon_loss'], color='g', linestyle='--',
                        label=f'Test={test_m["recon_loss"]:.4f}')
        ax3.set_xlabel('Epoch'); ax3.set_ylabel('MSE')
        ax3.set_title('Reconstruction Loss (MSE)'); ax3.legend(); ax3.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, 'training_curve.png'), dpi=150)
    plt.close(fig)
    print(f"训练曲线已保存至: {os.path.join(MODEL_DIR, 'training_curve.png')}")


if __name__ == "__main__":
    main()
