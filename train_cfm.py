"""
train_cfm.py — PINN-Flow 攻击检测器训练脚本
==============================================
使用物理信息流匹配损失训练 CFMDetector。

核心创新:
  L = L_cls + λ_fm·L_fm + λ_phys·L_phys

  其中:
    L_cls  = CrossEntropy (分类)
    L_fm   = MSE(v_θ, v_target)  (OT 流匹配)
    L_phys = max(0, mean(||r_phys||²) − κ·Tr(R))  (运动学 ODE 约束)

用法:
  python train_cfm.py                          # 默认训练
  python train_cfm.py --eval-only models/cfm_cls_best.pt
  python train_cfm.py --lambda-phys 0          # 消融: 无物理正则化
"""

import os
import sys
import math
import argparse
import numpy as np
from collections import defaultdict
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', message='Detected call of.*lr_scheduler.step.*before.*optimizer.step')
warnings.filterwarnings('ignore', message='.*non-writable tensor.*')

from detector.cfm_detector import (CFMDetector, compute_physics_residual,
                                    D_MODEL, NUM_TRANSFORMER_LAYERS, NUM_HEADS,
                                    DIM_FEEDFORWARD, NUM_FLOW_BLOCKS,
                                    DIM_FEEDFORWARD_FLOW, DROPOUT, TRACE_R)

# 从 train_classifier 复用数据集类
from train_classifier import PreprocessedDataset, ALL_ATTACK_TYPES, ATTACK_NAMES_CN

# ============================================================================
# 全局配置
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'dataset_win', 'config')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 训练超参数
BATCH_SIZE = 2048
NUM_WORKERS = 2
PREFETCH_FACTOR = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 150
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 50
GRAD_CLIP = 1.0
USE_AMP = True

# 损失权重
LAMBDA_FM = 3.0          # 流匹配损失权重
LAMBDA_PHYS = 1.0        # 物理正则化权重
KAPPA = 1.0              # 噪声临界值缩放
A0_FM_WEIGHT = 0.15      # A0 窗口的 FM 损失降权

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 1. PINN-Flow 损失函数
# ============================================================================

def pinn_flow_loss(model: CFMDetector,
                   x_norm: torch.Tensor,
                   cls_label: torch.Tensor,
                   atk_target: torch.Tensor,
                   normalizer: dict,
                   lambda_fm: float = LAMBDA_FM,
                   lambda_phys: float = LAMBDA_PHYS,
                   kappa: float = KAPPA,
                   a0_fm_weight: float = A0_FM_WEIGHT,
                   trace_R: float = TRACE_R) -> Tuple[torch.Tensor, dict]:
    """PINN-Flow 联合损失。

    物理残差计算需要物理单位。我们将归一化的 innovation/u_cmd 去归一化后,
    使用 innovation(物理单位) 作为 y_meas 的代理。

    近似合理性: r_phys_exact − r_phys_approx = O(Ts²), 对正则化损失足够。

    Args:
        x_norm:     (B, W, 5) 归一化输入 [innov_norm(3) | u_cmd_norm(2)]
        cls_label:  (B,)      攻击类别标签
        atk_target: (B, W, 3) 真实攻击信号 (物理单位)
        normalizer: dict 含 'feat_median'(3,), 'feat_iqr'(3,), 'cmd_max'(2,)

    Returns:
        loss_total, metrics, cls_logits
    """
    B = x_norm.shape[0]
    W = x_norm.shape[1]

    # ---- 去归一化 (用于物理损失) ----
    feat_median = normalizer['feat_median']  # (3,) numpy
    feat_iqr = normalizer['feat_iqr']        # (3,) numpy
    cmd_max = normalizer['cmd_max']          # (2,) numpy

    # 转为 tensor
    if not isinstance(feat_median, torch.Tensor):
        feat_median = torch.from_numpy(feat_median).float().to(x_norm.device)
        feat_iqr = torch.from_numpy(feat_iqr).float().to(x_norm.device)
        cmd_max = torch.from_numpy(cmd_max).float().to(x_norm.device)

    # innovation_phys = innov_norm * IQR + median
    innov_phys = x_norm[:, :, :3] * feat_iqr.view(1, 1, 3) + feat_median.view(1, 1, 3)
    # u_cmd_phys = u_cmd_norm * cmd_max
    u_cmd_phys = x_norm[:, :, 3:5] * cmd_max.view(1, 1, 2)

    # ---- 1. 编码 ----
    features = model.encode(x_norm)  # (B, W, d_model)

    # ---- 2. 分类损失 ----
    cls_logits = model.classify(features)
    loss_cls = F.cross_entropy(cls_logits, cls_label, label_smoothing=LABEL_SMOOTHING)

    # ---- 3. 流匹配损失 (OT 路径, 在攻击信号空间) ----
    x_0 = torch.randn(B, W, 3, device=x_norm.device)
    t = torch.rand(B, device=x_norm.device)
    t_expanded = t.view(B, 1, 1)
    x_t = (1 - t_expanded) * x_0 + t_expanded * atk_target
    v_target = atk_target - x_0
    v_pred = model.flow_head(t, x_t, features)

    loss_fm_per_sample = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=[1, 2])

    is_a0 = (cls_label == 0)
    fm_weight = torch.where(is_a0,
                            torch.tensor(a0_fm_weight, device=x_norm.device),
                            torch.tensor(1.0, device=x_norm.device))
    loss_fm = (loss_fm_per_sample * fm_weight).mean()

    # ---- 4. PINN 物理正则化 ----
    # y_rec = innov_phys − x_t  (x_t 是流样本, 物理单位)
    y_rec_t = innov_phys - x_t                            # (B, W, 3)
    r_phys = compute_physics_residual(y_rec_t, u_cmd_phys) # (B, W−1, 3)
    phys_mse_per_sample = (r_phys ** 2).mean(dim=[1, 2])   # (B,)

    # 仅惩罚超出噪声下限的部分 (Huber 风格)
    loss_phys_per_sample = F.relu(phys_mse_per_sample - kappa * trace_R)
    loss_phys = loss_phys_per_sample.mean()

    # ---- 总损失 ----
    loss_total = loss_cls + lambda_fm * loss_fm + lambda_phys * loss_phys

    metrics = {
        'loss_cls': loss_cls.item(),
        'loss_fm': loss_fm.item(),
        'loss_phys': loss_phys.item(),
        'phys_mse': phys_mse_per_sample.mean().item(),
    }
    return loss_total, metrics, cls_logits


# ============================================================================
# 2. 训练循环
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device,
                normalizer, scaler=None,
                lambda_fm=LAMBDA_FM, lambda_phys=LAMBDA_PHYS):
    """单 epoch 训练。"""
    model.train()
    total_loss = total_cls = total_fm = total_phys = 0.0
    correct = total = 0

    for x, cls_label, atk_seq in dataloader:
        x = x.to(device, non_blocking=True)
        cls_label = cls_label.to(device, non_blocking=True)
        atk_seq = atk_seq.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss, metrics, cls_logits = pinn_flow_loss(
                    model, x, cls_label, atk_seq, normalizer,
                    lambda_fm=lambda_fm, lambda_phys=lambda_phys)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            loss, metrics, cls_logits = pinn_flow_loss(
                model, x, cls_label, atk_seq, normalizer,
                lambda_fm=lambda_fm, lambda_phys=lambda_phys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_cls += metrics['loss_cls'] * bs
        total_fm += metrics['loss_fm'] * bs
        total_phys += metrics['loss_phys'] * bs
        correct += (cls_logits.argmax(dim=1) == cls_label).sum().item()
        total += bs

    n = max(total, 1)
    return (total_loss / n, total_cls / n, total_fm / n,
            total_phys / n, correct / n)


@torch.no_grad()
def evaluate(model, dataloader, device, normalizer,
             lambda_fm=LAMBDA_FM, lambda_phys=LAMBDA_PHYS):
    """评估: 分类 + FM 损失 + ODE 重建 MAE。"""
    model.eval()
    total_loss = total_cls = total_fm = total_phys = 0.0
    correct = total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    recon_mae_by_class = defaultdict(list)

    for x, cls_label, atk_seq in dataloader:
        x = x.to(device, non_blocking=True)
        cls_label = cls_label.to(device, non_blocking=True)
        atk_seq = atk_seq.to(device, non_blocking=True)

        B, W, _ = x.shape

        features = model.encode(x)
        cls_logits = model.classify(features)
        loss_cls = F.cross_entropy(cls_logits, cls_label,
                                    label_smoothing=LABEL_SMOOTHING)

        # FM 损失
        x_0 = torch.randn(B, W, 3, device=device)
        t = torch.rand(B, device=device)
        t_expanded = t.view(B, 1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * atk_seq
        v_target = atk_seq - x_0
        v_pred = model.flow_head(t, x_t, features)
        loss_fm = F.mse_loss(v_pred, v_target)

        # 物理损失 (评估时也计算, 使用去归一化的 innovation)
        feat_median = torch.from_numpy(normalizer['feat_median']).float().to(device)
        feat_iqr = torch.from_numpy(normalizer['feat_iqr']).float().to(device)
        cmd_max = torch.from_numpy(normalizer['cmd_max']).float().to(device)
        innov_phys = x[:, :, :3] * feat_iqr.view(1, 1, 3) + feat_median.view(1, 1, 3)
        u_cmd_phys = x[:, :, 3:5] * cmd_max.view(1, 1, 2)
        y_rec_t = innov_phys - x_t
        r_phys = compute_physics_residual(y_rec_t, u_cmd_phys)
        phys_mse = (r_phys ** 2).mean()
        loss_phys = F.relu(phys_mse - KAPPA * TRACE_R)

        loss = loss_cls + lambda_fm * loss_fm + lambda_phys * loss_phys

        total_loss += loss.item() * B
        total_cls += loss_cls.item() * B
        total_fm += loss_fm.item() * B
        total_phys += loss_phys.item() * B

        pred = cls_logits.argmax(dim=1)
        correct += (pred == cls_label).sum().item()
        total += B

        for i in range(B):
            gt = cls_label[i].item()
            class_total[gt] += 1
            if pred[i].item() == gt:
                class_correct[gt] += 1

        # ODE 重建 MAE (推理质量)
        a_hat = model.sample_ode(x, n_steps=10)
        mae = (a_hat - atk_seq).abs().mean(dim=[1, 2])
        for i in range(B):
            gt = cls_label[i].item()
            recon_mae_by_class[gt].append(mae[i].item())

    n = max(total, 1)
    per_class_acc = {}
    for cls_idx in range(9):
        if class_total[cls_idx] > 0:
            per_class_acc[ALL_ATTACK_TYPES[cls_idx]] = (
                class_correct[cls_idx] / class_total[cls_idx])

    per_class_mae = {}
    for cls_idx in range(9):
        if recon_mae_by_class[cls_idx]:
            per_class_mae[ALL_ATTACK_TYPES[cls_idx]] = np.mean(recon_mae_by_class[cls_idx])

    return (total_loss / n, total_cls / n, total_fm / n,
            total_phys / n, correct / n, per_class_acc, per_class_mae)


# ============================================================================
# 3. 可视化
# ============================================================================

def plot_training_curves(history: dict, save_path: str):
    """绘制训练曲线。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('PINN-Flow CFMDetector Training', fontsize=14, fontweight='bold')
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

    ax = axes[0, 2]
    ax.plot(epochs, history['train_fm'], 'b-', alpha=0.7, label='Train')
    ax.plot(epochs, history['val_fm'], 'r-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('FM Loss')
    ax.set_title('Flow Matching MSE'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history['train_phys'], 'b-', alpha=0.7, label='Train')
    ax.plot(epochs, history['val_phys'], 'r-', label='Val')
    ax.axhline(y=TRACE_R, color='gray', linestyle='--', alpha=0.5,
               label=f'Tr(R)={TRACE_R}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Physics Loss')
    ax.set_title('PINN Physics Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history['val_recon_mae'], 'r-', label='Val ODE MAE')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MAE')
    ax.set_title('ODE Reconstruction MAE (10-step Euler)'); ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
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
# 4. 主训练流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PINN-Flow CFMDetector 训练')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--lambda-fm', type=float, default=LAMBDA_FM)
    parser.add_argument('--lambda-phys', type=float, default=LAMBDA_PHYS)
    parser.add_argument('--d-model', type=int, default=D_MODEL)
    parser.add_argument('--transformer-layers', type=int, default=NUM_TRANSFORMER_LAYERS)
    parser.add_argument('--heads', type=int, default=NUM_HEADS)
    parser.add_argument('--flow-blocks', type=int, default=NUM_FLOW_BLOCKS)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--eval-only', type=str, default=None)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    # 验证数据
    if not os.path.exists(os.path.join(args.data_dir, 'X_train.npy')):
        print(f"[ERROR] 预处理数据未找到: {args.data_dir}/X_train.npy")
        print(f"请先运行: python preprocess_data.py")
        sys.exit(1)

    # 加载归一化参数
    norm_path = os.path.join(args.data_dir, 'normalizer.npz')
    if not os.path.exists(norm_path):
        print(f"[ERROR] 归一化参数未找到: {norm_path}")
        sys.exit(1)
    normalizer = dict(np.load(norm_path))

    print("=" * 60)
    print("PINN-Flow CFMDetector Training")
    print("=" * 60)
    print(f"  设备:         {DEVICE}")
    if DEVICE.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU:          {gpu_name} ({gpu_mem:.0f}GB)")
    print(f"  AMP:          {USE_AMP}")
    print(f"  d_model:      {args.d_model}")
    print(f"  Transformer:  {args.transformer_layers}L × {args.heads}H")
    print(f"  Flow blocks:  {args.flow_blocks}")
    print(f"  λ_fm:         {args.lambda_fm}")
    print(f"  λ_phys:       {args.lambda_phys}")
    print(f"  噪声临界值:   Tr(R) = {TRACE_R}")
    print(f"  归一化:       {norm_path}")
    print(f"    feat_median: {normalizer['feat_median']}")
    print(f"    feat_iqr:    {normalizer['feat_iqr']}")
    print(f"    cmd_max:     {normalizer['cmd_max']}")
    print("=" * 60)

    # ---- 加载数据 ----
    train_dataset = PreprocessedDataset(args.data_dir, 'train')
    val_dataset = PreprocessedDataset(args.data_dir, 'val')

    sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=True,
                              prefetch_factor=PREFETCH_FACTOR)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True, persistent_workers=True,
                            prefetch_factor=PREFETCH_FACTOR)

    # ---- 模型 ----
    sample_x, _, _ = train_dataset[0]
    in_channels = sample_x.shape[1]
    window_size = sample_x.shape[0]

    model = CFMDetector(
        in_channels=in_channels, window_size=window_size,
        d_model=args.d_model, num_classes=len(ALL_ATTACK_TYPES),
        num_transformer_layers=args.transformer_layers,
        num_heads=args.heads,
        dim_feedforward=DIM_FEEDFORWARD,
        num_flow_blocks=args.flow_blocks,
        dim_feedforward_flow=DIM_FEEDFORWARD_FLOW,
        dropout=args.dropout,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {n_params:,} (~{n_params/1e6:.2f}M)")

    # 仅评估
    if args.eval_only:
        print(f"\n加载模型: {args.eval_only}")
        state = torch.load(args.eval_only, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state, strict=False)
        _, _, _, _, val_acc, per_class, per_class_mae = evaluate(
            model, val_loader, DEVICE, normalizer, args.lambda_fm, args.lambda_phys)
        print(f"\n验证准确率: {val_acc:.4f}")
        print(f"\n每类准确率:")
        for cls_name in ALL_ATTACK_TYPES:
            acc = per_class.get(cls_name, 0.0)
            print(f"  {cls_name} ({ATTACK_NAMES_CN[cls_name]}): {acc*100:5.1f}%")
        print(f"\n每类 ODE 重建 MAE (Euler 10步):")
        for cls_name in ALL_ATTACK_TYPES:
            mae = per_class_mae.get(cls_name, float('nan'))
            print(f"  {cls_name} ({ATTACK_NAMES_CN[cls_name]}): {mae:.4f}")
        return

    # ---- 优化器 & 调度器 ----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.12, div_factor=25, final_div_factor=1000,
        anneal_strategy='cos')

    scaler = torch.amp.GradScaler('cuda') if USE_AMP and DEVICE.type == 'cuda' else None

    # ---- 训练 ----
    history = defaultdict(list)
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_path = os.path.join(MODEL_DIR, 'cfm_cls_best.pt')

    print(f"\n{'='*60}")
    print(f"开始训练 ({steps_per_epoch} steps/epoch, {args.epochs} epochs)")
    print(f"{'='*60}")

    import time as _time
    for epoch in range(1, args.epochs + 1):
        t0 = _time.time()

        train_loss, train_cls, train_fm, train_phys, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE,
            normalizer, scaler=scaler,
            lambda_fm=args.lambda_fm, lambda_phys=args.lambda_phys)

        val_loss, val_cls, val_fm, val_phys, val_acc, per_class, per_class_mae = evaluate(
            model, val_loader, DEVICE, normalizer,
            args.lambda_fm, args.lambda_phys)

        elapsed = _time.time() - t0
        val_recon_mae = np.mean(list(per_class_mae.values())) if per_class_mae else 0.0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_fm'].append(train_fm)
        history['train_phys'].append(train_phys)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_fm'].append(val_fm)
        history['val_phys'].append(val_phys)
        history['val_recon_mae'].append(val_recon_mae)
        history['val_per_class'].append(per_class)

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            marker = ' *'
        else:
            epochs_no_improve += 1
            marker = ''

        lr_now = optimizer.param_groups[0]['lr']
        print(f"E{epoch:3d} | LR={lr_now:.1e} | {elapsed:.0f}s | "
              f"Acc={val_acc:.3f}({best_val_acc:.3f}) | "
              f"FM={val_fm:.4f} Phys={val_phys:.4f} | "
              f"ODE_MAE={val_recon_mae:.4f} | no_imp={epochs_no_improve}{marker}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n早停: {EARLY_STOP_PATIENCE} epochs 未改善, epoch {epoch}")
            break

    # ---- 保存 ----
    final_path = os.path.join(MODEL_DIR, 'cfm_cls_final.pt')
    torch.save(model.state_dict(), final_path)

    config = {
        'in_channels': in_channels, 'window_size': window_size,
        'd_model': args.d_model, 'num_classes': len(ALL_ATTACK_TYPES),
        'num_transformer_layers': args.transformer_layers,
        'num_heads': args.heads,
        'dim_feedforward': DIM_FEEDFORWARD,
        'num_flow_blocks': args.flow_blocks,
        'dim_feedforward_flow': DIM_FEEDFORWARD_FLOW,
        'dropout': args.dropout,
        'model_type': 'cfm',
    }
    np.savez(os.path.join(MODEL_DIR, 'cfm_cls_config.npz'),
             **{k: np.array(v) if isinstance(v, list) else v
                for k, v in config.items()})

    # ---- 最终评估 ----
    print(f"\n{'='*60}")
    print("最终评估 (最佳模型)")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    _, _, _, _, val_acc, per_class, per_class_mae = evaluate(
        model, val_loader, DEVICE, normalizer,
        args.lambda_fm, args.lambda_phys)

    print(f"\n最佳验证准确率: {val_acc:.4f}")
    print(f"\n每类准确率:")
    for cls_name in ALL_ATTACK_TYPES:
        acc = per_class.get(cls_name, 0.0)
        bar = '#' * int(acc * 50) + '.' * (50 - int(acc * 50))
        print(f"  {cls_name} ({ATTACK_NAMES_CN[cls_name]}): {bar} {acc*100:5.1f}%")

    print(f"\n每类 ODE 重建 MAE (Euler 10步):")
    for cls_name in ALL_ATTACK_TYPES:
        mae = per_class_mae.get(cls_name, float('nan'))
        bar = '#' * min(int(mae * 200), 50) + '.' * max(50 - int(mae * 200), 0)
        print(f"  {cls_name} ({ATTACK_NAMES_CN[cls_name]}): {bar} {mae:.4f}")

    print(f"\n最佳 epoch: {best_epoch} | 最佳 Acc: {best_val_acc:.4f}")
    print(f"模型: {best_path} | {final_path}")

    if not args.no_plot:
        plot_training_curves(history, os.path.join(MODEL_DIR, 'cfm_curves.png'))

    print(f"\n训练完成!")


if __name__ == "__main__":
    main()
