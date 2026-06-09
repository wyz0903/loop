"""
visualize_errors.py — 可视化分类模型错误样本
==============================================
加载训练好的模型，在验证集上找分类错误，按类别分组可视化。

用法:
  python visualize_errors.py                          # 默认 config 划分
  python visualize_errors.py --data-dir dataset_win/trajectory
  python visualize_errors.py --model models/cls_best.pt --num 12
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# IEEE 论文标准字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, 'dataset_win', 'config')
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, 'models', 'cls_best.pt')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

ALL_ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
ATTACK_NAMES_CN = {
    'A0': 'Normal', 'A1': 'Constant Bias', 'A2': 'Sinusoidal',
    'A3': 'Drift', 'A4': 'Step', 'A5': 'Replay Attack',
    'A6': 'Dropout', 'A7': 'Scaling', 'A8': 'Freeze',
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 通道名称
CHANNEL_NAMES = ['ekf_innov_x', 'ekf_innov_y', 'ekf_innov_θ', 'u_cmd_v', 'u_cmd_ω']
CHANNEL_LABELS = ['Inn. x [m]', 'Inn. y [m]', 'Inn. θ [rad]', 'Cmd v [m/s]', 'Cmd ω [rad/s]']


def load_model(model_path: str):
    """加载训练好的模型"""
    from detector.attack_classifier import AttackClassifier
    from detector.config import ENC_CHANNELS, LATENT_DIM

    config_path = os.path.join(os.path.dirname(model_path), 'cls_config.npz')
    if os.path.exists(config_path):
        cfg = np.load(config_path, allow_pickle=True)
        in_channels = int(cfg['in_channels'])
        window_size = int(cfg['window_size'])
        latent_dim = int(cfg['latent_dim'])
        enc_channels = cfg['enc_channels'].tolist() if 'enc_channels' in cfg else ENC_CHANNELS
    else:
        in_channels, window_size, latent_dim = 5, 100, LATENT_DIM
        enc_channels = ENC_CHANNELS

    model = AttackClassifier(
        in_channels=in_channels, window_size=window_size,
        latent_dim=latent_dim, num_classes=len(ALL_ATTACK_TYPES),
        enc_channels=enc_channels
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Model loaded: {model_path}")
    print(f"  Params: in={in_channels}, win={window_size}, latent={latent_dim}, enc={enc_channels}")
    return model, window_size


def find_misclassified(model, X, cls_true, batch_size=1024):
    """找所有分类错误的样本"""
    model.eval()
    all_preds = []
    n = len(X)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.from_numpy(X[start:end]).to(DEVICE)
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    logits, _, _ = model(x)
            else:
                logits, _, _ = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds)
    cls_true = np.asarray(cls_true)

    # 找错误索引
    error_mask = all_preds != cls_true
    error_indices = np.where(error_mask)[0]
    error_true = cls_true[error_indices]
    error_pred = all_preds[error_indices]

    # 按 (true, pred) 对分组
    error_groups = defaultdict(list)
    for idx, t, p in zip(error_indices, error_true, error_pred):
        error_groups[(t, p)].append(idx)

    print(f"Total: {n:,}  Errors: {len(error_indices):,}  ({len(error_indices)/n*100:.2f}%)")
    return error_indices, error_true, error_pred, error_groups


def plot_confusion_summary(error_groups: dict, save_path: str):
    """绘制混淆摘要热力图 (仅错误部分)"""
    confusion = np.zeros((9, 9), dtype=int)
    for (t, p), indices in error_groups.items():
        confusion[t, p] = len(indices)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(confusion, cmap='YlOrRd', aspect='auto')

    for i in range(9):
        for j in range(9):
            if confusion[i, j] > 0:
                text_color = 'white' if confusion[i, j] > confusion.max() * 0.6 else 'black'
                ax.text(j, i, str(confusion[i, j]), ha='center', va='center',
                       fontsize=9, color=text_color, fontweight='bold')

    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels([f'{a}\n{ATTACK_NAMES_CN[a]}' for a in ALL_ATTACK_TYPES], fontsize=8)
    ax.set_yticklabels([f'{a}\n{ATTACK_NAMES_CN[a]}' for a in ALL_ATTACK_TYPES], fontsize=8)
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title('Classification Error Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Error Count')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Fig] Confusion summary: {save_path}")


def plot_error_samples(X, cls_true, error_groups, save_path: str,
                       max_per_group: int = 3, max_groups: int = 12):
    """可视化最有代表性的错误样本

    选择错误样本最多的前 max_groups 个 (true→pred) 组合，
    每个组合一行，每行展示 max_per_group 个样本。
    """
    sorted_groups = sorted(error_groups.items(), key=lambda x: -len(x[1]))
    selected = sorted_groups[:max_groups]
    n_groups = len(selected)

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

    fig, axes = plt.subplots(n_groups, max_per_group,
                             figsize=(max_per_group * 5, n_groups * 3))
    if n_groups == 1:
        axes = axes.reshape(1, -1)

    for row_idx, ((true_cls, pred_cls), indices) in enumerate(selected):
        sample_indices = indices[:max_per_group]
        true_name = f'{ALL_ATTACK_TYPES[true_cls]} {ATTACK_NAMES_CN[ALL_ATTACK_TYPES[true_cls]]}'
        pred_name = f'{ALL_ATTACK_TYPES[pred_cls]} {ATTACK_NAMES_CN[ALL_ATTACK_TYPES[pred_cls]]}'

        for col_idx in range(max_per_group):
            ax = axes[row_idx, col_idx]
            if col_idx < len(sample_indices):
                window = X[sample_indices[col_idx]]
                for ch in range(5):
                    ax.plot(window[:, ch], linewidth=0.7, color=colors[ch],
                           alpha=0.85)
                ax.set_xlabel('Time Step', fontsize=7)
                ax.set_ylabel('Signal Value', fontsize=7)
            else:
                ax.text(0.5, 0.5, '(insufficient samples)', transform=ax.transAxes,
                       ha='center', va='center', fontsize=9, color='gray')
                ax.set_xticks([]); ax.set_yticks([])

            ax.tick_params(labelsize=6)

            if col_idx == 0:
                ax.set_title(f'True: {true_name}\n-> Pred: {pred_name}',
                            fontsize=8, fontweight='bold', color='#d62728')
            else:
                ax.set_title(f'Sample {col_idx+1}', fontsize=8)

    handles = [plt.Line2D([0], [0], color=colors[i], linewidth=2,
                         label=f'{CHANNEL_NAMES[i]} ({CHANNEL_LABELS[i]})')
               for i in range(5)]
    fig.legend(handles=handles, loc='upper right', ncol=1, fontsize=7,
              bbox_to_anchor=(0.98, 0.97), framealpha=0.9)

    fig.suptitle('Classification Error Samples (sorted by error count)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"  [Fig] Error samples: {save_path}")


def plot_per_class_mistakes(X, cls_true, error_indices, error_true, error_pred, save_path: str):
    """按真实类别展示错误样本：挑最重要的两类误判各展示一例"""
    n_classes = len(ALL_ATTACK_TYPES)
    samples_per_class = 2

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

    fig, axes = plt.subplots(n_classes, samples_per_class,
                             figsize=(samples_per_class * 5, n_classes * 2.5))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for cls_idx in range(n_classes):
        cls_name = ALL_ATTACK_TYPES[cls_idx]
        # 该真实类别的错误 mask
        mask = error_true == cls_idx
        cls_err_indices = error_indices[mask]
        cls_err_preds = error_pred[mask]

        if len(cls_err_indices) == 0:
            for col in range(samples_per_class):
                ax = axes[cls_idx, col]
                ax.text(0.5, 0.5, f'{cls_name}\n({ATTACK_NAMES_CN[cls_name]})\nNo errors',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=9, color='#2ca02c')
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor('#2ca02c')
            continue

        # 按错误预测类型分组，取最多的两个
        from collections import Counter
        pred_counts = Counter(cls_err_preds)
        top_preds = [p for p, _ in pred_counts.most_common(samples_per_class)]

        for col, pred_cls in enumerate(top_preds):
            if col >= samples_per_class:
                break
            match = cls_err_indices[cls_err_preds == pred_cls]
            sample_idx = match[0]
            ax = axes[cls_idx, col]
            window = X[sample_idx]

            for ch in range(5):
                ax.plot(window[:, ch], linewidth=0.7, color=colors[ch], alpha=0.85)

            ax.set_title(f'True: {cls_name} {ATTACK_NAMES_CN[cls_name]}\n'
                        f'-> Miscls: {ALL_ATTACK_TYPES[pred_cls]} '
                        f'{ATTACK_NAMES_CN[ALL_ATTACK_TYPES[pred_cls]]}',
                        fontsize=7, color='#d62728')
            ax.set_xlabel('Time Step', fontsize=6)
            ax.set_ylabel('Signal Value', fontsize=6)
            ax.tick_params(labelsize=5)

        # 填充不足的列
        for col in range(len(top_preds), samples_per_class):
            ax = axes[cls_idx, col]
            ax.text(0.5, 0.5, '(only this error type)', transform=ax.transAxes,
                   ha='center', va='center', fontsize=8, color='gray')
            ax.set_xticks([]); ax.set_yticks([])

    # 图例
    handles = [plt.Line2D([0], [0], color=colors[i], linewidth=2,
                         label=f'{CHANNEL_NAMES[i]} ({CHANNEL_LABELS[i]})')
               for i in range(5)]
    fig.legend(handles=handles, loc='upper right', ncol=1, fontsize=6,
              bbox_to_anchor=(0.99, 0.99), framealpha=0.9)

    fig.suptitle('Typical Errors per Class (sorted by misclassification frequency)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"  [Fig] Per-class errors: {save_path}")


def print_error_stats(error_groups: dict):
    """打印错误统计"""
    print(f"\n{'='*60}")
    print("Error Statistics (True -> Predicted)")
    print(f"{'='*60}")
    sorted_groups = sorted(error_groups.items(), key=lambda x: -len(x[1]))
    for (t, p), indices in sorted_groups[:20]:
        true_name = f'{ALL_ATTACK_TYPES[t]} {ATTACK_NAMES_CN[ALL_ATTACK_TYPES[t]]}'
        pred_name = f'{ALL_ATTACK_TYPES[p]} {ATTACK_NAMES_CN[ALL_ATTACK_TYPES[p]]}'
        print(f"  {true_name:20s} → {pred_name:20s} : {len(indices):5d} 个")


def main():
    parser = argparse.ArgumentParser(description='可视化分类错误样本')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--num', type=int, default=12,
                       help='最大展示的错误组数')
    parser.add_argument('--prefix', type=str, default='err',
                       help='输出文件前缀')
    args = parser.parse_args()

    print("=" * 60)
    print("Classification Error Visualization")
    print("=" * 60)

    # 加载模型
    model, window_size = load_model(args.model)

    # 加载验证数据
    X_val = np.load(os.path.join(args.data_dir, 'X_val.npy')).astype(np.float32)
    y_val = np.load(os.path.join(args.data_dir, 'Y_val_cls.npy')).astype(np.int64)
    print(f"Validation set: {len(X_val):,} samples, shape={X_val.shape}")

    # 找错误
    error_indices, error_true, error_pred, error_groups = find_misclassified(
        model, X_val, y_val)

    if len(error_indices) == 0:
        print("\nNo classification errors!")
        return

    # 打印统计
    print_error_stats(error_groups)

    # 可视化
    prefix = os.path.join(RESULT_DIR, args.prefix)

    # 图1: 混淆矩阵
    plot_confusion_summary(error_groups, f'{prefix}_confusion.png')

    # 图2: 错误样本详情
    plot_error_samples(X_val, y_val, error_groups, f'{prefix}_samples.png',
                       max_per_group=3, max_groups=args.num)

    # 图3: 各类别错误
    plot_per_class_mistakes(X_val, y_val, error_indices, error_true, error_pred,
                           f'{prefix}_per_class.png')

    print(f"\nDone! Output saved to: {RESULT_DIR}/")
    print(f"  {prefix}_confusion.png  -- Error confusion matrix")
    print(f"  {prefix}_samples.png   -- High-frequency error samples")
    print(f"  {prefix}_per_class.png -- Per-class typical errors")


if __name__ == "__main__":
    main()
