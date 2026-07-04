"""
evaluate.py — 测试集分类评估与可视化
======================================
加载训练好的检测模型, 在测试集上推理并生成评估报告。

输出: eval/{model_name}_{timestamp}/
  ├── confusion_matrix.png
  ├── per_class_metrics.png
  ├── metrics.txt / metrics.csv
  ├── classification_report.txt
  └── analysis_report.md
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from sklearn.metrics import classification_report
from attack import ALL_ATTACK_TYPES, ATTACK_NAMES_CN, ATK_COLORS

DEFAULT_BATCH_SIZE = 2048


def parse_args():
    parser = argparse.ArgumentParser(description='检测器 测试集分类评估')
    parser.add_argument('--model-path', type=str,
                        default=os.path.join(PROJECT_ROOT, 'detector', 'models', 'cfm_cls_best.pt'))
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'dataset_win'))
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def load_model(weight_path: str, device: torch.device):
    from detector.detector import Detector
    model = Detector().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model


@torch.no_grad()
def run_inference(model, dataloader, device):
    model.eval()
    y_true_all, y_pred_all, y_conf_all = [], [], []
    for batch in dataloader:
        x_batch = batch[0].to(device)
        logits, _ = model(x_batch)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        y_pred_all.extend(pred.cpu().numpy().tolist())
        y_conf_all.extend(conf.cpu().numpy().tolist())
        y_true_all.extend(batch[1].numpy().tolist())
    return (np.array(y_true_all), np.array(y_pred_all), np.array(y_conf_all, dtype=np.float32))


def build_confusion_matrix(y_true, y_pred, num_classes=8):
    cm_counts = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm_counts[t, p] += 1
    cm_norm = cm_counts.astype(np.float64) / np.maximum(cm_counts.sum(axis=1, keepdims=True), 1)
    return cm_counts, cm_norm


def compute_per_class_metrics(y_true, y_pred, y_conf, class_names):
    rows = []
    for cls_idx, atk in enumerate(class_names):
        true_mask = y_true == cls_idx
        pred_mask = y_pred == cls_idx
        tp = int(np.sum(true_mask & pred_mask))
        n_true = int(np.sum(true_mask))
        n_pred = int(np.sum(pred_mask))
        accuracy = tp / max(n_true, 1)
        precision = tp / max(n_pred, 1)
        recall = tp / max(n_true, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        mean_conf = float(np.mean(y_conf[true_mask])) if n_true > 0 else 0.0
        rows.append({'attack_type': atk, 'name_cn': ATTACK_NAMES_CN.get(atk, ''),
                     'accuracy': accuracy, 'precision': precision, 'recall': recall,
                     'f1_score': f1, 'mean_confidence': mean_conf, 'n_samples': n_true})
    return pd.DataFrame(rows)


def generate_classification_report(y_true, y_pred, class_names, save_path):
    present = sorted(set(y_true.flat) | set(y_pred.flat))
    present = [i for i in present if i < len(class_names)]
    present_names = [f'{class_names[i]} ({ATTACK_NAMES_CN.get(class_names[i], "")})' for i in present]
    cr_text = classification_report(y_true, y_pred, labels=present,
                                     target_names=present_names, digits=4, zero_division=0)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("检测器 测试集分类报告\n" + "=" * 60 + "\n" + cr_text)
    print(f"  [Classification Report] {save_path}")


def plot_confusion_matrix(cm_counts, cm_norm, class_names, save_path):
    n = len(class_names)
    labels_cn = [f'{a}\n{ATTACK_NAMES_CN[a]}' for a in class_names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    fig.suptitle('攻击分类混淆矩阵', fontsize=13, fontweight='bold')

    im1 = ax1.imshow(cm_counts, cmap='Blues', aspect='equal')
    for i in range(n):
        for j in range(n):
            val = cm_counts[i, j]
            if val > 0:
                color = 'white' if val > cm_counts.max() * 0.5 else 'black'
                ax1.text(j, i, str(val), ha='center', va='center', fontsize=8,
                        color=color, fontweight='bold' if i == j else 'normal')
    ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
    ax1.set_xticklabels(labels_cn, fontsize=7); ax1.set_yticklabels(labels_cn, fontsize=7)
    ax1.set_xlabel('预测类别'); ax1.set_ylabel('真实类别')
    ax1.set_title('原始计数', fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='样本数')

    im2 = ax2.imshow(cm_norm * 100, cmap='RdYlGn', aspect='equal', vmin=0, vmax=100)
    for i in range(n):
        for j in range(n):
            val_pct = cm_norm[i, j] * 100
            if val_pct > 0:
                color = 'white' if val_pct > 50 else 'black'
                ax2.text(j, i, f'{val_pct:.1f}%', ha='center', va='center', fontsize=8,
                        color=color, fontweight='bold' if i == j else 'normal')
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels_cn, fontsize=7); ax2.set_yticklabels(labels_cn, fontsize=7)
    ax2.set_xlabel('预测类别'); ax2.set_ylabel('真实类别')
    ax2.set_title('行归一化 [%]', fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='百分比 [%]')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Confusion Matrix] {save_path}")


def plot_per_class_metrics(df, save_path):
    n = len(df)
    atks = df['attack_type'].tolist()
    x = np.arange(n)
    colors = [ATK_COLORS.get(a, '#888888') for a in atks]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))
    fig.suptitle('检测器 测试集各类别分类指标', fontsize=13, fontweight='bold')

    bars = ax1.bar(x, df['accuracy'] * 100, color=colors, alpha=0.85, edgecolor='black', lw=0.5)
    for bar, v in zip(bars, df['accuracy']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                 f'{v*100:.1f}%', ha='center', fontsize=7.5, fontweight='bold')
    ax1_f1 = ax1.twinx()
    ax1_f1.plot(x, df['f1_score'], 'D-', color='#E65100', lw=1.5, ms=6, label='F1')
    for i, f1 in enumerate(df['f1_score']):
        ax1_f1.annotate(f'{f1:.3f}', (i, f1), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=6.5, color='#BF360C')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{a}\n{ATTACK_NAMES_CN[a]}' for a in atks], fontsize=7)
    ax1.set_ylabel('准确率 [%]'); ax1_f1.set_ylabel('F1'); ax1_f1.set_ylim(0, 1.05)
    ax1.set_title('各类别准确率 & F1', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_f1.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)

    conf_bars = ax2.bar(x, df['mean_confidence'], color=colors, alpha=0.85, edgecolor='black', lw=0.5)
    for bar, v in zip(conf_bars, df['mean_confidence']):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{v:.3f}', ha='center', fontsize=7.5, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{a}\n{ATTACK_NAMES_CN[a]}' for a in atks], fontsize=7)
    ax2.set_ylabel('平均 softmax 置信度'); ax2.set_ylim(0, 1.1)
    ax2.set_title('各类别平均置信度', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Per-Class Metrics] {save_path}")


def write_metrics_txt(output_dir, model_name, model_path, n_samples, cm_counts, cm_norm, df):
    save_path = os.path.join(output_dir, 'metrics.txt')
    overall_acc = np.trace(cm_counts) / max(cm_counts.sum(), 1)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("检测器 测试集分类评估报告\n" + "=" * 70 + "\n\n")
        f.write(f"模型: {model_name}\n权重: {model_path}\n")
        f.write(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"测试样本: {n_samples:,} 窗口\n\n")

        f.write("-" * 70 + "\n一、总体指标\n" + "-" * 70 + "\n")
        f.write(f"  总准确率: {overall_acc*100:6.2f}%\n")
        f.write(f"  总样本数: {n_samples:,}\n\n")

        f.write("-" * 70 + "\n二、逐类别分类指标\n" + "-" * 70 + "\n")
        header = f"{'攻击':>6}  {'名称':<12}  {'准确率':>8}  {'精确率':>8}  {'召回率':>8}  {'F1':>8}  {'置信度':>8}  {'样本数':>8}"
        f.write(header + "\n" + "-" * 70 + "\n")
        for _, row in df.iterrows():
            f.write(f"{row['attack_type']:>6}  {row['name_cn']:<12}  "
                    f"{row['accuracy']*100:>7.1f}%  {row['precision']:>7.4f}  "
                    f"{row['recall']:>7.4f}  {row['f1_score']:>7.4f}  "
                    f"{row['mean_confidence']:>7.3f}  {int(row['n_samples']):>8,}\n")
        f.write("\n" + "-" * 70 + "\n三、混淆矩阵 (原始计数)\n" + "-" * 70 + "\n")
        atks_order = ALL_ATTACK_TYPES
        f.write(f"{'':>10}" + "".join(f"{a:>8}" for a in atks_order) + "\n")
        for i, a in enumerate(atks_order):
            f.write(f"{a:>10}" + "".join(f"{cm_counts[i, j]:>8}" for j in range(len(atks_order))) + "\n")
        f.write("\n混淆矩阵 (行归一化 %)\n" + "-" * 70 + "\n")
        f.write(f"{'':>10}" + "".join(f"{a:>8}" for a in atks_order) + "\n")
        for i, a in enumerate(atks_order):
            f.write(f"{a:>10}" + "".join(f"{cm_norm[i, j]*100:>7.1f}%" for j in range(len(atks_order))) + "\n")
    print(f"  [Metrics TXT] {save_path}")


def generate_markdown_report(output_dir, model_name, model_path, n_samples, cm_counts, cm_norm, df):
    save_path = os.path.join(output_dir, 'analysis_report.md')
    overall_acc = np.trace(cm_counts) / max(cm_counts.sum(), 1)
    lines = [
        "# 检测器 测试集分类评估报告", "",
        f"- **模型**: {model_name}", f"- **权重**: `{model_path}`",
        f"- **日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- **测试样本**: {n_samples:,} 窗口", "",
        "## 总体指标", "",
        "| 指标 | 数值 |", "|------|------|",
        f"| 总准确率 | {overall_acc*100:.2f}% |",
        f"| 总样本数 | {n_samples:,} |", "",
        "## 逐类别分类指标", "",
        "| 攻击 | 名称 | 准确率 | 精确率 | 召回率 | F1 | 置信度 | 样本数 |",
        "|------|------|--------|--------|--------|-----|--------|--------|",
    ]
    for _, row in df.iterrows():
        lines.append(f"| {row['attack_type']} | {row['name_cn']} | "
                     f"{row['accuracy']*100:.1f}% | {row['precision']:.4f} | "
                     f"{row['recall']:.4f} | {row['f1_score']:.4f} | "
                     f"{row['mean_confidence']:.3f} | {int(row['n_samples']):,} |")
    lines.append("")

    atks_order = ALL_ATTACK_TYPES
    lines.extend(["## 混淆矩阵 (行归一化 %)", "",
                  "| 真实\\预测 | " + " | ".join(atks_order) + " |",
                  "|" + "|".join(["------"] * (len(atks_order) + 1)) + "|"])
    for i, a in enumerate(atks_order):
        row = f"| {a} | " + " | ".join(f"{cm_norm[i, j]*100:.1f}%" for j in range(len(atks_order))) + " |"
        lines.append(row)

    best_idx = df['f1_score'].idxmax()
    worst_idx = df['f1_score'].idxmin()
    lines.extend(["", "## 关键发现", "",
                  f"- **最佳分类**: {df.loc[best_idx, 'attack_type']} "
                  f"({df.loc[best_idx, 'name_cn']}), F1={df.loc[best_idx, 'f1_score']:.4f}",
                  f"- **最差分类**: {df.loc[worst_idx, 'attack_type']} "
                  f"({df.loc[worst_idx, 'name_cn']}), F1={df.loc[worst_idx, 'f1_score']:.4f}", ""])

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  [Markdown Report] {save_path}")


def main():
    from torch.utils.data import DataLoader
    from detector.train import PreprocessedDataset

    args = parse_args()
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = args.output_dir or os.path.join(PROJECT_ROOT, 'eval', f'{model_name}_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    print("检测器 测试集分类评估")
    print(f"  模型: {args.model_path}")
    print(f"  数据: {args.data_dir}")
    print(f"  设备: {device}")
    print(f"  输出: {output_root}")

    print("\n[1/5] 加载模型...")
    model = load_model(args.model_path, device)

    print("\n[2/5] 加载测试集...")
    test_dataset = PreprocessedDataset(args.data_dir, split='test', load_clean=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    print(f"  测试窗口: {len(test_dataset):,}")

    print(f"\n[3/5] 批量推理...")
    y_true, y_pred, y_conf = run_inference(model, test_loader, device)
    print(f"  推理完成: {len(y_true):,} 样本")

    print(f"\n[4/5] 计算指标...")
    cm_counts, cm_norm = build_confusion_matrix(y_true, y_pred)
    overall_acc = np.trace(cm_counts) / max(cm_counts.sum(), 1)
    print(f"  总准确率: {overall_acc*100:.2f}%")
    df = compute_per_class_metrics(y_true, y_pred, y_conf, ALL_ATTACK_TYPES)
    generate_classification_report(y_true, y_pred, ALL_ATTACK_TYPES,
                                   os.path.join(output_root, 'classification_report.txt'))

    print(f"\n[5/5] 生成图表和报告...")
    plot_confusion_matrix(cm_counts, cm_norm, ALL_ATTACK_TYPES,
                          os.path.join(output_root, 'confusion_matrix.png'))
    plot_per_class_metrics(df, os.path.join(output_root, 'per_class_metrics.png'))
    write_metrics_txt(output_root, model_name, args.model_path, len(y_true), cm_counts, cm_norm, df)
    pd.DataFrame(df).to_csv(os.path.join(output_root, 'metrics.csv'), index=False, encoding='utf-8-sig')
    generate_markdown_report(output_root, model_name, args.model_path, len(y_true), cm_counts, cm_norm, df)

    print(f"\n评估完成! 总准确率: {overall_acc*100:.2f}%")


if __name__ == "__main__":
    main()
