"""
evaluate.py — 测试集分类评估与可视化
======================================
加载训练好的 CFMDetector 模型, 在测试集上推理并生成完整的评估报告。

输出结构:
  eval/{model_name}_{timestamp}/
  ├── confusion_matrix.png       # 双面板 (计数 + 行归一化%)
  ├── per_class_metrics.png      # 各类别准确率/F1/置信度汇总
  ├── metrics.txt                # 综合文本报告
  ├── metrics.csv                # 逐类指标 CSV
  ├── classification_report.txt  # sklearn 分类报告
  └── analysis_report.md         # Markdown 分析报告

用法:
  python detector/evaluate.py                           # 使用全部默认路径
  python detector/evaluate.py --model-path detector/models/cfm_cls_best.pt
  python detector/evaluate.py --data-dir dataset_win --batch-size 1024
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# IEEE 论文绘图样式 (支持中英文混排)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# ---- 项目路径 ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

# 攻击类型定义 (来自 attack.py 唯一数据源)
from attack import ALL_ATTACK_TYPES, ATTACK_NAMES_CN, ATK_COLORS

# ============================================================================
# 可调参数 (集中配置)
# ============================================================================
DEFAULT_BATCH_SIZE = 2048
DEFAULT_MODEL_PATH = 'detector/models/cfm_cls_best.pt'
DEFAULT_DATA_DIR = 'dataset_win'

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='CFMDetector 测试集分类评估')
    parser.add_argument('--model-path', type=str,
                        default=os.path.join(PROJECT_ROOT, DEFAULT_MODEL_PATH),
                        help=f'模型权重路径 (默认: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--config-path', type=str, default=None,
                        help='模型配置 npz 路径 (默认: 从 model-path 同目录自动查找)')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, DEFAULT_DATA_DIR),
                        help=f'预处理数据目录 (默认: {DEFAULT_DATA_DIR})')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: eval/{model_name}_{timestamp}/)')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'推理批大小 (默认: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--device', type=str, default=None,
                        help='推理设备 (默认: 自动检测 cuda/cpu)')
    return parser.parse_args()


# ============================================================================
# 目录创建
# ============================================================================

def build_eval_dirs(output_root: str) -> str:
    """创建评估输出目录, 返回根目录路径。"""
    os.makedirs(output_root, exist_ok=True)
    return output_root


# ============================================================================
# 模型加载
# ============================================================================

def load_model(config_path: str, weight_path: str, device: torch.device):
    """从配置和权重文件构建 CFMDetector 模型。

    Args:
        config_path: cfm_cls_config.npz 路径
        weight_path: .pt 权重路径
        device: torch device

    Returns:
        CFMDetector 模型 (已置于 eval 模式)
    """
    from detector.cfm_detector import CFMDetector

    # 读取配置, 缺失时使用模块默认值
    cfg = {}
    if os.path.exists(config_path):
        cfg = dict(np.load(config_path, allow_pickle=True))
        print(f"  配置加载: {config_path}")
    else:
        print(f"  [WARN] 配置文件不存在: {config_path}, 使用默认参数")

    # 将 npz 中 object 类型值转为 Python 原生类型
    def _cast(v):
        if isinstance(v, np.ndarray) and v.ndim == 0:
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    model = CFMDetector(
        in_channels=int(_cast(cfg.get('in_channels', 8))),
        window_size=int(_cast(cfg.get('window_size', 100))),
        d_model=int(_cast(cfg.get('d_model', 128))),
        num_classes=int(_cast(cfg.get('num_classes', 9))),
        backbone_type=str(_cast(cfg.get('backend_type', cfg.get('backbone_type', 'simple_conv')))),
        conv_channels=list(_cast(cfg.get('conv_channels', [64, 128, 128]))),
        conv_kernel_size=int(_cast(cfg.get('conv_kernel_size', 3))),
        pool_size=int(_cast(cfg.get('pool_size', 2))),
        use_channel_attn=bool(_cast(cfg.get('use_channel_attn', True))),
        channel_attn_heads=int(_cast(cfg.get('channel_attn_heads', 4))),
        channel_attn_dim=int(_cast(cfg.get('channel_attn_dim', 64))),
        num_transformer_layers=int(_cast(cfg.get('num_transformer_layers', 4))),
        num_heads=int(_cast(cfg.get('num_heads', 8))),
        dim_feedforward=int(_cast(cfg.get('dim_feedforward', 512))),
        dropout=float(_cast(cfg.get('dropout', 0.1))),
    ).to(device)

    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {param_count:,}")
    return model


# ============================================================================
# 推理
# ============================================================================

@torch.no_grad()
def run_inference(model, dataloader, device: torch.device):
    """在 DataLoader 上批量推理, 收集标签、预测和置信度。

    Returns:
        y_true: (N,) int64 — 真实标签索引
        y_pred: (N,) int64 — 预测标签索引
        y_conf: (N,) float32 — softmax 最大概率
    """
    model.eval()
    y_true_all, y_pred_all, y_conf_all = [], [], []

    for x_batch, cls_label in dataloader:
        x_batch = x_batch.to(device)
        logits, _ = model(x_batch)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        y_pred_all.extend(pred.cpu().numpy().tolist())
        y_conf_all.extend(conf.cpu().numpy().tolist())
        y_true_all.extend(cls_label.numpy().tolist())

    return (np.array(y_true_all),
            np.array(y_pred_all),
            np.array(y_conf_all, dtype=np.float32))


# ============================================================================
# 指标计算
# ============================================================================

def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           num_classes: int = 9):
    """构建混淆矩阵 (原始计数 + 行归一化)。

    Returns:
        cm_counts: (num_classes, num_classes) int — 原始计数
        cm_norm:   (num_classes, num_classes) float — 行归一化 [0,1]
    """
    cm_counts = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm_counts[t, p] += 1

    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_norm = cm_counts.astype(np.float64) / np.maximum(row_sums, 1)

    return cm_counts, cm_norm


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              y_conf: np.ndarray, class_names: list) -> pd.DataFrame:
    """计算各类别的准确率、精确率、召回率、F1、平均置信度和样本数。

    Returns:
        DataFrame 行=类别, 列=[accuracy, precision, recall, f1, mean_confidence, n_samples]
    """
    rows = []
    for cls_idx, atk in enumerate(class_names):
        # 真实为该类的样本
        true_mask = y_true == cls_idx
        # 预测为该类的样本
        pred_mask = y_pred == cls_idx

        tp = int(np.sum(true_mask & pred_mask))
        n_true = int(np.sum(true_mask))
        n_pred = int(np.sum(pred_mask))

        accuracy = tp / max(n_true, 1)
        precision = tp / max(n_pred, 1)
        recall = tp / max(n_true, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        mean_conf = float(np.mean(y_conf[true_mask])) if n_true > 0 else 0.0

        rows.append({
            'attack_type': atk,
            'name_cn': ATTACK_NAMES_CN.get(atk, ''),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_confidence': mean_conf,
            'n_samples': n_true,
        })

    return pd.DataFrame(rows)


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                   class_names: list, save_path: str) -> dict:
    """生成 sklearn classification_report 并保存为 txt。

    Returns:
        dict — sklearn classification_report(output_dict=True)
    """
    try:
        from sklearn.metrics import classification_report
    except ImportError:
        print("[WARN] sklearn 未安装, 跳过分类报告")
        return {}

    # 确定实际出现的类别
    present = sorted(set(y_true.flat) | set(y_pred.flat))
    present = [i for i in present if i < len(class_names)]
    present_names = [
        f'{class_names[i]} ({ATTACK_NAMES_CN.get(class_names[i], "")})'
        for i in present
    ]

    cr_text = classification_report(
        y_true, y_pred, labels=present, target_names=present_names,
        digits=4, zero_division=0,
    )
    cr_dict = classification_report(
        y_true, y_pred, labels=present, target_names=present_names,
        digits=4, output_dict=True, zero_division=0,
    )

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("CFMDetector 测试集分类报告\n")
        f.write("=" * 60 + "\n")
        f.write(cr_text)

    print(f"  [Classification Report] {save_path}")
    return cr_dict


# ============================================================================
# 绘图
# ============================================================================

def plot_confusion_matrix(cm_counts: np.ndarray, cm_norm: np.ndarray,
                          class_names: list, save_path: str):
    """双面板混淆矩阵: 左侧原始计数, 右侧行归一化百分比。"""
    n = len(class_names)
    labels_cn = [f'{a}\n{ATTACK_NAMES_CN[a]}' for a in class_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    fig.suptitle('攻击分类混淆矩阵 — CFMDetector (cls-only)', fontsize=13, fontweight='bold')

    # ---- 面板 1: 原始计数 ----
    im1 = ax1.imshow(cm_counts, cmap='Blues', aspect='equal')
    for i in range(n):
        for j in range(n):
            val = cm_counts[i, j]
            color = 'white' if val > cm_counts.max() * 0.5 else 'black'
            ax1.text(j, i, str(val) if val > 0 else '',
                     ha='center', va='center', fontsize=8, color=color,
                     fontweight='bold' if i == j else 'normal')
    ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
    ax1.set_xticklabels(labels_cn, fontsize=7)
    ax1.set_yticklabels(labels_cn, fontsize=7)
    ax1.set_xlabel('预测类别', fontsize=10)
    ax1.set_ylabel('真实类别', fontsize=10)
    ax1.set_title('原始计数', fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='样本数')

    # ---- 面板 2: 行归一化 ----
    im2 = ax2.imshow(cm_norm * 100, cmap='RdYlGn', aspect='equal', vmin=0, vmax=100)
    for i in range(n):
        for j in range(n):
            val_pct = cm_norm[i, j] * 100
            if val_pct > 0:
                color = 'white' if val_pct > 50 else 'black'
                ax2.text(j, i, f'{val_pct:.1f}%',
                         ha='center', va='center', fontsize=8, color=color,
                         fontweight='bold' if i == j else 'normal')
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels_cn, fontsize=7)
    ax2.set_yticklabels(labels_cn, fontsize=7)
    ax2.set_xlabel('预测类别', fontsize=10)
    ax2.set_ylabel('真实类别', fontsize=10)
    ax2.set_title('行归一化 [%]', fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='百分比 [%]')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Confusion Matrix] {save_path}")


def plot_per_class_metrics(df: pd.DataFrame, save_path: str):
    """各类别分类指标汇总图: 准确率+F1双轴柱状图 + 置信度子图。

    左面板: 各类别准确率(蓝色柱) + F1(橙色菱形折线), 双 y 轴
    右面板: 各类别平均 softmax 置信度 (绿柱), 单 y 轴
    """
    n = len(df)
    atks = df['attack_type'].tolist()
    x = np.arange(n)
    colors = [ATK_COLORS.get(a, '#888888') for a in atks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))
    fig.suptitle('CFMDetector 测试集各类别分类指标', fontsize=13, fontweight='bold')

    # ---- 面板 1: 准确率 + F1 ----
    bars = ax1.bar(x, df['accuracy'] * 100, color=colors, alpha=0.85,
                   edgecolor='black', lw=0.5, label='准确率 (Accuracy)')
    for bar, v in zip(bars, df['accuracy']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                 f'{v*100:.1f}%', ha='center', fontsize=7.5, fontweight='bold')

    ax1_f1 = ax1.twinx()
    ax1_f1.plot(x, df['f1_score'], 'D-', color='#E65100', lw=1.5, ms=6,
                label='F1 分数', zorder=5)
    for i, f1 in enumerate(df['f1_score']):
        ax1_f1.annotate(f'{f1:.3f}', (i, f1), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=6.5, color='#BF360C')

    ax1.set_xticks(x)
    labels_cn = [f'{a}\n{ATTACK_NAMES_CN[a]}' for a in atks]
    ax1.set_xticklabels(labels_cn, fontsize=7)
    ax1.set_ylabel('准确率 [%]', fontsize=10)
    ax1_f1.set_ylabel('F1 分数', fontsize=10)
    ax1_f1.set_ylim(0, 1.05)
    ax1.set_title('各类别准确率 & F1', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_f1.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)

    # ---- 面板 2: 平均置信度 ----
    conf_bars = ax2.bar(x, df['mean_confidence'], color=colors, alpha=0.85,
                        edgecolor='black', lw=0.5)
    for bar, v in zip(conf_bars, df['mean_confidence']):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{v:.3f}', ha='center', fontsize=7.5, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_cn, fontsize=7)
    ax2.set_ylabel('平均 softmax 置信度', fontsize=10)
    ax2.set_ylim(0, 1.1)
    ax2.set_title('各类别平均置信度', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Per-Class Metrics] {save_path}")


# ============================================================================
# 报告生成
# ============================================================================

def write_metrics_txt(output_dir: str, model_name: str, model_path: str,
                      n_samples: int, cm_counts: np.ndarray, cm_norm: np.ndarray,
                      df: pd.DataFrame, class_report: dict):
    """生成综合文本报告 metrics.txt。"""
    save_path = os.path.join(output_dir, 'metrics.txt')
    atks_order = ALL_ATTACK_TYPES

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("CFMDetector 测试集分类评估报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"权重: {model_path}\n")
        f.write(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"测试样本: {n_samples:,} 窗口\n\n")

        # 一、总体指标
        overall_acc = np.trace(cm_counts) / max(cm_counts.sum(), 1)
        f.write("-" * 70 + "\n")
        f.write("一、总体指标\n")
        f.write("-" * 70 + "\n")
        f.write(f"  总准确率:      {overall_acc*100:6.2f}%\n")
        if 'macro avg' in class_report:
            ma = class_report['macro avg']
            f.write(f"  宏观 F1:       {ma['f1-score']:7.4f}\n")
        if 'weighted avg' in class_report:
            wa = class_report['weighted avg']
            f.write(f"  加权 F1:       {wa['f1-score']:7.4f}\n")
        f.write(f"  总样本数:       {n_samples:,}\n\n")

        # 二、逐类别指标
        f.write("-" * 70 + "\n")
        f.write("二、逐类别分类指标\n")
        f.write("-" * 70 + "\n")
        header = (f"{'攻击':>6}  {'名称':<12}  {'准确率':>8}  {'精确率':>8}  "
                  f"{'召回率':>8}  {'F1':>8}  {'置信度':>8}  {'样本数':>8}")
        f.write(header + "\n")
        f.write("-" * 70 + "\n")
        for _, row in df.iterrows():
            f.write(f"{row['attack_type']:>6}  {row['name_cn']:<12}  "
                    f"{row['accuracy']*100:>7.1f}%  {row['precision']:>7.4f}  "
                    f"{row['recall']:>7.4f}  {row['f1_score']:>7.4f}  "
                    f"{row['mean_confidence']:>7.3f}  {int(row['n_samples']):>8,}\n")
        f.write("\n")

        # 三、混淆矩阵 (原始计数)
        f.write("-" * 70 + "\n")
        f.write("三、混淆矩阵 (原始计数)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'':>10}" + "".join(f"{a:>8}" for a in atks_order) + "\n")
        for i, a in enumerate(atks_order):
            f.write(f"{a:>10}" + "".join(f"{cm_counts[i, j]:>8}" for j in range(len(atks_order))) + "\n")
        f.write("\n")

        # 行归一化
        f.write("混淆矩阵 (行归一化 %)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'':>10}" + "".join(f"{a:>8}" for a in atks_order) + "\n")
        for i, a in enumerate(atks_order):
            f.write(f"{a:>10}" + "".join(f"{cm_norm[i, j]*100:>7.1f}%" for j in range(len(atks_order))) + "\n")
        f.write("\n")

        # 四、分类报告
        if class_report:
            f.write("-" * 70 + "\n")
            f.write("四、分类报告 (Precision/Recall/F1)\n")
            f.write("-" * 70 + "\n")
            for a in atks_order:
                key = f'{a} ({ATTACK_NAMES_CN[a]})'
                if key in class_report:
                    cr = class_report[key]
                    f.write(f"  {key:<25} P={cr['precision']:.4f}  R={cr['recall']:.4f}  "
                            f"F1={cr['f1-score']:.4f}  (n={cr['support']:.0f})\n")
            if 'macro avg' in class_report:
                ma = class_report['macro avg']
                f.write(f"\n  {'宏观平均 (macro avg)':<25} P={ma['precision']:.4f}  "
                        f"R={ma['recall']:.4f}  F1={ma['f1-score']:.4f}\n")
            if 'weighted avg' in class_report:
                wa = class_report['weighted avg']
                f.write(f"  {'加权平均 (weighted avg)':<25} P={wa['precision']:.4f}  "
                        f"R={wa['recall']:.4f}  F1={wa['f1-score']:.4f}\n")
            f.write("\n")

    print(f"  [Metrics TXT] {save_path}")


def export_metrics_csv(df: pd.DataFrame, output_dir: str):
    """导出逐类指标 CSV。"""
    save_path = os.path.join(output_dir, 'metrics.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"  [Metrics CSV] {save_path}")


def generate_markdown_report(output_dir: str, model_name: str, model_path: str,
                             n_samples: int, cm_counts: np.ndarray,
                             cm_norm: np.ndarray,
                             df: pd.DataFrame, class_report: dict):
    """生成 Markdown 格式分析报告。"""
    save_path = os.path.join(output_dir, 'analysis_report.md')
    overall_acc = np.trace(cm_counts) / max(cm_counts.sum(), 1)

    lines = []
    lines.append("# CFMDetector 测试集分类评估报告")
    lines.append("")
    lines.append(f"- **模型**: {model_name}")
    lines.append(f"- **权重**: `{model_path}`")
    lines.append(f"- **日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **测试样本**: {n_samples:,} 窗口")
    lines.append("")

    # 总体指标
    lines.append("## 总体指标")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 总准确率 | {overall_acc*100:.2f}% |")
    if 'macro avg' in class_report:
        lines.append(f"| 宏观 F1 | {class_report['macro avg']['f1-score']:.4f} |")
    if 'weighted avg' in class_report:
        lines.append(f"| 加权 F1 | {class_report['weighted avg']['f1-score']:.4f} |")
    lines.append(f"| 总样本数 | {n_samples:,} |")
    lines.append("")

    # 逐类别指标表
    lines.append("## 逐类别分类指标")
    lines.append("")
    lines.append("| 攻击 | 名称 | 准确率 | 精确率 | 召回率 | F1 | 置信度 | 样本数 |")
    lines.append("|------|------|--------|--------|--------|-----|--------|--------|")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['attack_type']} | {row['name_cn']} | "
            f"{row['accuracy']*100:.1f}% | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1_score']:.4f} | "
            f"{row['mean_confidence']:.3f} | {int(row['n_samples']):,} |"
        )
    lines.append("")

    # 混淆矩阵
    atks_order = ALL_ATTACK_TYPES
    lines.append("## 混淆矩阵 (行归一化 %)")
    lines.append("")
    header = "| 真实\\预测 | " + " | ".join(atks_order) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["------"] * (len(atks_order) + 1)) + "|")
    for i, a in enumerate(atks_order):
        row = f"| {a} | " + " | ".join(f"{cm_norm[i, j]*100:.1f}%" for j in range(len(atks_order))) + " |"
        lines.append(row)
    lines.append("")

    # 关键发现
    lines.append("## 关键发现")
    lines.append("")
    best_idx = df['f1_score'].idxmax()
    worst_idx = df['f1_score'].idxmin()
    lines.append(f"- **最佳分类**: {df.loc[best_idx, 'attack_type']} "
                 f"({df.loc[best_idx, 'name_cn']}), F1={df.loc[best_idx, 'f1_score']:.4f}")
    lines.append(f"- **最差分类**: {df.loc[worst_idx, 'attack_type']} "
                 f"({df.loc[worst_idx, 'name_cn']}), F1={df.loc[worst_idx, 'f1_score']:.4f}")
    lines.append("")

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  [Markdown Report] {save_path}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    from torch.utils.data import DataLoader
    from detector.train_cfm import PreprocessedDataset

    args = parse_args()

    # ---- 确定模型名称和输出目录 ----
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = args.output_dir or os.path.join(
        PROJECT_ROOT, 'eval', f'{model_name}_{timestamp}')

    # ---- 查找配置文件 ----
    if args.config_path:
        config_path = args.config_path
    else:
        model_dir = os.path.dirname(args.model_path)
        config_path = os.path.join(model_dir, 'cfm_cls_config.npz')

    # ---- 设备 ----
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print(f"CFMDetector 测试集分类评估")
    print(f"  模型: {args.model_path}")
    print(f"  配置: {config_path}")
    print(f"  数据: {args.data_dir}")
    print(f"  设备: {device}")
    print(f"  输出: {output_root}")
    print("=" * 60)

    # ---- 创建输出目录 ----
    build_eval_dirs(output_root)

    # ---- 加载模型 ----
    print("\n[1/5] 加载模型...")
    model = load_model(config_path, args.model_path, device)

    # ---- 加载测试集 ----
    print("\n[2/5] 加载测试集...")
    test_dataset = PreprocessedDataset(args.data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    print(f"  测试窗口: {len(test_dataset):,}")

    # ---- 推理 ----
    print(f"\n[3/5] 批量推理 (batch_size={args.batch_size})...")
    y_true, y_pred, y_conf = run_inference(model, test_loader, device)
    print(f"  推理完成: {len(y_true):,} 样本")

    # ---- 指标 ----
    print(f"\n[4/5] 计算指标...")
    cm_counts, cm_norm = build_confusion_matrix(y_true, y_pred)
    overall_acc = np.trace(cm_counts) / max(cm_counts.sum(), 1)
    print(f"  总准确率: {overall_acc*100:.2f}%")

    df = compute_per_class_metrics(y_true, y_pred, y_conf, ALL_ATTACK_TYPES)

    class_report = generate_classification_report(
        y_true, y_pred, ALL_ATTACK_TYPES,
        os.path.join(output_root, 'classification_report.txt'))

    # ---- 绘图 + 报告 ----
    print(f"\n[5/5] 生成图表和报告...")
    plot_confusion_matrix(cm_counts, cm_norm, ALL_ATTACK_TYPES,
                          os.path.join(output_root, 'confusion_matrix.png'))
    plot_per_class_metrics(df, os.path.join(output_root, 'per_class_metrics.png'))

    write_metrics_txt(output_root, model_name, args.model_path,
                      len(y_true), cm_counts, cm_norm, df, class_report)
    export_metrics_csv(df, output_root)
    generate_markdown_report(output_root, model_name, args.model_path,
                             len(y_true), cm_counts, cm_norm, df, class_report)

    # ---- 完成 ----
    print(f"\n{'='*60}")
    print(f"评估完成!")
    print(f"  总准确率: {overall_acc*100:.2f}%")
    print(f"  宏观 F1:  {class_report.get('macro avg', {}).get('f1-score', 'N/A')}")
    print(f"  输出目录: {output_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
