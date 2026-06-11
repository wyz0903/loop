"""
detector/test/analyze.py — CFMDetector DL 性能分析
====================================================
读取仿真结果 NPZ 文件，计算检测器 DL 指标并生成报告。

专注于:
  - 混淆矩阵构建
  - 分类准确率/置信度/延迟/虚警率
  - 攻击信号重建 MAE
  - 每类指标 CSV 导出

用法:
  python detector/test/analyze.py                    # 分析 results/ 中的 NPZ
  python detector/test/analyze.py --npz-dir eval/cfm_v1/npz/  # 指定目录
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# IEEE 论文标准字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

RESULT_DIR = os.path.join(PROJECT_ROOT, 'results')
ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
ATTACK_NAMES = {
    'A0': 'Normal', 'A1': 'Constant Bias', 'A2': 'Sinusoidal',
    'A3': 'Drift', 'A4': 'Step', 'A5': 'Replay Attack',
    'A6': 'Intermittent Dropout', 'A7': 'Scaling', 'A8': 'Sensor Freeze',
}


def find_sim_files(npz_dir: str = None):
    """扫描 NPZ 目录，按攻击类型组织文件"""
    if npz_dir is None:
        npz_dir = RESULT_DIR
    files = glob.glob(os.path.join(npz_dir, 'sim_*.npz'))
    organized = {}
    for path in files:
        basename = os.path.basename(path).replace('.npz', '')
        parts = basename.split('_')  # ['sim', 'A4', 'lissajous'] or similar
        for atk in ATTACK_TYPES:
            if atk in parts:
                organized[atk] = path
                break
    return organized


def _get_onset_idx(data: dict) -> int:
    """从仿真数据获取攻击起始索引（考虑窗口填充期）"""
    if 'attack_onset' in data:
        raw_onset = int(float(data['attack_onset']) / float(data.get('Ts', 0.05)))
        return max(raw_onset, 100)
    if 'attack_active' in data:
        active = data['attack_active']
        if hasattr(active, 'max') and active.max() > 0.5:
            raw_onset = int(np.argmax(active > 0.5))
            return max(raw_onset, 100)
    return max(int(15.0 / data.get('Ts', 0.05)), 100)


def compute_detector_metrics(data: dict, attack_type: str = None) -> dict:
    """计算检测器 DL 指标 (使用动态 onset, 考虑窗口填充期)"""
    if attack_type is None:
        attack_type = str(data.get('attack_type', 'A0'))

    onset_idx = _get_onset_idx(data)
    det_classes = data.get('det_class', np.array(['A0'] * len(data['t'])))
    det_confs = data.get('det_conf', np.zeros(len(data['t'])))

    post_classes = det_classes[onset_idx:]
    post_confs = det_confs[onset_idx:]

    if len(post_classes) == 0:
        return {'detection_accuracy': 0.0, 'mean_confidence': 0.0,
                'detection_latency_sec': 99.0, 'false_alarm_rate': 0.0,
                'recovery_rmse': 0.0}

    # 准确率
    correct = sum(1 for c in post_classes if str(c) == attack_type)
    accuracy = correct / len(post_classes)
    mean_conf = float(np.mean([float(c) for c in post_confs]))

    # 检测延迟
    latency_sec = len(post_classes) * data.get('Ts', 0.05)
    for i, c in enumerate(post_classes):
        if str(c) == attack_type and float(post_confs[i]) > 0.5:
            latency_sec = i * data.get('Ts', 0.05)
            break

    # 虚警率: 窗口填充后(100步)到攻击前
    pre_start = 100
    pre_classes = det_classes[pre_start:onset_idx]
    if len(pre_classes) > 0:
        false_alarms = sum(1 for c in pre_classes if str(c) != 'A0')
        far = false_alarms / len(pre_classes)
    else:
        far = 0.0

    # 恢复质量
    y_ekf_arr = data.get('y_ekf', np.zeros((1, 3)))
    true_arr = data.get('true_state', np.zeros((1, 3)))
    if len(y_ekf_arr) > onset_idx:
        recovery_rmse = float(np.sqrt(np.mean(
            np.sum((y_ekf_arr[onset_idx:] - true_arr[onset_idx:]) ** 2, axis=1))))
    else:
        recovery_rmse = 0.0

    return {
        'detection_accuracy': accuracy,
        'mean_confidence': mean_conf,
        'detection_latency_sec': latency_sec,
        'false_alarm_rate': far,
        'recovery_rmse': recovery_rmse,
    }


def compute_tracking_metrics(data: dict) -> dict:
    """从仿真数据计算跟踪指标"""
    onset_idx = _get_onset_idx(data)
    X_err = data['X_error']
    pos_err = np.sqrt(X_err[:, 0] ** 2 + X_err[:, 1] ** 2)
    ang_err = np.abs(X_err[:, 2])

    post_pos = pos_err[onset_idx:]

    return {
        'post_pos_rmse': float(np.sqrt(np.mean(post_pos ** 2))),
        'post_pos_max': float(np.max(post_pos)),
        'post_ang_rmse': float(np.sqrt(np.mean(ang_err[onset_idx:] ** 2))),
        'tracking_lost': float(np.max(post_pos)) > 0.5,
    }


def build_confusion_matrix_from_sim(npz_dir: str = None):
    """从仿真 NPZ 构建逐步混淆矩阵"""
    files_by_atk = find_sim_files(npz_dir)
    cm = np.zeros((9, 9), dtype=int)
    total_per_class = np.zeros(9, dtype=int)

    for atk_label in ATTACK_TYPES:
        atk_idx = int(atk_label[1])
        if atk_label not in files_by_atk:
            continue
        data = dict(np.load(files_by_atk[atk_label], allow_pickle=True))
        onset_idx = _get_onset_idx(data)
        det_classes = data.get('det_class', np.array([]))
        post_classes = det_classes[onset_idx:]
        total_per_class[atk_idx] = len(post_classes)
        for c in post_classes:
            c_str = str(c)
            if c_str in ATTACK_TYPES:
                cm[atk_idx, int(c_str[1])] += 1

    return cm, total_per_class


def plot_confusion_matrix(cm: np.ndarray, total_per_class: np.ndarray, output_dir: str):
    """绘制混淆矩阵"""
    labels = [f'{a}\n{ATTACK_NAMES[a]}' for a in ATTACK_TYPES]
    acc_per_class = np.zeros(9)
    for i in range(9):
        acc_per_class[i] = cm[i, i] / max(total_per_class[i], 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('CFMDetector Confusion Matrix', fontsize=14, fontweight='bold')

    # 左: 原始计数
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(9):
        for j in range(9):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=9, fontweight='bold' if i == j else 'normal',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels(labels, fontsize=7); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Counts')
    plt.colorbar(im, ax=ax)

    # 右: 行归一化百分比
    ax = axes[1]
    cm_pct = np.zeros_like(cm, dtype=float)
    for i in range(9):
        cm_pct[i] = cm[i] / max(total_per_class[i], 1) * 100
    im2 = ax.imshow(cm_pct, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f'{cm_pct[i, j]:.1f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels(labels, fontsize=7); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Row-Normalized [%]')
    plt.colorbar(im2, ax=ax)

    filepath = os.path.join(output_dir, 'confusion_matrix.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot] {filepath}")
    return acc_per_class


def build_metrics_table(npz_dir: str = None) -> pd.DataFrame:
    """构建 CFM 检测器指标表"""
    files_by_atk = find_sim_files(npz_dir)
    rows = []

    for atk in ATTACK_TYPES:
        if atk not in files_by_atk:
            continue
        data = dict(np.load(files_by_atk[atk], allow_pickle=True))
        m = compute_tracking_metrics(data)
        dm = compute_detector_metrics(data, attack_type=atk)

        row = {
            'Attack': atk,
            'Name': ATTACK_NAMES.get(atk, ''),
            'Post_RMSE_m': round(m['post_pos_rmse'], 4),
            'Post_Max_m': round(m['post_pos_max'], 4),
            'Detection_Acc_%': round(dm['detection_accuracy'] * 100, 1),
            'Mean_Confidence': round(dm['mean_confidence'], 3),
            'Latency_s': round(dm['detection_latency_sec'], 2),
            'Recovery_RMSE_m': round(dm['recovery_rmse'], 4),
            'False_Alarm_%': round(dm['false_alarm_rate'] * 100, 1),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def generate_summary_plot(df: pd.DataFrame, output_dir: str):
    """生成 CFM 检测器性能汇总图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CFMDetector Performance Summary', fontsize=14, fontweight='bold')

    attacks = df['Attack'].values
    x = np.arange(len(attacks))

    # 左: 检测准确率 + 跟踪 RMSE
    ax = axes[0]
    colors_acc = ['#2ca02c' if v >= 70 else '#ff7f0e' if v >= 40 else '#d62728'
                  for v in df['Detection_Acc_%'].values]
    bars = ax.bar(x - 0.2, df['Detection_Acc_%'].values, 0.35,
                  color=colors_acc, alpha=0.85, label='Detection Acc [%]')
    ax2 = ax.twinx()
    ax2.plot(x + 0.2, df['Post_RMSE_m'].values, 'bs-', markersize=8, linewidth=1.5,
             label='Post-Attack RMSE [m]')
    for i, (acc, rmse) in enumerate(zip(df['Detection_Acc_%'].values, df['Post_RMSE_m'].values)):
        ax.text(i - 0.2, acc + 1, f'{acc:.0f}%', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(attacks)
    ax.set_ylabel('Detection Accuracy [%]'); ax.set_ylim(0, 105)
    ax2.set_ylabel('Position RMSE [m]')
    ax.set_title('Detection Accuracy & Tracking RMSE')
    ax.grid(True, alpha=0.3, axis='y')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # 右: 检测延迟 + 虚警率
    ax = axes[1]
    bars1 = ax.bar(x - 0.15, df['Latency_s'].values, 0.3,
                   color='steelblue', alpha=0.8, label='Detection Latency [s]')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + 0.15, df['False_Alarm_%'].values, 0.3,
                    color='#d62728', alpha=0.6, label='False Alarm Rate [%]')
    for i, (lat, far) in enumerate(zip(df['Latency_s'].values, df['False_Alarm_%'].values)):
        if lat > 0.01:
            ax.text(i - 0.15, lat + 0.02, f'{lat:.2f}', ha='center', fontsize=7)
        if far > 0.01:
            ax2.text(i + 0.15, far + 0.1, f'{far:.1f}%', ha='center', fontsize=7, color='darkred')
    ax.set_xticks(x); ax.set_xticklabels(attacks)
    ax.set_ylabel('Latency [s]')
    ax2.set_ylabel('False Alarm Rate [%]')
    ax.set_title('Detection Latency & False Alarm Rate')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'detector_analysis.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot] {filepath}")
    return filepath


def generate_report(df: pd.DataFrame, output_dir: str):
    """生成 Markdown 分析报告"""
    report_path = os.path.join(output_dir, 'detector_analysis_report.md')

    lines = []
    lines.append("# CFMDetector Performance Analysis Report")
    lines.append("")
    lines.append(f"> Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("> Detector: CFMDetector — PINN-Flow Conditional Flow Matching + Transformer backbone")
    lines.append("")

    if len(df) > 0:
        mean_acc = df['Detection_Acc_%'].mean()
        mean_rmse = df['Post_RMSE_m'].mean()
        mean_lat = df['Latency_s'].mean()
        lines.append("## Overall Summary")
        lines.append("")
        lines.append(f"- **Mean detection accuracy**: {mean_acc:.1f}%")
        lines.append(f"- **Mean post-attack RMSE**: {mean_rmse:.4f} m")
        lines.append(f"- **Mean detection latency**: {mean_lat:.2f} s")
        lines.append("")

    lines.append("## Per-Attack Metrics")
    lines.append("")
    header = "| Attack | Name | RMSE [m] | Acc [%] | Conf | Latency [s] | FAR [%] | Recovery RMSE |"
    sep = "|--------|------|----------|---------|------|-------------|---------|---------------|"
    lines.append(header)
    lines.append(sep)
    for _, row in df.iterrows():
        lines.append(f"| {row['Attack']} | {row['Name']} | {row['Post_RMSE_m']:.4f} | "
                     f"{row['Detection_Acc_%']:.1f} | {row['Mean_Confidence']:.3f} | "
                     f"{row['Latency_s']:.2f} | {row['False_Alarm_%']:.1f} | "
                     f"{row['Recovery_RMSE_m']:.4f} |")
    lines.append("")

    report_text = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  [Report] {report_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser(description='CFMDetector DL Performance Analysis')
    parser.add_argument('--npz-dir', type=str, default=None,
                        help='NPZ file directory (default: results/)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots and reports')
    args = parser.parse_args()

    npz_dir = args.npz_dir or RESULT_DIR
    output_dir = args.output_dir or npz_dir

    print("=" * 60)
    print("CFMDetector Performance Analysis")
    print("=" * 60)

    # 扫描数据
    print(f"\n[1] Scanning simulation data in {npz_dir}...")
    files_by_atk = find_sim_files(npz_dir)
    if not files_by_atk:
        print("  [ERROR] No simulation NPZ files found.")
        print("  Run 'python simulate.py --all' first.")
        sys.exit(1)

    for atk in ATTACK_TYPES:
        status = '✓' if atk in files_by_atk else 'MISSING'
        print(f"  {atk}: {status}")
    print(f"  Total: {len(files_by_atk)}/{len(ATTACK_TYPES)} attack types")

    # 混淆矩阵
    print("\n[2] Building confusion matrix...")
    cm, total_per_class = build_confusion_matrix_from_sim(npz_dir)
    acc_per_class = plot_confusion_matrix(cm, total_per_class, output_dir)
    for i, (atk, acc) in enumerate(zip(ATTACK_TYPES, acc_per_class)):
        print(f"  {atk} ({ATTACK_NAMES[atk]}): {acc:.1f}% ({total_per_class[i]} samples)")

    # 指标表
    print("\n[3] Computing metrics...")
    df = build_metrics_table(npz_dir)

    # 保存 CSV
    csv_path = os.path.join(output_dir, 'detector_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"  [CSV] {csv_path}")

    # 汇总图
    print("\n[4] Generating summary plot...")
    generate_summary_plot(df, output_dir)

    # 报告
    print("\n[5] Generating report...")
    generate_report(df, output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
