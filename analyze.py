"""
analyze.py — 检测器性能分析与报告生成
============================================================================
读取 simulate_detector.py 批量仿真结果，生成：
  1. 定量指标对比表 (CSV + Markdown + LaTeX)
  2. 汇总可视化图
  3. 研究分析报告

用法：
  python analyze.py                    # 分析已有结果，生成报告
  python analyze.py --export-latex     # 额外导出 LaTeX 表格
"""

import os
import sys
import glob
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

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
ATTACK_NAMES = {
    'A0': 'Normal', 'A1': 'Constant Bias', 'A2': 'Sinusoidal',
    'A3': 'Drift', 'A4': 'Step', 'A5': 'Replay Attack',
    'A6': 'Intermittent Dropout', 'A7': 'Scaling', 'A8': 'Sensor Freeze',
}

TIER_COLORS = {'none': '#d62728', 'nn': '#2ca02c', 'oracle': '#1f77b4'}
TIER_LABELS = {
    'none': 'Tier 0: No Detector',
    'cfm': 'Tier 1: CFMDetector (PINN-Flow)',
    'oracle': 'Tier 2: Oracle (Upper Bound)',
}
TIER_LINESTYLE = {'none': '--', 'nn': '-', 'oracle': '-.'}


def find_sim_files():
    """扫描 results/ 目录，按攻击类型和 tier 组织文件"""
    files = glob.glob(os.path.join(RESULT_DIR, 'sim_det_*.npz'))
    organized = defaultdict(dict)  # {atk: {tier: path}}

    for path in files:
        basename = os.path.basename(path).replace('.npz', '')
        parts = basename.split('_')  # ['sim', 'det', 'A4', 'nn', 'lissajous']
        if len(parts) >= 4:
            atk = parts[2]
            tier = parts[3]
            if atk in ATTACK_TYPES and tier in ('none', 'nn', 'oracle'):
                organized[atk][tier] = path

    return dict(organized)


def _get_onset_idx(data: dict) -> int:
    """从仿真数据获取攻击起始索引（考虑窗口填充期）"""
    # 优先用 attack_onset 标量
    if 'attack_onset' in data:
        raw_onset = int(float(data['attack_onset']) / float(data.get('Ts', 0.05)))
        return max(raw_onset, 100)  # 窗口填充期: 前 100 步 NN 未激活
    # Fallback: attack_active 数组
    if 'attack_active' in data:
        active = data['attack_active']
        if hasattr(active, 'max') and active.max() > 0.5:
            raw_onset = int(np.argmax(active > 0.5))
            return max(raw_onset, 100)
    # 最后默认
    return max(int(15.0 / data.get('Ts', 0.05)), 100)


def compute_tracking_metrics(data: dict) -> dict:
    """从仿真数据计算跟踪指标 (使用动态 onset)"""
    onset_idx = _get_onset_idx(data)

    X_err = data['X_error']
    pos_err = np.sqrt(X_err[:, 0]**2 + X_err[:, 1]**2)
    ang_err = np.abs(X_err[:, 2])

    # 稳态: 窗口填充(100步)后 到 攻击前
    steady_start = max(int(5.0 / data.get('Ts', 0.05)), 100)
    steady_end = onset_idx

    pre_pos = pos_err[steady_start:steady_end]
    post_pos = pos_err[onset_idx:]

    return {
        'steady_pos_rmse': float(np.sqrt(np.mean(pre_pos**2))) if len(pre_pos) > 0 else 0.0,
        'post_pos_rmse': float(np.sqrt(np.mean(post_pos**2))),
        'post_pos_max': float(np.max(post_pos)),
        'post_ang_rmse': float(np.sqrt(np.mean(ang_err[onset_idx:]**2))),
        'tracking_lost': float(np.max(post_pos)) > 0.5,
    }


def compute_detector_metrics(data: dict, attack_type: str = None) -> dict:
    """计算检测器专用指标 (使用动态 onset, 考虑窗口填充期)

    Args:
        data:        仿真数据字典 (从 npz 加载)
        attack_type: 攻击类型标签
    """
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
            np.sum((y_ekf_arr[onset_idx:] - true_arr[onset_idx:])**2, axis=1))))
    else:
        recovery_rmse = 0.0

    return {
        'detection_accuracy': accuracy,
        'mean_confidence': mean_conf,
        'detection_latency_sec': latency_sec,
        'false_alarm_rate': far,
        'recovery_rmse': recovery_rmse,
    }


def build_metrics_table() -> pd.DataFrame:
    """构建完整的指标对比表（按 tier 分组）"""
    files_by_atk = find_sim_files()
    rows = []

    for atk in ATTACK_TYPES:
        if atk not in files_by_atk:
            continue

        tier_data = files_by_atk[atk]
        none_rmse = None

        for tier in ['none', 'nn', 'oracle']:
            if tier not in tier_data:
                continue

            data = dict(np.load(tier_data[tier], allow_pickle=True))
            m = compute_tracking_metrics(data)

            # 改善百分比 (相对于 none)
            if tier == 'none':
                none_rmse = m['post_pos_rmse']
                imp = 0.0
            elif none_rmse is not None and none_rmse > 1e-6:
                imp = (none_rmse - m['post_pos_rmse']) / none_rmse * 100
            else:
                imp = 0.0

            row = {
                'Attack': atk,
                'Name': ATTACK_NAMES.get(atk, ''),
                'Tier': tier,
                'Post_RMSE_m': round(m['post_pos_rmse'], 4),
                'Post_Max_m': round(m['post_pos_max'], 4),
                'Improvement_%': round(imp, 1),
            }

            # 检测器指标（仅 nn/oracle）
            if tier in ('nn', 'oracle'):
                dm = compute_detector_metrics(data, attack_type=atk)
                row.update({
                    'Detection_Acc_%': round(dm['detection_accuracy'] * 100, 1),
                    'Mean_Confidence': round(dm['mean_confidence'], 3),
                    'Latency_s': round(dm['detection_latency_sec'], 2),
                    'Recovery_RMSE_m': round(dm['recovery_rmse'], 4),
                    'False_Alarm_%': round(dm['false_alarm_rate'] * 100, 1),
                })

            rows.append(row)

    return pd.DataFrame(rows)


def generate_plots(df: pd.DataFrame):
    """生成汇总分析图"""
    attacks = sorted(df['Attack'].unique())
    tiers_in_data = ['none', 'nn', 'oracle']
    tiers_in_data = [t for t in tiers_in_data if t in df['Tier'].values]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('CFMDetector Performance — Flow Matching Recovery',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(attacks))

    # Panel 1: RMSE 柱状图（按 tier 分组）
    ax = axes[0, 0]
    n_tiers = len(tiers_in_data)
    w = 0.8 / n_tiers
    for i, tier in enumerate(tiers_in_data):
        rmse_vals = []
        for atk in attacks:
            row = df[(df['Attack'] == atk) & (df['Tier'] == tier)]
            rmse_vals.append(row['Post_RMSE_m'].values[0] if len(row) else 0)
        offset = (i - (n_tiers - 1) / 2) * w
        bars = ax.bar(x + offset, rmse_vals, w,
                      color=TIER_COLORS[tier], alpha=0.85,
                      label=TIER_LABELS[tier])
        for bar, val in zip(bars, rmse_vals):
            if val > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f'{val:.3f}', ha='center', fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylabel('Position RMSE [m]')
    ax.set_title('Post-Attack Position RMSE by Attack Type')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: 改善百分比（nn vs oracle）
    ax = axes[1, 0]
    compare_tiers = [t for t in ['nn', 'oracle'] if t in tiers_in_data]
    w2 = 0.35 / len(compare_tiers) if compare_tiers else 0.35
    for i, tier in enumerate(compare_tiers):
        imp_vals = []
        for atk in attacks:
            row = df[(df['Attack'] == atk) & (df['Tier'] == tier)]
            imp_vals.append(row['Improvement_%'].values[0] if len(row) else 0)
        offset = (i - (len(compare_tiers) - 1) / 2) * w2
        colors = ['#2ca02c' if v >= 0 else '#d62728' for v in imp_vals]
        bars = ax.bar(x + offset, imp_vals, w2, color=colors, alpha=0.8,
                      label=TIER_LABELS[tier])
        for bar, val in zip(bars, imp_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1 if val >= 0 else bar.get_height() - 4,
                    f'{val:+.1f}%', ha='center', fontsize=7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylabel('RMSE Improvement vs None [%]')
    ax.set_title('Tracking Improvement by Detector Tier')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: 检测准确率 vs 改善（仅 nn）
    ax = axes[0, 1]
    nn_data = df[df['Tier'] == 'nn']
    if 'Detection_Acc_%' in nn_data.columns and len(nn_data) > 0:
        ax.scatter(nn_data['Detection_Acc_%'], nn_data['Improvement_%'],
                   c='steelblue', s=100, alpha=0.7, edgecolors='black')
        for _, row in nn_data.iterrows():
            ax.annotate(row['Attack'],
                        (row['Detection_Acc_%'], row['Improvement_%']),
                        fontsize=8, xytext=(5, 5), textcoords='offset points')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Detection Accuracy [%]')
        ax.set_ylabel('RMSE Improvement [%]')
        ax.set_title('NN Detection Accuracy vs Tracking Improvement')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No NN detection data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    # Panel 4: RMSE 对比表
    ax = axes[1, 1]
    ax.axis('off')
    nn_data = df[df['Tier'] == 'nn']
    if len(nn_data) > 0:
        table_data = []
        table_cols = ['Atk', 'Name', 'None', 'NN', 'Oracle', 'NN_Imp']
        for atk in attacks:
            r_none = df[(df['Attack'] == atk) & (df['Tier'] == 'none')]
            r_nn = df[(df['Attack'] == atk) & (df['Tier'] == 'nn')]
            r_oracle = df[(df['Attack'] == atk) & (df['Tier'] == 'oracle')]
            none_v = f"{r_none['Post_RMSE_m'].values[0]:.4f}" if len(r_none) else '-'
            nn_v = f"{r_nn['Post_RMSE_m'].values[0]:.4f}" if len(r_nn) else '-'
            oracle_v = f"{r_oracle['Post_RMSE_m'].values[0]:.4f}" if len(r_oracle) else '-'
            imp_v = f"{r_nn['Improvement_%'].values[0]:+.1f}%" if len(r_nn) else '-'
            table_data.append([atk, ATTACK_NAMES.get(atk, '')[:12], none_v, nn_v, oracle_v, imp_v])

        table = ax.table(cellText=table_data, colLabels=table_cols,
                         cellLoc='center', loc='center',
                         colWidths=[0.08, 0.22, 0.18, 0.18, 0.18, 0.16])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.3)
        ax.set_title('Post-Attack Position RMSE Summary', fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(RESULT_DIR, 'det_analysis_summary.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot] Saved: {filepath}")
    return filepath


def export_latex_table(df: pd.DataFrame):
    """导出 LaTeX 格式表格 (论文用)"""
    latex_path = os.path.join(RESULT_DIR, 'det_metrics.tex')

    # 为每个攻击类型取 nn 行 + none 行做对比
    tex_rows = []
    for atk in ATTACK_TYPES:
        r_none = df[(df['Attack'] == atk) & (df['Tier'] == 'none')]
        r_nn = df[(df['Attack'] == atk) & (df['Tier'] == 'nn')]
        if len(r_none) and len(r_nn):
            tex_rows.append({
                'Attack': atk,
                'Name': ATTACK_NAMES.get(atk, ''),
                'RMSE_None': r_none['Post_RMSE_m'].values[0],
                'RMSE_NN': r_nn['Post_RMSE_m'].values[0],
                'Improvement': r_nn['Improvement_%'].values[0],
                'Det_Acc': r_nn['Detection_Acc_%'].values[0] if 'Detection_Acc_%' in r_nn else 0,
                'Latency': r_nn['Latency_s'].values[0] if 'Latency_s' in r_nn else 0,
            })

    tex_df = pd.DataFrame(tex_rows)
    latex_str = tex_df.to_latex(index=False, float_format="%.3f",
                                caption='CFMDetector Performance Summary',
                                label='tab:cfmdetector_performance')

    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    print(f"  [LaTeX] Saved: {latex_path}")


def generate_report(df: pd.DataFrame):
    """生成 Markdown 研究分析报告"""
    report_path = os.path.join(RESULT_DIR, 'detector_analysis_report.md')

    nn_data = df[df['Tier'] == 'nn']
    none_data = df[df['Tier'] == 'none']
    oracle_data = df[df['Tier'] == 'oracle']

    lines = []
    lines.append("# CFMDetector Performance Analysis Report")
    lines.append("")
    lines.append("> Generated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
    lines.append("> Detector: CFMDetector — PINN-Flow Conditional Flow Matching + Transformer backbone")
    lines.append("> Strategy: additive attacks via subtraction recovery, A5 replay -> internal kinematics dead reckoning")
    lines.append("")

    # 总体统计
    if len(nn_data) > 0:
        mean_imp = nn_data['Improvement_%'].mean()
        max_imp = nn_data['Improvement_%'].max()
        max_imp_row = nn_data.loc[nn_data['Improvement_%'].idxmax()]
        positive = (nn_data['Improvement_%'] > 0).sum()

        lines.append("## 1. Overall Summary")
        lines.append("")
        lines.append(f"- **Mean tracking improvement**: {mean_imp:+.1f}%")
        lines.append(f"- **Max improvement**: {max_imp:+.1f}% ({max_imp_row['Attack']})")
        lines.append(f"- **Positive improvement**: {positive}/{len(nn_data)} attacks")
        if 'Detection_Acc_%' in nn_data.columns:
            mean_acc = nn_data['Detection_Acc_%'].mean()
            lines.append(f"- **Mean detection accuracy**: {mean_acc:.1f}%")
        lines.append("")

    # 指标表
    lines.append("## 2. Quantitative Metrics")
    lines.append("")
    header = "| Attack | Name | None RMSE | NN RMSE | Oracle RMSE | NN Improv. | Det. Acc. | Latency |"
    sep = "|--------|------|-----------|---------|-------------|------------|-----------|---------|"
    lines.append(header)
    lines.append(sep)
    for atk in ATTACK_TYPES:
        r_none = none_data[none_data['Attack'] == atk]
        r_nn = nn_data[nn_data['Attack'] == atk]
        r_oracle = oracle_data[oracle_data['Attack'] == atk]
        none_v = f"{r_none['Post_RMSE_m'].values[0]:.4f}" if len(r_none) else '-'
        nn_v = f"{r_nn['Post_RMSE_m'].values[0]:.4f}" if len(r_nn) else '-'
        oracle_v = f"{r_oracle['Post_RMSE_m'].values[0]:.4f}" if len(r_oracle) else '-'
        imp_v = f"{r_nn['Improvement_%'].values[0]:+.1f}%" if len(r_nn) else '-'
        acc_v = f"{r_nn['Detection_Acc_%'].values[0]:.1f}%" if len(r_nn) and 'Detection_Acc_%' in r_nn else '-'
        lat_v = f"{r_nn['Latency_s'].values[0]:.2f}s" if len(r_nn) and 'Latency_s' in r_nn else '-'
        lines.append(f"| {atk} | {ATTACK_NAMES.get(atk, '')} | {none_v} | {nn_v} | {oracle_v} | {imp_v} | {acc_v} | {lat_v} |")
    lines.append("")

    # 分析
    lines.append("## 3. Analysis and Discussion")
    lines.append("")
    lines.append("### 3.1 Distribution Alignment")
    lines.append("")
    lines.append("Training uses **internal kinematics innovation** (identical to deployment computation),")
    lines.append("eliminating the distribution shift between EKF innovation (training) and internal kinematics innovation (deployment).")
    lines.append("Validation accuracy during training should directly reflect deployment detection performance.")
    lines.append("")
    lines.append("### 3.2 Unified Recovery Strategy")
    lines.append("")
    lines.append("Additive attacks (A0-A4, A6-A8) use unified `y_rec = y_meas - a(k)` recovery:")
    lines.append("- NN decoder reconstructs attack waveform a(k) from latent representation")
    lines.append("- Recovery quality depends on decoder reconstruction accuracy, not classification labels")
    lines.append("- A0 normal case: a ~ 0, automatically degenerates to pass-through")
    lines.append("")
    lines.append("A5 replay attack is the exception: non-additive attack, subtraction is meaningless; switches to internal kinematics dead reckoning.")
    lines.append("")

    # NN 与 Oracle 差距分析
    if len(nn_data) > 0 and len(oracle_data) > 0:
        lines.append("### 3.3 NN vs Oracle Gap")
        lines.append("")
        for atk in ATTACK_TYPES:
            r_nn = nn_data[nn_data['Attack'] == atk]
            r_oracle = oracle_data[oracle_data['Attack'] == atk]
            if len(r_nn) and len(r_oracle):
                nn_rmse = r_nn['Post_RMSE_m'].values[0]
                oracle_rmse = r_oracle['Post_RMSE_m'].values[0]
                gap = nn_rmse - oracle_rmse
                lines.append(f"- **{atk}** ({ATTACK_NAMES.get(atk, '')}): "
                           f"NN={nn_rmse:.4f}, Oracle={oracle_rmse:.4f}, "
                           f"Gap={gap:.4f}m")
        lines.append("")

    lines.append("### 3.4 Improvement Directions")
    lines.append("")
    lines.append("1. **Decoder accuracy**: reduce the RMSE gap between NN and Oracle")
    lines.append("2. **A5 dead reckoning optimization**: periodically reset internal kinematics state with EKF estimate to control drift")
    lines.append("3. **Multi-window voting**: leverage classification results from consecutive windows to reduce single-window misclassification")
    lines.append("")

    report_text = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  [Report] Saved: {report_path}")
    return report_text


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Detector Performance Analysis')
    parser.add_argument('--export-latex', action='store_true',
                        help='Export LaTeX table')
    args = parser.parse_args()

    print("=" * 60)
    print("CFMDetector Performance Analysis")
    print("=" * 60)

    # 检查数据
    print("\n[1] Scanning simulation data...")
    files_by_atk = find_sim_files()
    if not files_by_atk:
        print("  [ERROR] No simulation data found in results/")
        print("  Run 'python simulate_detector.py --all' first.")
        sys.exit(1)

    for atk in ATTACK_TYPES:
        if atk in files_by_atk:
            tiers = list(files_by_atk[atk].keys())
            print(f"  {atk}: {tiers}")
        else:
            print(f"  {atk}: MISSING")
    print(f"  Total: {len(files_by_atk)}/{len(ATTACK_TYPES)} attack types")

    # 构建指标表
    print("\n[2] Computing metrics...")
    df = build_metrics_table()

    # 打印摘要
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    # 按 tier 分组显示
    for tier in ['none', 'nn', 'oracle']:
        tier_df = df[df['Tier'] == tier]
        if len(tier_df) == 0:
            continue
        print(f"\n  {TIER_LABELS.get(tier, tier)}:")
        print(f"  {'Atk':<6} {'RMSE':<10} {'vs None':<10}")
        for _, row in tier_df.iterrows():
            imp_str = f"{row['Improvement_%']:+.1f}%" if row['Improvement_%'] != 0 else 'baseline'
            print(f"  {row['Attack']:<6} {row['Post_RMSE_m']:<10.4f} {imp_str:<10}")

    # 保存 CSV
    csv_path = os.path.join(RESULT_DIR, 'det_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  [CSV] Saved: {csv_path}")

    # 生成图
    print("\n[3] Generating plots...")
    generate_plots(df)

    # 生成报告
    print("\n[4] Generating report...")
    generate_report(df)

    # LaTeX
    if args.export_latex:
        export_latex_table(df)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
