"""
evaluate.py — 模型综合评估流水线
==================================
为训练好的检测器模型生成完整的评估报告, 输出到版本化的子文件夹中。

输出结构:
  eval/{model_name}/
  ├── metrics.txt              # 综合文本报告
  ├── metrics.csv              # 逐攻击类型指标 CSV
  ├── classification_report.txt # Precision/Recall/F1
  ├── confusion_matrix.png      # 9×9 混淆矩阵
  ├── reconstruction/
  │   ├── recon_A0.png ... recon_A8.png
  │   └── recon_summary.png
  ├── trajectory/
  │   ├── tracking_A0.png ... tracking_A8.png
  │   └── tracking_summary.png
  └── npz/                     # 仿真 NPZ (simulate 模式)

模式:
  simulate   — 运行闭环仿真 + 生成全部产物 (默认)
  from_npz   — 读取已有 NPZ + 生成全部产物
  validation — 在验证集上推理 + 分类报告 + 混淆矩阵

用法:
  python evaluate.py --model-path models/cfm_cls_best.pt
                     --output-dir eval/cfm_v1
                     --model-name cfm_v1

  python evaluate.py --mode from_npz --npz-dir results/
                     --output-dir eval/cfm_v1 --model-name cfm_v1

  python evaluate.py --mode validation --model-path models/cfm_cls_best.pt
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# IEEE 论文绘图样式 (支持中英文混排)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))  # 项目根目录 (导入 simulate, model 等)
sys.path.insert(0, SCRIPT_DIR)  # detector/ 目录

# 攻击类型定义 (与项目一致)
ALL_ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
ATTACK_NAMES_CN = {
    'A0': '正常', 'A1': '恒定偏移', 'A2': '正弦注入',
    'A3': '斜坡漂移', 'A4': '阶跃', 'A5': '重放攻击',
    'A6': '信号丢失', 'A7': '缩放攻击', 'A8': '传感器冻结',
}
ATK_COLORS = {
    'A0': '#4CAF50', 'A1': '#E91E63', 'A2': '#FF9800', 'A3': '#2196F3',
    'A4': '#F44336', 'A5': '#9C27B0', 'A6': '#795548', 'A7': '#00BCD4', 'A8': '#607D8B',
}

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='模型综合评估')
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型权重路径')
    parser.add_argument('--norm-path', type=str,
                        default=os.path.join(SCRIPT_DIR, '..', 'dataset_win', 'normalizer.npz'),
                        help='归一化参数路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出根目录 (默认: eval/{model_name}/)')
    parser.add_argument('--model-name', type=str, default='cfm_v1',
                        help='模型版本名 (用于子文件夹命名)')
    parser.add_argument('--mode', type=str, default='simulate',
                        choices=['simulate', 'from_npz', 'validation'],
                        help='评估模式')
    parser.add_argument('--npz-dir', type=str,
                        default=os.path.join(SCRIPT_DIR, '..', 'results'),
                        help='NPZ 文件目录 (from_npz 模式)')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(SCRIPT_DIR, '..', 'dataset_win'),
                        help='预处理数据目录 (validation 模式)')
    parser.add_argument('--attack-types', type=str, default=None,
                        help='攻击类型列表, 逗号分隔 (默认: 全部9种)')
    parser.add_argument('--trajectory', type=str, default='circular',
                        help='仿真轨迹类型 (默认 circular)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no-simulate', action='store_true',
                        help='跳过仿真, 仅使用已有 NPZ')
    return parser.parse_args()


# ============================================================================
# 目录创建
# ============================================================================

def build_eval_dirs(output_root: str) -> dict:
    """创建评估输出目录结构, 返回子目录路径字典."""
    dirs = {
        'root': output_root,
        'recon': os.path.join(output_root, 'reconstruction'),
        'tracking': os.path.join(output_root, 'trajectory'),
        'npz': os.path.join(output_root, 'npz'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ============================================================================
# 模式 1: 闭环仿真
# ============================================================================

def run_simulations(model_path: str, norm_path: str, npz_dir: str,
                    attack_types: list, trajectory: str, seed: int):
    """运行闭环仿真, 保存 NPZ 到 npz_dir。"""
    import simulate
    os.makedirs(npz_dir, exist_ok=True)

    print(f"\n运行闭环仿真: {len(attack_types)} 种攻击 × CFM")
    print(f"  轨迹: {trajectory}, seed: {seed}")
    print(f"  输出: {npz_dir}")

    for atk_idx, atk in enumerate(attack_types):
        print(f"  [{atk_idx+1}/{len(attack_types)}] {atk} ({ATTACK_NAMES_CN[atk]}) ...", end=' ', flush=True)
        try:
            data = simulate.run_simulation(
                attack_type=atk,
                trajectory_type=trajectory,
                use_detector=True,
                seed=seed,
                model_path=model_path,
                norm_path=norm_path,
            )
            # 保存 NPZ
            fname = f'sim_{atk}_{trajectory}_cfm.npz'
            save_dict = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    save_dict[k] = v
                elif isinstance(v, (str, int, float)):
                    save_dict[k] = np.array(v)
            np.savez_compressed(os.path.join(npz_dir, fname), **save_dict)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("仿真完成\n")


# ============================================================================
# NPZ 加载
# ============================================================================

def load_npz_files(npz_dir: str) -> dict:
    """扫描 NPZ 文件, 按攻击类型组织。Returns: {attack_type: [npz_data_dict, ...]}"""
    data = defaultdict(list)
    if not os.path.isdir(npz_dir):
        print(f"[WARN] NPZ 目录不存在: {npz_dir}")
        return data

    for fname in sorted(os.listdir(npz_dir)):
        if not fname.endswith('.npz'):
            continue
        # 文件名格式: sim_{ATK}_{TRAJ}_cfm.npz
        parts = fname.replace('.npz', '').split('_')
        if len(parts) < 3:
            continue
        file_atk = parts[1]
        if file_atk not in ALL_ATTACK_TYPES:
            continue
        fpath = os.path.join(npz_dir, fname)
        try:
            d = dict(np.load(fpath, allow_pickle=True))
            data[file_atk].append(d)
        except Exception as e:
            print(f"[WARN] 加载 {fname} 失败: {e}")

    print(f"加载 {sum(len(v) for v in data.values())} 个 NPZ ({len(data)} 种攻击)")
    return data


# ============================================================================
# 从仿真数据构建混淆矩阵
# ============================================================================

def build_confusion_matrix_from_sim(sim_data: dict) -> tuple:
    """从仿真 NPZ 数据构建 step 级别的混淆矩阵。

    每个受攻击时间步贡献一个 (true, pred) 对。
    Returns: (cm_counts(9,9), cm_norm(9,9), y_true_all, y_pred_all)
    """
    y_true_all = []
    y_pred_all = []
    atk_to_idx = {a: i for i, a in enumerate(ALL_ATTACK_TYPES)}

    for atk_true, npz_list in sim_data.items():
        if atk_true not in atk_to_idx:
            continue
        true_idx = atk_to_idx[atk_true]
        for d in npz_list:
            det_class = d.get('det_class', None)
            attack_active = d.get('attack_active', None)
            if det_class is None:
                continue
            # 将 det_class 字符串/对象转为索引
            pred_indices = np.array([atk_to_idx.get(str(c), 0) for c in det_class])
            # 受攻击窗口: attack_active > 0.5
            if attack_active is not None:
                mask = attack_active > 0.5
                # 对于 A0, 无攻击, 使用全部步
                if atk_true == 'A0' or mask.sum() == 0:
                    mask = np.ones(len(det_class), dtype=bool)
                    # 跳过前 100 步 (窗口填充)
                    mask[:100] = False
            else:
                mask = np.ones(len(det_class), dtype=bool)
                mask[:100] = False

            y_true_all.extend([true_idx] * mask.sum())
            y_pred_all.extend(pred_indices[mask].tolist())

    if not y_true_all:
        print("[WARN] 无分类数据, 返回空混淆矩阵")
        cm_counts = np.zeros((9, 9), dtype=int)
        cm_norm = np.zeros((9, 9))
        return cm_counts, cm_norm, np.array([]), np.array([])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # 混淆矩阵
    cm_counts = np.zeros((9, 9), dtype=int)
    for t, p in zip(y_true_all, y_pred_all):
        cm_counts[t, p] += 1

    # 行归一化
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_norm = cm_counts.astype(float) / np.maximum(row_sums, 1)

    print(f"混淆矩阵: {len(y_true_all)} 样本, "
          f"对角准确率={np.trace(cm_counts)/max(cm_counts.sum(), 1):.3f}")
    return cm_counts, cm_norm, y_true_all, y_pred_all


# ============================================================================
# 混淆矩阵绘图
# ============================================================================

def plot_confusion_matrix(cm_counts: np.ndarray, cm_norm: np.ndarray,
                          class_names: list, save_path: str):
    """双面板混淆矩阵: 左侧原始计数, 右侧行归一化百分比。"""
    n = len(class_names)
    labels_cn = [f'{a}\n{ATTACK_NAMES_CN[a]}' for a in class_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    fig.suptitle('攻击分类混淆矩阵 — CFMDetector (PINN-Flow)', fontsize=13, fontweight='bold')

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


# ============================================================================
# 重建对比图
# ============================================================================

def plot_reconstruction_comparison(npz_data: dict, attack_type: str, save_path: str):
    """3 通道 (x,y,θ) 真实攻击 vs 估计攻击时序对比。"""
    t_arr = npz_data.get('t', np.arange(len(npz_data.get('attack_signal', np.zeros((1,3))))))
    atk_true = npz_data.get('attack_signal', np.zeros((len(t_arr), 3)))
    atk_est = npz_data.get('det_attack_est', np.zeros((len(t_arr), 3)))
    atk_active = npz_data.get('attack_active', np.zeros(len(t_arr)))
    atk_onset = npz_data.get('attack_onset', 0)

    n = min(len(t_arr), len(atk_true))
    t_arr = t_arr[:n]; atk_true = atk_true[:n]; atk_est = atk_est[:n]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'攻击信号重建对比 — {attack_type} ({ATTACK_NAMES_CN[attack_type]})',
                 fontsize=13, fontweight='bold')

    channels = ['x', 'y', 'θ']
    units = ['m', 'm', 'rad']
    colors_true = ['#E91E63', '#2196F3', '#4CAF50']
    colors_est = ['#FF9800', '#00BCD4', '#8BC34A']

    for ch in range(3):
        ax = axes[ch]
        ax.plot(t_arr, atk_true[:, ch], color=colors_true[ch], lw=1.2, alpha=0.9,
                label=f'True a_{channels[ch]}')
        ax.plot(t_arr, atk_est[:, ch], color=colors_est[ch], lw=1.0, alpha=0.8,
                ls='--', label=f'Est a_{channels[ch]}')
        ax.axhline(0, color='gray', lw=0.5, ls=':', alpha=0.5)
        # 攻击窗着色
        if atk_onset > 0:
            ax.axvline(atk_onset, color='red', lw=0.8, ls=':', alpha=0.5)
        if atk_active.max() > 0:
            in_attack = atk_active > 0.5
            if in_attack.any():
                t_a = t_arr[in_attack]
                if len(t_a) > 0:
                    ax.axvspan(t_a[0], t_a[-1], alpha=0.08, color='red')
        # 通道 MAE
        mae = np.mean(np.abs(atk_true[:, ch] - atk_est[:, ch]))
        ax.set_ylabel(f'{channels[ch]} [{units[ch]}]')
        ax.set_title(f'{channels[ch]} 通道 (MAE={mae:.4f} {units[ch]})', fontsize=10)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('时间 [s]')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_reconstruction_summary(sim_data: dict, save_path: str):
    """3×3 攻击重建概览 (每个子图一个攻击类型, x 通道)。"""
    atks_order = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.suptitle('攻击信号重建总览 (x 通道)', fontsize=14, fontweight='bold')

    for idx, atk in enumerate(atks_order):
        ax = axes[idx // 3, idx % 3]
        if atk in sim_data:
            d = sim_data[atk][0]  # 取第一个 NPZ
            t_arr = d.get('t', None)
            atk_true = d.get('attack_signal', np.zeros((1, 3)))
            atk_est = d.get('det_attack_est', np.zeros((1, 3)))
            n = min(len(t_arr) if t_arr is not None else len(atk_true), len(atk_true))
            if t_arr is None:
                t_arr = np.arange(n) * 0.05
            else:
                t_arr = t_arr[:n]
            ax.plot(t_arr, atk_true[:n, 0], color='#E91E63', lw=1.0, alpha=0.8, label='True')
            ax.plot(t_arr, atk_est[:n, 0], color='#FF9800', lw=0.8, alpha=0.7,
                    ls='--', label='Est')
            mae = np.mean(np.abs(atk_true[:n, 0] - atk_est[:n, 0]))
            ax.set_title(f'{atk} ({ATTACK_NAMES_CN[atk]})\nMAE={mae:.4f}', fontsize=9)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.set_title(f'{atk}', fontsize=9)
        ax.axhline(0, color='gray', lw=0.5, ls=':', alpha=0.4)
        ax.grid(True, alpha=0.3)
        if idx % 3 == 0:
            ax.set_ylabel('a_x [m]', fontsize=8)
        if idx >= 6:
            ax.set_xlabel('时间 [s]', fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Recon Summary] {save_path}")


# ============================================================================
# 轨迹跟踪对比图
# ============================================================================

def plot_trajectory_comparison(npz_data: dict, attack_type: str, save_path: str):
    """2 面板: 2D 轨迹 + 位置误差时序。"""
    t_arr = npz_data.get('t', None)
    Upsilon_r = npz_data.get('Upsilon_r', None)
    true_state = npz_data.get('true_state', None)
    Upsilon_hat = npz_data.get('Upsilon_hat', None)
    X_error = npz_data.get('X_error', None)
    atk_active = npz_data.get('attack_active', None)
    atk_onset = npz_data.get('attack_onset', 0)

    if t_arr is None:
        t_arr = np.arange(len(true_state)) * 0.05

    n = min(len(t_arr), len(Upsilon_hat) if Upsilon_hat is not None else 0)
    if n == 0:
        print(f"  [WARN] {attack_type}: 无效数据, 跳过轨迹图")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle(f'轨迹跟踪对比 — {attack_type} ({ATTACK_NAMES_CN[attack_type]})',
                 fontsize=13, fontweight='bold')

    # ---- 面板 1: 2D 轨迹 ----
    ax1.plot(Upsilon_r[:n, 0], Upsilon_r[:n, 1], 'k--', lw=1.0, alpha=0.6, label='Reference')
    ax1.plot(Upsilon_hat[:n, 0], Upsilon_hat[:n, 1], '#2ca02c', lw=1.2, alpha=0.8,
             label='EKF Estimate')
    ax1.plot(true_state[:n, 0], true_state[:n, 1], '#1f77b4', lw=0.8, alpha=0.7,
             label='True State')
    # 攻击起始标记
    if atk_onset > 0 and atk_onset < t_arr[-1]:
        onset_k = int(atk_onset / 0.05)
        if onset_k < n:
            ax1.scatter(true_state[onset_k, 0], true_state[onset_k, 1],
                       marker='X', s=120, c='red', zorder=10, label=f'Attack onset (t={atk_onset:.0f}s)')
    ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
    ax1.set_title('2D 轨迹', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # ---- 面板 2: 位置误差 ----
    pos_err = np.sqrt(X_error[:n, 0]**2 + X_error[:n, 1]**2)
    ax2.plot(t_arr[:n], pos_err, 'r-', lw=1.0, alpha=0.9)
    ax2.set_xlabel('时间 [s]'); ax2.set_ylabel('位置误差 [m]')
    ax2.set_title('跟踪位置误差', fontweight='bold')
    if atk_onset > 0:
        ax2.axvline(atk_onset, color='red', lw=0.8, ls=':', alpha=0.5)
    if atk_active is not None and atk_active.max() > 0:
        in_attack = atk_active[:n] > 0.5
        if in_attack.any():
            t_a = t_arr[:n][in_attack]
            if len(t_a) > 0:
                ax2.axvspan(t_a[0], t_a[-1], alpha=0.08, color='red')
    ax2.grid(True, alpha=0.3)
    post_rmse = float(np.sqrt(np.mean(pos_err[int(atk_onset/0.05):]**2))) if atk_onset < t_arr[-1] else 0
    ax2.set_title(f'跟踪位置误差 (攻击后 RMSE={post_rmse:.4f}m)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_tracking_summary(sim_data: dict, save_path: str):
    """所有攻击类型的跟踪 RMSE 总览柱状图。"""
    atks_order = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
    rmse_list = {}
    for atk in atks_order:
        if atk in sim_data:
            d = sim_data[atk][0]
            X_error = d.get('X_error', None)
            atk_active = d.get('attack_active', None)
            if X_error is not None:
                if atk_active is not None and atk_active.max() > 0:
                    mask = atk_active > 0.5
                    if atk == 'A0':
                        rmse = float(np.sqrt(np.mean(X_error[:, 0]**2 + X_error[:, 1]**2)))
                    else:
                        pos = np.sqrt(X_error[mask, 0]**2 + X_error[mask, 1]**2)
                        rmse = float(np.sqrt(np.mean(pos**2))) if len(pos) > 0 else 0
                else:
                    rmse = float(np.sqrt(np.mean(X_error[:, 0]**2 + X_error[:, 1]**2)))
                rmse_list[atk] = rmse

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [ATK_COLORS.get(a, '#888888') for a in atks_order]
    values = [rmse_list.get(a, 0) for a in atks_order]
    bars = ax.bar(atks_order, values, color=colors, alpha=0.85, edgecolor='black', lw=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_ylabel('Position RMSE [m]')
    ax.set_title('攻击后跟踪位置 RMSE — CFMDetector', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 中文标注
    for i, a in enumerate(atks_order):
        if a in ATTACK_NAMES_CN:
            ax.annotate(ATTACK_NAMES_CN[a], (i, values[i]), textcoords="offset points",
                       xytext=(0, -18), ha='center', fontsize=6, color='gray', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Tracking Summary] {save_path}")


# ============================================================================
# 分类报告
# ============================================================================

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                   class_names: list, save_path: str) -> dict:
    """生成 sklearn classification_report 并保存为 txt。"""
    try:
        from sklearn.metrics import classification_report
    except ImportError:
        print("[WARN] sklearn 未安装, 跳过分类报告")
        return {}

    # 确定实际出现的类别
    present = sorted(set(y_true) | set(y_pred))
    present_names = [f'{class_names[i]} ({ATTACK_NAMES_CN[class_names[i]]})' for i in present]

    cr = classification_report(
        y_true, y_pred,
        labels=present,
        target_names=present_names,
        digits=4, output_dict=True, zero_division=0,
    )
    cr_text = classification_report(
        y_true, y_pred,
        labels=present,
        target_names=present_names,
        digits=4, zero_division=0,
    )

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("攻击分类报告 — CFMDetector (PINN-Flow)\n")
        f.write("=" * 60 + "\n")
        f.write(cr_text)
        f.write("\n")

    print(f"  [Classification Report] {save_path}")
    return cr


# ============================================================================
# 指标 TXT
# ============================================================================

def write_metrics_txt(sim_data: dict, cm_counts: np.ndarray, cm_norm: np.ndarray,
                      class_report: dict, output_dir: str, model_name: str,
                      model_path: str):
    """生成综合 metrics.txt。"""
    save_path = os.path.join(output_dir, 'metrics.txt')
    atks_order = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"CFMDetector 综合评估报告\n")
        f.write(f"=" * 70 + "\n\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"权重: {model_path}\n")
        f.write(f"日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # 1. 逐攻击类型指标
        f.write("-" * 70 + "\n")
        f.write("一、逐攻击类型跟踪指标\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'攻击':>6} {'类型名称':<12} {'攻击前RMSE':>12} {'攻击后RMSE':>12} {'最大误差':>10}\n")
        f.write("-" * 70 + "\n")

        for atk in atks_order:
            if atk in sim_data:
                d = sim_data[atk][0]
                X_error = d.get('X_error', None)
                atk_onset = d.get('attack_onset', 15.0)
                if X_error is not None:
                    n_steps = len(X_error)
                    onset_k = int(atk_onset / 0.05)
                    pre_mask = slice(100, onset_k) if onset_k > 100 else slice(100, n_steps//3)
                    atk_active = d.get('attack_active', None)
                    if atk_active is not None and atk_active.max() > 0 and atk != 'A0':
                        post_mask = atk_active > 0.5
                    else:
                        post_mask = slice(onset_k, n_steps) if atk != 'A0' else slice(100, n_steps)

                    pre_rmse = float(np.sqrt(np.mean(X_error[pre_mask, 0]**2 + X_error[pre_mask, 1]**2)))
                    if isinstance(post_mask, slice):
                        post_rmse = float(np.sqrt(np.mean(X_error[post_mask, 0]**2 + X_error[post_mask, 1]**2)))
                        post_max = float(np.max(np.sqrt(X_error[post_mask, 0]**2 + X_error[post_mask, 1]**2)))
                    else:
                        pos_post = np.sqrt(X_error[post_mask, 0]**2 + X_error[post_mask, 1]**2)
                        post_rmse = float(np.sqrt(np.mean(pos_post**2))) if len(pos_post) > 0 else 0
                        post_max = float(np.max(pos_post)) if len(pos_post) > 0 else 0

                    f.write(f"{atk:>6}  {ATTACK_NAMES_CN[atk]:<12} {pre_rmse:>12.4f} {post_rmse:>12.4f} {post_max:>10.4f}\n")
        f.write("\n")

        # 2. 混淆矩阵
        f.write("-" * 70 + "\n")
        f.write("二、混淆矩阵 (原始计数)\n")
        f.write("-" * 70 + "\n")
        header = f"{'':>10}" + "".join(f"{a:>8}" for a in atks_order)
        f.write(header + "\n")
        for i, a in enumerate(atks_order):
            row = f"{a:>10}" + "".join(f"{cm_counts[i,j]:>8}" for j in range(len(atks_order)))
            f.write(row + "\n")
        f.write("\n")

        # 行归一化
        f.write("混淆矩阵 (行归一化 %)\n")
        f.write("-" * 70 + "\n")
        f.write(header + "\n")
        for i, a in enumerate(atks_order):
            row = f"{a:>10}" + "".join(f"{cm_norm[i,j]*100:>7.1f}%" for j in range(len(atks_order)))
            f.write(row + "\n")
        f.write("\n")

        # 3. 分类报告
        if class_report:
            f.write("-" * 70 + "\n")
            f.write("三、分类报告 (Precision/Recall/F1)\n")
            f.write("-" * 70 + "\n")
            for cls_name in atks_order:
                key = f'{cls_name} ({ATTACK_NAMES_CN[cls_name]})'
                if key in class_report:
                    cr = class_report[key]
                    f.write(f"{key:<25} P={cr['precision']:.4f} R={cr['recall']:.4f} F1={cr['f1-score']:.4f} (n={cr['support']:.0f})\n")
            if 'macro avg' in class_report:
                ma = class_report['macro avg']
                f.write(f"\n{'宏观平均 (macro avg)':<25} P={ma['precision']:.4f} R={ma['recall']:.4f} F1={ma['f1-score']:.4f}\n")
            if 'weighted avg' in class_report:
                wa = class_report['weighted avg']
                f.write(f"{'加权平均 (weighted avg)':<25} P={wa['precision']:.4f} R={wa['recall']:.4f} F1={wa['f1-score']:.4f}\n")
            f.write("\n")

        # 4. 总体统计
        f.write("-" * 70 + "\n")
        f.write("四、总体统计\n")
        f.write("-" * 70 + "\n")
        overall_acc = np.trace(cm_counts) / max(cm_counts.sum(), 1)
        f.write(f"总准确率: {overall_acc*100:.2f}%\n")
        f.write(f"总样本数: {cm_counts.sum()}\n")

    print(f"  [Metrics TXT] {save_path}")


# ============================================================================
# CSV 导出
# ============================================================================

def export_metrics_csv(sim_data: dict, output_dir: str):
    """导出逐攻击类型指标 CSV。"""
    save_path = os.path.join(output_dir, 'metrics.csv')
    rows = []
    for atk in sorted(sim_data.keys()):
        for d in sim_data[atk]:
            X_error = d.get('X_error', None)
            atk_onset = d.get('attack_onset', 15.0)
            if X_error is None:
                continue
            n_steps = len(X_error)
            onset_k = int(atk_onset / 0.05)
            atk_active = d.get('attack_active', None)
            if atk_active is not None and atk_active.max() > 0 and atk != 'A0':
                post_mask = atk_active > 0.5
            else:
                post_mask = slice(onset_k, n_steps) if atk != 'A0' else slice(100, n_steps)

            if isinstance(post_mask, slice):
                post_rmse = float(np.sqrt(np.mean(X_error[post_mask, 0]**2 + X_error[post_mask, 1]**2)))
                post_max = float(np.max(np.sqrt(X_error[post_mask, 0]**2 + X_error[post_mask, 1]**2)))
            else:
                pos_post = np.sqrt(X_error[post_mask, 0]**2 + X_error[post_mask, 1]**2)
                post_rmse = float(np.sqrt(np.mean(pos_post**2))) if len(pos_post) > 0 else 0
                post_max = float(np.max(pos_post)) if len(pos_post) > 0 else 0

            rows.append({'attack_type': atk, 'name_cn': ATTACK_NAMES_CN.get(atk, ''),
                         'post_rmse_m': post_rmse, 'post_max_m': post_max})

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"  [Metrics CSV] {save_path}")
    return pd.DataFrame(rows) if rows else None


# ============================================================================
# Validation 模式
# ============================================================================

def run_validation_eval(model_path: str, norm_path: str, data_dir: str,
                        output_dir: str, dirs: dict):
    """在验证集上运行 CFMDetector 推理, 生成混淆矩阵和分类报告。"""
    import torch
    from .train_cfm import PreprocessedDataset, ALL_ATTACK_TYPES as ATK, ATTACK_NAMES_CN as ANC
    from .cfm_detector import CFMDetector

    print(f"\nValidation 模式: 验证集推理")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    # 加载模型
    config_path = os.path.join(os.path.dirname(model_path), 'cfm_cls_config.npz')
    cfg = {}
    if os.path.exists(config_path):
        cfg = dict(np.load(config_path, allow_pickle=True))

    # 检测模型版本, 兼容 v1/v2
    model_type = str(cfg.get('model_type', 'cfm'))
    d_model_cfg = int(cfg.get('d_model', 128))
    if model_type == 'cfm':
        # v1 旧模型: Transformer 骨干, 无正交分裂器
        model = CFMDetector(
            in_channels=int(cfg.get('in_channels', 5)),
            window_size=int(cfg.get('window_size', 100)),
            d_model=d_model_cfg,
            num_classes=int(cfg.get('num_classes', 9)),
            backbone_type='transformer',
            d_cls=d_model_cfg, d_fm=d_model_cfg,
            num_transformer_layers=int(cfg.get('num_transformer_layers', 4)),
            num_heads=int(cfg.get('num_heads', 8)),
            dim_feedforward=int(cfg.get('dim_feedforward', 512)),
            num_flow_blocks=int(cfg.get('num_flow_blocks', 4)),
            dropout=float(cfg.get('dropout', 0.1)),
        ).to(device)
    else:
        # v2 新模型: 从配置读取
        backbone_type = str(cfg.get('backbone_type', 'causal_conv'))
        dilations_raw = cfg.get('dilations', None)
        dilations = list(dilations_raw) if dilations_raw is not None else None
        model = CFMDetector(
            in_channels=int(cfg.get('in_channels', 5)),
            window_size=int(cfg.get('window_size', 100)),
            d_model=d_model_cfg,
            num_classes=int(cfg.get('num_classes', 9)),
            backbone_type=backbone_type,
            dilations=dilations,
            conv_kernel_size=int(cfg.get('conv_kernel_size', 3)),
            d_cls=int(cfg.get('d_cls', 64)),
            d_fm=int(cfg.get('d_fm', 64)),
            num_transformer_layers=int(cfg.get('num_transformer_layers', 4)),
            num_heads=int(cfg.get('num_heads', 8)),
            dim_feedforward=int(cfg.get('dim_feedforward', 512)),
            num_flow_blocks=int(cfg.get('num_flow_blocks', 4)),
            dim_feedforward_flow=int(cfg.get('dim_feedforward_flow', 192)),
            dropout=float(cfg.get('dropout', 0.1)),
        ).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 加载数据
    val_dataset = PreprocessedDataset(data_dir, 'val')
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=2, pin_memory=True)

    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, cls_label, _, _ in val_loader:
            x = x.to(device)
            features = model.encode(x)
            logits = model.classify(features)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred_all.extend(pred.tolist())
            y_true_all.extend(cls_label.numpy().tolist())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # 混淆矩阵
    cm_counts = np.zeros((9, 9), dtype=int)
    for t, p in zip(y_true_all, y_pred_all):
        cm_counts[t, p] += 1
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_norm = cm_counts.astype(float) / np.maximum(row_sums, 1)

    # 绘图
    class_names = ALL_ATTACK_TYPES
    plot_confusion_matrix(cm_counts, cm_norm, class_names,
                          os.path.join(output_dir, 'confusion_matrix.png'))

    # 分类报告
    class_report = generate_classification_report(
        y_true_all, y_pred_all, class_names,
        os.path.join(output_dir, 'classification_report.txt'))

    # 文本报告
    acc = np.trace(cm_counts) / max(cm_counts.sum(), 1)
    report_path = os.path.join(output_dir, 'metrics.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"CFMDetector 验证集评估\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"模型: {model_path}\n")
        f.write(f"总准确率: {acc*100:.2f}%\n")
        f.write(f"总样本数: {len(y_true_all)}\n\n")
        for i, a in enumerate(class_names):
            row_acc = cm_norm[i, i] * 100 if cm_counts[i].sum() > 0 else 0
            f.write(f"{a} ({ATTACK_NAMES_CN[a]}): {row_acc:.1f}% ({cm_counts[i].sum()} samples)\n")

    print(f"  Validation 完成: acc={acc:.4f}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    args = parse_args()

    # 确定输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = args.output_dir or os.path.join(
        SCRIPT_DIR, 'eval', f'{args.model_name}_{timestamp}')
    dirs = build_eval_dirs(output_root)

    print("=" * 60)
    print(f"CFMDetector 综合评估: {args.model_name}")
    print(f"模式: {args.mode}")
    print(f"输出: {output_root}")
    print("=" * 60)

    atk_types = (args.attack_types.split(',') if args.attack_types
                 else ALL_ATTACK_TYPES)
    atk_types = [a.strip() for a in atk_types if a.strip() in ALL_ATTACK_TYPES]

    # ---- 获取仿真数据 ----
    if args.mode == 'simulate' and not args.no_simulate:
        run_simulations(
            model_path=args.model_path,
            norm_path=args.norm_path,
            npz_dir=dirs['npz'],
            attack_types=atk_types,
            trajectory=args.trajectory,
            seed=args.seed,
        )

    sim_data = load_npz_files(dirs['npz'] if args.mode != 'from_npz' else args.npz_dir)

    if args.mode == 'validation':
        run_validation_eval(args.model_path, args.norm_path, args.data_dir,
                           output_root, dirs)
        print(f"\n评估完成! 结果: {output_root}")
        return

    if not sim_data:
        print("[ERROR] 无仿真数据可供分析。请先运行仿真或指定有效的 NPZ 目录。")
        sys.exit(1)

    # ---- 混淆矩阵 ----
    cm_counts, cm_norm, y_true, y_pred = build_confusion_matrix_from_sim(sim_data)
    plot_confusion_matrix(cm_counts, cm_norm, ALL_ATTACK_TYPES,
                          os.path.join(output_root, 'confusion_matrix.png'))

    # ---- 分类报告 ----
    class_report = {}
    if len(y_true) > 0:
        class_report = generate_classification_report(
            y_true, y_pred, ALL_ATTACK_TYPES,
            os.path.join(output_root, 'classification_report.txt'))

    # ---- 逐攻击重建图 ----
    print(f"\n生成重建对比图...")
    for atk in atk_types:
        if atk in sim_data:
            fp = os.path.join(dirs['recon'], f'recon_{atk}.png')
            plot_reconstruction_comparison(sim_data[atk][0], atk, fp)

    # ---- 重建总览 ----
    plot_reconstruction_summary(sim_data, os.path.join(dirs['recon'], 'recon_summary.png'))

    # ---- 逐攻击轨迹图 ----
    print(f"\n生成轨迹跟踪图...")
    for atk in atk_types:
        if atk in sim_data:
            fp = os.path.join(dirs['tracking'], f'tracking_{atk}.png')
            plot_trajectory_comparison(sim_data[atk][0], atk, fp)

    # ---- 跟踪总览 ----
    plot_tracking_summary(sim_data, os.path.join(dirs['tracking'], 'tracking_summary.png'))

    # ---- 指标 TXT ----
    write_metrics_txt(sim_data, cm_counts, cm_norm, class_report, output_root,
                      args.model_name, args.model_path or 'N/A')

    # ---- 指标 CSV ----
    export_metrics_csv(sim_data, output_root)

    print(f"\n{'='*60}")
    print(f"评估完成!")
    print(f"输出目录: {output_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
