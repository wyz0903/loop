"""
visualization.py — 论文可视化模块
================================
分步骤、全面的 WMR 仿真可视化。每种调试场景输出到独立的子目录。

用法:
  python visualization.py                     # 默认: 参考轨迹族参数多样性网格
  python visualization.py --mode dataset      # 数据集可视化 (2×4 攻击子图)
  python visualization.py --no-save           # 交互式显示 (不保存文件)

输出目录:
  results/trajectories/            参考轨迹可视化
  results/dataset/                 数据集攻击轨迹可视化
  results/simulations/             仿真结果 (由 simulate.py 输出)
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# IEEE 论文标准字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

from model import WMRParams, RandomizedTrajectory, SIM_STEPS

# 输出目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJ_RESULT_DIR = os.path.join(BASE_DIR, 'results', 'trajectories')
DATASET_RESULT_DIR = os.path.join(BASE_DIR, 'results', 'dataset')
os.makedirs(TRAJ_RESULT_DIR, exist_ok=True)
os.makedirs(DATASET_RESULT_DIR, exist_ok=True)

TRAJECTORY_FAMILIES = ['lissajous', 'circular', 'spiral', 'random_waypoint', 'square']

FAMILY_DISPLAY_NAMES = {
    'lissajous':       'Lissajous',
    'circular':        'Circular',
    'spiral':          'Spiral',
    'random_waypoint': 'Random Waypoint',
    'square':          'Square',
}

FAMILY_COLORS = {
    'lissajous':       '#2196F3',
    'circular':        '#4CAF50',
    'spiral':          '#FF9800',
    'random_waypoint': '#E91E63',
    'square':          '#9C27B0',
}

# 每族展示的轨迹数 (与 generate_dataset.py --num-per-family 默认值一致)
N_VARIANTS = 12


# ============================================================================
# 工具函数
# ============================================================================

def _generate_trajectory_path(family: str, seed: int) -> dict:
    """生成一条参考轨迹的完整路径 (仅运动学开环，无攻击/噪声)"""
    Ts = WMRParams().Ts
    traj = RandomizedTrajectory(Ts=Ts, family=family, seed=seed)
    traj.reset()

    n_steps = SIM_STEPS
    x_arr = np.zeros(n_steps)
    y_arr = np.zeros(n_steps)

    for step in range(n_steps):
        t = step * Ts
        state, _ = traj.step(t)
        x_arr[step] = state[0]
        y_arr[step] = state[1]

    return {
        'x': x_arr, 'y': y_arr,
        'info': traj.get_info(),
        'family': family,
        'seed': seed,
    }


def _format_params(family: str, info: dict) -> str:
    """格式化轨迹参数为简短可读字符串 (用于子图底部标注)"""
    if family == 'lissajous':
        return r'$v$={:.2f}, $\omega_f$={:.2f}'.format(info['v_r'], info['w_freq'])
    elif family == 'circular':
        return r'$v$={:.2f}, $\omega$={:.2f}'.format(info['v_r'], info['w_r'])
    elif family == 'spiral':
        v, r0, rmax = info['v'], info['R0'], info['Rmax']
        return r'$v$={:.2f}, $R_0$={:.2f}, $R_{{max}}$={:.1f}'.format(v, r0, rmax)
    elif family == 'random_waypoint':
        return r'$v$={:.2f}'.format(info['v_r'])
    elif family == 'square':
        return r'$L$={:.1f}, $R$={:.2f}, $v$={:.2f}'.format(
            info['side'], info['R'], info['v'])
    return ''


# ============================================================================
# 1. 参考轨迹族参数多样性网格
# ============================================================================

def plot_reference_trajectories(save: bool = True):
    """绘制所有参考轨迹族的完整参数多样性网格

    布局: {N_VARIANTS} 行 × 5 列 (每列一种轨迹族)。
    每行使用不同的随机种子，展示族内参数多样性。
    不含攻击，仅展示数据集生成时使用的参考轨迹。
    固定输出全部 {N_VARIANTS} 个参数变体 (与 generate_dataset.py 默认值一致)。
    """
    families = TRAJECTORY_FAMILIES
    n_cols = len(families)
    n_rows = N_VARIANTS
    b = WMRParams().pos_bound

    # 每子图 2.3" × 2.0"，总尺寸约 11.5" × 24" 适合纵向滚动查看
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.3 * n_cols, 2.0 * n_rows),
                             constrained_layout=True)

    for row in range(n_rows):
        for col, family in enumerate(families):
            ax = axes[row, col]
            seed = row * 100 + col
            data = _generate_trajectory_path(family, seed)
            color = FAMILY_COLORS[family]

            # 2D 轨迹
            ax.plot(data['x'], data['y'], linewidth=0.6, color=color, alpha=0.85)

            # 起点 / 终点
            ax.scatter(data['x'][0], data['y'][0], c='#2ca02c', s=30, marker='o',
                       zorder=5, edgecolors='white', linewidths=0.5)
            ax.scatter(data['x'][-1], data['y'][-1], c='#d62728', s=50, marker='*',
                       zorder=5, edgecolors='white', linewidths=0.5)

            # 空间安全边界
            ax.plot([-b, b, b, -b, -b], [-b, -b, b, b, -b], color='gray',
                    linewidth=0.5, linestyle=':', alpha=0.35)

            ax.set_xlim(-b - 0.25, b + 0.25)
            ax.set_ylim(-b - 0.25, b + 0.25)
            ax.set_aspect('equal')

            # 参数标注 (子图底部)
            param_str = _format_params(family, data['info'])
            ax.text(0.5, -0.08, param_str, transform=ax.transAxes, fontsize=7.5,
                    ha='center', va='top', style='italic', color='#555555')

            # 列标题 (仅第一行)
            if row == 0:
                ax.set_title(FAMILY_DISPLAY_NAMES[family],
                             fontsize=12, fontweight='bold', color=color, pad=8)

            # 行标签 (仅第一列)
            if col == 0:
                ax.set_ylabel(f'#{row+1}', fontsize=10, fontweight='bold',
                              rotation=0, labelpad=22, va='center')

            # 隐藏刻度
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(length=0)

    fig.suptitle('Reference Trajectory Families — Parameter Diversity (No Attack)',
                 fontsize=16, fontweight='bold')

    if save:
        save_path = os.path.join(TRAJ_RESULT_DIR, 'reference_trajectories.png')
        fig.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  [Plot] {save_path}")
        return save_path
    else:
        return fig


# ============================================================================
# 2. 数据集攻击轨迹可视化 (2×4 子图)
# ============================================================================

# 攻击类型常量
ATK_NAMES = {
    'A0': 'Normal', 'A1': 'Constant Bias', 'A2': 'Sinusoidal',
    'A3': 'Drift', 'A4': 'Replay Attack', 'A5': 'Intermittent Dropout',
    'A6': 'Scaling', 'A7': 'Sensor Freeze',
}

# 数据集可视化可调参数
DS_FIG_DPI = 150
DS_LINE_ALPHA_NORMAL = 0.55
DS_LINE_ALPHA_ATTACK = 0.95
DS_LINE_WIDTH = 0.8
DS_REF_ALPHA = 0.35
DS_ATTACK_LW_MULT = 2.2


def _find_latest_dataset(dataset_root='dataset'):
    """找到最新（按名称排序最大）的时间戳数据集子目录。"""
    root = Path(dataset_root)
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    subdirs.sort(reverse=True)
    return subdirs[0]


def _build_trajectory_label(row: pd.Series) -> str:
    """根据轨迹族和参数生成图标题。"""
    fam = row['trajectory_family']
    if fam == 'lissajous':
        return (f"Lissajous  |  "
                f"v_r={row['traj_v_r']:.3f},  "
                f"ω_f={row['traj_w_freq']:.2f},  "
                f"ω_a={row['traj_w_amp']:.2f}")
    elif fam == 'circular':
        return (f"Circular  |  "
                f"v_r={row['traj_v_r']:.3f},  "
                f"ω_r={row['traj_w_r']:+.3f}")
    elif fam == 'spiral':
        direction = 'CCW' if row['traj_direction'] > 0 else 'CW'
        return (f"Spiral ({direction})  |  "
                f"v={row['traj_v']:.3f},  "
                f"R0={row['traj_R0']:.2f},  "
                f"Rmax={row['traj_Rmax']:.2f}")
    elif fam == 'random_waypoint':
        return f"Random Waypoint  |  v_r={row['traj_v_r']:.3f}"
    elif fam == 'square':
        direction = 'CCW' if row['traj_direction'] > 0 else 'CW'
        return (f"Square ({direction})  |  "
                f"side={row['traj_side']:.2f},  "
                f"R={row['traj_R']:.2f},  "
                f"v={row['traj_v']:.3f}")
    return fam


def _build_dataset_filename(row: pd.Series) -> str:
    """根据轨迹族和参数生成文件名（不含扩展名）。"""
    fam = row['trajectory_family']
    cid = int(row['config_id'])
    if fam == 'lissajous':
        base = f"lissajous_v{row['traj_v_r']:.3f}_wf{row['traj_w_freq']:.2f}_wa{row['traj_w_amp']:.2f}"
    elif fam == 'circular':
        base = f"circular_v{row['traj_v_r']:.3f}_wr{row['traj_w_r']:+.3f}"
    elif fam == 'spiral':
        d = 'CCW' if row['traj_direction'] > 0 else 'CW'
        base = f"spiral_{d}_v{row['traj_v']:.3f}_R0_{row['traj_R0']:.2f}_Rmax{row['traj_Rmax']:.2f}"
    elif fam == 'random_waypoint':
        base = f"randwaypoint_v{row['traj_v_r']:.3f}"
    elif fam == 'square':
        d = 'CCW' if row['traj_direction'] > 0 else 'CW'
        base = f"square_{d}_side{row['traj_side']:.2f}_R{row['traj_R']:.2f}_v{row['traj_v']:.3f}"
    else:
        base = fam
    return f"cfg{cid:04d}_{base}"


def _plot_one_config(dataset_dir: Path, config_rows: pd.DataFrame, output_dir: Path,
                     swap_axes: bool = False):
    """为一个 config_id 生成 2×4 子图。

    Args:
        swap_axes: 若 True，横轴=y、纵轴=x（横版）；否则横轴=x、纵轴=y（竖版）
    """
    sample_row = config_rows.iloc[0]
    title_label = _build_trajectory_label(sample_row)
    fname = _build_dataset_filename(sample_row)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    axes = axes.flatten()

    # 颜色定义
    TRAJ_COLOR = '#64B5F6'   # 浅蓝色 — 正常轨迹段
    ATK_COLOR  = '#E74C3C'   # 红色   — 攻击段
    REF_COLOR  = 'gray'
    START_COLOR = '#2ca02c'  # 起点绿色
    END_COLOR   = '#d62728'  # 终点红色

    for idx, atk_type in enumerate(['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']):
        ax = axes[idx]
        atk_row = config_rows[config_rows['attack_type'] == atk_type]
        if atk_row.empty:
            ax.set_title(f"{atk_type} — Missing", fontsize=10)
            ax.set_aspect('equal')
            continue

        atk_row = atk_row.iloc[0]
        npz_path = dataset_dir / atk_row['filename']
        data = np.load(npz_path)

        t = data['t']
        y_meas = data['y_meas']
        Upsilon_r = data['Upsilon_r']

        onset_step = int(atk_row['attack_onset_step'])
        offset_step = int(atk_row['attack_offset_step'])

        # 横版/竖版坐标索引与轴标签
        xi, yi = (1, 0) if swap_axes else (0, 1)
        xlabel, ylabel = ('y [m]', 'x [m]') if swap_axes else ('x [m]', 'y [m]')

        # 参考轨迹（灰色虚线）
        ax.plot(Upsilon_r[:, xi], Upsilon_r[:, yi],
                color=REF_COLOR, linestyle='--', linewidth=0.6, alpha=DS_REF_ALPHA)

        if atk_type == 'A0' or onset_step >= len(t):
            ax.plot(y_meas[:, xi], y_meas[:, yi],
                    color=TRAJ_COLOR, linewidth=DS_LINE_WIDTH,
                    alpha=DS_LINE_ALPHA_NORMAL)
        else:
            # 攻击前段
            ax.plot(y_meas[:onset_step + 1, xi], y_meas[:onset_step + 1, yi],
                    color=TRAJ_COLOR, linewidth=DS_LINE_WIDTH,
                    alpha=DS_LINE_ALPHA_NORMAL)

            # 攻击段（高亮）
            end_atk = min(offset_step + 1, len(t))
            ax.plot(y_meas[onset_step:end_atk, xi], y_meas[onset_step:end_atk, yi],
                    color=ATK_COLOR, linewidth=DS_LINE_WIDTH * DS_ATTACK_LW_MULT,
                    alpha=DS_LINE_ALPHA_ATTACK)

            # 攻击后段
            if offset_step < len(t):
                ax.plot(y_meas[offset_step:, xi], y_meas[offset_step:, yi],
                        color=TRAJ_COLOR, linewidth=DS_LINE_WIDTH,
                        alpha=DS_LINE_ALPHA_NORMAL)

            # 攻击起止标记
            ax.scatter(y_meas[onset_step, xi], y_meas[onset_step, yi],
                       color='red', s=20, zorder=5, marker='o',
                       edgecolors='darkred', linewidths=0.5)
            ax.scatter(y_meas[end_atk - 1, xi], y_meas[end_atk - 1, yi],
                       color='darkred', s=20, zorder=5, marker='s',
                       edgecolors='black', linewidths=0.5)

        # 起点和终点标记（所有子图）
        ax.scatter(y_meas[0, xi], y_meas[0, yi],
                   c=START_COLOR, s=30, marker='o', zorder=6,
                   edgecolors='white', linewidths=0.5)
        ax.scatter(y_meas[-1, xi], y_meas[-1, yi],
                   c=END_COLOR, s=50, marker='*', zorder=6,
                   edgecolors='white', linewidths=0.5)

        ax.set_title(f"{atk_type}: {ATK_NAMES[atk_type]}", fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linewidth=0.4)

    # 统一图例（代理艺术家，不依赖具体子图数据）
    legend_handles = [
        mlines.Line2D([], [], color=REF_COLOR, linestyle='--', linewidth=0.6,
                      alpha=DS_REF_ALPHA, label='Reference'),
        mlines.Line2D([], [], color=TRAJ_COLOR, linewidth=DS_LINE_WIDTH,
                      label='Measured'),
        mlines.Line2D([], [], color=ATK_COLOR, linewidth=DS_LINE_WIDTH * DS_ATTACK_LW_MULT,
                      label='Attacked'),
        mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=6,
                      markeredgecolor='darkred', markeredgewidth=0.5, label='Onset'),
        mlines.Line2D([], [], color='darkred', marker='s', linestyle='None', markersize=6,
                      markeredgecolor='black', markeredgewidth=0.5, label='Offset'),
        mlines.Line2D([], [], color=START_COLOR, marker='o', linestyle='None', markersize=7,
                      markeredgecolor='white', markeredgewidth=0.5, label='Start'),
        mlines.Line2D([], [], color=END_COLOR, marker='*', linestyle='None', markersize=8,
                      markeredgecolor='white', markeredgewidth=0.5, label='End'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=7,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0))

    fig.suptitle(title_label, fontsize=13, fontweight='bold', y=1.02)

    save_path = output_dir / f"{fname}.png"
    fig.savefig(save_path, dpi=DS_FIG_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return save_path


def plot_dataset_trajectories(save: bool = True):
    """对最新数据集所有轨迹配置，生成竖版 + 横版 2×4 子图。

    竖版 (portrait):  横轴=x, 纵轴=y  →  results/dataset/portrait/
    横版 (landscape): 横轴=y, 纵轴=x  →  results/dataset/landscape/
    每个 config_id 两张图，文件名包含轨迹族名称和参数。
    """
    ds_dir = _find_latest_dataset()
    print(f"数据集: {ds_dir}")

    df = pd.read_csv(ds_dir / 'metadata.csv')
    config_ids = sorted(df['config_id'].unique())
    print(f"轨迹配置数: {len(config_ids)} (共 {len(df)} 个仿真文件)")

    # 竖版 / 横版输出子目录
    portrait_dir = Path(DATASET_RESULT_DIR) / 'portrait'
    landscape_dir = Path(DATASET_RESULT_DIR) / 'landscape'
    portrait_dir.mkdir(parents=True, exist_ok=True)
    landscape_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for cid in config_ids:
        config_rows = df[df['config_id'] == cid]

        # 竖版：横轴=x, 纵轴=y
        sp_p = _plot_one_config(ds_dir, config_rows, portrait_dir, swap_axes=False)
        saved.append(sp_p)

        # 横版：横轴=y, 纵轴=x
        sp_l = _plot_one_config(ds_dir, config_rows, landscape_dir, swap_axes=True)
        saved.append(sp_l)

        print(f"  [{cid:3d}/{len(config_ids) - 1}] {sp_p.name}  |  {sp_l.name}")

    print(f"\n完成。共 {len(saved)} 张图")
    print(f"  竖版目录: {portrait_dir.resolve()}")
    print(f"  横版目录: {landscape_dir.resolve()}")
    return saved


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='WMR Visualization Module — 分步骤论文可视化')
    parser.add_argument('--mode', choices=['trajectories', 'dataset'],
                        default='trajectories',
                        help='可视化模式: trajectories (参考轨迹), dataset (数据集攻击轨迹)')
    parser.add_argument('--no-save', action='store_true',
                        help='Show interactively instead of saving to file')

    args = parser.parse_args()

    if args.mode == 'trajectories':
        result = plot_reference_trajectories(save=not args.no_save)
    elif args.mode == 'dataset':
        result = plot_dataset_trajectories(save=not args.no_save)

    if not args.no_save and result is not None:
        count = len(result) if isinstance(result, list) else 1
        print(f"Saved: {count} file(s)")


if __name__ == '__main__':
    main()