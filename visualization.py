"""
visualization.py — 论文可视化模块
================================
分步骤、全面的 WMR 仿真可视化。每种调试场景输出到独立的子目录。

用法:
  python visualization.py           # 默认: 参考轨迹族参数多样性网格
  python visualization.py --no-save # 交互式显示 (不保存文件)

输出目录:
  results/trajectories/            参考轨迹可视化
  results/simulations/             仿真结果 (由 simulate.py 输出)
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# IEEE 论文标准字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

from model import WMRParams, RandomizedTrajectory, SIM_STEPS

# 输出到 results/trajectories/ 子目录
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'results', 'trajectories')
os.makedirs(RESULT_DIR, exist_ok=True)

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
        save_path = os.path.join(RESULT_DIR, 'reference_trajectories.png')
        fig.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  [Plot] {save_path}")
        return save_path
    else:
        return fig


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='WMR Visualization Module — 分步骤论文可视化')
    parser.add_argument('--no-save', action='store_true',
                        help='Show interactively instead of saving to file')

    args = parser.parse_args()

    result = plot_reference_trajectories(save=not args.no_save)
    if not args.no_save and result is not None:
        print(f"Saved: {result}")


if __name__ == '__main__':
    main()