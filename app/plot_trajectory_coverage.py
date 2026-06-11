"""
plot_trajectory_coverage.py — 临时脚本：绘制所有正常(A0)轨迹覆盖范围
==========================================================================
每种轨迹族一个子图，同族不同参数轨迹重叠绘制在同一子图中。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- IEEE 论文标准字体 ----
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['font.size'] = 10

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')
POS_BOUND = 2.5

# ---- 读取 metadata ----
df = pd.read_csv(os.path.join(RESULT_DIR, 'metadata.csv'))
df_a0 = df[df['attack_type'] == 'A0'].copy()

# ---- 按轨迹族分组 ----
families_order = ['lissajous', 'circular', 'spiral', 'random_waypoint', 'square']
families_cn = {
    'lissajous': 'Lissajous', 'circular': 'Circular', 'spiral': 'Spiral',
    'random_waypoint': 'Random Waypoint', 'square': 'Square',
}
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

# ---- 绘图 ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
ax_flat = axes.flatten()

for idx, fam in enumerate(families_order):
    ax = ax_flat[idx]
    sub = df_a0[df_a0['trajectory_family'] == fam]
    n_traj = len(sub)
    color = colors[idx]

    for i, (_, row) in enumerate(sub.iterrows()):
        fpath = os.path.join(RESULT_DIR, row['filename'])
        if not os.path.exists(fpath):
            continue
        data = np.load(fpath)
        Ur = data['Upsilon_r']  # (N, 3) reference trajectory
        x, y = Ur[:, 0], Ur[:, 1]
        alpha = max(0.25, 0.7 - n_traj * 0.006)  # 越多轨迹越透明
        ax.plot(x, y, color=color, linewidth=0.5, alpha=alpha)
        data.close()

    # 空间边界
    pb = POS_BOUND
    ax.plot([-pb, pb, pb, -pb, -pb], [-pb, -pb, pb, pb, -pb],
            'gray', linewidth=0.6, alpha=0.4, linestyle=':')
    ax.set_xlim(-pb - 0.3, pb + 0.3)
    ax.set_ylim(-pb - 0.3, pb + 0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'{families_cn[fam]}  (n={n_traj})',
                 fontsize=12, fontweight='bold', color=color)
    ax.grid(True, alpha=0.15)

# ---- 第6图：汇总统计 ----
ax6 = ax_flat[5]
totals = []
for fam in families_order:
    cnt = len(df_a0[df_a0['trajectory_family'] == fam])
    totals.append(cnt)
x_pos = np.arange(len(families_order))
bars = ax6.bar(x_pos, totals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
for bar, cnt in zip(bars, totals):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(cnt), ha='center', fontsize=10, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels([families_cn[f] for f in families_order], rotation=15, fontsize=9)
ax6.set_ylabel('Number of Trajectories')
ax6.set_title('Trajectory Count by Family', fontsize=12, fontweight='bold')
ax6.set_ylim(0, max(totals) * 1.25)
ax6.grid(True, alpha=0.2, axis='y')

fig.suptitle('Normal Trajectory Coverage — All Families (50 s, ±2.5 m boundary)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                       'fig_trajectory_coverage.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight')
print(f'图片已保存: {outpath}')
plt.show()
