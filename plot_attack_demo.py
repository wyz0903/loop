"""
plot_attack_demo.py — 临时脚本: 一条 Lissajous 轨迹遭遇 8 种攻击的可视化
==========================================================================
生成 3×3 子图 (左上=正常, 其余=A1~A8)。
每子图显示: 真实路径 vs 参考轨迹, 攻击激活期高亮。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict

from model import (WMRParams, WMRKinematics, EKFEstimator, SensorSimulator,
                   SIM_STEPS, SIM_TIME)
from controller import NMPCController, NMPCParams
from attack import SensorAttack, AttackConfig
from generate_dataset import (RandomizedTrajectory, _rk4_unicycle,
                              _internal_kinematic_step, ATTACK_NAMES)

# ---- IEEE 标准字体 ----
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['font.size'] = 8

Ts = 0.05

ATK_TYPES = ['A1','A2','A3','A4','A5','A6','A7','A8']
ATK_COLORS = ['#E91E63','#FF9800','#2196F3','#F44336',
              '#9C27B0','#795548','#00BCD4','#607D8B']


def run_one(traj, attack_type, onset, duration, ctrl, sim_seed):
    """运行一次闭环仿真"""
    wmr_params = WMRParams()
    robot = WMRKinematics(wmr_params)
    sensor = SensorSimulator()
    ekf = EKFEstimator(wmr_params)

    atk_cfg = AttackConfig(attack_duration=duration)
    attacker = SensorAttack(attack_type=attack_type, onset_time=onset,
                            config=atk_cfg, seed=sim_seed)

    traj.reset()
    # 机器人与 EKF 从轨迹起点附近随机初始化 (测试 NMPC 收敛性)
    perturb_rng = np.random.RandomState(sim_seed)
    init_state = np.array([
        traj._x_r + perturb_rng.uniform(-0.3, 0.3),
        traj._y_r + perturb_rng.uniform(-0.3, 0.3),
        traj._theta_r + perturb_rng.uniform(-0.2, 0.2),
    ])
    robot.reset(init_state)
    ekf.reset(init_state)
    ctrl.reset()
    attacker.reset()
    np.random.seed(sim_seed)

    data = defaultdict(list)
    u_cmd = np.zeros(2)
    internal_state = init_state.copy()
    _recalib_interval = 200
    _last_recalib_step = -_recalib_interval

    for step in range(SIM_STEPS):
        t = step * Ts
        Upsilon_r, u_r = traj.step(t)
        Ur_seq = traj.generate_sequence(t, NMPCParams().N)

        true_state = robot.state.copy()
        noise = sensor.noise_std * np.random.randn(3)
        y_clean = true_state + noise
        y_meas = attacker.inject(t, y_clean)
        attack_signal = y_meas - y_clean

        # EKF 估计 — NMPC 必须基于估计值而非真实值
        Upsilon_hat, ekf_innovation = ekf.step(y_meas, u_cmd)
        X_error = WMRKinematics.compute_error(Upsilon_r, Upsilon_hat)
        u_cmd = ctrl.solve(X_error, Ur_seq)
        u_a = WMRKinematics.clamp_control(u_cmd)
        robot.step(u_a)

        data['t'].append(t)
        data['Upsilon_r'].append(Upsilon_r.copy())
        data['true_state'].append(true_state.copy())
        data['y_meas'].append(y_meas.copy())
        data['attack_signal'].append(attack_signal.copy())
        data['attack_active'].append(1.0 if attacker._is_active(t) else 0.0)

    return {k: np.array(v) for k, v in data.items()}


def main():
    # 固定一条 Lissajous 轨迹 (用种子确定性生成)
    TRAJ_SEED = 123456  # 固定种子，确定一条中等大小的 8 字形
    traj = RandomizedTrajectory(Ts=Ts, seed=TRAJ_SEED, family='lissajous')
    info = traj.get_info()
    print(f"Trajectory: family={info['trajectory_family']}, "
          f"v_r={info.get('v_r','N/A'):.3f}, w_freq={info.get('w_freq','N/A'):.3f}")

    # NMPC 求解器 (共享)
    ctrl = NMPCController()
    ctrl.load_or_build()

    # 攻击参数 (固定以便对比)
    onset = 12.0
    duration = 8.0

    # ---- 运行 ----
    print("Running A0 (Normal)...")
    data_a0 = run_one(traj, 'A0', onset=50.0, duration=None, ctrl=ctrl, sim_seed=42)
    ctrl.reset()

    results = {}
    for atk in ATK_TYPES:
        print(f"Running {atk} ({ATTACK_NAMES[atk]})...")
        ctrl.reset()
        results[atk] = run_one(traj, atk, onset=onset, duration=duration,
                               ctrl=ctrl, sim_seed=42)

    # ---- 绘图 (3×3) ----
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    ax_flat = axes.flatten()
    pb = 2.5

    # 子图 0: A0 Normal
    ax = ax_flat[0]
    d = data_a0
    ax.plot(d['true_state'][:,0], d['true_state'][:,1],
            color='#4CAF50', linewidth=1.2, label='True path')
    ax.plot(d['Upsilon_r'][:,0], d['Upsilon_r'][:,1],
            color='#2196F3', linewidth=0.7, linestyle='--', alpha=0.7, label='Reference')
    ax.plot(d['true_state'][0,0], d['true_state'][0,1], 'ko', markersize=5, label='Start')
    ax.plot(d['true_state'][-1,0], d['true_state'][-1,1], 'r*', markersize=8, label='End')
    ax.set_title('A0 — Normal (baseline)', fontsize=11, fontweight='bold', color='#4CAF50')
    ax.legend(loc='upper right', fontsize=6.5, framealpha=0.8)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.15)
    ax.plot([-pb,pb,pb,-pb,-pb],[-pb,-pb,pb,pb,-pb],'gray',lw=0.5,ls=':',alpha=0.35)
    ax.set_xlim(-pb-0.3, pb+0.3); ax.set_ylim(-pb-0.3, pb+0.3)

    # 子图 1-8: A1-A8
    for idx, atk in enumerate(ATK_TYPES):
        ax = ax_flat[idx + 1]
        d = results[atk]
        color = ATK_COLORS[idx]

        # 攻击前后分段
        active = d['attack_active']
        onset_idx = int(onset / Ts)
        offset_idx = int((onset + duration) / Ts)
        offset_idx = min(offset_idx, len(active) - 1)

        # 攻击前 (灰色细线)
        if onset_idx > 0:
            ax.plot(d['true_state'][:onset_idx+1,0], d['true_state'][:onset_idx+1,1],
                    color='gray', linewidth=0.6, alpha=0.5, label='Pre-attack')
        # 攻击中 (粗彩色线)
        ax.plot(d['true_state'][onset_idx:offset_idx+1,0],
                d['true_state'][onset_idx:offset_idx+1,1],
                color=color, linewidth=2.2, alpha=0.9, label='Under attack')
        # 攻击后 (细线)
        if offset_idx < len(active) - 1:
            ax.plot(d['true_state'][offset_idx:,0], d['true_state'][offset_idx:,1],
                    color='gray', linewidth=0.6, alpha=0.4, linestyle=':', label='Post-attack')

        # 参考轨迹
        ax.plot(d['Upsilon_r'][:,0], d['Upsilon_r'][:,1],
                color='#2196F3', linewidth=0.5, linestyle='--', alpha=0.5)

        # 起止点
        ax.plot(d['true_state'][0,0], d['true_state'][0,1], 'ko', markersize=4)
        ax.plot(d['true_state'][-1,0], d['true_state'][-1,1], 'r*', markersize=7)
        # 攻击起始点
        ax.plot(d['true_state'][onset_idx,0], d['true_state'][onset_idx,1],
                'o', color='red', markersize=9, markerfacecolor='none',
                markeredgewidth=2.5, label=f'Onset t={onset:.0f}s')

        # 攻击信号范数作为文字标注
        atk_norm = np.linalg.norm(d['attack_signal'], axis=1)
        max_dev = atk_norm[onset_idx:offset_idx+1].max()
        ax.text(0.95, 0.05, f'Max dev: {max_dev:.3f}',
                transform=ax.transAxes, fontsize=7, ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

        ax.set_title(f'{atk} — {ATTACK_NAMES[atk]}', fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.15)
        ax.plot([-pb,pb,pb,-pb,-pb],[-pb,-pb,pb,pb,-pb],'gray',lw=0.5,ls=':',alpha=0.35)
        ax.set_xlim(-pb-0.3, pb+0.3); ax.set_ylim(-pb-0.3, pb+0.3)
        if idx == 7:  # 最后一个添加图例
            ax.legend(loc='upper right', fontsize=6, framealpha=0.8)

    fig.suptitle('Attack Impact on Lissajous Trajectory\n'
                 f'(onset={onset:.0f}s, duration={duration:.0f}s, 50s simulation)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'fig_attack_demo.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f'\nSaved: {outpath}')
    plt.show()


if __name__ == '__main__':
    main()
