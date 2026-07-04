"""
simulate.py — WMR 闭环仿真 (含 CFM 检测器集成)
================================================
统一的闭环仿真脚本，支持: 正常运行 / 攻击场景 / CFM 检测器介入。

模式:
  python simulate.py                        # CFM 模式 (默认 A1, lissajous)
  python simulate.py --no-detector          # 无检测器基线
  python simulate.py --attack A1            # 指定攻击类型
  python simulate.py --all                  # 批量所有 8 种攻击
  python simulate.py --compare              # 五族轨迹无攻击跟踪对比
  python simulate.py --no-plot              # 跳过图形显示
"""

import os
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

from model import (WMRParams, WMRKinematics, SensorSimulator,
                   LissajousTrajectory, CircularTrajectory, SIM_STEPS)
from controller import NMPCController, NMPCParams
from attack import SensorAttack, ALL_ATTACK_TYPES, ATTACK_NAMES
from backend import CFMDetectorBackend

# ============================================================================
# 全局配置
# ============================================================================

# 仿真总时长与步数严格对齐 model.SIM_STEPS，避免标签与实际不一致
SIM_TIME = SIM_STEPS * WMRParams().Ts   # = 1000 * 0.05 = 50.0 s
ATTACK_ONSET_DEFAULT = 15.0
ATTACK_ONSET_MIN = 5.0
ATTACK_ONSET_MAX = 30.0
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

# 机器人初始位姿偏移范围 (世界坐标系, 用于检验 NMPC 收敛能力)
# 每次仿真在 [0, max] 内均匀随机采样，符号随机正负; 全零时机器人严格从起点出发
INIT_POS_MAX = 0.15        # 位置偏移最大幅值 [m] (每个轴独立 U(0, max) × 随机符号)
INIT_HEADING_MAX = 0.2     # 朝向偏移最大幅值 [rad] (U(0, max) × 随机符号)

TRAJECTORY_FAMILIES = ['lissajous', 'circular', 'spiral', 'random_waypoint', 'square']


# ============================================================================
# 辅助函数
# ============================================================================

def _has_attack(data: dict) -> bool:
    """该次仿真是否存在真实攻击 (A0 或攻击从未触发时返回 False)"""
    if str(data.get('attack_type', 'A0')) == 'A0':
        return False
    active = data.get('attack_active', None)
    if active is not None and hasattr(active, 'max'):
        return bool(active.max() > 0.5)
    return True


def _get_onset_idx(data: dict) -> int:
    """从仿真数据中获取攻击起始索引 (考虑窗口填充期)"""
    n_steps = len(data['t']) if 't' in data else 700
    if 'attack_active' in data:
        active = data['attack_active']
        if hasattr(active, 'max') and active.max() > 0.5:
            raw_onset = int(np.argmax(active > 0.5))
            return min(max(raw_onset, 100), n_steps - 1)
    if 'attack_onset' in data:
        raw_onset = int(float(data['attack_onset']) / data.get('Ts', 0.05))
        return min(max(raw_onset, 100), n_steps - 1)
    return min(max(int(ATTACK_ONSET_DEFAULT / 0.05), 100), n_steps - 1)


def _create_trajectory(traj_type: str, Ts: float, seed: int):
    """创建指定类型的轨迹生成器"""
    from generate_dataset import RandomizedTrajectory

    if traj_type == 'lissajous':
        return LissajousTrajectory(Ts=Ts)
    elif traj_type == 'circular':
        return CircularTrajectory(Ts=Ts)
    elif traj_type in ('spiral', 'random_waypoint', 'square'):
        family_seeds = {'spiral': 0, 'random_waypoint': 500, 'square': 1500}
        base = family_seeds.get(traj_type, 0)
        for s in range(2000):
            traj = RandomizedTrajectory(Ts=Ts, seed=base + s)
            if traj.family == traj_type:
                return traj
        raise RuntimeError(f'无法创建 {traj_type} 轨迹')
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")


# ============================================================================
# 核心仿真
# ============================================================================

def run_simulation(attack_type: str = 'A1',
                   use_detector: bool = True,
                   trajectory_type: str = 'lissajous',
                   seed: int = 42,
                   attack_onset: float = None,
                   model_path: str = None,
                   norm_path: str = None,
                   show_progress: bool = True) -> dict:
    """运行单次闭环仿真

    Args:
        attack_type:     攻击类型 'A0'~'A7' (A0=无攻击)
        use_detector:    True=CFMDetectorBackend, False=y_meas 直送 NMPC
        trajectory_type: 轨迹类型 lissajous/circular/spiral/random_waypoint/square
        seed:            随机种子
        attack_onset:    攻击起始时间 [s] (None=随机[5,30]s, A0=永不攻击)
        model_path:      CFM 模型权重路径
        norm_path:       归一化参数路径

    Returns:
        data: 包含所有时序信号的字典
    """
    # ---- 初始化 ----
    wmr_params = WMRParams()
    Ts = wmr_params.Ts
    n_steps = SIM_STEPS  # 1000

    # 轨迹
    traj = _create_trajectory(trajectory_type, Ts, seed)

    # 攻击起始时间
    if attack_onset is None:
        rng = np.random.RandomState(seed)
        attack_onset = float(rng.uniform(ATTACK_ONSET_MIN, ATTACK_ONSET_MAX))
    if attack_type == 'A0':
        attack_onset = SIM_TIME + 1.0  # 永不攻击

    robot = WMRKinematics(wmr_params)
    sensor = SensorSimulator()
    ctrl = NMPCController(NMPCParams())
    ctrl.load_or_build()

    attacker = SensorAttack(attack_type=attack_type,
                            onset_time=attack_onset, seed=seed)

    # 检测器
    detector = None
    if use_detector:
        detector = CFMDetectorBackend(model_path=model_path, norm_path=norm_path)

    # ---- 重置 ----
    traj.reset()
    # 机器人初始位姿 = 参考轨迹起点 + 随机偏移 (检验 NMPC 收敛能力)
    init_rng = np.random.RandomState(seed)
    init_state = np.array([traj._x_r, traj._y_r, traj._theta_r])
    init_state[0] += init_rng.uniform(0.0, INIT_POS_MAX) * init_rng.choice([-1, 1])
    init_state[1] += init_rng.uniform(0.0, INIT_POS_MAX) * init_rng.choice([-1, 1])
    init_state[2] += init_rng.uniform(0.0, INIT_HEADING_MAX) * init_rng.choice([-1, 1])
    robot.reset(init_state)
    ctrl.reset()
    attacker.reset()
    if detector is not None:
        detector.reset()
    np.random.seed(seed)

    # ---- 仿真循环 ----
    data = defaultdict(list)
    u_cmd = np.zeros(2)

    for step in range(n_steps):
        t = step * Ts

        # 1. 参考轨迹
        Upsilon_r, u_r = traj.step(t)
        Ur_seq = traj.generate_sequence(t, NMPCParams().N)

        # 2. 传感器测量 (含攻击)
        true_state = robot.state.copy()
        noise = sensor.noise_std * np.random.randn(3)
        y_clean = true_state + noise
        y_meas = attacker.inject(t, y_clean)
        attack_signal = y_meas - y_clean

        # 3. 检测器处理
        if detector is None:
            y_rec = y_meas.copy()
            det_class = 'A0'
            det_conf = 0.0
            det_attack_est = np.zeros(3)
        else:
            result = detector.detect(y_meas)
            y_rec = result.y_recovered
            det_class = result.attack_class
            det_conf = result.confidence
            det_attack_est = result.attack_estimate

        # 4. 状态估计 = 恢复后的测量 (替代 EKF)
        # y_rec 直接作为位姿估计，送入 NMPC

        # 5. 跟踪误差
        X_error = WMRKinematics.compute_error(Upsilon_r, y_rec)

        # 6. NMPC 控制
        u_cmd = ctrl.solve(X_error, Ur_seq)

        # 7. 通知检测器控制指令
        if detector is not None:
            detector.set_control(u_cmd)

        # 8. 限幅
        u_a = WMRKinematics.clamp_control(u_cmd)

        # 9. 机器人状态更新
        robot.step(u_a)

        # 10. 记录
        data['t'].append(t)
        data['Upsilon_r'].append(Upsilon_r.copy())
        data['true_state'].append(true_state.copy())
        data['y_meas'].append(y_meas.copy())
        data['y_rec'].append(y_rec.copy())
        data['attack_signal'].append(attack_signal.copy())
        data['Upsilon_hat'].append(y_rec.copy())
        data['X_error'].append(X_error.copy())
        data['u_cmd'].append(u_cmd.copy())
        data['u_a'].append(u_a.copy())
        data['det_class'].append(det_class)
        data['det_conf'].append(det_conf)
        data['det_attack_est'].append(det_attack_est.copy())
        data['attack_active'].append(1.0 if (t >= attack_onset and attack_type != 'A0') else 0.0)

        # 进度
        if show_progress and (step % 200 == 0 or step == n_steps - 1):
            pos_err = np.linalg.norm(X_error[:2])
            det_str = 'CFM' if use_detector else 'NONE'
            print(f"  [{det_str:3s}] t={t:5.1f}s | "
                  f"|e_xy|={pos_err:.4f}m | "
                  f"det={det_class} conf={det_conf:.2f}", flush=True)

    # 转数组
    result = {}
    for k, v in data.items():
        try:
            result[k] = np.array(v, dtype=float)
        except ValueError:
            result[k] = np.array(v, dtype=object)

    result['Ts'] = Ts
    result['attack_type'] = attack_type
    result['use_detector'] = use_detector
    result['trajectory_type'] = trajectory_type
    result['attack_onset'] = attack_onset
    return result


# ============================================================================
# 指标计算
# ============================================================================

def compute_metrics(data: dict) -> dict:
    """从仿真数据计算定量指标"""
    onset_idx = _get_onset_idx(data)
    X_err = data['X_error']
    pos_err = np.sqrt(X_err[:, 0] ** 2 + X_err[:, 1] ** 2)
    ang_err = np.abs(X_err[:, 2])

    steady_start = max(int(5.0 / data['Ts']), 100)
    steady_end = onset_idx

    post_pos = pos_err[onset_idx:] if onset_idx < len(pos_err) else pos_err[-100:]

    return {
        'attack_type': data['attack_type'],
        'trajectory_type': data.get('trajectory_type', 'lissajous'),
        'attack_onset': float(data.get('attack_onset', ATTACK_ONSET_DEFAULT)),
        'steady_pos_rmse': float(np.sqrt(np.mean(pos_err[steady_start:steady_end] ** 2)))
                          if steady_end > steady_start else 0.0,
        'post_pos_rmse': float(np.sqrt(np.mean(post_pos ** 2))) if len(post_pos) > 0 else 0.0,
        'post_pos_max': float(np.max(post_pos)) if len(post_pos) > 0 else 0.0,
        'post_ang_rmse': float(np.sqrt(np.mean(ang_err[onset_idx:] ** 2)))
                        if onset_idx < len(ang_err) else 0.0,
        'use_detector': data.get('use_detector', True),
    }


def compute_detector_metrics(data: dict) -> dict:
    """计算检测器 DL 指标"""
    onset_idx = _get_onset_idx(data)
    attack_type = data['attack_type']
    det_classes = data['det_class']
    det_confs = data['det_conf']

    post_classes = det_classes[onset_idx:]
    post_confs = det_confs[onset_idx:]

    if len(post_classes) == 0:
        return {'detection_accuracy': 0.0, 'mean_confidence': 0.0,
                'detection_latency_steps': 999, 'detection_latency_sec': 99.0,
                'false_alarm_rate': 0.0, 'recovery_rmse': 0.0}

    correct = sum(1 for c in post_classes if str(c) == attack_type)
    accuracy = correct / len(post_classes)
    mean_conf = np.mean([float(c) for c in post_confs])

    latency_steps = len(post_classes)
    for i, c in enumerate(post_classes):
        if str(c) == attack_type and i < len(post_confs) and float(post_confs[i]) > 0.5:
            latency_steps = i
            break

    pre_start = 100
    pre_classes = det_classes[pre_start:onset_idx]
    if len(pre_classes) > 0:
        false_alarms = sum(1 for c in pre_classes if str(c) != 'A0')
        far = false_alarms / len(pre_classes)
    else:
        far = 0.0

    y_rec_arr = data['y_rec'][onset_idx:]
    true_arr = data['true_state'][onset_idx:]
    recovery_rmse = float(np.sqrt(np.mean(np.sum((y_rec_arr - true_arr) ** 2, axis=1))))

    return {
        'detection_accuracy': accuracy,
        'mean_confidence': mean_conf,
        'detection_latency_steps': latency_steps,
        'detection_latency_sec': latency_steps * data['Ts'],
        'false_alarm_rate': far,
        'recovery_rmse': recovery_rmse,
    }


# ============================================================================
# 可视化
# ============================================================================

def plot_results(data: dict, save_path: str = None):
    """绘制单次仿真结果 (2x2 面板, IEEE 论文标准)

    自适应布局:
      - 上排 (恒定): 2D 轨迹 + 位置跟踪误差
      - 下排 (含检测器): 攻击估计 + 检测器输出
      - 下排 (无检测器): 分轴跟踪误差 + NMPC 控制指令
    A0 / 无攻击场景不绘制任何攻击起始标记。
    """
    t = data['t']
    Ts = data['Ts']
    atk_type = data.get('attack_type', 'A0')
    use_det = data.get('use_detector', True)
    det_label = 'CFMDetector' if use_det else 'No Detector'
    has_attack = _has_attack(data)
    onset_idx = _get_onset_idx(data) if has_attack else None
    onset_time = onset_idx * Ts if has_attack else None

    ref = data['Upsilon_r']
    true_state = data['true_state']
    est = data['Upsilon_hat']
    X_err = data['X_error']
    pos_err = np.sqrt(X_err[:, 0] ** 2 + X_err[:, 1] ** 2)
    b = WMRParams().pos_bound

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'WMR Closed-Loop Tracking — Attack {atk_type} '
                 f'({ATTACK_NAMES.get(atk_type, "")}) — {det_label}',
                 fontsize=14, fontweight='bold')

    # ---- Panel 1: 2D 轨迹 ----
    ax = axes[0, 0]
    ax.plot(ref[:, 0], ref[:, 1], 'k--', linewidth=1.5, alpha=0.6, label='Reference')
    ax.plot(true_state[:, 0], true_state[:, 1], color='#1f77b4', linewidth=1.6,
            alpha=0.9, label='Robot (actual)')
    if use_det:
        ax.plot(est[:, 0], est[:, 1], color='#9467bd', linewidth=1.0, alpha=0.6,
                label='Estimate (recovered)')
    # 起点 / 终点标记
    ax.scatter(true_state[0, 0], true_state[0, 1], c='#2ca02c', s=110, marker='o',
               zorder=6, edgecolors='white', linewidths=1.3, label='Start')
    ax.scatter(true_state[-1, 0], true_state[-1, 1], c='#d62728', s=200, marker='*',
               zorder=6, edgecolors='white', linewidths=1.0, label='End')
    if has_attack:
        ax.scatter(true_state[onset_idx, 0], true_state[onset_idx, 1], c='red', s=120,
                   marker='X', zorder=7, edgecolors='white', linewidths=1.0,
                   label='Attack onset')
    # 空间安全边界
    ax.plot([-b, b, b, -b, -b], [-b, -b, b, b, -b], color='gray',
            linewidth=0.8, linestyle=':', alpha=0.5)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title('2D Trajectory')
    ax.legend(fontsize=7, loc='best'); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # ---- Panel 2: 位置跟踪误差 ----
    ax = axes[0, 1]
    ax.plot(t, pos_err, color='#1f77b4', linewidth=1.3, alpha=0.9)
    if has_attack:
        ax.axvline(x=onset_time, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label='Attack onset')
        ax.legend(fontsize=8)
        rmse = np.sqrt(np.mean(pos_err[onset_idx:] ** 2))
        box_txt = f'Post-attack RMSE: {rmse:.4f} m'
    else:
        steady = max(int(5.0 / Ts), 100)
        rmse = np.sqrt(np.mean(pos_err[steady:] ** 2))
        box_txt = f'Steady-state RMSE: {rmse:.4f} m'
    ax.text(0.02, 0.95, box_txt, transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Position error [m]')
    ax.set_title(r'Tracking Error $\|e_{xy}\|$')
    ax.grid(True, alpha=0.3)

    if use_det:
        # ---- Panel 3: 检测器攻击估计 ----
        ax = axes[1, 0]
        atk_norm = np.linalg.norm(data['det_attack_est'], axis=1)
        ax.plot(t, atk_norm, color='#d62728', linewidth=1.0, alpha=0.85)
        if has_attack:
            ax.axvline(x=onset_time, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time [s]'); ax.set_ylabel(r'$\|\hat{a}\|$')
        ax.set_title('Detected Attack Signal')
        ax.grid(True, alpha=0.3)

        # ---- Panel 4: 检测器输出 ----
        ax = axes[1, 1]
        det_classes = data['det_class']
        det_confs = data['det_conf']
        cls_ints = np.array([int(str(c)[1:]) if str(c).startswith('A') else 0
                             for c in det_classes])
        ax.fill_between(t, 0, cls_ints, alpha=0.3, color='#9467bd', step='post')
        ax.set_ylim(-0.5, 7.5); ax.set_yticks(range(8))
        ax.set_yticklabels([f'A{i}' for i in range(8)])
        if has_attack:
            ax.axvline(x=onset_time, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(t, det_confs, color='#1f77b4', linewidth=1.0, alpha=0.7)
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax2.set_ylim(0, 1.05); ax2.set_ylabel('Confidence')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Detected class')
        ax.set_title('Detector Output (Class + Confidence)')
        ax.grid(True, alpha=0.3)
    else:
        # ---- Panel 3: 分轴跟踪误差 (本体坐标系) ----
        ax = axes[1, 0]
        ax.plot(t, X_err[:, 0], color='#1f77b4', linewidth=1.1, label=r'$x_e$')
        ax.plot(t, X_err[:, 1], color='#ff7f0e', linewidth=1.1, label=r'$y_e$')
        ax.plot(t, X_err[:, 2], color='#2ca02c', linewidth=1.1, alpha=0.85,
                label=r'$\theta_e$')
        ax.axhline(y=0.0, color='gray', linewidth=0.6, alpha=0.5)
        if has_attack:
            ax.axvline(x=onset_time, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Error')
        ax.set_title('Per-Axis Tracking Error (body frame)')
        ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.3)

        # ---- Panel 4: NMPC 控制指令 ----
        ax = axes[1, 1]
        u = data['u_a']
        ax.plot(t, u[:, 0], color='#1f77b4', linewidth=1.1, label=r'$v$ [m/s]')
        ax.plot(t, u[:, 1], color='#ff7f0e', linewidth=1.1, label=r'$\omega$ [rad/s]')
        for lim, c in [(0.3, '#1f77b4'), (1.76, '#ff7f0e')]:
            ax.axhline(y=lim, color=c, linestyle=':', linewidth=0.8, alpha=0.5)
            ax.axhline(y=-lim, color=c, linestyle=':', linewidth=0.8, alpha=0.5)
        if has_attack:
            ax.axvline(x=onset_time, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Control input')
        ax.set_title('NMPC Control Commands')
        ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        fname = f'sim_{atk_type}_{data.get("trajectory_type", "lissajous")}.png'
        save_path = os.path.join(RESULT_DIR, fname)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot] {save_path}")
    return save_path


def plot_summary(all_metrics: list):
    """批量运行汇总图: 所有攻击类型的 RMSE + 检测准确率"""
    df = pd.DataFrame(all_metrics)
    attacks = sorted(df['attack_type'].unique())
    x = np.arange(len(attacks))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('CFMDetector — All Attack Types Summary', fontsize=14, fontweight='bold')

    # Panel 1: Post-Attack RMSE
    ax = axes[0]
    rmse_vals = [df[df['attack_type'] == a]['post_pos_rmse'].values[0]
                 if len(df[df['attack_type'] == a]) else 0 for a in attacks]
    colors = ['#4CAF50' if v < 0.05 else '#FF9800' if v < 0.2 else '#F44336' for v in rmse_vals]
    bars = ax.bar(x, rmse_vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(attacks)
    ax.set_ylabel('Position RMSE [m]')
    ax.set_title('Post-Attack Position RMSE')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Detection Accuracy (if available)
    ax = axes[1]
    if 'detection_accuracy' in df.columns:
        acc_vals = [df[df['attack_type'] == a]['detection_accuracy'].values[0]
                    if len(df[df['attack_type'] == a]) else 0 for a in attacks]
        acc_pct = [v * 100 for v in acc_vals]
        colors_acc = ['#4CAF50' if v >= 70 else '#FF9800' if v >= 40 else '#F44336' for v in acc_pct]
        bars = ax.bar(x, acc_pct, color=colors_acc, alpha=0.85)
        for bar, val in zip(bars, acc_pct):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_ylabel('Accuracy [%]')
    else:
        ax.text(0.5, 0.5, 'No detector metrics available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(attacks)
    ax.set_title('Detection Accuracy')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(RESULT_DIR, 'sim_summary.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot] {filepath}")
    return filepath


# ============================================================================
# 五族轨迹无攻击对比 (保留自旧 simulate.py)
# ============================================================================

def run_single_track(traj, robot, sensor, ctrl, nmpc_params, Ts, n_steps):
    """用给定的轨迹生成器执行一次无攻击闭环仿真。y_meas 直接作为位姿估计送入 NMPC。"""
    data = defaultdict(list)
    u_cmd = np.zeros(2)

    for step in range(n_steps):
        t = step * Ts
        Upsilon_r, u_r = traj.step(t)
        Ur_seq = traj.generate_sequence(t, nmpc_params.N)

        true_state = robot.state.copy()
        y_meas = sensor.measure(true_state)
        X_error = WMRKinematics.compute_error(Upsilon_r, y_meas)
        u_cmd = ctrl.solve(X_error, Ur_seq)
        u_a = WMRKinematics.clamp_control(u_cmd)
        robot.step(u_a)

        data['t'].append(t)
        data['Upsilon_r'].append(Upsilon_r.copy())
        data['true_state'].append(true_state.copy())
        data['Upsilon_hat'].append(y_meas.copy())
        data['X_error'].append(X_error.copy())
        data['u_cmd'].append(u_cmd.copy())

    return {k: np.array(v) for k, v in data.items()}


def plot_all_trajectories():
    """五族轨迹无攻击跟踪对比图"""
    from generate_dataset import RandomizedTrajectory

    wmr_params = WMRParams()
    Ts = wmr_params.Ts
    n_steps = SIM_STEPS
    nmpc_params = NMPCParams()

    def _force_family(family: str, ts: float):
        family_seeds = {'spiral': 0, 'random_waypoint': 500, 'square': 1500}
        base = family_seeds.get(family, 0)
        for s in range(2000):
            traj = RandomizedTrajectory(Ts=ts, seed=base + s)
            if traj.family == family:
                return traj
        raise RuntimeError(f'无法创建 {family} 轨迹')

    TRAJ_CONFIGS = [
        ('Lissajous',       '#2196F3', lambda: LissajousTrajectory(Ts=Ts)),
        ('Circular',        '#4CAF50', lambda: CircularTrajectory(Ts=Ts)),
        ('Spiral',          '#FF9800', lambda: _force_family('spiral', Ts)),
        ('Random Waypoint', '#E91E63', lambda: _force_family('random_waypoint', Ts)),
        ('Square',          '#9C27B0', lambda: _force_family('square', Ts)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes_flat = axes.flatten()
    all_rms = {}

    for idx, (name, color, factory) in enumerate(TRAJ_CONFIGS):
        robot = WMRKinematics(wmr_params)
        sensor = SensorSimulator()
        ctrl = NMPCController(nmpc_params)
        ctrl.load_or_build()
        traj = factory()

        traj.reset()
        # 机器人初始位姿 = 参考轨迹起点 + 随机偏移
        init_rng = np.random.RandomState(42 + idx)
        init_state = np.array([traj._x_r, traj._y_r, traj._theta_r])
        init_state[0] += init_rng.uniform(0.0, INIT_POS_MAX) * init_rng.choice([-1, 1])
        init_state[1] += init_rng.uniform(0.0, INIT_POS_MAX) * init_rng.choice([-1, 1])
        init_state[2] += init_rng.uniform(0.0, INIT_HEADING_MAX) * init_rng.choice([-1, 1])
        robot.reset(init_state); ctrl.reset()
        np.random.seed(42)

        print(f"[{idx+1}/5] 仿真 {name} ...", end='', flush=True)
        data = run_single_track(traj, robot, sensor, ctrl, nmpc_params, Ts, n_steps)

        Ur = data['Upsilon_r']
        TrueState = data['true_state']
        X_err = data['X_error']
        pos_err = np.sqrt(X_err[:, 0] ** 2 + X_err[:, 1] ** 2)
        rms_xy = np.sqrt(np.mean(pos_err ** 2))
        all_rms[name] = rms_xy

        ax = axes_flat[idx]
        ax.plot(Ur[:, 0], Ur[:, 1], 'k--', linewidth=1.2, alpha=0.7, label='Ref')
        ax.plot(TrueState[:, 0], TrueState[:, 1], color=color, linewidth=1.4, label='Track')
        ax.plot(TrueState[0, 0], TrueState[0, 1], 'go', markersize=6)
        ax.plot(TrueState[-1, 0], TrueState[-1, 1], 'r*', markersize=8)
        b = wmr_params.pos_bound
        ax.plot([-b, b, b, -b, -b], [-b, -b, b, b, -b], 'gray', linewidth=0.8, alpha=0.4, linestyle=':')
        ax.set_xlim(-b - 0.3, b + 0.3); ax.set_ylim(-b - 0.3, b + 0.3)
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
        ax.set_title(f'{name}  (RMS$_{{xy}}$={rms_xy:.4f} m)', fontsize=12, fontweight='bold', color=color)
        ax.set_aspect('equal'); ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.25)
        print(f' RMS_xy={rms_xy:.4f}m')

    # 汇总柱状图
    ax6 = axes_flat[5]
    names = list(all_rms.keys())
    values = list(all_rms.values())
    colors_bar = [cfg[1] for cfg in TRAJ_CONFIGS]
    bars = ax6.bar(names, values, color=colors_bar, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Position RMS$_{xy}$ [m]')
    ax6.set_title('Tracking Error Comparison', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim(0, max(values) * 1.25)

    fig.suptitle('Five Trajectory Families — No-Attack NMPC Tracking (50s)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============================================================================
# 批量运行
# ============================================================================

def run_batch(attack_types: list = None,
              trajectory_type: str = 'lissajous',
              seed: int = 42,
              use_detector: bool = True,
              randomize_onset: bool = True,
              do_plot: bool = True,
              model_path: str = None,
              norm_path: str = None):
    """批量运行所有攻击类型的仿真"""
    if attack_types is None:
        attack_types = ALL_ATTACK_TYPES

    all_metrics = []
    rng = np.random.RandomState(seed)

    for atk in attack_types:
        if randomize_onset and atk != 'A0':
            attack_onset = float(rng.uniform(ATTACK_ONSET_MIN, ATTACK_ONSET_MAX))
        elif atk == 'A0':
            attack_onset = SIM_TIME + 1.0
        else:
            attack_onset = ATTACK_ONSET_DEFAULT

        print(f"\n{'#'*60}")
        print(f"# Attack: {atk} ({ATTACK_NAMES[atk]}) @ onset={attack_onset:.1f}s")
        print(f"{'#'*60}")

        data = run_simulation(
            attack_type=atk, use_detector=use_detector,
            trajectory_type=trajectory_type, seed=seed,
            attack_onset=attack_onset,
            model_path=model_path, norm_path=norm_path
        )

        # 保存 NPZ
        det_str = 'cfm' if use_detector else 'none'
        fname = f'sim_{atk}_{trajectory_type}_{det_str}.npz'
        save_dict = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                save_dict[k] = v
            elif isinstance(v, str):
                save_dict[k] = np.array(v)
        np.savez_compressed(os.path.join(RESULT_DIR, fname), **save_dict)

        # 指标
        metrics = compute_metrics(data)
        if use_detector:
            metrics.update(compute_detector_metrics(data))
        all_metrics.append(metrics)

        print(f"  Post-Attack RMSE: {metrics['post_pos_rmse']:.4f}m", end='')
        if use_detector and 'detection_accuracy' in metrics:
            print(f" | Det Acc: {metrics['detection_accuracy']:.1%}", end='')
        print()

        if do_plot:
            plot_results(data)

    # 保存指标 CSV
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        csv_path = os.path.join(RESULT_DIR, 'sim_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n  [CSV] {csv_path}")

    # 汇总图
    if do_plot and all_metrics:
        plot_summary(all_metrics)

    return all_metrics


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='WMR Closed-Loop Simulation with CFM Detector')

    parser.add_argument('--attack', type=str, default='A1',
                        help='Attack type A0-A7 (default: A1)')
    parser.add_argument('--all', action='store_true',
                        help='Run all 8 attack types')
    parser.add_argument('--no-detector', action='store_true',
                        help='Disable CFM detector (direct y_meas to NMPC)')
    parser.add_argument('--trajectory', type=str, default='lissajous',
                        choices=TRAJECTORY_FAMILIES,
                        help='Reference trajectory type (default: lissajous)')
    parser.add_argument('--compare', action='store_true',
                        help='Five-family no-attack tracking comparison')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-randomize-onset', action='store_true',
                        help='Use fixed onset (15s) instead of randomized [5,30]s')
    parser.add_argument('--model-path', type=str, default=None,
                        help='CFM model weights path')
    parser.add_argument('--norm-path', type=str, default=None,
                        help='Normalizer params path')

    args = parser.parse_args()

    use_detector = not args.no_detector
    randomize_onset = not args.no_randomize_onset

    print("=" * 60)
    print("WMR Closed-Loop Simulation")
    print("=" * 60)
    print(f"  Sim Time: {SIM_TIME}s")
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Detector: {'CFM' if use_detector else 'None (baseline)'}")
    print(f"  Seed: {args.seed}")

    # 五族轨迹对比
    if args.compare:
        print(f"  Mode: COMPARE (5-family no-attack)")
        plot_all_trajectories()
        return

    # 批量模式
    if args.all:
        print(f"  Mode: BATCH (all attacks)")
        print(f"  Attack Onset: {'randomized [5,30]s' if randomize_onset else f'{ATTACK_ONSET_DEFAULT}s (fixed)'}")
        run_batch(trajectory_type=args.trajectory, seed=args.seed,
                  use_detector=use_detector, randomize_onset=randomize_onset,
                  do_plot=not args.no_plot,
                  model_path=args.model_path, norm_path=args.norm_path)
        return

    # 单次模式
    print(f"  Mode: SINGLE (attack={args.attack})")

    if randomize_onset and args.attack != 'A0':
        rng = np.random.RandomState(args.seed)
        attack_onset = float(rng.uniform(ATTACK_ONSET_MIN, ATTACK_ONSET_MAX))
        print(f"  Attack Onset: {attack_onset:.1f}s (randomized)")
    elif args.attack == 'A0':
        attack_onset = SIM_TIME + 1.0
        print(f"  Attack Onset: never (A0)")
    else:
        attack_onset = ATTACK_ONSET_DEFAULT
        print(f"  Attack Onset: {attack_onset}s (fixed)")

    data = run_simulation(
        attack_type=args.attack, use_detector=use_detector,
        trajectory_type=args.trajectory, seed=args.seed,
        attack_onset=attack_onset,
        model_path=args.model_path, norm_path=args.norm_path
    )

    # 指标
    X_err = data['X_error']
    if _has_attack(data):
        onset_idx = _get_onset_idx(data)
        seg = np.sqrt(X_err[onset_idx:, 0] ** 2 + X_err[onset_idx:, 1] ** 2)
        rmse_label = 'Post-attack RMSE'
    else:
        steady = max(int(5.0 / data['Ts']), 100)
        seg = np.sqrt(X_err[steady:, 0] ** 2 + X_err[steady:, 1] ** 2)
        rmse_label = 'Steady-state RMSE'
    rmse = np.sqrt(np.mean(seg ** 2))
    max_err = np.max(seg)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Detector: {'CFM' if use_detector else 'None'}")
    print(f"  {rmse_label}: {rmse:.4f}m, Max: {max_err:.4f}m")

    if use_detector:
        det_m = compute_detector_metrics(data)
        print(f"  Detection Acc: {det_m['detection_accuracy']:.1%}, "
              f"Latency: {det_m['detection_latency_sec']:.2f}s, "
              f"Recovery RMSE: {det_m['recovery_rmse']:.4f}m")

    # 保存 NPZ
    det_str = 'cfm' if use_detector else 'none'
    fname = f'sim_{args.attack}_{args.trajectory}_{det_str}.npz'
    save_dict = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = v
        elif isinstance(v, str):
            save_dict[k] = np.array(v)
    np.savez_compressed(os.path.join(RESULT_DIR, fname), **save_dict)
    print(f"  NPZ saved: {fname}")

    if not args.no_plot and not args.all:
        plot_results(data)

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
