"""
simulate_detector.py — 多级检测器对比仿真
============================================================================
对比不同检测器级别在传感器攻击下的跟踪精度恢复能力。

检测器分级:
  Tier 0 (none)  : 无检测 — y_meas 直接入 EKF (最差基线)
  Tier 1 (cfm)   : CFMDetector — PINN-Flow 流匹配 + Transformer 统一架构
  Tier 2 (oracle): OracleDetector — 已知 ground truth a(k)，理论上界

轨迹类型 (5族，与训练数据一致):
  lissajous / circular / spiral / random_waypoint / square

攻击起始时间: 随机 [5, 30]s (匹配训练数据分布)

用法:
  python simulate_detector.py --attack A4
  python simulate_detector.py --attack A4 --tiers none,nn,oracle
  python simulate_detector.py --all
  python simulate_detector.py --all --trajectory circular
"""

import os
import sys
import time
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

from model import (WMRParams, WMRKinematics, EKFEstimator, SensorSimulator,
                   LissajousTrajectory, CircularTrajectory, SIM_STEPS)
from controller import NMPCController, NMPCParams
from attack import SensorAttack, AttackConfig
from detector import (OracleDetector,
                      DetectionResult, create_detector)

# ============================================================================
# 全局配置
# ============================================================================

SIM_TIME = 35.0
ATTACK_ONSET = 15.0              # 默认攻击起始 (--randomize-onset 时随机化)
ATTACK_ONSET_MIN = 5.0           # 随机起始最小值
ATTACK_ONSET_MAX = 30.0          # 随机起始最大值
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

TIER_COLORS = {
    'none':   '#d62728',   # 红色 — 最差基线
    'cfm':    '#9467bd',   # 紫色 — PINN-Flow CFM
    'oracle': '#1f77b4',   # 蓝色 — 理论上界
}
TIER_LABELS = {
    'none':   'Tier 0: No Detector',
    'cfm':    'Tier 1: CFMDetector (PINN-Flow)',
    'oracle': 'Tier 2: Oracle (Upper Bound)',
}
TIER_LINESTYLE = {
    'none':   '--',
    'cfm':    ':',
    'oracle': '-.',
}


# ============================================================================
# 核心仿真函数
# ============================================================================

def run_simulation(attack_type: str = 'A4',
                   detector_tier: str = 'none',
                   trajectory_type: str = 'lissajous',
                   seed: int = 42,
                   attack_onset: float = None,
                   model_path: str = None,
                   norm_path: str = None) -> dict:
    """运行单次闭环仿真

    Args:
        attack_type:     攻击类型 'A0'~'A8'
        detector_tier:   检测器级别 'none'/'cfm'/'oracle'
        trajectory_type: 轨迹类型 (5族: lissajous/circular/spiral/random_waypoint/square)
        seed:            随机种子
        attack_onset:    攻击开始时间 [s] (None=默认15s, A0=永不攻击)

    Returns:
        data: 包含所有时序信号的字典
    """
    # ---- 初始化 ----
    wmr_params = WMRParams()
    Ts = wmr_params.Ts
    n_steps = SIM_STEPS  # 700

    # 轨迹: 支持 5 种类型
    traj = _create_trajectory(trajectory_type, Ts, seed)

    # 攻击起始时间
    if attack_onset is None:
        attack_onset = ATTACK_ONSET
    if attack_type == 'A0':
        attack_onset = SIM_TIME + 1.0  # 永不攻击

    robot = WMRKinematics(wmr_params)
    sensor = SensorSimulator()
    ekf = EKFEstimator(wmr_params)
    ctrl = NMPCController(NMPCParams())
    ctrl.load_or_build()

    attacker = SensorAttack(attack_type=attack_type,
                            onset_time=attack_onset, seed=seed)
    detector = create_detector(detector_tier, attack_type=attack_type, seed=seed,
                              model_path=model_path, norm_path=norm_path)

    # ---- 重置 ----
    traj.reset()
    robot.reset()
    ekf.reset()
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
        if detector_tier == 'none' or detector is None:
            y_ekf = y_meas.copy()
            det_class = 'A0'
            det_conf = 0.0
            det_attack_est = np.zeros(3)
        elif detector_tier == 'oracle':
            result = detector.detect(y_meas, a_true=attack_signal)
            y_ekf = result.y_recovered
            det_class = result.attack_class
            det_conf = result.confidence
            det_attack_est = result.attack_estimate
        else:
            # nn: 标准 detect() 接口
            result = detector.detect(y_meas)
            y_ekf = result.y_recovered
            det_class = result.attack_class
            det_conf = result.confidence
            det_attack_est = result.attack_estimate

        # 4. EKF 状态估计
        Upsilon_hat, ekf_innovation = ekf.step(y_ekf, u_cmd)

        # 5. 通知检测器 EKF 状态 (供周期性内部状态重校准)
        if detector is not None and hasattr(detector, 'set_ekf_state'):
            detector.set_ekf_state(Upsilon_hat)

        # 6. 跟踪误差
        X_error = WMRKinematics.compute_error(Upsilon_r, Upsilon_hat)

        # 7. NMPC 控制
        u_cmd = ctrl.solve(X_error, Ur_seq)

        # 8. 通知检测器控制指令
        if detector is not None and hasattr(detector, 'set_control'):
            detector.set_control(u_cmd)

        # 9. 限幅
        u_a = WMRKinematics.clamp_control(u_cmd)

        # 10. 机器人状态更新
        robot.step(u_a)

        # 11. 记录
        data['t'].append(t)
        data['Upsilon_r'].append(Upsilon_r.copy())
        data['true_state'].append(true_state.copy())
        data['y_meas'].append(y_meas.copy())
        data['y_ekf'].append(y_ekf.copy())
        data['attack_signal'].append(attack_signal.copy())
        data['Upsilon_hat'].append(Upsilon_hat.copy())
        data['ekf_innovation'].append(ekf_innovation.copy())
        data['X_error'].append(X_error.copy())
        data['u_cmd'].append(u_cmd.copy())
        data['u_a'].append(u_a.copy())
        data['det_class'].append(det_class)
        data['det_conf'].append(det_conf)
        data['det_attack_est'].append(det_attack_est.copy())
        data['attack_active'].append(1.0 if (t >= attack_onset and attack_type != 'A0') else 0.0)

        # 进度
        if step % 200 == 0 or step == n_steps - 1:
            pos_err = np.linalg.norm(X_error[:2])
            print(f"  [{detector_tier.upper():6s}] t={t:5.1f}s | "
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
    result['detector_tier'] = detector_tier
    result['trajectory_type'] = trajectory_type
    result['attack_onset'] = attack_onset
    return result


def _create_trajectory(traj_type: str, Ts: float, seed: int):
    """创建轨迹生成器 (支持 5 种类型)"""
    from generate_dataset import RandomizedTrajectory

    if traj_type in ('lissajous',):
        return LissajousTrajectory(Ts=Ts)
    elif traj_type in ('circular',):
        return CircularTrajectory(Ts=Ts)
    elif traj_type in ('spiral', 'random_waypoint', 'square'):
        # 使用 RandomizedTrajectory 的特定族
        traj = RandomizedTrajectory(Ts=Ts, seed=seed)
        # 强制选择指定族 (通过反复创建直到匹配)
        for _ in range(100):
            if traj.family == traj_type:
                break
            traj = RandomizedTrajectory(Ts=Ts, seed=seed + _ + 1)
        return traj
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")


# ============================================================================
# 多级对比运行
# ============================================================================

def run_comparison(attack_type: str = 'A4',
                   tiers: list = None,
                   trajectory_type: str = 'lissajous',
                   seed: int = 42,
                   attack_onset: float = None,
                   model_path: str = None,
                   norm_path: str = None) -> dict:
    """运行多个检测器级别的对比仿真

    Returns:
        {tier_name: data_dict} 字典
    """
    if tiers is None:
        tiers = ['none', 'cfm', 'oracle']

    onset = attack_onset if attack_onset is not None else ATTACK_ONSET
    print(f"\n{'='*60}")
    print(f"Attack: {attack_type} | Trajectory: {trajectory_type} | Onset: {onset}s")
    print(f"Tiers: {tiers}")
    print(f"{'='*60}")

    results = {}
    for tier in tiers:
        print(f"\n--- {TIER_LABELS[tier]} ---")
        t0 = time.time()
        results[tier] = run_simulation(
            attack_type=attack_type,
            detector_tier=tier,
            trajectory_type=trajectory_type,
            seed=seed,
            attack_onset=attack_onset,
            model_path=model_path,
            norm_path=norm_path
        )
        elapsed = time.time() - t0
        # 攻击后位置 RMSE
        data = results[tier]
        onset_idx = _get_onset_idx(data)
        X_err = data['X_error']
        pos_err = np.sqrt(X_err[onset_idx:, 0]**2 + X_err[onset_idx:, 1]**2)
        rmse = np.sqrt(np.mean(pos_err**2))
        print(f"  -> post-attack pos RMSE: {rmse:.4f}m ({elapsed:.1f}s)")

    return results


def _get_onset_idx(data: dict) -> int:
    """从仿真数据中获取攻击起始索引

    优先使用 attack_active 数组, 其次 attack_onset 标量, 最后默认 300 (15s)。
    考虑窗口填充期: 前 100 步 NN 未激活, 取 max(onset_idx, 100)。
    对于 A0 (无攻击), 返回仿真后半作为"post-attack"参考点。
    """
    n_steps = len(data['t']) if 't' in data else 700
    if 'attack_active' in data:
        active = data['attack_active']
        if hasattr(active, 'max') and active.max() > 0.5:
            raw_onset = int(np.argmax(active > 0.5))
            return min(max(raw_onset, 100), n_steps - 1)
    if 'attack_onset' in data:
        raw_onset = int(float(data['attack_onset']) / data.get('Ts', 0.05))
        # A0: onset > sim_time, clamp to last step
        return min(max(raw_onset, 100), n_steps - 1)
    return min(max(int(ATTACK_ONSET / 0.05), 100), n_steps - 1)


# ============================================================================
# 指标计算
# ============================================================================

def compute_metrics(data: dict) -> dict:
    """从仿真数据计算定量指标 (使用动态 onset)"""
    onset_idx = _get_onset_idx(data)
    X_err = data['X_error']
    pos_err = np.sqrt(X_err[:, 0]**2 + X_err[:, 1]**2)
    ang_err = np.abs(X_err[:, 2])

    # 稳态: 窗口填充后 到 攻击前 (min 100步窗口填充)
    steady_start = max(int(5.0 / data['Ts']), 100)  # 5s 后 + 窗口就绪
    steady_end = onset_idx

    # 攻击后指标
    post_pos = pos_err[onset_idx:] if onset_idx < len(pos_err) else pos_err[-100:]
    post_ang = ang_err[onset_idx:] if onset_idx < len(ang_err) else ang_err[-100:]

    metrics = {
        'attack_type': data['attack_type'],
        'detector_tier': data['detector_tier'],
        'trajectory_type': data.get('trajectory_type', 'lissajous'),
        'attack_onset': float(data.get('attack_onset', ATTACK_ONSET)),
        'steady_pos_rmse': float(np.sqrt(np.mean(pos_err[steady_start:steady_end]**2)))
                          if steady_end > steady_start else 0.0,
        'post_pos_rmse': float(np.sqrt(np.mean(post_pos**2))) if len(post_pos) > 0 else 0.0,
        'post_pos_max': float(np.max(post_pos)) if len(post_pos) > 0 else 0.0,
        'post_ang_rmse': float(np.sqrt(np.mean(post_ang**2))) if len(post_ang) > 0 else 0.0,
        'ekf_res_norm_mean': float(np.mean(
            np.linalg.norm(data['ekf_innovation'][onset_idx:], axis=1)))
            if onset_idx < len(data['ekf_innovation']) else 0.0,
        'u_v_rmse': float(np.sqrt(np.mean(data['u_cmd'][onset_idx:, 0]**2)))
                   if onset_idx < len(data['u_cmd']) else 0.0,
        'u_w_rmse': float(np.sqrt(np.mean(data['u_cmd'][onset_idx:, 1]**2)))
                   if onset_idx < len(data['u_cmd']) else 0.0,
    }
    return metrics


def compute_detector_metrics(data: dict) -> dict:
    """计算检测器专用指标 (使用动态 onset)"""
    onset_idx = _get_onset_idx(data)
    det_classes = data['det_class']
    det_confs = data['det_conf']
    attack_type = data['attack_type']

    post_classes = det_classes[onset_idx:]
    post_confs = det_confs[onset_idx:]

    if len(post_classes) == 0:
        return {'detection_accuracy': 0.0, 'mean_confidence': 0.0,
                'detection_latency_steps': 999, 'detection_latency_sec': 99.0,
                'false_alarm_rate': 0.0, 'recovery_rmse': 0.0}

    correct = sum(1 for c in post_classes if str(c) == attack_type)
    accuracy = correct / len(post_classes)
    mean_conf = np.mean([float(c) for c in post_confs])

    # 检测延迟
    latency_steps = len(post_classes)
    for i, c in enumerate(post_classes):
        if str(c) == attack_type and i < len(post_confs) and float(post_confs[i]) > 0.5:
            latency_steps = i
            break
    latency_sec = latency_steps * data['Ts']

    # 虚警率: 窗口填充后(100步)到攻击前
    pre_start = 100  # 跳过窗口填充期
    pre_classes = det_classes[pre_start:onset_idx]
    if len(pre_classes) > 0:
        false_alarms = sum(1 for c in pre_classes if str(c) != 'A0')
        far = false_alarms / len(pre_classes)
    else:
        far = 0.0

    # 恢复质量
    y_ekf_arr = data['y_ekf'][onset_idx:]
    true_arr = data['true_state'][onset_idx:]
    recovery_rmse = float(np.sqrt(np.mean(np.sum((y_ekf_arr - true_arr)**2, axis=1))))

    return {
        'detection_accuracy': accuracy,
        'mean_confidence': mean_conf,
        'detection_latency_steps': latency_steps,
        'detection_latency_sec': latency_sec,
        'false_alarm_rate': far,
        'recovery_rmse': recovery_rmse,
    }


# ============================================================================
# 可视化
# ============================================================================

def plot_comparison(all_data: dict, attack_type: str,
                    trajectory_type: str = 'lissajous'):
    """绘制多级检测器对比图 (四联图)"""
    tiers = list(all_data.keys())
    onset_idx = _get_onset_idx(all_data[tiers[0]])
    attack_onset_time = onset_idx * all_data[tiers[0]]['Ts']
    t = all_data[tiers[0]]['t']

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    traj_label = '8-shaped Lissajous' if trajectory_type == 'lissajous' else 'Circular'
    fig.suptitle(f'Attack {attack_type}: Detector Tier Comparison ({traj_label})',
                 fontsize=14, fontweight='bold')

    # ---- Panel 1: 2D 轨迹 ----
    ax = axes[0, 0]
    ref = all_data[tiers[0]]['Upsilon_r']
    ax.plot(ref[:, 0], ref[:, 1], 'k--', linewidth=1.5, alpha=0.5, label='Reference')
    for tier in tiers:
        Upsilon_hat = all_data[tier]['Upsilon_hat']
        ax.plot(Upsilon_hat[:, 0], Upsilon_hat[:, 1],
                color=TIER_COLORS[tier], linestyle=TIER_LINESTYLE[tier],
                linewidth=1.2, alpha=0.8, label=TIER_LABELS[tier])
    ax.scatter(ref[onset_idx, 0], ref[onset_idx, 1],
               c='red', s=80, marker='x', zorder=5, linewidths=2,
               label='Attack Onset')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title('2D Trajectory (Estimated States)')
    ax.legend(fontsize=7); ax.axis('equal'); ax.grid(True, alpha=0.3)

    # ---- Panel 2: 位置误差时序 ----
    ax = axes[0, 1]
    for tier in tiers:
        X_err = all_data[tier]['X_error']
        pos_err = np.sqrt(X_err[:, 0]**2 + X_err[:, 1]**2)
        ax.plot(t, pos_err,
                color=TIER_COLORS[tier], linestyle=TIER_LINESTYLE[tier],
                linewidth=1.3, alpha=0.85, label=TIER_LABELS[tier])
    ax.axvline(x=attack_onset_time, color='red', linestyle='--',
               linewidth=1.5, alpha=0.7, label='Attack Onset')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Position Error [m]')
    ax.set_title(r'Tracking Error $\|e_{xy}\|$')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ---- Panel 3: EKF 新息范数 ----
    ax = axes[1, 0]
    for tier in tiers:
        ekf_norm = np.linalg.norm(all_data[tier]['ekf_innovation'], axis=1)
        ax.plot(t, ekf_norm,
                color=TIER_COLORS[tier], linestyle=TIER_LINESTYLE[tier],
                linewidth=1.0, alpha=0.8, label=TIER_LABELS[tier])
    ax.axvline(x=ATTACK_ONSET, color='red', linestyle='--',
               linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Innovation Norm')
    ax.set_title('EKF Innovation')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ---- Panel 4: 攻击后 RMSE 柱状图 ----
    ax = axes[1, 1]
    x = np.arange(len(tiers))
    rmse_values = []
    max_values = []
    for tier in tiers:
        X_err = all_data[tier]['X_error']
        pos_err = np.sqrt(X_err[onset_idx:, 0]**2 + X_err[onset_idx:, 1]**2)
        rmse_values.append(np.sqrt(np.mean(pos_err**2)))
        max_values.append(np.max(pos_err))

    w = 0.35
    bars1 = ax.bar(x - w/2, rmse_values, w, label='RMSE [m]',
                   color=[TIER_COLORS[t] for t in tiers], alpha=0.8)
    bars2 = ax.bar(x + w/2, max_values, w, label='Max Error [m]',
                   color=[TIER_COLORS[t] for t in tiers], alpha=0.35,
                   edgecolor=[TIER_COLORS[t] for t in tiers], linewidth=1.5)

    for bar, val in zip(bars1, rmse_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', fontsize=8, fontweight='bold')
    for bar, val in zip(bars2, max_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', fontsize=7, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tiers], fontsize=9)
    ax.set_ylabel('Error [m]')
    ax.set_title('Post-Attack Position Error (t > 15s)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = f'det_comparison_{attack_type}_{trajectory_type}.png'
    filepath = os.path.join(RESULT_DIR, fname)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot] {filepath}")
    return filepath


def plot_summary(all_metrics: list):
    """汇总图: 所有攻击 × 所有 tier 的 RMSE 改善百分比"""
    df = pd.DataFrame(all_metrics)

    attacks = sorted(df['attack_type'].unique())
    tier_order = ['none', 'cfm', 'oracle']
    tiers_in_data = sorted(df['detector_tier'].unique(),
                           key=lambda t: tier_order.index(t)
                           if t in tier_order else 99)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Detector Tier Comparison — All Attack Types',
                 fontsize=14, fontweight='bold')

    # ---- Panel 1: 绝对 RMSE 对比 ----
    ax = axes[0]
    x = np.arange(len(attacks))
    n_tiers = len(tiers_in_data)
    w = 0.8 / n_tiers

    for i, tier in enumerate(tiers_in_data):
        rmse_vals = []
        for atk in attacks:
            row = df[(df['attack_type'] == atk) & (df['detector_tier'] == tier)]
            rmse_vals.append(row['post_pos_rmse'].values[0] if len(row) else 0)
        offset = (i - (n_tiers - 1) / 2) * w
        ax.bar(x + offset, rmse_vals, w,
               color=TIER_COLORS.get(tier, 'gray'), alpha=0.85,
               label=TIER_LABELS.get(tier, tier))

    ax.set_xticks(x); ax.set_xticklabels(attacks)
    ax.set_ylabel('Position RMSE [m]')
    ax.set_title('Post-Attack Position RMSE by Attack Type')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    # ---- Panel 2: 相对改善 (vs none) ----
    ax = axes[1]
    compare_tiers = [t for t in tiers_in_data if t != 'none']
    n_ct = len(compare_tiers)
    w2 = 0.8 / n_ct

    for i, tier in enumerate(compare_tiers):
        improvements = []
        for atk in attacks:
            row_none = df[(df['attack_type'] == atk) & (df['detector_tier'] == 'none')]
            row_tier = df[(df['attack_type'] == atk) & (df['detector_tier'] == tier)]
            if len(row_none) and len(row_tier):
                base = row_none['post_pos_rmse'].values[0]
                val = row_tier['post_pos_rmse'].values[0]
                imp = (base - val) / max(base, 1e-6) * 100
            else:
                imp = 0
            improvements.append(imp)
        offset = (i - (n_ct - 1) / 2) * w2
        bars = ax.bar(x + offset, improvements, w2,
                      color=TIER_COLORS.get(tier, 'gray'), alpha=0.85,
                      label=TIER_LABELS.get(tier, tier))
        for bar, val in zip(bars, improvements):
            if abs(val) > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1 if val > 0 else bar.get_height() - 3,
                        f'{val:+.1f}%', ha='center', fontsize=7)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(attacks)
    ax.set_ylabel('RMSE Improvement vs Baseline [%]')
    ax.set_title('Tracking Improvement over No-Detector Baseline')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(RESULT_DIR, 'det_summary.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot] {filepath}")
    return filepath


# ============================================================================
# 批量运行
# ============================================================================

def batch_all(attack_types: list = None,
              tiers: list = None,
              trajectory_type: str = 'lissajous',
              seed: int = 42,
              randomize_onset: bool = True,
              do_plot: bool = True,
              model_path: str = None,
              norm_path: str = None):
    """批量运行所有攻击 × 所有 tier 的对比仿真"""
    if attack_types is None:
        attack_types = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
    if tiers is None:
        tiers = ['none', 'cfm', 'oracle']

    all_metrics = []
    rng = np.random.RandomState(seed)

    for atk in attack_types:
        # 随机攻击起始 (匹配训练数据 [5, 30]s 分布)
        if randomize_onset and atk != 'A0':
            attack_onset = float(rng.uniform(ATTACK_ONSET_MIN, ATTACK_ONSET_MAX))
        elif atk == 'A0':
            attack_onset = SIM_TIME + 1.0
        else:
            attack_onset = ATTACK_ONSET

        print(f"\n{'#'*60}")
        print(f"# Attack: {atk} ({SensorAttack.ATTACK_NAMES[atk]}) @ onset={attack_onset:.1f}s")
        print(f"{'#'*60}")

        all_data = run_comparison(
            attack_type=atk, tiers=tiers,
            trajectory_type=trajectory_type, seed=seed,
            attack_onset=attack_onset,
            model_path=model_path, norm_path=norm_path
        )

        # 保存 NPZ
        for tier, data in all_data.items():
            fname = f'sim_det_{atk}_{tier}_{trajectory_type}.npz'
            save_dict = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    save_dict[k] = v
                elif isinstance(v, str):
                    save_dict[k] = np.array(v)
            np.savez_compressed(os.path.join(RESULT_DIR, fname), **save_dict)

            metrics = compute_metrics(data)
            if tier in ('cfm', 'oracle'):
                metrics.update(compute_detector_metrics(data))
            all_metrics.append(metrics)

        # 画对比图
        if do_plot:
            plot_comparison(all_data, atk, trajectory_type)

        # 快速摘要
        onset_idx = _get_onset_idx(all_data['none'])
        none_pos = np.sqrt(all_data['none']['X_error'][onset_idx:, 0]**2 +
                          all_data['none']['X_error'][onset_idx:, 1]**2)
        none_val = np.sqrt(np.mean(none_pos**2))
        print(f"  Summary: ", end='')
        for tier in tiers:
            X_err = all_data[tier]['X_error']
            pos = np.sqrt(X_err[onset_idx:, 0]**2 +
                         X_err[onset_idx:, 1]**2)
            val = np.sqrt(np.mean(pos**2))
            imp = (none_val - val) / max(none_val, 1e-6) * 100
            print(f"{tier}={val:.4f}m ({imp:+.1f}%)  ", end='')
        print()

    # 保存指标 CSV
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        csv_path = os.path.join(RESULT_DIR, 'det_metrics.csv')
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
        description='WMR Multi-Tier Detector Comparison Simulation')

    parser.add_argument('--attack', type=str, default='A4',
                        help='Attack type A0-A8 (default: A4)')
    parser.add_argument('--all', action='store_true',
                        help='Run all 9 attack types')
    parser.add_argument('--tiers', type=str, default='none,nn,oracle',
                        help='Detector tiers, comma-separated (default: none,nn,oracle)')
    parser.add_argument('--trajectory', type=str, default='lissajous',
                        choices=['lissajous', 'circular', 'spiral',
                                'random_waypoint', 'square'],
                        help='Reference trajectory type (default: lissajous)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-randomize-onset', action='store_true',
                        help='Use fixed onset (15s) instead of randomized [5,30]s')
    parser.add_argument('--model-path', type=str, default=None,
                        help='NN model weights path')
    parser.add_argument('--norm-path', type=str, default=None,
                        help='Normalizer params path')

    args = parser.parse_args()
    tiers = [t.strip() for t in args.tiers.split(',')]
    randomize_onset = not args.no_randomize_onset

    print("=" * 60)
    print("WMR Multi-Tier Detector Comparison")
    print("=" * 60)
    print(f"  Sim Time: {SIM_TIME}s")
    print(f"  Attack Onset: {'randomized [5,30]s' if randomize_onset else f'{ATTACK_ONSET}s (fixed)'}")
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Tiers: {tiers}")
    print(f"  Seed: {args.seed}")

    if args.all:
        print(f"  Mode: BATCH (all attacks)")
        batch_all(tiers=tiers, trajectory_type=args.trajectory,
                  seed=args.seed, randomize_onset=randomize_onset,
                  do_plot=not args.no_plot,
                  model_path=args.model_path, norm_path=args.norm_path)
    else:
        print(f"  Mode: SINGLE (attack={args.attack})")
        if randomize_onset and args.attack != 'A0':
            rng = np.random.RandomState(args.seed)
            attack_onset = float(rng.uniform(ATTACK_ONSET_MIN, ATTACK_ONSET_MAX))
        elif args.attack == 'A0':
            attack_onset = SIM_TIME + 1.0
        else:
            attack_onset = ATTACK_ONSET

        all_data = run_comparison(
            attack_type=args.attack, tiers=tiers,
            trajectory_type=args.trajectory, seed=args.seed,
            attack_onset=attack_onset,
            model_path=args.model_path, norm_path=args.norm_path
        )

        # 指标摘要
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")

        onset_idx = _get_onset_idx(all_data[tiers[0]])
        baseline_rmse = None

        for tier in tiers:
            data = all_data[tier]
            X_err = data['X_error']
            pos_err = np.sqrt(X_err[onset_idx:, 0]**2 + X_err[onset_idx:, 1]**2)
            rmse = np.sqrt(np.mean(pos_err**2))
            max_err = np.max(pos_err)

            if baseline_rmse is None:
                baseline_rmse = rmse
                imp_str = "(baseline)"
            else:
                imp = (baseline_rmse - rmse) / max(baseline_rmse, 1e-6) * 100
                imp_str = f"({imp:+.1f}%)"

            print(f"  {TIER_LABELS.get(tier, tier)}")
            print(f"    Post-attack RMSE: {rmse:.4f}m, Max: {max_err:.4f}m {imp_str}")

            if tier in ('cfm', 'oracle'):
                det_m = compute_detector_metrics(data)
                print(f"    Detection Acc: {det_m['detection_accuracy']:.1%}, "
                      f"Latency: {det_m['detection_latency_sec']:.2f}s, "
                      f"Recovery RMSE: {det_m['recovery_rmse']:.4f}m")

        # 保存 NPZ
        for tier, data in all_data.items():
            fname = f'sim_det_{args.attack}_{tier}_{args.trajectory}.npz'
            np.savez_compressed(
                os.path.join(RESULT_DIR, fname),
                **{k: v for k, v in data.items() if isinstance(v, np.ndarray)}
            )

        if not args.no_plot:
            plot_comparison(all_data, args.attack, args.trajectory)

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
