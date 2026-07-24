"""
simulate.py — WMR 闭环仿真 (含检测器集成)
================================================
统一的闭环仿真脚本，支持: 正常运行 / 攻击场景 / 检测器介入。

"""

import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from model import (WMRParams, WMRKinematics, SensorSimulator,
                   RandomizedTrajectory, SIM_STEPS)
from controller import NMPCController, NMPCParams
from attack import SensorAttack, AttackConfig, ALL_ATTACK_TYPES, ATTACK_NAMES
from backend import DetectorBackend

# ============================================================================
# 全局配置
# ============================================================================

# 仿真总时长与步数严格对齐 model.SIM_STEPS，避免标签与实际不一致
SIM_TIME = SIM_STEPS * WMRParams().Ts   # = 1000 * 0.05 = 50.0 s
ATTACK_ONSET_DEFAULT = 15.0
ATTACK_ONSET_MIN = 10.0
ATTACK_ONSET_MAX = 40.0
# 攻击协议与数据集生成 (generate_dataset.py) 完全统一: 时长 5.0s (100 步 < 窗口 128 步)
# → 任何窗口恒含干净步; onset ∈ [10, 40]s
ATTACK_DURATION = 5.0
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'simulations')
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
    return str(data.get('attack_type', 'A0')) != 'A0'


def _get_onset_idx(data: dict) -> int:
    return min(max(int(float(data['attack_onset']) / data['Ts']), 100), len(data['t']) - 1)


def _create_trajectory(traj_type: str, Ts: float, seed: int):
    # lissajous/circular 使用默认参数（向后兼容旧 LissajousTrajectory / CircularTrajectory）
    use_defaults = traj_type in ('lissajous', 'circular')
    return RandomizedTrajectory(Ts=Ts, family=traj_type, seed=seed,
                                use_defaults=use_defaults)


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
                   show_progress: bool = True,
                   oracle: bool = False) -> dict:
    """运行单次闭环仿真

    Args:
        attack_type:     攻击类型 'A0'~'A7' (A0=无攻击)
        use_detector:    True=DetectorBackend, False=y_meas 直送 NMPC
        trajectory_type: 轨迹类型 lissajous/circular/spiral/random_waypoint/square
        seed:            随机种子
        attack_onset:    攻击起始时间 [s] (None=随机[10,40]s, A0=永不攻击)
        model_path:      检测器模型权重路径
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
    attack_offset = attack_onset + ATTACK_DURATION

    robot = WMRKinematics(wmr_params)
    sensor = SensorSimulator()
    ctrl = NMPCController(NMPCParams())
    ctrl.load_or_build()

    attacker = SensorAttack(attack_type=attack_type,
                            onset_time=attack_onset,
                            config=AttackConfig(attack_duration=ATTACK_DURATION),
                            seed=seed)

    # 检测器
    detector = None
    if use_detector:
        detector = DetectorBackend(model_path=model_path, norm_path=norm_path)

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
            if oracle:
                true_class = (attack_type if (attack_onset <= t < attack_offset
                              and attack_type != 'A0') else 'A0')
                result = detector.detect(y_meas, oracle_class=true_class)
            else:
                result = detector.detect(y_meas)
            y_rec = result.y_recovered
            det_class = result.attack_class
            det_conf = result.confidence
            det_attack_est = result.attack_estimate

        # 4. 状态估计 = 检测器输出 (测量直通)
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
        data['attack_active'].append(1.0 if (attack_onset <= t < attack_offset and attack_type != 'A0') else 0.0)

        # 进度
        if show_progress and (step % 200 == 0 or step == n_steps - 1):
            pos_err = np.linalg.norm(X_error[:2])
            det_str = 'DET' if use_detector else 'NONE'
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
    result['attack_offset'] = attack_offset
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

    # 攻击期窗口 [onset, offset): 与 generate_dataset 约定一致, 避免恢复期稀释
    offset_idx = min(int(float(data.get('attack_offset', data['t'][-1] + 1.0)) / data['Ts']),
                     len(pos_err))
    post_pos = pos_err[onset_idx:offset_idx] if onset_idx < offset_idx else pos_err[-100:]

    return {
        'attack_type': data['attack_type'],
        'trajectory_type': data.get('trajectory_type', 'lissajous'),
        'attack_onset': float(data.get('attack_onset', ATTACK_ONSET_DEFAULT)),
        'steady_pos_rmse': float(np.sqrt(np.mean(pos_err[steady_start:steady_end] ** 2)))
                          if steady_end > steady_start else 0.0,
        'post_pos_rmse': float(np.sqrt(np.mean(post_pos ** 2))) if len(post_pos) > 0 else 0.0,
        'post_pos_max': float(np.max(post_pos)) if len(post_pos) > 0 else 0.0,
        'post_ang_rmse': float(np.sqrt(np.mean(ang_err[onset_idx:offset_idx] ** 2)))
                        if onset_idx < offset_idx else 0.0,
        'use_detector': data.get('use_detector', True),
    }


def compute_detector_metrics(data: dict) -> dict:
    """计算检测器 DL 指标"""
    onset_idx = _get_onset_idx(data)
    offset_idx = min(int(float(data.get('attack_offset', data['t'][-1] + 1.0)) / data['Ts']),
                     len(data['det_class']))
    attack_type = data['attack_type']
    det_classes = data['det_class']
    det_confs = data['det_conf']

    post_classes = det_classes[onset_idx:offset_idx]
    post_confs = det_confs[onset_idx:offset_idx]

    if len(post_classes) == 0:
        return {'detection_accuracy': 0.0, 'mean_confidence': 0.0,
                'detection_latency_steps': 999, 'detection_latency_sec': 99.0,
                'false_alarm_rate': 0.0}

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

    return {
        'detection_accuracy': accuracy,
        'mean_confidence': mean_conf,
        'detection_latency_steps': latency_steps,
        'detection_latency_sec': latency_steps * data['Ts'],
        'false_alarm_rate': far,
    }


# ============================================================================
# 批量运行
# ============================================================================

def run_batch(attack_types: list = None,
              trajectory_type: str = 'lissajous',
              seed: int = 42,
              use_detector: bool = True,
              randomize_onset: bool = True,
              model_path: str = None,
              norm_path: str = None,
              oracle: bool = False):
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
            model_path=model_path, norm_path=norm_path,
            oracle=oracle
        )

        # 保存 NPZ
        det_str = 'det' if use_detector else 'none'
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

        print(f"  Attack-Phase RMSE: {metrics['post_pos_rmse']:.4f}m", end='')
        if use_detector and 'detection_accuracy' in metrics:
            print(f" | Det Acc: {metrics['detection_accuracy']:.1%}", end='')
        print()

    # 保存指标 CSV
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        csv_path = os.path.join(RESULT_DIR, 'sim_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n  [CSV] {csv_path}")

    return all_metrics


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='WMR Closed-Loop Simulation with Detector')

    parser.add_argument('--attack', type=str, default='A1',
                        help='Attack type A0-A7 (default: A1)')
    parser.add_argument('--all', action='store_true',
                        help='Run all 8 attack types')
    parser.add_argument('--no-detector', action='store_true',
                        help='Disable detector (direct y_meas to NMPC)')
    parser.add_argument('--trajectory', type=str, default='lissajous',
                        choices=TRAJECTORY_FAMILIES,
                        help='Reference trajectory type (default: lissajous)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-randomize-onset', action='store_true',
                        help='Use fixed onset (15s) instead of randomized [10,40]s')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Detector model weights path')
    parser.add_argument('--norm-path', type=str, default=None,
                        help='Normalizer params path')
    parser.add_argument('--oracle', action='store_true',
                        help='Oracle 分类：用 ground truth 类别（隔离恢复逻辑）')

    args = parser.parse_args()

    use_detector = not args.no_detector
    randomize_onset = not args.no_randomize_onset

    print("=" * 60)
    print("WMR Closed-Loop Simulation")
    print("=" * 60)
    print(f"  Sim Time: {SIM_TIME}s")
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Detector: {'Enabled' if use_detector else 'None (baseline)'}")
    print(f"  Seed: {args.seed}")

    # 批量模式
    if args.all:
        print(f"  Mode: BATCH (all attacks)")
        print(f"  Attack Onset: {'randomized [10,40]s' if randomize_onset else f'{ATTACK_ONSET_DEFAULT}s (fixed)'}")
        run_batch(trajectory_type=args.trajectory, seed=args.seed,
                  use_detector=use_detector, randomize_onset=randomize_onset,
                  model_path=args.model_path, norm_path=args.norm_path,
                  oracle=args.oracle)
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
        model_path=args.model_path, norm_path=args.norm_path,
        oracle=args.oracle
    )

    # 指标
    X_err = data['X_error']
    if _has_attack(data):
        onset_idx = _get_onset_idx(data)
        offset_idx = min(int(float(data['attack_offset']) / data['Ts']), len(X_err))
        seg = np.sqrt(X_err[onset_idx:offset_idx, 0] ** 2 + X_err[onset_idx:offset_idx, 1] ** 2)
        rmse_label = 'Attack-phase RMSE'
    else:
        steady = max(int(5.0 / data['Ts']), 100)
        seg = np.sqrt(X_err[steady:, 0] ** 2 + X_err[steady:, 1] ** 2)
        rmse_label = 'Steady-state RMSE'
    rmse = np.sqrt(np.mean(seg ** 2))
    max_err = np.max(seg)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Detector: {'Enabled' if use_detector else 'None'}")
    print(f"  {rmse_label}: {rmse:.4f}m, Max: {max_err:.4f}m")

    if use_detector:
        det_m = compute_detector_metrics(data)
        print(f"  Detection Acc: {det_m['detection_accuracy']:.1%}, "
              f"Latency: {det_m['detection_latency_sec']:.2f}s, "
              f"False Alarm: {det_m['false_alarm_rate']:.1%}")

    # 保存 NPZ
    det_str = 'det' if use_detector else 'none'
    fname = f'sim_{args.attack}_{args.trajectory}_{det_str}.npz'
    save_dict = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = v
        elif isinstance(v, str):
            save_dict[k] = np.array(v)
    np.savez_compressed(os.path.join(RESULT_DIR, fname), **save_dict)
    print(f"  NPZ saved: {fname}")

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
