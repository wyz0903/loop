"""
generate_dataset.py — 多样化轨迹攻击数据集生成器
================================================================================
生成 WMR 在多种路径和攻击下的运行数据，供神经网络训练使用。

关键设计：
  1. 随机化参考轨迹参数 — 确保模型泛化性
  2. 记录详尽信号 — 内部运动学新息、控制指令、测量值、攻击真值等
  3. 静态分布 — 不引入检测器反馈，仅记录开环观测数据
  4. 输出格式 — 每轮仿真一个 .npz 文件 + 全局 metadata.csv 索引

轨迹类型 (随机采样，共5族):
  - lissajous       : 8字形, 随机 v_r ∈ [0.1,0.3], ω_freq ∈ [0.05,0.8]
  - circular        : 圆形,   随机 v_r ∈ [0.05,0.3], ω_r ∈ [±0.03,±0.5]
  - spiral          : 螺旋线, 半径从 R₀ 逐渐扩展到 Rmax
  - random_waypoint : 随机路径点, ω_r 方波切换产生折线/锯齿轨迹
  - square          : 正方形(圆角), 直行段 + 90°圆弧转弯，边长2-5m随机

每条轨迹 × 8 种攻击 (A0~A7) × N 组随机参数 = 多样化训练集

用法:
  python generate_dataset.py                           # 默认: 50 组轨迹 × 8 攻击 = 400 轮
  python generate_dataset.py --num-configs 100         # 100 组轨迹参数
  python generate_dataset.py --quick                   # 快速测试: 3 组 × 3 攻击
  python generate_dataset.py --attack A1               # 只生成一种攻击的数据,以A1为例。
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from model import (WMRParams, WMRKinematics, SensorSimulator,
                   RandomizedTrajectory,
                   SIM_TIME, SIM_STEPS)
from controller import NMPCController, NMPCParams
from attack import SensorAttack, AttackConfig, ALL_ATTACK_TYPES, ATTACK_NAMES

# ============================================================================
# 全局配置
# ============================================================================

DEFAULT_ATTACK_ONSET = 15.0      # 默认攻击开始时间
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================================
# 1. 单轮仿真运行器
# ============================================================================

def run_single_simulation(traj: RandomizedTrajectory,
                          attack_type: str,
                          attack_onset: float = DEFAULT_ATTACK_ONSET,
                          attack_duration: float = None,
                          seed: int = 42) -> dict:
    """运行一次完整的闭环仿真，记录所有信号

    Args:
        traj:         随机化的参考轨迹生成器
        attack_type:  攻击类型 A0~A7
        attack_onset: 攻击开始时间
        seed:         随机种子

    Returns:
        data: 包含所有时间序列的字典，每个值都是 shape (N_steps,) 或 (N_steps, D) 的 ndarray
    """
    # ---- 初始化组件 ----
    wmr_params = WMRParams()
    Ts = wmr_params.Ts
    n_steps = SIM_STEPS   # 1000

    robot = WMRKinematics(wmr_params)
    sensor = SensorSimulator()
    ctrl = NMPCController(NMPCParams())
    ctrl.load_or_build()

    atk_cfg = AttackConfig(attack_duration=attack_duration)
    attacker = SensorAttack(attack_type=attack_type,
                            onset_time=attack_onset, config=atk_cfg, seed=seed)

    # ---- 重置 ----
    traj.reset()
    # 机器人从轨迹起点附近随机初始化 (测试 NMPC 收敛性)
    perturb_rng = np.random.RandomState(seed)
    init_state = np.array([
        traj._x_r + perturb_rng.uniform(-0.3, 0.3),
        traj._y_r + perturb_rng.uniform(-0.3, 0.3),
        traj._theta_r + perturb_rng.uniform(-0.2, 0.2),
    ])
    robot.reset(init_state)
    ctrl.reset()
    attacker.reset()
    np.random.seed(seed)

    # ---- 仿真循环 ----
    data = defaultdict(list)
    u_cmd = np.zeros(2)

    for step in range(n_steps):
        t = step * Ts

        # 1. 参考轨迹
        Upsilon_r, u_r = traj.step(t)
        Ur_seq = traj.generate_sequence(t, NMPCParams().N)

        # 2. 真实状态 + 测量 (统一走过 inject 接口)
        true_state = robot.state.copy()
        noise = sensor.noise_std * np.random.randn(3)
        y_clean = true_state + noise
        y_meas = attacker.inject(t, y_clean)
        attack_signal = y_meas - y_clean  # 等效攻击信号 (重放攻击下非加性)

        # 3. 跟踪误差 (测量直接作为位姿估计)
        X_error = WMRKinematics.compute_error(Upsilon_r, y_meas)

        # 4. NMPC 控制
        u_cmd = ctrl.solve(X_error, Ur_seq)
        u_a = WMRKinematics.clamp_control(u_cmd)

        # 5. 机器人运动
        robot.step(u_a)

        # 6. 记录数据
        data['t'].append(t)
        data['true_state'].append(true_state.copy())       # 真实位姿 (3,)
        data['y_meas'].append(y_meas.copy())               # 含攻击测量 (3,)
        data['attack_signal'].append(attack_signal.copy()) # 攻击真值 (3,)
        data['y_clean'].append(y_clean.copy())             # 干净传感器信号 (3,) ★ 训练目标
        data['sensor_noise'].append(noise.copy())          # 传感器噪声 (3,)
        data['Upsilon_r'].append(Upsilon_r.copy())         # 参考位姿 (3,)
        data['u_r'].append(u_r.copy())                     # 参考指令 (2,)
        data['Upsilon_hat'].append(y_meas.copy())             # 状态估计 = 测量 (3,)
        data['X_error'].append(X_error.copy())             # 跟踪误差 (3,)
        data['u_cmd'].append(u_cmd.copy())                 # 控制指令 (2,)
        data['u_a'].append(u_a.copy())                     # 实际执行指令 (2,)
        data['attack_active'].append(1.0 if t >= attack_onset else 0.0)

        # 进度
        if step % 350 == 0:
            pos_err = np.linalg.norm(X_error[:2])
            print(f"    t={t:5.1f}s | |e_xy|={pos_err:.4f}m | "
                  f"|a|={np.linalg.norm(attack_signal):.4f}", flush=True)

    # 转为 ndarray
    result = {}
    for k, v in data.items():
        try:
            result[k] = np.array(v, dtype=float)
        except (ValueError, TypeError):
            result[k] = np.array(v, dtype=object)

    # 附加元信息
    result['Ts'] = Ts
    result['attack_type_label'] = attack_type
    result['attack_onset'] = attack_onset
    result['attack_offset'] = attack_onset + attack_duration if attack_duration else SIM_TIME + 1.0
    result['traj_info'] = traj.get_info()
    result['sim_time'] = SIM_TIME
    result['seed'] = seed

    return result


# ============================================================================
# 2. 数据集生成主循环
# ============================================================================

# 轨迹族固定顺序 (确定性遍历)
FAMILIES_ORDER = ['lissajous', 'circular', 'spiral', 'random_waypoint', 'square']


def generate_dataset(num_configs: int = None,
                     num_per_family: int = 12,
                     attack_types: list = None,
                     seed: int = 42) -> pd.DataFrame:
    """生成完整训练数据集

    Args:
        num_configs:   轨迹参数组数 (向后兼容，随机选族)
        num_per_family: 每族轨迹条数 (默认 12, 5族×12=60配置)
        attack_types:  要生成的攻击列表，默认全部 8 种
        seed:          全局随机种子

    Returns:
        metadata_df: 每轮仿真的元信息 DataFrame
    """
    if attack_types is None:
        attack_types = ALL_ATTACK_TYPES

    # 确定配置序列: num_per_family 优先, num_configs 向后兼容
    if num_configs is not None:
        # 旧模式: 随机选族
        config_families = None  # RandomizedTrajectory 随机选择
        n_configs = num_configs
    else:
        # 新模式: 固定每族条数
        config_families = []
        for fam in FAMILIES_ORDER:
            for i in range(num_per_family):
                config_families.append(fam)
        n_configs = len(config_families)

    total_runs = n_configs * len(attack_types)
    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**31 - 1, size=total_runs)

    print("=" * 65)
    print("数据集生成器 — 多样化轨迹 + 攻击")
    print("=" * 65)
    if config_families is not None:
        print(f"  轨迹参数组数: {n_configs} (每族 {num_per_family} 条)")
    else:
        print(f"  轨迹参数组数: {n_configs} (随机选族)")
    print(f"  攻击类型:     {attack_types}")
    print(f"  总仿真轮数:   {total_runs}")
    print(f"  仿真时长:     {SIM_TIME}s, Ts=0.05s, {SIM_STEPS}步/轮")
    print(f"  输出目录:     {RESULT_DIR}")
    print("=" * 65)

    metadata_rows = []
    run_idx = 0

    for cfg_idx in range(n_configs):
        # 轨迹种子: 确定性方案 (同 cfg_idx 始终生成同一条轨迹)
        if config_families is not None:
            fam_idx = FAMILIES_ORDER.index(config_families[cfg_idx])
            traj_idx = cfg_idx - fam_idx * num_per_family
            traj_seed = seed + fam_idx * 1000 + traj_idx
            traj = RandomizedTrajectory(seed=traj_seed, family=config_families[cfg_idx])
        else:
            traj_seed = seeds[cfg_idx * len(attack_types)]
            traj = RandomizedTrajectory(seed=traj_seed)
        traj_info = traj.get_info()

        print(f"\n[配置 {cfg_idx+1}/{n_configs}] "
              f"族={traj_info['trajectory_family']}, seed={traj_seed}")

        for atk_idx, atk_type in enumerate(attack_types):
            run_seed = seeds[run_idx]
            # 攻击开始时间在 [5, 35]s 内随机 (50s 仿真留出更多空间)
            if atk_type == 'A0':
                attack_onset = SIM_TIME + 1.0  # Normal: 永远不触发
                attack_duration = None
                attack_offset = SIM_TIME + 1.0
            else:
                attack_onset = float(rng.uniform(5.0, 35.0))
                # 攻击持续时间在 [5, 20]s 内随机，但不晚于仿真结束
                max_dur = SIM_TIME - attack_onset
                attack_duration = float(rng.uniform(5.0, min(20.0, max_dur)))
                attack_offset = attack_onset + attack_duration

            print(f"  [{run_idx+1:4d}/{total_runs}] "
                  f"{atk_type} ({ATTACK_NAMES[atk_type]}) ...", end='', flush=True)

            t0 = time.time()
            data = run_single_simulation(
                traj=traj, attack_type=atk_type,
                attack_onset=attack_onset,
                attack_duration=attack_duration,
                seed=run_seed
            )
            elapsed = time.time() - t0

            # 保存 .npz
            fname = f'sim_{cfg_idx:04d}_{atk_type}.npz'
            filepath = os.path.join(RESULT_DIR, fname)
            # 保存时去掉 traj_info (dict 无法直接存入 npz)
            traj_info_copy = data.pop('traj_info', {})
            np.savez_compressed(filepath, **{k: v for k, v in data.items()
                                            if isinstance(v, np.ndarray)})
            data['traj_info'] = traj_info_copy  # 恢复

            # 记录元信息（RMSE 仅计攻击激活期 [onset, offset]）
            X_err = data['X_error']
            onset_idx = int(round(attack_onset / data['Ts'])) if attack_onset < SIM_TIME else len(X_err)
            offset_idx = int(round(attack_offset / data['Ts'])) if attack_offset < SIM_TIME else len(X_err)
            offset_idx = max(offset_idx, onset_idx + 1)
            pos_err_active = np.sqrt(X_err[onset_idx:offset_idx, 0]**2 + X_err[onset_idx:offset_idx, 1]**2)
            pos_rmse = float(np.sqrt(np.mean(pos_err_active**2))) if len(pos_err_active) > 0 else 0.0

            Ts_data = float(data['Ts'])
            attack_onset_step = int(round(attack_onset / Ts_data))
            attack_offset_step = int(round(attack_offset / Ts_data))

            metadata_rows.append({
                'run_id': run_idx,
                'config_id': cfg_idx,
                'filename': fname,
                'attack_type': atk_type,
                'attack_name': ATTACK_NAMES[atk_type],
                'attack_onset': attack_onset,
                'attack_onset_step': attack_onset_step,
                'attack_duration': attack_duration if attack_duration else 0.0,
                'attack_offset': attack_offset,
                'attack_offset_step': attack_offset_step,
                'trajectory_family': traj_info['trajectory_family'],
                'traj_seed': traj_seed,
                'sim_seed': run_seed,
                'pos_rmse_post_attack': pos_rmse,
            })
            # 添加轨迹参数到 metadata
            for k, v in traj_info.items():
                if k != 'trajectory_family':
                    metadata_rows[-1][f'traj_{k}'] = v

            run_idx += 1
            print(f" RMSE={pos_rmse:.4f}m ({elapsed:.1f}s)")

    # 保存 metadata
    df = pd.DataFrame(metadata_rows)
    csv_path = os.path.join(RESULT_DIR, 'metadata.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*65}")
    print(f"数据集生成完成！")
    print(f"  总轮数:     {total_runs}")
    print(f"  总时间步:   {total_runs * SIM_STEPS:,}")
    print(f"  Metadata:   {csv_path}")
    print(f"  数据文件:   {RESULT_DIR}/sim_*.npz")
    print(f"{'='*65}")

    return df


# ============================================================================
# 3. 数据验证工具
# ============================================================================

def validate_dataset(df: pd.DataFrame):
    """验证生成的数据集完整性"""
    print("\n=== 数据集验证 ===")

    # 文件存在性
    missing = []
    for fname in df['filename']:
        path = os.path.join(RESULT_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname)
    if missing:
        print(f"  [错误] 缺失 {len(missing)} 个文件: {missing[:5]}...")
    else:
        print(f"  [OK] 全部 {len(df)} 个文件存在")

    # 攻击分布
    print("\n  攻击类型分布:")
    for atk in ALL_ATTACK_TYPES:
        count = len(df[df['attack_type'] == atk])
        print(f"    {atk} ({ATTACK_NAMES[atk]}): {count}")

    # 轨迹族分布
    print("\n  轨迹族分布:")
    for fam in sorted(df['trajectory_family'].unique()):
        count = len(df[df['trajectory_family'] == fam])
        print(f"    {fam}: {count}")

    # RMSE 统计
    if 'pos_rmse_post_attack' in df.columns:
        rmse_col = df['pos_rmse_post_attack']
        print(f"\n  攻击后 RMSE 统计:")
        print(f"    Mean: {rmse_col.mean():.4f}m")
        print(f"    Std:  {rmse_col.std():.4f}m")
        print(f"    Min:  {rmse_col.min():.4f}m")
        print(f"    Max:  {rmse_col.max():.4f}m")

    print()


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='生成 WMR 多样化轨迹攻击训练数据集')

    parser.add_argument('--num-per-family', type=int, default=12,
                        help='每族轨迹条数 (默认 12, 5族×12=60 配置)')
    parser.add_argument('--num-configs', type=int, default=None,
                        help='轨迹参数组数 (旧模式: 随机选族)')
    parser.add_argument('--attack', type=str, default=None,
                        help='只生成指定攻击类型 (如 A1), 默认全部')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试: 每族2条 × 3攻击 (A0,A1,A2)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认 ./dataset/)')
    parser.add_argument('--seed', type=int, default=42,
                        help='全局随机种子')

    args = parser.parse_args()

    if args.output_dir:
        global RESULT_DIR
        RESULT_DIR = args.output_dir
        os.makedirs(RESULT_DIR, exist_ok=True)

    if args.quick:
        num_per_family = 2
        attack_types = ['A0', 'A1', 'A2']
        print(f"[快速模式] 每族 {num_per_family} 条 × 3 种攻击 = "
              f"{num_per_family * 5 * 3} 轮")
        df = generate_dataset(
            num_per_family=num_per_family,
            attack_types=attack_types,
            seed=args.seed
        )
    elif args.num_configs is not None:
        df = generate_dataset(
            num_configs=args.num_configs,
            attack_types=[args.attack] if args.attack else ALL_ATTACK_TYPES,
            seed=args.seed
        )
    else:
        attack_types = [args.attack] if args.attack else ALL_ATTACK_TYPES
        df = generate_dataset(
            num_per_family=args.num_per_family,
            attack_types=attack_types,
            seed=args.seed
        )

    validate_dataset(df)


if __name__ == "__main__":
    main()
