"""
generate_dataset.py — 多样化轨迹攻击数据集生成器
================================================================================
生成 WMR 在多种路径和攻击下的运行数据，供神经网络训练使用。

关键设计：
  1. 随机化参考轨迹参数 — 确保模型泛化性（不再是固定 8 字形）
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
import sys
import time
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple

from model import (WMRParams, WMRKinematics, SensorSimulator,
                   SIM_STEPS,
                   _rk4_front_axle)
from controller import NMPCController, NMPCParams
from attack import SensorAttack, AttackConfig, ALL_ATTACK_TYPES, ATTACK_NAMES

# ============================================================================
# 全局配置
# ============================================================================

SIM_TIME = 50.0
DEFAULT_ATTACK_ONSET = 15.0      # 默认攻击开始时间
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================================
# 1. 多样化参考轨迹生成器
# ============================================================================

class RandomizedTrajectory:
    """随机参数参考轨迹生成器

    每次初始化随机采样轨迹类型和参数，确保训练数据覆盖广泛的运动模式。

    轨迹族:
      'lissajous'       : 8字形 Lissajous (光滑曲线)
      'circular'        : 定曲率圆形 (光滑曲线)
      'spiral'          : 阿基米德螺旋线 — 半径线性外扩 (光滑曲线)
      'random_waypoint' : 随机路径点 — ω_r 方波切换，产生折线/锯齿形轨迹
      'square'          : 正方形(圆角) — 边长随机，直行+90°圆弧转弯
    """

    def __init__(self, Ts: float = 0.05, seed: int = None, pos_bound: float = 2.5,
                 family: str = None):
        self.Ts = Ts
        self.pos_bound = pos_bound
        self.rng = np.random.RandomState(seed)

        # 指定族或随机选择
        if family is not None:
            self.family = family
        else:
            self.family = self.rng.choice(
                ['lissajous', 'circular', 'spiral', 'random_waypoint', 'square'])
        self._init_params()
        # 固定初始朝向 (验证与实际运行共用，确保一致性)
        self._init_theta = self.rng.uniform(-0.1, 0.1)
        # 预验证 + 居中: 确保完整轨迹自然不超界且中心对齐原点
        self._verify_and_center_trajectory()
        self.reset()

    def _init_params(self):
        """随机采样轨迹参数 (宽范围 + 自动过滤确保 40-95% 覆盖 ±2.5m)"""
        pb = self.pos_bound  # 2.5

        if self.family == 'lissajous':
            # 8字形: 包络半径 ≈ v / w_freq
            # 宽范围采样 → 由 _verify_and_center_trajectory 过滤 v/w_freq∈[1.0,2.5]
            self.v_const = self.rng.uniform(0.10, 0.30)
            self.w_freq = self.rng.uniform(0.04, 0.35)
            self.w_amp = 2.4048 * self.w_freq
            self._gen_func = self._step_lissajous

        elif self.family == 'circular':
            # 圆形: 解析居中保证圆心在原点, R_target 直接控制覆盖
            # 宽范围 → 由解析居中和最小覆盖检查过滤
            self.v_const = self.rng.uniform(0.08, 0.30)
            R_target = self.rng.uniform(0.40, 2.40)          # 圆半径 [m]
            direction = self.rng.choice([-1, 1])
            self.w_const = direction * (self.v_const / R_target)
            # 限幅角速度在合理范围 (w_max=1.76)
            self.w_const = float(np.clip(self.w_const, -1.70, 1.70))
            if abs(self.w_const) < 0.03:
                self.w_const = direction * 0.03
            self._gen_func = self._step_circular

        elif self.family == 'spiral':
            # 阿基米德螺旋线: R(t) = R0 * (1 + alpha*t), 半径线性外扩
            # 宽范围 → 圈数检查 N≥1.8 自动过滤
            self._spiral_R0 = self.rng.uniform(0.03, 0.25)     # 初始半径 [m]
            self._spiral_Rmax = self.rng.uniform(0.80, 2.30)   # 终点半径 [m]
            self._spiral_v = self.rng.uniform(0.15, 0.30)      # 线速度 [m/s]
            self._spiral_dir = self.rng.choice([-1, 1])         # 旋转方向
            # 增长率: R(T_sim) = Rmax => alpha = (Rmax/R0 - 1)/T_sim
            self._spiral_alpha = (self._spiral_Rmax / self._spiral_R0 - 1.0) / SIM_TIME
            self._gen_func = self._step_spiral

        elif self.family == 'random_waypoint':
            # 随机航点: 宽速度范围 + 长直行段 → 最小覆盖检查过滤
            # 预生成全部 SIM_TIME 内的切换序列 (保证预验证与实际一致)
            self.v_const = self.rng.uniform(0.12, 0.30)
            self._waypoint_w_values = []
            self._waypoint_intervals = []
            t_now = 0.0
            while t_now < SIM_TIME:
                self._waypoint_w_values.append(self.rng.uniform(-0.50, 0.50))
                dt = self.rng.uniform(1.5, 5.5)
                self._waypoint_intervals.append(dt)
                t_now += dt
            self._waypoint_next_switch = 0.0
            self._waypoint_idx = -1
            self._gen_func = self._step_random_waypoint

        elif self.family == 'square':
            # 正方形(圆角): 宽范围 → 约束完整周期 ≤ SIM_STEPS 步
            self._sq_side = self.rng.uniform(1.5, 3.0)
            self._sq_v_straight = self.rng.uniform(0.20, 0.30)
            self._sq_v_turn = self.rng.uniform(0.08, 0.24)
            self._sq_w_turn = self.rng.uniform(0.70, 1.60)
            # 步数取整
            self._sq_n_straight = max(1, int(np.round(
                self._sq_side / (self._sq_v_straight * self.Ts))))
            self._sq_n_turn = max(1, int(np.round(
                (0.5 * np.pi) / (self._sq_w_turn * self.Ts))))
            # 约束: 完整周期 ≤ SIM_STEPS 步 (保证居中)
            n_cycle = 4 * (self._sq_n_straight + self._sq_n_turn)
            if n_cycle > SIM_STEPS:
                s = float(SIM_STEPS) / n_cycle
                self._sq_n_straight = max(10, int(self._sq_n_straight * s))
                self._sq_n_turn = max(10, int(self._sq_n_turn * s))
                # 缩放取整后可能还有 1-2 步超出，微调
                while 4 * (self._sq_n_straight + self._sq_n_turn) > SIM_STEPS:
                    self._sq_n_straight = max(10, self._sq_n_straight - 1)
            # 反算 v/ω 保证精确边长和精确 90° (RK4 下无漂移)
            self._sq_v_straight = self._sq_side / (self._sq_n_straight * self.Ts)
            self._sq_w_turn = (0.5 * np.pi) / (self._sq_n_turn * self.Ts)
            self._sq_phase = 0
            self._sq_step_cnt = 0
            self._sq_edge_count = 0
            self._gen_func = self._step_square

    def _verify_and_center_trajectory(self, max_attempts: int = 100):
        """预模拟完整轨迹，计算居中偏移并验证边界。

        1. circular 族: 解析几何计算圆心 (无需完整一圈即可正确居中)
        2. 其他族: 从 (0,0) 出发模拟，记录 x/y 的 min/max
        3. 边界检查: 居中后半宽 ≤ pos_bound
        4. 最小覆盖检查: 居中后半宽 ≥ 0.4*pos_bound (1.0m)
        5. 螺旋圈数检查: N ≥ 1.8
        6. 不通过则重新采样参数
        """
        n_steps = SIM_STEPS  # 1000

        for attempt in range(max_attempts):
            # ---- circular 族: 解析居中 ----
            if self.family == 'circular':
                R = self.v_const / abs(self.w_const)
                if R > self.pos_bound:
                    self._init_params()
                    self._init_theta = self.rng.uniform(-0.1, 0.1)
                    continue
                # 最小覆盖检查
                if R < 0.4 * self.pos_bound:
                    self._init_params()
                    self._init_theta = self.rng.uniform(-0.1, 0.1)
                    continue
                # 解析圆心: 使圆心在原点
                self._cx = self.v_const * np.sin(self._init_theta) / self.w_const
                self._cy = -self.v_const * np.cos(self._init_theta) / self.w_const
                return  # 通过

            # ---- 其他族: 预模拟居中 ----
            x, y, theta = 0.0, 0.0, self._init_theta
            x_min, x_max = 0.0, 0.0
            y_min, y_max = 0.0, 0.0
            for step_idx in range(n_steps):
                u_r = self._gen_func(step_idx * self.Ts)
                x, y, theta = _rk4_front_axle(x, y, theta, u_r[0], u_r[1], self.Ts)
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)
            # 居中偏移量 (使轨迹 bounding box 中心对齐原点)
            self._cx = -(x_min + x_max) / 2.0
            self._cy = -(y_min + y_max) / 2.0
            # 居中后的半宽
            hw = (x_max - x_min) / 2.0
            hh = (y_max - y_min) / 2.0

            # ---- 边界检查 ----
            if hw > self.pos_bound or hh > self.pos_bound:
                self._init_params()
                self._init_theta = self.rng.uniform(-0.1, 0.1)
                continue

            # ---- 最小覆盖检查 (≥ 40% pos_bound = 1.0m) ----
            if max(hw, hh) < 0.4 * self.pos_bound:
                self._init_params()
                self._init_theta = self.rng.uniform(-0.1, 0.1)
                continue

            # ---- 螺旋圈数检查 ----
            if self.family == 'spiral':
                N_turns = (SIM_TIME * self._spiral_v
                           * np.log(self._spiral_Rmax / self._spiral_R0)
                           / (2.0 * np.pi * (self._spiral_Rmax - self._spiral_R0)))
                if N_turns < 1.8:
                    self._init_params()
                    self._init_theta = self.rng.uniform(-0.1, 0.1)
                    continue

            return  # 通过

        raise RuntimeError(
            f'{self.family}: 无法在 {max_attempts} 次内生成不超界的轨迹参数，'
            f'请检查参数范围设置')

    def reset(self):
        """重置位姿到居中起始点 (轨迹 bounding box 中心对齐原点)"""
        self._x_r = self._cx
        self._y_r = self._cy
        self._theta_r = self._init_theta  # 使用与预验证相同的初始朝向
        # 重置状态机变量
        if self.family == 'random_waypoint':
            self._waypoint_next_switch = 0.0
            self._waypoint_idx = -1
        elif self.family == 'square':
            self._sq_phase = 0
            self._sq_step_cnt = 0
            self._sq_edge_count = 0

    # ---- 各轨迹族的 step 实现 ----

    def _step_lissajous(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        v_r = self.v_const
        w_r = self.w_amp * np.cos(self.w_freq * t)
        return np.array([v_r, w_r])

    def _step_circular(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([self.v_const, self.w_const])

    def _step_spiral(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """阿基米德螺旋线: R(t)=R0*(1+alpha*t), w(t)=v/R(t), 半径线性外扩"""
        R = self._spiral_R0 * (1.0 + self._spiral_alpha * t)
        v_r = self._spiral_v
        w_r = self._spiral_dir * v_r / R
        # 限幅角速度以保证可跟踪性
        w_r = float(np.clip(w_r, -1.50, 1.50))
        return np.array([v_r, w_r])

    def _step_random_waypoint(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """ω_r 方波切换，模拟路径点之间的折线运动"""
        if t >= self._waypoint_next_switch:
            self._waypoint_idx += 1
            # 按需扩展预生成序列
            while self._waypoint_idx >= len(self._waypoint_w_values):
                self._waypoint_w_values.append(
                    self.rng.uniform(-0.50, 0.50))
                self._waypoint_intervals.append(
                    self.rng.uniform(1.0, 3.5))
            self._waypoint_next_switch = t + self._waypoint_intervals[self._waypoint_idx]
        w_r = self._waypoint_w_values[self._waypoint_idx]
        return np.array([self.v_const, w_r])

    def _step_square(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """正方形(圆角)轨迹: 步数计数器精确控制 (无 fence-post 误差)

        直行: _sq_n_straight 步, w=0
        转弯: _sq_n_turn 步, w=-w_turn (精确 90°)
        """
        # 1. 基于当前 phase 选择控制
        if self._sq_phase == 0:
            v_r = self._sq_v_straight
            w_r = 0.0
        else:
            v_r = self._sq_v_turn
            w_r = -self._sq_w_turn

        # 2. 步数计数 → phase 切换
        self._sq_step_cnt += 1
        if self._sq_phase == 0:
            if self._sq_step_cnt >= self._sq_n_straight:
                self._sq_phase = 1
                self._sq_step_cnt = 0
        else:
            if self._sq_step_cnt >= self._sq_n_turn:
                self._sq_phase = 0
                self._sq_step_cnt = 0
                self._sq_edge_count += 1
                if self._sq_edge_count >= 4:
                    self._sq_edge_count = 0  # 循环

        return np.array([v_r, w_r])

    # ---- 统一接口 ----

    def step(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """单步推进

        Returns:
            Upsilon_r: 参考位姿 (3,)
            u_r:       参考指令 [v_r, w_r] (2,)
        """
        u_r = self._gen_func(t)

        # RK4 积分 (前端偏置运动学，与 WMRKinematics 一致)
        self._x_r, self._y_r, self._theta_r = _rk4_front_axle(
            self._x_r, self._y_r, self._theta_r,
            u_r[0], u_r[1], self.Ts)

        return np.array([self._x_r, self._y_r, self._theta_r]), u_r

    def generate_sequence(self, t_start: float, N: int) -> np.ndarray:
        """生成未来 N 步参考指令序列 (MPC 用)

        注意: 对于有状态机的轨迹族 (random_waypoint, square),
        保存并恢复状态以避免影响主仿真循环。
        """
        Ur_seq = np.zeros((2, N))
        # 保存状态机变量 (如有)
        saved = {}
        if self.family == 'random_waypoint':
            saved = {'idx': self._waypoint_idx, 'next': self._waypoint_next_switch}
        elif self.family == 'square':
            saved = {'phase': self._sq_phase, 'cnt': self._sq_step_cnt,
                     'edge': self._sq_edge_count}
        # 生成序列
        for k in range(N):
            Ur_seq[:, k] = self._gen_func(t_start + k * self.Ts)
        # 恢复状态
        if self.family == 'random_waypoint':
            self._waypoint_idx = saved['idx']
            self._waypoint_next_switch = saved['next']
        elif self.family == 'square':
            self._sq_phase = saved['phase']
            self._sq_step_cnt = saved['cnt']
            self._sq_edge_count = saved['edge']
        return Ur_seq

    def get_info(self) -> dict:
        """返回轨迹元信息"""
        info = {'trajectory_family': self.family}
        if self.family == 'lissajous':
            info.update(v_r=self.v_const, w_freq=self.w_freq, w_amp=self.w_amp)
        elif self.family == 'circular':
            info.update(v_r=self.v_const, w_r=self.w_const)
        elif self.family == 'spiral':
            info.update(v=self._spiral_v, R0=self._spiral_R0, Rmax=self._spiral_Rmax,
                       direction=self._spiral_dir)
        elif self.family == 'random_waypoint':
            info.update(v_r=self.v_const)
        elif self.family == 'square':
            info.update(side=self._sq_side, v_straight=self._sq_v_straight,
                       v_turn=self._sq_v_turn, w_turn=self._sq_w_turn)
        return info


# ============================================================================
# 2. 单轮仿真运行器
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

    # 内部运动学预测起点 — 第一步用初始位姿 (匹配推理端上一帧恢复测量锚定)
    y_meas_prev = init_state.copy()

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

        # 3. 内部运动学新息 (锚定到上一帧测量)
        X_pred_internal = WMRKinematics.kinematic_predict(y_meas_prev, u_cmd)
        internal_innovation = y_meas - X_pred_internal
        internal_innovation[2] = np.arctan2(np.sin(internal_innovation[2]),
                                              np.cos(internal_innovation[2]))  # θ 包裹到[-π,π]

        # 4. 锚定到当前测量 (供下一步运动学预测使用)
        y_meas_prev = y_meas.copy()

        # 5. 跟踪误差 (测量直接作为位姿估计)
        X_error = WMRKinematics.compute_error(Upsilon_r, y_meas)

        # 6. NMPC 控制
        u_cmd = ctrl.solve(X_error, Ur_seq)
        u_a = WMRKinematics.clamp_control(u_cmd)

        # 7. 机器人运动
        robot.step(u_a)

        # 8. 记录数据
        data['t'].append(t)
        data['true_state'].append(true_state.copy())       # 真实位姿 (3,)
        data['y_meas'].append(y_meas.copy())               # 含攻击测量 (3,)
        data['attack_signal'].append(attack_signal.copy()) # 攻击真值 (3,)
        data['y_clean'].append(y_clean.copy())             # 干净传感器信号 (3,) ★ 训练目标
        data['sensor_noise'].append(noise.copy())          # 传感器噪声 (3,)
        data['Upsilon_r'].append(Upsilon_r.copy())         # 参考位姿 (3,)
        data['u_r'].append(u_r.copy())                     # 参考指令 (2,)
        data['Upsilon_hat'].append(y_meas.copy())             # 状态估计 = 测量 (3,)
        data['internal_innovation'].append(internal_innovation.copy())  # 内部运动学新息 (3,) ★ 键名与 preprocess_data.py INPUT_CHANNELS 对应
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
# 4. 数据集生成主循环
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
# 5. 数据验证工具
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
