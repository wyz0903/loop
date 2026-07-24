"""
model.py -- WMR Kinematic Model, Reference Trajectories, Sensor Simulator
==========================================================================
基于两轮差速轮式移动机器人(WMR)前端位姿(head posture)运动学。
物理参数来源：TurtleBot4 安全模式
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# 仿真全局常量
SIM_TIME = 50.0        # 仿真总时长 [s]
SIM_STEPS = 1000       # 仿真总步数 = round(SIM_TIME / 0.05)，避免 IEEE 754 截断误差



# ============================================================================
# 1. 物理参数定义
# ============================================================================

@dataclass
class WMRParams:
    """两轮差速WMR物理参数

    参数来源 (TurtleBot4 安全模式)：
      alpha = 0.17 : 前端距几何中心距离 (配置文档 Section 2.3)
      Ts    = 0.05 : 控制采样周期 (配置文档 Section 1)
      v_max = 0.3  : 最大线速度 m/s (配置文档 Section 3.1)
      w_max = 1.76 : 最大角速度 rad/s (配置文档 Section 3.1)
    """
    # TurtleBot4 安全模式: v_max = 0.3 m/s, ω_max = b = a/α ≈ 1.76 rad/s
    alpha: float = 0.17        # 前端偏置距离 [m] (α = a/b = 0.3/1.76)
    Ts: float = 0.05           # 采样时间 [s]
    v_max: float = 0.3         # 线速度上限 [m/s] (= a, 论文 U 约束)
    w_max: float = 1.76        # 角速度上限 [rad/s] (= b = a/α, 论文 U 约束)
    pos_bound: float = 2.5     # 空间位置边界 [m] (±2.5m 安全运行范围)


# ============================================================================
# 2. 参考轨迹生成器 — 5族统一 (Lissajous / Circular / Spiral / RandomWaypoint / Square)
# ============================================================================

def _rk4_front_axle(x: float, y: float, theta: float, v: float, w: float, Ts: float,
                     alpha: float = 0.17):
    """前端偏置运动学 RK4 积分（与 WMRKinematics 一致）

    运动学: ẋ = v·cos(θ) − α·w·sin(θ)
            ẏ = v·sin(θ) + α·w·cos(θ)
            θ̇ = w

    参考轨迹与机器人使用同一运动学模型，确保参考轨迹物理可行，
    NMPC 可在无扰动条件下将跟踪误差驱动至零。
    """
    h = Ts
    def _f(_x, _y, _t):
        return (v * np.cos(_t) - alpha * w * np.sin(_t),
                v * np.sin(_t) + alpha * w * np.cos(_t),
                w)
    k1x, k1y, k1t = _f(x, y, theta)
    k2x, k2y, k2t = _f(x + h/2*k1x, y + h/2*k1y, theta + h/2*k1t)
    k3x, k3y, k3t = _f(x + h/2*k2x, y + h/2*k2y, theta + h/2*k2t)
    k4x, k4y, k4t = _f(x + h*k3x, y + h*k3y, theta + h*k3t)
    xn = x + h/6.0 * (k1x + 2*k2x + 2*k3x + k4x)
    yn = y + h/6.0 * (k1y + 2*k2y + 2*k3y + k4y)
    tn = theta + h/6.0 * (k1t + 2*k2t + 2*k3t + k4t)
    tn = np.arctan2(np.sin(tn), np.cos(tn))
    return xn, yn, tn

class RandomizedTrajectory:
    """5族随机参数参考轨迹生成器（数据集生成 + 仿真统一入口）

    每次初始化随机采样轨迹类型和参数，确保训练数据覆盖广泛的运动模式。
    当指定 family 且 use_defaults=True 时使用默认参数（向后兼容）。

    轨迹族:
      'lissajous'       : 8字形 Lissajous (光滑曲线)
      'circular'        : 定曲率圆形 (光滑曲线)
      'spiral'          : 阿基米德螺旋线 — 半径线性外扩 (光滑曲线)
      'random_waypoint' : 随机路径点 — ω_r 方波切换，产生折线/锯齿形轨迹
      'square'          : 圆角正方形 — 4 段直边 + clothoid 过渡 + 圆弧，G² 连续
    """

    # 默认参数（use_defaults=True 时使用，满足 _verify_and_center_trajectory 边界门限 [1.0, 2.5]m）
    _DEFAULTS = {
        'lissajous':       dict(v_const=0.15, w_freq=0.16),
        'circular':        dict(v_const=0.15, w_const=0.10),
        'spiral':          dict(_spiral_R0=0.10, _spiral_Rmax=1.5, _spiral_v=0.20, _spiral_dir=1),
        'random_waypoint': dict(v_const=0.20),
        'square':          dict(_sq_side=2.0, _sq_R=0.20, _sq_v=0.22, _sq_dir=-1),
    }

    def __init__(self, Ts: float = 0.05, seed: int = None, pos_bound: float = 2.5,
                 family: str = None, use_defaults: bool = False):
        self.Ts = Ts
        self.pos_bound = pos_bound
        self.use_defaults = use_defaults
        self.rng = np.random.RandomState(seed)

        if family is not None:
            self.family = family
        else:
            self.family = self.rng.choice(
                ['lissajous', 'circular', 'spiral', 'random_waypoint', 'square'])
        self._init_params()
        if self.family != 'square':
            self._init_theta = 0.0 if use_defaults else self.rng.uniform(-0.1, 0.1)
        self._verify_and_center_trajectory()
        self.reset()

    def _init_params(self):
        """随机采样轨迹参数（宽范围 + 自动过滤确保覆盖）"""
        if self.use_defaults:
            self._init_default_params()
            return

        if self.family == 'lissajous':
            self.v_const = self.rng.uniform(0.10, 0.30)
            self.w_freq = self.rng.uniform(0.04, 0.35)
            self.w_amp = 2.4048 * self.w_freq

        elif self.family == 'circular':
            self.v_const = self.rng.uniform(0.08, 0.30)
            R_target = self.rng.uniform(0.40, 2.40)
            direction = self.rng.choice([-1, 1])
            self.w_const = direction * (self.v_const / R_target)
            self.w_const = float(np.clip(self.w_const, -1.70, 1.70))
            if abs(self.w_const) < 0.03:
                self.w_const = direction * 0.03

        elif self.family == 'spiral':
            self._spiral_R0 = self.rng.uniform(0.03, 0.25)
            self._spiral_Rmax = self.rng.uniform(0.80, 2.30)
            self._spiral_v = self.rng.uniform(0.15, 0.30)
            self._spiral_dir = self.rng.choice([-1, 1])
            self._spiral_alpha = (self._spiral_Rmax / self._spiral_R0 - 1.0) / SIM_TIME

        elif self.family == 'random_waypoint':
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

        elif self.family == 'square':
            self._sq_side = self.rng.uniform(1.5, 3.0)
            self._sq_v = self.rng.uniform(0.15, 0.30)
            R_min = max(0.10, self._sq_v / WMRParams().w_max)
            self._sq_R = self.rng.uniform(R_min, min(0.45, self._sq_side / 2.0))
            self._sq_dir = self.rng.choice([-1, 1])
            self._build_square_path()

        if self.family != 'square':
            self._gen_func = getattr(self, f'_step_{self.family}')

    def _init_default_params(self):
        """使用族默认参数（指定 family 且 use_defaults=True 时）"""
        d = self._DEFAULTS[self.family]
        if self.family == 'lissajous':
            self.v_const = d['v_const']
            self.w_freq = d['w_freq']
            self.w_amp = 2.4048 * self.w_freq
        elif self.family == 'circular':
            self.v_const = d['v_const']
            self.w_const = d['w_const']
        elif self.family == 'spiral':
            for k, v in d.items():
                setattr(self, k, v)
            self._spiral_alpha = (self._spiral_Rmax / self._spiral_R0 - 1.0) / SIM_TIME
        elif self.family == 'random_waypoint':
            self.v_const = d['v_const']
            self._waypoint_w_values = [0.3, -0.3, 0.2, -0.2]
            self._waypoint_intervals = [3.0, 3.0, 3.0, 3.0]
            self._waypoint_next_switch = 0.0
            self._waypoint_idx = -1
        elif self.family == 'square':
            self._sq_side = d['_sq_side']
            self._sq_R = d['_sq_R']
            self._sq_v = d['_sq_v']
            self._sq_dir = d['_sq_dir']
            self._build_square_path()
        if self.family != 'square':
            self._gen_func = getattr(self, f'_step_{self.family}')

    def _build_square_path(self):
        """预计算一个完整周期的圆角方形前端轴位姿和控制量。

        每个拐角: straight → clothoid-in (w 0→w_peak) → arc (const w) → clothoid-out (w_peak→0).
        clothoid 过渡段使曲率连续变化，消除前端轴路径在直行/转弯交接处的切线方向突变 (G¹ kink).
        """
        alpha = WMRParams().alpha
        w_max = WMRParams().w_max
        v_max = WMRParams().v_max
        Ts = self.Ts

        # 1. 直边步数
        n_straight = max(5, int(np.round(self._sq_side / (self._sq_v * Ts))))
        v_straight = self._sq_side / (n_straight * Ts)

        # 2. 转弯: 总弧长 ≈ π/2 * R，分配为过渡段 + 定曲率段
        total_corner = max(6, int(np.round((0.5 * np.pi * self._sq_R) / (self._sq_v * Ts))))
        n_trans = max(2, total_corner // 6)
        n_turn = max(2, total_corner - 2 * n_trans)

        # w_peak 使 clothoid-in + arc + clothoid-out 总转角恰好 90°
        # 过渡段平均 w = w_peak/2, 总计: w_peak * Ts * (n_trans + n_turn) = π/2
        w_peak = self._sq_dir * (0.5 * np.pi) / (Ts * (n_trans + n_turn))
        v_turn = abs(w_peak) * self._sq_R

        if abs(w_peak) > w_max:
            w_peak = self._sq_dir * w_max
            v_turn = w_max * self._sq_R
        if v_turn > v_max:
            v_turn = v_max

        # 3. 逐段积分 — 一个完整周期: 4×(直边 + 过渡入 + 圆弧 + 过渡出)
        segs = []
        for _ in range(4):
            segs.append(('straight',  n_straight, v_straight, 0.0))
            segs.append(('ramp_in',  n_trans,    v_turn,     w_peak))
            segs.append(('turn',     n_turn,     v_turn,     w_peak))
            segs.append(('ramp_out', n_trans,    v_turn,     w_peak))

        cycle_len = sum(n for _, n, _, _ in segs)
        Xc = np.zeros((cycle_len, 3))
        Ur = np.zeros((cycle_len, 2))
        state = np.array([0.0, 0.0, 0.0])  # 中心路径: 原点, 朝东

        idx = 0
        for seg_type, n_steps, v_seg, w_seg in segs:
            for k in range(n_steps):
                Xc[idx] = state

                if seg_type == 'straight':
                    w = 0.0
                elif seg_type == 'turn':
                    w = w_seg
                elif seg_type == 'ramp_in':
                    w = w_seg * (k / n_steps)
                else:  # ramp_out
                    w = w_seg * (1.0 - k / n_steps)

                Ur[idx] = [v_seg, w]

                # unicycle Euler 积分 (Ts=0.05 足够精确)
                state = state + Ts * np.array([
                    v_seg * np.cos(state[2]),
                    v_seg * np.sin(state[2]),
                    w
                ])
                state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))
                idx += 1

        # 4. 变换到前端轴坐标: x_f = x_c + α·cosθ, y_f = y_c + α·sinθ
        Xr = Xc.copy()
        Xr[:, 0] = Xc[:, 0] + alpha * np.cos(Xc[:, 2])
        Xr[:, 1] = Xc[:, 1] + alpha * np.sin(Xc[:, 2])

        # 5. 居中
        x_min, x_max = Xr[:, 0].min(), Xr[:, 0].max()
        y_min, y_max = Xr[:, 1].min(), Xr[:, 1].max()
        Xr[:, 0] -= (x_min + x_max) / 2.0
        Xr[:, 1] -= (y_min + y_max) / 2.0

        # 6. 随机起始相位
        start_offset = self.rng.randint(0, cycle_len)
        Xr = np.roll(Xr, -start_offset, axis=0)
        Ur = np.roll(Ur, -start_offset, axis=0)

        # 7. 存储
        self._sq_Xr_cycle = Xr
        self._sq_Ur_cycle = Ur
        self._sq_cycle_len = cycle_len
        self._cx = 0.0
        self._cy = 0.0
        self._init_theta = float(Xr[0, 2])

    def _verify_and_center_trajectory(self, max_attempts: int = 100):
        """预模拟完整轨迹，计算居中偏移并验证边界。

        use_defaults 模式下只做验证（失败则抛异常），不重采样。
        方形由 _build_square_path 解析构造，跳过预模拟。
        """
        if self.family == 'square':
            return
        for attempt in range(1 if self.use_defaults else max_attempts):
            if self.family == 'circular':
                R = self.v_const / abs(self.w_const)
                if R > self.pos_bound or R < 0.4 * self.pos_bound:
                    if self.use_defaults:
                        raise ValueError(f'circular defaults exceed pos_bound={self.pos_bound}')
                    self._init_params()
                    self._init_theta = self.rng.uniform(-0.1, 0.1)
                    continue
                self._cx = self.v_const * np.sin(self._init_theta) / self.w_const
                self._cy = -self.v_const * np.cos(self._init_theta) / self.w_const
                return

            x, y, theta = 0.0, 0.0, self._init_theta
            x_min, x_max = 0.0, 0.0
            y_min, y_max = 0.0, 0.0
            for step_idx in range(SIM_STEPS):
                u_r = self._gen_func(step_idx * self.Ts)
                x, y, theta = _rk4_front_axle(x, y, theta, u_r[0], u_r[1], self.Ts)
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)
            self._cx = -(x_min + x_max) / 2.0
            self._cy = -(y_min + y_max) / 2.0
            hw = (x_max - x_min) / 2.0
            hh = (y_max - y_min) / 2.0

            if hw > self.pos_bound or hh > self.pos_bound:
                if self.use_defaults:
                    raise ValueError(f'{self.family} defaults exceed pos_bound={self.pos_bound}')
                self._init_params()
                self._init_theta = self.rng.uniform(-0.1, 0.1)
                continue
            if max(hw, hh) < 0.4 * self.pos_bound:
                if self.use_defaults:
                    raise ValueError(f'{self.family} defaults too small for pos_bound={self.pos_bound}')
                self._init_params()
                self._init_theta = self.rng.uniform(-0.1, 0.1)
                continue
            if self.family == 'spiral':
                N_turns = (SIM_TIME * self._spiral_v
                           * np.log(self._spiral_Rmax / self._spiral_R0)
                           / (2.0 * np.pi * (self._spiral_Rmax - self._spiral_R0)))
                if N_turns < 1.8:
                    if self.use_defaults:
                        raise ValueError(f'spiral defaults: N_turns={N_turns:.1f} < 1.8')
                    self._init_params()
                    self._init_theta = self.rng.uniform(-0.1, 0.1)
                    continue
            return

        raise RuntimeError(
            f'{self.family}: 无法在 {max_attempts} 次内生成不超界的轨迹参数')

    def reset(self):
        """重置位姿到居中起始点（轨迹 bounding box 中心对齐原点）"""
        if self.family == 'square':
            self._sq_idx = 0
            self._x_r, self._y_r, self._theta_r = self._sq_Xr_cycle[0]
            return
        self._x_r = self._cx
        self._y_r = self._cy
        self._theta_r = self._init_theta
        if self.family == 'random_waypoint':
            self._waypoint_next_switch = 0.0
            self._waypoint_idx = -1

    # ---- 各轨迹族的 step 实现 ----

    def _step_lissajous(self, t: float):
        v_r = self.v_const
        w_r = self.w_amp * np.cos(self.w_freq * t)
        return np.array([v_r, w_r])

    def _step_circular(self, t: float):
        return np.array([self.v_const, self.w_const])

    def _step_spiral(self, t: float):
        """阿基米德螺旋线: R(t)=R0*(1+alpha*t), w(t)=v/R(t)"""
        R = self._spiral_R0 * (1.0 + self._spiral_alpha * t)
        v_r = self._spiral_v
        w_r = self._spiral_dir * v_r / R
        w_r = float(np.clip(w_r, -1.50, 1.50))
        return np.array([v_r, w_r])

    def _step_random_waypoint(self, t: float):
        """ω_r 方波切换，模拟路径点之间的折线运动"""
        if t >= self._waypoint_next_switch:
            self._waypoint_idx += 1
            while self._waypoint_idx >= len(self._waypoint_w_values):
                self._waypoint_w_values.append(
                    self.rng.uniform(-0.50, 0.50))
                self._waypoint_intervals.append(
                    self.rng.uniform(1.0, 3.5))
            self._waypoint_next_switch = t + self._waypoint_intervals[self._waypoint_idx]
        w_r = self._waypoint_w_values[self._waypoint_idx]
        return np.array([self.v_const, w_r])

    # ---- 统一接口 ----

    def step(self, t: float):
        """单步推进参考轨迹

        Returns:
            Upsilon_r: 参考位姿 [x_r, y_r, theta_r] (3,)
            u_r:       参考指令 [v_r, w_r] (2,)
        """
        if self.family == 'square':
            i = self._sq_idx % self._sq_cycle_len
            u_r = self._sq_Ur_cycle[i].copy()
            self._x_r, self._y_r, self._theta_r = self._sq_Xr_cycle[i]
            self._sq_idx += 1
            return np.array([self._x_r, self._y_r, self._theta_r]), u_r

        u_r = self._gen_func(t)
        self._x_r, self._y_r, self._theta_r = _rk4_front_axle(
            self._x_r, self._y_r, self._theta_r,
            u_r[0], u_r[1], self.Ts)
        return np.array([self._x_r, self._y_r, self._theta_r]), u_r

    def generate_sequence(self, t_start: float, N: int) -> np.ndarray:
        """生成未来 N 步参考指令序列 (用于 MPC)"""
        if self.family == 'square':
            Ur = np.zeros((2, N))
            for k in range(N):
                Ur[:, k] = self._sq_Ur_cycle[(self._sq_idx + k) % self._sq_cycle_len]
            return Ur

        Ur_seq = np.zeros((2, N))
        saved = {}
        if self.family == 'random_waypoint':
            saved = {'idx': self._waypoint_idx, 'next': self._waypoint_next_switch}
        for k in range(N):
            Ur_seq[:, k] = self._gen_func(t_start + k * self.Ts)
        if self.family == 'random_waypoint':
            self._waypoint_idx = saved['idx']
            self._waypoint_next_switch = saved['next']
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
            info.update(side=self._sq_side, R=self._sq_R, v=self._sq_v,
                       direction=self._sq_dir)
        return info

# ============================================================================
# 3. WMR 前端位姿运动学
# ============================================================================

class WMRKinematics:
    """两轮差速WMR前端位姿(head posture)运动学模型
    
    状态: Upsilon_h = [x_h, y_h, theta_h]
    控制: u = [v_c, w_c]
    
    连续运动学方程:
      d(Upsilon_h)/dt = F_h(theta_h) * u
      F_h = [cos(th)  -alpha*sin(th)
             sin(th)   alpha*cos(th)
             0         1            ]
    
    误差动力学:
      X = [x_e, y_e, theta_e] 在机器人本体坐标系下
      dX/dt = f(X, u_r) + G(X) * u
    """

    def __init__(self, params: WMRParams):
        self.p = params
        self.state = np.zeros(3)  # [x_h, y_h, theta_h]

    def reset(self, init_state: np.ndarray = None):
        """重置机器人状态~
        
        Args:
            init_state: 初始位姿 [x, y, theta]，默认 [0, 0.1, 0]
        """
        if init_state is None:
            self.state = np.array([0.0, 0.1, 0.0])
        else:
            self.state = init_state.copy()

    def input_matrix(self, theta: float) -> np.ndarray:
        """计算输入矩阵 F_h(theta) (论文 Eq.2)"""
        return np.array([
            [np.cos(theta), -self.p.alpha * np.sin(theta)],
            [np.sin(theta),  self.p.alpha * np.cos(theta)],
            [0.0,            1.0]
        ])

    def kinematics_rhs(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """运动学右侧: d(Upsilon_h)/dt = F_h(theta) * u"""
        theta = state[2]
        Fh = self.input_matrix(theta)
        return Fh @ u

    @staticmethod
    def kinematic_predict(state: np.ndarray, u_cmd: np.ndarray,
                          Ts: float = 0.05, alpha: float = 0.17) -> np.ndarray:
        """内部运动学 Euler 一步预测 (与检测器内部运动学一致)

        WMR 前端位姿运动学:
          dX/dt = F_h(theta) * u
          F_h = [[cos(θ),  -α·sin(θ)],
                 [sin(θ),   α·cos(θ)],
                 [0,        1        ]]

        Args:
            state: 当前状态 (3,) [x, y, theta]
            u_cmd: 控制指令 (2,) [v, w]
            Ts:    采样周期 [s] (默认 0.05)
            alpha: 前端偏移量 [m] (默认 0.17)

        Returns:
            next_state: 预测状态 (3,)
        """
        v, w = u_cmd[0], u_cmd[1]
        theta = state[2]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = v * cos_t - alpha * w * sin_t
        dy = v * sin_t + alpha * w * cos_t
        return state + Ts * np.array([dx, dy, w])

    def rk4_step(self, state: np.ndarray, u: np.ndarray,
                 Ts: float = None) -> np.ndarray:
        """RK4 一步积分
        
        Args:
            state: 当前状态 (3,)
            u:     控制输入 (2,)
            Ts:    步长，默认使用 params.Ts
            
        Returns:
            next_state: 下一步状态 (3,)
        """
        if Ts is None:
            Ts = self.p.Ts
        h = Ts
        k1 = self.kinematics_rhs(state, u)
        k2 = self.kinematics_rhs(state + h/2 * k1, u)
        k3 = self.kinematics_rhs(state + h/2 * k2, u)
        k4 = self.kinematics_rhs(state + h * k3, u)
        next_state = state + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        # 角度归一化
        next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))
        return next_state

    def step(self, u: np.ndarray) -> np.ndarray:
        """执行一步仿真，更新内部状态

        Args:
            u: 控制输入 [v, w] (2,)

        Returns:
            更新后的状态 (3,)
        """
        u_clamped = self.clamp_control(u)  # 防御性限幅
        self.state = self.rk4_step(self.state, u_clamped)
        # 空间安全边界: 位置限幅在 ±pos_bound 内
        self.state[0] = np.clip(self.state[0], -self.p.pos_bound, self.p.pos_bound)
        self.state[1] = np.clip(self.state[1], -self.p.pos_bound, self.p.pos_bound)
        return self.state

    # ---------- 坐标变换 ----------

    @staticmethod
    def compute_error(Upsilon_r: np.ndarray, Upsilon_h: np.ndarray) -> np.ndarray:
        """计算机器人本体坐标系下的跟踪误差
        
        Args:
            Upsilon_r: 参考位姿 [x_r, y_r, theta_r]
            Upsilon_h: 实际位姿 [x_h, y_h, theta_h]
            
        Returns:
            X_error: [x_e, y_e, theta_e] 在机器人坐标系下
        """
        x_r, y_r, theta_r = Upsilon_r
        x_h, y_h, theta_h = Upsilon_h

        # 旋转变换矩阵 R(theta_h)
        c = np.cos(theta_h)
        s = np.sin(theta_h)

        # 位置误差 (论文 Eq.5)
        dx = x_r - x_h
        dy = y_r - y_h
        x_e = c * dx + s * dy
        y_e = -s * dx + c * dy

        # 角度误差 (论文 Eq.6)
        theta_e = theta_r - theta_h
        theta_e = np.arctan2(np.sin(theta_e), np.cos(theta_e))

        return np.array([x_e, y_e, theta_e])

    @staticmethod
    def clamp_control(u: np.ndarray, v_max: float = 0.3, w_max: float = 1.76) -> np.ndarray:
        """控制输入限幅 (独立轴盒约束)"""
        u_clamped = u.copy()
        u_clamped[0] = np.clip(u_clamped[0], -v_max, v_max)
        u_clamped[1] = np.clip(u_clamped[1], -w_max, w_max)
        return u_clamped


# ============================================================================
# 4. 传感器模拟器
# ============================================================================


class SensorSimulator:
    """传感器模拟器：在真实状态上叠加高斯噪声（可选）

    噪声参数: 默认关闭 (无噪声)，可通过 noise_std 显式开启。
    """

    def __init__(self, noise_std: np.ndarray = None):
        if noise_std is None:
            self.noise_std = np.array([0.0, 0.0, 0.0])
        else:
            self.noise_std = noise_std

    def measure(self, true_state: np.ndarray) -> np.ndarray:
        """生成含噪测量值 (角度自动归一化)"""
        noise = self.noise_std * np.random.randn(3)
        y_meas = true_state + noise
        y_meas[2] = np.arctan2(np.sin(y_meas[2]), np.cos(y_meas[2]))
        return y_meas


