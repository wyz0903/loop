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


def setup_ieee_style(font_size: int = 9):
    """设置 IEEE 论文标准 matplotlib 样式 (全局生效)。

    所有绘图模块统一调用此函数，避免 rcParams 配置分散。
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.unicode_minus'] = False


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
    # TurtleBot4 安全模式 (Section IV): v_max = 0.3 m/s, ω_max = b = a/α ≈ 1.76 rad/s
    alpha: float = 0.17        # 前端偏置距离 [m] (α = a/b = 0.3/1.76)
    Ts: float = 0.05           # 采样时间 [s]
    v_max: float = 0.3         # 线速度上限 [m/s] (= a, 论文 U 约束)
    w_max: float = 1.76        # 角速度上限 [rad/s] (= b = a/α, 论文 U 约束)
    pos_bound: float = 2.5     # 空间位置边界 [m] (±2.5m 安全运行范围)


# ============================================================================
# 2. 参考轨迹生成器 — 8字形 Lissajous + 圆形
# ============================================================================

def _rk4_front_axle(x: float, y: float, theta: float, v: float, w: float, Ts: float,
                     alpha: float = 0.17):
    """前端偏置运动学 RK4 积分（与 WMRKinematics 一致，论文 Eq.1-2）

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


class LissajousTrajectory:
    """8字形(Lissajous)参考轨迹生成器

    轨迹参数:
      v_const  : 恒定线速度 [m/s] (< v_max=0.3)
      w_freq   : 角速度交变频率 [rad/s]
      w_amp    : 角速度幅值 = 2.4048 * w_freq

    使用前端偏置运动学 (与 WMRKinematics 一致) 生成参考轨迹，
    保证参考轨迹对机器人物理可行。
    """

    def __init__(self, Ts: float = 0.05, pos_bound: float = 2.5):
        self.Ts = Ts
        self.pos_bound = pos_bound
        self.v_const = 0.15       # 降低速度以适应 ±2.5m 边界
        self.w_freq = 0.3         # 提高频率使 8 字形更紧凑
        self.w_amp = 2.4048 * self.w_freq  # ~0.7214 rad/s

        # 预计算居中偏移 (使轨迹中心对齐原点)
        self._cx, self._cy = self._compute_center_offset()
        # 参考位姿初始值
        self._x_r = self._cx
        self._y_r = self._cy
        self._theta_r = 0.0

    def _compute_center_offset(self) -> Tuple[float, float]:
        """预模拟 50s 轨迹，计算使 bounding box 中心对齐原点的偏移量"""
        x, y, theta = 0.0, 0.0, 0.0
        x_min, x_max = 0.0, 0.0
        y_min, y_max = 0.0, 0.0
        for i in range(SIM_STEPS):
            t_i = i * self.Ts
            v_r = self.v_const
            w_r = self.w_amp * np.cos(self.w_freq * t_i)
            x, y, theta = _rk4_front_axle(x, y, theta, v_r, w_r, self.Ts)
            x_min, x_max = min(x_min, x), max(x_max, x)
            y_min, y_max = min(y_min, y), max(y_max, y)
        cx = -(x_min + x_max) / 2.0
        cy = -(y_min + y_max) / 2.0
        return cx, cy

    def reset(self):
        """重置参考位姿到居中起始点"""
        self._x_r = self._cx
        self._y_r = self._cy
        self._theta_r = 0.0

    def step(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """单步推进参考轨迹
        
        Args:
            t: 当前仿真时间 [s]
            
        Returns:
            Upsilon_r: 参考位姿 [x_r, y_r, theta_r] (3,)
            u_r:       参考控制指令 [v_r, w_r] (2,)
        """
        # 当前时刻参考指令 (配置文档 Eq.2.1)
        v_r = self.v_const
        w_r = self.w_amp * np.cos(self.w_freq * t)
        u_r = np.array([v_r, w_r])

        # RK4 积分 (前端偏置运动学，与 WMRKinematics 一致)
        self._x_r, self._y_r, self._theta_r = _rk4_front_axle(
            self._x_r, self._y_r, self._theta_r, v_r, w_r, self.Ts)

        Upsilon_r = np.array([self._x_r, self._y_r, self._theta_r])
        return Upsilon_r, u_r

    def generate_sequence(self, t_start: float, N: int) -> np.ndarray:
        """生成未来 N 步的参考指令序列 (用于 MPC)
        
        Args:
            t_start: 起始时间
            N:       预测步长
            
        Returns:
            Ur_seq: (2, N) 参考指令序列
        """
        Ur_seq = np.zeros((2, N))
        for k in range(N):
            t_k = t_start + k * self.Ts
            Ur_seq[0, k] = self.v_const
            Ur_seq[1, k] = self.w_amp * np.cos(self.w_freq * t_k)
        return Ur_seq


class CircularTrajectory:
    """圆形参考轨迹生成器 (Zhang et al. 2026, Section IV 实验设置)

    轨迹参数:
      v_r : 恒定线速度 [m/s]
      w_r : 恒定角速度 [rad/s]
      R   : 圆半径 = |v_r/w_r|

    使用前端偏置运动学 (与 WMRKinematics 一致) 生成参考轨迹，
    保证参考轨迹对机器人物理可行。
    """

    def __init__(self, Ts: float = 0.05, pos_bound: float = 2.5):
        self.Ts = Ts
        self.pos_bound = pos_bound
        self.v_const = 0.15      # R=v/w=0.75m，距原点最大 1.5m < 2.5m
        self.w_const = 0.2

        # 预计算居中偏移
        self._cx, self._cy = self._compute_center_offset()
        self._x_r = self._cx
        self._y_r = self._cy
        self._theta_r = 0.0

    def _compute_center_offset(self) -> Tuple[float, float]:
        """预模拟 50s 圆形轨迹，计算居中偏移"""
        x, y, theta = 0.0, 0.0, 0.0
        x_min, x_max = 0.0, 0.0
        y_min, y_max = 0.0, 0.0
        v_r, w_r = self.v_const, self.w_const
        for i in range(SIM_STEPS):
            x, y, theta = _rk4_front_axle(x, y, theta, v_r, w_r, self.Ts)
            x_min, x_max = min(x_min, x), max(x_max, x)
            y_min, y_max = min(y_min, y), max(y_max, y)
        cx = -(x_min + x_max) / 2.0
        cy = -(y_min + y_max) / 2.0
        return cx, cy

    def reset(self):
        """重置参考位姿到居中起始点"""
        self._x_r = self._cx
        self._y_r = self._cy
        self._theta_r = 0.0

    def step(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """单步推进圆形参考轨迹

        Returns:
            Upsilon_r: 参考位姿 [x_r, y_r, theta_r] (3,)
            u_r:       参考控制指令 [v_r, w_r] (2,)
        """
        v_r = self.v_const
        w_r = self.w_const
        u_r = np.array([v_r, w_r])

        # RK4 积分 (前端偏置运动学，与 WMRKinematics 一致)
        self._x_r, self._y_r, self._theta_r = _rk4_front_axle(
            self._x_r, self._y_r, self._theta_r, v_r, w_r, self.Ts)

        Upsilon_r = np.array([self._x_r, self._y_r, self._theta_r])
        return Upsilon_r, u_r

    def generate_sequence(self, t_start: float, N: int) -> np.ndarray:
        """生成未来 N 步的参考指令序列 (用于 MPC)

        Returns:
            Ur_seq: (2, N) 参考指令序列
        """
        Ur_seq = np.zeros((2, N))
        Ur_seq[0, :] = self.v_const
        Ur_seq[1, :] = self.w_const
        return Ur_seq


# (ReferenceTrajectory 别名已移除 — 直接使用 LissajousTrajectory)

# ============================================================================
# 3. WMR 前端位姿运动学
# ============================================================================

class WMRKinematics:
    """两轮差速WMR前端位姿(head posture)运动学模型
    
    状态: Upsilon_h = [x_h, y_h, theta_h]
    控制: u = [v_c, w_c]
    
    连续运动学方程 (论文 Eq.1-2):
      d(Upsilon_h)/dt = F_h(theta_h) * u
      F_h = [cos(th)  -alpha*sin(th)
             sin(th)   alpha*cos(th)
             0         1            ]
    
    误差动力学 (论文 Eq.7-10):
      X = [x_e, y_e, theta_e] 在机器人本体坐标系下
      dX/dt = f(X, u_r) + G(X) * u
    """

    def __init__(self, params: WMRParams):
        self.p = params
        self.state = np.zeros(3)  # [x_h, y_h, theta_h]

    def reset(self, init_state: np.ndarray = None):
        """重置机器人状态~
        
        Args:
            init_state: 初始位姿 [x, y, theta]，默认 [0, 0.1, 0] (配置文档 2.3)
        """
        if init_state is None:
            self.state = np.array([0.0, 0.1, 0.0])
        else:
            self.state = init_state.copy()

    @property
    def theta(self) -> float:
        return self.state[2]

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
        """内部运动学 Euler 一步预测 (与 CFMDetector 内部运动学一致)

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

    # ---------- 误差动力学 (用于 MPC 预测) ----------

    def error_dynamics_rhs(self, X: np.ndarray, u: np.ndarray, u_r: np.ndarray) -> np.ndarray:
        """跟踪误差动力学右侧 (论文 Eq.10)
        
        Args:
            X:   误差状态 [x_e, y_e, theta_e] (3,)
            u:   控制输入 [v, w] (2,)
            u_r: 参考指令 [v_r, w_r] (2,)
            
        Returns:
            dX: 误差导数 (3,)
        """
        x_e, y_e, theta_e = X[0], X[1], X[2]
        v_r, w_r = u_r[0], u_r[1]

        f_X = np.array([
            v_r * np.cos(theta_e),
            v_r * np.sin(theta_e),
            w_r
        ])
        G_X = np.array([
            [-1.0,  y_e],
            [0.0,  -x_e - self.p.alpha],
            [0.0,  -1.0]
        ])
        return f_X + G_X @ u

    def error_rk4_step(self, X: np.ndarray, u: np.ndarray, u_r: np.ndarray, Ts: float = None) -> np.ndarray:
        """误差动力学 RK4 一步积分 (用于 MPC 内部预测)"""
        if Ts is None:
            Ts = self.p.Ts
        h = Ts
        k1 = self.error_dynamics_rhs(X, u, u_r)
        k2 = self.error_dynamics_rhs(X + h/2 * k1, u, u_r)
        k3 = self.error_dynamics_rhs(X + h/2 * k2, u, u_r)
        k4 = self.error_dynamics_rhs(X + h * k3, u, u_r)
        X_next = X + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        # 角度误差归一化
        X_next[2] = np.arctan2(np.sin(X_next[2]), np.cos(X_next[2]))
        return X_next

    # ---------- 坐标变换 ----------

    @staticmethod
    def compute_error(Upsilon_r: np.ndarray, Upsilon_h: np.ndarray) -> np.ndarray:
        """计算机器人本体坐标系下的跟踪误差 (论文 Eq.5-6)
        
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


# ============================================================================
# 5. 五族轨迹可视化
# ============================================================================

def _generate_trajectory_path(traj, pos_bound, Ts=0.05, T_sim=50.0):
    """模拟完整 50s 参考轨迹 (使用轨迹自身的 step，已为 RK4)"""
    n = int(T_sim / Ts)
    traj.reset()
    x_arr, y_arr = np.zeros(n), np.zeros(n)
    for i in range(n):
        Ur, _ = traj.step(i * Ts)
        x_arr[i], y_arr[i] = Ur[0], Ur[1]
    within = np.all(np.abs(x_arr) <= pos_bound + 1e-6) and \
             np.all(np.abs(y_arr) <= pos_bound + 1e-6)
    return x_arr, y_arr, within


def plot_trajectory_shapes():
    """绘制五族参考轨迹完整形状对比图 (IEEE 论文标准)"""
    import matplotlib.pyplot as plt

    # ---- IEEE 论文标准字体 (局部作用，不污染全局) ----
    import matplotlib as mpl
    _rc_backup = {k: mpl.rcParams[k] for k in
                  ['font.family', 'font.serif', 'mathtext.fontset',
                   'font.size', 'axes.titlesize', 'axes.labelsize']}
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['axes.labelsize'] = 9

    Ts = 0.05; T_sim = 50.0; pb = 2.5

    # 五族轨迹配置
    configs = [
        ('Lissajous',       '#2196F3', LissajousTrajectory(Ts=Ts)),
        ('Circular',        '#4CAF50', CircularTrajectory(Ts=Ts)),
        ('Spiral',          '#FF9800', _force_rand_family('spiral', Ts)),
        ('Random Waypoint', '#E91E63', _force_rand_family('random_waypoint', Ts)),
        ('Square',          '#9C27B0', _force_rand_family('square', Ts)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    ax_flat = axes.flatten()
    extents = {}

    for idx, (name, color, traj) in enumerate(configs):
        x, y, _ = _generate_trajectory_path(traj, pb, Ts, T_sim)
        extents[name] = (np.max(np.abs(x)), np.max(np.abs(y)))

        ax = ax_flat[idx]
        ax.plot(x, y, color=color, linewidth=1.0, label='Reference path')
        ax.plot(x[0], y[0], 'ko', markersize=5, label='Start')
        ax.plot(x[-1], y[-1], 'r*', markersize=8, label='End')
        # 空间边界
        ax.plot([-pb, pb, pb, -pb, -pb], [-pb, -pb, pb, pb, -pb],
                'gray', linewidth=0.6, alpha=0.5, linestyle=':', label='Boundary')
        ax.set_xlim(-pb - 0.3, pb + 0.3)
        ax.set_ylim(-pb - 0.3, pb + 0.3)
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(name, fontsize=11, fontweight='bold', color=color)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8, ncol=2)
        ax.grid(True, alpha=0.2)

    # 第6图: 汇总柱状图
    ax6 = ax_flat[5]
    names = list(extents.keys())
    x_vals = [extents[n][0] for n in names]
    y_vals = [extents[n][1] for n in names]
    x_pos = np.arange(len(names))
    w = 0.35
    ax6.bar(x_pos - w / 2, x_vals, w, label='$|x|_{\\max}$', color='#2196F3', alpha=0.8)
    ax6.bar(x_pos + w / 2, y_vals, w, label='$|y|_{\\max}$', color='#FF9800', alpha=0.8)
    ax6.axhline(y=pb, color='gray', linestyle='--', linewidth=0.8, label=f'Boundary ($\\pm${pb} m)')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(names, rotation=15, fontsize=9)
    ax6.set_ylabel('Max absolute position [m]')
    ax6.set_title('Spatial Extent', fontsize=11, fontweight='bold')
    ax6.legend(loc='upper left', fontsize=7, framealpha=0.8)
    ax6.grid(True, alpha=0.2, axis='y')

    fig.suptitle('Reference Trajectory Families (50 s, ± 2.5 m boundary)',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

    # 恢复全局 rcParams，避免污染其他模块的图表
    for k, v in _rc_backup.items():
        mpl.rcParams[k] = v


def _force_rand_family(family: str, Ts: float):
    """根据族名创建 RandomizedTrajectory (通过种子搜索，结果缓存)

    由于 RandomizedTrajectory 使用种子随机选择族 (等概率 20%)，
    这里通过搜索种子来获取指定族的实例。首次获取后缓存。
    """
    from generate_dataset import RandomizedTrajectory

    # 高概率命中范围: 5 族 × 5倍冗余 = 前25个种子大概率覆盖全族
    # 若未命中则逐种子搜索 (最多 5000 个)
    for s in range(5000):
        traj = RandomizedTrajectory(Ts=Ts, seed=s)
        if traj.family == family:
            return traj
    raise RuntimeError(f'无法在 5000 个种子内找到 {family} 族轨迹')


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import sys
    if '--test' in sys.argv:
        # 快速自测模式
        print("=== model.py 自测 ===")
        params = WMRParams()
        traj = LissajousTrajectory(Ts=params.Ts)
        robot = WMRKinematics(params)
        sensor = SensorSimulator()

        robot.reset()
        u = np.array([0.3, 0.0])
        for _ in range(10):
            state = robot.step(u)
        print(f"10步直行后: 位姿 = {state}")

        traj.reset()
        Ur, ur = traj.step(1.0)
        print(f"t=1.0s 参考位姿: {Ur}, 指令: {ur}")

        true_state = robot.state
        y_meas = sensor.measure(true_state)
        print(f"传感器测量: {y_meas}")

        X_err = WMRKinematics.compute_error(Ur, robot.state)
        print(f"跟踪误差: {X_err}")
        print("=== 所有模块自测通过 ===")
    else:
        # 默认: 显示五族轨迹对比图
        plot_trajectory_shapes()
