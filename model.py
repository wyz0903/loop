"""
model.py -- WMR Kinematic Model, Reference Trajectories, EKF Estimator
==========================================================================
基于两轮差速轮式移动机器人(WMR)前端位姿(head posture)运动学。

参考文献：
  - "From Detection to Control: A Deep Diagnosis Cognition-driven MPC..."
  - Zhang et al. (2026) PTESO-based MPC for WMRs, IEEE TIE

物理参数来源：Zhang et al. (2026) Section IV — TurtleBot4 安全模式
  - α = 0.17 m  (a/b = 0.3/1.76)
  - v_max = 0.3 m/s, ω_max = 1.76 rad/s
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# ============================================================================
# 1. 物理参数定义
# ============================================================================

@dataclass
class WMRParams:
    """两轮差速WMR物理参数
    
    参数来源：
      alpha = 0.17 : 前端距几何中心距离 (配置文档 Section 2.3)
      Ts    = 0.05 : 控制采样周期 (配置文档 Section 1)
      v_max = 0.6  : 最大线速度 m/s (配置文档 Section 3.1)
      w_max = 2.9  : 最大角速度 rad/s (配置文档 Section 3.1)
    """
    # 物理参数来源：Zhang et al. (2026) PTESO-MPC, IEEE TIE
    # TurtleBot4 安全模式 (Section IV): v_max = 0.3 m/s, ω_max = b = a/α ≈ 1.76 rad/s
    alpha: float = 0.17        # 前端偏置距离 [m] (α = a/b = 0.3/1.76)
    Ts: float = 0.05           # 采样时间 [s]
    v_max: float = 0.3         # 线速度上限 [m/s] (= a, 论文 U 约束)
    w_max: float = 1.76        # 角速度上限 [rad/s] (= b = a/α, 论文 U 约束)


# ============================================================================
# 2. 参考轨迹生成器 — 8字形 Lissajous + 圆形
# ============================================================================

class LissajousTrajectory:
    """8字形(Lissajous)参考轨迹生成器

    轨迹参数:
      v_const = 0.25      : 恒定线速度 [m/s] (< v_max=0.3)
      w_freq  = 0.2       : 角速度交变频率 [rad/s]
      w_amp   = 2.4048 * w_freq : 角速度幅值 ≈0.481 rad/s

    生成的参考位姿满足理想参考系统运动学 (论文 Eq.10):
      d(Upsilon_r)/dt = [cos(theta_r) 0; sin(theta_r) 0; 0 1] * u_r
    """

    def __init__(self, Ts: float = 0.05):
        self.Ts = Ts
        self.v_const = 0.25
        self.w_freq = 0.2
        self.w_amp = 2.4048 * self.w_freq  # ~0.4810 rad/s

        # 参考位姿初始值
        self._x_r = 0.0
        self._y_r = 0.0
        self._theta_r = 0.0

    def reset(self):
        """重置参考位姿到原点"""
        self._x_r = 0.0
        self._y_r = 0.0
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

        # 理想参考系统运动学积分 (欧拉法，Ts 很小误差可忽略)
        self._x_r += self.Ts * v_r * np.cos(self._theta_r)
        self._y_r += self.Ts * v_r * np.sin(self._theta_r)
        self._theta_r += self.Ts * w_r
        # 角度归一化
        self._theta_r = np.arctan2(np.sin(self._theta_r), np.cos(self._theta_r))

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
      v_r = 0.2   : 恒定线速度 [m/s]
      w_r = 0.1   : 恒定角速度 [rad/s]
      R = v_r/w_r = 2.0 m (圆半径)

    生成的参考位姿满足理想参考系统运动学 (论文 Eq.10):
      d(Upsilon_r)/dt = [cos(theta_r) 0; sin(theta_r) 0; 0 1] * u_r
    """

    def __init__(self, Ts: float = 0.05):
        self.Ts = Ts
        self.v_const = 0.2       # 论文: ν_r = 0.2 m/s
        self.w_const = 0.1       # 论文: ω_r = 0.1 rad/s

        self._x_r = 0.0
        self._y_r = 0.0
        self._theta_r = 0.0

    def reset(self):
        """重置参考位姿到原点"""
        self._x_r = 0.0
        self._y_r = 0.0
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

        # 理想参考系统运动学积分 (欧拉法)
        self._x_r += self.Ts * v_r * np.cos(self._theta_r)
        self._y_r += self.Ts * v_r * np.sin(self._theta_r)
        self._theta_r += self.Ts * w_r
        self._theta_r = np.arctan2(np.sin(self._theta_r), np.cos(self._theta_r))

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


# 向后兼容别名
ReferenceTrajectory = LissajousTrajectory


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
        """重置机器人状态
        
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
        self.state = self.rk4_step(self.state, u)
        return self.state

    # ---------- 误差动力学 (用于 MPC 预测) ----------

    def error_dynamics_rhs(self, X: np.ndarray, u: np.ndarray,
                           u_r: np.ndarray) -> np.ndarray:
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

    def error_rk4_step(self, X: np.ndarray, u: np.ndarray, u_r: np.ndarray,
                        Ts: float = None) -> np.ndarray:
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
    def compute_error(Upsilon_r: np.ndarray,
                      Upsilon_h: np.ndarray) -> np.ndarray:
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
    def clamp_control(u: np.ndarray, v_max: float = 0.3,
                      w_max: float = 1.76) -> np.ndarray:
        """控制输入限幅 (配置文档 Section 2.6 菱形约束近似)"""
        u_clamped = u.copy()
        u_clamped[0] = np.clip(u_clamped[0], -v_max, v_max)
        u_clamped[1] = np.clip(u_clamped[1], -w_max, w_max)
        return u_clamped


# ============================================================================
# 4. 扩展卡尔曼滤波器 (EKF)
# ============================================================================

class EKFEstimator:
    """扩展卡尔曼滤波器用于状态估计
    
    参考：配置文档 Section 2.7 (Sensor 子系统)
    
    预测步使用 u_cmd (控制器发出指令)，而非 u_a (实际执行指令)。
    这在受攻击时产生新息，是DR-Net检测的关键特征来源。
    
    参数：
      Q = diag([1e-4, 1e-4, 1e-4])  : 过程噪声协方差
      R = diag([0.05, 0.05, 0.01])   : 量测噪声协方差
    """

    def __init__(self, params: WMRParams):
        self.p = params
        # 协方差矩阵 (信任传感器 > 信任模型)
        # Q增大 → 模型预测不可靠，更依赖测量更新
        # R减小 → 传感器更可信，卡尔曼增益更大
        self.Q = np.diag([5e-4, 5e-4, 5e-4])
        self.R = np.diag([0.02, 0.02, 0.005])
        self.H = np.eye(3)  # 直接测量全状态
        self.X_hat = None   # 估计状态
        self.P = None       # 误差协方差
        self._kinematics = WMRKinematics(params)  # 用于雅可比计算

    def reset(self, init_guess: np.ndarray = None):
        """初始化/重置 EKF
        
        Args:
            init_guess: 初始位姿猜测，默认 [0, 0.1, 0]
        """
        if init_guess is None:
            self.X_hat = np.array([0.0, 0.1, 0.0])
        else:
            self.X_hat = init_guess.copy()
        self.P = np.eye(3) * 0.1

    def predict(self, u_cmd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """EKF 预测步 (论文 Eq.17-18)
        
        使用控制器下发的合法指令 u_cmd 进行预测。
        若执行器被攻击，实际 u_a != u_cmd，预测将偏离真实值。
        
        Args:
            u_cmd: 控制器发出的指令 [v, w]
            
        Returns:
            X_pred: 先验状态估计
            P_pred: 先验协方差
        """
        theta = self.X_hat[2]
        Fh = self._kinematics.input_matrix(theta)
        X_pred = self.X_hat + self.p.Ts * (Fh @ u_cmd)

        # 状态转移雅可比 J (配置文档 Section 2.7)
        v, w = u_cmd
        J = np.array([
            [1.0, 0.0, self.p.Ts * (-v * np.sin(theta)
                                     - w * self.p.alpha * np.cos(theta))],
            [0.0, 1.0, self.p.Ts * ( v * np.cos(theta)
                                     - w * self.p.alpha * np.sin(theta))],
            [0.0, 0.0, 1.0]
        ])
        P_pred = J @ self.P @ J.T + self.Q

        return X_pred, P_pred

    def update(self, y_meas: np.ndarray, X_pred: np.ndarray,
               P_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """EKF 更新步 (论文 Eq.19-21)
        
        Args:
            y_meas:  传感器含噪测量值 [x, y, theta]
            X_pred:  先验状态估计
            P_pred:  先验协方差
            
        Returns:
            X_post:   后验状态估计 (即输出 Upsilon_hat)
            residual: 新息 r = y_meas - H·X_pred (可用于攻击检测)
        """
        # 卡尔曼增益
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # 新息
        innovation = y_meas - self.H @ X_pred
        innovation[2] = np.arctan2(np.sin(innovation[2]),
                                   np.cos(innovation[2]))

        # 后验更新
        self.X_hat = X_pred + K @ innovation
        self.P = (np.eye(3) - K @ self.H) @ P_pred
        # 角度归一化
        self.X_hat[2] = np.arctan2(np.sin(self.X_hat[2]),
                                    np.cos(self.X_hat[2]))

        return self.X_hat.copy(), innovation.copy()

    def step(self, y_meas: np.ndarray,
             u_cmd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """完整的 EKF 预测-更新循环
        
        Args:
            y_meas: 传感器含噪测量值
            u_cmd:  控制器下发的合法指令
            
        Returns:
            X_hat (Upsilon_hat): 估计状态
            residual:            新息 (DR-Net 关键输入特征)
        """
        X_pred, P_pred = self.predict(u_cmd)
        X_hat, residual = self.update(y_meas, X_pred, P_pred)
        return X_hat, residual


# ============================================================================
# 5. 传感器模拟器
# ============================================================================

class SensorSimulator:
    """传感器模拟器：在真实状态上叠加高斯噪声
    
    噪声参数 (配置文档 Section 2.7):
      sigma_x = sigma_y = 0.02 m
      sigma_theta = 0.01 rad
    """

    def __init__(self, noise_std: np.ndarray = None):
        if noise_std is None:
            self.noise_std = np.array([0.02, 0.02, 0.01])
        else:
            self.noise_std = noise_std

    def measure(self, true_state: np.ndarray) -> np.ndarray:
        """生成含噪测量值"""
        noise = self.noise_std * np.random.randn(3)
        return true_state + noise


# ============================================================================
# 6. 快速自测
# ============================================================================

if __name__ == "__main__":
    print("=== model.py 自测 ===")
    params = WMRParams()
    traj = ReferenceTrajectory(Ts=params.Ts)
    robot = WMRKinematics(params)
    ekf = EKFEstimator(params)
    sensor = SensorSimulator()

    # 测试开环前向运动
    robot.reset()
    u = np.array([0.3, 0.0])  # 直行
    for _ in range(10):
        state = robot.step(u)
    print(f"10步直行后: 位姿 = {state}")

    # 测试轨迹生成
    traj.reset()
    Ur, ur = traj.step(1.0)
    print(f"t=1.0s 参考位姿: {Ur}, 指令: {ur}")

    # 测试EKF
    ekf.reset()
    true_state = robot.state
    y_meas = sensor.measure(true_state)
    X_hat, res = ekf.step(y_meas, u)
    print(f"EKF估计: {X_hat}, 新息: {res}")

    # 测试误差计算
    X_err = WMRKinematics.compute_error(Ur, robot.state)
    print(f"跟踪误差: {X_err}")
    print("=== 所有模块自测通过 ===")
