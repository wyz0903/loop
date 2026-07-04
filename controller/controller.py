"""
controller.py -- Fixed NMPC Controller for WMR Trajectory Tracking
==========================================================================
基于 CasADi Opti 构建非线性模型预测控制器。

控制策略：
  - 全非线性误差动力学 RK4 预测 (无线性化，适配大偏差)
  - 菱形控制约束 (|v/v_max| + |w/w_max| <= 1)
  - 控制增量惩罚 (Delta u) 实现平滑切换
  - 固定权重，不做自适应调整

参数来源：配置文档 Section 3.1, 仿真系统配置说明文档

使用方法：
  python controller.py          # 编译并保存 .casadi 求解器文件
"""

import os
import numpy as np
import casadi as ca


# ============================================================================
# 控制器参数 (配置文档 Section 3.1)
# ============================================================================

class NMPCParams:
    """NMPC 控制器超参数
    
    所有数值来源见配置文档 Section 3.1
    """
    N: int = 30                # 预测步长 (调优: 20→30, 1.0s→1.5s, review_03)
    Ts: float = 0.05           # 采样时间 [s]
    alpha: float = 0.17        # 前端偏置 [m]

    # 状态与输入权重 (调优结果: Q_xy=0.15, R=0.00125, review_02)
    # 调优前 Q=diag([0.1,0.1,0.01]), R=diag([0.002,0.002]), Avg RMS_xy=0.0381
    # 调优后 Q=diag([0.15,0.15,0.01]), R=diag([0.00125,0.00125]), Avg RMS_xy=0.0272
    Q: np.ndarray = np.diag([0.15, 0.15, 0.01])
    R: np.ndarray = np.diag([0.00125, 0.00125])

    # 控制约束
    v_max: float = 0.3         # 线速度上限 [m/s] (= a)
    w_max: float = 1.76        # 角速度上限 [rad/s] (= b = a/α)

    # 状态约束
    x_bound: float = 2.5       # 位置误差边界 [m]

    # IPOPT 求解器设置
    max_iter: int = 100
    acceptable_tol: float = 1e-4
    print_level: int = 0


# ============================================================================
# NMPC 问题构建器
# ============================================================================

class NMPCBuilder:
    """基于 CasADi Opti 构建 NMPC 优化问题并编译为独立求解器文件
    
    构建流程：
      1. 定义符号变量 (状态序列 X, 控制序列 U)
      2. 构建误差动力学 RK4 离散化
      3. 施加菱形约束和状态边界
      4. 定义增量代价函数
      5. 编译为 IPOPT 求解器 -> 保存为 .casadi 文件
    """

    def __init__(self, params: NMPCParams = None):
        self.p = params if params is not None else NMPCParams()

    def _build_error_dynamics_rk4(self):
        """构建误差动力学的 RK4 离散化 CasADi Function
        
        误差动力学 (论文 Eq.10):
          dX/dt = f(X, u_r) + G(X) * u
          f = [v_r*cos(theta_e), v_r*sin(theta_e), w_r]^T
          G = [[-1, y_e], [0, -x_e-alpha], [0, -1]]
        """
        p = self.p
        X_sym = ca.MX.sym('X', 3, 1)    # [x_e, y_e, theta_e]
        u_sym = ca.MX.sym('u', 2, 1)    # [v, w]
        ur_sym = ca.MX.sym('ur', 2, 1)  # [v_r, w_r]

        x_e, y_e, theta_e = X_sym[0], X_sym[1], X_sym[2]
        v_r, w_r = ur_sym[0], ur_sym[1]

        # 自然动态 f(X, u_r)
        f_X = ca.vertcat(
            v_r * ca.cos(theta_e),
            v_r * ca.sin(theta_e),
            w_r
        )
        # 输入矩阵 G(X)
        G_X = ca.horzcat(
            ca.vertcat(-1.0, 0.0, 0.0),
            ca.vertcat(y_e, -x_e - p.alpha, -1.0)
        )
        X_dot = f_X + ca.MX(G_X) @ u_sym

        # 构建 CasADi Function: f_dyn(X, u, ur) -> X_dot
        f_dyn = ca.Function('f_dyn', [X_sym, u_sym, ur_sym], [X_dot])

        # RK4 离散化: F_RK4(X, u, ur) -> X_next
        h = p.Ts
        k1 = f_dyn(X_sym, u_sym, ur_sym)
        k2 = f_dyn(X_sym + h/2 * k1, u_sym, ur_sym)
        k3 = f_dyn(X_sym + h/2 * k2, u_sym, ur_sym)
        k4 = f_dyn(X_sym + h * k3, u_sym, ur_sym)
        X_next_raw = X_sym + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        # 角度归一化: 防止 θ_e 在预测范围内偏离 [-π, π] 导致代价函数惩罚失准
        X_next = ca.vertcat(
            X_next_raw[0],
            X_next_raw[1],
            ca.atan2(ca.sin(X_next_raw[2]), ca.cos(X_next_raw[2]))
        )

        self._F_RK4 = ca.Function('F_RK4', [X_sym, u_sym, ur_sym], [X_next])

    def build(self) -> ca.Function:
        """构建并返回 NMPC 求解器 CasADi Function
        
        Returns:
            nmpc_solver: Function(X0, U_prev, Ur_seq) -> U_opt
              输入: X0 (3,), U_prev (2,), Ur_seq (2, N)
              输出: U_opt (2,) -- 第一个最优控制动作
        """
        self._build_error_dynamics_rk4()
        p = self.p
        N = p.N

        # ---- Opti 优化问题 ----
        opti = ca.Opti()

        # 决策变量
        X = opti.variable(3, N + 1)    # 状态序列
        U = opti.variable(2, N)        # 控制序列

        # 参数
        X0 = opti.parameter(3, 1)      # 当前误差状态
        Ur_seq = opti.parameter(2, N)  # 参考指令序列
        U_prev = opti.parameter(2, 1)  # 上一时刻控制指令

        # ---- 初始条件约束 ----
        opti.subject_to(X[:, 0] == X0)

        # ---- 代价函数与动力学约束 ----
        J = 0
        for k in range(N):
            # 动力学约束: X_{k+1} = F_RK4(X_k, U_k, Ur_k)
            opti.subject_to(
                X[:, k + 1] == self._F_RK4(X[:, k], U[:, k], Ur_seq[:, k])
            )

            # 状态代价
            J += X[:, k + 1].T @ p.Q @ X[:, k + 1]

            # 控制增量代价 (论文 Eq.27 无扰切换)
            if k == 0:
                delta_u = U[:, 0] - U_prev
            else:
                delta_u = U[:, k] - U[:, k - 1]
            J += delta_u.T @ p.R @ delta_u

            # ---- 菱形控制约束 (配置文档 Eq.3.1) ----
            opti.subject_to(
                U[0, k] / p.v_max + U[1, k] / p.w_max <= 1
            )
            opti.subject_to(
                -U[0, k] / p.v_max + U[1, k] / p.w_max <= 1
            )
            opti.subject_to(
                U[0, k] / p.v_max - U[1, k] / p.w_max <= 1
            )
            opti.subject_to(
                -U[0, k] / p.v_max - U[1, k] / p.w_max <= 1
            )

            # ---- 状态约束 ----
            opti.subject_to(X[0, k] >= -p.x_bound)
            opti.subject_to(X[0, k] <=  p.x_bound)
            opti.subject_to(X[1, k] >= -p.x_bound)
            opti.subject_to(X[1, k] <=  p.x_bound)

        opti.minimize(J)

        # ---- IPOPT 求解器配置 ----
        p_opts = {'expand': True, 'print_time': False}
        s_opts = {
            'max_iter': p.max_iter,
            'print_level': p.print_level,
            'acceptable_tol': p.acceptable_tol,
            'sb': 'yes'
        }
        opti.solver('ipopt', p_opts, s_opts)

        # ---- 导出为 Function ----
        M = opti.to_function(
            'nmpc_solver',
            [X0, U_prev, Ur_seq],
            [U[:, 0]],
            ['X0', 'U_prev', 'Ur_seq'],
            ['U_opt']
        )
        return M


# ============================================================================
# NMPC 控制器接口
# ============================================================================

class NMPCController:
    """固定 NMPC 跟踪控制器
    
    封装 CasADi 编译求解器的加载与调用。
    
    使用方法:
      ctrl = NMPCController()
      ctrl.load_or_build()           # 首次运行编译，后续直接加载
      u_opt = ctrl.solve(X_error, u_prev, Ur_seq)
    """

    def __init__(self, params: NMPCParams = None,
                 solver_path: str = None):
        self.p = params if params is not None else NMPCParams()
        if solver_path is None:
            solver_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'nmpc_solver.casadi'
            )
        self.solver_path = solver_path
        self._solver = None
        self._u_prev = np.zeros(2)  # 上一时刻控制指令缓存

    def load_or_build(self, force_rebuild: bool = False):
        """加载已有求解器，或构建新的

        Args:
            force_rebuild: 强制重新编译
        """
        import hashlib
        import json
        hash_path = self.solver_path + '.hash'

        # 计算当前参数哈希
        current_hash = hashlib.md5(json.dumps({
            'N': self.p.N,
            'Q': self.p.Q.flatten().tolist(),
            'R': self.p.R.flatten().tolist(),
            'v_max': self.p.v_max,
            'w_max': self.p.w_max,
            'x_bound': self.p.x_bound,
            'Ts': self.p.Ts,
            'alpha': self.p.alpha,
        }, sort_keys=True).encode()).hexdigest()

        if not force_rebuild and os.path.exists(self.solver_path):
            # 参数哈希验证: 检测参数变更，自动触发重新编译
            if os.path.exists(hash_path):
                with open(hash_path, 'r') as f:
                    saved_hash = f.read().strip()
                if current_hash == saved_hash:
                    print(f"[NMPC] 加载已有求解器: {self.solver_path}")
                    self._solver = ca.Function.load(self.solver_path)
                    self._warmup()
                    return
                else:
                    print(f"[NMPC] 参数已变更，重新编译...")
            else:
                print(f"[NMPC] 无哈希文件，重新编译以确保一致性...")

        print("[NMPC] 正在构建 CasADi 优化问题并编译...")
        builder = NMPCBuilder(self.p)
        solver = builder.build()
        solver.save(self.solver_path)
        # 保存参数哈希
        with open(hash_path, 'w') as f:
            f.write(current_hash)
        print(f"[NMPC] 求解器已保存: {self.solver_path}")
        self._solver = solver

        # 预热：执行一次虚拟求解以初始化 IPOPT 内部结构
        self._warmup()

    def _warmup(self):
        """预热求解器 (首次求解通常较慢)"""
        X0_warm = np.zeros((3, 1))
        U_prev_warm = np.zeros((2, 1))
        Ur_seq_warm = np.zeros((2, self.p.N))
        Ur_seq_warm[0, :] = 0.25
        try:
            _ = self._solver(X0_warm, U_prev_warm, Ur_seq_warm)
        except Exception:
            pass  # 预热失败不影响后续使用

    def solve(self, X_error: np.ndarray, Ur_seq: np.ndarray,
              u_prev: np.ndarray = None) -> np.ndarray:
        """求解 NMPC 优化问题
        
        Args:
            X_error: 当前跟踪误差 [x_e, y_e, theta_e] (3,)
            Ur_seq:  未来 N 步参考指令序列 (2, N)
            u_prev:  上一时刻控制指令 (2,)，默认使用内部缓存
            
        Returns:
            u_opt: 最优控制指令 [v, w] (2,)
            
        Raises:
            RuntimeError: 求解器未初始化
        """
        if self._solver is None:
            raise RuntimeError("求解器未初始化，请先调用 load_or_build()")

        if u_prev is None:
            u_prev = self._u_prev

        # 输入整形
        X0 = X_error.reshape((3, 1))
        U_prev = u_prev.reshape((2, 1))
        Ur = np.asarray(Ur_seq).reshape((2, self.p.N))

        try:
            result = self._solver(X0, U_prev, Ur)
            u_opt = np.array(result).flatten()
            # 安全检查：防止求解器发散
            if np.any(np.isnan(u_opt)) or np.linalg.norm(u_opt) > 1e6:
                print("[NMPC WARN] 求解器异常，回退至上一步指令")
                u_opt = self._u_prev.copy()
            else:
                self._u_prev = u_opt.copy()
        except Exception as e:
            print(f"[NMPC ERROR] 求解失败: {e}，回退至上一步指令")
            u_opt = self._u_prev.copy()

        # 限幅
        u_opt[0] = np.clip(u_opt[0], -self.p.v_max, self.p.v_max)
        u_opt[1] = np.clip(u_opt[1], -self.p.w_max, self.p.w_max)

        return u_opt

    def reset(self):
        """重置控制器内部状态"""
        self._u_prev = np.zeros(2)


# ============================================================================
# 主入口：编译求解器
# ============================================================================

if __name__ == "__main__":
    print("=== NMPC 控制器编译 ===")
    ctrl = NMPCController()
    ctrl.load_or_build(force_rebuild=True)
    print("=== 编译完成，求解器就绪 ===")
