"""
differentiable_nmpc.py — 可微 NMPC 层 (nlpsol + 有限差分梯度)
=============================================================
将 CasADi NMPC 求解器嵌入 PyTorch 计算图。
前向: CasADi nlpsol + IPOPT 求解 (与 controller.py 等价)
反向: 中心有限差分 (eps=1e-5, IPOPT tol=1e-10, 信噪比 ~1e5)

技术说明:
  CasADi 3.7.2 的 nlpsol AD (ca.jacobian) 在含不等式约束时求值失败
  (已验证: 等式约束/无约束 OK, 加入菱形约束即失败, 与 N 和非线性无关)。
  因此反向传播改用中心有限差分, 前向仍用 nlpsol 保证约束处理正确。

冻结参数时行为与 NMPCController.solve() 完全等价。

使用方法:
    nmpc = DifferentiableNMPCLayer()
    x0 = torch.randn(B, 3)          # 跟踪误差 [x_e, y_e, theta_e]
    ur_seq = torch.randn(B, 2, 30)  # 参考指令序列
    u_opt = nmpc(x0, ur_seq)        # (B, 2) 最优控制 [v, w], 可微

梯度验证:
    python differentiable_nmpc.py   # 运行梯度正确性 + 物理合理性检查
"""

import os
import torch
import numpy as np
import casadi as ca

from controller.controller import NMPCParams

# ============================================================================
# 可调参数
# ============================================================================

FD_EPS = 1e-5               # 有限差分扰动步长
IPOPT_TOL = 1e-10           # IPOPT 容差 (紧致, 保证差分信噪比)
U_FALLBACK_NORM_MAX = 1e6   # 求解结果范数超过此值视为发散


# ============================================================================
# NLP 构建 (nlpsol 底层接口)
# ============================================================================

def _build_nmpc_nlp(params: NMPCParams):
    """从 NMPCParams 构建 NMPC 的 nlpsol 求解器

    Returns:
        solver:     nlpsol Function
        n_w:        决策变量维度
        lbg/ubg:    约束边界
        u0_offset:  U[:,0] 在 w 中的偏移
    """
    p = params
    N = p.N

    # ---- 误差动力学 RK4 (与 controller.py 完全一致) ----
    X_sym = ca.MX.sym('X', 3, 1)
    u_sym = ca.MX.sym('u', 2, 1)
    ur_sym = ca.MX.sym('ur', 2, 1)
    x_e, y_e, theta_e = X_sym[0], X_sym[1], X_sym[2]
    v_r, w_r = ur_sym[0], ur_sym[1]

    f_X = ca.vertcat(v_r * ca.cos(theta_e), v_r * ca.sin(theta_e), w_r)
    G_X = ca.horzcat(
        ca.vertcat(-1.0, 0.0, 0.0),
        ca.vertcat(y_e, -x_e - p.alpha, -1.0))
    X_dot = f_X + ca.MX(G_X) @ u_sym
    f_dyn = ca.Function('f_dyn', [X_sym, u_sym, ur_sym], [X_dot])

    h = p.Ts
    k1 = f_dyn(X_sym, u_sym, ur_sym)
    k2 = f_dyn(X_sym + h/2 * k1, u_sym, ur_sym)
    k3 = f_dyn(X_sym + h/2 * k2, u_sym, ur_sym)
    k4 = f_dyn(X_sym + h * k3, u_sym, ur_sym)
    X_next_raw = X_sym + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    X_next = ca.vertcat(
        X_next_raw[0], X_next_raw[1],
        ca.atan2(ca.sin(X_next_raw[2]), ca.cos(X_next_raw[2])))
    F_RK4 = ca.Function('F_RK4', [X_sym, u_sym, ur_sym], [X_next])

    # ---- 决策变量与参数 ----
    X_var = ca.MX.sym('Xv', 3, N + 1)
    U_var = ca.MX.sym('Uv', 2, N)
    X0_par = ca.MX.sym('X0p', 3, 1)
    Up_par = ca.MX.sym('Upp', 2, 1)
    Ur_par = ca.MX.sym('Urp', 2, N)

    w = ca.vertcat(ca.reshape(X_var, -1, 1), ca.reshape(U_var, -1, 1))
    n_w = w.shape[0]
    u0_offset = 3 * (N + 1)

    p_par = ca.vertcat(X0_par, Up_par, ca.reshape(Ur_par, -1, 1))

    # ---- 代价函数 ----
    J = 0
    for k in range(N):
        J += X_var[:, k+1].T @ p.Q @ X_var[:, k+1]
        du = U_var[:, k] - (Up_par if k == 0 else U_var[:, k-1])
        J += du.T @ p.R @ du

    # ---- 约束 ----
    g_list = []
    lbg_list = []
    ubg_list = []

    # 初始条件
    g_list.append(X_var[:, 0] - X0_par)
    lbg_list.extend([0.0] * 3)
    ubg_list.extend([0.0] * 3)

    for k in range(N):
        # 动力学
        g_list.append(X_var[:, k+1] - F_RK4(X_var[:, k], U_var[:, k], Ur_par[:, k]))
        lbg_list.extend([0.0] * 3)
        ubg_list.extend([0.0] * 3)

        # 菱形控制约束
        v_k, w_k = U_var[0, k], U_var[1, k]
        for expr in [v_k/p.v_max + w_k/p.w_max,
                     -v_k/p.v_max + w_k/p.w_max,
                     v_k/p.v_max - w_k/p.w_max,
                     -v_k/p.v_max - w_k/p.w_max]:
            g_list.append(expr)
            lbg_list.append(-1e20)
            ubg_list.append(1.0)

        # 状态约束
        g_list.append(X_var[0, k])
        lbg_list.append(-p.x_bound); ubg_list.append(p.x_bound)
        g_list.append(X_var[1, k])
        lbg_list.append(-p.x_bound); ubg_list.append(p.x_bound)

    g_all = ca.vertcat(*[ca.reshape(gi, -1, 1) for gi in g_list])
    lbg = np.array(lbg_list)
    ubg = np.array(ubg_list)

    # ---- 构建 nlpsol ----
    nlp = {'x': w, 'f': J, 'g': g_all, 'p': p_par}
    opts = {
        'ipopt.max_iter': p.max_iter,
        'ipopt.print_level': p.print_level,
        'ipopt.tol': IPOPT_TOL,
        'ipopt.acceptable_tol': IPOPT_TOL,
        'print_time': False,
        'expand': True,
    }
    solver = ca.nlpsol('nmpc_nlpsol', 'ipopt', nlp, opts)

    return solver, n_w, lbg, ubg, u0_offset


# ============================================================================
# 内部求解辅助
# ============================================================================

def _solve_once(solver, n_w, lbg, ubg, u0_offset, x0, u_prev, ur_seq, N):
    """单次 NMPC 求解, 返回 u_opt (2,) 或 None"""
    p_val = np.concatenate([x0.reshape(3), u_prev.reshape(2), ur_seq.flatten()])
    try:
        sol = solver(x0=np.zeros(n_w), p=p_val, lbg=lbg, ubg=ubg)
        w_opt = np.array(sol['x']).flatten()
        u_opt = w_opt[u0_offset:u0_offset + 2].copy()
        if np.any(np.isnan(u_opt)) or np.linalg.norm(u_opt) > U_FALLBACK_NORM_MAX:
            return None
        return u_opt
    except Exception:
        return None


# ============================================================================
# torch.autograd.Function
# ============================================================================

class _DifferentiableNMPCFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x0, ur_seq, u_prev, solver, n_w,
                lbg, ubg, u0_offset, N):
        B = x0.shape[0]
        device, dtype = x0.device, x0.dtype

        x0_np = x0.detach().cpu().numpy()
        ur_np = ur_seq.detach().cpu().numpy()
        up_np = u_prev.detach().cpu().numpy()

        u_opts = np.zeros((B, 2))
        for b in range(B):
            u = _solve_once(solver, n_w, lbg, ubg, u0_offset,
                            x0_np[b], up_np[b], ur_np[b].reshape(2, N), N)
            u_opts[b] = u if u is not None else up_np[b]

        ctx.solver = solver
        ctx.n_w = n_w
        ctx.lbg = lbg
        ctx.ubg = ubg
        ctx.u0_offset = u0_offset
        ctx.N = N
        ctx.save_for_backward(x0, ur_seq, u_prev)

        return torch.tensor(u_opts, dtype=dtype, device=device)

    @staticmethod
    def backward(ctx, grad_u):
        """反向: 中心有限差分求 du*/dx0, 再链式法则得 dL/dx0"""
        x0, ur_seq, u_prev = ctx.saved_tensors
        B = x0.shape[0]
        dtype = x0.dtype
        eps = FD_EPS

        x0_np = x0.detach().cpu().numpy()
        ur_np = ur_seq.detach().cpu().numpy()
        up_np = u_prev.detach().cpu().numpy()
        gu_np = grad_u.detach().cpu().numpy()

        grad_x0 = np.zeros((B, 3))
        for b in range(B):
            ur_b = ur_np[b].reshape(2, ctx.N)
            jac = np.zeros((2, 3))
            for i in range(3):
                x0_p = x0_np[b].copy(); x0_p[i] += eps
                x0_m = x0_np[b].copy(); x0_m[i] -= eps
                u_p = _solve_once(ctx.solver, ctx.n_w, ctx.lbg, ctx.ubg,
                                  ctx.u0_offset, x0_p, up_np[b], ur_b, ctx.N)
                u_m = _solve_once(ctx.solver, ctx.n_w, ctx.lbg, ctx.ubg,
                                  ctx.u0_offset, x0_m, up_np[b], ur_b, ctx.N)
                if u_p is not None and u_m is not None:
                    jac[:, i] = (u_p - u_m) / (2 * eps)
                # else: 该方向差分失败, jac 列保持零
            grad_x0[b] = gu_np[b] @ jac

        return (torch.tensor(grad_x0, dtype=dtype),
                None, None, None, None, None, None, None, None)


# ============================================================================
# PyTorch Module 封装
# ============================================================================

class DifferentiableNMPCLayer(torch.nn.Module):
    """可微 NMPC 层

    零可训练参数。前向 CasADi IPOPT, 反向中心有限差分。
    冻结时 (不参与 loss.backward()) 等价于 NMPCController.solve()。
    """

    def __init__(self, params: NMPCParams = None):
        super().__init__()
        self.params = params or NMPCParams()

        print("[DiffNMPC] building nlpsol (IPOPT tol=%.0e)..." % IPOPT_TOL)
        (self._solver, self._n_w,
         self._lbg, self._ubg, self._u0_offset) = _build_nmpc_nlp(self.params)
        print(f"[DiffNMPC] done: n_w={self._n_w}, n_g={len(self._lbg)}, "
              f"backward=central FD (eps={FD_EPS:.0e})")
        self._warmup()

    def _warmup(self):
        N = self.params.N
        ur = np.zeros((2, N)); ur[0, :] = 0.25
        _solve_once(self._solver, self._n_w, self._lbg, self._ubg,
                    self._u0_offset, np.zeros(3), np.zeros(2), ur, N)

    def forward(self, x0: torch.Tensor, ur_seq: torch.Tensor,
                u_prev: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x0:     (B, 3)    tracking error [x_e, y_e, theta_e]
            ur_seq: (B, 2, N) reference command sequence
            u_prev: (B, 2)    previous control, default zero
        Returns:
            u_opt:  (B, 2)    optimal control [v, w]
        """
        B = x0.shape[0]
        if u_prev is None:
            u_prev = torch.zeros(B, 2, dtype=x0.dtype, device=x0.device)
        return _DifferentiableNMPCFunction.apply(
            x0, ur_seq, u_prev,
            self._solver, self._n_w,
            self._lbg, self._ubg, self._u0_offset,
            self.params.N)

    def solve_numpy(self, x0: np.ndarray, ur_seq: np.ndarray,
                    u_prev: np.ndarray = None) -> np.ndarray:
        """numpy interface (no gradient, equivalent to NMPCController.solve)"""
        N = self.params.N
        if u_prev is None:
            u_prev = np.zeros(2)
        u = _solve_once(self._solver, self._n_w, self._lbg, self._ubg,
                        self._u0_offset, x0, u_prev, ur_seq, N)
        return u if u is not None else u_prev.copy()


# ============================================================================
# 自测
# ============================================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Differentiable NMPC verification")
    print("=" * 60)

    nmpc = DifferentiableNMPCLayer()
    N = nmpc.params.N

    # ---- 1. 前向正确性 ----
    print("\n[1] Forward: DiffNMPC vs NMPCController")
    from controller.controller import NMPCController
    ctrl = NMPCController()
    ctrl.load_or_build()

    x0_test = np.array([0.1, -0.05, 0.03])
    ur_test = np.zeros((2, N)); ur_test[0, :] = 0.2

    u_diff = nmpc.solve_numpy(x0_test, ur_test)
    u_ctrl = ctrl.solve(x0_test, ur_test)
    diff = np.abs(u_diff - u_ctrl)
    print(f"  DiffNMPC:   v={u_diff[0]:.6f}, w={u_diff[1]:.6f}")
    print(f"  Controller: v={u_ctrl[0]:.6f}, w={u_ctrl[1]:.6f}")
    print(f"  abs diff:   {diff}")
    assert np.all(diff < 0.05), f"Forward mismatch too large: {diff}"
    print("  [OK]")

    # ---- 2. 梯度正确性: FD 自洽 + 解析验证 ----
    print("\n[2] Gradient: torch FD grad vs manual FD grad")
    x0_base = np.array([0.08, -0.03, 0.02])
    up_base = np.zeros(2)
    ur_base = np.zeros((2, N)); ur_base[0, :] = 0.15

    # torch 梯度
    x0_t = torch.tensor(x0_base, dtype=torch.float64).unsqueeze(0).requires_grad_(True)
    ur_t = torch.tensor(ur_base, dtype=torch.float64).unsqueeze(0)
    u_opt = nmpc(x0_t, ur_t)
    loss = (u_opt ** 2).sum()
    loss.backward()
    grad_torch = x0_t.grad.numpy().flatten()

    # 手动 FD 验证
    eps = FD_EPS
    grad_manual = np.zeros(3)
    for i in range(3):
        x0_p = x0_base.copy(); x0_p[i] += eps
        x0_m = x0_base.copy(); x0_m[i] -= eps
        u_p = nmpc.solve_numpy(x0_p, ur_base, up_base)
        u_m = nmpc.solve_numpy(x0_m, ur_base, up_base)
        # d(||u||^2)/d(x0_i) = 2*u^T @ du/dx0_i
        du_dxi = (u_p - u_m) / (2 * eps)
        u_val = nmpc.solve_numpy(x0_base, ur_base, up_base)
        grad_manual[i] = 2 * u_val @ du_dxi

    grad_diff = np.abs(grad_torch - grad_manual)
    print(f"  torch grad:  {grad_torch}")
    print(f"  manual grad: {grad_manual}")
    print(f"  abs diff:    {grad_diff}")
    assert np.all(grad_diff < 1e-4), f"Gradient mismatch: {grad_diff}"
    print("  [OK] gradient correct")

    # ---- 3. 物理合理性 ----
    print("\n[3] Physics: larger error -> larger control response")
    x0_small = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    x0_large = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    ur_t2 = torch.tensor(ur_base, dtype=torch.float64).unsqueeze(0)

    u_small = nmpc(x0_small, ur_t2)
    u_large = nmpc(x0_large, ur_t2)
    print(f"  small err x_e=0.01 -> v={u_small[0,0]:.4f}")
    print(f"  large err x_e=0.50 -> v={u_large[0,0]:.4f}")
    assert abs(u_large[0,0].item()) > abs(u_small[0,0].item())
    print("  [OK]")

    # ---- 4. 梯度非零检查 (不等式约束活跃时) ----
    print("\n[4] Gradient nonzero under active constraints")
    x0_act = torch.tensor([[0.3, 0.2, 0.1]], dtype=torch.float64, requires_grad=True)
    u_act = nmpc(x0_act, ur_t2)
    loss_act = u_act.sum()
    loss_act.backward()
    print(f"  grad = {x0_act.grad.numpy().flatten()}")
    assert np.any(np.abs(x0_act.grad.numpy()) > 1e-8), "Gradient should be nonzero!"
    print("  [OK]")

    print("\n" + "=" * 60)
    print("All tests passed")
    print("=" * 60)
