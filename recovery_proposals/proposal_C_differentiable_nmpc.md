# 方案 C：统一物理融合 + 可微 NMPC 闭环训练（端到端路线）

## 1. 方案概述

在方案 A 的物理引导解码器基础上，加入**类别条件 embedding**，并用**可微 NMPC 将跟踪误差梯度回传到恢复器**，实现闭环训练。这是创新性最强、从根本上解决闭环分布偏移的路线。

**核心公式**：
```
ŷ_k = y_kin_k + δ_k
δ_k = Decoder(features_k, class_emb_c)

闭环训练 loss：
L_track = mean( ||compute_error(Upsilon_r, ŷ)||^2 )  # 跟踪误差
梯度链路：L_track → ∂L/∂ŷ → ∂ŷ/∂δ → ∂δ/∂θ_decoder  # 可微 NMPC 提供 ∂L/∂ŷ
```

**两类攻击的统一处理**（同方案 A，但由闭环训练逼出）：
- 信息保留攻击：δ 学有效修正（减小跟踪误差）
- 信息丢失攻击：δ 学 ≈0（因为非零修正引入脏测量误差，跟踪误差反而增大，loss 惩罚）

> **关键创新**：两类的分化不是手工设计，而是**跟踪误差 loss 在闭环里逼出来的**。δ 自动学到"何时修正、何时退化"。

## 2. 数据流

### 2.1 原始数据（已有，无需改动）
同方案 A。

### 2.2 预处理（已有，无需改动）
同方案 A。

### 2.3 闭环训练数据（新增）
可微 NMPC 训练不需要预处理窗口，而是**在线生成闭环轨迹**：
```
每个训练 batch：
  1. 从 dataset/<ts>/ 采样一条轨迹（含 y_meas, y_clean, u_cmd, Upsilon_r）
  2. 用当前恢复器 ŷ = y_kin + δ 替代 y_meas 送入 NMPC
  3. NMPC 输出 u_cmd → 运动学积分 → 新 true_state → 新 y_meas（攻击注入）
  4. 计算跟踪误差 L_track
  5. 反向传播：L_track → 可微 NMPC → ∂L/∂ŷ → ∂ŷ/∂δ → 更新 δ 网络
```

## 3. 架构设计

### 3.1 恢复器（同方案 A 升级）
- `Detector` 加 `class_emb`
- `decode(features, x_norm, class_ids)` → `y_pred = y_kin + delta`
- 详见方案 A 第 3.2 节

### 3.2 可微 NMPC 模块（新建 `differentiable_nmpc.py`）

**核心思想**：NMPC 求解器 `u* = argmin J(x, u)` 的解 `u*` 对参数 `x`（当前状态/恢复位姿）的导数，通过**隐函数定理（Implicit Function Theorem, IFT）** 对 KKT 最优性条件求导获得。

```python
"""
differentiable_nmpc.py — 可微 NMPC 层
======================================
用 CasADi 隐函数定理将 NMPC 解对参数的导数嵌入 PyTorch 计算图。
"""
import torch
import numpy as np
import casadi as ca

class DifferentiableNMPC(torch.autograd.Function):
    """可微 NMPC：前向用 CasADi IPOPT 求解，反向用 IFT 求梯度"""

    @staticmethod
    def forward(ctx, x0_tensor, ur_seq_tensor, solver, nmpc_params):
        """
        Args:
            x0_tensor: (B, 3) 当前误差状态 [x_e, y_e, theta_e]
            ur_seq_tensor: (B, 2, N) 参考指令序列
            solver: CasADi nlpsol 求解器
            nmpc_params: NMPCParams
        Returns:
            u_opt: (B, 2) 最优控制 [v, w]
        """
        B = x0_tensor.shape[0]
        N = nmpc_params.N
        u_opts = []
        # 保存 KKT 信息供反向
        kkt_infos = []

        for b in range(B):
            x0 = x0_tensor[b].detach().cpu().numpy().reshape(3, 1)
            ur = ur_seq_tensor[b].detach().cpu().numpy().reshape(2, N)
            u_prev = np.zeros((2, 1))

            # CasADi 求解
            sol = solver(x0=x0, p=ca.vertcat(u_prev, ur.reshape(-1, 1)))
            u_opt = np.array(sol['x']).flatten()[:2]  # 取前2个（v, w）
            u_opts.append(u_opt)

            # 保存 KKT 乘子（用于 IFT 反向）
            kkt_infos.append({
                'lam_g': np.array(sol['lam_g']),
                'lam_x': np.array(sol['lam_x']),
                'x0': x0, 'ur': ur, 'u_opt': u_opt
            })

        ctx.solver = solver
        ctx.nmpc_params = nmpc_params
        ctx.kkt_infos = kkt_infos
        ctx.save_for_backward(x0_tensor, ur_seq_tensor)

        return torch.tensor(np.array(u_opts), dtype=x0_tensor.dtype, device=x0_tensor.device)

    @staticmethod
    def backward(ctx, grad_u):
        """
        反向：∂L/∂x0 = ∂L/∂u* · ∂u*/∂x0
        用 IFT：∂u*/∂x0 = -[∂²L/∂u²]^{-1} · ∂²L/∂u∂x0
        CasADi 可通过 hessian 或 finite difference 近似
        """
        x0_tensor, ur_seq_tensor = ctx.saved_tensors
        B = x0_tensor.shape[0]
        grad_x0 = torch.zeros_like(x0_tensor)

        for b in range(B):
            info = ctx.kkt_infos[b]
            x0 = info['x0']
            u_opt = info['u_opt']
            gu = grad_u[b].detach().cpu().numpy()

            # 有限差分近似 ∂u*/∂x0（简化版，生产用 CasADi hessian）
            eps = 1e-5
            du_dx0 = np.zeros((2, 3))
            for i in range(3):
                x0_p = x0.copy(); x0_p[i] += eps
                x0_m = x0.copy(); x0_m[i] -= eps
                # 重新求解（或用 warm start）
                sol_p = ctx.solver(x0=x0_p, p=ca.vertcat(
                    np.zeros((2,1)), info['ur'].reshape(-1,1)))
                sol_m = ctx.solver(x0=x0_m, p=ca.vertcat(
                    np.zeros((2,1)), info['ur'].reshape(-1,1)))
                u_p = np.array(sol_p['x']).flatten()[:2]
                u_m = np.array(sol_m['x']).flatten()[:2]
                du_dx0[:, i] = (u_p - u_m) / (2 * eps)

            # ∂L/∂x0 = gu^T · du_dx0
            grad_x0[b] = torch.tensor(gu @ du_dx0, dtype=x0_tensor.dtype)

        return grad_x0, None, None, None


class DifferentiableNMPCLayer(torch.nn.Module):
    """PyTorch 封装的可微 NMPC 层"""

    def __init__(self, solver_path='controller/nmpc_solver.casadi'):
        super().__init__()
        self.solver = ca.Function.load(solver_path)
        self.nmpc_params = None  # 从 controller 导入

    def forward(self, x0, ur_seq):
        return DifferentiableNMPC.apply(x0, ur_seq, self.solver, self.nmpc_params)
```

### 3.3 可选实现：可微 NMPC 方式

| 方式 | 实现 | 优点 | 缺点 |
|------|------|------|------|
| **CasADi IFT + 有限差分**（推荐起步） | 上述代码 | 原生 CasADi，无需新依赖 | 反向慢（每参数2次求解） |
| CasADi IFT + 解析 Hessian | CasADi `hessian` API | 反向快 | 实现复杂，KKT 条件推导 |
| acados 可微接口 | `acados` Python API + GPU | 最快（GPU 批量） | 需装 acados，学习曲线 |
| 展开近似 Unrolled MPC | 展开 N 步预测成可微层 | 最简，纯 PyTorch | 精度有限，不考虑约束 |

### 3.4 可选实现：训练策略

| 策略 | 实现 | 适用 |
|------|------|------|
| **两阶段**（推荐） | 阶段1开环监督预训练 δ → 阶段2可微 NMPC 闭环微调 | 稳定，收敛快 |
| 端到端 | 直接从随机初始化做闭环训练 | 理论最优，但不稳定 |
| Curriculum | 攻击强度从弱到强渐进 | 防止早期梯度爆炸 |

### 3.5 信号流图（闭环训练）
```
y_meas(被攻击) → Detector(return_recon=True, class_ids)
  → ŷ = y_kin + δ(features, class_emb)
                    ↓
Upsilon_r → compute_error(Upsilon_r, ŷ) → X_error
                    ↓
X_error + Ur_seq → DifferentiableNMPC → u_cmd
                    ↓
u_cmd → WMRKinematics.rk4_step(true_state, u_cmd) → true_state_next
                    ↓
true_state_next → SensorAttack.inject → y_meas_next（下一拍）
                    ↓
L_track = mean(||X_error||^2)
                    ↓ 反向传播
∂L/∂u_cmd → 可微 NMPC（IFT）→ ∂L/∂ŷ → ∂ŷ/∂δ → 更新 Decoder 参数
```

## 4. 训练流程

### 4.1 阶段1：开环监督预训练（同方案 A）
```bash
python detector/train.py --data-dir dataset_win/<ts>/
```
- 目标：`MSE(y_pred, y_clean)` + 末步加权
- 产出：`detector/models/nn_cls_best.pt`（含 class_emb + decoder）

### 4.2 阶段2：可微 NMPC 闭环微调（新建 `train_closed_loop.py`）

```python
"""
train_closed_loop.py — 闭环可微训练
====================================
用可微 NMPC 将跟踪误差梯度回传到恢复器 δ 网络。
"""
import torch
from detector.classifier import Detector
from differentiable_nmpc import DifferentiableNMPCLayer
from model import WMRKinematics, WMRParams, RandomizedTrajectory
from attack import SensorAttack, AttackConfig
from controller import NMPCParams

# 配置
CLOSED_LOOP_LR = 1e-5       # 闭环微调学习率（远小于预训练）
CLOSED_LOOP_EPOCHS = 20
ATTACK_TYPES = ['A1', 'A2', 'A3', 'A6']  # 先训信息保留攻击
TRAJ_FAMILIES = ['lissajous', 'circular']

def closed_loop_step(model, nmpc_layer, traj, attacker, robot, Ts, n_steps):
    """单条轨迹的闭环前向 + loss 计算"""
    traj.reset()
    robot.reset(np.array([traj._x_r, traj._y_r, traj._theta_r]))
    model.reset_buffers()  # 清空滑动窗口

    total_loss = 0.0
    for step in range(n_steps):
        t = step * Ts
        Upsilon_r, _ = traj.step(t)
        Ur_seq = traj.generate_sequence(t, NMPCParams().N)

        true_state = robot.state.copy()
        y_clean = true_state.copy()
        y_meas = attacker.inject(t, y_clean)

        # 恢复器前向
        result = model.detect_differentiable(y_meas)  # 需实现可微版 detect
        y_rec = result.y_recovered  # (3,) 可微张量

        # 跟踪误差
        X_error = WMRKinematics.compute_error_torch(Upsilon_r, y_rec)  # 需 torch 版
        total_loss += torch.sum(X_error[:2] ** 2)

        # 可微 NMPC
        u_cmd = nmpc_layer(X_error.unsqueeze(0), Ur_seq.unsqueeze(0)).squeeze(0)

        # 机器人更新（可微运动学）
        robot.step_torch(u_cmd)  # 需 torch 版 rk4_step

    return total_loss / n_steps

def main():
    # 加载预训练模型
    model = Detector(...)
    model.load_state_dict(torch.load('detector/models/nn_cls_best.pt'))

    # 冻结骨干，只微调 decoder + class_emb
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.cls_head.parameters():
        p.requires_grad = False

    nmpc_layer = DifferentiableNMPCLayer()
    optimizer = torch.optim.Adam(
        list(model.decoder.parameters()) + [model.class_emb.weight],
        lr=CLOSED_LOOP_LR)

    for epoch in range(CLOSED_LOOP_EPOCHS):
        for atk in ATTACK_TYPES:
            for fam in TRAJ_FAMILIES:
                traj = RandomizedTrajectory(family=fam, seed=epoch*100+hash(atk)%1000)
                attacker = SensorAttack(atk, onset_time=15.0, config=AttackConfig(attack_duration=5.0))
                robot = WMRKinematics(WMRParams())

                optimizer.zero_grad()
                loss = closed_loop_step(model, nmpc_layer, traj, attacker, robot, 0.05, 1000)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        print(f"Epoch {epoch}: loss={loss.item():.6f}")

    torch.save(model.state_dict(), 'detector/models/nn_cls_closed_loop.pt')
```

### 4.3 需新增的 torch 版工具函数

```python
# model.py 新增
@staticmethod
def compute_error_torch(Upsilon_r, Upsilon_h):
    """torch 版跟踪误差（可微）"""
    x_r, y_r, theta_r = Upsilon_r[0], Upsilon_r[1], Upsilon_r[2]
    x_h, y_h, theta_h = Upsilon_h[0], Upsilon_h[1], Upsilon_h[2]
    c = torch.cos(theta_h); s = torch.sin(theta_h)
    dx = x_r - x_h; dy = y_r - y_h
    x_e = c * dx + s * dy
    y_e = -s * dx + c * dy
    theta_e = torch.atan2(torch.sin(theta_r - theta_h), torch.cos(theta_r - theta_h))
    return torch.stack([x_e, y_e, theta_e])

# model.py 新增
def rk4_step_torch(self, u):
    """torch 版 RK4（可微）"""
    # 同 rk4_step 但用 torch 运算
    ...
```

## 5. 部署流程

### 5.1 backend.py 改动
同方案 A（`return_recon=True`，`y_recovered = y_pred[0,-1]`，A0 门控）。
加载 `nn_cls_closed_loop.pt` 替代 `nn_cls_best.pt`。

### 5.2 simulate.py 无需改动

### 5.3 闭环仿真
```bash
python simulate.py --attack A1 --trajectory lissajous --model-path detector/models/nn_cls_closed_loop.pt
python simulate.py --all --model-path detector/models/nn_cls_closed_loop.pt
```

## 6. 优缺点

### 优点
- **解决闭环分布偏移**：恢复器在闭环里训练，见过自己输出回流。这是方案 A/B 无法做到的。
- **对齐跟踪目标**：loss 是跟踪误差，不是重构误差。δ 直接为"跟踪好"优化。
- **两类分化自动逼出**：信息丢失攻击下 δ→0 是 loss 惩罚的结果，不是手工设计。
- **创新性最强**：可微 NMPC + 条件先验 + 物理引导，论文卖点足。
- **不违反约束**：同方案 A。

### 缺点
- **工程量大**：可微 NMPC 实现复杂（IFT/有限差分/acados），torch 版运动学，闭环训练循环。预计 1-2 周。
- **训练不稳定风险**：闭环梯度可能爆炸/消失，需仔细调学习率、梯度裁剪、curriculum。
- **反向慢**：有限差分 IFT 每参数需 2 次 NMPC 求解，1000 步轨迹 × 3 参数 = 6000 次求解/样本。需 warm start 或 acados GPU 加速。
- **依赖预训练**：必须先有方案 A 的预训练权重，否则随机初始化闭环训练不收敛。
- **信息丢失攻击仍退化**：δ→0 时 ŷ≈y_kin，本质仍是模型依赖。但这是闭环 loss 逼出的最优行为，不是设计缺陷。

## 7. 实现细节

### 7.1 代码改动清单

| 文件 | 改动 | 行数估计 |
|------|------|----------|
| `detector/classifier.py` | 同方案 A（class_emb + decode 签名） | ~30行 |
| `differentiable_nmpc.py`（新建） | 可微 NMPC 层（IFT + 有限差分） | ~120行 |
| `train_closed_loop.py`（新建） | 闭环训练脚本 | ~150行 |
| `model.py` | 新增 `compute_error_torch`, `rk4_step_torch` | ~40行 |
| `backend.py` | 同方案 A | ~25行 |
| `detector/train.py` | 同方案 A（预训练） | ~15行 |

### 7.2 关键参数

| 参数 | 值 | 来源 |
|------|-----|------|
| `CLOSED_LOOP_LR` | 1e-5 | 需调（1e-6 ~ 1e-4） |
| `CLOSED_LOOP_EPOCHS` | 20 | 需调 |
| `GRAD_CLIP` | 1.0 | 防梯度爆炸 |
| `IFT_EPS` | 1e-5 | 有限差分步长 |
| 冻结层 | backbone + cls_head | 只微调 decoder + class_emb |

### 7.3 依赖
- 现有环境（torch, casadi, numpy）
- 可选：`acados`（GPU 加速可微 NMPC）
- 无需新增 pip 包（CasADi 已有）

## 8. 验证计划

### 8.1 可微 NMPC 梯度验证（先做，关键）
```python
# scripts/verify_nmpc_gradient.py
# 用 torch.autograd.gradcheck 验证有限差分 IFT 梯度 vs 数值梯度
# 对简单 x0 扰动，比较 ∂u*/∂x0 的解析（有限差分）和数值（中心差分）
```
**判据**：相对误差 < 1e-3 → 可微 NMPC 正确

### 8.2 闭环训练收敛验证
```bash
python train_closed_loop.py
# 观察 loss 曲线：应单调下降或震荡下降
# 若 loss 爆炸：降 LR、加 GRAD_CLIP、用 curriculum
```

### 8.3 闭环仿真验证
```bash
python simulate.py --all --model-path detector/models/nn_cls_closed_loop.pt
```
**指标**：
- `post_pos_rmse` 应 < 方案 A（因为对齐了跟踪目标）
- 信息丢失攻击（A4/A5/A7）：δ 应 ≈0，ŷ≈y_kin，轨迹跳动但攻击后收敛

### 8.4 消融实验
- 方案 A（开环监督）vs 方案 C（闭环微调）：验证闭环训练是否改善跟踪
- 有/无 class_emb：验证类别条件是否帮助分化
- 有限差分 IFT vs acados：速度/精度对比

## 9. 审查修复（子智能体审查后必修）

### 修复1 [严重/阻断]：CasADi solver 接口不匹配
当前 solver 是 `opti.to_function` 导出的 `ca.Function(X0, U_prev, Ur_seq) → U[:,0]`，不是 `nlpsol`。无 `sol['x']`/`sol['lam_g']`/`sol['lam_x']`。
- **修复**：重构 `NMPCBuilder.build()` 同时导出 `nlpsol` 版本（返回完整 `{x, lam_g, lam_x, g}` 字典），用于可微训练。或改用 CasADi `ca.Function.jacobian()` 做原生前向 AD，绕过 KKT 乘子。

### 修复2 [严重/架构]：闭环梯度链在 robot/traj/attack 处断裂
`traj.step`/`attacker.inject` 是 numpy，`y_meas_{k+1}` 从 numpy 重新进入 → 梯度链断裂。可微 NMPC backward 计算的 `∂L/∂X_error` 无法通过 `u_cmd → robot → next_y_meas` 传播。
- **修复（短期，推荐）**：重新表述方案为 **"NMPC-aware 跟踪误差监督"**——在跟踪误差度量中使用 NMPC 处理后的误差，而非直接重构误差。去掉可微 NMPC backward，改为：
```python
# 替代可微 NMPC：直接用跟踪误差作为 loss，无需 NMPC 梯度
L_track = mean(||compute_error_torch(Upsilon_r, y_rec)||^2)
L_track.backward()  # 梯度直接流向 y_rec → δ → decoder
```
这仍是"闭环训练"（在闭环轨迹上计算 loss），只是不用 NMPC 的梯度。
- **修复（长期）**：全 torch 化 `RandomizedTrajectory`（或预生成序列）、`SensorAttack.inject`（detach 桥接）、`WMRKinematics`。工程量 ~200 行，远超原估。

### 修复3 [中等]：有限差分精度 vs IPOPT tol
eps=1e-5 与 IPOPT `acceptable_tol=1e-4` 不匹配，差分商信噪比 < 1。
- **修复**：改用 CasADi 原生 AD（`Function.jacobian()`），或提高 IPOPT 精度至 `tol=1e-8`，或增大 eps 至 1e-3。

### 修复4 [中等]：1000 步展开复杂度过高
每轨迹 7000 次 NMPC 求解，总训练 40-300 分钟。
- **修复**：截断 BPTT 至 50-100 步；或只训练攻击窗口（onset±50 步）；warm start 加速。

### 修复5 [前置依赖]：强依赖方案 A
方案 C 的 `class_emb`/`decode(class_ids)` 签名需方案 A 先完成。
- **执行顺序**：方案 A → 方案 C 阶段1（=方案A）→ 方案 C 阶段2。

### 修复6 [遗漏]：闭环训练中 class_ids 来源
分类器输出 argmax 不可微。若用 soft probability 则可行但方案未说明。
- **修复**：闭环训练阶段用 ground truth class（因为训练数据有标签），部署时用 predicted class。或改用 soft probability + Gumbel-Softmax。
