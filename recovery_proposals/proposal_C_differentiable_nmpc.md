# 方案 C：统一物理融合 + 可微 NMPC 闭环训练（端到端路线）

## 1. 方案概述

在方案 A 的物理引导解码器基础上，加入**类别条件 embedding**，并用**可微 NMPC 将跟踪误差梯度回传到恢复器**，实现端到端闭环训练。这是创新性最强、从根本上解决闭环分布偏移的路线。

**核心公式**：
```
ŷ_k = y_kin_k + δ_k
δ_k = Decoder(features_k, class_emb_c)

闭环训练 loss：
L_track = mean( ||compute_error(Upsilon_r, ŷ)||^2 )  # 跟踪误差

梯度链路（端到端）：
直接路径：L_track → ∂L/∂ŷ_k → ∂ŷ_k/∂δ_k → ∂δ_k/∂θ
间接路径：L_track → ∂L/∂u_k → ∂u_k/∂X_error_k → ∂X_error_k/∂ŷ_k → ∂ŷ_k/∂δ_k  （可微 NMPC）
```

**可微 NMPC 的独特价值**：
- **无可微 NMPC**：δ_k 只优化当前步跟踪误差（短视）
- **有可微 NMPC**：δ_k 通过 u_k 影响未来状态 → 未来跟踪误差，优化**长期跟踪好**（远视）

**两类攻击的统一处理**（同方案 A，但由闭环训练逼出）：
- 信息保留攻击：δ 学有效修正（减小长期跟踪误差）
- 信息丢失攻击：δ 学 ≈0（因为非零修正引入脏测量误差，长期跟踪误差反而增大，loss 惩罚）

> **关键创新**：两类的分化不是手工设计，而是**长期跟踪误差 loss 在闭环里逼出来的**。δ 自动学到"何时修正、何时退化"。

## 2. 数据流

### 2.1 原始数据（已有，无需改动）
同方案 A。

### 2.2 预处理（已有，无需改动）
同方案 A。

### 2.3 闭环训练数据（在线生成）
闭环训练不需要预处理窗口，而是**在线生成闭环轨迹**：
```
每个训练 batch：
  1. 预生成参考轨迹序列 Upsilon_r（torch tensor）
  2. 用当前恢复器 ŷ = y_kin + δ 替代 y_meas 送入 NMPC
  3. 可微 NMPC 输出 u_cmd → torch 运动学积分 → 新 true_state → torch 攻击注入 → 新 y_meas
  4. 计算跟踪误差 L_track
  5. 反向传播：L_track → 可微 NMPC → ∂L/∂ŷ → ∂ŷ/∂δ → 更新 δ 网络
```

## 3. 架构设计

### 3.1 恢复器（同方案 A 升级）
- `Detector` 加 `class_emb`
- `decode(features, x_norm, class_ids)` → `y_pred = y_kin + delta`
- 详见方案 A 第 3.2 节

### 3.2 可微 NMPC 模块（新建 `differentiable_nmpc.py`）

**核心思想**：NMPC 解 `u*` 对初始状态 `x0` 的导数，通过 **CasADi 原生自动微分**获得（不用有限差分，避免精度问题）。

**实现要点**：
1. **CasADi 原生 AD**：用 `ca.jacobian(U_opt, X0)` 构建解析雅可比函数，替代有限差分（避免 eps=1e-5 与 IPOPT tol=1e-4 的信噪比问题）
2. **截断 BPTT**：只展开 50-100 步（攻击窗口附近），避免 1000 步展开的内存/计算问题
3. **Warm start**：将上一步 NMPC 解作为初始猜测，加速收敛

```python
"""
differentiable_nmpc.py — 可微 NMPC 层
======================================
用 CasADi 原生 AD 将 NMPC 解对参数的导数嵌入 PyTorch 计算图。
"""
import torch
import numpy as np
import casadi as ca

class DifferentiableNMPC(torch.autograd.Function):
    """可微 NMPC：前向用 CasADi 求解，反向用 CasADi 原生 AD 求梯度"""

    @staticmethod
    def forward(ctx, x0_tensor, ur_seq_tensor, solver_fn, jacobian_fn, nmpc_params):
        B = x0_tensor.shape[0]
        N = nmpc_params.N
        u_opts = []
        for b in range(B):
            x0 = x0_tensor[b].detach().cpu().numpy().reshape(3, 1)
            ur = ur_seq_tensor[b].detach().cpu().numpy().reshape(2, N)
            u_opt = np.array(solver_fn(x0, np.zeros((2, 1)), ur)).flatten()
            u_opts.append(u_opt)
        ctx.jacobian_fn = jacobian_fn
        ctx.save_for_backward(x0_tensor, ur_seq_tensor)
        return torch.tensor(np.array(u_opts), dtype=x0_tensor.dtype,
                            device=x0_tensor.device)

    @staticmethod
    def backward(ctx, grad_u):
        x0_tensor, ur_seq_tensor = ctx.saved_tensors
        B = x0_tensor.shape[0]
        grad_x0 = torch.zeros_like(x0_tensor)
        for b in range(B):
            x0 = x0_tensor[b].detach().cpu().numpy().reshape(3, 1)
            ur = ur_seq_tensor[b].detach().cpu().numpy().reshape(2, ctx.nmpc_params.N)
            gu = grad_u[b].detach().cpu().numpy()
            # CasADi 原生 AD：∂u*/∂x0（解析雅可比，无有限差分误差）
            du_dx0 = np.array(ctx.jacobian_fn(x0, np.zeros((2, 1)), ur)).reshape(2, 3)
            grad_x0[b] = torch.tensor(gu @ du_dx0, dtype=x0_tensor.dtype)
        return grad_x0, None, None, None, None


def build_differentiable_solver(nmpc_params):
    """构建可微 NMPC solver + jacobian（CasADi 原生 AD）"""
    from controller.controller import NMPCBuilder
    builder = NMPCBuilder(nmpc_params)
    solver_fn = builder.build()  # ca.Function: (X0, U_prev, Ur_seq) → U_opt

    # CasADi 原生前向 AD：对 X0 求雅可比
    X0_sym = ca.MX.sym('X0', 3, 1)
    U_prev_sym = ca.MX.sym('U_prev', 2, 1)
    Ur_sym = ca.MX.sym('Ur', 2, nmpc_params.N)
    U_opt = solver_fn(X0_sym, U_prev_sym, Ur_sym)
    jac_fn = ca.Function('jac_nmpc', [X0_sym, U_prev_sym, Ur_sym],
                         [ca.jacobian(U_opt, X0_sym)])
    return solver_fn, jac_fn


class DifferentiableNMPCLayer(torch.nn.Module):
    """PyTorch 封装的可微 NMPC 层"""

    def __init__(self, nmpc_params):
        super().__init__()
        self.solver_fn, self.jacobian_fn = build_differentiable_solver(nmpc_params)
        self.nmpc_params = nmpc_params

    def forward(self, x0, ur_seq):
        return DifferentiableNMPC.apply(x0, ur_seq, self.solver_fn,
                                        self.jacobian_fn, self.nmpc_params)
```

### 3.3 全 torch 化闭环组件

可微 NMPC backward 要求**整条闭环链在 torch 计算图内**。需 torch 化：

```python
# model.py 新增
def rk4_step_torch(state, u, Ts=0.05, alpha=0.17):
    """torch 版 RK4（可微）"""
    def f(s, u):
        v, w = u[0], u[1]
        c, s_ = torch.cos(s[2]), torch.sin(s[2])
        return torch.stack([v * c - alpha * w * s_,
                            v * s_ + alpha * w * c, w])
    k1 = f(state, u)
    k2 = f(state + Ts / 2 * k1, u)
    k3 = f(state + Ts / 2 * k2, u)
    k4 = f(state + Ts * k3, u)
    next_state = state + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    next_state = torch.stack([
        next_state[0], next_state[1],
        torch.atan2(torch.sin(next_state[2]), torch.cos(next_state[2]))
    ])
    return next_state

def compute_error_torch(Upsilon_r, Upsilon_h):
    """torch 版跟踪误差（可微）"""
    x_r, y_r, theta_r = Upsilon_r[0], Upsilon_r[1], Upsilon_r[2]
    x_h, y_h, theta_h = Upsilon_h[0], Upsilon_h[1], Upsilon_h[2]
    c = torch.cos(theta_h); s = torch.sin(theta_h)
    dx = x_r - x_h; dy = y_r - y_h
    x_e = c * dx + s * dy
    y_e = -s * dx + c * dy
    theta_e = torch.atan2(torch.sin(theta_r - theta_h),
                          torch.cos(theta_r - theta_h))
    return torch.stack([x_e, y_e, theta_e])

# attack.py 新增
def inject_torch(t, y_clean, attack_type, onset, duration, params):
    """torch 版攻击注入（可微/detach 桥接）"""
    active = (t >= onset) & (t < onset + duration)
    if not active:
        return y_clean
    if attack_type == 'A1':      # 恒定偏置（可微）
        return y_clean + params['bias']
    elif attack_type == 'A2':    # 正弦（可微）
        val = params['amp'] * torch.sin(2 * np.pi * params['freq'] * t)
        return y_clean + torch.stack([val, val * 0.7, val * 0.3])
    elif attack_type == 'A3':    # 斜坡（可微）
        dt = t - onset
        return y_clean + torch.stack([params['rate'] * dt,
                                       params['rate'] * dt * 0.8,
                                       params['rate_t'] * dt])
    elif attack_type == 'A6':    # 缩放（可微）
        return params['scale'] * y_clean
    else:
        # A4/A5/A7 非加性：detach 桥接（梯度仍可通过 y_clean 流动）
        return y_clean.detach()  # 或具体攻击逻辑的 torch 实现
```

> **梯度链完整性**：全 torch 化后，`y_rec_k → X_error_k → NMPC → u_k → robot → true_state_{k+1} → y_meas_{k+1} → y_rec_{k+1}` 整条链在 torch 图内。可微 NMPC backward 的梯度可从未来 loss 传回当前 δ_k。
>
> **非加性攻击 (A4/A5/A7)**：`inject_torch` 用 detach 桥接，梯度不完全端到端，但仍可通过 y_clean 流动。这是信息丢失攻击的固有限制。

### 3.4 信号流图（端到端闭环训练）
```
y_meas(被攻击) → Detector(return_recon=True, class_ids)
  → ŷ = y_kin + δ(features, class_emb)
                    ↓
Upsilon_r → compute_error_torch(Upsilon_r, ŷ) → X_error
                    ↓
X_error + Ur_seq → DifferentiableNMPC → u_cmd  [可微，CasADi AD]
                    ↓
u_cmd → rk4_step_torch(true_state, u_cmd) → true_state_next  [可微]
                    ↓
true_state_next → inject_torch → y_meas_next  [可微/detach桥接]
                    ↓
L_track = mean(||X_error||^2)
                    ↓ 反向传播（端到端）
∂L/∂u_cmd → 可微 NMPC（CasADi AD）→ ∂L/∂X_error → ∂L/∂ŷ → ∂ŷ/∂δ → Decoder
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
train_closed_loop.py — 端到端可微 NMPC 闭环训练
================================================
用可微 NMPC 将长期跟踪误差梯度回传到恢复器 δ 网络。
"""
import torch
import numpy as np
from detector.classifier import Detector
from differentiable_nmpc import DifferentiableNMPCLayer
from model import WMRKinematics, WMRParams, RandomizedTrajectory, rk4_step_torch, compute_error_torch
from attack import SensorAttack, AttackConfig, inject_torch
from controller import NMPCParams

# 配置
CLOSED_LOOP_LR = 1e-5
CLOSED_LOOP_EPOCHS = 20
ATTACK_TYPES = ['A1', 'A2', 'A3', 'A6']  # 先训信息保留攻击（可微）
TRAJ_FAMILIES = ['lissajous', 'circular']
BPTT_STEPS = 100  # 截断 BPTT

def precompute_traj_sequence(traj, Ts, n_steps, N):
    """预生成参考轨迹序列（torch tensor），避免 traj.step 的 numpy 边界"""
    traj.reset()
    Upsilon_seq = []
    Ur_seq = []
    for step in range(n_steps):
        t = step * Ts
        Upsilon_r, _ = traj.step(t)
        Ur = traj.generate_sequence(t, N)
        Upsilon_seq.append(torch.tensor(Upsilon_r, dtype=torch.float32))
        Ur_seq.append(torch.tensor(Ur, dtype=torch.float32))
    return Upsilon_seq, Ur_seq

def closed_loop_step(model, nmpc_layer, Upsilon_seq, Ur_seq,
                     attacker_params, robot_state, Ts, onset_step, n_steps):
    """单条轨迹的端到端闭环前向 + 长期跟踪误差 loss"""
    total_loss = 0.0
    start = max(0, onset_step - 20)
    end = min(n_steps, onset_step + BPTT_STEPS)

    for step in range(n_steps):
        Upsilon_r = Upsilon_seq[step]
        Ur = Ur_seq[step]

        true_state = robot_state
        y_clean = true_state
        y_meas = inject_torch(step * Ts, y_clean, **attacker_params)

        # 恢复器前向（torch）
        y_rec = model.detect_torch(y_meas)

        # 跟踪误差 loss（展开窗口内）
        if start <= step < end:
            X_error = compute_error_torch(Upsilon_r, y_rec)
            total_loss += torch.sum(X_error[:2] ** 2)

        # 可微 NMPC（torch，参与梯度）
        u_cmd = nmpc_layer(X_error.unsqueeze(0), Ur.unsqueeze(0)).squeeze(0)

        # 机器人更新（torch，参与梯度）
        robot_state = rk4_step_torch(robot_state, u_cmd)

    return total_loss / max(1, end - start)

def main():
    model = Detector(...)
    model.load_state_dict(torch.load('detector/models/nn_cls_best.pt'))

    # 冻结骨干，只微调 decoder + class_emb
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.cls_head.parameters():
        p.requires_grad = False

    nmpc_layer = DifferentiableNMPCLayer(NMPCParams())
    optimizer = torch.optim.Adam(
        list(model.decoder.parameters()) + [model.class_emb.weight],
        lr=CLOSED_LOOP_LR)

    for epoch in range(CLOSED_LOOP_EPOCHS):
        for atk in ATTACK_TYPES:
            for fam in TRAJ_FAMILIES:
                traj = RandomizedTrajectory(family=fam, seed=epoch * 100 + hash(atk) % 1000)
                onset = 15.0
                onset_step = int(onset / 0.05)
                attacker_params = dict(attack_type=atk, onset=onset, duration=5.0,
                                       params=AttackConfig().__dict__)
                robot_state = torch.tensor([traj._x_r, traj._y_r, traj._theta_r],
                                           dtype=torch.float32)

                Upsilon_seq, Ur_seq = precompute_traj_sequence(traj, 0.05, 1000, NMPCParams().N)

                optimizer.zero_grad()
                loss = closed_loop_step(model, nmpc_layer, Upsilon_seq, Ur_seq,
                                        attacker_params, robot_state, 0.05,
                                        onset_step, 1000)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.decoder.parameters()) + [model.class_emb.weight], 1.0)
                optimizer.step()

        print(f"Epoch {epoch}: loss={loss.item():.6f}")

    torch.save(model.state_dict(), 'detector/models/nn_cls_closed_loop.pt')
```

### 4.3 消融 baseline：跟踪误差监督（无可微 NMPC）
作为消融实验，可去掉可微 NMPC backward，直接用跟踪误差 loss（NMPC 用 numpy，不参与梯度）：
```python
# 无可微 NMPC：u_cmd 用 numpy NMPC，梯度断
X_error_np = X_error.detach().numpy()
u_cmd = ctrl.solve(X_error_np, Ur_seq.numpy())  # numpy
robot_state_np = robot.step(u_cmd)  # numpy
robot_state = torch.tensor(robot_state_np)  # 重新进入 torch（梯度断）
```
对比"有/无可微 NMPC"的跟踪效果，验证可微 NMPC 的增量价值（长期 vs 短视优化）。

## 5. 部署流程

### 5.1 backend.py 改动
同方案 A（两阶段推理，`y_recovered = y_pred[0,-1]`，A0 门控）。
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
- **长期跟踪优化**：可微 NMPC 让 δ 优化未来多步跟踪误差，而非短视当前步。这是可微 NMPC 的独特价值。
- **对齐跟踪目标**：loss 是跟踪误差，不是重构误差。δ 直接为"跟踪好"优化。
- **两类分化自动逼出**：信息丢失攻击下 δ→0 是长期 loss 惩罚的结果，不是手工设计。
- **创新性最强**：可微 NMPC + 条件先验 + 物理引导，论文卖点足。
- **不违反约束**：同方案 A。

### 缺点
- **训练不稳定风险**：闭环梯度可能爆炸/消失，需仔细调学习率、梯度裁剪、curriculum。
- **依赖预训练**：必须先有方案 A 的预训练权重，否则随机初始化闭环训练不收敛。
- **信息丢失攻击仍退化**：δ→0 时 ŷ≈y_kin，本质仍是模型依赖。但这是闭环 loss 逼出的最优行为，不是设计缺陷。
- **非加性攻击梯度桥接**：A4/A5/A7 的 `inject_torch` 需 detach 桥接，梯度不完全端到端。

## 7. 实现细节

### 7.1 代码改动清单

| 文件 | 改动 |
|------|------|
| `detector/classifier.py` | 同方案 A（class_emb + decode 签名）+ `detect_torch` |
| `differentiable_nmpc.py`（新建） | 可微 NMPC 层（CasADi 原生 AD） |
| `train_closed_loop.py`（新建） | 端到端闭环训练脚本 |
| `model.py` | 新增 `rk4_step_torch`, `compute_error_torch` |
| `attack.py` | 新增 `inject_torch` |
| `backend.py` | 同方案 A |
| `detector/train.py` | 同方案 A（预训练） |

### 7.2 关键参数

| 参数 | 值 | 来源 |
|------|-----|------|
| `CLOSED_LOOP_LR` | 1e-5 | 需调（1e-6 ~ 1e-4） |
| `CLOSED_LOOP_EPOCHS` | 20 | 需调 |
| `GRAD_CLIP` | 1.0 | 防梯度爆炸 |
| `BPTT_STEPS` | 100 | 截断 BPTT 步数 |
| 冻结层 | backbone + cls_head | 只微调 decoder + class_emb |

### 7.3 依赖
- 现有环境（torch, casadi, numpy）
- 无需新增依赖（CasADi 原生 AD 已支持）

### 7.4 前置依赖
方案 C 的 `class_emb`/`decode(class_ids)` 签名需方案 A 先完成。执行顺序：方案 A → 方案 C 阶段1（=方案A）→ 方案 C 阶段2。

### 7.5 闭环训练中 class_ids 来源
训练数据有 ground truth 标签，闭环训练阶段直接用 GT class 作为 `class_ids`。部署时用 predicted class（同方案 A 两阶段推理）。

## 8. 验证计划

### 8.1 可微 NMPC 梯度验证（先做，关键）
```python
# scripts/verify_nmpc_gradient.py
# 用 torch.autograd.gradcheck 验证 CasADi AD 梯度 vs 数值梯度
# 对简单 x0 扰动，比较 ∂u*/∂x0 的解析（CasADi AD）和数值（中心差分）
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
- `post_pos_rmse` 应 < 方案 A（因为对齐了跟踪目标 + 长期优化）
- 信息丢失攻击（A4/A5/A7）：δ 应 ≈0，ŷ≈y_kin，轨迹跳动但攻击后收敛

### 8.4 消融实验
- **有/无可微 NMPC**：验证长期跟踪优化的增量价值（核心消融）
- 方案 A（开环监督）vs 方案 C（闭环微调）：验证闭环训练是否改善跟踪
- 有/无 class_emb：验证类别条件是否帮助分化
- BPTT_STEPS = 50/100/200：验证截断长度对收敛的影响
