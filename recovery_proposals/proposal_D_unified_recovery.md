# 方案 D：物理信息统一恢复（最终方案）

## 1. 核心思想

**恢复是唯一主任务，分类是辅助探针。**

现有方案 A/B/C 的共同缺陷：将"识别攻击类型"作为恢复的前置条件，导致分类与恢复两个目标在共享表征上梯度冲突、级联失败、补丁堆叠。

本方案的出发点：KinematicFeatureLayer 输出的创新信号 `innov = y_meas − y_kin` 的时域模式已经完整描述了攻击特征——恒定偏移、正弦振荡、斜坡增长、信号消失——网络无需先贴标签再处理，直接从 innov 模式学习修正。

**恢复公式**：
```
ŷ = y_kin + δ(features)
```
- `y_kin`：可微分运动学层的前向积分预测（零参数，不滞后，不依赖当前脏测量）
- `δ`：轻量解码器输出的漂移修正（小幅度、时域平滑）
- 无 class_emb、无置信度门控、无两阶段推理

**训练目标**：跟踪误差，而非重构误差。

`MSE(ŷ, y_clean)` 拟合的是"无恢复闭环下的真实轨迹"——部署恢复后轨迹改变，目标分布不再存在。正确的目标是：**你输出的 ŷ 喂给 NMPC 后，跟踪有多好。**

## 2. 架构

```
输入: x (B, 128, 5) 归一化 [y_meas(3), u_cmd(2)]
  │
  ▼ KinematicFeatureLayer (零参数, 已有)
  (B, 11, 128) = [y_meas, u_cmd, y_kin, innov]
  │
  ▼ MultiScaleDSConvBackbone (已有, 共享, 11→32→64→96, 128→64→32→16)
  features (B, 16, 96)
  │
  ├─→ 恢复路径 (主任务, ~6K 参数)
  │     Decoder: ConvTranspose1d 上采样 16→128 → δ_norm (B, 128, 3)
  │     ŷ_norm = y_kin_norm + δ_norm
  │     ŷ_phys = ŷ_norm × scale + median
  │
  └─→ 分类路径 (辅助, 已有, ~10K 参数)
        注意力池化 → cls_head → (B, 8)
```

**总参数量 ~34K**（骨干 18K + 分类头 10K + 解码器 6K），仍适合嵌入式。

### 与方案 A/B/C 的对比

| 组件 | 方案 A | 方案 B | 方案 C | **方案 D** |
|------|--------|--------|--------|-----------|
| class_emb 条件注入 | ✓ | — | ✓ | **✗ 删除** |
| 置信度门控 | ✓ | ✓ (R 插值) | ✓ | **✗ 删除** |
| 两阶段推理 | ✓ | — | ✓ | **✗ 单次前向** |
| A0 特殊处理 | ✓ | — | ✓ | **✗ 不需要** |
| 可微 NMPC | — | — | ✓ | **✓ (阶段 2)** |
| 分类角色 | 驱动恢复 | 调度 R | 驱动恢复 | **辅助探针** |

## 3. 训练

### 3.1 阶段 1：开环预训练

在现有预处理窗口数据上训练，引入参考轨迹 Upsilon_r。

**Loss**：
```python
# 主: 跟踪误差代理 (物理空间)
X_error = compute_error_torch(Upsilon_r, ŷ_phys)      # (B, 128, 3)
L_track = mean(X_error[:,:,0]² + X_error[:,:,1]²)     # 位置跟踪误差

# 正则: 防止退化为纯轨迹回放 (ŷ → Upsilon_r, 忽略传感器)
L_reg = MSE(ŷ_phys, y_clean)

# 辅助: 分类探针
L_cls = CrossEntropy(cls_logits, label)

L = L_track + λ_reg × L_reg + λ_cls × L_cls
```

| 参数 | 值 | 说明 |
|------|-----|------|
| λ_reg | 0.1 | 正则化权重，锚定传感器信息 |
| λ_cls | 0.05 | 辅助分类，不驱动骨干 |
| lr | 1e-3 | Adam |
| epochs | 80 | 含 A0 降采样 |

**产出**：`detector/models/nn_recovery_pretrain.pt`

### 3.2 阶段 2：可微 NMPC 闭环微调

在线生成闭环轨迹，用可微 NMPC 将长期跟踪误差梯度回传到 Decoder。

```
每个训练样本 (在线生成):
  for step in range(BPTT_STEPS):
    y_meas = inject_torch(true_state, attack)       # 攻击注入
    ŷ = model.recover(y_meas_window, u_cmd_window)  # 恢复
    X_error = compute_error_torch(Upsilon_r, ŷ)     # 跟踪误差
    L_track += ||X_error||²
    u_cmd = DiffNMPC(X_error, Ur_seq)               # 可微 NMPC
    true_state = rk4_step_torch(true_state, u_cmd)   # 机器人更新
  L_track.backward()  # 梯度穿过 DiffNMPC → Decoder
```

**梯度链路**：
```
L_track → ∂L/∂u_cmd → DiffNMPC (FD 雅可比) → ∂L/∂X_error
  → ∂L/∂ŷ → ∂ŷ/∂δ → Decoder 参数
```

| 参数 | 值 | 说明 |
|------|-----|------|
| lr | 1e-5 | 微调，防破坏预训练 |
| BPTT_STEPS | 100 | 截断反向传播 (攻击窗口附近) |
| epochs | 20 | |
| 冻结 | backbone + cls_head | 只微调 Decoder |
| 攻击 | A1, A2, A3, A6 | 先训信息保留攻击 (可微) |
| 轨迹 | lissajous, circular | 先训光滑轨迹 |

**产出**：`detector/models/nn_recovery_closed_loop.pt`

### 3.3 消融 baseline

去掉 DiffNMPC 梯度（u_cmd 用 numpy，不参与反向传播），对比"长期优化 vs 短视优化"。

## 4. 数据

### 4.1 数据生成改动 (generate_dataset.py)

```python
# 新增: 保存参考轨迹
npz['Upsilon_r'] = Upsilon_r_seq  # (1000, 3), 物理空间
```

### 4.2 预处理改动 (preprocess_data.py)

新增输出：

| 文件 | 形状 | 说明 |
|------|------|------|
| `Y_{split}_ref.npy` | (N, 128, 3) | 参考轨迹窗口，物理空间，不归一化 |

### 4.3 闭环训练数据 (阶段 2, 在线生成)

不需要预处理窗口。每个 epoch 在线生成闭环轨迹：
- 随机采样轨迹族 + 攻击类型 + onset 时间
- 用当前模型跑闭环仿真
- 计算 L_track 并反向传播

## 5. 部署

### 5.1 backend.py

```python
def detect(self, y_meas):
    # 窗口构建 (同现有)
    x_tensor = self._build_input_tensor(ymeas_window, ucmd_window)

    # 单次前向: 分类 + 恢复同时出
    cls_logits, features, y_pred = self._model(x_tensor, return_recon=True)
    probs = softmax(cls_logits)
    attack_class = ALL_ATTACK_TYPES[probs.argmax()]
    confidence = probs.max()

    # 恢复: 窗口末步 = 当前步
    y_recovered = y_pred[0, -1, :].cpu().numpy()
    y_recovered[2] = atan2(sin(y_recovered[2]), cos(y_recovered[2]))

    return DetectionResult(attack_class, confidence, y_recovered, ...)
```

无门控、无条件、无两阶段。A0 时 innov≈0 → δ≈0 → ŷ≈y_kin≈y_meas，自然退化。

### 5.2 simulate.py

无需改动。`result.y_recovered` 已接入 NMPC 闭环。

## 6. 攻击恢复机制

### 信息保留攻击 (A1/A2/A3/A6)

```
y_meas = true_state + attack
y_kin  = 控制指令前向积分 (不受当前脏测量影响)
innov  = y_meas − y_kin ≈ attack + 小漂移  ← 攻击签名
δ      = Decoder(innov 模式) ≈ −attack + 漂移修正
ŷ      = y_kin + δ ≈ true_state            ← NMPC 拿到正确估计
```

### 信息丢失攻击 (A4/A5/A7)

```
y_meas 不含当前状态信息 (冻结/丢失/重放)
innov  随时间异常增长                       ← 网络识别为不可信
δ      → 0                                 ← 网络学到: 修正反而更差
ŷ      ≈ y_kin                             ← 退化为死算预测 (信息论极限)
攻击结束后 innov 回归 → δ 恢复 → ŷ 收敛
```

**两类分化不是手工设计，是 L_track 在闭环里逼出来的。**

## 7. 验证计划

| 阶段 | 内容 | 判据 |
|------|------|------|
| 可微 NMPC 梯度 | 已完成 ✓ | FD 自洽，物理合理 |
| 阶段 1 开环 | val L_track 收敛 | 分 8 类末步 RMSE |
| 阶段 2 闭环 | L_track 下降 | loss 曲线单调/震荡下降 |
| 闭环仿真 | simulate.py --all | post_pos_rmse < 方案 A |
| 消融 | 有/无 DiffNMPC | 验证长期优化增量 |
| 消融 | 有/无 L_reg | 验证正则化防退化 |
| 消融 | 有/无 cls 辅助 | 验证分类探针是否帮助表征 |

## 8. 代码改动清单

| 文件 | 改动 | 状态 |
|------|------|------|
| `differentiable_nmpc.py` | 可微 NMPC 层 | **已完成 ✓** |
| `detector/classifier.py` | 加 Decoder, 改 forward 签名 | 待做 |
| `detector/train.py` | 加 L_track + L_reg, 加载 Y_ref | 待做 |
| `train_closed_loop.py` (新建) | 阶段 2 闭环微调 | 待做 |
| `generate_dataset.py` | 保存 Upsilon_r | 待做 |
| `detector/preprocess_data.py` | 输出 Y_ref 窗口 | 待做 |
| `model.py` | 加 rk4_step_torch, compute_error_torch | 待做 |
| `attack.py` | 加 inject_torch | 待做 |
| `backend.py` | 单次前向恢复 | 待做 |

## 9. 论文叙事

> 现有传感器攻击恢复工作通常采用"先检测攻击类型，再设计对应恢复策略"的两阶段范式。本文提出物理信息统一恢复框架：利用可微分运动学层提供的创新信号（innovation），网络无需攻击分类即可学习统一修正。进一步，通过可微 NMPC 将长期跟踪误差梯度回传到恢复器，实现端到端闭环训练，从根本上解决开环训练的分布偏移问题。实验表明，恢复驱动的共享表征同时支持高精度攻击分类（辅助任务），验证了创新信号的结构充分性。
