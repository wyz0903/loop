# 方案 A：物理引导解码器恢复（数据驱动路线）

## 1. 方案概述

复用现有物理引导解码器 `y_pred = y_kin + delta_pred`，加入**类别条件 embedding** 使 δ 网络在检测器分类引导下学习不同恢复行为，通过**开环监督训练**后直接接入闭环。这是改动最小、复用现有工作最多的路线。

**核心公式**：
```
ŷ_k = y_kin_k + δ_k
y_kin_k = rollout(y_meas[0], u[0:k])      # 运动学外推（物理先验，不依赖当前脏测量）
δ_k = Decoder(features_k, class_emb_c)     # 类别条件修正项
```

**两类攻击的统一处理**：
- 信息保留攻击（A1/A2/A3/A6）：δ 学到有效修正 → ŷ ≈ y_clean
- 信息丢失攻击（A4/A5/A7）：δ 学到 ≈0 → ŷ ≈ y_kin（自动退化到运动学外推）

## 2. 数据流

### 2.1 原始数据生成（已有，无需改动）
```
python generate_dataset.py --num-per-family 12
```
- 输出：`dataset/<ts>/sim_*.npz` + `metadata.csv`
- 每个 npz 含：`y_meas(1000,3)`, `y_clean(1000,3)`, `u_cmd(1000,2)`, `true_state(1000,3)`, `attack_type_label`, `attack_onset`, `attack_offset`
- 攻击参数固定（attack.py 写死），暂不随机化

### 2.2 预处理（已有，需小改）
```
python detector/preprocess_data.py --input-dir dataset/<ts>/
```
- 输出：`dataset_win/<ts>/X_{train,val,test}.npy`, `Y_{train,val,test}_cls.npy`, `Y_{train,val,test}_clean.npy`, `normalizer.npz`
- **需新增**：`Y_{train,val,test}_cls_window.npy`——窗口级类别标签（当前 `Y_cls.npy` 是窗口级，已满足）
- 窗口格式：`X (N, 128, 8)` = `[y_meas(3) + innov_anchored(3) + u_cmd(2)]`
- `Y_clean (N, 128, 3)` = 干净位姿窗口（重构目标）
- `Y_cls (N,)` = 窗口级攻击类别（0-7）

### 2.3 训练数据格式（升级后）
- 输入：`X (B, 128, 8)` + `Y_cls (B,)` → 类别 embedding
- 目标：`Y_clean (B, 128, 3)` → MSE 监督
- 类别 embedding：`nn.Embedding(8, 16)` → 拼接到解码器特征

## 3. 架构设计

### 3.1 检测器（已有，`detector/classifier.py`）
```
MultiScaleDSConvBackbone (8→32→64→96, 128→64→32→16)
  → KinematicConsistencyBias (零参数物理先验)
  → Attention Pooling → cls_head (96→8)
  → PhysicsGuidedDecoder (96→64→32→16→3, upsample 16→128)
```
- 输出：`cls_logits (B,8)`, `y_pred (B,128,3) = y_kin + delta_pred`

### 3.2 恢复器升级（改动点）

**升级1：类别条件注入**

在 `PhysicsGuidedDecoder` 输入端拼接类别 embedding：
```python
# classifier.py 修改
class Detector(nn.Module):
    def __init__(self, ..., num_classes=8, class_emb_dim=16):
        ...
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)
        # decoder 输入通道：d_model + class_emb_dim = 96 + 16 = 112
        self.decoder = PhysicsGuidedDecoder(d_model=96 + class_emb_dim)

    def decode(self, features, x_norm, class_ids=None):
        y0_phys = x_norm[:, 0, :3] * self.ymeas_scale + self.ymeas_median
        u_phys = x_norm[:, :, -2:] * self.cmd_max
        with torch.no_grad():
            y_kin = batch_kinematic_rollout(y0_phys, u_phys)
        # 类别条件注入
        if class_ids is not None:
            emb = self.class_emb(class_ids)  # (B, 16)
            emb = emb.unsqueeze(1).expand(-1, features.shape[1], -1)  # (B, 16, 16)
            features_cond = torch.cat([features, emb], dim=-1)  # (B, 16, 112)
        else:
            features_cond = features
        delta_pred = self.decoder(features_cond)
        return y_kin + delta_pred, delta_pred

    def forward(self, x, return_recon=False, class_ids=None):
        features = self.encode(x)
        cls_logits = self.classify(features, x)
        if return_recon and self.decoder is not None:
            y_pred, delta_pred = self.decode(features, x, class_ids)
            return cls_logits, features, y_pred, delta_pred
        return cls_logits, features
```

**升级2：`PhysicsGuidedDecoder` 输入通道适配**
```python
class PhysicsGuidedDecoder(nn.Module):
    def __init__(self, d_model=96, class_emb_dim=16):
        super().__init__()
        ci = d_model + class_emb_dim  # 112
        layers = []
        for co in DECODER_CHANNELS:  # [64, 32, 16]
            layers.extend([nn.ConvTranspose1d(ci, co, 4, 2, 1),
                           nn.BatchNorm1d(co), nn.GELU()])
            ci = co
        layers.append(nn.Conv1d(ci, 3, 5, padding=2))
        self.upsample = nn.Sequential(*layers)
```

### 3.3 可选实现：类别条件注入方式

| 方式 | 实现 | 优点 | 缺点 |
|------|------|------|------|
| **Embedding 拼接**（推荐） | `cat([features, emb.expand(...)], dim=-1)` | 简单，不改变骨干 | 参数量略增 |
| FiLM 调制 | `γ(c)·features + β(c)` | 调制更精细 | 需额外网络学 γ,β |
| 独立 Head | 每类一个 decoder head | 类间隔离 | 参数量×8，过拟合风险 |

### 3.4 信号流图
```
y_meas(被攻击) → Detector.detect()
  ├→ cls_logits → softmax → class_id, confidence
  ├→ features (B,16,96)
  └→ decode(features, x_norm, class_id)
       ├→ y_kin = rollout(y_meas[0], u)  [物理先验]
       └→ delta = Decoder(cat[features, emb(class_id)])  [数据驱动修正]
       → y_pred = y_kin + delta
       → y_recovered = y_pred[0, -1, :]  [窗口末步=当前步]
```

## 4. 训练流程

### 4.1 阶段1：开环监督预训练（升级现有 train.py）

```python
# train.py 修改
# 在 train_epoch 和 evaluate 中：
cls_logits, _, y_pred, _ = model(x, return_recon=True, class_ids=cls_label)
loss_cls = F.cross_entropy(cls_logits, cls_label)
loss_recon = F.mse_loss(y_pred, y_clean)
# 可选：末步加权（强化当前步恢复精度）
loss_recon_last = F.mse_loss(y_pred[:, -1, :], y_clean[:, -1, :])
loss = loss_cls + RECON_LAMBDA * (0.7 * loss_recon + 0.3 * loss_recon_last)
```

**关键参数**：
- `RECON_LAMBDA = 0.3`（保持现有）
- `RECON_WARMUP = 20`（前20 epoch 只训分类）
- `class_emb_dim = 16`
- 训练命令：`python detector/train.py --data-dir dataset_win/<ts>/`

### 4.2 可选实现：训练目标

| 目标 | 公式 | 适用场景 |
|------|------|----------|
| 全窗 MSE（现有） | `MSE(y_pred, y_clean)` | 通用，整窗恢复 |
| 末步加权 MSE | `0.7*MSE(全窗) + 0.3*MSE(末步)` | 强化当前步精度（推荐） |
| 跟踪误差代理 | `MSE(compute_error(Upsilon_r, y_pred), 0)` | 需参考轨迹，更贴近目标 |

## 5. 部署流程

### 5.1 backend.py 改动

```python
# backend.py 修改
class DetectorBackend:
    CONFIDENCE_THRESHOLD = 0.5  # 恢复门控阈值

    def detect(self, y_meas):
        ...
        if not self._is_window_ready():
            return DetectionResult(attack_class='A0', confidence=0.0,
                                   y_recovered=y_meas.copy(), ...)

        ymeas_window = np.array(list(self._ymeas_buffer))
        ucmd_window = np.array(list(self._ucmd_buffer))
        x_tensor = self._build_input_tensor(ymeas_window, ucmd_window)
        attack_class, confidence, y_pred = self._classify_and_recover(x_tensor)

        # 恢复策略：A0门控 + 置信度门控
        if attack_class == 'A0' or confidence < self.CONFIDENCE_THRESHOLD:
            y_recovered = y_meas.copy()
            attack_estimate = np.zeros(3)
        else:
            y_recovered = y_pred[0, -1, :].cpu().numpy()  # 窗口末步
            y_recovered[2] = np.arctan2(np.sin(y_recovered[2]), np.cos(y_recovered[2]))
            attack_estimate = y_meas - y_recovered

        return DetectionResult(attack_class=attack_class, confidence=confidence,
                               y_recovered=y_recovered, attack_estimate=attack_estimate, ...)

    def _classify_and_recover(self, x_tensor):
        with torch.no_grad():
            cls_logits, _, y_pred, _ = self._model(x_tensor, return_recon=True)
        probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
        pred_cls = int(probs.argmax())
        return ALL_ATTACK_TYPES[pred_cls], float(probs[pred_cls]), y_pred
```

### 5.2 simulate.py 无需改动
- 已支持 `result.y_recovered` 送入 NMPC（`simulate.py:175-184`）

### 5.3 闭环仿真
```bash
python simulate.py --attack A1 --trajectory lissajous
python simulate.py --all  # 批量8种攻击
```

## 6. 优缺点

### 优点
- **改动最小**：复用现有 Detector + PhysicsGuidedDecoder，只加 class embedding
- **快速验证**：1-2天可完成升级+训练+闭环验证
- **不违反约束**：y_kin 是运动学（合法），delta 从数据学（不预设攻击表达式）
- **统一处理两类**：δ 自动分化（信息保留→修正，信息丢失→≈0）

### 缺点
- **闭环分布偏移未解决**：训练数据来自无恢复闭环，部署接恢复后分布变
- **信息丢失攻击退化到 y_kin**：本质是弱开环（导师可能仍有疑虑）
- **固定参数限制**：当前数据集攻击参数固定，泛化性未验证
- **末步精度不保证**：整窗 MSE 不专门优化末步，可能末步偏差大

## 7. 实现细节

### 7.1 代码改动清单

| 文件 | 改动 | 行数估计 |
|------|------|----------|
| `detector/classifier.py` | 加 `class_emb`，改 `decode()` 和 `forward()` 签名，改 `PhysicsGuidedDecoder.__init__` | ~30行 |
| `detector/train.py` | `train_epoch`/`evaluate` 传 `class_ids`，可选末步加权 loss | ~15行 |
| `backend.py` | `_classify` → `_classify_and_recover`，`detect` 加恢复门控 | ~25行 |
| `simulate.py` | 无需改动 | 0 |

### 7.2 关键参数

| 参数 | 值 | 来源 |
|------|-----|------|
| `WINDOW_SIZE` | 128 | 现有 |
| `class_emb_dim` | 16 | 新增（可调 8-32） |
| `CONFIDENCE_THRESHOLD` | 0.5 | 新增（可调 0.3-0.7） |
| `RECON_LAMBDA` | 0.3 | 现有 |
| `RECON_WARMUP` | 20 | 现有 |

### 7.3 依赖
- 现有环境（torch, casadi, numpy, matplotlib）
- 无需新增依赖

## 8. 验证计划

### 8.1 开环验证（先做，30分钟）
```bash
# 1. 确认 decoder 权重存在且收敛
python detector/train.py --eval-only detector/models/nn_cls_best.pt

# 2. 写开环末步 RMSE 脚本（新建 scripts/eval_open_loop.py）
# 对 test 集每窗口：model(return_recon=True) → y_pred[:,-1] vs y_clean[:,-1]
# 分8类统计 RMSE
```
**判据**：A1/A2/A3/A6 末步 RMSE < 0.02m → 可接闭环；否则需重训

### 8.2 闭环验证
```bash
# 单攻击验证
python simulate.py --attack A1 --trajectory lissajous
python simulate.py --attack A5 --trajectory lissajous  # 信息丢失，看退化行为

# 批量验证
python simulate.py --all
```
**指标**：
- `post_pos_rmse`：攻击期位置 RMSE（< 0.1m 为可接受）
- `detection_accuracy`：检测准确率（> 80%）
- 轨迹图：攻击 onset 后是否"跳动→收敛"

### 8.3 消融实验
- 无类别条件 vs 有类别条件：验证 class embedding 是否改善分化
- 全用 y_pred vs A0 门控：验证门控是否减少正常段重构误差
- 末步加权 vs 全窗 MSE：验证末步精度是否改善闭环收敛

## 9. 审查修复（子智能体审查后必修）

### 修复1 [严重]：删除 Upgrade 2，只保留 Upgrade 1
Upgrade 1 和 Upgrade 2 对 `PhysicsGuidedDecoder` 输入通道的修改互相矛盾（双重叠加导致 128≠112 RuntimeError）。
- **修复**：只保留 Upgrade 1（调用侧传 `PhysicsGuidedDecoder(d_model=96+16)`），**不改** `PhysicsGuidedDecoder.__init__`。现有 `ci = d_model` 已能正确接收 112 通道。

### 修复2 [严重]：backend 改两阶段推理
部署时 `_classify_and_recover` 未传 `class_ids`，decoder 期望 112 通道但收到 96 → RuntimeError。
- **修复**：先分类获取 `pred_cls`，再用 `pred_cls` 作为 `class_ids` 执行恢复：
```python
def _classify_and_recover(self, x_tensor):
    with torch.no_grad():
        cls_logits, features = self._model(x_tensor, return_recon=False)
        probs = torch.softmax(cls_logits, dim=1)
        pred_cls = probs.argmax(dim=1)
        _, _, y_pred, _ = self._model(x_tensor, return_recon=True, class_ids=pred_cls)
    probs_np = probs.cpu().numpy()[0]
    pred_cls_int = int(pred_cls[0])
    return ALL_ATTACK_TYPES[pred_cls_int], float(probs_np[pred_cls_int]), y_pred
```

### 修复3 [中等]：训练时也用 predicted class（消除 train-test mismatch）
训练用 ground truth label、部署用 predicted label，分类错误时 decoder 收到错误 embedding。
- **修复**（推荐策略a）：训练时也用 predicted class + detach：
```python
cls_logits, _, y_pred, _ = model(x, return_recon=True,
                                  class_ids=cls_logits.argmax(dim=1).detach())
```

### 修复4 [遗漏]：声明旧权重不兼容
架构改动后 `nn_cls_best.pt` 缺少 `class_emb` 和 decoder 新通道权重，`strict=False` 会静默降级到随机初始化。
- **修复**：方案明确"架构改动后必须重训"。backend 加载时检查必要 key 是否存在，缺失则报错。

### 修复5 [遗漏]：eval-only 模式也需传 class_ids
`train.py` 的 `evaluate()` 中 `model(x, return_recon=True)` 未传 `class_ids`，同样 RuntimeError。
- **修复**：evaluate 函数加 `class_ids=cls_label` 参数。
