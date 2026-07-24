# 方案 B：自适应观测器恢复（控制论路线）

## 1. 方案概述

用 **EKF/UKF 状态观测器** 作为恢复核心，**检测器类别先验驱动测量噪声协方差 R 调度**，实现闭环可辩护的状态估计。这是控制论白盒路线，导师最易接受。

**核心公式**（EKF 预测-修正）：
```
预测：x̂_k^- = f(x̂_{k-1}, u_{k-1})           # 运动学模型
      P_k^- = F_k P_{k-1} F_k^T + Q          # 协方差预测
修正：K_k = P_k^- H^T (H P_k^- H^T + R_c)^{-1}  # 卡尔曼增益
      x̂_k = x̂_k^- + K_k (y_k - H x̂_k^-)      # 状态修正
      P_k = (I - K_k H) P_k^-                # 协方差修正
```
其中 `R_c = R_schedule(class_c)` 由检测器类别驱动的测量噪声协方差。

**两类攻击的统一处理**：
- 信息保留攻击（A1/A2/A3/A6）：R_c 中等 → K 中等 → 测量修正有效但降权
- 信息丢失攻击（A4/A5/A7）：R_c → ∞ → K → 0 → 退化到纯预测（x̂ ≈ x̂^-）
- 正常（A0）：R_c = R_nominal → K 正常 → 全信测量

> **导师反对开环的回应**：K→0 是 EKF 在测量不可信时的**自适应降权**（标准滤波行为），不是"切换到开环控制策略"。观测器始终是闭环框架，只是增益自适应。

## 2. 数据流

### 2.1 原始数据（已有，无需改动）
同方案 A。

### 2.2 预处理（无需改动）
同方案 A（EKF 不需要预处理窗口，直接用实时 y_meas + u_cmd）。

### 2.3 R 调度表（从训练数据统计）

从 `dataset/<ts>/metadata.csv` 关联 npz 文件，按攻击类型统计 `y_meas - y_clean` 的协方差：
```python
# scripts/compute_R_schedule.py
import numpy as np, pandas as pd, os

data_dir = 'dataset/<ts>'
df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
R_schedule = {}
for _, row in df.iterrows():
    data = np.load(os.path.join(data_dir, row['filename']))
    atk = row['attack_type']
    onset = int(row['attack_onset_step'])
    offset = int(row['attack_offset_step'])
    if atk == 'A0':
        residual = data['y_meas'] - data['y_clean']
    else:
        residual = data['y_meas'][onset:offset] - data['y_clean'][onset:offset]
    R = np.cov(residual.T)
    R_schedule.setdefault(atk, []).append(R)

R_table = {k: np.mean(v, axis=0) for k, v in R_schedule.items()}
# A0 噪声关闭时 residual≈0，直接用标称值
R_table['A0'] = np.diag([1e-4, 1e-4, 1e-4])
np.savez('detector/models/R_schedule.npz', **R_table)
```

输出：`detector/models/R_schedule.npz`，含 8 个 `(3,3)` 协方差矩阵。

## 3. 架构设计

### 3.1 观测器模块（新建 `observer.py`）

```python
"""
observer.py — 自适应 EKF/UKF 状态观测器
========================================
检测器类别先验驱动测量噪声 R 调度，实现闭环可辩护的状态估计。
"""
import numpy as np
from model import WMRKinematics, WMRParams

# 信息丢失攻击 (A4/A5/A7) 的大 R fallback
R_LARGE = np.diag([10.0, 10.0, 5.0])

class AdaptiveEKF:
    """自适应扩展卡尔曼滤波器"""

    def __init__(self, R_schedule_path='detector/models/R_schedule.npz'):
        p = WMRParams()
        self.Ts = p.Ts
        self.alpha = p.alpha
        self.P = np.eye(3) * 0.01
        self.Q = np.diag([1e-4, 1e-4, 1e-5])
        self.H = np.eye(3)
        self.R_nominal = np.diag([1e-4, 1e-4, 1e-4])
        data = np.load(R_schedule_path)
        self.R_table = {k: data[k] for k in data.files}
        self.x_hat = np.zeros(3)
        self.initialized = False

    def reset(self, init_state=None):
        self.x_hat = init_state.copy() if init_state is not None else np.zeros(3)
        self.P = np.eye(3) * 0.01
        self.initialized = True

    def predict(self, u_cmd):
        """运动学预测（EKF 预测步）"""
        theta = self.x_hat[2]
        v, w = u_cmd[0], u_cmd[1]
        c, s = np.cos(theta), np.sin(theta)
        F = np.array([
            [1, 0, -self.Ts * (v * s + self.alpha * w * c)],
            [0, 1,  self.Ts * (v * c - self.alpha * w * s)],
            [0, 0, 1]
        ])
        self.x_hat = WMRKinematics.kinematic_predict(self.x_hat, u_cmd, self.Ts, self.alpha)
        self.x_hat[2] = np.arctan2(np.sin(self.x_hat[2]), np.cos(self.x_hat[2]))
        self.P = F @ self.P @ F.T + self.Q
        return self.x_hat.copy()

    def update(self, y_meas, attack_class='A0', confidence=1.0):
        """测量修正（EKF 修正步），R 由类别先验 + 置信度联合驱动"""
        if not self.initialized:
            self.reset(y_meas)
            return self.x_hat.copy()

        # R 调度：类别先验 → 测量噪声协方差
        if attack_class in ('A4', 'A5', 'A7'):
            R_class = self.R_table.get(attack_class, R_LARGE)
        else:
            R_class = self.R_table.get(attack_class, self.R_nominal)
        # 置信度门控：低置信度时 R 趋向大值（降权测量）
        R_c = confidence * R_class + (1.0 - confidence) * R_LARGE
        R_c = R_c + np.eye(3) * 1e-8

        S = self.H @ self.P @ self.H.T + R_c
        K = self.P @ self.H.T @ np.linalg.inv(S)

        innov = y_meas - self.H @ self.x_hat
        innov[2] = np.arctan2(np.sin(innov[2]), np.cos(innov[2]))

        self.x_hat = self.x_hat + K @ innov
        self.x_hat[2] = np.arctan2(np.sin(self.x_hat[2]), np.cos(self.x_hat[2]))

        I_KH = np.eye(3) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_c @ K.T
        return self.x_hat.copy()

    def step(self, y_meas, u_cmd, attack_class='A0', confidence=1.0):
        """完整一步：预测 + 修正"""
        self.predict(u_cmd)
        return self.update(y_meas, attack_class, confidence)
```

### 3.2 可选实现：观测器类型

| 类型 | 实现 | 优点 | 缺点 |
|------|------|------|------|
| **EKF**（推荐起步） | 解析雅可比 F | 轻量，可解释 | 非线性精度有限 |
| UKF | Sigma 点采样 | 非线性精度高 | 计算量×(2n+1)=7倍 |
| 自适应 EKF | R 在线估计（残差驱动） | 不依赖类别先验 | 收敛慢，需调参 |

### 3.3 可选实现：R 调度方式

| 方式 | 实现 | 优点 | 缺点 |
|------|------|------|------|
| **类别查表 + 置信度插值**（推荐） | `R_c = conf·R_table[class] + (1-conf)·R_LARGE` | 简单，鲁棒 | 依赖分类正确 |
| 残差驱动自适应 | `R_c = α·innov·innov^T + β·R_nominal` | 不依赖类别 | 收敛慢，需调 α,β |
| 混合 | 类别先验 + 残差微调 | 最鲁棒 | 复杂度高 |

### 3.4 可选实现：攻击期处理

| 方式 | 实现 | 适用 |
|------|------|------|
| **降权测量**（推荐） | R_c 增大 → K 减小 | 所有攻击 |
| 冻结增益 | 攻击期 K=0，攻击后恢复 | 信息丢失攻击 |
| 多模型并行 MMAE | 跑多个 R 的 EKF，按残差一致性加权 | 分类不确定时 |

### 3.5 信号流图
```
y_meas(被攻击) → Detector.detect() → class_id, confidence
                                      ↓
u_cmd → AdaptiveEKF.predict(u_cmd) → x̂^-（预测）
                                      ↓
y_meas + class_id + confidence → AdaptiveEKF.update(...)
  ├→ R_c = conf·R_table[class] + (1-conf)·R_LARGE  [类别先验+置信度]
  ├→ K = P H^T (H P H^T + R_c)^{-1}  [自适应增益]
  └→ x̂ = x̂^- + K·(y_meas - H·x̂^-)  [状态修正]
                                      ↓
                              y_recovered = x̂
```

## 4. 训练流程

### 4.1 无需神经网络训练
EKF 是白盒，不需要训练。唯一需要的是 **R 调度表**（从数据统计，见 2.3）。

### 4.2 可选：R 调度参数微调
若类别查表效果不佳，可用验证集微调 R 缩放因子：
```python
# scripts/tune_R_scale.py
# 对验证集闭环仿真，网格搜索每类的 R 缩放因子 s_c
# 目标：最小化 post_pos_rmse
# R_c_tuned = s_c * R_table[class]
```

## 5. 部署流程

### 5.1 backend.py 改动

```python
# backend.py 修改
from observer import AdaptiveEKF

class DetectorBackend:
    def __init__(self, ...):
        ...
        self.ekf = AdaptiveEKF(R_schedule_path='detector/models/R_schedule.npz')

    def reset(self):
        ...
        # EKF lazy init：不在 reset 时初始化，等首次窗口就绪

    def detect(self, y_meas):
        self._step_count += 1
        y_meas = np.asarray(y_meas, dtype=float).ravel()
        self._ymeas_buffer.append(y_meas.copy())
        self._ucmd_buffer.append(self._u_cmd.copy())

        if not self._is_window_ready():
            # 窗口填充期：测量直通，不做 EKF
            return DetectionResult(attack_class='A0', confidence=0.0,
                                   y_recovered=y_meas.copy(), attack_estimate=np.zeros(3), ...)

        # 分类
        ymeas_window = np.array(list(self._ymeas_buffer))
        ucmd_window = np.array(list(self._ucmd_buffer))
        x_tensor = self._build_input_tensor(ymeas_window, ucmd_window)
        attack_class, confidence = self._classify(x_tensor)

        # Lazy init：首次窗口就绪时用当前 y_meas 初始化 EKF
        if not self.ekf.initialized:
            self.ekf.reset(y_meas)

        # EKF 恢复（类别先验 + 置信度门控）
        y_rec = self.ekf.step(y_meas, self._u_cmd, attack_class, confidence)

        return DetectionResult(attack_class=attack_class, confidence=confidence,
                               y_recovered=y_rec, attack_estimate=y_meas - y_rec, ...)
```

### 5.2 simulate.py 无需改动
同方案 A。

### 5.3 闭环仿真
```bash
python simulate.py --attack A1 --trajectory lissajous
python simulate.py --all
```

## 6. 优缺点

### 优点
- **闭环可辩护**：EKF 是标准闭环滤波器，K→0 是自适应降权，不是开环策略切换。**导师最易接受**。
- **不依赖解码器训练**：白盒，无需 GPU 训练，部署轻量。
- **可解释**：每步有 P, K, R 可分析，论文好写。
- **统一处理两类**：R 调度自动处理（信息保留→R 中等，信息丢失→R→∞→K→0）。
- **不违反"不知道表达式"约束**：R 从数据统计，不预设攻击形式。

### 缺点
- **EKF 非线性精度有限**：运动学非线性（cos/sin），EKF 一阶近似可能不够。UKF 可改善但计算量大。
- **R 调度依赖分类正确**：分类错误 → R 错 → K 错 → 估计偏。置信度门控可缓解。
- **信息丢失攻击仍退化到预测**：K→0 时 x̂ ≈ x̂^-（运动学外推），本质仍是弱模型依赖。但这是滤波器自适应行为，不是开环控制。
- **无数据驱动修正**：纯白盒，不能利用训练数据里的攻击模式信息（方案 A/C 的 δ 可以）。

## 7. 实现细节

### 7.1 代码改动清单

| 文件 | 改动 |
|------|------|
| `observer.py`（新建） | AdaptiveEKF 类 |
| `scripts/compute_R_schedule.py`（新建） | 从 metadata.csv 统计 R 表 |
| `backend.py` | 加 EKF 实例，detect 里调用 ekf.step（lazy init + 置信度门控） |
| `detector/classifier.py` | 无需改动 |
| `detector/train.py` | 无需改动 |
| `simulate.py` | 无需改动 |

### 7.2 关键参数

| 参数 | 值 | 来源 |
|------|-----|------|
| `Q`（过程噪声） | `diag([1e-4, 1e-4, 1e-5])` | 需调（运动学模型误差） |
| `R_nominal` | `diag([1e-4, 1e-4, 1e-4])` | 标称测量噪声 |
| `R_LARGE` | `diag([10, 10, 5])` | 信息丢失攻击 fallback |
| `P_init` | `eye(3) * 0.01` | 初始不确定性 |
| `R_table` | 从数据统计 | `scripts/compute_R_schedule.py` |

### 7.3 依赖
- 现有环境（numpy, casadi）
- 无需 torch（EKF 纯 numpy）
- 无需新增依赖

## 8. 验证计划

### 8.1 R 调度表验证（先做）
```bash
python scripts/compute_R_schedule.py
# 检查输出的 R 表：
# A0: R = diag([1e-4, 1e-4, 1e-4])（标称值）
# A1/A2/A3/A6: R 中等（~0.01-0.1）
# A4/A5/A7: R 应极大（~1-10）
```

### 8.2 开环验证
```bash
# 写 scripts/eval_ekf_open_loop.py
# 对 test 集每步：ekf.step(y_meas, u_cmd, true_class) → x̂ vs y_clean
# 分8类统计 RMSE
```
**判据**：A1/A2/A3/A6 RMSE < 0.05m → 可接闭环

### 8.3 闭环验证
```bash
python simulate.py --attack A1 --trajectory lissajous
python simulate.py --attack A5 --trajectory lissajous
python simulate.py --all
```
**指标**：同方案 A。

### 8.4 消融实验
- EKF vs UKF：非线性精度对比
- 类别查表 R vs 残差自适应 R：鲁棒性对比
- 有/无置信度门控：分类错误时的鲁棒性
