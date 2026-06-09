# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在处理本仓库中的代码时提供指南。

## 项目概要

这是一个用于**轮式移动机器人（WMR）传感器攻击检测与恢复**的研究仿真系统。一个类似TurtleBot4的差速驱动机器人在NMPC控制下跟踪参考轨迹，同时其传感器测量数据可能在随机时刻受到攻击。一个神经网络检测器（NNDetector）用于分类攻击类型并恢复干净信号——且无需修改EKF或NMPC。本项目为一个科研项目，目标是在IEEE TIE上发表论文。

## 工作习惯

1. 项目使用虚拟环境：D:\anaconda\envs\learning&control
2. 使用中文思考和回复
3. 编写代码时，在顶端统一配置可调参数
4. 以科研思维而不是项目思维思考，调整模型架构时需要考虑创新性、简洁性等，而不是通过补丁维护运行
5. 总是给出面向论文读者和编辑的优秀IEEE标准可视化
6. 在代码顶部编写使用说明，而不是修改CLAUDE.md加入说明
7. 每次调整完架构和关键设计，都要更新系统设计说明文档。说明文档应当简明扼要。避免防御性说明
8. 任何时候，描述同一事物的术语必须统一，不得随意更改。

## 工作流程与命令

1. 通过controller.py编译 NMPC 求解器（仅限首次运行，需要 CasADi + IPOPT）
2. 数据集通过generate_dataset.py生成数据集。将生成 `.npz` 文件输出到 `dataset/` 目录中，并生成一个 `metadata.csv`
3. 通过preprocess_data.py对数据进行预处理。输出 `dataset_win/config/X_train.npy`、`Y_train_cls.npy`、`Y_train_atk.npy`、`normalizer.npz` 等文件
4. 训练攻击分类器，通过train_classifier.py。保存模型至 `models/cls_best.pt` 和 `models/cls_config.npz`
5. 运行检测器仿真，通过simulate_detector.py。输出 `results/sim_det_*.npz` 和对比图表
6. 通过analyze.py导出分析结果

### 独立测试脚本

```
python simulate.py              # 基准闭环仿真（正常运行，无攻击）
python simulate.py --no-plot    # 跳过图形显示
python simulate.py --compare    # 五族轨迹无攻击跟踪对比图
python visualize_errors.py      # 在验证集上可视化分类器的错误
python attack.py                # 打印攻击类型目录 + 攻击模块自检
python model.py                 # 运行运动学 / EKF 模块自检
```

## 模型架构

### 信号流

```
ReferenceTrajectory(参考轨迹) → NMPC → u_cmd → WMRKinematics(RK4运动学) → X_true(真实状态)
                                              ↘ (通过 set_control 传入)  ↓
                                               NNDetector(检测器)       Sensor(传感器) + Noise(噪声)
                                                   ↑                    ↓
                                      y_meas ← SensorAttack.inject()(注入攻击)
                                        ↓
                               NNDetector.detect(y_meas)(执行检测)
                                 内部运动学 → 产生新息 (innovation)
                                 滑动窗口 → NN 分类预测 + 攻击量估计 â(k)
                                 信号恢复: y_rec = y_meas − â(k)
                                        ↓
                               EKF.predict(u_cmd) → update(y_rec) → X_hat(状态估计)
                                        ↓
                               compute_error(Upsilon_r, X_hat) → X_error(误差状态)
                                        ↓
                               NMPC.solve(X_error, Ur_seq) → u_cmd  (循环)
```

### 模块映射表

| 文件                   | 作用                                                         |
| ---------------------- | ------------------------------------------------------------ |
| `model.py`             | WMR 运动学 (RK4)，EKF 估计器，李萨如 (Lissajous)/圆形 (Circular) 轨迹生成器，传感器模拟器 |
| `controller.py`        | CasADi Opti NMPC：误差动力学 RK4 预测，菱形约束，控制增量代价函数，IPOPT 求解器 |
| `attack.py`            | 8 种加性传感器攻击类型 (A1–A8) + 正常情况 (A0)；统一的 `inject()` 注入接口 |
| `detector.py`          | NNDetector（滑动窗口 + 神经网络推理 + 信号恢复 + 用于 A5 重放攻击的航位推算），OracleDetector（作为理想上限的检测器） |
| `generate_dataset.py`  | 开环数据生成：5 种随机轨迹系列 × 9 种攻击                    |
| `preprocess_data.py`   | 100 步滑动窗口，RobustScaler（基于四分位距 IQR）+ 物理量归一化，防数据泄漏的文件级拆分 |
| `train_classifier.py`  | AttackClassifier / FreqAwareClassifier（包含频率感知路径的 ResDown 编码器 + FiLM 解码器），约 90 万参数 |
| `simulate.py`          | 包含图表的基准闭环仿真（无检测器介入）                       |
| `simulate_detector.py` | 3 层级对比运行脚本：none（无） / nn（神经网络） / oracle（神谕理想状态） |
| `analyze.py`           | 计算指标 + 生成 CSV/Markdown/LaTeX 报告 + 汇总图表           |

### 攻击类型

每种攻击具有严格的物理含义：

| 标签 | 名称 | 类型 |
| ---- | ---- | ---- |
| A0   | Normal (正常) | — |
| A1   | Constant Bias (恒定偏移) | 加性 |
| A2   | Sinusoidal (正弦注入) | 加性 |
| A3   | Drift (斜坡漂移) | 加性 |
| A4   | Step(阶跃) | 加性 |
| A5   | Replay Attack (重放攻击) | 非加性 |
| A6   | Intermittent Dropout (信号丢失) | 非加性 |
| A7   | Scaling (缩放攻击) | 乘性 |
| A8   | Sensor Freeze (传感器冻结) | 非加性 |

### 轨迹系列

1. **lissajous (李萨如)** — 8字形，随机的 [v_r, ω_freq] 参数
2. **circular (圆形)** — 恒定曲率，随机的 [v_r, ω_r] 参数（在 `trajectory` 拆分模式中保留用于跨系列测试）
3. **spiral (螺旋线)** — 半径从 R₀ 逐渐扩展到 Rmax 的阿基米德螺旋线
4. **random_waypoint (随机航点)** — 具有随机切换的、分段恒定的 ω_r
5. **square (方形)** — 直行段 + 90°圆弧转弯，边长和速度随机

### 关键设计决策

- **内部运动学新息 (Internal kinematics innovation)**（非 EKF 新息）：NNDetector 维护着一个由 `u_cmd` 驱动的独立开环运动学模型。新息计算公式为 = `y_meas − X_pred_kinematic`。这使得检测器自成一体——它不需要读取 EKF 的内部状态，从而实现了真正的即插即用部署。
- **分层恢复策略**：
  - 加性攻击 (A1–A4)：`y_rec = y_meas − â(k)` — 直接减去估计攻击信号
  - 非加性攻击 (A5 重放, A6 信号丢失, A8 传感器冻结)：检测到后切换到航位推算模式（纯运动学模型预测）
  - 乘性攻击 (A7 缩放)：检测到后用估计的缩放因子逆变换 `y_rec = diag(ŝ)⁻¹ · y_meas`
- **停留时间投票 + EMA (Dwell-time voting + EMA)**：每一步的原始神经网络分类结果会通过多数投票（10 步滑动窗口）和 EMA 软分类（衰减率 α=0.3）进行融合，以减少预测的抖动。
- **尾部加权平均**：信号恢复利用了解码器最后 10 步的输出进行指数加权（α=0.7），而不是仅仅使用最后一步的输出。
- **防泄漏数据拆分**：来自同一个 `.npz` 仿真文件的窗口数据始终会被完整地分配到训练集或验证集中，防止信息泄漏。
- **基于置信度的门控恢复**：当置信度在 0.6–0.85 之间时，进行部分信号恢复混合（系数 0.25）；置信度 > 0.85 时，进行完全混合（系数 0.5）；置信度 < 0.6 时，直接放行（不做恢复干预）。

### 物理常量（TurtleBot4 安全模式设定）

- `α = 0.17 m` (前端偏移量)
- `v_max = 0.3 m/s`, `ω_max = 1.76 rad/s`
- `Ts = 0.05 s` (20 Hz 控制频率), `T_sim = 35 s` (700 步)
- 空间安全边界：`±2.5 m` (x, y 位置硬限幅)
- 传感器噪声（低噪声设置）：`σ_xy = 0.008 m`, `σ_θ = 0.004 rad`
