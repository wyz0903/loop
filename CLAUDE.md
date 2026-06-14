# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在处理本仓库中的代码时提供指南。

## 项目概要

这是一个用于**轮式移动机器人（WMR）传感器攻击检测与恢复**的研究仿真系统。一个类似TurtleBot4的差速驱动机器人在NMPC控制下跟踪参考轨迹，同时其传感器测量数据可能在随机时刻受到攻击。一个基于 SimpleConvBackbone + 通道自注意力的分类检测器（CFMDetector, cls-only）用于分类攻击类型（9类: A0-A8），检测到攻击时切换到运动学死推算作为位姿估计。本项目为一个科研项目，目标是在IEEE TIE上发表论文。

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
3. 通过detector/preprocess_data.py对数据进行预处理。输出 `dataset_win/X_train.npy`、`Y_train_cls.npy`、`Y_train_atk.npy`、`normalizer.npz` 等文件。默认按轨迹族分层 IID 划分为 train/val/test (70/15/15)
4. 训练CFM检测器，通过detector/train_cfm.py。保存模型至 `detector/models/cfm_cls_best.pt`
5. 运行检测器仿真，通过simulate.py。输出 `results/sim_*.npz` 和图表
6. 运行测试集评估，通过detector/evaluate.py。输出 `eval/{model_name}_{timestamp}/` 含混淆矩阵、分类指标、Markdown 报告
7. 通过app/interactive_app.py启动交互式可视化GUI

### 独立测试脚本

```
python simulate.py                    # CFM模式闭环仿真（默认A4+lissajous）
python simulate.py --no-detector      # 无检测器基线
python simulate.py --attack A0        # 无攻击正常运行
python simulate.py --compare          # 五族轨迹无攻击跟踪对比图
python simulate.py --all              # 批量所有9种攻击
python attack.py                      # 打印攻击类型目录 + 攻击模块自检
python model.py                       # 运行运动学 / 传感器模块自检
```

## 模型架构

### 信号流

```
ReferenceTrajectory(参考轨迹) → NMPC → u_cmd → WMRKinematics(RK4运动学) → X_true(真实状态)
                                              ↘ (通过 set_control 传入)  ↓
                                               CFMDetector(检测器)      Sensor(传感器) + Noise(噪声)
                                                   ↑                    ↓
                                      y_meas ← SensorAttack.inject()(注入攻击)
                                        ↓
                               CFMDetector.detect(y_meas)(执行检测)
                                 内部运动学预测 → 新息 innov
                                 滑动窗口: [y_meas(3) + innov(3) + u_cmd(2)] = 8 通道
                                 SimpleConvBackbone (3块 Conv-BN-ReLU-Pool)
                                   → 注意力池化 → 9 类 softmax (A0-A8)
                                 恢复策略: A0/低置信度 → y_meas 直通; 攻击 → 运动学死推算
                                        ↓
                               y_rec → 直接作为位姿估计 (替代 EKF)
                                        ↓
                               compute_error(Upsilon_r, X_hat) → X_error(误差状态)
                                        ↓
                               NMPC.solve(X_error, Ur_seq) → u_cmd  (循环)
```

### 模块映射表

| 文件 | 作用 |
| ---------------------- | ------------------------------------------------------------ |
| `model.py` | WMR 运动学 (RK4)，李萨如 (Lissajous)/圆形 (Circular) 轨迹生成器，传感器模拟器 |
| `controller/controller.py` | CasADi Opti NMPC：误差动力学 RK4 预测，菱形约束，控制增量代价函数，IPOPT 求解器 |
| `attack.py` | 8 种传感器攻击类型 (A1–A8) + 正常情况 (A0)；统一的 `inject()` 注入接口 |
| `simulate.py` | 统一闭环仿真：CFM检测器 / 无检测器基线 / 五族轨迹对比 |
| `generate_dataset.py` | 开环数据生成：5 种随机轨迹系列 × 9 种攻击 |
| `cfm_backend.py` | CFMDetectorBackend 推理包装器：滑动窗口缓冲 + 内部运动学 + 分类 + 恢复策略路由 |
| `detector/cfm_detector.py` | CFMDetector 模型定义：SimpleConvBackbone (3块 Conv-BN-ReLU-Pool) + 注意力池化分类头 |
| `detector/preprocess_data.py` | 100 步滑动窗口，物理锚点归一化 (RobustNormalizer)，防数据泄漏的文件级拆分 |
| `detector/train_cfm.py` | CFMDetector 训练脚本：L = L_cls（纯交叉熵分类，label smoothing 0.05），A0 每 epoch 随机降采样 |
| `detector/evaluate.py` | 测试集分类评估：混淆矩阵 + 逐类精度/召回/F1 + 置信度 + 汇总图 + Markdown 报告 |
| `detector/models/` | 训练好的模型权重 (cfm_cls_best.pt, cfm_cls_config.npz) |

| `app/interactive_app.py` | tkinter 交互式 GUI：轨迹/攻击自由组合 + 时间滑块回放 + 6面板实时显示 |
| `app/plot_attack_demo.py` | 攻击演示图生成（3×3子图，论文用） |
| `app/plot_trajectory_coverage.py` | 五族轨迹空间覆盖范围图 |

### 攻击类型

每种攻击具有严格的物理含义：

| 标签 | 名称 | 类型 |
| ---- | ---- | ---- |
| A0 | Normal (正常) | — |
| A1 | Constant Bias (恒定偏移) | 加性 |
| A2 | Sinusoidal (正弦注入) | 加性 |
| A3 | Drift (斜坡漂移) | 加性 |
| A4 | Step (阶跃) | 加性 |
| A5 | Replay Attack (重放攻击) | 非加性 |
| A6 | Intermittent Dropout (信号丢失) | 非加性 |
| A7 | Scaling (缩放攻击) | 乘性 |
| A8 | Sensor Freeze (传感器冻结) | 非加性 |

### 轨迹系列

1. **lissajous (李萨如)** — 8字形，随机的 [v_r, ω_freq] 参数
2. **circular (圆形)** — 恒定曲率，随机的 [v_r, ω_r] 参数
3. **spiral (螺旋线)** — 半径从 R₀ 逐渐扩展到 Rmax 的阿基米德螺旋线
4. **random_waypoint (随机航点)** — 具有随机切换的、分段恒定的 ω_r
5. **square (方形)** — 直行段 + 90°圆弧转弯，边长和速度随机

### 关键设计决策

- **简单卷积骨干 + 通道自注意力 (SimpleConvBackbone + ChannelSelfAttention)**：输入层对 8 个原始通道做多头自注意力（4 头，proj_dim=64），显式学习通道间物理耦合关系（如 innov ↔ y_meas ↔ u_cmd）。注意力矩阵 8×8 可直接可视化为论文图。其后接 3 个 Conv-BN-ReLU-MaxPool 块，通道 8→64→128→128，时序 100→50→25→12。总参数量 ~108K（含通道注意力 ~30K），远小于旧版 TCN 的 1.3M。
- **物理锚点归一化 (Physical-Anchor Normalization)**：y_meas 用工作空间边界 [2.5m, 2.5m, π] 作为尺度锚点，创新通道用物理异常阈值 [0.5m, 0.5m, 0.3rad] 作为尺度。避免 IQR 归一化将常规攻击信号放大至 10³-10⁵ 导致梯度爆炸。物理含义清晰，论文可辩护。
- **注意力池化分类头**：可学习的 attn_query 对特征序列加权求和，自适应地聚焦于攻击窗口最具判别力的时间步，替代简单的全局平均池化。
- **纯分类训练**：单一交叉熵损失 + label smoothing 0.05，无类别权重，通过每 epoch 随机降采样 50% A0 窗口平衡类别分布。
- **恢复策略路由**：A0 正常或低置信度 (<0.5) → y_meas 直通，避免注入估计误差；检测到攻击 (A1-A8) → 运动学死推算作为位姿估计。无信号重建能力，简洁可解释。
- **防泄漏分层 IID 拆分**：按轨迹族分层抽样为 train/val/test (70/15/15)，同一 config 的所有窗口整体进入同一划分，保证各划分同分布且无信息泄漏。

### 物理常量（TurtleBot4 安全模式设定）

- `α = 0.17 m` (前端偏移量)
- `v_max = 0.3 m/s`, `ω_max = 1.76 rad/s`
- `Ts = 0.05 s` (20 Hz 控制频率), `T_sim = 50 s` (1000 步)
- 空间安全边界：`±2.5 m` (x, y 位置硬限幅)
- 传感器噪声：`σ_xy = 0.0 m`, `σ_θ = 0.0 rad`（关闭，简化问题）
