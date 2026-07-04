# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在处理本仓库中的代码时提供指南。

## 项目概要

这是一个用于**轮式移动机器人（WMR）传感器攻击检测与恢复**的研究仿真系统。一个类似TurtleBot4的差速驱动机器人在NMPC控制下跟踪参考轨迹，同时其传感器测量数据可能在随机时刻受到攻击。一个基于 MultiScaleDSConvBackbone + 运动学一致性偏置注意力的分类检测器（CFMDetector）用于分类攻击类型（8类: A0-A7），检测到攻击时通过解码器重建位姿估计（运动学递推 + 学习修正）。

## 工作习惯

1. 项目使用虚拟环境：D:\anaconda\envs\learning&control
2. 使用中文思考和回复
3. 编写代码时，在顶端统一配置可调参数
4. 以科研思维而不是项目思维思考，调整模型架构时需要考虑创新性、简洁性等，而不是通过补丁维护运行
5. 总是给出面向论文读者和编辑的优秀IEEE标准可视化
6. 在代码顶部编写使用说明
7. 每次调整完架构和关键设计，都要更新系统设计说明文档。说明文档应当简明扼要。避免防御性说明
8. 任何时候，描述同一事物的术语必须统一，不得随意更改。

## 编码标准

**1. Think Before Coding — 先想再写**

- 实现前先陈述假设，不确定就明确说出来。
- 存在多种解释时列出所有选项，不静默选择。
- 有更简单的方法就指出，必要时反对过度设计。
- 遇到不清晰的地方立刻停下来，指出困惑点。

**2. Simplicity First — 简洁优先**

- 只写解决问题所需的最少代码，不做投机性设计。
- 不为单一用途的代码创建抽象层。
- 不添加未被要求的"灵活性"或"可配置性"。
- 不可能发生的场景不需要写错误处理。
- 200 行能搞定的事写成 50 行——重写。

**3. Surgical Changes — 外科手术式修改**

- 只碰必须改的部分，不顺手"改进"无关代码、注释、格式。
- 不重构没有坏的东西。
- 匹配已有代码风格，即使不是你偏好的风格。
- 如果发现无关的死代码，口头提及即可——不要擅自删除。
- 修改产生的 orphan（无用 import/变量/函数）必须清理。

**4. Goal-Driven Execution — 目标驱动执行**

- 把任务转化为可验证的目标（例如"修 bug"→"先写复现测试，再修"）。
- 多步骤任务先给出简洁计划再执行。
- 循环直至验证通过，不以"应该没问题"收尾。

## 工作流程与命令

1. 通过controller.py编译 NMPC 求解器（仅限首次运行，需要 CasADi + IPOPT）
2. 数据集通过generate_dataset.py生成数据集。将生成 `.npz` 文件输出到 `dataset/` 目录中，并生成一个 `metadata.csv`
3. 通过detector/preprocess_data.py对数据进行预处理。输出 `dataset_win/X_train.npy`、`Y_train_cls.npy`、`Y_train_atk.npy`、`normalizer.npz` 等文件。默认按轨迹族分层 IID 划分为 train/val/test (70/15/15)
4. 训练CFM检测器，通过detector/train.py。保存模型至 `detector/models/cfm_cls_best.pt`
5. 运行检测器仿真，通过simulate.py。输出 `results/sim_*.npz` 和图表
6. 运行测试集评估，通过detector/evaluate.py。输出 `eval/{model_name}_{timestamp}/` 含混淆矩阵、分类指标、Markdown 报告
7. 通过app/interactive_app.py启动交互式可视化GUI

### 独立测试脚本

```
python simulate.py                    # CFM模式闭环仿真（默认A1+lissajous）
python simulate.py --no-detector      # 无检测器基线
python simulate.py --attack A0        # 无攻击正常运行
python simulate.py --compare          # 五族轨迹无攻击跟踪对比图
python simulate.py --all              # 批量所有8种攻击
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
                                 innov_anchored = y_meas - rollout(y_meas[0], u_cmd)  (在线窗口级计算)
                                 滑动窗口: [y_meas(3) + innov_anchored(3) + u_cmd(2)] = 8 通道
                                 MultiScaleDSConvBackbone (3块膨胀深度可分离卷积, 自适应K)
                                   → 运动学一致性偏置注意力 → 8 类 softmax (A0-A7)
                                 恢复策略: A0/低置信度 → y_meas 直通; 攻击 → 解码器重建 (y_kin + delta_pred)
                                        ↓
                               y_rec → 直接作为位姿估计 (替代 EKF)
                                        ↓
                               compute_error(Upsilon_r, X_hat) → X_error(误差状态)
                                        ↓
                               NMPC.solve(X_error, Ur_seq) → u_cmd  (循环)
```

### 模块映射表


| 文件                          | 作用                                                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `model.py`                    | WMR 运动学 (RK4)，李萨如 (Lissajous)/圆形 (Circular) 轨迹生成器，传感器模拟器                                                              |
| `controller/controller.py`    | CasADi Opti NMPC：误差动力学 RK4 预测，菱形约束，控制增量代价函数，IPOPT 求解器                                                            |
| `attack.py`                   | 7 种传感器攻击类型 (A1–A7) + 正常情况 (A0)；统一的`inject()` 注入接口                                                                     |
| `simulate.py`                 | 统一闭环仿真：CFM检测器 / 无检测器基线 / 五族轨迹对比                                                                                      |
| `generate_dataset.py`         | 开环数据生成：5 种随机轨迹系列 × 8 种攻击                                                                                                 |
| `backend.py`                  | CFMDetectorBackend 推理包装器：滑动窗口缓冲 + 内部运动学 + 分类 + 恢复策略路由                                                             |
| `detector/cfm_detector.py`    | CFMDetector 模型定义：MultiScaleDSConvBackbone (膨胀深度可分离卷积) + 运动学一致性偏置注意力 + 物理引导解码器                              |
| `detector/preprocess_data.py` | 100 步滑动窗口，物理锚点归一化 (RobustNormalizer)，防数据泄漏的文件级拆分                                                                  |
| `detector/train.py`           | CFMDetector 训练脚本：L = L_cls + λ_recon*L_recon（联合训练，λ_recon=0.3，warmup=20 epoch，label smoothing 0.0），A0 每 epoch 随机降采样 |
| `detector/evaluate.py`        | 测试集分类评估：混淆矩阵 + 逐类精度/召回/F1 + 置信度 + 汇总图 + Markdown 报告                                                              |
| `detector/models/`            | 训练好的模型权重 (cfm_cls_best.pt, cfm_cls_config.npz)                                                                                     |

| `app/interactive_app.py` | tkinter 交互式 GUI：轨迹/攻击自由组合 + 时间滑块回放 + 6面板实时显示 |

### 攻击类型

每种攻击具有严格的物理含义：


| 标签 | 名称                            | 类型   |
| ---- | ------------------------------- | ------ |
| A0   | Normal (正常)                   | —     |
| A1   | Constant Bias (恒定偏移)        | 加性   |
| A2   | Sinusoidal (正弦注入)           | 加性   |
| A3   | Drift (斜坡漂移)                | 加性   |
| A4   | Replay Attack (重放攻击)        | 非加性 |
| A5   | Intermittent Dropout (信号丢失) | 非加性 |
| A6   | Scaling (缩放攻击)              | 乘性   |
| A7   | Sensor Freeze (传感器冻结)      | 非加性 |

### 轨迹系列

1. **lissajous (李萨如)** — 8字形，随机的 [v_r, ω_freq] 参数
2. **circular (圆形)** — 恒定曲率，随机的 [v_r, ω_r] 参数
3. **spiral (螺旋线)** — 半径从 R₀ 逐渐扩展到 Rmax 的阿基米德螺旋线
4. **random_waypoint (随机航点)** — 具有随机切换的、分段恒定的 ω_r
5. **square (方形)** — 直行段 + 90°圆弧转弯，边长和速度随机

### 关键设计决策

- **多尺度膨胀深度可分离卷积骨干 (MultiScaleDSConvBackbone)**：3 块膨胀深度可分离卷积 (dilation=1,3,9)，每块 3 个并行膨胀深度卷积 + Pointwise Conv 混合。不同膨胀率对应不同运动学时间尺度 (0.35s / 0.95s / 2.75s)，Pointwise Conv 隐式学习最优尺度加权——自适应 K。通道 8→32→64→96，时序 100→50→25→12。骨干参数量 ~27K，远小于旧版 Standard Conv 骨干 ~76K。
- **窗口锚定运动学新息 (Unified Anchored Innovation)**：`innov[t] = y_meas[t] - rollout(y_meas[0], u_cmd[0:t])[t]`。统一替代旧版 1-step innov + kin_res 双尺度，打破非加性攻击自指涉污染反馈环。归一化复用 y_meas 物理空间尺度 [2.5m, 2.5m, π]，简洁统一。
- **运动学一致性偏置注意力 (Kinematics-Guided Attention Pooling)**：零参数物理先验——从 innov_anchored 的 L2 范数计算每时间步的运动学一致性偏置，注入注意力分数。攻击步 (innov 大 → bias < 0) 被抑制，正常步 (innov ≈ 0 → bias ≈ 0) 获得更多关注。可学习标量 α 控制物理先验强度。
- **总参数量 ~64K**：骨干27K + 分类头1K + 解码器36K，极轻量适合嵌入式部署。
- **物理锚点归一化 (Physical-Anchor Normalization)**：y_meas 用工作空间边界 [2.5m, 2.5m, π] 作为尺度锚点，创新通道用物理异常阈值 [0.5m, 0.5m, 0.3rad] 作为尺度。避免 IQR 归一化将常规攻击信号放大至 10³-10⁵ 导致梯度爆炸。物理含义清晰，论文可辩护。
- **注意力池化分类头**：可学习的 attn_query 对特征序列加权求和，自适应地聚焦于攻击窗口最具判别力的时间步，替代简单的全局平均池化。
- **联合训练**：L_cls + λ_recon * L_recon（λ_recon=0.3，warmup=20 epoch），label smoothing 0.0（关闭），无类别权重，通过每 epoch 随机降采样 50% A0 窗口平衡类别分布。
- **恢复策略路由**：A0 正常或低置信度 (<0.5) → y_meas 直通；检测到攻击 (A1-A7) → 解码器重建 y_kin + delta_pred（物理引导：运动学递推 + 学习修正）。
- **防泄漏分层 IID 拆分**：按轨迹族分层抽样为 train/val/test (70/15/15)，同一 config 的所有窗口整体进入同一划分，保证各划分同分布且无信息泄漏。

### 物理常量（TurtleBot4 安全模式设定）

- `α = 0.17 m` (前端偏移量)
- `v_max = 0.3 m/s`, `ω_max = 1.76 rad/s`
- `Ts = 0.05 s` (20 Hz 控制频率), `T_sim = 50 s` (1000 步)
- 空间安全边界：`±2.5 m` (x, y 位置硬限幅)
- 传感器噪声：`σ_xy = 0.0 m`, `σ_θ = 0.0 rad`（关闭，简化问题）
