# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在处理本仓库中的代码时提供指南。

## 项目概要

这是一个用于**轮式移动机器人（WMR）传感器攻击检测与恢复**的研究仿真系统。一个类似TurtleBot4的差速驱动机器人在NMPC控制下跟踪参考轨迹，同时其传感器测量数据可能在随机时刻受到攻击。一个基于运动学约束的状态估计网络同时完成攻击分类（8类: A0-A7）和干净传感器信号预测（供NMPC恢复）。

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
2. 数据集通过generate_dataset.py生成数据集。输出到 `dataset/YYYYMMDD_HHMMSS/` 时间戳子目录，内含 `.npz` 文件、`metadata.csv` 和 `README.md`
3. 通过detector/preprocess_data.py对数据进行预处理。默认输入 `dataset/<ts>/`，输出自动推导为 `dataset_win/<ts>/`。输出 `X_*.npy`、`Y_*_cls.npy`、`Y_*_clean.npy`、`normalizer.npz` 等文件。默认按轨迹族分层 IID 划分为 train/val/test (70/15/15)
4. 训练NN检测器，通过detector/train.py。保存模型至 `detector/models/nn_cls_best.pt`
5. 运行检测器仿真，通过simulate.py。输出到 `results/simulations/` 目录（`.npz` 数据和 `.csv` 指标）
6. 运行测试集评估，通过detector/evaluate.py。输出 `eval/{model_name}_{timestamp}/` 含混淆矩阵、分类指标、Markdown 报告
7. 通过app/interactive_app.py启动交互式可视化GUI

### 独立测试脚本

```
python simulate.py                    # NN模式闭环仿真（默认A1+lissajous）
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
                                              检测器                   Sensor(传感器) + Noise(噪声)
                                                   ↑                    ↓
                                      y_meas ← SensorAttack.inject()(注入攻击)
                                        ↓
                               检测器.detect(y_meas)(执行检测)
                                 滑动窗口: [y_meas(3) + u_cmd(2)] = 5 通道
                                 Encoder → q (干净状态估计, 3ch)
                                 攻击信号自然浮现: attack_res = y_meas - q
                                 分类: 注意力池化(F) + ||y_meas - q|| → 8 类 softmax
                                 恢复: q 替代被攻击量测 → 送入 NMPC
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
| `simulate.py`                 | 统一闭环仿真：NN检测器 / 无检测器基线 / 五族轨迹对比                                                                                      |
| `generate_dataset.py`         | 开环数据生成：5 种随机轨迹系列 × 8 种攻击                                                                                                 |
| `backend.py`                  | DetectorBackend 推理包装器：滑动窗口缓冲 + 8 类攻击分类（纯检测，无恢复）                                              |
| `detector/classifier.py`      | 检测模型定义：TemporalEncoder (Embedding + 4×DilatedResidualBlock) + StateHead (q) + AttackClassifier (注意力池化 + 攻击残差反馈) |
| `detector/preprocess_data.py` | 128 步滑动窗口，物理锚点归一化 (RobustNormalizer)，防数据泄漏的文件级拆分                                                                  |
| `detector/train.py`           | 训练脚本：L = L_recon(MSE q vs y_clean) + λ_kin·L_kin(运动学一致性) + λ_cls·L_cls(交叉熵)，随机位置块掩码，A0 降采样 50% |
| `detector/evaluate.py`        | 测试集分类评估：混淆矩阵 + 逐类精度/召回/F1 + 置信度 + 汇总图 + Markdown 报告                                                              |
| `detector/models/`            | 训练好的模型权重 (nn_cls_best.pt)                                                                                                        |

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

- **状态估计而非手工特征**：编码器直接输出干净位姿估计 q (3ch)，运动学方程作为损失约束 (L_kin) 而非输入特征。网络受约束学习"什么状态服从物理"，而非被告知"看什么特征"。灵感来自 JEPA 的表示空间预测思想。
- **运动学一致性损失 (L_kin)**：MSE(q[t], Euler_norm(q[t-1], u[t-1]))，在归一化空间中计算。约束编码器输出的 q 必须时间自洽。通过去归一化→物理 Euler 递推→再归一化实现，零可学习参数。这才是真正的 PINN——物理方程在损失中约束学习过程。
- **攻击残差自然浮现**：`attack_res = y_meas - q`，网络学习分解 y_meas = q + attack。攻击残差的大小直接指示攻击强度，作为辅助信号输入分类器。
- **TemporalEncoder 骨干**：Embedding (5→48, k=5) + 4×DilatedResidualBlock (48ch, d=1/3/9/27, 无降采样)。感受野覆盖 0.25s→8.25s。无降采样避免 U-Net 的信息瓶颈，无多尺度并行分支避免参数膨胀。参数量 ~58K。
- **随机位置块掩码** vs 尾部掩码：掩码位置在窗口中随机选取，防止模型学"前面总是干净"的位置捷径。仅掩码 y_meas (u_cmd 始终可见以维持运动学递推)。
- **q 的双重角色**：既是训练中的中间表示（受 L_recon+L_kin 约束），也是推理时的最终输出（反归一化后送入 NMPC 替代被攻击量测，实现传感器恢复）。
- **物理锚点归一化**：y_meas / [2.5m, 2.5m, π]，u_cmd / [0.3, 1.76]。y_clean 使用相同参数归一化以匹配合并输出空间。
- **防泄漏分层 IID 拆分**：按轨迹族分层抽样为 train/val/test (70/15/15)。
- **物理锚点归一化 (Physical-Anchor Normalization)**：y_meas 用工作空间边界 [2.5m, 2.5m, π] 作为尺度锚点，u_cmd 用物理上限 [0.3, 1.76] 归一化。避免数据驱动归一化（IQR/Z-score）的梯度问题，物理含义清晰。
- **u_cmd 缓冲索引约定**：运行时 `_ucmd_buffer` slot j 存 `u_{j-1→j}`（simulate.py 中 detect 先于 set_control 调用），与训练数据 (y_k, u_{k→k+1}) 有一步错位。分类器输入保持现状（修正需重训）。
- **防泄漏分层 IID 拆分**：按轨迹族分层抽样为 train/val/test (70/15/15)，同一 config 的所有窗口整体进入同一划分，保证各划分同分布且无信息泄漏。
- **交叉熵分类**：label smoothing 0.0（关闭），无类别权重，通过每 epoch 随机降采样 50% A0 窗口平衡类别分布。

### 物理常量（TurtleBot4 安全模式设定）

- `α = 0.17 m` (前端偏移量)
- `v_max = 0.3 m/s`, `ω_max = 1.76 rad/s`
- `Ts = 0.05 s` (20 Hz 控制频率), `T_sim = 50 s` (1000 步)
- 空间安全边界：`±2.5 m` (x, y 位置硬限幅)
- 传感器噪声：`σ_xy = 0.0 m`, `σ_θ = 0.0 rad`（关闭，简化问题）
- 攻击协议（数据集生成与闭环仿真完全统一）：时长固定 `5.0s`（100 步），onset∈[10,40]s
