"""
simulate.py -- WMR 闭环仿真主程序
==========================================================================
整合模型、控制器、传感器、EKF，执行完整的轨迹跟踪闭环仿真。

仿真流程：
  t=0..T: 参考轨迹生成 -> EKF估计 -> 误差计算 -> NMPC求解 -> 机器人运动
  每一步记录：真实状态、估计状态、传感器测量、控制指令、EKF新息

参考：配置文档 Section 1 (求解器设置 T=35s, Ts=0.05)

用法：
  python simulate.py                # 运行仿真并显示结果
  python simulate.py --no-plot      # 仅运行不显示图表
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from model import WMRParams, ReferenceTrajectory, WMRKinematics
from model import EKFEstimator, SensorSimulator
from controller import NMPCController, NMPCParams


# ============================================================================
# 仿真配置
# ============================================================================

# 仿真时间参数 (配置文档 Section 1)
SIM_TIME = 35.0       # 仿真总时长 [s]
# Ts 由 WMRParams 定义 (0.05s)

# 结果保存路径
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================================
# 数据记录器
# ============================================================================

class DataLogger:
    """仿真数据记录器，存储每一步的时间序列数据"""
    
    def __init__(self):
        self.data = defaultdict(list)
    
    def record(self, **kwargs):
        """记录一步数据"""
        for key, value in kwargs.items():
            self.data[key].append(value)
    
    def to_arrays(self):
        """将所有记录转换为 numpy 数组"""
        return {k: np.array(v) for k, v in self.data.items()}
    
    def save(self, filepath: str):
        """保存为 .npz 文件"""
        arrays = self.to_arrays()
        np.savez(filepath, **arrays)
        print(f"[LOG] 数据已保存: {filepath}")


# ============================================================================
# 闭环仿真循环
# ============================================================================

def run_simulation(show_plot: bool = True):
    """执行一次完整的闭环仿真
    
    Args:
        show_plot: 是否显示结果图表
        
    Returns:
        logger: 包含所有记录数据的 DataLogger
    """
    # ---- 初始化组件 ----
    wmr_params = WMRParams()
    Ts = wmr_params.Ts
    n_steps = int(SIM_TIME / Ts)  # 700 步

    traj = ReferenceTrajectory(Ts=Ts)
    robot = WMRKinematics(wmr_params)
    sensor = SensorSimulator()
    ekf = EKFEstimator(wmr_params)
    
    nmpc_params = NMPCParams()
    ctrl = NMPCController(nmpc_params)
    ctrl.load_or_build()

    # 重置所有组件
    traj.reset()
    robot.reset()              # [0, 0.1, 0]
    ekf.reset()                # [0, 0.1, 0]
    ctrl.reset()
    np.random.seed(42)         # 固定随机种子以确保可复现

    logger = DataLogger()
    
    # 初始化控制指令
    u_cmd = np.zeros(2)        # 初始控制指令
    u_a = np.zeros(2)          # 实际执行指令 (Normal下 = u_cmd)

    print(f"[SIM] 开始闭环仿真: T={SIM_TIME}s, Ts={Ts}s, Steps={n_steps}")
    t_start = time.time()

    for step in range(n_steps):
        t = step * Ts

        # 1. 生成参考轨迹
        Upsilon_r, u_r = traj.step(t)
        Ur_seq = traj.generate_sequence(t, nmpc_params.N)

        # 2. 传感器测量 (含噪声)
        true_state = robot.state.copy()
        y_meas = sensor.measure(true_state)

        # 3. EKF 状态估计
        Upsilon_hat, ekf_innovation = ekf.step(y_meas, u_cmd)

        # 4. 计算跟踪误差 (在机器人本体坐标系下)
        X_error = WMRKinematics.compute_error(Upsilon_r, Upsilon_hat)

        # 5. NMPC 求解最优控制
        u_cmd = ctrl.solve(X_error, Ur_seq)

        # 6. 控制指令限幅
        u_a = WMRKinematics.clamp_control(u_cmd)

        # 7. 机器人运动 (RK4积分)
        robot.step(u_a)

        # 8. 记录数据
        logger.record(
            t=t,
            Upsilon_r=Upsilon_r.copy(),
            u_r=u_r.copy(),
            true_state=true_state.copy(),
            y_meas=y_meas.copy(),
            Upsilon_hat=Upsilon_hat.copy(),
            ekf_innovation=ekf_innovation.copy(),
            X_error=X_error.copy(),
            u_cmd=u_cmd.copy(),
            u_a=u_a.copy(),
        )

        # 每 100 步输出进度
        if (step + 1) % 100 == 0:
            elapsed = time.time() - t_start
            err_norm = np.linalg.norm(X_error[:2])
            print(f"  Step {step+1:4d}/{n_steps} | t={t:5.1f}s | "
                  f"|e_xy|={err_norm:.4f} | 耗时={elapsed:.1f}s")

    t_total = time.time() - t_start
    print(f"[SIM] 仿真完成！总耗时: {t_total:.1f}s "
          f"(平均 {t_total/n_steps*1000:.2f} ms/步)")

    # 保存数据
    logger.save(os.path.join(RESULT_DIR, 'sim_normal.npz'))

    # 可视化
    if show_plot:
        plot_results(logger.to_arrays())

    return logger


# ============================================================================
# 结果可视化
# ============================================================================

def plot_results(data: dict):
    """绘制仿真结果四联图
    
    图1: 2D轨迹对比 (参考 vs 实际)
    图2: 跟踪误差时序 (x_e, y_e, theta_e)
    图3: 控制指令时序 (v, w 对比)
    图4: EKF 新息时序
    """
    t = data['t']
    Ur = data['Upsilon_r']       # (N, 3)
    TrueState = data['true_state']  # (N, 3)
    Upsilon_hat = data['Upsilon_hat']  # (N, 3)
    X_err = data['X_error']      # (N, 3)
    u_cmd = data['u_cmd']        # (N, 2)
    u_a = data['u_a']            # (N, 2)
    ekf_res = data['ekf_innovation']  # (N, 3)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('WMR 8字形轨迹跟踪闭环仿真 (Normal)', fontsize=14, fontweight='bold')

    # ---- 图1: 2D轨迹 ----
    ax1 = axes[0, 0]
    ax1.plot(Ur[:, 0], Ur[:, 1], 'k--', linewidth=1.5, label='Reference (8-shape)')
    ax1.plot(TrueState[:, 0], TrueState[:, 1], 'b-', linewidth=1.2, label='Actual')
    ax1.plot(TrueState[0, 0], TrueState[0, 1], 'go', markersize=8, label='Start')
    ax1.plot(TrueState[-1, 0], TrueState[-1, 1], 'r*', markersize=10, label='End')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('2D Trajectory Tracking')
    ax1.axis('equal')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # ---- 图2: 跟踪误差 ----
    ax2 = axes[0, 1]
    ax2.plot(t, X_err[:, 0], 'r-', linewidth=1.2, label='x_e')
    ax2.plot(t, X_err[:, 1], 'g-', linewidth=1.2, label='y_e')
    ax2.plot(t, X_err[:, 2], 'b-', linewidth=1.2, label='theta_e')
    ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Tracking Error')
    ax2.set_title('Tracking Error Time Series')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # ---- 图3: 控制指令 ----
    ax3 = axes[1, 0]
    ax3.plot(t, u_cmd[:, 0], 'b-', linewidth=1.2, label='v_cmd')
    ax3.plot(t, u_cmd[:, 1], 'r-', linewidth=1.2, label='w_cmd')
    ax3.axhline(y=0.6, color='b', linestyle=':', alpha=0.4, label='v_max')
    ax3.axhline(y=-0.6, color='b', linestyle=':', alpha=0.4)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Control Command')
    ax3.set_title('NMPC Control Commands')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # ---- 图4: EKF 新息 ----
    ax4 = axes[1, 1]
    ax4.plot(t, ekf_res[:, 0], 'r-', linewidth=1.0, alpha=0.7, label='res_x')
    ax4.plot(t, ekf_res[:, 1], 'g-', linewidth=1.0, alpha=0.7, label='res_y')
    ax4.plot(t, ekf_res[:, 2], 'b-', linewidth=1.0, alpha=0.7, label='res_theta')
    ax4.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Innovation (新息)')
    ax4.set_title('EKF Innovation (新息)')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, 'sim_normal.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[PLOT] 图表已保存: {save_path}")
    plt.show()


# ============================================================================
# 定量评估
# ============================================================================

def print_metrics(data: dict):
    """输出仿真结果的关键定量指标"""
    X_err = data['X_error']
    u_cmd = data['u_cmd']
    u_a = data['u_a']

    # 位置误差统计
    pos_err = np.sqrt(X_err[:, 0]**2 + X_err[:, 1]**2)
    print("\n========== 定量评估指标 ==========")
    print(f"位置误差 RMS:        {np.sqrt(np.mean(pos_err**2)):.4f} m")
    print(f"位置误差 Max:        {np.max(pos_err):.4f} m")
    print(f"位置误差稳态 Mean:   {np.mean(pos_err[-200:]):.4f} m "
          f"(最后200步)")
    print(f"角度误差 RMS:        {np.sqrt(np.mean(X_err[:, 2]**2)):.4f} rad")
    print(f"线速度 RMS:          {np.sqrt(np.mean(u_cmd[:, 0]**2)):.4f} m/s")
    print(f"角速度 RMS:          {np.sqrt(np.mean(u_cmd[:, 1]**2)):.4f} rad/s")
    print(f"控制饱和率 (v):      {np.mean(np.abs(u_cmd[:, 0]) >= 0.59)*100:.1f}%")
    print(f"控制饱和率 (w):      {np.mean(np.abs(u_cmd[:, 1]) >= 2.89)*100:.1f}%")
    print("====================================\n")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    show = '--no-plot' not in sys.argv
    logger = run_simulation(show_plot=show)
    print_metrics(logger.to_arrays())
