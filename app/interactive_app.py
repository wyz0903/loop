"""
interactive_app.py — WMR 传感器攻击仿真交互式浏览器
==============================================================================
提供图形界面用于自由探索 WMR 在不同轨迹和攻击组合下的行为。

功能:
  - 双模式轨迹选择: 从数据集选择或自定义参数
  - 8 种攻击类型 (A0~A7), 可调起始时间和持续时间
  - 后台运行仿真 (NMPC+IPOPT, 约 1000 步 / 20 秒)
  - 时间滑块拖拽回放: 6 面板实时更新 (轨迹/误差/控制/新息/攻击/测量对比)
  - 键盘快捷键: F5 运行, Space 播放, ← → 步进, Home/End 跳转

用法:
  python interactive_app.py           # 启动 GUI
  run_app.bat                         # 双击启动 (Windows)

依赖:
  - tkinter (标准库)
  - matplotlib + TkAgg 后端
  - conda 环境: D:\\anaconda\\envs\\learning&control
"""

import os
import sys
import time
import queue
import threading
import argparse
import traceback

# 将项目根目录加入 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

# ---- matplotlib: TkAgg 后端 ----
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ---- tkinter ----
import tkinter as tk
from tkinter import ttk, messagebox

# ---- 项目模块 ----
from model import WMRParams, WMRKinematics, SensorSimulator
from controller import NMPCController, NMPCParams
from attack import SensorAttack, AttackConfig, ALL_ATTACK_TYPES, ATTACK_NAMES, ATK_COLORS
from backend import DetectorBackend

# ============================================================================
# 全局常量
# ============================================================================

DEFAULT_TS = 0.05
DEFAULT_SIM_TIME = 50.0
SIM_STEPS_DEFAULT = int(DEFAULT_SIM_TIME / DEFAULT_TS)  # 1000

FAMILIES_ORDER = ['lissajous', 'circular', 'spiral', 'random_waypoint', 'square']
FAMILY_NAMES_ZH = {
    'lissajous': '李萨如 (Lissajous)',
    'circular': '圆形 (Circular)',
    'spiral': '螺旋线 (Spiral)',
    'random_waypoint': '随机路径点 (Random Waypoint)',
    'square': '正方形 (Square)',
}

# 攻击颜色方案 (与项目 plot_attack_demo.py 一致)
# IEEE 论文绘图样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 辅助函数
# ============================================================================

def load_dataset_configs(metadata_path: str) -> list[dict]:
    """从 metadata.csv 加载全部轨迹配置"""
    df = pd.read_csv(metadata_path)
    configs = []
    for config_id in sorted(df['config_id'].unique()):
        row = df[df['config_id'] == config_id].iloc[0]
        cfg = {
            'config_id': int(config_id),
            'family': str(row['trajectory_family']),
            'traj_seed': int(row['traj_seed']),
        }
        # 解析各族的参数
        family = cfg['family']
        if family == 'lissajous':
            cfg['v_r'] = float(row.get('traj_v_r', 0.15))
            cfg['w_freq'] = float(row.get('traj_w_freq', 0.14))
            cfg['w_amp'] = float(row.get('traj_w_amp', 2.4048 * cfg['w_freq']))
        elif family == 'circular':
            cfg['v_r'] = float(row.get('traj_v_r', 0.15))
            cfg['w_r'] = float(row.get('traj_w_r', 0.3))
        elif family == 'spiral':
            cfg['v'] = float(row.get('traj_v', 0.20))
            cfg['R0'] = float(row.get('traj_R0', 0.10))
            cfg['Rmax'] = float(row.get('traj_Rmax', 1.50))
            cfg['direction'] = int(row.get('traj_direction', 1))
        elif family == 'random_waypoint':
            cfg['v_r'] = float(row.get('traj_v_r', 0.20))
        elif family == 'square':
            cfg['side'] = float(row.get('traj_side', 2.0))
            cfg['v_straight'] = float(row.get('traj_v_straight', 0.25))
            cfg['v_turn'] = float(row.get('traj_v_turn', 0.15))
            cfg['w_turn'] = float(row.get('traj_w_turn', 1.0))
        configs.append(cfg)
    return configs


# ============================================================================
# PerfectDetectorBackend — 理想检测器 (100% 检测, 运动学死推算恢复, 仅用于对比基线)
# ============================================================================

class PerfectDetectorBackend:
    """理想检测器: 检测 100% 准确 (基于 ground truth)，恢复使用运动学死推算。

    检测: 通过 set_ground_truth() 获知真实攻击状态 → detect() 返回 100% 正确的类别。
    恢复: 攻击中完全开环运动学死推算 (不再使用 y_clean 作弊)，正常时信任传感器直通。
    API 与 DetectorBackend 一致: detect(y_meas) → DetectionResult。
    """

    def __init__(self):
        self._attack_type = 'A0'
        self._attack_signal = np.zeros(3)
        self._internal_state = np.zeros(3)    # [x, y, theta] 用于运动学死推算
        self._u_cmd = np.zeros(2)             # [v, w] 控制指令缓存

    def set_ground_truth(self, attack_type: str, attack_signal: np.ndarray):
        """每步调用: 传入当前步的真实攻击信息 (由仿真循环提供)。"""
        self._attack_type = attack_type
        self._attack_signal = np.asarray(attack_signal, dtype=float).ravel()

    def detect(self, y_meas: np.ndarray) -> 'DetectionResult':
        """返回: 正确攻击类别 + 运动学死推算恢复 (不再使用 y_clean 作弊)。

        检测仍为 100% 准确 (基于 ground truth attack_signal)。
        恢复策略: 正常时信任传感器直通，攻击时完全开环运动学死推算。
        """
        from backend import DetectionResult
        y_meas = np.asarray(y_meas, dtype=float).ravel()

        # 根据真实攻击信号判断当前是否处于攻击中
        is_under_attack = np.any(np.abs(self._attack_signal) > 1e-10)

        if is_under_attack:
            # 攻击中 → 完全开环运动学死推算, 不再相信传感器
            X_pred = WMRKinematics.kinematic_predict(self._internal_state, self._u_cmd)
            y_recovered = X_pred.copy()
        else:
            # 正常 → 信任传感器测量直通
            y_recovered = y_meas.copy()

        # 内部状态传播: 为下一步死推算做准备
        self._internal_state = y_recovered.copy()

        attack_class = self._attack_type if is_under_attack else 'A0'
        attack_estimate = self._attack_signal.copy() if is_under_attack else np.zeros(3)

        return DetectionResult(
            attack_class=attack_class, confidence=1.0,
            y_recovered=y_recovered,
            attack_estimate=attack_estimate,
            features={'detector': 'perfect'})

    def set_control(self, u_cmd: np.ndarray) -> None:
        self._u_cmd = np.asarray(u_cmd, dtype=float).ravel()

    def reset(self) -> None:
        self._internal_state = np.zeros(3)
        self._u_cmd = np.zeros(2)


# ============================================================================
# SimulationWorker — 后台仿真线程
# ============================================================================

class SimulationWorker(threading.Thread):
    """在后台线程中运行完整 NMPC 闭环仿真

    通过 queue 向主线程报告进度和结果:
      ('progress', step, t, elapsed) — 每 10 步
      ('done', result_dict)          — 仿真完成
      ('error', error_message)       — 异常
    """

    def __init__(self, result_queue: queue.Queue,
                 traj_family: str, traj_seed: int,
                 attack_type: str, attack_onset: float,
                 attack_duration: float, sim_time: float,
                 detector_mode: str = 'cfm'):
        super().__init__(daemon=True)
        self._q = result_queue
        self._traj_family = traj_family
        self._traj_seed = traj_seed
        self._attack_type = attack_type
        self._attack_onset = attack_onset
        self._attack_duration = attack_duration
        self._sim_time = sim_time
        self._detector_mode = detector_mode  # 'cfm' | 'perfect' | 'none'

    def run(self):
        try:
            self._run_simulation()
        except Exception as e:
            tb = traceback.format_exc()
            self._q.put(('error', f'{e}\n\n{tb}'))

    def _run_simulation(self):
        # ---- 延迟导入重型模块 ----
        from model import RandomizedTrajectory

        Ts = DEFAULT_TS
        n_steps = int(self._sim_time / Ts)

        # ---- 轨迹 ----
        traj = RandomizedTrajectory(seed=self._traj_seed, family=self._traj_family)

        # ---- 初始化组件 ----
        wmr_params = WMRParams()
        robot = WMRKinematics(wmr_params)
        sensor = SensorSimulator()
        ctrl = NMPCController(NMPCParams())
        ctrl.load_or_build()

        atk_cfg = AttackConfig(attack_duration=self._attack_duration)
        attacker = SensorAttack(attack_type=self._attack_type,
                                onset_time=self._attack_onset,
                                config=atk_cfg, seed=self._traj_seed)

        # ---- 重置 (初始偏移与 simulate.py 一致) ----
        traj.reset()
        _INIT_POS_MAX = 0.15
        _INIT_HEADING_MAX = 0.2
        perturb_rng = np.random.RandomState(self._traj_seed)
        init_state = np.array([
            traj._x_r + perturb_rng.uniform(0.0, _INIT_POS_MAX) * perturb_rng.choice([-1, 1]),
            traj._y_r + perturb_rng.uniform(0.0, _INIT_POS_MAX) * perturb_rng.choice([-1, 1]),
            traj._theta_r + perturb_rng.uniform(0.0, _INIT_HEADING_MAX) * perturb_rng.choice([-1, 1]),
        ])
        robot.reset(init_state)
        ctrl.reset()
        attacker.reset()
        np.random.seed(self._traj_seed)

        # ---- 检测器 ----
        detector_mode = self._detector_mode
        model_dir = os.path.join(PROJECT_ROOT, 'detector', 'models')
        norm_dir = os.path.join(PROJECT_ROOT, 'dataset_win', 'normalizer.npz')
        cfm_model = os.path.join(model_dir, 'cfm_cls_best.pt')

        detector = None
        if detector_mode == 'cfm':
            if not os.path.exists(cfm_model):
                self._q.put(('warning', f'CFM模型未找到: {cfm_model}\n回退为无检测器'))
            else:
                detector = DetectorBackend(model_path=cfm_model, norm_path=norm_dir)
                detector.reset()
        elif detector_mode == 'perfect':
            detector = PerfectDetectorBackend()

        # ---- 仿真循环 ----
        data = {
            't': np.zeros(n_steps, dtype=np.float32),
            'true_state': np.zeros((n_steps, 3), dtype=np.float32),
            'y_meas': np.zeros((n_steps, 3), dtype=np.float32),
            'y_rec': np.zeros((n_steps, 3), dtype=np.float32),
            'attack_signal': np.zeros((n_steps, 3), dtype=np.float32),
            'sensor_noise': np.zeros((n_steps, 3), dtype=np.float32),
            'Upsilon_r': np.zeros((n_steps, 3), dtype=np.float32),
            'u_r': np.zeros((n_steps, 2), dtype=np.float32),
            'Upsilon_hat': np.zeros((n_steps, 3), dtype=np.float32),
            'X_error': np.zeros((n_steps, 3), dtype=np.float32),
            'u_cmd': np.zeros((n_steps, 2), dtype=np.float32),
            'u_a': np.zeros((n_steps, 2), dtype=np.float32),
            'attack_active': np.zeros(n_steps, dtype=np.float32),
            'det_class': np.zeros(n_steps, dtype=object),
            'det_conf': np.zeros(n_steps, dtype=np.float32),
            'det_attack_est': np.zeros((n_steps, 3), dtype=np.float32),
        }

        u_cmd = np.zeros(2)

        t_start_wall = time.time()

        for step in range(n_steps):
            t = step * Ts

            # 1. 参考轨迹
            Upsilon_r, u_r = traj.step(t)
            Ur_seq = traj.generate_sequence(t, NMPCParams().N)

            # 2. 测量 (含攻击)
            true_state = robot.state.copy()
            noise = sensor.noise_std * np.random.randn(3)
            y_clean = true_state + noise
            y_meas = attacker.inject(t, y_clean)
            attack_signal = y_meas - y_clean

            # 3. 检测器
            if detector is None:
                y_rec = y_meas.copy()
                det_class = 'A0'
                det_conf = 0.0
                det_attack_est = np.zeros(3)
            else:
                # 理想检测器: 注入真实攻击信息
                if isinstance(detector, PerfectDetectorBackend):
                    detector.set_ground_truth(self._attack_type, attack_signal)
                detector.set_control(u_cmd)
                result = detector.detect(y_meas)
                y_rec = result.y_recovered.copy()
                det_class = result.attack_class
                det_conf = result.confidence
                det_attack_est = result.attack_estimate.copy()

            # 4. 跟踪误差
            X_error = WMRKinematics.compute_error(Upsilon_r, y_rec)

            # 5. NMPC 控制
            u_cmd = ctrl.solve(X_error, Ur_seq)
            u_a = WMRKinematics.clamp_control(u_cmd)

            # 7. 机器人运动
            robot.step(u_a)

            # 8. 记录数据
            data['t'][step] = t
            data['true_state'][step] = true_state
            data['y_meas'][step] = y_meas
            data['y_rec'][step] = y_rec
            data['attack_signal'][step] = attack_signal
            data['sensor_noise'][step] = noise
            data['Upsilon_r'][step] = Upsilon_r
            data['u_r'][step] = u_r
            data['Upsilon_hat'][step] = y_rec
            data['X_error'][step] = X_error
            data['u_cmd'][step] = u_cmd
            data['u_a'][step] = u_a
            data['attack_active'][step] = 1.0 if attacker._is_active(t) else 0.0
            data['det_class'][step] = det_class
            data['det_conf'][step] = det_conf
            data['det_attack_est'][step] = det_attack_est

            # 进度报告
            if step % 10 == 0:
                elapsed = time.time() - t_start_wall
                self._q.put(('progress', step, t, elapsed))

        # 完成
        elapsed_total = time.time() - t_start_wall

        # 添加元信息
        data['Ts'] = Ts
        data['attack_type_label'] = self._attack_type
        data['attack_onset'] = self._attack_onset
        data['attack_offset'] = (self._attack_onset + self._attack_duration
                                 if self._attack_duration else self._sim_time + 1.0)
        data['traj_info'] = traj.get_info()
        data['sim_time'] = self._sim_time
        data['seed'] = self._traj_seed
        data['elapsed'] = elapsed_total
        data['use_detector'] = detector is not None
        data['detector_mode'] = detector_mode

        # 计算攻击后 RMSE
        onset_idx = int(round(self._attack_onset / Ts)) if self._attack_onset < self._sim_time else n_steps
        offset_idx = int(round(data['attack_offset'] / Ts)) if data['attack_offset'] < self._sim_time else n_steps
        offset_idx = max(offset_idx, onset_idx + 1)
        pos_err = np.sqrt(data['X_error'][onset_idx:offset_idx, 0]**2 +
                          data['X_error'][onset_idx:offset_idx, 1]**2)
        data['pos_rmse'] = float(np.sqrt(np.mean(pos_err**2))) if len(pos_err) > 0 else 0.0

        self._q.put(('done', data))


# ============================================================================
# ControlPanel — 顶部控制面板
# ============================================================================

class ControlPanel(ttk.Frame):
    """顶部控制面板: 轨迹/攻击选择、运行按钮、进度条、时间滑块"""

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = app
        self._dataset_configs = []
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)
        self.columnconfigure(4, weight=1)
        self.columnconfigure(5, weight=1)

        row = 0

        # ---- 轨迹来源 ----
        ttk.Label(self, text='轨迹来源:', font=('', 10, 'bold')).grid(
            row=row, column=0, sticky='w', padx=5, pady=2)
        self._src_var = tk.StringVar(value='custom')
        ttk.Radiobutton(self, text='数据集记录', variable=self._src_var,
                        value='dataset', command=self._on_source_change).grid(
            row=row, column=1, sticky='w', padx=2)
        ttk.Radiobutton(self, text='自定义参数', variable=self._src_var,
                        value='custom', command=self._on_source_change).grid(
            row=row, column=2, sticky='w', padx=2)
        row += 1

        # ---- 数据集模式面板 ----
        self._frame_dataset = ttk.LabelFrame(self, text='数据集记录选择')
        self._frame_dataset.grid(row=row, column=0, columnspan=6, sticky='ew', padx=5, pady=2)
        self._frame_dataset.columnconfigure(0, weight=1)

        self._ds_var = tk.StringVar()
        self._ds_combo = ttk.Combobox(self._frame_dataset, textvariable=self._ds_var,
                                       state='readonly', width=70)
        self._ds_combo.grid(row=0, column=0, sticky='ew', padx=5, pady=3)
        self._ds_combo.bind('<<ComboboxSelected>>', self._on_dataset_select)

        ttk.Label(self._frame_dataset, text='(从 dataset/metadata.csv 加载)',
                  font=('', 8)).grid(row=1, column=0, sticky='w', padx=5)
        row += 1

        # ---- 自定义参数面板 ----
        self._frame_custom = ttk.LabelFrame(self, text='自定义轨迹参数')
        self._frame_custom.grid(row=row, column=0, columnspan=6, sticky='ew', padx=5, pady=2)
        self._frame_custom.columnconfigure(1, weight=1)

        ttk.Label(self._frame_custom, text='轨迹族:').grid(
            row=0, column=0, padx=5, pady=3, sticky='w')
        self._family_var = tk.StringVar(value='lissajous')
        self._family_combo = ttk.Combobox(
            self._frame_custom, textvariable=self._family_var,
            values=[f'{k} — {FAMILY_NAMES_ZH[k]}' for k in FAMILIES_ORDER],
            state='readonly', width=35)
        self._family_combo.grid(row=0, column=1, sticky='w', padx=5, pady=3)
        self._family_combo.bind('<<ComboboxSelected>>', self._on_family_change)

        ttk.Label(self._frame_custom, text='Seed:').grid(
            row=0, column=2, padx=5, pady=3, sticky='w')
        self._seed_var = tk.IntVar(value=42)
        ttk.Spinbox(self._frame_custom, from_=0, to=99999, textvariable=self._seed_var,
                    width=8).grid(row=0, column=3, padx=5, pady=3, sticky='w')

        # 动态参数区 (根据族切换)
        self._param_frame = ttk.Frame(self._frame_custom)
        self._param_frame.grid(row=1, column=0, columnspan=5, sticky='ew', padx=5, pady=3)
        self._param_widgets = {}  # {param_name: tk.Variable}
        self._build_family_params('lissajous')
        row += 1

        # ---- 攻击设置 ----
        self._frame_attack = ttk.LabelFrame(self, text='攻击设置')
        self._frame_attack.grid(row=row, column=0, columnspan=6, sticky='ew', padx=5, pady=2)

        ttk.Label(self._frame_attack, text='攻击类型:').grid(
            row=0, column=0, padx=5, pady=3, sticky='w')
        self._atk_var = tk.StringVar(value='A1  Constant Bias (恒定偏移)')
        atk_values = [f'{a}  {ATTACK_NAMES[a]}' for a in ALL_ATTACK_TYPES]
        self._atk_combo = ttk.Combobox(self._frame_attack, textvariable=self._atk_var,
                                        values=atk_values, state='readonly', width=40)
        self._atk_combo.grid(row=0, column=1, padx=5, pady=3, sticky='w')

        ttk.Label(self._frame_attack, text='起始时间 [s]:').grid(
            row=0, column=2, padx=5, pady=3, sticky='w')
        self._onset_var = tk.DoubleVar(value=15.0)
        ttk.Spinbox(self._frame_attack, from_=0.0, to=50.0, increment=0.5,
                    textvariable=self._onset_var, width=7).grid(
            row=0, column=3, padx=5, pady=3, sticky='w')

        ttk.Label(self._frame_attack, text='持续 [s]:').grid(
            row=0, column=4, padx=5, pady=3, sticky='w')
        self._dur_var = tk.DoubleVar(value=10.0)
        ttk.Spinbox(self._frame_attack, from_=1.0, to=50.0, increment=1.0,
                    textvariable=self._dur_var, width=7).grid(
            row=0, column=5, padx=5, pady=3, sticky='w')

        # 检测器选择
        ttk.Label(self._frame_attack, text='检测器:').grid(
            row=1, column=0, padx=5, pady=3, sticky='w')
        self._detector_var = tk.StringVar(value='cfm')
        ttk.Radiobutton(self._frame_attack, text='CFM (NN)',
                        variable=self._detector_var, value='cfm').grid(
            row=1, column=1, padx=2, pady=2, sticky='w')
        ttk.Radiobutton(self._frame_attack, text='Perfect (理想)',
                        variable=self._detector_var, value='perfect').grid(
            row=1, column=2, padx=2, pady=2, sticky='w')
        ttk.Radiobutton(self._frame_attack, text='None',
                        variable=self._detector_var, value='none').grid(
            row=1, column=3, padx=2, pady=2, sticky='w')

        row += 1

        # ---- 仿真控制 ----
        self._frame_sim = ttk.Frame(self)
        self._frame_sim.grid(row=row, column=0, columnspan=6, sticky='ew', padx=5, pady=2)

        ttk.Label(self._frame_sim, text='仿真时长 [s]:').pack(side='left', padx=5)
        self._simtime_var = tk.DoubleVar(value=50.0)
        ttk.Spinbox(self._frame_sim, from_=5.0, to=50.0, increment=5.0,
                    textvariable=self._simtime_var, width=6).pack(side='left', padx=2)

        self._btn_run = ttk.Button(self._frame_sim, text='>>>  运行仿真 (F5)  <<<',
                                    command=self.app._on_run, style='Accent.TButton')
        self._btn_run.pack(side='left', padx=20)

        # 播放速度
        ttk.Label(self._frame_sim, text='播放速度:').pack(side='left', padx=(20, 5))
        self._speed_var = tk.DoubleVar(value=5.0)
        ttk.Scale(self._frame_sim, from_=0.5, to=20.0, variable=self._speed_var,
                  orient='horizontal', length=120).pack(side='left', padx=5)
        self._speed_label = ttk.Label(self._frame_sim, text='5×')
        self._speed_label.pack(side='left')
        self._speed_var.trace_add('write', lambda *_: self._speed_label.config(
            text=f'{self._speed_var.get():.1f}×'))

        row += 1

        # ---- 进度条 ----
        self._progress = ttk.Progressbar(self, orient='horizontal', mode='determinate',
                                          maximum=SIM_STEPS_DEFAULT)
        self._progress.grid(row=row, column=0, columnspan=5, sticky='ew', padx=5, pady=2)

        self._progress_label = ttk.Label(self, text='就绪')
        self._progress_label.grid(row=row, column=5, sticky='e', padx=5, pady=2)
        row += 1

        # ---- 时间滑块 ----
        ttk.Label(self, text='时间轴:').grid(row=row, column=0, sticky='w', padx=5)

        self._time_var = tk.IntVar(value=0)
        self._time_slider = ttk.Scale(
            self, from_=0, to=SIM_STEPS_DEFAULT - 1, orient='horizontal',
            variable=self._time_var, command=self.app._on_slider)
        self._time_slider.grid(row=row, column=1, columnspan=4, sticky='ew', padx=5)
        self._time_slider['state'] = 'disabled'

        self._time_label = ttk.Label(self, text='t = 0.00 s')
        self._time_label.grid(row=row, column=5, sticky='e', padx=5)
        row += 1

        # ---- 状态栏 ----
        self._status_var = tk.StringVar(value='就绪 — 请设置参数后按 F5 运行仿真')
        self._status_bar = ttk.Label(self, textvariable=self._status_var,
                                      relief='sunken', anchor='w', font=('', 9))
        self._status_bar.grid(row=row, column=0, columnspan=6, sticky='ew', padx=2, pady=2)

    # ---- 轨迹族参数面板 ----

    def _build_family_params(self, family: str):
        """根据轨迹族动态重建参数输入区"""
        for w in self._param_frame.winfo_children():
            w.destroy()
        self._param_widgets.clear()

        params = {
            'lissajous': [
                ('v_r', 0.15, 0.05, 0.30, 0.01, 'm/s'),
                ('w_freq', 0.14, 0.04, 0.80, 0.01, 'Hz'),
                ('w_amp', 0.34, 0.0, 2.0, 0.01, 'rad/s (auto)'),
            ],
            'circular': [
                ('v_r', 0.15, 0.05, 0.30, 0.01, 'm/s'),
                ('w_r', 0.30, 0.03, 1.76, 0.01, 'rad/s'),
                ('方向', 1, -1, 1, 2, '±1'),
            ],
            'spiral': [
                ('v', 0.20, 0.10, 0.30, 0.01, 'm/s'),
                ('R0', 0.10, 0.03, 0.25, 0.01, 'm (初始半径)'),
                ('Rmax', 1.50, 0.80, 2.30, 0.05, 'm (终点半径)'),
                ('方向', 1, -1, 1, 2, '±1'),
            ],
            'random_waypoint': [
                ('v_r', 0.20, 0.10, 0.30, 0.01, 'm/s'),
                ('w_min', -0.50, -1.00, 0.00, 0.05, 'rad/s'),
                ('w_max', 0.50, 0.00, 1.00, 0.05, 'rad/s'),
            ],
            'square': [
                ('边长', 2.0, 1.5, 3.0, 0.1, 'm'),
                ('v_straight', 0.25, 0.20, 0.30, 0.01, 'm/s'),
                ('w_turn', 1.0, 0.70, 1.60, 0.05, 'rad/s'),
            ],
        }

        if family not in params:
            return

        for i, (name, default, vmin, vmax, step, unit) in enumerate(params[family]):
            col_start = (i % 2) * 3
            row_off = i // 2

            ttk.Label(self._param_frame, text=f'{name}:').grid(
                row=row_off, column=col_start, padx=(15 if i == 0 else 8, 2), pady=3, sticky='w')

            if name == 'w_amp':
                # 自动计算
                var = tk.DoubleVar(value=default)
                lbl = ttk.Label(self._param_frame, text=f'{default:.3f}', width=10,
                                relief='sunken', anchor='center')
                lbl.grid(row=row_off, column=col_start + 1, padx=2, pady=2, sticky='w')
                self._param_widgets[name] = var
                # 绑定 w_freq 变化时自动更新
                if 'w_freq' in self._param_widgets:
                    self._param_widgets['w_freq'].trace_add('write', lambda *_: self._update_w_amp())
            elif name == '方向':
                var = tk.IntVar(value=default)
                ttk.Spinbox(self._param_frame, from_=vmin, to=vmax, increment=step,
                            textvariable=var, width=5).grid(
                    row=row_off, column=col_start + 1, padx=2, pady=2, sticky='w')
                self._param_widgets[name] = var
            else:
                var = tk.DoubleVar(value=default)
                ttk.Spinbox(self._param_frame, from_=vmin, to=vmax, increment=step,
                            textvariable=var, width=8).grid(
                    row=row_off, column=col_start + 1, padx=2, pady=2, sticky='w')
                self._param_widgets[name] = var

            ttk.Label(self._param_frame, text=unit, font=('', 7)).grid(
                row=row_off, column=col_start + 2, padx=1, pady=2, sticky='w')

    def _update_w_amp(self):
        """自动更新 lissajous w_amp = 2.4048 * w_freq"""
        if 'w_freq' in self._param_widgets:
            w_freq = self._param_widgets['w_freq'].get()
            w_amp = 2.4048 * w_freq
            self._param_widgets['w_amp'].set(w_amp)
            # 更新显示标签 (找到对应的 Label)
            for w in self._param_frame.winfo_children():
                if isinstance(w, ttk.Label) and w.cget('relief') == 'sunken':
                    w.config(text=f'{w_amp:.3f}')

    def _on_family_change(self, event=None):
        family = self._family_combo.get().split(' — ')[0]
        self._build_family_params(family)

    def _on_source_change(self):
        mode = self._src_var.get()
        if mode == 'dataset':
            self._frame_dataset.grid()
            self._frame_custom.grid_remove()
            if not self._dataset_configs:
                self._load_dataset()
        else:
            self._frame_dataset.grid_remove()
            self._frame_custom.grid()

    def _load_dataset(self):
        """加载 metadata.csv 并填充下拉列表"""
        metadata_path = os.path.join(PROJECT_ROOT, 'dataset', 'metadata.csv')
        if not os.path.exists(metadata_path):
            messagebox.showwarning('数据集未找到',
                                    f'metadata.csv 不存在:\n{metadata_path}\n\n请先运行 generate_dataset.py')
            return
        try:
            self._dataset_configs = load_dataset_configs(metadata_path)
        except Exception as e:
            messagebox.showerror('加载数据集失败', str(e))
            return

        items = []
        for cfg in self._dataset_configs:
            cid = cfg['config_id']
            fam = cfg['family']
            # 简要参数
            if fam == 'lissajous':
                params_str = f'v_r={cfg["v_r"]:.2f} w_freq={cfg["w_freq"]:.2f}'
            elif fam == 'circular':
                params_str = f'v_r={cfg["v_r"]:.2f} w_r={cfg["w_r"]:.3f}'
            elif fam == 'spiral':
                params_str = f'v={cfg["v"]:.2f} R0={cfg["R0"]:.2f} Rmax={cfg["Rmax"]:.2f}'
            elif fam == 'random_waypoint':
                params_str = f'v_r={cfg["v_r"]:.2f}'
            elif fam == 'square':
                params_str = f'side={cfg["side"]:.1f} v={cfg["v_straight"]:.2f}'
            else:
                params_str = ''
            items.append(f'config_{cid:02d} | {fam:16s} | seed={cfg["traj_seed"]:4d} | {params_str}')
        self._ds_combo['values'] = items
        self._ds_combo.current(0)

    def _on_dataset_select(self, event=None):
        idx = self._ds_combo.current()
        if idx >= 0 and idx < len(self._dataset_configs):
            cfg = self._dataset_configs[idx]
            self._status_var.set(
                f'已选: config_{cfg["config_id"]:02d} | {cfg["family"]} | seed={cfg["traj_seed"]}')

    def get_selected_config(self) -> dict:
        """获取当前选择的轨迹配置 (数据集模式)"""
        idx = self._ds_combo.current()
        if idx >= 0 and idx < len(self._dataset_configs):
            return self._dataset_configs[idx]
        return None

    def get_attack_type(self) -> str:
        return self._atk_var.get().split('  ')[0].strip()

    def get_attack_onset(self) -> float:
        return self._onset_var.get()

    def get_attack_duration(self) -> float:
        return self._dur_var.get()

    def get_sim_time(self) -> float:
        return self._simtime_var.get()

    def get_detector_mode(self) -> str:
        return self._detector_var.get()  # 'cfm' | 'perfect' | 'none'

    def set_progress(self, step: int, total: int, t: float, elapsed: float):
        self._progress['maximum'] = total
        self._progress['value'] = step
        self._progress_label.config(text=f'Step {step}/{total}, t={t:.1f}s, 用时={elapsed:.0f}s')

    def set_ready(self):
        self._progress['value'] = 0
        self._progress_label.config(text='就绪')
        self._btn_run['state'] = 'normal'
        self._time_slider['state'] = 'disabled'

    def set_running(self):
        self._btn_run['state'] = 'disabled'
        self._progress_label.config(text='仿真中...')

    def enable_slider(self, n_steps: int):
        self._time_slider.config(to=n_steps - 1)
        self._time_slider['state'] = 'normal'
        self._time_var.set(0)

    def update_time_label(self, k: int, t: float):
        self._time_label.config(text=f't = {t:.2f} s')
        self._time_var.set(k)


# ============================================================================
# InteractiveApp — 主应用窗口
# ============================================================================

class InteractiveApp(tk.Tk):
    """WMR 传感器攻击仿真交互式浏览器 — 主窗口"""

    def __init__(self):
        super().__init__()

        self.title('WMR Sensor Attack Research — Interactive Explorer')
        self.geometry('1800x1080')
        self.minsize(1280, 800)

        # 状态
        self._sim_data = None       # 仿真结果 dict
        self._worker = None        # SimulationWorker
        self._worker_queue = None  # queue.Queue
        self._n_steps = 1000
        self._playing = False
        self._play_job = None
        self._cursor_lines = []     # 游标线引用
        self._cursor_markers = []   # 游标标记引用

        # ---- 构建 UI ----
        self._build_control_panel()
        self._build_figure()
        self._build_status_bar()

        # 绑定快捷键
        self.bind('<F5>', lambda e: self._on_run())
        self.bind('<Left>', lambda e: self._step_slider(-1))
        self.bind('<Right>', lambda e: self._step_slider(1))
        self.bind('<Control-Left>', lambda e: self._step_slider(-50))
        self.bind('<Control-Right>', lambda e: self._step_slider(50))
        self.bind('<Home>', lambda e: self._jump_to(0))
        self.bind('<End>', lambda e: self._jump_to(self._n_steps - 1))
        self.bind('<space>', lambda e: self._toggle_play())
        self.bind('<Escape>', lambda e: self._stop_play())

        # 窗口关闭时清理
        self.protocol('WM_DELETE_WINDOW', self._on_close)

        # 加载数据集
        self._panel._load_dataset()

    def _build_control_panel(self):
        self._panel = ControlPanel(self, self)
        self._panel.pack(side='top', fill='x', padx=5, pady=5)

    def _build_figure(self):
        """创建 matplotlib Figure 并嵌入 tkinter

        布局: 左侧大图 (2D轨迹, 占全部5行) + 右侧5行时间序列竖列
        每个子图均可单独保存为图片文件。
        """
        self._fig = Figure(figsize=(18, 6.8), dpi=100)
        gs = GridSpec(5, 2, figure=self._fig, hspace=0.78, wspace=0.22,
                      left=0.05, right=0.98, top=0.97, bottom=0.05,
                      width_ratios=[1.5, 1.0])

        # 左侧: 2D 轨迹大图 (占据全部5行)
        self._axes = {
            'trajectory': self._fig.add_subplot(gs[:, 0]),
        }

        # 右侧: 5 个时间序列子图 (竖列)
        self._axes['tracking_error'] = self._fig.add_subplot(gs[0, 1])
        self._axes['control'] = self._fig.add_subplot(gs[1, 1])
        self._axes['innovation'] = self._fig.add_subplot(gs[2, 1])
        self._axes['attack'] = self._fig.add_subplot(gs[3, 1])
        self._axes['measurement'] = self._fig.add_subplot(gs[4, 1])

        # ---- Canvas ----
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(side='top', fill='both', expand=True, padx=5, pady=2)

        # ---- 工具栏 ----
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side='bottom', fill='x', padx=5)
        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._toolbar.update()

        # 保存全景图按钮
        ttk.Button(toolbar_frame, text='💾 保存全景图',
                   command=self._save_current_figure).pack(side='right', padx=10)

        # ---- 单个子图保存按钮 ----
        save_frame = ttk.LabelFrame(self, text='保存单个子图 (独立图片文件)')
        save_frame.pack(side='bottom', fill='x', padx=5, pady=2)

        _PLOT_SAVE_KEYS = [
            ('trajectory',      '2D 轨迹'),
            ('tracking_error',  '跟踪误差'),
            ('control',         '控制指令'),
            ('innovation',      '检测器攻击估计'),
            ('attack',          '攻击信号'),
            ('measurement',     '测量对比'),
        ]
        for key, name in _PLOT_SAVE_KEYS:
            btn = ttk.Button(save_frame, text=f'💾 {name}',
                             command=lambda k=key: self._save_single_plot(k))
            btn.pack(side='left', padx=3, pady=2)

        # ---- 通道选择 (用于 measurement 子图) ----
        self._ch_frame = ttk.Frame(self)
        self._ch_frame.pack(side='bottom', fill='x', padx=5, pady=2)
        ttk.Label(self._ch_frame, text='测量对比通道:').pack(side='left', padx=5)
        self._ch_var = tk.IntVar(value=0)
        for i, name in enumerate(['X', 'Y', 'Theta']):
            ttk.Radiobutton(self._ch_frame, text=name, variable=self._ch_var,
                            value=i, command=self._on_channel_change).pack(side='left', padx=5)

    def _build_status_bar(self):
        self._detail_var = tk.StringVar(value='就绪')
        status = ttk.Label(self, textvariable=self._detail_var,
                           relief='sunken', anchor='w', font=('Consolas', 9))
        status.pack(side='bottom', fill='x', padx=2, pady=1)

    # ==================================================================
    # 事件处理
    # ==================================================================

    def _on_run(self):
        """F5: 运行仿真"""
        if self._worker and self._worker.is_alive():
            return  # 已在运行

        self._stop_play()

        # 收集参数
        if self._panel._src_var.get() == 'dataset':
            cfg = self._panel.get_selected_config()
            if cfg is None:
                messagebox.showwarning('请选择数据集记录', '请先从下拉列表中选择一条数据集记录')
                return
            traj_family = cfg['family']
            traj_seed = cfg['traj_seed']
        else:
            family_key = self._panel._family_combo.get().split(' — ')[0]
            traj_family = family_key
            traj_seed = self._panel._seed_var.get()

        attack_type = self._panel.get_attack_type()
        attack_onset = self._panel.get_attack_onset()
        attack_duration = self._panel.get_attack_duration()
        sim_time = self._panel.get_sim_time()

        if attack_type == 'A0':
            attack_onset = sim_time + 1.0  # Normal: 永远不触发

        # 计算步数
        self._n_steps = int(sim_time / DEFAULT_TS)

        # 清空旧数据
        self._sim_data = None
        self._clear_all_axes()

        # 禁用控件
        self._panel.set_running()
        self._panel._time_slider['state'] = 'disabled'
        self._detail_var.set('仿真运行中...')

        # 启动后台线程
        detector_mode = self._panel.get_detector_mode()
        self._worker_queue = queue.Queue()
        self._worker = SimulationWorker(
            self._worker_queue,
            traj_family=traj_family,
            traj_seed=traj_seed,
            attack_type=attack_type,
            attack_onset=attack_onset,
            attack_duration=attack_duration,
            sim_time=sim_time,
            detector_mode=detector_mode,
        )
        self._worker.start()
        self.after(100, self._poll_worker)

    def _poll_worker(self):
        """轮询 worker 进度"""
        if self._worker_queue is None:
            return
        try:
            while True:
                msg = self._worker_queue.get_nowait()
                msg_type = msg[0]

                if msg_type == 'progress':
                    _, step, t, elapsed = msg
                    self._panel.set_progress(step, self._n_steps, t, elapsed)
                    self._detail_var.set(f'仿真中... Step {step}/{self._n_steps}, t={t:.1f}s, 用时={elapsed:.0f}s')

                elif msg_type == 'done':
                    _, data = msg
                    self._sim_data = data
                    self._n_steps = len(data['t'])
                    self._panel.set_progress(self._n_steps, self._n_steps,
                                              data['sim_time'], data['elapsed'])
                    self._panel._btn_run['state'] = 'normal'
                    self._draw_static_plots()
                    self._panel.enable_slider(self._n_steps)
                    self._update_cursor(0)
                    atk = data.get('attack_type_label', 'A0')
                    det_mode = data.get('detector_mode', 'none')
                    det_label = {'cfm': 'CFM', 'perfect': 'Perfect', 'none': 'None'}.get(det_mode, det_mode)
                    rmse = data.get('pos_rmse', 0)
                    self._detail_var.set(
                        f'仿真完成 | 攻击={atk} | 检测器={det_label} | RMSE={rmse:.4f}m | '
                        f'用时={data["elapsed"]:.1f}s | 拖拽滑块回放')
                    self._panel._status_var.set(
                        f'仿真完成 | {atk} | det={det_label} | pos_rmse={rmse:.4f}m | 总用时={data["elapsed"]:.1f}s')

                elif msg_type == 'warning':
                    _, warning_msg = msg
                    messagebox.showwarning('检测器初始化警告', warning_msg)

                elif msg_type == 'error':
                    _, error_msg = msg
                    messagebox.showerror('仿真错误', error_msg)
                    self._panel.set_ready()
                    self._detail_var.set('仿真失败')
                    self._panel._status_var.set('仿真出错')
                    return  # 停止轮询
        except queue.Empty:
            pass

        # 继续轮询 (如果 worker 仍在运行)
        if self._worker and self._worker.is_alive():
            self._poll_id = self.after(100, self._poll_worker)

    def _on_slider(self, val):
        """时间滑块拖动回调"""
        k = int(float(val))
        self._update_cursor(k)

    def _step_slider(self, delta: int):
        """键盘步进"""
        if self._sim_data is None:
            return
        k = max(0, min(self._n_steps - 1, self._panel._time_var.get() + delta))
        self._panel.update_time_label(k, k * DEFAULT_TS)
        self._update_cursor(k)

    def _jump_to(self, k: int):
        if self._sim_data is None:
            return
        k = max(0, min(self._n_steps - 1, k))
        self._panel.update_time_label(k, k * DEFAULT_TS)
        self._update_cursor(k)

    def _toggle_play(self):
        if self._sim_data is None:
            return
        if self._playing:
            self._stop_play()
        else:
            self._playing = True
            self._detail_var.set('▶ 播放中 (Space 暂停)')
            self._play_step()

    def _play_step(self):
        if not self._playing:
            return
        k = self._panel._time_var.get() + 1
        if k >= self._n_steps:
            self._stop_play()
            return
        self._panel.update_time_label(k, k * DEFAULT_TS)
        self._update_cursor(k)
        speed = self._panel._speed_var.get()
        delay = int(50 / speed)  # 基础 50ms × speed 倍
        self._play_job = self.after(max(10, delay), self._play_step)

    def _stop_play(self):
        self._playing = False
        if self._play_job:
            self.after_cancel(self._play_job)
            self._play_job = None

    def _on_channel_change(self):
        """测量对比通道切换"""
        if self._sim_data is not None:
            self._redraw_measurement_subplot()
            k = self._panel._time_var.get()
            self._update_cursor(k)

    # ==================================================================
    # 保存图片
    # ==================================================================

    def _save_current_figure(self):
        """保存当前全部 6 个面板的全景图"""
        from tkinter import filedialog
        if self._sim_data is None:
            messagebox.showinfo('提示', '请先运行仿真再保存图片')
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG 图片', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg'), ('全部', '*.*')],
            initialdir=PROJECT_ROOT,
            title='保存仿真结果全景图'
        )
        if filepath:
            self._fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self._detail_var.set(f'图片已保存: {os.path.basename(filepath)}')
            print(f'[Save] {filepath}')

    # ==================================================================
    # 单个子图独立保存
    # ==================================================================

    _PLOT_KEYS_ZH = {
        'trajectory': '2D轨迹', 'tracking_error': '跟踪误差',
        'control': '控制指令', 'innovation': '检测器攻击估计',
        'attack': '攻击信号', 'measurement': '测量对比',
    }

    def _save_single_plot(self, plot_key: str):
        """保存单个子图为独立图片文件 (论文级质量)"""
        from tkinter import filedialog
        if self._sim_data is None:
            messagebox.showinfo('提示', '请先运行仿真再保存图片')
            return

        data = self._sim_data
        atk_type = data.get('attack_type_label', 'A0')
        name_zh = self._PLOT_KEYS_ZH.get(plot_key, plot_key)

        # 根据子图类型选择 figsize
        if plot_key == 'trajectory':
            fig, ax = plt.subplots(figsize=(7, 6.5))
        else:
            fig, ax = plt.subplots(figsize=(10, 3.2))

        # 绘制
        draw_func = {
            'trajectory':      self._save_plot_trajectory,
            'tracking_error':  self._save_plot_tracking_error,
            'control':         self._save_plot_control,
            'innovation':      self._save_plot_innovation,
            'attack':          self._save_plot_attack,
            'measurement':     self._save_plot_measurement,
        }.get(plot_key)
        if draw_func:
            draw_func(ax, data)

        # 标题 (含攻击类型)
        atk_name = ATTACK_NAMES.get(atk_type, atk_type)
        fig.suptitle(f'{name_zh}  —  {atk_type} {atk_name}', fontsize=12, fontweight='bold', y=0.98)

        # 保存对话框
        default_name = f'{plot_key}_{atk_type}.png'
        filepath = filedialog.asksaveasfilename(
            defaultextension='.png',
            initialfile=default_name,
            filetypes=[('PNG 图片', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')],
            initialdir=PROJECT_ROOT,
            title=f'保存子图: {name_zh}'
        )
        if filepath:
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self._detail_var.set(f'子图已保存: {os.path.basename(filepath)}')

        plt.close(fig)

    # ---- 独立子图绘制函数 (论文用, 不含游标和交互元素) ----

    @staticmethod
    def _save_plot_trajectory(ax, data):
        """2D 轨迹图 (独立)"""
        ax.plot(data['Upsilon_r'][:, 0], data['Upsilon_r'][:, 1],
                'b--', lw=1.2, alpha=0.7, label='Reference')
        ax.plot(data['true_state'][:, 0], data['true_state'][:, 1],
                'gray', lw=1.0, alpha=0.8, label='Actual')
        ax.scatter(*data['true_state'][0, :2], c='green', marker='o', s=60, zorder=5, label='Start')
        ax.scatter(*data['true_state'][-1, :2], c='red', marker='*', s=80, zorder=5, label='End')
        pb = 2.5
        ax.plot([-pb, pb, pb, -pb, -pb], [-pb, -pb, pb, pb, -pb],
                'gray', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlim(-pb - 0.3, pb + 0.3)
        ax.set_ylim(-pb - 0.3, pb + 0.3)
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title('2D Trajectory', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    @staticmethod
    def _save_plot_tracking_error(ax, data):
        """跟踪误差图 (独立)"""
        t_arr = data['t']
        ax.plot(t_arr, data['X_error'][:, 0], 'r-', lw=1.0, label=r'$x_e$ [m]')
        ax.plot(t_arr, data['X_error'][:, 1], 'g-', lw=1.0, label=r'$y_e$ [m]')
        ax.plot(t_arr, data['X_error'][:, 2], 'b-', lw=1.0, label=r'$\theta_e$ [rad]')
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_xlim(0, data['sim_time'])
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Error')
        ax.set_title('Tracking Error (Body Frame)', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _save_plot_control(ax, data):
        """控制指令图 (独立)"""
        t_arr = data['t']
        ax.plot(t_arr, data['u_cmd'][:, 0], 'b-', lw=1.0, label=r'$v$ [m/s]')
        ax.plot(t_arr, data['u_cmd'][:, 1], 'r-', lw=1.0, label=r'$\omega$ [rad/s]')
        ax.axhline(0.3, color='blue', lw=0.5, ls='--', alpha=0.4)
        ax.axhline(-0.3, color='blue', lw=0.5, ls='--', alpha=0.4)
        ax.axhline(1.76, color='red', lw=0.5, ls='--', alpha=0.4)
        ax.axhline(-1.76, color='red', lw=0.5, ls='--', alpha=0.4)
        ax.set_xlim(0, data['sim_time'])
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Control')
        ax.set_title('NMPC Control Commands', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _save_plot_innovation(ax, data):
        """检测器攻击估计图 (独立)"""
        t_arr = data['t']
        det_a_norm = np.linalg.norm(data['det_attack_est'], axis=1)
        ax.plot(t_arr, det_a_norm, '#d62728', lw=1.2, alpha=0.9, label=r'$\|\hat{a}_{\mathrm{det}}\|$')
        ax.axhline(0.01, color='gray', lw=0.5, ls='--', alpha=0.5)
        ax.set_xlim(0, data['sim_time'])
        ax.set_xlabel('Time [s]'); ax.set_ylabel(r'$\|\hat{a}\|$ [m/rad]')
        ax.set_title('Detected Attack Signal', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _save_plot_attack(ax, data):
        """攻击信号图 (独立)"""
        t_arr = data['t']
        atk_norm = np.linalg.norm(data['attack_signal'], axis=1)
        ax.plot(t_arr, atk_norm, 'k-', lw=1.5, alpha=0.9, label=r'$\|a(k)\|$')
        ax.plot(t_arr, data['attack_signal'][:, 0], 'r--', lw=0.7, alpha=0.5, label=r'$a_x$')
        ax.plot(t_arr, data['attack_signal'][:, 1], 'g--', lw=0.7, alpha=0.5, label=r'$a_y$')
        ax.plot(t_arr, data['attack_signal'][:, 2], 'b--', lw=0.7, alpha=0.5, label=r'$a_\theta$')
        # 检测器估计覆盖
        if data.get('use_detector', False):
            det_a_norm = np.linalg.norm(data['det_attack_est'], axis=1)
            ax.plot(t_arr, det_a_norm, color='cyan', lw=1.2, alpha=0.8, ls='-.',
                    label=r'$\|\hat{a}_{\mathrm{det}}(k)\|$')
        ax.set_xlim(0, data['sim_time'])
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Attack [m/rad]')
        ax.set_title('Attack Signal (Ground Truth + Detector Estimate)', fontweight='bold')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _save_plot_measurement(ax, data):
        """测量对比图 (独立, 默认 X 通道)"""
        t_arr = data['t']
        ax.plot(t_arr, data['true_state'][:, 0], 'k-', lw=1.0, label='True x')
        ax.plot(t_arr, data['y_meas'][:, 0], 'r-', lw=0.8, alpha=0.7, label='Measured x')
        ax.plot(t_arr, data['Upsilon_hat'][:, 0], 'b--', lw=1.0, alpha=0.8, label='Est. x')
        ax.set_xlim(0, data['sim_time'])
        ax.set_xlabel('Time [s]'); ax.set_ylabel('x [m]')
        ax.set_title('Measurement vs True vs Estimate (X channel)', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # ==================================================================
    # 绘图
    # ==================================================================

    def _clear_all_axes(self):
        """清空全部子图"""
        for ax in self._axes.values():
            ax.clear()
        self._cursor_lines.clear()
        self._cursor_markers.clear()
        self._canvas.draw_idle()

    def _draw_static_plots(self):
        """仿真完成后，绘制全部时间序列 (静态部分)"""
        data = self._sim_data
        if data is None:
            return
        t_arr = data['t']
        atk_onset = data['attack_onset']
        atk_offset = data.get('attack_offset', atk_onset)
        atk_type = data.get('attack_type_label', 'A0')

        # 攻击窗颜色
        atk_alpha = 0.12
        atk_face = 'red'

        # ---- (0,0) 2D 轨迹 ----
        ax = self._axes['trajectory']
        ax.clear()
        ax.plot(data['Upsilon_r'][:, 0], data['Upsilon_r'][:, 1],
                'b--', lw=1.0, alpha=0.7, label='Reference')
        # 攻击段着色
        onset_k = int(atk_onset / DEFAULT_TS)
        offset_k = int(atk_offset / DEFAULT_TS)
        if onset_k < len(t_arr) and atk_type != 'A0':
            ax.plot(data['true_state'][onset_k:offset_k, 0],
                    data['true_state'][onset_k:offset_k, 1],
                    color=ATK_COLORS.get(atk_type, 'red'), lw=1.8, alpha=0.8,
                    label=f'Under Attack ({atk_type})')
            # 攻击前
            ax.plot(data['true_state'][:onset_k, 0], data['true_state'][:onset_k, 1],
                    'gray', lw=0.8, alpha=0.6, label='Pre-attack')
            if offset_k < len(t_arr):
                ax.plot(data['true_state'][offset_k:, 0], data['true_state'][offset_k:, 1],
                        'gray', lw=0.8, alpha=0.4)
        else:
            ax.plot(data['true_state'][:, 0], data['true_state'][:, 1],
                    'gray', lw=1.0, alpha=0.7, label='Actual')

        # 起终点
        ax.scatter(*data['true_state'][0, :2], c='green', marker='o', s=60, zorder=5, label='Start')
        ax.scatter(*data['true_state'][-1, :2], c='red', marker='*', s=80, zorder=5, label='End')

        # 边界框
        pb = 2.5
        ax.plot([-pb, pb, pb, -pb, -pb], [-pb, -pb, pb, pb, -pb],
                'gray', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlim(-pb - 0.3, pb + 0.3)
        ax.set_ylim(-pb - 0.3, pb + 0.3)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('2D Trajectory', fontweight='bold')
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # 当前位置标记 (动态)
        pos_marker, = ax.plot([], [], 'o', ms=9, mfc='yellow', mec='black', mew=1.5, zorder=10)
        self._cursor_markers.append(pos_marker)

        # ---- 时间序列子图通用设置 ----
        def setup_time_axis(ax, title, ylabel, legend_loc='best'):
            ax.clear()
            ax.set_xlim(0, data['sim_time'])
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if atk_type != 'A0':
                ax.axvspan(atk_onset, min(atk_offset, data['sim_time']),
                           alpha=atk_alpha, color=atk_face)
            cursor, = ax.plot([0, 0], ax.get_ylim(), 'k--', lw=1.2, alpha=0.7)
            self._cursor_lines.append(cursor)
            # 攻击起始线
            if atk_type != 'A0':
                ax.axvline(atk_onset, color='red', lw=0.8, ls=':', alpha=0.5)
            return ax

        # ---- (0,1) 跟踪误差 ----
        ax = setup_time_axis(self._axes['tracking_error'], 'Tracking Error',
                             'Error [m], [rad]')
        ax.plot(t_arr, data['X_error'][:, 0], 'r-', lw=1.0, label=r'$x_e$')
        ax.plot(t_arr, data['X_error'][:, 1], 'g-', lw=1.0, label=r'$y_e$')
        ax.plot(t_arr, data['X_error'][:, 2], 'b-', lw=1.0, label=r'$\theta_e$')
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.legend(loc='upper right', fontsize=7, ncol=3)

        # ---- (1,0) 控制指令 ----
        ax = setup_time_axis(self._axes['control'], 'Control Commands',
                             'v [m/s], ω [rad/s]')
        ax.plot(t_arr, data['u_cmd'][:, 0], 'b-', lw=1.0, label='v_cmd')
        ax.plot(t_arr, data['u_cmd'][:, 1], 'r-', lw=1.0, label='ω_cmd')
        ax.axhline(0.3, color='blue', lw=0.5, ls='--', alpha=0.4)
        ax.axhline(-0.3, color='blue', lw=0.5, ls='--', alpha=0.4)
        ax.axhline(1.76, color='red', lw=0.5, ls='--', alpha=0.4)
        ax.axhline(-1.76, color='red', lw=0.5, ls='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=7)

        # ---- (1,1) 检测器攻击估计 ----
        ax = setup_time_axis(self._axes['innovation'], 'Detected Attack',
                             '||Attack Est.|| [m]')
        det_a_norm_2 = np.linalg.norm(data['det_attack_est'], axis=1)
        ax.plot(t_arr, det_a_norm_2, '#d62728', lw=1.2, alpha=0.9, label='||â_det||')
        ax.axhline(0.01, color='gray', lw=0.5, ls='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=7)

        # ---- (2,0) 攻击信号 ----
        ax = setup_time_axis(self._axes['attack'], 'Attack Signal',
                             'Attack [m], [rad]')
        atk_norm = np.linalg.norm(data['attack_signal'], axis=1)
        ax.plot(t_arr, atk_norm, 'k-', lw=1.5, alpha=0.9, label='||a(k)||')
        ax.plot(t_arr, data['attack_signal'][:, 0], 'r--', lw=0.7, alpha=0.6, label='a_x')
        ax.plot(t_arr, data['attack_signal'][:, 1], 'g--', lw=0.7, alpha=0.6, label='a_y')
        ax.plot(t_arr, data['attack_signal'][:, 2], 'b--', lw=0.7, alpha=0.6, label='a_θ')
        # 检测器估计覆盖层
        if data.get('use_detector', False):
            det_a_norm = np.linalg.norm(data['det_attack_est'], axis=1)
            ax.plot(t_arr, det_a_norm, color='cyan', lw=1.2, alpha=0.8, ls='-.',
                    label='||â_det(k)||')
        ax.legend(loc='upper right', fontsize=7)

        # ---- (2,1) 测量 vs 真值 vs 估计 ----
        ch = self._ch_var.get()
        ch_names = ['x', 'y', 'θ']
        ax = setup_time_axis(self._axes['measurement'],
                             f'Measurement vs True vs Estimate ({ch_names[ch]})',
                             f'{ch_names[ch]} [m/rad]')
        ax.plot(t_arr, data['true_state'][:, ch], 'k-', lw=1.0, label='True')
        ax.plot(t_arr, data['y_meas'][:, ch], 'r-', lw=0.8, alpha=0.7, label='Measured')
        ax.plot(t_arr, data['Upsilon_hat'][:, ch], 'b--', lw=1.0, alpha=0.8, label='State Est.')
        ax.legend(loc='upper right', fontsize=7)

        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _redraw_measurement_subplot(self):
        """仅重绘 measurement 子图 (通道切换时)"""
        data = self._sim_data
        if data is None:
            return
        t_arr = data['t']
        atk_onset = data['attack_onset']
        atk_offset = data.get('attack_offset', atk_onset)
        atk_type = data.get('attack_type_label', 'A0')
        ch = self._ch_var.get()
        ch_names = ['x', 'y', 'θ']

        ax = self._axes['measurement']
        ax.clear()
        ax.set_xlim(0, data['sim_time'])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'{ch_names[ch]} [m/rad]')
        ax.set_title(f'Measurement vs True vs Estimate ({ch_names[ch]})', fontweight='bold')
        ax.grid(True, alpha=0.3)
        if atk_type != 'A0':
            ax.axvspan(atk_onset, min(atk_offset, data['sim_time']), alpha=0.12, color='red')
            ax.axvline(atk_onset, color='red', lw=0.8, ls=':', alpha=0.5)

        ax.plot(t_arr, data['true_state'][:, ch], 'k-', lw=1.0, label='True')
        ax.plot(t_arr, data['y_meas'][:, ch], 'r-', lw=0.8, alpha=0.7, label='Measured')
        ax.plot(t_arr, data['Upsilon_hat'][:, ch], 'b--', lw=1.0, alpha=0.8, label='State Est.')
        ax.legend(loc='upper right', fontsize=7)

        # 重建游标线
        if hasattr(self, '_cursor_lines') and len(self._cursor_lines) >= 5:
            old = self._cursor_lines.pop()  # 移除旧的
            old.remove() if old else None
        cursor, = ax.plot([0, 0], ax.get_ylim(), 'k--', lw=1.2, alpha=0.7)
        self._cursor_lines.append(cursor)

        self._canvas.draw_idle()

    def _update_cursor(self, k: int):
        """更新所有游标线到 step k"""
        data = self._sim_data
        if data is None:
            return
        t_k = data['t'][k]
        true_state_k = data['true_state'][k]
        n_curves = 5  # 5 个时间序列子图

        # 更新竖直游标线
        for i, line in enumerate(self._cursor_lines):
            if line is not None and i < n_curves:
                line.set_xdata([t_k, t_k])
                # 调整 y 范围
                ax = line.axes
                ylim = ax.get_ylim()
                line.set_ydata(ylim)

        # 更新 2D 位置标记
        if self._cursor_markers:
            self._cursor_markers[0].set_data([true_state_k[0]], [true_state_k[1]])

        # 更新状态栏
        atk_active = bool(data['attack_active'][k]) if k < len(data['attack_active']) else False
        atk_status = '[攻击中]' if atk_active else '[正常]'
        x_err = data['X_error'][k]
        pos_err = np.sqrt(x_err[0]**2 + x_err[1]**2)
        atk_norm = np.linalg.norm(data['attack_signal'][k])

        self._detail_var.set(
            f'k={k:4d} | t={t_k:6.2f}s | '
            f'x={true_state_k[0]:+.3f}m y={true_state_k[1]:+.3f}m θ={true_state_k[2]:+.3f}rad | '
            f'|e_xy|={pos_err:.3f}m | {atk_status} | ||a||={atk_norm:.3f}'
            f' | Det:{data["det_class"][k]}({data["det_conf"][k]:.2f})')

        # 更新滑块标签
        self._panel.update_time_label(k, t_k)

        self._canvas.draw_idle()

    # ==================================================================
    # 生命周期
    # ==================================================================

    def _on_close(self):
        self._stop_play()
        self.destroy()


# ============================================================================
# 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='WMR 传感器攻击仿真交互式浏览器')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    args = parser.parse_args()

    app = InteractiveApp()
    if args.debug:
        print('[DEBUG] 交互式应用已启动')
    app.mainloop()


if __name__ == "__main__":
    main()
