"""
web_app.py — WMR 传感器攻击仿真 Web 浏览器
==============================================================================
基于 Flask + Plotly.js 的响应式 Web 应用，替代 tkinter 桌面版。

功能:
  - 数据集记录浏览: 选择即显示全部 6 面板图表
  - 自定义仿真: 选择轨迹族/攻击参数, 后台运行 NMPC 仿真
  - 时间滑块回放: 拖拽或键盘控制, 6 面板同步更新
  - 自适应布局: CSS Grid 响应式, 适配笔记本和外接显示器

用法:
  python app/web_app.py
  浏览器打开 http://localhost:5000

依赖: Flask (已在 conda 环境), Plotly.js (CDN 加载, 无需安装)
"""

import os
import sys
import json
import uuid
import time
import threading
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ============================================================================
# Flask 应用
# ============================================================================

app = Flask(__name__)

DEFAULT_TS = 0.05
SIM_STEPS_DEFAULT = 1000

# 仿真任务存储 {task_id: {status, progress, data, elapsed}}
_sim_tasks = {}

# ============================================================================
# 数据集工具
# ============================================================================

FAMILIES_ORDER = ['lissajous', 'circular', 'spiral', 'random_waypoint', 'square']
FAMILY_NAMES_ZH = {
    'lissajous': '李萨如', 'circular': '圆形', 'spiral': '螺旋线',
    'random_waypoint': '随机路径点', 'square': '正方形',
}


def _list_dataset_dirs():
    base = os.path.join(PROJECT_ROOT, 'dataset')
    if not os.path.exists(base):
        return []
    dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    dirs.sort(key=lambda d: os.path.getctime(os.path.join(base, d)), reverse=True)
    return [os.path.join(base, d) for d in dirs]


def _load_metadata(ds_dir):
    metadata_path = os.path.join(ds_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        return []
    df = pd.read_csv(metadata_path)
    records = []
    for _, row in df.iterrows():
        records.append({
            'config_id': int(row['config_id']),
            'family': str(row['trajectory_family']),
            'traj_seed': int(row['traj_seed']),
            'attack_type': str(row['attack_type']),
            'attack_onset': float(row['attack_onset']),
            'attack_duration': float(row.get('attack_duration', 0.0)),
            'filename': str(row['filename']),
        })
    return records


# ============================================================================
# 路由
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html',
                           families=FAMILIES_ORDER,
                           family_names=FAMILY_NAMES_ZH)


@app.route('/api/dataset_dirs')
def dataset_dirs():
    return jsonify(_list_dataset_dirs())


@app.route('/api/dataset_records')
def dataset_records():
    ds_dir = request.args.get('dir', '')
    return jsonify(_load_metadata(ds_dir))


@app.route('/api/load_record')
def load_record():
    """加载单个 npz 文件, 返回所有数组为 JSON"""
    ds_dir = request.args.get('dir', '')
    filename = request.args.get('file', '')

    npz_path = os.path.join(ds_dir, filename)
    if not os.path.exists(npz_path):
        return jsonify({'error': f'文件不存在: {npz_path}'}), 404

    try:
        raw = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    result = {}
    for key in raw.keys():
        arr = raw[key]
        result[key] = arr.tolist() if hasattr(arr, 'tolist') else arr

    return jsonify(result)


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """启动后台仿真, 返回 task_id 用于轮询"""
    params = request.get_json() or {}
    task_id = uuid.uuid4().hex[:8]

    _sim_tasks[task_id] = {'status': 'running', 'progress': 0.0, 'data': None, 'elapsed': 0.0}

    t = threading.Thread(target=_run_simulation, args=(task_id, params), daemon=True)
    t.start()

    return jsonify({'task_id': task_id})


@app.route('/api/simulate_status/<task_id>')
def simulate_status(task_id):
    task = _sim_tasks.get(task_id)
    if task is None:
        return jsonify({'error': 'Unknown task_id'}), 404
    return jsonify(task)


# ============================================================================
# 仿真引擎 (独立, 不依赖 tkinter/matplotlib)
# ============================================================================

def _run_simulation(task_id, params):
    try:
        from model import WMRParams, WMRKinematics, SensorSimulator, RandomizedTrajectory
        from controller import NMPCController, NMPCParams
        from attack import SensorAttack, AttackConfig
        from backend import DetectorBackend

        traj_family = params.get('family', 'lissajous')
        traj_seed = int(params.get('seed', 42))
        attack_type = params.get('attack_type', 'A1')
        attack_onset = float(params.get('attack_onset', 15.0))
        attack_duration = float(params.get('attack_duration', 10.0))
        sim_time = float(params.get('sim_time', 50.0))
        detector_mode = params.get('detector_mode', 'nn')

        # A0: 永远不触发攻击
        if attack_type == 'A0':
            attack_onset = sim_time + 1.0
            attack_duration = 0.0

        Ts = DEFAULT_TS
        n_steps = int(sim_time / Ts)

        # 轨迹
        traj = RandomizedTrajectory(seed=traj_seed, family=traj_family)

        # 初始化
        wmr_params = WMRParams()
        robot = WMRKinematics(wmr_params)
        sensor = SensorSimulator()
        ctrl = NMPCController(NMPCParams())
        ctrl.load_or_build()

        atk_cfg = AttackConfig(attack_duration=attack_duration if attack_type != 'A0' else 0.0)
        attacker = SensorAttack(attack_type=attack_type, onset_time=attack_onset,
                                config=atk_cfg, seed=traj_seed)

        traj.reset()
        perturb_rng = np.random.RandomState(traj_seed)
        init_state = np.array([
            traj._x_r + perturb_rng.uniform(0.0, 0.15) * perturb_rng.choice([-1, 1]),
            traj._y_r + perturb_rng.uniform(0.0, 0.15) * perturb_rng.choice([-1, 1]),
            traj._theta_r + perturb_rng.uniform(0.0, 0.2) * perturb_rng.choice([-1, 1]),
        ])
        robot.reset(init_state)
        ctrl.reset()
        attacker.reset()
        np.random.seed(traj_seed)

        # 检测器
        detector = None
        if detector_mode == 'nn':
            model_dir = os.path.join(PROJECT_ROOT, 'detector', 'models')
            nn_model = os.path.join(model_dir, 'nn_cls_best.pt')
            if os.path.exists(nn_model):
                win_base = os.path.join(PROJECT_ROOT, 'dataset_win')
                win_ts = None
                if os.path.exists(win_base):
                    ds = sorted([d for d in os.listdir(win_base)
                                if os.path.isdir(os.path.join(win_base, d))],
                               key=lambda d: os.path.getctime(os.path.join(win_base, d)),
                               reverse=True)
                    if ds:
                        win_ts = ds[0]
                norm_path = os.path.join(win_base, win_ts, 'normalizer.npz') if win_ts else os.path.join(win_base, 'normalizer.npz')
                if os.path.exists(norm_path):
                    detector = DetectorBackend(model_path=nn_model, norm_path=norm_path)
                    detector.reset()

        # 预分配
        ts = np.zeros(n_steps, dtype=np.float32)
        true_state = np.zeros((n_steps, 3), dtype=np.float32)
        y_meas = np.zeros((n_steps, 3), dtype=np.float32)
        y_rec = np.zeros((n_steps, 3), dtype=np.float32)
        attack_signal = np.zeros((n_steps, 3), dtype=np.float32)
        Upsilon_r = np.zeros((n_steps, 3), dtype=np.float32)
        X_error = np.zeros((n_steps, 3), dtype=np.float32)
        u_cmd = np.zeros((n_steps, 2), dtype=np.float32)
        attack_active = np.zeros(n_steps, dtype=np.float32)
        det_attack_est = np.zeros((n_steps, 3), dtype=np.float32)
        det_class = np.full(n_steps, 'A0', dtype=object)

        u = np.zeros(2)
        t_wall = time.time()

        for step in range(n_steps):
            t = step * Ts

            Ur, ur = traj.step(t)
            Ur_seq = traj.generate_sequence(t, NMPCParams().N)

            xs = robot.state.copy()
            noise = sensor.noise_std * np.random.randn(3)
            y_clean = xs + noise
            ym = attacker.inject(t, y_clean)
            a_sig = ym - y_clean

            if detector is None:
                yr = ym.copy()
                dc = 'A0'
                dconf = 0.0
                aest = np.zeros(3)
            else:
                detector.set_control(u)
                res = detector.detect(ym)
                yr = res.y_recovered.copy()
                dc = res.attack_class
                dconf = res.confidence
                aest = res.attack_estimate.copy()

            Xe = WMRKinematics.compute_error(Ur, yr)
            u = ctrl.solve(Xe, Ur_seq)
            ua = WMRKinematics.clamp_control(u)
            robot.step(ua)

            ts[step] = t
            true_state[step] = xs
            y_meas[step] = ym
            y_rec[step] = yr
            attack_signal[step] = a_sig
            Upsilon_r[step] = Ur
            X_error[step] = Xe
            u_cmd[step] = u
            attack_active[step] = 1.0 if attacker._is_active(t) else 0.0
            det_attack_est[step] = aest
            det_class[step] = dc

            if step % 50 == 0:
                _sim_tasks[task_id]['progress'] = step / n_steps
                _sim_tasks[task_id]['elapsed'] = time.time() - t_wall

        elapsed_total = time.time() - t_wall

        result = {
            't': ts.tolist(),
            'true_state': true_state.tolist(),
            'y_meas': y_meas.tolist(),
            'y_rec': y_rec.tolist(),
            'attack_signal': attack_signal.tolist(),
            'Upsilon_r': Upsilon_r.tolist(),
            'X_error': X_error.tolist(),
            'u_cmd': u_cmd.tolist(),
            'attack_active': attack_active.tolist(),
            'det_attack_est': det_attack_est.tolist(),
            'det_class': det_class.tolist(),
        }
        result['_meta'] = {
            'elapsed': round(elapsed_total, 1),
            'attack_type': attack_type,
            'attack_onset': attack_onset,
            'attack_offset': attack_onset + attack_duration if attack_type != 'A0' else sim_time + 1,
            'attack_duration': attack_duration,
            'sim_time': sim_time,
            'detector_mode': detector_mode,
            'use_detector': detector is not None,
            'Ts': Ts,
            'n_steps': n_steps,
            'family': traj_family,
            'seed': traj_seed,
        }

        _sim_tasks[task_id] = {'status': 'done', 'progress': 1.0, 'data': result, 'elapsed': elapsed_total}

    except Exception as e:
        _sim_tasks[task_id] = {
            'status': 'error',
            'progress': 0.0,
            'data': None,
            'elapsed': 0.0,
            'error': str(e) + '\n' + traceback.format_exc(),
        }


# ============================================================================
# 清理过期任务 (定期)
# ============================================================================

@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """清理超过 10 分钟的旧任务"""
    now = time.time()
    stale = [tid for tid, t in _sim_tasks.items()
             if t['status'] in ('done', 'error') and now - t.get('_created', now) > 600]
    for tid in stale:
        del _sim_tasks[tid]
    return jsonify({'cleaned': len(stale)})


# ============================================================================
# 入口
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='WMR Web Explorer')
    parser.add_argument('--port', type=int, default=5000, help='HTTP 端口 (默认 5000)')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    print('=' * 60)
    print('  WMR Sensor Attack — Web Explorer')
    print(f'  打开浏览器: http://localhost:{args.port}')
    print('=' * 60)

    app.run(debug=args.debug, host='127.0.0.1', port=args.port)
