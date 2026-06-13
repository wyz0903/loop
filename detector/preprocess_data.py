"""
preprocess_data.py — 密集滑动窗口 + 物理锚点归一化 + .npy 缓存
==============================================================================
从 .npz 仿真文件提取滑动窗口，预生成为 .npy 数组供训练直接加载。

关键设计:
  - 窗口大小: 100 步 (5 秒 @ Ts=0.05)
  - 步长 stride=1: 密集采样最大化样本量
  - 物理锚点归一化: 创新通道用固定物理尺度 σ (非数据依赖的 IQR), u_cmd 用物理上限
  - 输入通道: internal_innovation(3) + u_cmd(2) = 5 通道
  - 抗泄漏: 同一 .npz 文件的所有窗口始终整体进入同一划分
  - IID 分层划分: 按轨迹族分层抽样, 确保 train/val/test 同分布

输出文件 (保存在 dataset_win/ 目录):
  X_train.npy, X_val.npy, X_test.npy          — 输入窗口 (N, 100, 5) float32
  Y_train_cls.npy, Y_val_cls.npy, Y_test_cls.npy  — 攻击分类标签 (N,) int64
  Y_train_atk.npy, Y_val_atk.npy, Y_test_atk.npy  — 攻击信号窗口 (N, 100, 3) float32
  split_info.npz                                — 划分信息 (train/val/test 文件列表)
  normalizer.npz                                — 归一化参数

用法:
  python preprocess_data.py
  python preprocess_data.py --train-ratio 0.7 --val-ratio 0.15
  python preprocess_data.py --window 150 --stride 2
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'dataset_win')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
ATTACK_NAMES = {
    'A0': 'Normal', 'A1': 'ConstantBias', 'A2': 'Sinusoidal',
    'A3': 'Drift', 'A4': 'Step', 'A5': 'ReplayAttack',
    'A6': 'Dropout', 'A7': 'Scaling', 'A8': 'Freeze',
}

# 输入通道: 传感器测量 + 内部运动学新息 + 控制上下文
INPUT_CHANNELS = ['y_meas', 'internal_innovation', 'u_cmd']   # y_meas(3) + innov(3) + u_cmd(2) = 8 通道
from model import SIM_STEPS, SIM_TIME
N_STEPS = SIM_STEPS       # 1000，避免 IEEE 754 截断误差
WINDOW_SIZE = 100
STRIDE = 1
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15    # test = 1 - train - val = 0.17
SPLIT_SEED = 42     # 分层抽样固定种子, 保证可复现


# 物理上限 (TurtleBot4 安全模式)
PHYSICAL_MAX = np.array([0.3, 1.76], dtype=np.float32)  # v_max, omega_max
# y_meas 通道物理锚点尺度: 工作空间边界
Y_MEAS_SCALE = np.array([2.5, 2.5, np.pi], dtype=np.float32)  # [x_max, y_max, pi]


class RobustNormalizer:
    """物理锚点归一化器 (Physical-Anchor Normalization)

    通道布局 (8 通道):
      - y_meas (前 3 通道):  (x - median_y) / Y_MEAS_SCALE  -- 工作空间物理锚点
      - innovation (中间 3 通道): (x - median_innov) / INNOV_SCALE  -- 运动学异常锚点
      - u_cmd (最后 2 通道):    x / cmd_max                  -- 控制物理上限

    设计原理:
      单步运动学预测误差 (innovation) 在正常运行时 ~ 0 (Ts=0.05s, 无传感器噪声)。
      若用 IQR 归一化 `(x-median)/IQR`, IQR 可低至 1e-6 量级, 除以极小值
      将常规攻击信号 (0.1-0.5 m/rad) 放大到 10^3-10^5, 导致 NN 训练梯度爆炸。

      改用固定物理锚点尺度 sigma 作为分母:
        - y_meas 尺度: [2.5m, 2.5m, π rad]  -- 工作空间边界
        - sigma_xy = 0.5 m   -- 单步位置误差 > 0.5m 在任何工况下均不可能正常发生
        - sigma_th = 0.3 rad -- 单步航向误差 > 0.3rad (17deg) 明确异常

      优势: 数值天然稳定, 物理含义清晰, 论文可辩护。
      所有通道均除以该通道的物理特征尺度。

    所有统计量从训练集计算，验证集/测试集复用，保证无信息泄漏。
    """

    # 创新通道物理锚点尺度 (sigma): 代表"明确异常"的单步误差阈值
    INNOV_SCALE = np.array([0.5, 0.5, 0.3], dtype=np.float32)  # [x, y, theta]

    def __init__(self):
        self.ymeas_median = None    # (3,) y_meas 通道中位数 (去中心化)
        self.ymeas_scale = Y_MEAS_SCALE.copy()  # (3,) y_meas 通道锚点尺度
        self.feat_median = None     # (3,) 创新通道中位数 (去中心化)
        self.innov_scale = None     # (3,) 创新通道锚点尺度 (= INNOV_SCALE)
        self.cmd_max = PHYSICAL_MAX.copy()  # (2,) 控制指令物理上限

    def fit(self, X: np.ndarray):
        """从训练集计算归一化参数

        Args:
            X: (N, W, 8) 训练窗口, 布局 [y_meas(3)+innov(3)+u_cmd(2)]
        """
        # y_meas 通道 (0:3)
        ymeas_data = X[:, :, 0:3].reshape(-1, 3)
        self.ymeas_median = np.median(ymeas_data, axis=0).astype(np.float32)
        # 创新通道 (3:6)
        innov_data = X[:, :, 3:6].reshape(-1, 3)
        self.feat_median = np.median(innov_data, axis=0).astype(np.float32)
        # 物理锚点: 固定 sigma, 不依赖数据分布 (避免 IQR 对尖峰分布的放大效应)
        self.innov_scale = self.INNOV_SCALE.copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """应用归一化"""
        X_norm = np.zeros_like(X, dtype=np.float32)
        # y_meas 通道 (0:3): (x - median_y) / y_meas_scale
        X_norm[:, :, 0:3] = (X[:, :, 0:3] - self.ymeas_median) / self.ymeas_scale
        # 创新通道 (3:6): (x - median_innov) / innov_scale
        X_norm[:, :, 3:6] = (X[:, :, 3:6] - self.feat_median) / self.innov_scale
        # u_cmd 通道 (6:8): x / cmd_max
        X_norm[:, :, 6:8] = X[:, :, 6:8] / self.cmd_max
        return X_norm

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def unnormalize_y_meas(self, Y_norm: np.ndarray) -> np.ndarray:
        """将归一化的 y_clean_hat 还原为物理单位

        Args:
            Y_norm: (..., 3) 归一化干净信号
        Returns:
            (..., 3) 物理单位
        """
        return Y_norm * self.ymeas_scale + self.ymeas_median

    def save(self, filepath: str):
        np.savez(filepath,
                 ymeas_median=self.ymeas_median, ymeas_scale=self.ymeas_scale,
                 feat_median=self.feat_median, innov_scale=self.innov_scale,
                 cmd_max=self.cmd_max)

    @classmethod
    def load(cls, filepath: str) -> 'RobustNormalizer':
        data = np.load(filepath)
        obj = cls()
        # 向后兼容: 旧格式无 ymeas_median/ymeas_scale
        if 'ymeas_median' in data:
            obj.ymeas_median = data['ymeas_median']
            obj.ymeas_scale = data['ymeas_scale']
        else:
            obj.ymeas_median = np.zeros(3, dtype=np.float32)
            obj.ymeas_scale = Y_MEAS_SCALE.copy()
        obj.feat_median = data['feat_median']
        # 向后兼容旧格式 'feat_iqr'
        if 'innov_scale' in data:
            obj.innov_scale = data['innov_scale']
        elif 'feat_iqr' in data:
            obj.innov_scale = data['feat_iqr']
        else:
            obj.innov_scale = cls.INNOV_SCALE.copy()
        obj.cmd_max = data['cmd_max']
        return obj


def load_npz_file(filepath: str) -> dict:
    """加载 .npz 仿真文件"""
    return dict(np.load(filepath, allow_pickle=True))


def build_windows(data: dict, window_size: int, stride: int) -> tuple:
    """从单个仿真数据中提取滑动窗口

    Args:
        data:      .npz 加载的字典，含各信号的时间序列
        window_size: 窗口大小
        stride:      步长

    Returns:
        X_windows:       (N, W, 8)  模型输入特征窗口 [y_meas(3)+innov(3)+u_cmd(2)]
        y_clean_windows: (N, W, 3)  干净传感器信号窗口 (流匹配目标)
        y_meas_windows:  (N, W, 3)  原始测量窗口 (物理单位, 用于物理损失)
    """
    # 拼接模型输入通道 (y_meas + innovation + u_cmd)
    ch_arrays = []
    for ch_name in INPUT_CHANNELS:
        arr = data[ch_name]
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        ch_arrays.append(arr)
    X_all = np.concatenate(ch_arrays, axis=1).astype(np.float32)  # (T, 8)

    # theta 创新包裹到 [-pi, pi] (防御性: 确保角度差物理含义)
    # 创新通道在索引 5 (y_meas 3ch + innov_theta 是第 6 个通道 = 索引 5)
    X_all[:, 5] = np.arctan2(np.sin(X_all[:, 5]), np.cos(X_all[:, 5]))

    # 构建 y_clean 目标: 从 .npz 读取 (新数据) 或从 y_meas - attack_signal 回算 (旧数据)
    if 'y_clean' in data:
        y_clean_all = data['y_clean'].astype(np.float32)           # (T, 3)
    else:
        # 向后兼容: 从现有数据计算
        y_meas_all = data['y_meas'].astype(np.float32)
        atk_all = data['attack_signal'].astype(np.float32)
        y_clean_all = y_meas_all - atk_all

    y_meas_all = data['y_meas'].astype(np.float32)               # (T, 3) 物理单位

    # 滑动窗口
    n_windows = (N_STEPS - window_size) // stride + 1
    X_windows = np.zeros((n_windows, window_size, X_all.shape[1]), dtype=np.float32)
    y_clean_windows = np.zeros((n_windows, window_size, y_clean_all.shape[1]), dtype=np.float32)
    y_meas_windows = np.zeros((n_windows, window_size, y_meas_all.shape[1]), dtype=np.float32)

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        X_windows[i] = X_all[start:end]
        y_clean_windows[i] = y_clean_all[start:end]
        y_meas_windows[i] = y_meas_all[start:end]

    return X_windows, y_clean_windows, y_meas_windows


def main():
    parser = argparse.ArgumentParser(
        description='数据集预处理：密集滑动窗口 + 归一化 + .npy 缓存')
    parser.add_argument('--window', type=int, default=WINDOW_SIZE,
                        help=f'窗口大小 (默认 {WINDOW_SIZE})')
    parser.add_argument('--stride', type=int, default=STRIDE,
                        help=f'步长 (默认 {STRIDE})')
    parser.add_argument('--input-dir', type=str, default=DATASET_DIR)
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--train-ratio', type=float, default=TRAIN_RATIO,
                        help=f'训练集比例 (默认 {TRAIN_RATIO})')
    parser.add_argument('--val-ratio', type=float, default=VAL_RATIO,
                        help=f'验证集比例 (默认 {VAL_RATIO}), test = 1 - train - val')
    parser.add_argument('--split-seed', type=int, default=SPLIT_SEED,
                        help=f'分层抽样随机种子 (默认 {SPLIT_SEED})')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 metadata
    metadata_path = os.path.join(args.input_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"[ERROR] metadata.csv 未找到: {metadata_path}")
        sys.exit(1)
    df = pd.read_csv(metadata_path)

    # ---- 分层 IID 划分: 按轨迹族分层抽样 (同一 config 整体进入同一划分, 防泄漏) ----
    rng = np.random.RandomState(args.split_seed)

    # 构建 config_id -> trajectory_family 映射
    config_family = {}
    for _, row in df.iterrows():
        cid = int(row['config_id'])
        fam = str(row.get('trajectory_family', 'unknown'))
        config_family[cid] = fam

    # 按族分组 config_id, 打乱, 切分
    families = defaultdict(list)
    for cid, fam in config_family.items():
        families[fam].append(cid)
    for fam in families:
        families[fam] = sorted(set(families[fam]))

    train_configs, val_configs, test_configs = set(), set(), set()
    for fam, cids in sorted(families.items()):
        cids_sorted = sorted(cids)
        rng.shuffle(cids_sorted)
        n = len(cids_sorted)
        n_train = max(1, int(np.round(n * args.train_ratio)))
        n_val = max(1, int(np.round(n * args.val_ratio)))
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
        train_configs.update(cids_sorted[:n_train])
        val_configs.update(cids_sorted[n_train:n_train + n_val])
        test_configs.update(cids_sorted[n_train + n_val:])

    # 按 config_id 反查文件名
    train_files, val_files, test_files = set(), set(), set()
    for _, row in df.iterrows():
        cid = int(row['config_id'])
        fname = row['filename']
        if cid in train_configs:
            train_files.add(fname)
        elif cid in val_configs:
            val_files.add(fname)
        else:
            test_files.add(fname)

    # 打印划分统计
    train_fams = set(config_family[c] for c in train_configs)
    val_fams = set(config_family[c] for c in val_configs)
    test_fams = set(config_family[c] for c in test_configs)
    print(f"分层 IID 划分 (train/val/test = {args.train_ratio:.0%}/{args.val_ratio:.0%}/"
          f"{1-args.train_ratio-args.val_ratio:.0%}, seed={args.split_seed})")
    print(f"  Train configs: {len(train_configs)} ({sorted(train_configs)[:5]}...)  "
          f"families: {sorted(train_fams)}")
    print(f"  Val   configs: {len(val_configs)} ({sorted(val_configs)})  "
          f"families: {sorted(val_fams)}")
    print(f"  Test  configs: {len(test_configs)} ({sorted(test_configs)})  "
          f"families: {sorted(test_fams)}")
    for fam in sorted(families.keys()):
        t_c = sorted([c for c in families[fam] if c in train_configs])
        v_c = sorted([c for c in families[fam] if c in val_configs])
        te_c = sorted([c for c in families[fam] if c in test_configs])
        print(f"    {fam}: train={t_c}, val={v_c}, test={te_c}")

    # 保存划分信息
    np.savez(os.path.join(args.output_dir, 'split_info.npz'),
             train_ratio=np.array(args.train_ratio),
             val_ratio=np.array(args.val_ratio),
             split_seed=np.array(args.split_seed),
             train_files=np.array(sorted(train_files)),
             val_files=np.array(sorted(val_files)),
             test_files=np.array(sorted(test_files)))

    # 收集数据 (按文件整体分配，不拆分窗口)
    train_X, val_X, test_X = [], [], []
    train_clean, val_clean, test_clean = [], [], []
    train_cls, val_cls, test_cls = [], [], []
    train_ymeas, val_ymeas, test_ymeas = [], [], []

    n_per_file = (N_STEPS - args.window) // args.stride + 1
    A0_LABEL = ALL_ATTACK_TYPES.index('A0')

    print(f"窗口大小: {args.window}, 步长: {args.stride}")
    print(f"每文件窗口数: {n_per_file}")
    print(f"总文件数: {len(df)} (train: {len(train_files)}, val: {len(val_files)}, "
          f"test: {len(test_files)})")

    for idx, row in df.iterrows():
        fname = row['filename']
        filepath = os.path.join(args.input_dir, fname)
        atk_type = row['attack_type']
        attack_onset = float(row.get('attack_onset', 15.0))
        in_train = fname in train_files
        in_val = fname in val_files

        data = load_npz_file(filepath)
        X_w, y_clean_w, y_meas_w = build_windows(data, args.window, args.stride)

        # 逐窗口标注（考虑攻击结束时间）
        is_normal = (atk_type == 'A0')
        n_w = len(X_w)
        cls_per_window = np.full(n_w, A0_LABEL, dtype=np.int64)

        if not is_normal:
            atk_label = ALL_ATTACK_TYPES.index(atk_type)
            if 'attack_onset_step' in row.index and not pd.isna(row['attack_onset_step']):
                onset_step = int(row['attack_onset_step'])
            else:
                onset_step = int(round(attack_onset / 0.05))
            attack_offset = float(row.get('attack_offset', SIM_TIME + 1.0))
            if 'attack_offset_step' in row.index and not pd.isna(row['attack_offset_step']):
                offset_step = int(row['attack_offset_step'])
                if attack_offset >= SIM_TIME:
                    offset_step = N_STEPS + 1
            else:
                offset_step = int(round(attack_offset / 0.05)) if attack_offset < SIM_TIME else N_STEPS + 1
            for w in range(n_w):
                w_end = w * args.stride + args.window
                w_start = w * args.stride
                if w_end > onset_step and w_start < offset_step:
                    cls_per_window[w] = atk_label

        if in_train:
            train_X.append(X_w)
            train_clean.append(y_clean_w)
            train_cls.append(cls_per_window)
            train_ymeas.append(y_meas_w)
        elif in_val:
            val_X.append(X_w)
            val_clean.append(y_clean_w)
            val_cls.append(cls_per_window)
            val_ymeas.append(y_meas_w)
        else:
            test_X.append(X_w)
            test_clean.append(y_clean_w)
            test_cls.append(cls_per_window)
            test_ymeas.append(y_meas_w)

        if (idx + 1) % 20 == 0:
            print(f"  处理进度: {idx+1}/{len(df)}")

    # 合并
    print("合并数据...")
    X_train_all = np.concatenate(train_X, axis=0).astype(np.float32)
    X_val_all = np.concatenate(val_X, axis=0).astype(np.float32)
    X_test_all = np.concatenate(test_X, axis=0).astype(np.float32)
    clean_train_all = np.concatenate(train_clean, axis=0).astype(np.float32)
    clean_val_all = np.concatenate(val_clean, axis=0).astype(np.float32)
    clean_test_all = np.concatenate(test_clean, axis=0).astype(np.float32)
    cls_train_all = np.concatenate(train_cls, axis=0)
    cls_val_all = np.concatenate(val_cls, axis=0)
    cls_test_all = np.concatenate(test_cls, axis=0)
    y_meas_train_all = np.concatenate(train_ymeas, axis=0).astype(np.float32)
    y_meas_val_all = np.concatenate(val_ymeas, axis=0).astype(np.float32)
    y_meas_test_all = np.concatenate(test_ymeas, axis=0).astype(np.float32)

    print(f"\n原始数据:")
    print(f"  Train: X={X_train_all.shape}, cls={cls_train_all.shape}, "
          f"y_clean={clean_train_all.shape}, y_meas={y_meas_train_all.shape}")
    print(f"  Val:   X={X_val_all.shape}, cls={cls_val_all.shape}, "
          f"y_clean={clean_val_all.shape}, y_meas={y_meas_val_all.shape}")
    print(f"  Test:  X={X_test_all.shape}, cls={cls_test_all.shape}, "
          f"y_clean={clean_test_all.shape}, y_meas={y_meas_test_all.shape}")

    # 归一化: 训练集计算统计量，验证集/测试集复用 (避免信息泄漏)
    print("\n归一化: y_meas -> [2.5, 2.5, π], 创新 -> [0.5, 0.5, 0.3], u_cmd -> 物理上限")
    normalizer = RobustNormalizer()
    X_train_all = normalizer.fit_transform(X_train_all)
    X_val_all = normalizer.transform(X_val_all)
    X_test_all = normalizer.transform(X_test_all)
    normalizer.save(os.path.join(args.output_dir, 'normalizer.npz'))
    print(f"  y_meas median:   {normalizer.ymeas_median}")
    print(f"  y_meas scale:    {normalizer.ymeas_scale}")
    print(f"  创新通道 median: {normalizer.feat_median}")
    print(f"  创新通道 scale:  {normalizer.innov_scale}")
    print(f"  u_cmd max:       {normalizer.cmd_max}")

    # 保存
    print("\n保存 .npy 文件...")
    np.save(os.path.join(args.output_dir, 'X_train.npy'), X_train_all)
    np.save(os.path.join(args.output_dir, 'X_val.npy'), X_val_all)
    np.save(os.path.join(args.output_dir, 'X_test.npy'), X_test_all)
    np.save(os.path.join(args.output_dir, 'Y_train_cls.npy'), cls_train_all)
    np.save(os.path.join(args.output_dir, 'Y_val_cls.npy'), cls_val_all)
    np.save(os.path.join(args.output_dir, 'Y_test_cls.npy'), cls_test_all)
    np.save(os.path.join(args.output_dir, 'Y_train_clean.npy'), clean_train_all)
    np.save(os.path.join(args.output_dir, 'Y_val_clean.npy'), clean_val_all)
    np.save(os.path.join(args.output_dir, 'Y_test_clean.npy'), clean_test_all)
    # y_meas (物理单位, 用于 FM 先验 + 物理损失计算)
    np.save(os.path.join(args.output_dir, 'Y_meas_train.npy'), y_meas_train_all)
    np.save(os.path.join(args.output_dir, 'Y_meas_val.npy'), y_meas_val_all)
    np.save(os.path.join(args.output_dir, 'Y_meas_test.npy'), y_meas_test_all)

    # 统计
    total_mb = (X_train_all.nbytes + X_val_all.nbytes + X_test_all.nbytes +
                clean_train_all.nbytes + clean_val_all.nbytes + clean_test_all.nbytes +
                cls_train_all.nbytes + cls_val_all.nbytes + cls_test_all.nbytes +
                y_meas_train_all.nbytes + y_meas_val_all.nbytes + y_meas_test_all.nbytes) / 1024**2

    print(f"\n{'='*60}")
    print(f"预处理完成！")
    print(f"{'='*60}")
    print(f"  分层 IID 划分: train/val/test = {args.train_ratio:.0%}/{args.val_ratio:.0%}/"
          f"{1-args.train_ratio-args.val_ratio:.0%}")
    print(f"  窗口大小: {args.window} 步 ({args.window*0.05:.1f}s)")
    print(f"  步长:     {args.stride}")
    print(f"  训练窗口: {len(X_train_all):,}")
    print(f"  验证窗口: {len(X_val_all):,}")
    print(f"  测试窗口: {len(X_test_all):,}")
    print(f"  总数据量: {total_mb:.0f} MB")
    print(f"  输出目录: {args.output_dir}")

    # 类别分布
    for split_name, cls_arr in [('Train', cls_train_all), ('Val', cls_val_all),
                                 ('Test', cls_test_all)]:
        counts = defaultdict(int)
        for lbl in cls_arr:
            counts[ALL_ATTACK_TYPES[lbl]] += 1
        print(f"\n  {split_name} 类别分布:")
        for atk in ALL_ATTACK_TYPES:
            print(f"    {atk} ({ATTACK_NAMES[atk]}): {counts[atk]:,}")


if __name__ == "__main__":
    main()
