"""
preprocess_data.py — 密集滑动窗口 + 归一化 + .npy 缓存
==============================================================================
从 .npz 仿真文件提取滑动窗口，预生成为 .npy 数组供训练直接加载。

关键设计:
  - 窗口大小: 100 步 (5 秒 @ Ts=0.05)
  - 步长 stride=1: 密集采样最大化样本量
  - RobustScaler + 物理归一化: 特征通道用 (x−median)/IQR, u_cmd 用物理上限
  - 输入通道: ekf_innovation(3) + u_cmd(2) = 5 通道
  - 抗泄漏: 同一 .npz 文件的所有窗口始终整体进入同一划分

划分模式 (--split-mode):
  config      — 按 config_id 划分 (默认, train/val = 80/20)
  trajectory  — 按轨迹族划分 (circular→test, 其余→train, 测试泛化到新轨迹类型)

输出文件 (保存在 dataset_win/ 目录):
  X_train.npy, X_val.npy          — 输入窗口 (N, 100, 5) float32
  Y_train_cls.npy, Y_val_cls.npy  — 攻击分类标签 (N,) int64
  Y_train_atk.npy, Y_val_atk.npy  — 攻击信号窗口 (N, 100, 3) float32
  split_info.npz                   — 划分信息 (模式、train/test 文件列表)

用法:
  python preprocess_data.py
  python preprocess_data.py --split-mode trajectory
  python preprocess_data.py --window 150 --stride 2
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'dataset_win')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
ATTACK_NAMES = {
    'A0': 'Normal', 'A1': 'ConstantBias', 'A2': 'Sinusoidal',
    'A3': 'RampDrift', 'A4': 'StepAttack', 'A5': 'ReplayAttack',
    'A6': 'PulseTrain', 'A7': 'ChirpSweep', 'A8': 'MultiTone',
}

# 输入通道: 内部运动学新息 + 控制上下文
INPUT_CHANNELS = ['internal_innovation', 'u_cmd']   # internal_innovation(3) + u_cmd(2) = 5 通道
N_STEPS = int(35.0 / 0.05)  # 700
WINDOW_SIZE = 100
STRIDE = 1
TRAIN_RATIO = 0.8  # config 0-15 train, 16-19 val


# 物理上限 (TurtleBot4 安全模式)
PHYSICAL_MAX = np.array([0.3, 1.76], dtype=np.float32)  # v_max, ω_max


class RobustNormalizer:
    """全局鲁棒归一化器

    - 特征通道 (前 N−2 个): RobustScaler — (x − median) / IQR
    - u_cmd (最后 2 通道):   物理归一化 — x / [v_max, ω_max]

    所有统计量从训练集计算，验证集复用，保证无信息泄漏。
    """

    def __init__(self):
        self.feat_median = None   # (C−2,) 特征通道的中位数
        self.feat_iqr = None      # (C−2,) 特征通道的 IQR
        self.cmd_max = PHYSICAL_MAX.copy()  # (2,) 物理上限

    def fit(self, X: np.ndarray):
        """从训练集计算归一化参数

        Args:
            X: (N, W, C) 训练窗口, 最后 2 通道为 u_cmd
        """
        n_feat = X.shape[2] - 2
        feat_data = X[:, :, :n_feat].reshape(-1, n_feat)
        self.feat_median = np.median(feat_data, axis=0).astype(np.float32)
        q25 = np.percentile(feat_data, 25, axis=0).astype(np.float32)
        q75 = np.percentile(feat_data, 75, axis=0).astype(np.float32)
        self.feat_iqr = np.maximum(q75 - q25, 1e-6)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """应用归一化"""
        n_feat = X.shape[2] - 2
        X_norm = np.zeros_like(X, dtype=np.float32)
        X_norm[:, :, :n_feat] = (X[:, :, :n_feat] - self.feat_median) / self.feat_iqr
        X_norm[:, :, n_feat:] = X[:, :, n_feat:] / self.cmd_max
        return X_norm

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, filepath: str):
        np.savez(filepath, feat_median=self.feat_median, feat_iqr=self.feat_iqr,
                 cmd_max=self.cmd_max)

    @classmethod
    def load(cls, filepath: str) -> 'RobustNormalizer':
        data = np.load(filepath)
        obj = cls()
        obj.feat_median = data['feat_median']
        obj.feat_iqr = data['feat_iqr']
        obj.cmd_max = data['cmd_max']
        return obj


def load_npz_file(filepath: str) -> dict:
    """加载 .npz 仿真文件"""
    return dict(np.load(filepath, allow_pickle=True))


def build_windows(data: dict, window_size: int, stride: int) -> dict:
    """从单个仿真数据中提取滑动窗口

    Args:
        data:      .npz 加载的字典，含各信号的时间序列
        window_size: 窗口大小
        stride:      步长

    Returns:
        X_windows:  (N, W, 8)  输入特征窗口
        atk_windows: (N, W, 3) 攻击信号窗口 (重建目标)
    """
    # 拼接输入通道
    ch_arrays = []
    for ch_name in INPUT_CHANNELS:
        arr = data[ch_name]  # (700,) or (700, D)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        ch_arrays.append(arr)
    X_all = np.concatenate(ch_arrays, axis=1).astype(np.float32)  # (700, 5)

    atk_all = data['attack_signal'].astype(np.float32)  # (700, 3)

    # 滑动窗口
    n_windows = (N_STEPS - window_size) // stride + 1
    X_windows = np.zeros((n_windows, window_size, X_all.shape[1]), dtype=np.float32)
    atk_windows = np.zeros((n_windows, window_size, atk_all.shape[1]), dtype=np.float32)

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        X_windows[i] = X_all[start:end]
        atk_windows[i] = atk_all[start:end]

    return X_windows, atk_windows


def main():
    parser = argparse.ArgumentParser(
        description='数据集预处理：密集滑动窗口 + 归一化 + .npy 缓存')
    parser.add_argument('--window', type=int, default=WINDOW_SIZE,
                        help=f'窗口大小 (默认 {WINDOW_SIZE})')
    parser.add_argument('--stride', type=int, default=STRIDE,
                        help=f'步长 (默认 {STRIDE})')
    parser.add_argument('--input-dir', type=str, default=DATASET_DIR)
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--split-mode', type=str, default='config',
                        choices=['config', 'trajectory'],
                        help='划分模式: config=按config_id (默认), trajectory=按轨迹族')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 metadata
    metadata_path = os.path.join(args.input_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"[ERROR] metadata.csv 未找到: {metadata_path}")
        sys.exit(1)
    df = pd.read_csv(metadata_path)

    # ---- 确定文件→划分映射 (抗泄漏: 同一文件所有窗口整体进同一划分) ----
    train_files = set()
    val_files = set()

    if args.split_mode == 'config':
        num_configs = df['config_id'].nunique()
        train_cutoff = int(num_configs * TRAIN_RATIO)
        for _, row in df.iterrows():
            if int(row['config_id']) < train_cutoff:
                train_files.add(row['filename'])
            else:
                val_files.add(row['filename'])
        print(f"划分模式: config (train=config 0~{train_cutoff-1}, "
              f"val=config {train_cutoff}~{num_configs-1})")

    elif args.split_mode == 'trajectory':
        # circular 轨迹族整体进入验证集，其余进训练集
        # 测试泛化到全新轨迹类型的能力
        TEST_FAMILIES = {'circular'}
        for _, row in df.iterrows():
            traj_family = str(row.get('trajectory_family', 'unknown'))
            if traj_family in TEST_FAMILIES:
                val_files.add(row['filename'])
            else:
                train_files.add(row['filename'])
        train_families = set()
        val_families = set()
        for _, row in df.iterrows():
            fam = str(row.get('trajectory_family', 'unknown'))
            if row['filename'] in train_files:
                train_families.add(fam)
            else:
                val_families.add(fam)
        print(f"划分模式: trajectory")
        print(f"  Train 轨迹族: {sorted(train_families)}")
        print(f"  Val   轨迹族: {sorted(val_families)}")

    # 保存划分信息
    split_info = {
        'split_mode': args.split_mode,
        'train_files': sorted(train_files),
        'val_files': sorted(val_files),
    }
    np.savez(os.path.join(args.output_dir, 'split_info.npz'),
             split_mode=np.array(args.split_mode),
             train_files=np.array(sorted(train_files)),
             val_files=np.array(sorted(val_files)))

    # 收集数据 (按文件整体分配，不拆分窗口)
    train_X, val_X = [], []
    train_atk, val_atk = [], []
    train_cls, val_cls = [], []

    n_per_file = (N_STEPS - args.window) // args.stride + 1
    A0_LABEL = ALL_ATTACK_TYPES.index('A0')

    print(f"窗口大小: {args.window}, 步长: {args.stride}")
    print(f"每文件窗口数: {n_per_file}")
    print(f"攻击起点: 随机 [5, 30]s (逐文件记录在 metadata.csv)")
    print(f"总文件数: {len(df)} (train: {len(train_files)}, val: {len(val_files)})")

    for idx, row in df.iterrows():
        fname = row['filename']
        filepath = os.path.join(args.input_dir, fname)
        atk_type = row['attack_type']
        attack_onset = float(row.get('attack_onset', 15.0))
        in_train = fname in train_files

        data = load_npz_file(filepath)
        X_w, atk_w = build_windows(data, args.window, args.stride)

        # 逐窗口标注（考虑攻击结束时间）
        is_normal = (atk_type == 'A0')
        n_w = len(X_w)
        cls_per_window = np.full(n_w, A0_LABEL, dtype=np.int64)

        if not is_normal:
            atk_label = ALL_ATTACK_TYPES.index(atk_type)
            onset_step = int(attack_onset / 0.05)
            # 攻击结束时间（默认 inf = 永不结束）
            attack_offset = float(row.get('attack_offset', 35.0 + 1.0))
            offset_step = int(attack_offset / 0.05) if attack_offset < 35.0 else N_STEPS + 1
            for w in range(n_w):
                w_end = w * args.stride + args.window
                w_start = w * args.stride
                # 窗口结束在攻击开始后 且 窗口开始在攻击结束前 → 含攻击
                if w_end > onset_step and w_start < offset_step:
                    cls_per_window[w] = atk_label

        if in_train:
            train_X.append(X_w)
            train_atk.append(atk_w)
            train_cls.append(cls_per_window)
        else:
            val_X.append(X_w)
            val_atk.append(atk_w)
            val_cls.append(cls_per_window)

        if (idx + 1) % 20 == 0:
            print(f"  处理进度: {idx+1}/{len(df)}")

    # 合并
    print("合并数据...")
    X_train_all = np.concatenate(train_X, axis=0).astype(np.float32)
    X_val_all = np.concatenate(val_X, axis=0).astype(np.float32)
    atk_train_all = np.concatenate(train_atk, axis=0).astype(np.float32)
    atk_val_all = np.concatenate(val_atk, axis=0).astype(np.float32)
    cls_train_all = np.concatenate(train_cls, axis=0)
    cls_val_all = np.concatenate(val_cls, axis=0)

    print(f"\n原始数据:")
    print(f"  Train: X={X_train_all.shape}, cls={cls_train_all.shape}, atk={atk_train_all.shape}")
    print(f"  Val:   X={X_val_all.shape}, cls={cls_val_all.shape}, atk={atk_val_all.shape}")

    # 归一化: 训练集计算统计量，验证集复用 (避免信息泄漏)
    print("\n归一化: 特征通道 → RobustScaler, u_cmd → 物理上限")
    normalizer = RobustNormalizer()
    X_train_all = normalizer.fit_transform(X_train_all)
    X_val_all = normalizer.transform(X_val_all)
    normalizer.save(os.path.join(args.output_dir, 'normalizer.npz'))
    print(f"  特征通道 median: {normalizer.feat_median}")
    print(f"  特征通道 IQR:    {normalizer.feat_iqr}")
    print(f"  u_cmd max:       {normalizer.cmd_max}")

    # 保存
    print("\n保存 .npy 文件...")
    np.save(os.path.join(args.output_dir, 'X_train.npy'), X_train_all)
    np.save(os.path.join(args.output_dir, 'X_val.npy'), X_val_all)
    np.save(os.path.join(args.output_dir, 'Y_train_cls.npy'), cls_train_all)
    np.save(os.path.join(args.output_dir, 'Y_val_cls.npy'), cls_val_all)
    np.save(os.path.join(args.output_dir, 'Y_train_atk.npy'), atk_train_all)
    np.save(os.path.join(args.output_dir, 'Y_val_atk.npy'), atk_val_all)

    # 统计
    total_mb = (X_train_all.nbytes + X_val_all.nbytes +
                atk_train_all.nbytes + atk_val_all.nbytes +
                cls_train_all.nbytes + cls_val_all.nbytes) / 1024**2

    print(f"\n{'='*60}")
    print(f"预处理完成！")
    print(f"{'='*60}")
    print(f"  划分模式: {args.split_mode}")
    print(f"  窗口大小: {args.window} 步 ({args.window*0.05:.1f}s)")
    print(f"  步长:     {args.stride}")
    print(f"  训练窗口: {len(X_train_all):,}")
    print(f"  验证窗口: {len(X_val_all):,}")
    print(f"  总数据量: {total_mb:.0f} MB")
    print(f"  输出目录: {args.output_dir}")

    # 类别分布
    for split_name, cls_arr in [('Train', cls_train_all), ('Val', cls_val_all)]:
        counts = defaultdict(int)
        for lbl in cls_arr:
            counts[ALL_ATTACK_TYPES[lbl]] += 1
        print(f"\n  {split_name} 类别分布:")
        for atk in ALL_ATTACK_TYPES:
            print(f"    {atk} ({ATTACK_NAMES[atk]}): {counts[atk]:,}")


if __name__ == "__main__":
    main()
