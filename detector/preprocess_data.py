"""
preprocess_data.py — 密集滑动窗口 + 归一化 + .npy 缓存
==============================================================================
从 .npz 仿真文件提取滑动窗口，预生成为 .npy 数组供训练直接加载。

关键设计:
  - 窗口大小: 100 步 (5 秒 @ Ts=0.05)
  - 步长 stride=1: 密集采样最大化样本量
  - RobustScaler + 物理归一化: 特征通道用 (x−median)/IQR, u_cmd 用物理上限
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
DATASET_DIR = os.path.join(SCRIPT_DIR, '..', 'dataset')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'dataset_win')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
ATTACK_NAMES = {
    'A0': 'Normal', 'A1': 'ConstantBias', 'A2': 'Sinusoidal',
    'A3': 'Drift', 'A4': 'Step', 'A5': 'ReplayAttack',
    'A6': 'Dropout', 'A7': 'Scaling', 'A8': 'Freeze',
}

# 输入通道: 内部运动学新息 + 控制上下文
INPUT_CHANNELS = ['internal_innovation', 'u_cmd']   # internal_innovation(3) + u_cmd(2) = 5 通道
from model import SIM_STEPS, SIM_TIME
N_STEPS = SIM_STEPS       # 1000，避免 IEEE 754 截断误差
WINDOW_SIZE = 100
STRIDE = 1
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15    # test = 1 - train - val = 0.15
SPLIT_SEED = 42     # 分层抽样固定种子, 保证可复现


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


def build_windows(data: dict, window_size: int, stride: int) -> tuple:
    """从单个仿真数据中提取滑动窗口

    Args:
        data:      .npz 加载的字典，含各信号的时间序列
        window_size: 窗口大小
        stride:      步长

    Returns:
        X_windows:      (N, W, 5)  模型输入特征窗口 (innovation + u_cmd)
        atk_windows:    (N, W, 3)  攻击信号窗口 (重建目标)
        y_meas_windows: (N, W, 3)  原始测量窗口 (物理单位, 仅用于物理损失)
    """
    # 拼接模型输入通道 (innovation + u_cmd)
    ch_arrays = []
    for ch_name in INPUT_CHANNELS:
        arr = data[ch_name]
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        ch_arrays.append(arr)
    X_all = np.concatenate(ch_arrays, axis=1).astype(np.float32)  # (T, 5)

    atk_all = data['attack_signal'].astype(np.float32)           # (T, 3)
    y_meas_all = data['y_meas'].astype(np.float32)               # (T, 3) 物理单位

    # 滑动窗口
    n_windows = (N_STEPS - window_size) // stride + 1
    X_windows = np.zeros((n_windows, window_size, X_all.shape[1]), dtype=np.float32)
    atk_windows = np.zeros((n_windows, window_size, atk_all.shape[1]), dtype=np.float32)
    y_meas_windows = np.zeros((n_windows, window_size, y_meas_all.shape[1]), dtype=np.float32)

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        X_windows[i] = X_all[start:end]
        atk_windows[i] = atk_all[start:end]
        y_meas_windows[i] = y_meas_all[start:end]

    return X_windows, atk_windows, y_meas_windows


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
    # 算法:
    #   1. 按 trajectory_family 分组 config_id
    #   2. 每组内用固定 seed 随机打乱 config_id
    #   3. 按 train/val/test 比例切分
    # 结果: 每个轨迹族均按比例分布在 train/val/test 中, 保证 IID
    rng = np.random.RandomState(args.split_seed)

    # 构建 config_id → trajectory_family 映射
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
        # 确保至少各1个, test 拿剩下的
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
    # 逐族明细
    for fam in sorted(families.keys()):
        t_c = sorted([c for c in families[fam] if c in train_configs])
        v_c = sorted([c for c in families[fam] if c in val_configs])
        te_c = sorted([c for c in families[fam] if c in test_configs])
        print(f"    {fam}: train={t_c}, val={v_c}, test={te_c}")

    # 保存划分信息
    split_info = {
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'split_seed': args.split_seed,
        'train_files': sorted(train_files),
        'val_files': sorted(val_files),
        'test_files': sorted(test_files),
    }
    np.savez(os.path.join(args.output_dir, 'split_info.npz'),
             train_ratio=np.array(args.train_ratio),
             val_ratio=np.array(args.val_ratio),
             split_seed=np.array(args.split_seed),
             train_files=np.array(sorted(train_files)),
             val_files=np.array(sorted(val_files)),
             test_files=np.array(sorted(test_files)))

    # 收集数据 (按文件整体分配，不拆分窗口)
    train_X, val_X, test_X = [], [], []
    train_atk, val_atk, test_atk = [], [], []
    train_cls, val_cls, test_cls = [], [], []
    train_ymeas, val_ymeas, test_ymeas = [], [], []

    n_per_file = (N_STEPS - args.window) // args.stride + 1
    A0_LABEL = ALL_ATTACK_TYPES.index('A0')

    print(f"窗口大小: {args.window}, 步长: {args.stride}")
    print(f"每文件窗口数: {n_per_file}")
    print(f"攻击起点: 随机 [5, 30]s (逐文件记录在 metadata.csv)")
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
        X_w, atk_w, y_meas_w = build_windows(data, args.window, args.stride)

        # 逐窗口标注（考虑攻击结束时间）
        is_normal = (atk_type == 'A0')
        n_w = len(X_w)
        cls_per_window = np.full(n_w, A0_LABEL, dtype=np.int64)

        if not is_normal:
            atk_label = ALL_ATTACK_TYPES.index(atk_type)
            # 优先使用元数据中的整数步索引 (避免浮点精度损失)
            if 'attack_onset_step' in row.index and not pd.isna(row['attack_onset_step']):
                onset_step = int(row['attack_onset_step'])
            else:
                onset_step = int(round(attack_onset / 0.05))
            # 攻击结束时间 (默认 inf = 永不结束)
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
                # 窗口结束在攻击开始后 且 窗口开始在攻击结束前 → 含攻击
                if w_end > onset_step and w_start < offset_step:
                    cls_per_window[w] = atk_label

        if in_train:
            train_X.append(X_w)
            train_atk.append(atk_w)
            train_cls.append(cls_per_window)
            train_ymeas.append(y_meas_w)
        elif in_val:
            val_X.append(X_w)
            val_atk.append(atk_w)
            val_cls.append(cls_per_window)
            val_ymeas.append(y_meas_w)
        else:
            test_X.append(X_w)
            test_atk.append(atk_w)
            test_cls.append(cls_per_window)
            test_ymeas.append(y_meas_w)

        if (idx + 1) % 20 == 0:
            print(f"  处理进度: {idx+1}/{len(df)}")

    # 合并
    print("合并数据...")
    X_train_all = np.concatenate(train_X, axis=0).astype(np.float32)
    X_val_all = np.concatenate(val_X, axis=0).astype(np.float32)
    X_test_all = np.concatenate(test_X, axis=0).astype(np.float32)
    atk_train_all = np.concatenate(train_atk, axis=0).astype(np.float32)
    atk_val_all = np.concatenate(val_atk, axis=0).astype(np.float32)
    atk_test_all = np.concatenate(test_atk, axis=0).astype(np.float32)
    cls_train_all = np.concatenate(train_cls, axis=0)
    cls_val_all = np.concatenate(val_cls, axis=0)
    cls_test_all = np.concatenate(test_cls, axis=0)
    y_meas_train_all = np.concatenate(train_ymeas, axis=0).astype(np.float32)
    y_meas_val_all = np.concatenate(val_ymeas, axis=0).astype(np.float32)
    y_meas_test_all = np.concatenate(test_ymeas, axis=0).astype(np.float32)

    print(f"\n原始数据:")
    print(f"  Train: X={X_train_all.shape}, cls={cls_train_all.shape}, atk={atk_train_all.shape}, "
          f"y_meas={y_meas_train_all.shape}")
    print(f"  Val:   X={X_val_all.shape}, cls={cls_val_all.shape}, atk={atk_val_all.shape}, "
          f"y_meas={y_meas_val_all.shape}")
    print(f"  Test:  X={X_test_all.shape}, cls={cls_test_all.shape}, atk={atk_test_all.shape}, "
          f"y_meas={y_meas_test_all.shape}")

    # 归一化: 训练集计算统计量，验证集/测试集复用 (避免信息泄漏)
    print("\n归一化: 特征通道 → RobustScaler, u_cmd → 物理上限")
    normalizer = RobustNormalizer()
    X_train_all = normalizer.fit_transform(X_train_all)
    X_val_all = normalizer.transform(X_val_all)
    X_test_all = normalizer.transform(X_test_all)
    normalizer.save(os.path.join(args.output_dir, 'normalizer.npz'))
    print(f"  特征通道 median: {normalizer.feat_median}")
    print(f"  特征通道 IQR:    {normalizer.feat_iqr}")
    print(f"  u_cmd max:       {normalizer.cmd_max}")

    # 保存
    print("\n保存 .npy 文件...")
    np.save(os.path.join(args.output_dir, 'X_train.npy'), X_train_all)
    np.save(os.path.join(args.output_dir, 'X_val.npy'), X_val_all)
    np.save(os.path.join(args.output_dir, 'X_test.npy'), X_test_all)
    np.save(os.path.join(args.output_dir, 'Y_train_cls.npy'), cls_train_all)
    np.save(os.path.join(args.output_dir, 'Y_val_cls.npy'), cls_val_all)
    np.save(os.path.join(args.output_dir, 'Y_test_cls.npy'), cls_test_all)
    np.save(os.path.join(args.output_dir, 'Y_train_atk.npy'), atk_train_all)
    np.save(os.path.join(args.output_dir, 'Y_val_atk.npy'), atk_val_all)
    np.save(os.path.join(args.output_dir, 'Y_test_atk.npy'), atk_test_all)
    # y_meas (物理单位, 仅用于物理损失计算)
    np.save(os.path.join(args.output_dir, 'Y_meas_train.npy'), y_meas_train_all)
    np.save(os.path.join(args.output_dir, 'Y_meas_val.npy'), y_meas_val_all)
    np.save(os.path.join(args.output_dir, 'Y_meas_test.npy'), y_meas_test_all)

    # 统计
    total_mb = (X_train_all.nbytes + X_val_all.nbytes + X_test_all.nbytes +
                atk_train_all.nbytes + atk_val_all.nbytes + atk_test_all.nbytes +
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
