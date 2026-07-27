"""
preprocess_data.py — 滑动窗口 + 归一化 + .npy 缓存
====================================================
从 .npz 仿真文件提取滑动窗口, 预生成为 .npy 数组供训练直接加载。

关键设计:
  - 窗口 128 步 (6.4s), stride=10 密集采样
  - 物理锚点归一化: y_meas 用工作空间边界 [2.5m, 2.5m, π], u_cmd 用物理上限 [0.3, 1.76]
  - 5 通道: [y_meas(3) + u_cmd(2)]
  - 抗泄漏: 同一 config 的所有窗口整体进入同一划分
  - IID 分层: 按轨迹族分层抽样 train/val/test = 70/15/15

输出: dataset_win/<ts>/X_*.npy, Y_*_cls.npy, Y_*_clean.npy, normalizer.npz, split_info.npz
      (output-dir 自动从 input-dir 推导: dataset/<ts>/ → dataset_win/<ts>/)
"""

import os, sys, argparse
import numpy as np
import pandas as pd
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'dataset_win')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _latest_dataset_dir() -> str:
    """返回 dataset/ 下最新的时间戳子目录，若无则回退到 dataset/ 根目录"""
    if not os.path.isdir(DATASET_DIR):
        return DATASET_DIR
    subdirs = sorted(
        [d for d in os.listdir(DATASET_DIR)
         if os.path.isdir(os.path.join(DATASET_DIR, d))],
        reverse=True)
    if subdirs:
        return os.path.join(DATASET_DIR, subdirs[0])
    return DATASET_DIR


def _auto_output_dir(input_dir: str) -> str:
    """从 input-dir 自动推断 output-dir: dataset/<ts>/ → dataset_win/<ts>/"""
    input_abs = os.path.abspath(input_dir)
    dataset_abs = os.path.abspath(DATASET_DIR)
    if input_abs.startswith(dataset_abs) and len(input_abs) > len(dataset_abs):
        suffix = input_abs[len(dataset_abs):].lstrip(os.sep)
        if suffix:
            return os.path.join(OUTPUT_DIR, suffix)
    return OUTPUT_DIR

from attack import ALL_ATTACK_TYPES, ATTACK_NAMES

WINDOW_SIZE = 128
STRIDE = 10
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SPLIT_SEED = 42

Y_MEAS_SCALE = np.array([2.5, 2.5, np.pi], dtype=np.float32)
PHYSICAL_MAX = np.array([0.3, 1.76], dtype=np.float32)

ATTACK_EPS = 1e-6
MIN_ACTIVE_RATIO = 0.05


class RobustNormalizer:
    """物理锚点归一化器: y_meas→工作空间尺度, u_cmd→控制上限"""

    def __init__(self):
        self.ymeas_median = None
        self.ymeas_scale = Y_MEAS_SCALE.copy()
        self.cmd_max = PHYSICAL_MAX.copy()

    def fit(self, X):
        self.ymeas_median = np.median(X[:, :, 0:3].reshape(-1, 3), axis=0).astype(np.float32)
        return self

    def transform(self, X):
        X_norm = np.zeros_like(X, dtype=np.float32)
        X_norm[:, :, 0:3] = (X[:, :, 0:3] - self.ymeas_median) / self.ymeas_scale
        X_norm[:, :, 3:5] = X[:, :, 3:5] / self.cmd_max
        return X_norm

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, filepath):
        np.savez(filepath, ymeas_median=self.ymeas_median, ymeas_scale=self.ymeas_scale,
                 cmd_max=self.cmd_max)


def build_windows(data, window_size, stride):
    """从单个仿真数据提取滑动窗口 → (X(5ch), y_clean(3ch))"""
    y_meas_all = data['y_meas'].astype(np.float32)
    u_cmd_all = data['u_cmd'].astype(np.float32)
    y_clean_all = data['y_clean'].astype(np.float32)
    n_steps = len(y_meas_all)

    n_windows = (n_steps - window_size) // stride + 1
    X_w = np.zeros((n_windows, window_size, 5), dtype=np.float32)
    y_clean_w = np.zeros((n_windows, window_size, 3), dtype=np.float32)

    for i in range(n_windows):
        s, e = i * stride, i * stride + window_size
        X_w[i, :, 0:3] = y_meas_all[s:e]
        X_w[i, :, 3:5] = u_cmd_all[s:e]
        y_clean_w[i] = y_clean_all[s:e]

    return X_w, y_clean_w


def main():
    parser = argparse.ArgumentParser(description='数据集预处理')
    parser.add_argument('--window', type=int, default=WINDOW_SIZE)
    parser.add_argument('--stride', type=int, default=STRIDE)
    parser.add_argument('--input-dir', type=str, default=None,
                        help='输入目录 (默认: dataset/ 下最新时间戳子目录)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: 从 input-dir 自动推导)')
    parser.add_argument('--train-ratio', type=float, default=TRAIN_RATIO)
    parser.add_argument('--val-ratio', type=float, default=VAL_RATIO)
    parser.add_argument('--split-seed', type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    if args.input_dir is None:
        args.input_dir = _latest_dataset_dir()
        print(f"自动选择最新数据集: {args.input_dir}")
    if args.output_dir is None:
        args.output_dir = _auto_output_dir(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(args.input_dir, 'metadata.csv'))
    rng = np.random.RandomState(args.split_seed)

    # ---- 分层 IID 划分: 按轨迹族 ----
    config_family = {int(row['config_id']): str(row.get('trajectory_family', 'unknown'))
                     for _, row in df.iterrows()}
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
        n_train = max(1, min(int(np.round(n * args.train_ratio)), n - 2))
        n_val = max(1, min(int(np.round(n * args.val_ratio)), n - n_train - 1))
        train_configs.update(cids_sorted[:n_train])
        val_configs.update(cids_sorted[n_train:n_train + n_val])
        test_configs.update(cids_sorted[n_train + n_val:])

    train_files = {row['filename'] for _, row in df.iterrows()
                   if int(row['config_id']) in train_configs}
    val_files = {row['filename'] for _, row in df.iterrows()
                 if int(row['config_id']) in val_configs}
    test_files = {row['filename'] for _, row in df.iterrows()
                  if int(row['config_id']) in test_configs}

    np.savez(os.path.join(args.output_dir, 'split_info.npz'),
             train_files=np.array(sorted(train_files)),
             val_files=np.array(sorted(val_files)),
             test_files=np.array(sorted(test_files)))

    print(f"IID 分层划分: train={len(train_configs)}/{len(val_configs)}/{len(test_configs)} configs")

    # ---- 收集窗口 ----
    A0_LABEL = ALL_ATTACK_TYPES.index('A0')
    train_X, val_X, test_X = [], [], []
    train_clean, val_clean, test_clean = [], [], []
    train_cls, val_cls, test_cls = [], [], []

    for idx, row in df.iterrows():
        fname = row['filename']
        filepath = os.path.join(args.input_dir, fname)
        atk_type = row['attack_type']
        if fname in train_files:
            bucket_X, bucket_clean, bucket_cls = train_X, train_clean, train_cls
        elif fname in val_files:
            bucket_X, bucket_clean, bucket_cls = val_X, val_clean, val_cls
        else:
            bucket_X, bucket_clean, bucket_cls = test_X, test_clean, test_cls

        data = dict(np.load(filepath, allow_pickle=True))
        X_w, y_clean_w = build_windows(data, args.window, args.stride)

        if atk_type == 'A0':
            cls_arr = np.full(len(X_w), A0_LABEL, dtype=np.int64)
        else:
            atk_label = ALL_ATTACK_TYPES.index(atk_type)
            onset = int(row['attack_onset_step'])
            offset = int(row['attack_offset_step'])
            cls_arr = np.full(len(X_w), A0_LABEL, dtype=np.int64)
            for w in range(len(X_w)):
                w_end = w * args.stride + args.window
                if w_end > onset and w * args.stride < offset:
                    mag = np.linalg.norm(y_clean_w[w] - X_w[w, :, :3], axis=1)
                    if atk_type == 'A5':
                        if np.any(mag > ATTACK_EPS):
                            cls_arr[w] = atk_label
                    elif np.mean(mag > ATTACK_EPS) >= MIN_ACTIVE_RATIO:
                        cls_arr[w] = atk_label

        bucket_X.append(X_w)
        bucket_clean.append(y_clean_w)
        bucket_cls.append(cls_arr)

        if (idx + 1) % 20 == 0:
            print(f"  处理: {idx+1}/{len(df)}")

    # ---- 合并 + 归一化 ----
    X_train = np.concatenate(train_X).astype(np.float32)
    X_val = np.concatenate(val_X).astype(np.float32)
    X_test = np.concatenate(test_X).astype(np.float32)

    normalizer = RobustNormalizer()
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)
    normalizer.save(os.path.join(args.output_dir, 'normalizer.npz'))

    # ---- 保存 ----
    for split, X, clean, cls in [
        ('train', X_train, np.concatenate(train_clean).astype(np.float32), np.concatenate(train_cls)),
        ('val', X_val, np.concatenate(val_clean).astype(np.float32), np.concatenate(val_cls)),
        ('test', X_test, np.concatenate(test_clean).astype(np.float32), np.concatenate(test_cls)),
    ]:
        np.save(os.path.join(args.output_dir, f'X_{split}.npy'), X)
        np.save(os.path.join(args.output_dir, f'Y_{split}_cls.npy'), cls)
        np.save(os.path.join(args.output_dir, f'Y_{split}_clean.npy'), clean)

    # ---- 统计 ----
    total_mb = sum(arr.nbytes for arr in [X_train, X_val, X_test] +
                   [np.concatenate(c).astype(np.float32) for c in [train_clean, val_clean, test_clean]]) / 1024**2
    print(f"\n预处理完成: 窗口 {len(X_train):,}/{len(X_val):,}/{len(X_test):,} (train/val/test), "
          f"{total_mb:.0f}MB")

    for split_name, cls_arr in [('Train', np.concatenate(train_cls)),
                                 ('Val', np.concatenate(val_cls)),
                                 ('Test', np.concatenate(test_cls))]:
        counts = defaultdict(int)
        for lbl in cls_arr:
            counts[ALL_ATTACK_TYPES[lbl]] += 1
        print(f"  {split_name}: " + ", ".join(f"{a}={counts[a]}" for a in ALL_ATTACK_TYPES))


if __name__ == "__main__":
    main()
