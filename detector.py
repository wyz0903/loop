"""
detector.py — 传感器攻击检测与信号恢复模块
============================================================================
即插即用的攻击检测器，不改动 EKF 或 NMPC。

架构位置：
  Sensor → [Attack] → y_meas → [Detector] → y_rec → [EKF] → X_hat → [NMPC]

检测器功能：
  1. 分类攻击类别 (A0~A8)
  2. 恢复干净传感器信号 (攻击移除)

NNDetector 策略：
  - 内部运动学模型预测位姿 → 新息暴露攻击信号
  - 滑动窗口缓冲 (100步) → AttackClassifier 推理 → 攻击类型 + 置信度 + â(k)
  - 分类冻结：连续高置信度后锁定类型，跳过后续 NN 推理
  - 类型特定恢复路由器：A0 直通, A1/A4/A6 统计估计, A2/A7/A8 NN 解码器,
    A3 趋势外推, A5 内部运动学死推算

Oracle 策略：
  - 已知 ground truth a(k)，完美移除攻击 + 传感器噪声滤波（理论上界）

参考文献：
  - Zhang et al. (2026) PTESO-MPC, IEEE TIE
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from collections import deque

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 模块常量
# ============================================================================

STATE_DIM = 3             # 传感器测量维度 [x, y, theta]
TS = 0.05                 # 采样周期 [s]
ALPHA = 0.17              # 前端偏置距离 [m] (与 model.py 同步)
NN_WINDOW_SIZE = 100      # NN 输入窗口 (5s @ 50ms)


# ============================================================================
# 检测结果数据结构
# ============================================================================

@dataclass
class DetectionResult:
    """单步检测器的完整输出

    Attributes:
        attack_class:   攻击类别标签 'A0'~'A8'
        confidence:     分类置信度 [0, 1]
        y_recovered:    恢复后的传感器信号 (3,) — 输入 EKF
        attack_estimate: 估计的攻击分量 (3,) — y_meas - y_recovered
        features:        附加信息字典
    """
    attack_class: str
    confidence: float
    y_recovered: np.ndarray
    attack_estimate: np.ndarray
    features: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"DetectionResult(class={self.attack_class}, "
                f"conf={self.confidence:.3f}, "
                f"|a_est|={np.linalg.norm(self.attack_estimate):.4f})")


# ============================================================================
# NN 检测器 — 即插即用
# ============================================================================

class NNDetector:
    """神经网络攻击检测器 — 即插即用，不改动控制系统

    对外仅暴露 detect(y_meas) → DetectionResult。
    通过 set_control(u_cmd) 接收控制指令，供内部运动学模型使用。

    核心原理:
      1. 内部运动学模型用 u_cmd 做开环位姿预测
      2. 新息 = y_meas - X_pred 暴露攻击信号
      3. NN 解码器从新息窗口重建攻击信号 â(k)
      4. 统一恢复: y_rec = y_meas - â(k)
      5. A5（重放攻击）例外: 非加性攻击，切换为内部运动学死推算

    论文核心贡献:
      端到端学习的攻击信号估计器配合内部运动学模型，
      实现对全部加性攻击的统一信号恢复，无需按攻击类型分别设计恢复策略。
    """

    ALL_ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']

    # 滞留时间参数: 过去 N 步分类历史的多数投票 → 防抖同时允许攻击结束后切换回 A0
    DWELL_STEPS = 10               # 投票窗口大小 (0.5s @ 50ms, 减半以加快响应)

    # 加权尾部平均: 恢复时使用解码器最后 N 步的指数加权平均 (非仅最后一步)
    TAIL_AVG_STEPS = 10            # 尾部平均步数
    TAIL_ALPHA = 0.7               # 指数衰减因子 (越近权重越大)

    # EMA 分类: 软置信度融合 (更快响应, 补充多数投票)
    CLASS_EMA_ALPHA = 0.3          # EMA 更新率 (越大越敏感)

    # A5 死推算防抖: 需连续 N 步原始 NN 输出为 A5 才激活
    A5_CONSECUTIVE_REQUIRED = 5    # 连续 A5 步数要求 (0.25s)
    A5_DEACTIVATE_CONSECUTIVE = 3  # 连续非 A5 步数要求才退出

    # 置信度门控恢复: 仅当分类置信度 > 阈值时才应用 NN 恢复
    RECOVERY_CONFIDENCE_THRESH = 0.6   # 中等置信度即可应用部分恢复
    RECOVERY_CONFIDENCE_FULL = 0.85     # 高置信度 → 全量恢复
    RECOVERY_BLEND_ALPHA = 0.5          # 全量恢复时的混合比例
    RECOVERY_BLEND_PARTIAL = 0.25       # 中等置信度时的部分混合比例

    # FM 不确定性参数
    FM_N_SAMPLES = 3                  # FM 多采样次数 (不确定性估计)
    FM_UNCERTAINTY_THRESH = 0.3       # 不确定性阈值 (超过 → 加倍保守)
    FM_K_STEPS = 4                    # FM ODE 积分步数 (默认值, 从 config 覆盖)

    # 内部状态重校准: 周期性用 EKF 估计重置内部运动学状态 (控制漂移)
    RECALIB_INTERVAL = 200         # 重校准间隔 [步] (10s @ 50ms)
    RECALIB_CONFIDENCE_THRESH = 0.8 # 仅在高置信度且非 A5 时重校准

    def __init__(self, model_path: str = None, norm_path: str = None,
                 window_size: int = NN_WINDOW_SIZE, device: str = None):
        """
        Args:
            model_path:  训练好的 AttackClassifier 权重路径
            norm_path:   RobustNormalizer 参数路径 (normalizer.npz)
            window_size: NN 输入窗口大小 (默认 100)
            device:      推理设备 ('cuda' / 'cpu')
        """
        if model_path is None:
            model_path = os.path.join(SCRIPT_DIR, 'models', 'cls_best.pt')
        if norm_path is None:
            norm_path = os.path.join(SCRIPT_DIR, 'dataset_win', 'config', 'normalizer.npz')

        self.window_size = window_size

        # ---- 设备 ----
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

        # ---- 加载模型 ----
        self._model, self._in_channels, self._is_fm_model = self._load_model(model_path)
        self._model.to(self._device)
        self._model.eval()

        # ---- FM 推理配置 ----
        self._fm_k_steps = self.FM_K_STEPS

        # ---- 加载归一化参数 ----
        self._load_normalizer(norm_path)

        # ---- 内部运动学状态 ----
        self._internal_state = np.array([0.0, 0.1, 0.0])  # [x, y, theta]
        self._u_cmd = np.zeros(2)                          # 最近一次控制指令
        self._step_count = 0

        # ---- 滑动窗口缓冲区 ----
        self._innov_buffer = deque(maxlen=window_size)   # (window_size, 3)
        self._ucmd_buffer = deque(maxlen=window_size)    # (window_size, 2)

        # ---- A5 死推算状态 ----
        self._dead_reckon_active = False
        self._a5_consecutive_count = 0       # 连续 A5 原始预测计数
        self._non_a5_consecutive_count = 0   # 连续非 A5 原始预测计数

        # ---- 滞留时间共识投票 ----
        self._class_history = deque(maxlen=self.DWELL_STEPS)
        self._confidence_history = deque(maxlen=self.DWELL_STEPS)

        # ---- EMA 软分类状态: 更快的置信度响应 ----
        self._class_ema = np.zeros(len(self.ALL_ATTACK_TYPES))  # 各类别的 EMA 概率

        # ---- 内部状态重校准 ----
        self._last_recalib_step = 0
        self._ekf_state_external = None  # 外部 EKF 估计 (由 set_ekf_state 设置)

        # ---- 解码器尾部缓冲区: 加权平均用 ----
        self._decoder_tail_buffer = deque(maxlen=self.TAIL_AVG_STEPS)

        # ---- 创新息持久偏移检测: NN 分类 A0 时的兜底补偿 ----
        self._innov_ema = np.zeros(3)          # 创新息长期 EMA
        self._innov_ema_alpha = 0.01           # EMA 衰减因子 (100步时间常数)
        self._innov_offset_thresh = 0.02       # 持久偏移阈值 [m] (约 2× 传感器噪声)

        print(f"[NNDetector] 模型已加载: {model_path}")
        fm_info = f", FM(K={self._fm_k_steps}, N={self.FM_N_SAMPLES})" if self._is_fm_model else ""
        print(f"  设备: {self._device}, 窗口: {window_size}, "
              f"Dwell={self.DWELL_STEPS}步, TailAvg={self.TAIL_AVG_STEPS}步{fm_info}")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def detect(self, y_meas: np.ndarray) -> DetectionResult:
        """主检测接口 — 每步调用一次

        检测策略 (v3):
          - 每步运行完整 NN 推理（分类 + 解码器全窗口输出）
          - 多数投票 (DWELL_STEPS=10) + EMA 软分类双轨融合
          - 加权尾部平均: 用解码器最后 TAIL_AVG_STEPS 步的指数加权平均做恢复
          - 周期性内部状态重校准: 高置信度时用 EKF 估计重置运动学状态
          - A5 死推算决策绕过投票，直接使用原始 NN 输出

        Args:
            y_meas: 当前传感器测量值 (3,) [x, y, theta]

        Returns:
            DetectionResult
        """
        self._step_count += 1
        y_meas = np.asarray(y_meas, dtype=float).ravel()

        # 1. 内部运动学预测 → 新息 + 保存 X_pred
        X_pred, innovation = self._compute_innovation(y_meas)

        # 1.5 更新创新息长期 EMA (持久偏移检测)
        self._innov_ema = (self._innov_ema_alpha * innovation +
                           (1 - self._innov_ema_alpha) * self._innov_ema)

        # 2. 推入缓冲区
        self._innov_buffer.append(innovation)
        self._ucmd_buffer.append(self._u_cmd.copy())

        # 3. 更新内部运动学状态
        self._internal_state = X_pred.copy()

        # 4. 周期性内部状态重校准: 高置信度 + 非 A5 → 用 EKF 估计重置
        self._maybe_recalibrate()

        # 5. 窗口未就绪 → 直通
        if not self._is_window_ready():
            return DetectionResult(
                attack_class='A0',
                confidence=0.0,
                y_recovered=y_meas.copy(),
                attack_estimate=np.zeros(3),
                features={'status': 'window_filling',
                         'readiness': self._window_readiness()}
            )

        # 6. NN 推理 — 每步完整执行, 返回全窗口解码 + FM 不确定性
        innov_window = np.array(list(self._innov_buffer))
        ucmd_window = np.array(list(self._ucmd_buffer))
        attack_class_raw, confidence_raw, nn_attack_est, attack_seq_full, fm_uncertainty = (
            self._nn_infer(innov_window, ucmd_window))

        # 7. EMA 软分类: 快速响应
        cls_idx = self.ALL_ATTACK_TYPES.index(attack_class_raw)
        one_hot = np.zeros(len(self.ALL_ATTACK_TYPES))
        one_hot[cls_idx] = 1.0
        self._class_ema = (self.CLASS_EMA_ALPHA * one_hot +
                          (1 - self.CLASS_EMA_ALPHA) * self._class_ema)

        # 8. 滞留时间共识投票: 防抖
        self._class_history.append(attack_class_raw)
        self._confidence_history.append(confidence_raw)

        from collections import Counter
        class_counts = Counter(self._class_history)
        attack_class_voted = class_counts.most_common(1)[0][0]
        vote_confidence = class_counts[attack_class_voted] / len(self._class_history)

        # 融合: 如果 EMA 和投票一致 → 用投票; 不一致 → EMA 主导(更快响应变化)
        ema_class = self.ALL_ATTACK_TYPES[int(np.argmax(self._class_ema))]
        ema_conf = float(np.max(self._class_ema))

        if ema_class == attack_class_voted:
            attack_class = attack_class_voted
            confidence = max(vote_confidence, ema_conf)
        else:
            # EMA 权重更高 → 更快检测攻击开始/结束
            if ema_conf > 0.6:
                attack_class = ema_class
                confidence = ema_conf
            else:
                attack_class = attack_class_voted
                confidence = vote_confidence

        # 9. 加权尾部平均恢复
        nn_attack_est_tail = self._tail_weighted_average(attack_seq_full)

        # 10. 统一恢复 (含 FM 不确定性门控)
        y_recovered, attack_estimate = self._recover(
            y_meas, attack_class, nn_attack_est_tail, X_pred,
            attack_class_raw=attack_class_raw,
            fm_uncertainty=fm_uncertainty)

        return DetectionResult(
            attack_class=attack_class,
            confidence=confidence,
            y_recovered=y_recovered,
            attack_estimate=attack_estimate,
            features={'step': self._step_count,
                     'dead_reckon': self._dead_reckon_active,
                     'ema_class': ema_class,
                     'ema_conf': ema_conf,
                     'fm_uncertainty': fm_uncertainty}
        )

    def set_control(self, u_cmd: np.ndarray) -> None:
        """记录控制指令，供内部运动学模型使用"""
        self._u_cmd = np.asarray(u_cmd, dtype=float).ravel()

    def set_ekf_state(self, ekf_state: np.ndarray) -> None:
        """接收外部 EKF 估计，供周期性内部状态重校准使用

        在仿真循环中，每次 EKF 更新后调用。
        """
        self._ekf_state_external = np.asarray(ekf_state, dtype=float).ravel()

    def _maybe_recalibrate(self):
        """周期性内部运动学状态重校准

        当满足以下条件时，用外部 EKF 估计重置内部状态以控制漂移:
          - 距上次重校准已超过 RECALIB_INTERVAL 步
          - 分类置信度高 (非 A5)
          - 外部 EKF 状态可用
        """
        if self._ekf_state_external is None:
            return
        if self._step_count - self._last_recalib_step < self.RECALIB_INTERVAL:
            return
        # 仅在高置信度非 A5 时重校准（避免在攻击活跃期用被破坏的 EKF 估计）
        if self._dead_reckon_active:
            return
        ema_class = self.ALL_ATTACK_TYPES[int(np.argmax(self._class_ema))]
        ema_conf = float(np.max(self._class_ema))
        if ema_class == 'A5':
            return
        if ema_conf < self.RECALIB_CONFIDENCE_THRESH and self._step_count > self.window_size:
            return

        self._internal_state = self._ekf_state_external.copy()
        self._last_recalib_step = self._step_count

    def _tail_weighted_average(self, attack_seq_full: np.ndarray) -> np.ndarray:
        """解码器尾部指数加权平均

        用最后 TAIL_AVG_STEPS 步的指数加权平均替代仅取最后一步,
        减少单步估计的随机波动, 提高恢复稳定性。

        Args:
            attack_seq_full: (W, 3) 解码器全窗口输出

        Returns:
            attack_est: (3,) 加权平均攻击估计
        """
        tail = attack_seq_full[-self.TAIL_AVG_STEPS:, :]  # (T, 3)
        # 指数衰减权重: alpha^(T-1-i), 最近步权重最大
        weights = np.array([self.TAIL_ALPHA ** (self.TAIL_AVG_STEPS - 1 - i)
                           for i in range(self.TAIL_AVG_STEPS)])
        weights = weights / weights.sum()
        return (tail * weights[:, np.newaxis]).sum(axis=0)

    def reset(self) -> None:
        """重置所有内部状态"""
        self._internal_state = np.array([0.0, 0.1, 0.0])
        self._u_cmd = np.zeros(2)
        self._step_count = 0
        self._innov_buffer.clear()
        self._ucmd_buffer.clear()
        self._dead_reckon_active = False
        self._class_history.clear()
        self._confidence_history.clear()
        self._innov_ema = np.zeros(3)  # 重置持久偏移 EMA
        self._class_ema = np.zeros(len(self.ALL_ATTACK_TYPES))
        self._a5_consecutive_count = 0
        self._non_a5_consecutive_count = 0
        self._last_recalib_step = 0
        self._ekf_state_external = None
        self._decoder_tail_buffer.clear()

    # ------------------------------------------------------------------
    # 内部运动学模型
    # ------------------------------------------------------------------

    @staticmethod
    def _kinematic_step(state: np.ndarray, u_cmd: np.ndarray) -> np.ndarray:
        """WMR 前端位姿运动学 Euler 积分 (与 model.py 一致)

        dX/dt = F_h(theta) * u
        F_h = [[cos(θ),  -α·sin(θ)],
               [sin(θ),   α·cos(θ)],
               [0,        1        ]]
        """
        v, w = u_cmd[0], u_cmd[1]
        theta = state[2]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = v * cos_t - ALPHA * w * sin_t
        dy = v * sin_t + ALPHA * w * cos_t
        return state + TS * np.array([dx, dy, w])

    def _compute_innovation(self, y_meas: np.ndarray) -> tuple:
        """内部新息: innov = y_meas - X_pred

        内部运动学模型用 u_cmd 做开环预测。
        预测不含攻击信息，新息直接暴露攻击信号。

        Returns:
            X_pred:     运动学预测位姿 (3,) — 同时用于状态更新和 A5 恢复
            innovation: 新息 = y_meas - X_pred (3,)
        """
        X_pred = self._kinematic_step(self._internal_state, self._u_cmd)
        return X_pred, y_meas - X_pred

    # ------------------------------------------------------------------
    # 窗口管理
    # ------------------------------------------------------------------

    def _is_window_ready(self) -> bool:
        return len(self._innov_buffer) >= self.window_size

    def _window_readiness(self) -> int:
        return max(0, self.window_size - len(self._innov_buffer))

    # ------------------------------------------------------------------
    # 模型加载与归一化
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str):
        """加载训练好的 AttackClassifier 或 FreqAwareClassifier

        兼容新旧配置格式:
          - 旧版: freq_extractor_ch / FREQ_EXTRACTOR_CH
          - 新版(v3): freq_channels / FREQ_CHANNELS
          - FM 模型: use_fm, fm_latent_dim, fm_k_steps
        """
        from train_classifier import (AttackClassifier, FreqAwareClassifier,
                                       ENC_CHANNELS, LATENT_DIM,
                                       FREQ_CHANNELS, FM_LATENT_DIM)

        config_path = os.path.join(os.path.dirname(model_path), 'cls_config.npz')
        if os.path.exists(config_path):
            cfg = np.load(config_path, allow_pickle=True)
            in_channels = int(cfg['in_channels'])
            window_size_cfg = int(cfg['window_size'])
            latent_dim = int(cfg['latent_dim'])
            enc_channels = cfg['enc_channels'].tolist() if 'enc_channels' in cfg else ENC_CHANNELS
            dec_channels = cfg['dec_channels'].tolist() if 'dec_channels' in cfg else None
            model_type = str(cfg.get('model_type', 'baseline'))
            # FM 配置 (默认 False — 兼容旧模型)
            use_fm = bool(cfg.get('use_fm', False))
            fm_latent_dim = int(cfg.get('fm_latent_dim', FM_LATENT_DIM))
            # 兼容新旧频率通道配置键名
            if 'freq_channels' in cfg:
                freq_channels = cfg['freq_channels'].tolist()
            elif 'freq_extractor_ch' in cfg:
                freq_channels = cfg['freq_extractor_ch'].tolist()
            else:
                freq_channels = FREQ_CHANNELS
            if window_size_cfg != self.window_size:
                print(f"  [WARN] 模型窗口={window_size_cfg}, 检测器窗口={self.window_size}")
        else:
            in_channels, latent_dim = 5, LATENT_DIM
            enc_channels = ENC_CHANNELS
            dec_channels = None
            freq_channels = FREQ_CHANNELS
            model_type = 'baseline'
            use_fm = False
            fm_latent_dim = FM_LATENT_DIM

        # Fallback: 从权重文件推断 dec_channels (兼容旧模型)
        if model_type == 'baseline' and dec_channels is None:
            try:
                sd = torch.load(model_path, map_location='cpu', weights_only=True)
                dec_keys = sorted([k for k in sd.keys()
                                   if 'dec_blocks' in k and k.endswith('.up.weight')])
                if dec_keys:
                    dec_channels = [sd[k].shape[1] for k in dec_keys]
            except Exception:
                pass

        if model_type == 'freqaware':
            model = FreqAwareClassifier(
                in_channels=in_channels, window_size=self.window_size,
                latent_dim=latent_dim, num_classes=len(self.ALL_ATTACK_TYPES),
                enc_channels=enc_channels,
                freq_channels=freq_channels,
                use_fm=use_fm, fm_latent_dim=fm_latent_dim,
            )
        else:
            model = AttackClassifier(
                in_channels=in_channels, window_size=self.window_size,
                latent_dim=latent_dim, num_classes=len(self.ALL_ATTACK_TYPES),
                enc_channels=enc_channels,
                dec_channels_override=dec_channels
            )
        model.load_state_dict(torch.load(model_path, map_location=self._device,
                                         weights_only=True))
        return model, in_channels, use_fm

    def _load_normalizer(self, norm_path: str):
        """加载归一化参数"""
        if os.path.exists(norm_path):
            data = np.load(norm_path)
            self._feat_median = data['feat_median']
            self._feat_iqr = data['feat_iqr']
            self._cmd_max = data['cmd_max']
        else:
            print(f"  [WARN] 归一化参数未找到: {norm_path}，使用默认值")
            self._feat_median = np.zeros(3)
            self._feat_iqr = np.ones(3) * 0.05
            self._cmd_max = np.array([0.3, 1.76])

    def _normalize(self, innov_window: np.ndarray, ucmd_window: np.ndarray) -> np.ndarray:
        """归一化: innov → RobustScaler, ucmd → /cmd_max → 拼接 (W, 5)"""
        innov_norm = (innov_window - self._feat_median) / np.maximum(self._feat_iqr, 1e-6)
        ucmd_norm = ucmd_window / self._cmd_max
        return np.concatenate([innov_norm, ucmd_norm], axis=1)

    # ------------------------------------------------------------------
    # NN 推理
    # ------------------------------------------------------------------

    def _nn_infer(self, innov_window: np.ndarray, ucmd_window: np.ndarray):
        """单次 NN 前向 — 返回全窗口解码输出 + FM 不确定性

        确定性模型: teacher forcing 直出
        FM 模型: teacher forcing 分类 + FM 多采样集成重建 + 不确定性估计

        Returns:
            attack_class:     分类标签
            confidence:       置信度
            attack_est_last:  (3,) 最后一步攻击估计
            attack_seq_full:  (W, 3) 全窗口解码输出 (确定性=直接, FM=集成均值)
            fm_uncertainty:   float, FM 不确定性 (非 FM 模型 = 0.0)
        """
        x_norm = self._normalize(innov_window, ucmd_window)
        x_tensor = torch.from_numpy(x_norm.astype(np.float32)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            if self._device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits, attack_seq, _ = self._model(x_tensor)
            else:
                logits, attack_seq, _ = self._model(x_tensor)

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_cls = int(probs.argmax())
        confidence = float(probs[pred_cls])
        attack_class = self.ALL_ATTACK_TYPES[pred_cls]
        fm_uncertainty = 0.0

        if self._is_fm_model:
            # FM 多采样: 生成 N 个 decoder latent → 批量 decoder → 集成均值 + 标准差
            cls_probs = F.softmax(logits.float(), dim=1)
            class_embed_t = cls_probs @ self._model.class_embedding

            z_samples = self._model.fm_integrate_multi(
                class_embed_t, k_steps=self._fm_k_steps,
                n_samples=self.FM_N_SAMPLES
            )  # (N, 1, fm_latent_dim)

            # 批量 decoder forward: repeat input N 次
            z_batch = z_samples.squeeze(1)  # (N, fm_latent_dim)
            x_batch = x_tensor.repeat(self.FM_N_SAMPLES, 1, 1)  # (N, 100, 5)
            with torch.no_grad():
                if self._device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        _, atk_batch, _ = self._model(x_batch, z_dec_override=z_batch)
                else:
                    _, atk_batch, _ = self._model(x_batch, z_dec_override=z_batch)

            atk_samples = atk_batch.detach().cpu().numpy()  # (N, 100, 3)
            attack_seq_full = atk_samples.mean(axis=0)  # (W, 3) 集成均值
            attack_seq_std = atk_samples.std(axis=0)     # (W, 3) 逐步标准差
            fm_uncertainty = float(attack_seq_std.mean())  # 标量不确定性
        else:
            attack_seq_full = attack_seq.cpu().numpy()[0]  # (W, 3)

        attack_est_last = attack_seq_full[-1, :]  # (3,)

        return attack_class, confidence, attack_est_last, attack_seq_full, fm_uncertainty

    # ------------------------------------------------------------------
    # 统一恢复
    # ------------------------------------------------------------------

    def _recover(self, y_meas: np.ndarray, attack_class: str,
                 nn_attack_est: np.ndarray,
                 X_pred: np.ndarray = None,
                 attack_class_raw: str = None,
                 fm_uncertainty: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """统一恢复策略 (含 FM 不确定性门控)

        加性攻击 (A0-A4, A6-A8): y_rec = y_meas - â(k)

        非加性攻击 (A5 重放): y_rec = X_pred（运动学死推算）
          - A5 死推算需连续 A5_CONSECUTIVE_REQUIRED 步原始 NN 输出为 A5 才激活
          - 需连续 A5_DEACTIVATE_CONSECUTIVE 步非 A5 才退出
          - 防止 A0→A5 误分类导致的错误死推算

        FM 不确定性门控: 当多采样标准差 > 阈值时，对重建信号施加更保守的混合比例。
        """
        # A5 死推算决策: 连续确认 + 立即退出
        if attack_class_raw is not None:
            if attack_class_raw == 'A5':
                self._a5_consecutive_count += 1
                self._non_a5_consecutive_count = 0
            else:
                self._non_a5_consecutive_count += 1
                self._a5_consecutive_count = 0

        # 激活: 连续 A5_CONSECUTIVE_REQUIRED 步
        if not self._dead_reckon_active:
            if self._a5_consecutive_count >= self.A5_CONSECUTIVE_REQUIRED:
                self._dead_reckon_active = True
        # 退出: 连续非 A5 步
        else:
            if self._non_a5_consecutive_count >= self.A5_DEACTIVATE_CONSECUTIVE:
                self._dead_reckon_active = False

        if self._dead_reckon_active:
            if X_pred is not None:
                return X_pred.copy(), np.zeros(3)
            X_pred = self._kinematic_step(self._internal_state, self._u_cmd)
            return X_pred.copy(), np.zeros(3)
        else:
            # 渐进式置信度门控: 根据 EMA 置信度选择恢复强度
            # 中等置信度(0.6-0.85) → 部分恢复(25%); 高置信度(>0.85) → 全量恢复(50%)
            # 低置信度(<0.6)或 A0 → 直通
            ema_conf = float(np.max(self._class_ema))
            a_est = np.clip(nn_attack_est, -0.5, 0.5)

            # 创新息持久偏移检测: NN 判定 A0 但测量中存在持续非零偏移
            # 此时可能为未检测到的恒定偏置攻击 (A1), 应用弱恢复作为兜底
            innov_offset = np.linalg.norm(self._innov_ema)
            if attack_class == 'A0' and ema_conf < self.RECOVERY_CONFIDENCE_THRESH:
                if innov_offset > self._innov_offset_thresh:
                    # 持久偏移 → 弱补偿: y_rec = y_meas - 0.2 * innov_ema
                    weak_a = np.clip(0.2 * self._innov_ema, -0.3, 0.3)
                    return y_meas - weak_a, weak_a
                return y_meas.copy(), np.zeros(3)
            elif attack_class == 'A0' or ema_conf < self.RECOVERY_CONFIDENCE_THRESH:
                return y_meas.copy(), np.zeros(3)
            elif ema_conf >= self.RECOVERY_CONFIDENCE_FULL:
                blend_alpha = self.RECOVERY_BLEND_ALPHA  # 0.5
            else:
                blend_alpha = self.RECOVERY_BLEND_PARTIAL  # 0.25

            # FM 不确定性门控: 模型内部分歧大 → 加倍保守
            if fm_uncertainty > self.FM_UNCERTAINTY_THRESH:
                blend_alpha *= 0.5
            y_rec = y_meas - blend_alpha * a_est
            return y_rec, blend_alpha * a_est


# ============================================================================
# Oracle 检测器 — 仿真理论性能上界
# ============================================================================

class OracleDetector:
    """Oracle 检测器 — 已知 ground truth 攻击信号

    在仿真中直接访问真实攻击信号 a_true(k)，执行完美信号恢复：
      y_rec = y_meas - a_true(k)

    代表检测器在"完美知道攻击"情况下的理论性能上界。
    """

    def __init__(self, attack_type: str = 'A0', seed: int = 42):
        self.attack_type = attack_type
        self._step_count = 0

    def reset(self):
        self._step_count = 0

    def detect(self, y_meas: np.ndarray,
               a_true: np.ndarray = None) -> DetectionResult:
        """Oracle 检测：减去已知攻击

        Args:
            y_meas: 当前传感器测量值 (3,) — 含攻击
            a_true: 真实攻击信号 (3,) — 仅在仿真中可获得

        Returns:
            DetectionResult (attack_class 正确, confidence=1.0)
        """
        self._step_count += 1
        y_meas = np.asarray(y_meas, dtype=float).ravel()
        if a_true is None:
            a_true = np.zeros(3)
        a_true = np.asarray(a_true, dtype=float).ravel()

        y_rec = y_meas - a_true

        return DetectionResult(
            attack_class=self.attack_type,
            confidence=1.0,
            y_recovered=y_rec,
            attack_estimate=a_true.copy(),
            features={'detector': 'oracle', 'step': self._step_count}
        )


# ============================================================================
# 检测器工厂函数
# ============================================================================

def create_detector(tier: str, attack_type: str = 'A0', seed: int = 42,
                    model_path: str = None, norm_path: str = None):
    """根据 tier 创建对应的检测器实例

    Args:
        tier: 检测器级别
            'none'   — 无检测，y_rec = y_meas
            'nn'     — NNDetector (神经网络分类 + 类型特定恢复)
            'oracle' — OracleDetector, 已知 ground truth (理论上界)
        attack_type: 攻击类型标签 (仅 oracle tier 使用)
        seed:        随机种子
        model_path:  NN 模型权重路径 (仅 nn tier)
        norm_path:   归一化参数路径 (仅 nn tier)

    Returns:
        检测器实例 或 None (tier='none')
    """
    tier = tier.lower()
    if tier == 'none':
        return None
    elif tier == 'nn':
        return NNDetector(model_path=model_path, norm_path=norm_path)
    elif tier == 'oracle':
        return OracleDetector(attack_type=attack_type, seed=seed)
    else:
        raise ValueError(f"Unknown detector tier: {tier}. "
                         f"Choose from: none, nn, oracle")


# ============================================================================
# 模块自测
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("detector.py — 攻击检测器模块自测")
    print("=" * 60)

    # ---- 测试 OracleDetector ----
    print("\n[1] OracleDetector 测试")
    oracle = OracleDetector(attack_type='A4', seed=42)
    a_true = np.array([0.25, -0.15, 0.18])
    result = oracle.detect(y_meas=np.array([1.0, 0.5, 0.3]), a_true=a_true)
    assert result.attack_class == 'A4', f"分类错误: {result.attack_class}"
    assert result.confidence == 1.0, f"置信度错误: {result.confidence}"
    assert np.allclose(result.y_recovered, np.array([0.75, 0.65, 0.12])), \
        f"恢复错误: {result.y_recovered}"
    print(f"  [OK] OracleDetector: {result}")

    # ---- 测试 NNDetector (无 GPU 时用 CPU) ----
    print("\n[2] NNDetector 测试")
    model_path = os.path.join(SCRIPT_DIR, 'models', 'cls_best.pt')
    norm_path = os.path.join(SCRIPT_DIR, 'dataset_win', 'config', 'normalizer.npz')

    if not os.path.exists(model_path):
        print(f"  [SKIP] 模型文件不存在: {model_path}")
        print(f"  请先运行 train_classifier.py 训练模型")
    elif not os.path.exists(norm_path):
        print(f"  [SKIP] 归一化参数不存在: {norm_path}")
        print(f"  请先运行 preprocess_data.py 预处理数据")
    else:
        detector = NNDetector(model_path=model_path, norm_path=norm_path)

        # 窗口填充期测试 — 前 99 步应返回 A0
        print("  测试窗口填充期...")
        for i in range(50):
            y = np.array([0.5, -0.3, 0.1]) + 0.02 * np.random.randn(3)
            detector.set_control(np.array([0.25, 0.1]))
            r = detector.detect(y)
            if i < detector.window_size - 1:
                assert r.confidence == 0.0, f"step {i}: 窗口未就绪应 conf=0"
        print(f"  [OK] 窗口填充期: {detector._window_readiness()} 步后就绪")

        # 继续填充并测试推理
        print("  测试 NN 推理...")
        for i in range(50):
            y = np.array([0.5, -0.3, 0.1]) + 0.15 * np.sin(2 * np.pi * 0.8 * i * TS)
            y += 0.02 * np.random.randn(3)
            detector.set_control(np.array([0.25, 0.1]))
            r = detector.detect(y)
        print(f"  分类: {r.attack_class}, 置信度: {r.confidence:.3f}")
        print(f"  |a_est|: {np.linalg.norm(r.attack_estimate):.4f}")
        print(f"  冻结状态: {r.features.get('frozen', False)}")
        print("  [OK] NN 推理测试通过")

        # 测试 reset
        detector.reset()
        assert detector._step_count == 0
        assert not detector._dead_reckon_active
        print("  [OK] Reset 测试通过")

    # ---- 测试工厂函数 ----
    print("\n[3] create_detector 工厂测试")
    assert create_detector('none') is None, "none tier 应返回 None"
    print("  [OK] none → None")

    if os.path.exists(model_path) and os.path.exists(norm_path):
        det = create_detector('nn', model_path=model_path, norm_path=norm_path)
        assert isinstance(det, NNDetector), f"nn tier 应返回 NNDetector, 实际: {type(det)}"
        print("  [OK] nn → NNDetector")

    det = create_detector('oracle', attack_type='A2')
    assert isinstance(det, OracleDetector), f"oracle tier 应返回 OracleDetector, 实际: {type(det)}"
    print("  [OK] oracle → OracleDetector")

    try:
        create_detector('ewma')
        assert False, "ewma tier 应抛出异常"
    except ValueError:
        print("  [OK] 'ewma' → ValueError (已废弃)")

    print("\n" + "=" * 60)
    print("detector.py self-test PASSED")
    print("=" * 60)
