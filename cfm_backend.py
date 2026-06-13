"""
cfm_backend.py — 攻击分类检测器推理后端 (cls-only 分支)
===========================================================
精简推理: 滑动窗口 + 分类 → 根据检测结果路由恢复策略。

恢复策略:
  - A0 (正常) → y_meas 直通
  - A5 (重放) → 运动学死推算
  - 其他攻击 → 运动学死推算 (cls-only 无信号重建能力)
  - 低置信度 → y_meas 直通

公用 API:
  - CFMDetectorBackend — 推理后端主类
  - DetectionResult — 检测结果数据结构
  - detect(y_meas) → DetectionResult
  - set_control(u_cmd)
  - reset()
"""

import os
import time
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Dict

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 模块常量
# ============================================================================

STATE_DIM = 3             # 传感器测量维度 [x, y, theta]
TS = 0.05                 # 采样周期 [s]
ALPHA = 0.17              # 前端偏置距离 [m]
NN_WINDOW_SIZE = 100      # 神经网络输入窗口大小 [步]


# ============================================================================
# 检测结果数据结构
# ============================================================================

@dataclass
class DetectionResult:
    """单步检测器的完整输出

    Attributes:
        attack_class:   攻击类别标签 'A0'~'A8'
        confidence:     分类置信度 [0, 1]
        y_recovered:    恢复后的传感器信号 (3,) — 用作位姿估计
        attack_estimate: 估计的攻击分量 (3,) — cls-only 分支为零向量
        features:        附加信息字典
    """
    attack_class: str
    confidence: float
    y_recovered: np.ndarray
    attack_estimate: np.ndarray
    features: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"DetectionResult(class={self.attack_class}, "
                f"conf={self.confidence:.3f})")


# ============================================================================
# CFMDetector 推理后端 (cls-only)
# ============================================================================

class CFMDetectorBackend:
    """攻击分类检测器 — 即插即用, 不改动控制系统 (cls-only 分支)。

    核心设计:
      - 8 通道输入: [y_meas(3) + innov(3) + u_cmd(2)]
      - 仅分类, 无信号重建
      - 检测到攻击时 → 运动学死推算作为位姿估计
    """

    ALL_ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']

    # ---- 后处理 ----
    CONFIDENCE_THRESHOLD = 0.5     # 低于此阈值 → 直通不恢复
    A5_DEAD_RECKON = True          # A5 重放攻击 → 运动学死推算

    def __init__(self, model_path: str = None, norm_path: str = None,
                 window_size: int = NN_WINDOW_SIZE, device: str = None):
        if model_path is None:
            model_path = os.path.join(SCRIPT_DIR, 'detector', 'models', 'cfm_cls_best.pt')
        if norm_path is None:
            norm_path = os.path.join(SCRIPT_DIR, 'dataset_win', 'normalizer.npz')

        self.window_size = window_size

        # ---- 设备 ----
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

        # ---- 加载模型 ----
        self._model = self._load_model(model_path)
        self._model.to(self._device)
        self._model.eval()

        # ---- 加载归一化参数 ----
        self._load_normalizer(norm_path)

        # ---- 内部运动学状态 ----
        self._internal_state = np.array([0.0, 0.1, 0.0])
        self._u_cmd = np.zeros(2)
        self._step_count = 0

        # ---- 滑动窗口缓冲区 ----
        self._ymeas_buffer = deque(maxlen=window_size)
        self._innov_buffer = deque(maxlen=window_size)
        self._ucmd_buffer = deque(maxlen=window_size)

        # ---- 统计分析 ----
        self._inference_count = 0
        self._total_inference_time = 0.0

        print(f"[CFMDetector] 模型已加载: {model_path}")
        print(f"  设备: {self._device}, 窗口: {window_size}")
        print(f"  模式: cls-only (分类 + 死推算回退)")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def detect(self, y_meas: np.ndarray) -> DetectionResult:
        """主检测接口 — 每步调用一次。

        检测策略:
          1. 内部运动学预测 → 新息
          2. 推入滑动窗口 (y_meas, innov, u_cmd)
          3. 窗口就绪后 → NN 分类
          4. A0 或低置信度 → 直通; 否则 → 死推算
        """
        self._step_count += 1
        y_meas = np.asarray(y_meas, dtype=float).ravel()

        # 1. 内部运动学预测 → 新息
        X_pred, innovation = self._compute_innovation(y_meas)

        # 2. 推入缓冲区
        self._ymeas_buffer.append(y_meas.copy())
        self._innov_buffer.append(innovation)
        self._ucmd_buffer.append(self._u_cmd.copy())

        # 3. 更新内部运动学状态为当前测量 (锚定)
        self._internal_state = y_meas.copy()

        # 4. 窗口未就绪 → 直通
        if not self._is_window_ready():
            return DetectionResult(
                attack_class='A0', confidence=0.0,
                y_recovered=y_meas.copy(), attack_estimate=np.zeros(3),
                features={'status': 'window_filling',
                          'readiness': self._window_readiness()})

        # 5. NN 分类
        t0 = time.time()
        attack_class, confidence = self._nn_infer()
        self._inference_count += 1
        self._total_inference_time += time.time() - t0

        # 6. 恢复: A0/低置信度 → 直通; 其他 → 死推算
        y_recovered = self._recover(y_meas, attack_class, confidence, X_pred)

        # 7. 锚定内部运动学状态到恢复测量
        self._internal_state = y_recovered.copy()

        return DetectionResult(
            attack_class=attack_class, confidence=confidence,
            y_recovered=y_recovered, attack_estimate=np.zeros(3),
            features={'step': self._step_count})

    def set_control(self, u_cmd: np.ndarray) -> None:
        self._u_cmd = np.asarray(u_cmd, dtype=float).ravel()

    def reset(self) -> None:
        self._internal_state = np.array([0.0, 0.1, 0.0])
        self._u_cmd = np.zeros(2)
        self._step_count = 0
        self._ymeas_buffer.clear()
        self._innov_buffer.clear()
        self._ucmd_buffer.clear()
        self._inference_count = 0
        self._total_inference_time = 0.0

    # ------------------------------------------------------------------
    # 内部运动学模型
    # ------------------------------------------------------------------

    @staticmethod
    def _kinematic_step(state: np.ndarray, u_cmd: np.ndarray) -> np.ndarray:
        """WMR 前端位姿运动学 Euler 积分。"""
        v, w = u_cmd[0], u_cmd[1]
        theta = state[2]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = v * cos_t - ALPHA * w * sin_t
        dy = v * sin_t + ALPHA * w * cos_t
        return state + TS * np.array([dx, dy, w])

    def _compute_innovation(self, y_meas: np.ndarray):
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
        """加载 CFMDetector (cls-only 兼容)。"""
        from detector.cfm_detector import CFMDetector

        config_path = os.path.join(os.path.dirname(model_path), 'cfm_cls_config.npz')
        cfg = {}
        if os.path.exists(config_path):
            cfg = dict(np.load(config_path, allow_pickle=True))

        model = CFMDetector(
            in_channels=int(cfg.get('in_channels', 8)),
            window_size=int(cfg.get('window_size', self.window_size)),
            d_model=int(cfg.get('d_model', 128)),
            num_classes=int(cfg.get('num_classes', 9)),
            backbone_type=str(cfg.get('backbone_type', 'causal_conv')),
            dilations=list(cfg.get('dilations', [1, 2, 4, 8, 16, 32])) if 'dilations' in cfg else None,
            conv_kernel_size=int(cfg.get('conv_kernel_size', 3)),
            num_transformer_layers=int(cfg.get('num_transformer_layers', 4)),
            num_heads=int(cfg.get('num_heads', 8)),
            dim_feedforward=int(cfg.get('dim_feedforward', 512)),
        )

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self._device,
                                     weights_only=True)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  [INFO] 缺少的权重键: {len(missing)} 个")
            if unexpected:
                print(f"  [INFO] 多余的权重键: {len(unexpected)} 个")
        else:
            print(f"  [WARN] 模型未找到: {model_path}, 使用随机初始化权重")

        return model

    def _load_normalizer(self, norm_path: str):
        """加载归一化参数。"""
        if os.path.exists(norm_path):
            data = np.load(norm_path)
            # y_meas 参数
            self._ymeas_median = data.get('ymeas_median', np.zeros(3, dtype=np.float32))
            self._ymeas_scale = data.get('ymeas_scale',
                                          np.array([2.5, 2.5, np.pi], dtype=np.float32))
            # 创新参数
            self._feat_median = data['feat_median']
            if 'innov_scale' in data:
                self._innov_scale = data['innov_scale']
            elif 'feat_iqr' in data:
                self._innov_scale = data['feat_iqr']
            else:
                self._innov_scale = np.array([0.5, 0.5, 0.3], dtype=np.float32)
            # u_cmd 参数
            self._cmd_max = data['cmd_max']
        else:
            print(f"  [WARN] 归一化参数未找到: {norm_path}, 使用默认值")
            self._ymeas_median = np.zeros(3, dtype=np.float32)
            self._ymeas_scale = np.array([2.5, 2.5, np.pi], dtype=np.float32)
            self._feat_median = np.zeros(3)
            self._innov_scale = np.array([0.5, 0.5, 0.3], dtype=np.float32)
            self._cmd_max = np.array([0.3, 1.76])

    def _normalize(self, ymeas_window: np.ndarray, innov_window: np.ndarray,
                   ucmd_window: np.ndarray) -> np.ndarray:
        """归一化: y_meas + innov + u_cmd → (W, 8)。"""
        ymeas_norm = (ymeas_window - self._ymeas_median) / np.maximum(self._ymeas_scale, 1e-6)
        innov_norm = (innov_window - self._feat_median) / np.maximum(self._innov_scale, 1e-6)
        ucmd_norm = ucmd_window / self._cmd_max
        return np.concatenate([ymeas_norm, innov_norm, ucmd_norm], axis=1)

    # ------------------------------------------------------------------
    # NN 推理
    # ------------------------------------------------------------------

    def _nn_infer(self):
        """单次 NN 前向 — 仅分类。

        Returns:
            attack_class:  分类标签 'A0'~'A8'
            confidence:    分类置信度 [0, 1]
        """
        ymeas_window = np.array(list(self._ymeas_buffer))
        innov_window = np.array(list(self._innov_buffer))
        ucmd_window = np.array(list(self._ucmd_buffer))
        x_norm = self._normalize(ymeas_window, innov_window, ucmd_window)  # (100, 8)
        x_tensor = torch.from_numpy(x_norm.astype(np.float32)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            cls_logits, _ = self._model(x_tensor)

        probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
        pred_cls = int(probs.argmax())
        confidence = float(probs[pred_cls])
        attack_class = self.ALL_ATTACK_TYPES[pred_cls]

        return attack_class, confidence

    # ------------------------------------------------------------------
    # 恢复策略
    # ------------------------------------------------------------------

    def _recover(self, y_meas: np.ndarray, attack_class: str,
                 confidence: float, X_pred: np.ndarray) -> np.ndarray:
        """恢复策略 (cls-only: 无信号重建, 仅路由)。

        - A0 或低置信度 → y_meas 直通
        - 其他 (检测到攻击) → 运动学死推算
        """
        # 低置信度 → 直通
        if confidence < self.CONFIDENCE_THRESHOLD:
            return y_meas.copy()

        # A0 正常 → 直通
        if attack_class == 'A0':
            return y_meas.copy()

        # 检测到攻击 → 运动学死推算 (cls-only 无信号重建能力)
        return X_pred.copy()
