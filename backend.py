"""
backend.py — 攻击检测器推理后端
================================
推理: 滑动窗口 + 8 类攻击分类 (A0-A7)，纯检测，无恢复。
"""

import os
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Dict

import torch

from model import WMRKinematics
from attack import ALL_ATTACK_TYPES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_SIZE = 128


@dataclass
class DetectionResult:
    """单步检测输出"""
    attack_class: str
    confidence: float
    y_recovered: np.ndarray
    attack_estimate: np.ndarray
    features: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"DetectionResult(class={self.attack_class}, "
                f"conf={self.confidence:.3f})")


class DetectorBackend:
    """攻击检测器推理后端 — 对控制系统透明。

    8 通道输入: [y_meas(3) + innov(3) + u_cmd(2)]
    纯检测: 输出攻击类别与置信度，测量值直通。
    """

    def __init__(self, model_path: str = None, norm_path: str = None,
                 window_size: int = WINDOW_SIZE, device: str = None):
        if model_path is None:
            model_path = os.path.join(SCRIPT_DIR, 'detector', 'models', 'nn_cls_best.pt')
        if norm_path is None:
            # 自动解析最新预处理批次 dataset_win/<ts>/normalizer.npz
            win_root = os.path.join(SCRIPT_DIR, 'dataset_win')
            batches = sorted(d for d in os.listdir(win_root)
                             if os.path.exists(os.path.join(win_root, d, 'normalizer.npz')))
            norm_path = os.path.join(win_root, batches[-1], 'normalizer.npz')

        self.window_size = window_size
        self._device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self._model = self._load_model(model_path)
        self._model.to(self._device)
        self._model.eval()

        self._load_normalizer(norm_path)
        self._model.set_norm_params(
            ymeas_scale=self._ymeas_scale, ymeas_median=self._ymeas_median,
            cmd_max=self._cmd_max, feat_median=self._feat_median,
            feat_scale=self._feat_scale)

        self._u_cmd = np.zeros(2)
        self._step_count = 0
        self._ymeas_buffer = deque(maxlen=window_size)
        self._ucmd_buffer = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def detect(self, y_meas: np.ndarray) -> DetectionResult:
        self._step_count += 1
        y_meas = np.asarray(y_meas, dtype=float).ravel()

        self._ymeas_buffer.append(y_meas.copy())
        self._ucmd_buffer.append(self._u_cmd.copy())

        if not self._is_window_ready():
            return DetectionResult(
                attack_class='A0', confidence=0.0,
                y_recovered=y_meas.copy(), attack_estimate=np.zeros(3),
                features={'status': 'window_filling'})

        ymeas_window = np.array(list(self._ymeas_buffer))
        ucmd_window = np.array(list(self._ucmd_buffer))
        x_tensor = self._build_input_tensor(ymeas_window, ucmd_window)
        attack_class, confidence = self._classify(x_tensor)

        return DetectionResult(
            attack_class=attack_class, confidence=confidence,
            y_recovered=y_meas.copy(), attack_estimate=np.zeros(3),
            features={'step': self._step_count})

    def set_control(self, u_cmd: np.ndarray) -> None:
        self._u_cmd = np.asarray(u_cmd, dtype=float).ravel()

    def reset(self) -> None:
        self._u_cmd = np.zeros(2)
        self._step_count = 0
        self._ymeas_buffer.clear()
        self._ucmd_buffer.clear()

    # ------------------------------------------------------------------
    # 内部: 运动学 + 窗口
    # ------------------------------------------------------------------

    def _compute_innovation_window(self, ymeas: np.ndarray, ucmd: np.ndarray) -> np.ndarray:
        """窗口锚定运动学新息: innov[t] = y_meas[t] - rollout(y_meas[0], u_cmd[0:t])[t]"""
        W = len(ymeas)
        y_kin = np.zeros((W, 3), dtype=np.float32)
        y = ymeas[0].copy()
        y_kin[0] = y
        for k in range(W - 1):
            y = WMRKinematics.kinematic_predict(y, ucmd[k])
            y_kin[k + 1] = y
        innov = ymeas - y_kin
        innov[:, 2] = np.arctan2(np.sin(innov[:, 2]), np.cos(innov[:, 2]))
        return innov

    def _is_window_ready(self) -> bool:
        return len(self._ymeas_buffer) >= self.window_size

    # ------------------------------------------------------------------
    # 内部: 模型加载
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str):
        from detector.classifier import Detector

        model = Detector()
        state_dict = torch.load(model_path, map_location=self._device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        return model

    def _load_normalizer(self, norm_path: str):
        data = np.load(norm_path)
        self._ymeas_median = data['ymeas_median']
        self._ymeas_scale = data['ymeas_scale']
        self._feat_median = data['feat_median']
        self._feat_scale = data['feat_scale']
        self._cmd_max = data['cmd_max']

    # ------------------------------------------------------------------
    # 内部: 归一化
    # ------------------------------------------------------------------

    def _normalize(self, ymeas_window: np.ndarray, innov_window: np.ndarray,
                   ucmd_window: np.ndarray) -> np.ndarray:
        ymeas_norm = (ymeas_window - self._ymeas_median) / np.maximum(self._ymeas_scale, 1e-6)
        innov_norm = (innov_window - self._feat_median) / np.maximum(self._feat_scale, 1e-6)
        ucmd_norm = ucmd_window / self._cmd_max
        return np.concatenate([ymeas_norm, innov_norm, ucmd_norm], axis=1)

    def _build_input_tensor(self, ymeas_window: np.ndarray,
                            ucmd_window: np.ndarray) -> torch.Tensor:
        innov_window = self._compute_innovation_window(ymeas_window, ucmd_window)
        x_norm = self._normalize(ymeas_window, innov_window, ucmd_window)
        return torch.from_numpy(x_norm.astype(np.float32)).unsqueeze(0).to(self._device)

    # ------------------------------------------------------------------
    # 内部: 分类
    # ------------------------------------------------------------------

    def _classify(self, x_tensor: torch.Tensor):
        with torch.no_grad():
            cls_logits, _ = self._model(x_tensor, return_recon=False)
        probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
        pred_cls = int(probs.argmax())
        return ALL_ATTACK_TYPES[pred_cls], float(probs[pred_cls])

