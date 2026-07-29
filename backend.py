"""
backend.py — 攻击检测器推理后端
================================
推理: 滑动窗口 + 8 类攻击分类 (A0-A7) + 位姿恢复。

5 通道输入: [y_meas(3) + u_cmd(2)]
恢复: y_recovered = y_kin + delta (物理空间)
"""

import os
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Dict

import torch

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

    5 通道输入: [y_meas(3) + u_cmd(2)]
    恢复输出: y_recovered = y_kin + delta (物理空间)
    若模型无 decoder (旧版), y_recovered = y_meas (直通)
    """

    def __init__(self, model_path: str = None, norm_path: str = None,
                 window_size: int = WINDOW_SIZE, device: str = None):
        if model_path is None:
            # 优先恢复模型, 回退到纯分类模型
            recovery_path = os.path.join(SCRIPT_DIR, 'detector', 'models', 'nn_recovery_best.pt')
            cls_path = os.path.join(SCRIPT_DIR, 'detector', 'models', 'nn_cls_best.pt')
            model_path = recovery_path if os.path.exists(recovery_path) else cls_path
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
            cmd_max=self._cmd_max)

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
        attack_class, confidence, y_recovered = self._classify_and_recover(x_tensor)

        if y_recovered is None:
            y_recovered = y_meas.copy()  # 旧模型 fallback: 直通

        attack_estimate = y_meas.copy() - y_recovered if y_recovered is not None else np.zeros(3)

        return DetectionResult(
            attack_class=attack_class, confidence=confidence,
            y_recovered=y_recovered, attack_estimate=attack_estimate,
            features={'step': self._step_count})

    def set_control(self, u_cmd: np.ndarray) -> None:
        self._u_cmd = np.asarray(u_cmd, dtype=float).ravel()

    def reset(self) -> None:
        self._u_cmd = np.zeros(2)
        self._step_count = 0
        self._ymeas_buffer.clear()
        self._ucmd_buffer.clear()

    # ------------------------------------------------------------------
    # 内部: 窗口
    # ------------------------------------------------------------------

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
        self._cmd_max = data['cmd_max']

    # ------------------------------------------------------------------
    # 内部: 归一化
    # ------------------------------------------------------------------

    def _normalize(self, ymeas_window: np.ndarray, ucmd_window: np.ndarray) -> np.ndarray:
        ymeas_norm = (ymeas_window - self._ymeas_median) / np.maximum(self._ymeas_scale, 1e-6)
        ucmd_norm = ucmd_window / self._cmd_max
        return np.concatenate([ymeas_norm, ucmd_norm], axis=1)

    def _build_input_tensor(self, ymeas_window: np.ndarray,
                            ucmd_window: np.ndarray) -> torch.Tensor:
        x_norm = self._normalize(ymeas_window, ucmd_window)
        return torch.from_numpy(x_norm.astype(np.float32)).unsqueeze(0).to(self._device)

    # ------------------------------------------------------------------
    # 内部: 分类 + 恢复
    # ------------------------------------------------------------------

    def _classify_and_recover(self, x_tensor: torch.Tensor):
        """单次前向: 分类 + 恢复 (y_kin 直出, 不做 δ 修正)"""
        with torch.no_grad():
            # 运行运动学层获取 y_kin
            x_perm = x_tensor.permute(0, 2, 1)
            x_11 = self._model.kinematic_layer(x_perm)      # (1, 11, 128)
            y_kin_norm = x_11[:, 5:8, :]                     # (1, 3, 128)
            scale = self._model.ymeas_scale.view(1, 3, 1)
            median = self._model.ymeas_median.view(1, 3, 1)
            y_kin = (y_kin_norm * scale + median).permute(0, 2, 1)  # (1, 128, 3)

            # 分类 (仅作辅助指标)
            features = self._model.backbone(x_11.permute(0, 2, 1))
            cls_logits = self._model.classify(features)

            # 恢复: 直接用 y_kin 末步 (物理前推, 不依赖当前脏测量)
            y_recovered = y_kin[0, -1, :].cpu().numpy()
            y_recovered[2] = np.arctan2(np.sin(y_recovered[2]),
                                         np.cos(y_recovered[2]))

        probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
        pred_cls = int(probs.argmax())
        return ALL_ATTACK_TYPES[pred_cls], float(probs[pred_cls]), y_recovered
