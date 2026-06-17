"""
cfm_backend.py — 攻击分类检测器推理后端 (编码器-解码器)
===========================================================
推理: 滑动窗口 + 分类 + 物理引导解码器重建 → 恢复路由策略。

恢复策略:
  - A0 (正常) → y_meas 直通
  - 低置信度 → y_meas 直通
  - 检测到攻击 → 解码器重建 y_pred[-1] (物理引导: 运动学递推 + 学习修正)

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
from typing import Dict, Optional

import torch

from model import WMRKinematics
from attack import ALL_ATTACK_TYPES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 模块常量
# ============================================================================

STATE_DIM = 3             # 传感器测量维度 [x, y, theta]
NN_WINDOW_SIZE = 100      # 滑动窗口长度 (与 detector 一致)


# ============================================================================
# 检测结果数据结构
# ============================================================================

@dataclass
class DetectionResult:
    """单步检测器的完整输出

    Attributes:
        attack_class:   攻击类别标签 'A0'~'A7'
        confidence:     分类置信度 [0, 1]
        y_recovered:    恢复后的传感器信号 (3,) — 用作位姿估计
        attack_estimate: 估计的攻击分量 (3,) — 解码器修正量 delta_pred
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
# CFMDetector 推理后端 (编码器-解码器)
# ============================================================================

class CFMDetectorBackend:
    """攻击分类检测器 — 即插即用, 不改动控制系统。

    核心设计:
      - 8 通道输入: [y_meas(3) + innov(3) + u_cmd(2)]
      - 编码器分类 + 解码器重建
      - 检测到攻击时 → 解码器恢复信号 (物理引导: 运动学+学习修正)
    """

    # ---- 后处理 ----
    CONFIDENCE_THRESHOLD = 0.5     # 低于此阈值 → 直通不恢复

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

        # ---- 同步归一化参数到模型 ----
        self._sync_norm_to_model()

        # ---- 解码器可用性 ----
        self._has_decoder = (self._model.use_decoder and
                             self._model.decoder is not None)
        mode_str = "编码器-解码器 (物理引导重建)" if self._has_decoder else "cls-only (死推算回退)"

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
        print(f"  模式: {mode_str}")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def detect(self, y_meas: np.ndarray) -> DetectionResult:
        """主检测接口 — 每步调用一次。

        检测策略:
          1. 内部运动学预测 → 新息
          2. 推入滑动窗口
          3. 编码器+分类头分类 (107K, 每次必跑)
          4. 检测到攻击 → 解码器重建 (+43.6K, 按需)
          5. 正常/低置信度 → 直通; 攻击 → 解码器恢复
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

        # 5. 分类: 编码器+分类头 (107K, 每次必跑)
        t0 = time.time()
        x_tensor = self._build_input_tensor()
        attack_class, confidence, features = self._nn_classify(x_tensor)

        # 6. 按需解码: 仅检测到攻击时运行解码器 (+43.6K)
        y_decoded = None
        delta = np.zeros(3)
        need_reconstruct = (confidence >= self.CONFIDENCE_THRESHOLD
                            and attack_class != 'A0')
        if need_reconstruct:
            y_decoded, delta = self._nn_reconstruct(features, x_tensor)

        self._inference_count += 1
        self._total_inference_time += time.time() - t0

        # 7. 恢复: 正常→直通, 攻击→解码器重建
        y_recovered = self._recover(y_meas, attack_class, confidence,
                                    X_pred, y_decoded)

        # 8. 锚定内部运动学状态到恢复测量
        self._internal_state = y_recovered.copy()

        return DetectionResult(
            attack_class=attack_class, confidence=confidence,
            y_recovered=y_recovered, attack_estimate=delta,
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

    def _compute_innovation(self, y_meas: np.ndarray):
        X_pred = WMRKinematics.kinematic_predict(self._internal_state, self._u_cmd)
        return X_pred, y_meas - X_pred

    def _compute_kinres_window(self, ymeas: np.ndarray, ucmd: np.ndarray) -> np.ndarray:
        """计算窗口锚定运动学残差 (整窗)。

        从窗口第一帧 y_meas[0] 出发, 沿 u_cmd 做 100 步运动学递推,
        与实测值逐帧比较: kin_res = y_meas - y_kin。

        打破非加性攻击的自指涉污染反馈环:
          - 递推仅用 u_cmd (控制指令), 不引入 y_meas[1:] 的污染
          - 累积误差暴露 A4(重放)/A5(丢包)/A7(冻结) 的长期不一致

        Args:
            ymeas: (W, 3) y_meas 窗口
            ucmd:  (W, 2) u_cmd 窗口

        Returns:
            kin_res: (W, 3) 窗口锚定运动学残差
        """
        W = len(ymeas)
        y_kin = np.zeros((W, 3), dtype=np.float32)
        y = ymeas[0].copy()
        y_kin[0] = y
        for k in range(W - 1):
            v, w = ucmd[k, 0], ucmd[k, 1]
            cos_t, sin_t = np.cos(y[2]), np.sin(y[2])
            dx = v * cos_t - 0.17 * w * sin_t
            dy = v * sin_t + 0.17 * w * cos_t
            y = y + 0.05 * np.array([dx, dy, w])
            y_kin[k + 1] = y
        kin_res = ymeas - y_kin
        # theta 包裹到 [-pi, pi]
        kin_res[:, 2] = np.arctan2(np.sin(kin_res[:, 2]), np.cos(kin_res[:, 2]))
        return kin_res

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
        """加载 CFMDetector, 兼容旧版配置。"""
        from detector.cfm_detector import CFMDetector, IN_CHANNELS

        config_path = os.path.join(os.path.dirname(model_path), 'cfm_cls_config.npz')
        cfg = {}
        if os.path.exists(config_path):
            cfg = dict(np.load(config_path, allow_pickle=True))

        # ---- 向后兼容: 旧配置 → 新架构 ----
        backbone_type = str(cfg.get('backbone_type', 'simple_conv'))
        if backbone_type == 'causal_conv':
            print(f"  [WARN] 旧配置 backbone_type='causal_conv' → 回退为 'simple_conv'")
            backbone_type = 'simple_conv'

        in_channels = int(cfg.get('in_channels', IN_CHANNELS))
        if in_channels != IN_CHANNELS:
            print(f"  [WARN] 旧配置 in_channels={in_channels} → 强制使用 {IN_CHANNELS}")
            in_channels = IN_CHANNELS

        use_channel_attn = bool(cfg.get('use_channel_attn', True))
        use_decoder = bool(cfg.get('use_decoder', True))

        model = CFMDetector(
            in_channels=in_channels,
            window_size=int(cfg.get('window_size', self.window_size)),
            d_model=int(cfg.get('d_model', 128)),
            num_classes=int(cfg.get('num_classes', 8)),
            backbone_type=backbone_type,
            use_channel_attn=use_channel_attn,
            use_decoder=use_decoder,
            conv_channels=list(cfg.get('conv_channels', [64, 128, 128])) if 'conv_channels' in cfg else None,
            conv_kernel_size=int(cfg.get('conv_kernel_size', 3)),
            pool_size=int(cfg.get('pool_size', 2)),
            num_transformer_layers=int(cfg.get('num_transformer_layers', 4)),
            num_heads=int(cfg.get('num_heads', 8)),
            dim_feedforward=int(cfg.get('dim_feedforward', 512)),
        )

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self._device,
                                     weights_only=True)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            loaded_keys = len(state_dict) - len(unexpected)
            total_keys = len(state_dict)
            if missing:
                missing_pct = len(missing) / max(total_keys, 1) * 100
                dec_missing = sum(1 for k in missing if 'decoder' in k)
                print(f"  [INFO] 缺少的权重键: {len(missing)} 个 ({missing_pct:.0f}%)"
                      f"{', 解码器: ' + str(dec_missing) + ' 个' if dec_missing else ''}")
                if missing_pct > 50:
                    print(f"  [WARN] 模型权重严重不匹配 — 请重新训练!")
            if unexpected:
                dec_unexpected = sum(1 for k in unexpected if 'decoder' in k)
                print(f"  [INFO] 多余的权重键: {len(unexpected)} 个"
                      f"{', 解码器: ' + str(dec_unexpected) + ' 个' if dec_unexpected else ''}")
        else:
            print(f"  [WARN] 模型未找到: {model_path}, 使用随机初始化权重")

        return model

    def _load_normalizer(self, norm_path: str):
        """加载归一化参数。"""
        if os.path.exists(norm_path):
            data = np.load(norm_path)
            self._ymeas_median = data.get('ymeas_median', np.zeros(3, dtype=np.float32))
            self._ymeas_scale = data.get('ymeas_scale',
                                          np.array([2.5, 2.5, np.pi], dtype=np.float32))
            self._feat_median = data['feat_median']
            if 'innov_scale' in data:
                self._innov_scale = data['innov_scale']
            elif 'feat_iqr' in data:
                self._innov_scale = data['feat_iqr']
            else:
                self._innov_scale = np.array([0.5, 0.5, 0.3], dtype=np.float32)
            # 窗口运动学残差归一化参数 (向后兼容旧 normalizer 无此字段)
            self._kinres_median = data.get('kinres_median', np.zeros(3, dtype=np.float32))
            self._kinres_scale = data.get('kinres_scale',
                                           np.array([2.5, 2.5, np.pi], dtype=np.float32))
            self._cmd_max = data['cmd_max']
        else:
            print(f"  [WARN] 归一化参数未找到: {norm_path}, 使用默认值")
            self._ymeas_median = np.zeros(3, dtype=np.float32)
            self._ymeas_scale = np.array([2.5, 2.5, np.pi], dtype=np.float32)
            self._feat_median = np.zeros(3)
            self._innov_scale = np.array([0.5, 0.5, 0.3], dtype=np.float32)
            self._kinres_median = np.zeros(3, dtype=np.float32)
            self._kinres_scale = np.array([2.5, 2.5, np.pi], dtype=np.float32)
            self._cmd_max = np.array([0.3, 1.76])

    def _sync_norm_to_model(self):
        """将归一化参数同步到模型的 registered buffers。"""
        if hasattr(self._model, 'set_norm_params'):
            self._model.set_norm_params(
                ymeas_scale=self._ymeas_scale,
                ymeas_median=self._ymeas_median,
                cmd_max=self._cmd_max,
            )

    def _normalize(self, ymeas_window: np.ndarray, innov_window: np.ndarray,
                   kinres_window: np.ndarray, ucmd_window: np.ndarray) -> np.ndarray:
        """归一化: y_meas + innov + kin_res + u_cmd → (W, 11)。"""
        ymeas_norm = (ymeas_window - self._ymeas_median) / np.maximum(self._ymeas_scale, 1e-6)
        innov_norm = (innov_window - self._feat_median) / np.maximum(self._innov_scale, 1e-6)
        kinres_norm = (kinres_window - self._kinres_median) / np.maximum(self._kinres_scale, 1e-6)
        ucmd_norm = ucmd_window / self._cmd_max
        return np.concatenate([ymeas_norm, innov_norm, kinres_norm, ucmd_norm], axis=1)

    # ------------------------------------------------------------------
    # NN 推理 (两步: 分类 → 按需解码)
    # ------------------------------------------------------------------

    def _build_input_tensor(self) -> torch.Tensor:
        """从缓冲区构建归一化输入张量 (100, 11)。"""
        ymeas_window = np.array(list(self._ymeas_buffer))
        innov_window = np.array(list(self._innov_buffer))
        ucmd_window = np.array(list(self._ucmd_buffer))
        # 窗口锚定运动学残差: 在线计算 (~0.1ms)
        kinres_window = self._compute_kinres_window(ymeas_window, ucmd_window)
        x_norm = self._normalize(ymeas_window, innov_window, kinres_window, ucmd_window)
        return torch.from_numpy(x_norm.astype(np.float32)).unsqueeze(0).to(self._device)

    def _nn_classify(self, x_tensor: torch.Tensor):
        """仅分类: 编码器 + 分类头 (107K 参数)。

        Returns:
            attack_class, confidence, features
        """
        with torch.no_grad():
            cls_logits, features = self._model(x_tensor, return_recon=False)

        probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
        pred_cls = int(probs.argmax())
        confidence = float(probs[pred_cls])
        attack_class = ALL_ATTACK_TYPES[pred_cls]

        return attack_class, confidence, features

    def _nn_reconstruct(self, features: torch.Tensor, x_tensor: torch.Tensor
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """按需解码: 解码器重建 (仅在检测到攻击时调用, +43.6K 参数)。

        Returns:
            y_decoded: 重建的 y_pred[-1] (3,) 物理单位
            delta:     攻击修正量 delta_pred[-1] (3,) 物理单位
        """
        if not self._has_decoder:
            return None, np.zeros(3)

        with torch.no_grad():
            y_pred, delta_pred = self._model.decode(features, x_tensor)
            y_decoded = y_pred[0, -1, :].cpu().numpy()
            delta = delta_pred[0, -1, :].cpu().numpy()
            y_decoded[2] = np.arctan2(np.sin(y_decoded[2]), np.cos(y_decoded[2]))

        return y_decoded, delta

    # ------------------------------------------------------------------
    # 恢复策略
    # ------------------------------------------------------------------

    def _recover(self, y_meas: np.ndarray, attack_class: str,
                 confidence: float, X_pred: np.ndarray,
                 y_decoded: Optional[np.ndarray] = None) -> np.ndarray:
        """恢复策略。

        - A0 或低置信度 → y_meas 直通
        - 检测到攻击 + 解码器可用 → 解码器重建
        - 检测到攻击 + 无解码器 → 运动学死推算 (回退)
        """
        if confidence < self.CONFIDENCE_THRESHOLD:
            return y_meas.copy()

        if attack_class == 'A0':
            return y_meas.copy()

        if y_decoded is not None:
            return y_decoded.copy()

        return X_pred.copy()
