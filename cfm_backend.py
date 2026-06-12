"""
cfm_backend.py — CFMDetector 推理后端
======================================
即插即用检测器, 使用 PINN-Flow 条件流匹配模型。

后处理策略 (精简高效):
  1. ODE 求解器 (Euler 10步) — 模型推理的组成部分
  2. 置信度阈值: confidence < 0.5 → 直通不恢复

公用 API:
  - CFMDetectorBackend — 推理后端主类
  - DetectionResult — 检测结果数据结构
  - detect(y_meas) → DetectionResult
  - set_control(u_cmd)
  - set_ekf_state(ekf_state)
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
ALPHA = 0.17              # 前端偏置距离 [m] (与 model.py 同步)
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
# CFMDetector 推理后端
# ============================================================================

class CFMDetectorBackend:
    """PINN-Flow 攻击检测器 — 即插即用, 不改动控制系统。

    核心设计:
      - 单一 Transformer 主干 + AdaLN-Zero 流匹配生成
      - 仅 2 项后处理: ODE 求解 + 置信度阈值
      - 统一恢复: y_rec = y_meas − â
    """

    ALL_ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']

    # ---- 推理配置 ----
    ODE_STEPS = 10                 # Euler 积分步数

    # ---- 后处理 (仅 2 项) ----
    CONFIDENCE_THRESHOLD = 0.5     # 低于此阈值 → 直通不恢复
    A5_DEAD_RECKON = True          # A5 重放攻击 → 运动学死推算

    def __init__(self, model_path: str = None, norm_path: str = None,
                 window_size: int = NN_WINDOW_SIZE, device: str = None,
                 ode_steps: int = None):
        """
        Args:
            model_path:  CFMDetector 权重路径
            norm_path:   normalizer.npz 路径
            window_size: NN 输入窗口 (默认 100)
            device:      推理设备
            ode_steps:   ODE 求解器步数 (默认 10)
        """
        if model_path is None:
            model_path = os.path.join(SCRIPT_DIR, 'detector', 'models', 'cfm_cls_best.pt')
        if norm_path is None:
            norm_path = os.path.join(SCRIPT_DIR, 'dataset_win', 'normalizer.npz')

        self.window_size = window_size
        if ode_steps is not None:
            self.ODE_STEPS = ode_steps

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
        self._internal_state = np.array([0.0, 0.1, 0.0])  # [x, y, θ]
        self._u_cmd = np.zeros(2)
        self._step_count = 0

        # ---- 滑动窗口缓冲区 ----
        self._innov_buffer = deque(maxlen=window_size)
        self._ucmd_buffer = deque(maxlen=window_size)

        # ---- 统计分析 ----
        self._inference_count = 0
        self._total_inference_time = 0.0

        print(f"[CFMDetector] 模型已加载: {model_path}")
        print(f"  设备: {self._device}, 窗口: {window_size}, "
              f"ODE 步数: {self.ODE_STEPS}")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def detect(self, y_meas: np.ndarray) -> DetectionResult:
        """主检测接口 — 每步调用一次。

        检测策略:
          1. 内部运动学预测 → 新息
          2. 推入滑动窗口
          3. 窗口就绪后 → NN 推理 (分类 + ODE 生成攻击估计)
          4. 置信度 > 阈值 → 减法恢复; 否则 → 直通
          5. A5 → 死推算

        Args:
            y_meas: 当前传感器测量值 [x, y, θ] (3,)

        Returns:
            DetectionResult
        """
        self._step_count += 1
        y_meas = np.asarray(y_meas, dtype=float).ravel()

        # 1. 内部运动学预测 → 新息
        X_pred, innovation = self._compute_innovation(y_meas)

        # 2. 推入缓冲区
        self._innov_buffer.append(innovation)
        self._ucmd_buffer.append(self._u_cmd.copy())

        # 3. 更新内部运动学状态
        self._internal_state = X_pred.copy()

        # 4. 窗口未就绪 → 直通
        if not self._is_window_ready():
            return DetectionResult(
                attack_class='A0',
                confidence=0.0,
                y_recovered=y_meas.copy(),
                attack_estimate=np.zeros(3),
                features={'status': 'window_filling',
                          'readiness': self._window_readiness()}
            )

        # 5. NN 推理
        t0 = time.time()
        attack_class, confidence, nn_attack_est = self._nn_infer()
        self._inference_count += 1
        self._total_inference_time += time.time() - t0

        # 6. 恢复
        y_recovered, attack_estimate = self._recover(
            y_meas, attack_class, confidence, nn_attack_est, X_pred)

        return DetectionResult(
            attack_class=attack_class,
            confidence=confidence,
            y_recovered=y_recovered,
            attack_estimate=attack_estimate,
            features={'step': self._step_count,
                       'ode_steps': self.ODE_STEPS}
        )

    def set_control(self, u_cmd: np.ndarray) -> None:
        """记录控制指令, 供内部运动学模型使用。"""
        self._u_cmd = np.asarray(u_cmd, dtype=float).ravel()

    def set_ekf_state(self, ekf_state: np.ndarray) -> None:
        """接收 EKF 后验估计 Upsilon_hat，锚定内部运动学状态。

        EKF 接收的是检测器恢复后的信号 y_rec，其估计始终是全系统对真实位姿
        的最优估计。每步锚定可彻底切断开环 Euler 积分的误差累积。
        """
        self._internal_state = np.asarray(ekf_state, dtype=float).ravel().copy()

    def reset(self) -> None:
        """重置所有内部状态。"""
        self._internal_state = np.array([0.0, 0.1, 0.0])
        self._u_cmd = np.zeros(2)
        self._step_count = 0
        self._innov_buffer.clear()
        self._ucmd_buffer.clear()
        self._inference_count = 0
        self._total_inference_time = 0.0

    # ------------------------------------------------------------------
    # 内部运动学模型
    # ------------------------------------------------------------------

    @staticmethod
    def _kinematic_step(state: np.ndarray, u_cmd: np.ndarray) -> np.ndarray:
        """WMR 前端位姿运动学 Euler 积分 (与 model.py 严格一致)。

        dX/dt = F_h(θ) · u
        其中 F_h = [[cos(θ),  -α·sin(θ)],
                   [sin(θ),   α·cos(θ)],
                   [0,        1        ]]
        """
        v, w = u_cmd[0], u_cmd[1]
        theta = state[2]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = v * cos_t - ALPHA * w * sin_t
        dy = v * sin_t + ALPHA * w * cos_t
        return state + TS * np.array([dx, dy, w])

    def _compute_innovation(self, y_meas: np.ndarray):
        """内部新息: innov = y_meas − X_pred。

        内部运动学模型用 u_cmd 做开环预测, 新息直接暴露攻击信号。
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
        """加载训练好的 CFMDetector (v1/v2 兼容)。"""
        from detector.cfm_detector import CFMDetector

        # 尝试加载配置
        config_path = os.path.join(os.path.dirname(model_path), 'cfm_cls_config.npz')
        cfg = {}
        if os.path.exists(config_path):
            cfg = dict(np.load(config_path, allow_pickle=True))

        # 检测模型版本
        model_type = str(cfg.get('model_type', 'cfm'))
        if model_type == 'cfm':
            # v1 旧模型: Transformer 骨干, 无正交分裂器
            model = CFMDetector(
                in_channels=int(cfg.get('in_channels', 5)),
                window_size=int(cfg.get('window_size', self.window_size)),
                d_model=int(cfg.get('d_model', 128)),
                num_classes=int(cfg.get('num_classes', 9)),
                backbone_type='transformer',
                d_cls=int(cfg.get('d_model', 128)),   # 无分裂 → 恒等
                d_fm=int(cfg.get('d_model', 128)),
                num_transformer_layers=int(cfg.get('num_transformer_layers', 4)),
                num_heads=int(cfg.get('num_heads', 8)),
                dim_feedforward=int(cfg.get('dim_feedforward', 512)),
                num_flow_blocks=int(cfg.get('num_flow_blocks', 4)),
                dim_feedforward_flow=int(cfg.get('dim_feedforward_flow', 192)),
                dropout=float(cfg.get('dropout', 0.1)),
            )
        else:
            # v2 新模型: 从配置读取骨干和子空间参数
            backbone_type = str(cfg.get('backbone_type', 'causal_conv'))
            dilations_raw = cfg.get('dilations', None)
            dilations = list(dilations_raw) if dilations_raw is not None else None
            model = CFMDetector(
                in_channels=int(cfg.get('in_channels', 5)),
                window_size=int(cfg.get('window_size', self.window_size)),
                d_model=int(cfg.get('d_model', 128)),
                num_classes=int(cfg.get('num_classes', 9)),
                backbone_type=backbone_type,
                dilations=dilations,
                conv_kernel_size=int(cfg.get('conv_kernel_size', 3)),
                d_cls=int(cfg.get('d_cls', 64)),
                d_fm=int(cfg.get('d_fm', 64)),
                num_transformer_layers=int(cfg.get('num_transformer_layers', 4)),
                num_heads=int(cfg.get('num_heads', 8)),
                dim_feedforward=int(cfg.get('dim_feedforward', 512)),
                num_flow_blocks=int(cfg.get('num_flow_blocks', 4)),
                dim_feedforward_flow=int(cfg.get('dim_feedforward_flow', 192)),
                dropout=float(cfg.get('dropout', 0.1)),
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
        """加载 RobustNormalizer 参数。"""
        if os.path.exists(norm_path):
            data = np.load(norm_path)
            self._feat_median = data['feat_median']
            self._feat_iqr = data['feat_iqr']
            self._cmd_max = data['cmd_max']
        else:
            print(f"  [WARN] 归一化参数未找到: {norm_path}, 使用默认值")
            self._feat_median = np.zeros(3)
            self._feat_iqr = np.ones(3) * 0.05
            self._cmd_max = np.array([0.3, 1.76])

    def _normalize(self, innov_window: np.ndarray, ucmd_window: np.ndarray) -> np.ndarray:
        """归一化: innov → RobustScaler, ucmd → /cmd_max → 拼接 (W, 5)。"""
        innov_norm = (innov_window - self._feat_median) / np.maximum(self._feat_iqr, 1e-6)
        ucmd_norm = ucmd_window / self._cmd_max
        return np.concatenate([innov_norm, ucmd_norm], axis=1)

    # ------------------------------------------------------------------
    # NN 推理
    # ------------------------------------------------------------------

    def _nn_infer(self):
        """单次 NN 前向 — ODE 生成攻击估计。

        Returns:
            attack_class:  分类标签 'A0'~'A8'
            confidence:    分类置信度 [0, 1]
            attack_est:    (3,) 攻击估计 â (窗口最后一步)
        """
        innov_window = np.array(list(self._innov_buffer))
        ucmd_window = np.array(list(self._ucmd_buffer))
        x_norm = self._normalize(innov_window, ucmd_window)
        x_tensor = torch.from_numpy(x_norm.astype(np.float32)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            if self._device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    a_hat = self._model.sample_ode(x_tensor, n_steps=self.ODE_STEPS)
                    features = self._model.encode(x_tensor)
                    cls_logits = self._model.classify(features)
            else:
                a_hat = self._model.sample_ode(x_tensor, n_steps=self.ODE_STEPS)
                features = self._model.encode(x_tensor)
                cls_logits = self._model.classify(features)

        probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
        pred_cls = int(probs.argmax())
        confidence = float(probs[pred_cls])
        attack_class = self.ALL_ATTACK_TYPES[pred_cls]

        attack_seq = a_hat.cpu().numpy()[0]  # (W, 3)
        attack_est = attack_seq[-1, :]       # (3,) — 窗口最后一步

        return attack_class, confidence, attack_est

    # ------------------------------------------------------------------
    # 统一恢复
    # ------------------------------------------------------------------

    def _recover(self, y_meas: np.ndarray, attack_class: str,
                 confidence: float, nn_attack_est: np.ndarray,
                 X_pred: np.ndarray) -> tuple:
        """统一恢复策略。

        后处理规则 (仅 2 条):
          1. 置信度 < 阈值 → 直通
          2. A5 (重放, 非加性) → 死推算

        其余所有攻击类型: y_rec = y_meas − â
        (无混合系数, 无分类特定路由, 无削波, 无偏移补偿)
        """
        # 规则 1: 低置信度 → 直通
        if confidence < self.CONFIDENCE_THRESHOLD:
            return y_meas.copy(), np.zeros(3)

        # 规则 2: A5 重放攻击 → 死推算 (非加性攻击)
        if self.A5_DEAD_RECKON and attack_class == 'A5':
            return X_pred.copy(), np.zeros(3)

        # 默认: 减法恢复 (统一处理 A0-A4, A6-A8)
        y_rec = y_meas - nn_attack_est
        return y_rec, nn_attack_est.copy()
