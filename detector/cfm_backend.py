"""
detector/cfm_backend.py — CFMDetector 推理后端
================================================
即插即用检测器, 使用 PINN-Flow 模型替代 NNDetector。

后处理策略 (仅 2 项, 对比当前 8 项):
  1. ODE 求解器 (Euler 10步) — 模型推理的组成部分
  2. 置信度阈值: confidence < 0.5 → 直通不恢复

  移除的机制: EMA, 多数投票, 尾部加权平均, A5 迟滞计数器,
  创新息持久偏移检测, 周期性内部状态重校准, 输出裁剪.

公用 API 与 NNDetector 完全兼容:
  - detect(y_meas) → DetectionResult
  - set_control(u_cmd)
  - set_ekf_state(ekf_state)
  - reset()
"""

import os
import sys
import numpy as np
from collections import deque

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 向后兼容导入
from detector.backend import DetectionResult, STATE_DIM, TS, ALPHA, NN_WINDOW_SIZE


# ============================================================================
# CFMDetector 推理后端
# ============================================================================

class CFMDetectorBackend:
    """PINN-Flow 攻击检测器 — 即插即用, 不改动控制系统。

    相较于 NNDetector 的核心差异:
      - 使用单一 Transformer 主干 + 流匹配生成 (无攻击类型特定模块)
      - 仅 2 项后处理: ODE 求解 + 置信度阈值
      - 统一恢复: y_rec = y_meas − â  (A5 除外, 使用死推算)
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
            model_path = os.path.join(SCRIPT_DIR, '..', 'models', 'cfm_cls_best.pt')
        if norm_path is None:
            norm_path = os.path.join(SCRIPT_DIR, '..', 'dataset_win', 'config', 'normalizer.npz')

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
        import time as _time
        t0 = _time.time()
        attack_class, confidence, nn_attack_est = self._nn_infer()
        self._inference_count += 1
        self._total_inference_time += _time.time() - t0

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
        """接收外部 EKF 估计 (保留以备将来使用, 当前不使用周期性重校准)。"""
        pass

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
        """加载训练好的 CFMDetector。"""
        from detector.cfm_detector import CFMDetector

        # 尝试加载配置
        config_path = os.path.join(os.path.dirname(model_path), 'cfm_cls_config.npz')
        cfg = {}
        if os.path.exists(config_path):
            cfg = dict(np.load(config_path, allow_pickle=True))

        model = CFMDetector(
            in_channels=int(cfg.get('in_channels', 5)),
            window_size=int(cfg.get('window_size', self.window_size)),
            d_model=int(cfg.get('d_model', 128)),
            num_classes=int(cfg.get('num_classes', 9)),
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


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CFMDetectorBackend 自测")
    print("=" * 60)

    # 使用随机权重测试 (无已训练模型)
    # 创建临时模型权重
    import tempfile
    from detector.cfm_detector import CFMDetector

    model = CFMDetector()
    tmp_dir = tempfile.mkdtemp()
    model_path = os.path.join(tmp_dir, 'cfm_cls_best.pt')
    config_path = os.path.join(tmp_dir, 'cfm_cls_config.npz')
    torch.save(model.state_dict(), model_path)
    np.savez(config_path,
             in_channels=5, window_size=100, d_model=128,
             num_classes=9, num_transformer_layers=4, num_heads=8,
             dim_feedforward=512, num_flow_blocks=4,
             dim_feedforward_flow=192, dropout=0.1,
             model_type='cfm')

    # 需要 normalizer.npz — 使用 dataset_win 路径或创建虚拟
    import tempfile
    norm_path = os.path.join(tmp_dir, 'normalizer.npz')
    np.savez(norm_path, feat_median=np.zeros(3), feat_iqr=np.ones(3)*0.05,
             cmd_max=np.array([0.3, 1.76]))

    detector = CFMDetectorBackend(
        model_path=model_path, norm_path=norm_path, device='cpu')

    detector.set_control(np.array([0.1, 0.0]))

    # 模拟传感器数据
    y_meas = np.array([0.15, 0.22, 0.05])

    # 窗口未就绪时 → 直通
    result = detector.detect(y_meas)
    print(f"窗口未就绪: class={result.attack_class}, "
          f"conf={result.confidence:.3f}, readiness={result.features.get('readiness', '?')}")

    # 填充窗口
    for _ in range(detector.window_size + 5):
        y_meas += np.array([0.01, 0.008, 0.002]) * np.random.randn(3)
        result = detector.detect(y_meas)
        detector.set_control(np.array([0.1, 0.05 * np.sin(_ * 0.05)]))

    print(f"窗口就绪: class={result.attack_class}, "
          f"conf={result.confidence:.3f}, "
          f"|a_est|={np.linalg.norm(result.attack_estimate):.4f}")

    # 清洗
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\n自测通过!")
