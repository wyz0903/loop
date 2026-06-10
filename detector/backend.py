"""
detector/backend.py — 传感器攻击检测与信号恢复模块
=====================================================
即插即用的攻击检测器，不改动 EKF 或 NMPC。

架构位置：
  Sensor → [Attack] → y_meas → [Detector] → y_rec → [EKF] → X_hat → [NMPC]

检测器功能：
  1. 分类攻击类别 (A0~A8)
  2. 恢复干净传感器信号 (攻击移除)

CFMDetector 策略 (主检测器):
  - Transformer 主干 + AdaLN-Zero 流匹配头 + 分类头
  - 条件流匹配 (OT-CFM) 学习攻击信号的条件概率分布
  - ODE 采样从噪声逐步重建攻击信号
  - 物理信息正则化 (PINN): 运动学 ODE 残差约束

Oracle 策略 (理论上界):
  - 已知 ground truth a(k)，完美移除攻击信号

参考文献：
  - Lipman et al. (2023) Flow Matching for Generative Modeling, ICLR
  - Tong et al. (2024) Improving and Generalizing Flow-Based Generative Models
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from collections import deque

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 模块常量
# ============================================================================

STATE_DIM = 3             # 传感器测量维度 [x, y, theta]
TS = 0.05                 # 采样周期 [s]
ALPHA = 0.17              # 前端偏置距离 [m] (与 model.py 同步)


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
            'cfm'    — CFMDetectorBackend (PINN-Flow 流匹配 + Transformer)
            'oracle' — OracleDetector, 已知 ground truth (理论上界)
        attack_type: 攻击类型标签 (仅 oracle tier 使用)
        seed:        随机种子
        model_path:  模型权重路径 (cfm tier)
        norm_path:   归一化参数路径 (cfm tier)

    Returns:
        检测器实例 或 None (tier='none')
    """
    tier = tier.lower()
    if tier == 'none':
        return None
    elif tier == 'cfm':
        from detector.cfm_backend import CFMDetectorBackend  # 惰性导入, 避免循环依赖
        return CFMDetectorBackend(model_path=model_path, norm_path=norm_path)
    elif tier == 'oracle':
        return OracleDetector(attack_type=attack_type, seed=seed)
    else:
        raise ValueError(f"Unknown detector tier: {tier}. "
                         f"Choose from: none, cfm, oracle")
