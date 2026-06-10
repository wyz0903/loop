"""
detector — 传感器攻击检测与信号恢复模块
==========================================
即插即用的攻击检测器，不改动 EKF 或 NMPC。

公共 API:
  NNDetector      — 神经网络攻击检测器
  OracleDetector  — 理想检测器 (理论上界)
  DetectionResult — 检测结果数据结构
  create_detector — 检测器工厂函数
"""

from detector.backend import (NNDetector, OracleDetector,
                               DetectionResult, create_detector)
from detector.cfm_backend import CFMDetectorBackend

__all__ = ['NNDetector', 'OracleDetector', 'CFMDetectorBackend',
           'DetectionResult', 'create_detector']
