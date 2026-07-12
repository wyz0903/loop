"""
detector — 攻击检测器模型定义与训练评估
============================================
纯数据集层面的深度学习流水线: 模型定义、数据预处理、训练、评估。

公共 API:
  Detector — 攻击分类检测器
"""

from detector.classifier import Detector

__all__ = ['Detector']
