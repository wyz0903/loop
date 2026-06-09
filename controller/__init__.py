"""
controller — NMPC 轨迹跟踪控制器
===================================
基于 CasADi Opti 构建非线性模型预测控制器。

公共 API:
  NMPCController — 固定 NMPC 跟踪控制器
  NMPCParams     — 控制器超参数
"""

from controller.controller import NMPCController, NMPCParams

__all__ = ['NMPCController', 'NMPCParams']
