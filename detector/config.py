"""
detector/config.py — 检测器模块共享常量
==========================================
模型架构常量、攻击类型列表等。
"""

# 攻击类型
ALL_ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']

# 编码器通道配置
ENC_CHANNELS = [48, 96, 192, 256]

# 潜在空间维度
LATENT_DIM = 256

# FreqAware 频率路径通道
FREQ_CHANNELS = [24, 32]

# FiLM 分类嵌入维度
CLASS_EMBED_DIM = 48

