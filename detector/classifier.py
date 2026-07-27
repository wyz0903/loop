"""
detector/classifier.py — 攻击检测模型 (PINN: 可微分运动学 + 膨胀深度可分离卷积 + 注意力池化)
==================================================================================
5 通道外部输入: [y_meas(3) + u_cmd(2)]
网络内部通过可微分运动学层扩展为 11 通道:
  [y_meas(3) + u_cmd(2) + y_kin(3) + innov(3)]
其中 y_kin = RK4(y_meas[0], u_cmd), innov = y_meas - y_kin

输出: cls_logits (B,8)

架构: 可微分运动学层 → MultiScaleDSConvBackbone → 注意力池化 → 分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

D_MODEL = 96
NUM_CLASSES = 8
RAW_CHANNELS = 5           # 外部输入: [y_meas(3), u_cmd(2)]
KIN_CHANNELS = 6           # PINN 运动学特征: [y_kin(3), innov(3)]
IN_CHANNELS = RAW_CHANNELS + KIN_CHANNELS  # = 11
WINDOW_SIZE = 128
CONV_CHANNELS = [32, 64, 96]
CONV_KERNEL_SIZES = [7, 5, 3]
CONV_DILATIONS = [1, 3, 9]
POOL_SIZE = 2

TS = 0.05
ALPHA = 0.17


# ============================================================================
# 可微分运动学层 (PINN 物理先验)
# ============================================================================

class KinematicFeatureLayer(nn.Module):
    """可微分运动学特征层: 从归一化输入计算归一化的 y_kin 和 innov

    核心思想 (PINN):
      轮式机器人运动学方程是已知的精确物理模型。
      内部分三步: 去归一化 → 物理空间运动学递推 → 再归一化。
      网络因此获得:
        - y_kin: "控制指令预测的轨迹"
        - innov: "测量与运动学预测的偏差" (归一化后, 与 y_meas 同尺度)

    与旧版手工新息的区别:
      - 旧版在预处理中独立计算, 需要手工设置 innov 归一化锚点 [0.5, 0.5, 0.3]
      - 本层 innov 直接用 y_meas 的尺度归一化 (y_meas_scale), 物理含义一致
    """

    def __init__(self, ts=TS, alpha=ALPHA,
                 ymeas_scale=None, ymeas_median=None, cmd_max=None):
        super().__init__()
        self.ts = ts
        self.alpha = alpha

        # 注册为 buffer 以便 to(device) 自动跟随
        ymeas_scale = ymeas_scale or [2.5, 2.5, 3.141592653589793]
        ymeas_median = ymeas_median or [0., 0., 0.]
        cmd_max = cmd_max or [0.3, 1.76]

        self.register_buffer('ymeas_scale', torch.tensor(ymeas_scale, dtype=torch.float32))
        self.register_buffer('ymeas_median', torch.tensor(ymeas_median, dtype=torch.float32))
        self.register_buffer('cmd_max', torch.tensor(cmd_max, dtype=torch.float32))

    def set_norm_params(self, ymeas_scale, ymeas_median, cmd_max):
        for name, val in [('ymeas_scale', ymeas_scale), ('ymeas_median', ymeas_median),
                           ('cmd_max', cmd_max)]:
            getattr(self, name).copy_(torch.as_tensor(val, dtype=torch.float32))

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_norm: (B, 5, T) — 归一化的 [y_meas, u_cmd]
        Returns:
            (B, 11, T) — [y_meas_norm, u_cmd_norm, y_kin_norm, innov_norm]
        """
        B, _, T = x_norm.shape

        # ---- 1. 去归一化到物理空间 ----
        # y_meas_norm = (y_phys - median) / scale  →  y_phys = y_norm * scale + median
        y_meas_n = x_norm[:, :3, :]
        u_cmd_n = x_norm[:, 3:5, :]

        scale = self.ymeas_scale.view(1, 3, 1)
        median = self.ymeas_median.view(1, 3, 1)
        cmd_s = self.cmd_max.view(1, 2, 1)

        y_meas_p = y_meas_n * scale + median
        u_cmd_p = u_cmd_n * cmd_s

        # ---- 2. 物理空间运动学递推 ----
        y_kin_p = torch.zeros(B, 3, T, device=x_norm.device, dtype=x_norm.dtype)
        y = y_meas_p[:, :, 0]           # 窗口第一帧作为初始状态
        y_kin_p[:, :, 0] = y

        for t in range(T - 1):
            v = u_cmd_p[:, 0, t]
            w = u_cmd_p[:, 1, t]
            cos_t = torch.cos(y[:, 2])
            sin_t = torch.sin(y[:, 2])

            y_next = torch.stack([
                y[:, 0] + self.ts * (v * cos_t - self.alpha * w * sin_t),
                y[:, 1] + self.ts * (v * sin_t + self.alpha * w * cos_t),
                y[:, 2] + self.ts * w,
            ], dim=-1)

            y_kin_p[:, :, t + 1] = y_next
            y = y_next

        innov_p = y_meas_p - y_kin_p
        innov_p[:, 2, :] = torch.atan2(torch.sin(innov_p[:, 2, :]),
                                        torch.cos(innov_p[:, 2, :]))

        # ---- 3. 再归一化回网络空间 ----
        # y_kin 和 innov 与 y_meas 同物理量纲, 共用相同的 scale/median
        y_kin_n = (y_kin_p - median) / scale
        innov_n = innov_p / scale  # innov 物理值已中心化 (median=0), 仅除 scale

        return torch.cat([x_norm, y_kin_n, innov_n], dim=1)


# ============================================================================
# 多尺度膨胀深度可分离卷积骨干
# ============================================================================

class MultiScaleDSConvBlock(nn.Module):
    """3 并行膨胀深度卷积 (d=1,3,9) → Pointwise 混合 → BN+GELU+Pool"""

    def __init__(self, in_channels, out_channels, kernel_size=7,
                 dilations=None, pool_size=POOL_SIZE):
        super().__init__()
        dilations = dilations or CONV_DILATIONS
        self.depthwise_convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size,
                      padding=(kernel_size // 2) * d,
                      dilation=d, groups=in_channels, bias=False)
            for d in dilations
        ])
        self.pointwise = nn.Conv1d(in_channels * 3, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        h = torch.cat([dw(x) for dw in self.depthwise_convs], dim=1)
        return self.pool(F.gelu(self.bn(self.pointwise(h))))


class MultiScaleDSConvBackbone(nn.Module):
    """3 块逐通道扩增 + 降采样: 11→32→64→96, 128→64→32→16"""

    def __init__(self, in_channels=IN_CHANNELS):
        super().__init__()
        self.d_model = CONV_CHANNELS[-1]
        blocks = []
        ci = in_channels
        for i, co in enumerate(CONV_CHANNELS):
            blocks.append(MultiScaleDSConvBlock(ci, co, CONV_KERNEL_SIZES[i]))
            ci = co
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = nn.LayerNorm(self.d_model)

    def forward(self, x):
        h = x.permute(0, 2, 1)
        h = self.blocks(h)
        return self.norm_out(h.permute(0, 2, 1))


# ============================================================================
# 检测器
# ============================================================================

class Detector(nn.Module):
    """PINN 攻击检测模型: 可微分运动学层 + 膨胀深度可分离卷积骨干 + 注意力池化"""

    def __init__(self, ymeas_scale=None, ymeas_median=None, cmd_max=None):
        super().__init__()
        ymeas_scale = ymeas_scale or [2.5, 2.5, 3.141592653589793]
        ymeas_median = ymeas_median or [0., 0., 0.]
        cmd_max = cmd_max or [0.3, 1.76]

        # 归一化参数 (buffer)
        self.register_buffer('ymeas_scale', torch.tensor(ymeas_scale, dtype=torch.float32))
        self.register_buffer('ymeas_median', torch.tensor(ymeas_median, dtype=torch.float32))
        self.register_buffer('cmd_max', torch.tensor(cmd_max, dtype=torch.float32))

        # PINN 运动学层 (零参数, 纯物理计算, 含去归一化/再归一化)
        self.kinematic_layer = KinematicFeatureLayer(
            ymeas_scale=ymeas_scale, ymeas_median=ymeas_median, cmd_max=cmd_max)

        self.backbone = MultiScaleDSConvBackbone()

        self.cls_norm = nn.LayerNorm(D_MODEL)
        self.cls_head = nn.Linear(D_MODEL, NUM_CLASSES)
        self.cls_dropout = nn.Dropout(0.3)
        self.attn_query = nn.Parameter(torch.randn(D_MODEL) * 0.02)

        nn.init.xavier_uniform_(self.cls_head.weight)
        nn.init.zeros_(self.cls_head.bias)

    def set_norm_params(self, ymeas_scale, ymeas_median, cmd_max):
        """从 normalizer 加载后更新归一化参数 (同步到运动学层)"""
        for name, val in [('ymeas_scale', ymeas_scale), ('ymeas_median', ymeas_median),
                           ('cmd_max', cmd_max)]:
            getattr(self, name).copy_(torch.as_tensor(val, dtype=torch.float32))
        self.kinematic_layer.set_norm_params(ymeas_scale, ymeas_median, cmd_max)

    def encode(self, x):
        return self.backbone(x)

    def classify(self, features):
        d = features.shape[-1]
        scores = torch.matmul(features, self.attn_query) / (d ** 0.5)
        attn = torch.softmax(scores, dim=1)
        pooled = (features * attn.unsqueeze(-1)).sum(dim=1)
        return self.cls_head(self.cls_dropout(self.cls_norm(pooled)))

    def forward(self, x):
        # x: (B, T, RAW_CHANNELS)  归一化后的原始输入
        # 1. 置换 → (B, C, T) 供 Conv1d 使用
        x = x.permute(0, 2, 1)

        # 2. PINN: 可微分运动学特征注入
        x = self.kinematic_layer(x)  # (B, 5, T) → (B, 11, T)

        # 3. 骨干 + 分类
        features = self.encode(x.permute(0, 2, 1))
        cls_logits = self.classify(features)
        return cls_logits, features
