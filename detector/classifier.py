"""
detector/classifier.py — 运动学约束状态估计网络 (Kinematic State Estimation Network)
==================================================================================
设计哲学 (JEPA 启发):
  不在原始信号空间做手工特征或预测, 在表示空间学习。
  编码器直接输出干净位姿估计 q, 运动学方程作为约束 (L_kin) 而非输入特征。
  攻击信号自然浮现于 y_meas - q, 分类器读取时序特征 + 攻击残差。

输入: (B, 128, 5) [y_meas(3), u_cmd(2)] 归一化
输出: cls_logits (B, 8), q (B, 128, 3) — 干净传感器估计

架构: Embedding → 4×DilatedResidualBlock → StateHead + Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ============================================================================
# 常量
# ============================================================================
D_MODEL = 48
NUM_CLASSES = 8
IN_CHANNELS = 5            # [y_meas(3), u_cmd(2)]
STATE_CHANNELS = 3         # q = [x, y, θ]
WINDOW_SIZE = 128
DILATIONS = [1, 3, 9, 27]  # 感受野: 5, 21, 57, 165 步
KERNEL_SIZE = 3
TS = 0.05
ALPHA = 0.17


# ============================================================================
# 归一化空间运动学递推 (工具函数)
# ============================================================================

def kinematic_step_norm(q_norm: torch.Tensor, u_norm: torch.Tensor,
                        scale: torch.Tensor, median: torch.Tensor,
                        cmd_max: torch.Tensor,
                        ts: float = TS, alpha: float = ALPHA) -> torch.Tensor:
    """归一化空间中的一步 Euler 运动学递推。

    从归一化状态 q_norm 和控制 u_norm 出发,
    去归一化 → 物理空间递推 → 再归一化, 得到下一步状态的归一化预测。

    Args:
        q_norm: (..., 3) 归一化 [x, y, θ]
        u_norm: (..., 2) 归一化 [v, ω]
        scale, median: (3,) y_meas 归一化参数
        cmd_max: (2,) u_cmd 归一化参数
    Returns:
        q_next_norm: (..., 3) 预测的下一步归一化状态
    """
    # 去归一化
    θ_phys = q_norm[..., 2] * scale[2] + median[2]
    v_phys = u_norm[..., 0] * cmd_max[0]
    ω_phys = u_norm[..., 1] * cmd_max[1]

    cos_t = torch.cos(θ_phys)
    sin_t = torch.sin(θ_phys)

    # 物理空间递推, 再归一化
    dx = ts * (v_phys * cos_t - alpha * ω_phys * sin_t) / scale[0]
    dy = ts * (v_phys * sin_t + alpha * ω_phys * cos_t) / scale[1]
    dθ = ts * ω_phys / scale[2]

    return q_norm + torch.stack([dx, dy, dθ], dim=-1)


# ============================================================================
# 膨胀残差卷积块
# ============================================================================

class DilatedResidualBlock(nn.Module):
    """膨胀残差块: Conv(d) → BN → GELU → Conv(d) → BN, out = GELU(in + residual)

    保持通道数和序列长度不变, 膨胀率控制感受野大小。
    """

    def __init__(self, channels: int = D_MODEL, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, KERNEL_SIZE,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, KERNEL_SIZE,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(residual + h)


# ============================================================================
# 时序编码器
# ============================================================================

class TemporalEncoder(nn.Module):
    """时序编码器: Embedding + 4× 膨胀残差块 (无降采样, 保持 128 步分辨率)"""

    def __init__(self, in_channels: int = IN_CHANNELS, d_model: int = D_MODEL):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(d_model, d) for d in DILATIONS
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → F: (B, D_MODEL, T)"""
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return h


# ============================================================================
# 攻击分类器
# ============================================================================

class AttackClassifier(nn.Module):
    """攻击分类器: 注意力池化(F) + 攻击残差强度 → Linear → 8 类

    双路信息融合:
      - f_pooled: 从时序特征 F 通过可学习 query 注意力池化得到
      - att_mag_mean: ||y_meas - q|| 的窗口均值, 直接指示攻击存在/强度
    """

    def __init__(self, d_model: int = D_MODEL, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.attn_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.head = nn.Linear(d_model + 1, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, F: torch.Tensor, y_meas: torch.Tensor,
                q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F: (B, D, T) 编码器时序特征
            y_meas: (B, 3, T) 原始量测 (可能被掩码/攻击)
            q: (B, 3, T) 干净状态估计
        Returns:
            cls_logits: (B, num_classes)
        """
        # 注意力池化
        F_t = F.permute(0, 2, 1)                             # (B, T, D)
        d = F_t.shape[-1]
        scores = torch.matmul(F_t, self.attn_query) / (d ** 0.5)
        attn = torch.softmax(scores, dim=1)
        f_pooled = (F_t * attn.unsqueeze(-1)).sum(dim=1)     # (B, D)

        # 攻击残差强度
        att_mag = torch.norm(y_meas - q, dim=1)              # (B, T)
        att_mag_mean = att_mag.mean(dim=1, keepdim=True)     # (B, 1)

        return self.head(torch.cat([f_pooled, att_mag_mean], dim=1))


# ============================================================================
# 检测器
# ============================================================================

class Detector(nn.Module):
    """运动学约束状态估计检测器

    训练: y_meas 随机块掩码 → Encoder → q (L_recon + L_kin) + cls (L_cls)
    推理: 完整序列 → Encoder → q (干净状态) + cls (攻击诊断)

    q 直接替代被攻击测量值送入 NMPC 进行传感器恢复。
    """

    def __init__(self, ymeas_scale=None, ymeas_median=None, cmd_max=None,
                 mask_min: float = 0.1, mask_max: float = 0.4):
        super().__init__()
        self.mask_min = mask_min
        self.mask_max = mask_max

        ymeas_scale = ymeas_scale or [2.5, 2.5, 3.141592653589793]
        ymeas_median = ymeas_median or [0., 0., 0.]
        cmd_max = cmd_max or [0.3, 1.76]

        # 归一化参数 (buffer)
        self.register_buffer('ymeas_scale', torch.tensor(ymeas_scale, dtype=torch.float32))
        self.register_buffer('ymeas_median', torch.tensor(ymeas_median, dtype=torch.float32))
        self.register_buffer('cmd_max', torch.tensor(cmd_max, dtype=torch.float32))

        self.encoder = TemporalEncoder()
        self.state_head = nn.Conv1d(D_MODEL, STATE_CHANNELS, kernel_size=1)
        self.classifier = AttackClassifier()

    # ------------------------------------------------------------------
    # 归一化参数
    # ------------------------------------------------------------------

    def set_norm_params(self, ymeas_scale, ymeas_median, cmd_max):
        for name, val in [('ymeas_scale', ymeas_scale), ('ymeas_median', ymeas_median),
                           ('cmd_max', cmd_max)]:
            getattr(self, name).copy_(torch.as_tensor(val, dtype=torch.float32))

    # ------------------------------------------------------------------
    # 掩码
    # ------------------------------------------------------------------

    def _apply_mask(self, x: torch.Tensor, ratio: float
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机位置连续块掩码 y_meas 通道 (u_cmd 始终可见)。

        Args:
            x: (B, C, T) 归一化输入
            ratio: 掩码比例
        Returns:
            x_masked: (B, C, T), mask: (B, T)  1=未掩码, 0=掩码
        """
        B, _, T = x.shape
        mask_len = max(1, int(T * ratio))
        start_max = T - mask_len

        mask = torch.ones(B, T, device=x.device, dtype=x.dtype)
        starts = torch.randint(0, start_max + 1, (B,), device=x.device)
        for b in range(B):
            mask[b, starts[b]:starts[b] + mask_len] = 0.0

        x_masked = x.clone()
        x_masked[:, :3, :] = x_masked[:, :3, :] * mask.unsqueeze(1)
        return x_masked, mask

    # ------------------------------------------------------------------
    # 运动学一致性损失
    # ------------------------------------------------------------------

    def compute_kin_loss(self, q: torch.Tensor, u_cmd: torch.Tensor
                         ) -> torch.Tensor:
        """计算运动学一致性损失: q[t] 应 ≈ Euler(q[t-1], u[t-1])。

        Args:
            q: (B, 3, T) 干净状态估计 (归一化空间)
            u_cmd: (B, 2, T) 控制指令 (归一化空间)
        Returns:
            L_kin: 标量
        """
        # 转为 (B, T, C) 格式供 kinematic_step_norm (期望 state 在最后一维)
        q_t = q.permute(0, 2, 1)         # (B, T, 3)
        u_t = u_cmd.permute(0, 2, 1)     # (B, T, 2)

        q_prev = q_t[:, :-1, :]          # (B, T-1, 3): t = 0..T-2
        u_prev = u_t[:, :-1, :]          # (B, T-1, 2): u = 0..T-2
        q_curr = q_t[:, 1:, :]           # (B, T-1, 3): t = 1..T-1

        q_pred = kinematic_step_norm(
            q_prev, u_prev,
            self.ymeas_scale, self.ymeas_median, self.cmd_max)
        return F.mse_loss(q_curr, q_pred)

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                mask_ratio: Optional[float] = None,
                return_all: bool = False):
        """
        Args:
            x: (B, T, C) 归一化输入 [y_meas(3), u_cmd(2)]
            mask_ratio: 掩码比例 (None=推理模式)
            return_all: 是否返回 mask 和 attack_res (训练用)

        Returns:
            推理: cls_logits (B, 8), q (B, T, 3)
            训练: cls_logits, q, mask (B, T), attack_res (B, T, 3)
        """
        x_c = x.permute(0, 2, 1)       # (B, C, T)
        y_meas = x_c[:, :3, :]
        u_cmd = x_c[:, 3:5, :]

        mask = None
        if mask_ratio is not None and mask_ratio > 0:
            x_c, mask = self._apply_mask(x_c, mask_ratio)
            y_meas = x_c[:, :3, :]

        # 编码
        F_enc = self.encoder(x_c)      # (B, D_MODEL, T)

        # 干净状态估计
        q = self.state_head(F_enc)     # (B, 3, T)

        # 分类
        cls_logits = self.classifier(F_enc, y_meas, q)

        if return_all:
            attack_res = y_meas - q
            return (cls_logits,
                    q.permute(0, 2, 1),
                    mask,
                    attack_res.permute(0, 2, 1))
        return cls_logits, q.permute(0, 2, 1)
