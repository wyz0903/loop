"""
detector/losses.py — 训练损失函数
====================================
FocalLoss: 多分类 Focal Loss，聚焦难分类样本
composite_recon_loss: 复合重建损失 (Pearson + 幅度 + MSE + 频谱)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    自动聚焦于难分类样本, 降低易分类样本的损失贡献。
    对 A1(偏置)/A3(斜坡) 等难检测攻击类型特别有效。
    """
    def __init__(self, gamma: float = 2.0, alpha=None,
                 label_smoothing: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Tensor of shape (num_classes,) or None
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none',
                                   label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            at = self.alpha.gather(0, targets)
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# 复合重建损失
# ============================================================================

def composite_recon_loss(atk_pred, atk_seq, cls_label,
                          pearson_w: float = 1.0,
                          amplitude_w: float = 0.3,
                          mse_w: float = 0.5,
                          spectral_w: float = 0.05,
                          a0_weight: float = 0.15):
    """复合重建损失: 分离形状、幅度、频域监督

    针对旧版"输出≈0"先验问题的根本修复:
      - Pearson 相关损失: 尺度无关的形状保持 (解决幅度低估)
      - 幅度比损失: 对数尺度惩罚幅度不匹配
      - MSE: 基线平滑监督
      - 频谱损失: 频域精度 (A7 扫频关键)

    A0 窗口(target≈0): Pearson/幅度/频谱无定义, 退化为纯 MSE。
    """
    B = atk_pred.shape[0]
    device = atk_pred.device

    # 展平用于相关计算
    pred_f = atk_pred.reshape(B, -1)   # (B, W*3)
    target_f = atk_seq.reshape(B, -1)

    # ---- 1. Pearson 相关损失 (1 - |r|): 尺度无关形状保持 ----
    pred_m = pred_f.mean(dim=1, keepdim=True)
    target_m = target_f.mean(dim=1, keepdim=True)
    pred_c = pred_f - pred_m
    target_c = target_f - target_m

    pred_std = torch.sqrt((pred_c ** 2).sum(dim=1) + 1e-8)
    target_std = torch.sqrt((target_c ** 2).sum(dim=1) + 1e-8)

    correlation = (pred_c * target_c).sum(dim=1) / (pred_std * target_std)
    pearson_loss = 1.0 - correlation.abs()  # (B,)

    # ---- 2. 幅度比损失: log(|pred_std / target_std|) ----
    amp_ratio = pred_std / target_std.clamp(min=1e-8)
    amplitude_loss = torch.abs(torch.log(amp_ratio.clamp(min=1e-4, max=1e4)))

    # ---- 3. MSE 损失 ----
    mse_loss = ((atk_pred - atk_seq) ** 2).mean(dim=[1, 2])  # (B,)

    # ---- 4. 频谱损失: FFT 幅度差 (零填充到 128 以兼容 FP16 cuFFT) ----
    # cuFFT 在半精度下要求信号长度为 2 的幂
    n_fft = 128  # 下一个 2 的幂 > 100
    pred_pad = F.pad(atk_pred.permute(0, 2, 1), (0, n_fft - atk_pred.shape[1]))  # (B, 3, 128)
    target_pad = F.pad(atk_seq.permute(0, 2, 1), (0, n_fft - atk_seq.shape[1]))
    pred_fft = torch.fft.rfft(pred_pad.float(), dim=2, norm='ortho').abs()  # (B, 3, 65)
    target_fft = torch.fft.rfft(target_pad.float(), dim=2, norm='ortho').abs()
    spectral_loss = ((pred_fft - target_fft) ** 2).mean(dim=[1, 2])  # (B,)

    # ---- 逐样本加权: A0 只用 MSE, 非 A0 用全部组件 ----
    is_a0 = (cls_label == 0)
    # A0: 纯 MSE * a0_weight (降低零先验)
    loss_a0 = mse_loss * a0_weight
    # 非 A0: 全部组件
    loss_attack = (pearson_w * pearson_loss +
                   amplitude_w * amplitude_loss +
                   mse_w * mse_loss +
                   spectral_w * spectral_loss)

    loss = torch.where(is_a0, loss_a0, loss_attack)
    return loss.mean()


def _per_sample_recon_loss(atk_pred, atk_seq, cls_label, *args, **kwargs):
    """向后兼容包装: 旧版 per-class weighted MSE"""
    a0_weight = kwargs.get('a0_weight', 0.15)
    per_sample_mse = ((atk_pred - atk_seq) ** 2).mean(dim=[1, 2])
    weights = torch.where(cls_label == 0,
                          torch.tensor(a0_weight, device=cls_label.device),
                          torch.tensor(1.0, device=cls_label.device))
    return (per_sample_mse * weights).mean()
