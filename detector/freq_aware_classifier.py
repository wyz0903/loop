"""
detector/freq_aware_classifier.py — 多尺度频率感知攻击分类与重建网络 (v4)
=============================================================================
FreqAwareClassifier: 双路径架构，编码器潜在信息通过 latent_to_film 统一注入 FiLM 条件。

核心创新:
  1. 频率路径保持 50 步分辨率 (Nyquist=5Hz) — 覆盖 A7 4Hz 扫频
  2. 正弦位置编码 — 解码器感知绝对时间位置，可生成时变频率
  3. 膨胀卷积块 (rate 1,2,4) — 多尺度时间上下文
  4. FiLM 调制 — 分类嵌入 + 潜在投影统一调制解码器，类别条件生成
  5. DC 偏置支路 — latent→FC→3，提供恒定/阶跃直流分量
  6. 3 层解码器 — 比旧版(2层)更深，更好表达复杂波形

参数量: ~830K
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from detector.config import (ENC_CHANNELS, LATENT_DIM, FREQ_CHANNELS,
                              CLASS_EMBED_DIM)
from detector.nn_blocks import ResDownBlock


class FreqAwareClassifier(nn.Module):
    """多尺度频率感知攻击分类与重建网络 (v4)

    核心创新:
      1. 频率路径保持 50 步分辨率 (Nyquist=5Hz) — 覆盖 A7 4Hz 扫频
      2. 正弦位置编码 — 解码器感知绝对时间位置，可生成时变频率
      3. 膨胀卷积块 (rate 1,2,4) — 多尺度时间上下文
      4. 统一 FiLM 条件 — film_input = class_embed + latent_to_film(z)
      5. DC 偏置支路 — latent→FC→3，提供恒定/阶跃直流分量
      6. 3 层解码器 — 比旧版(2层)更深，更好表达复杂波形

    v4 变更: 移除 Conditional Flow Matching 模块。
      - 旧 FM 子模块 (dec_latent_proj, fm_vecfield, dec_latent_adapter)
        保留为 LEGACY 空壳，仅用于兼容旧权重加载。
      - 替换为 latent_to_film (256→48) 单层投影，将编码器实例级信息
        注入 FiLM 条件向量，实现统一的类别条件生成。

    参数量: ~830K (旧版含 FM: ~900K)
    """

    def __init__(self, in_channels: int = 5, window_size: int = 100,
                 latent_dim: int = LATENT_DIM, num_classes: int = 9,
                 enc_channels: list = None,
                 freq_channels: list = None,
                 class_embed_dim: int = CLASS_EMBED_DIM,
                 use_fm: bool = False,
                 fm_latent_dim: int = 128):
        """Args:
            use_fm: [LEGACY] 保留用于兼容旧 FM 权重加载。
                    设为 True 时仍构造 FM 子模块（空壳），但前向路径不使用。
        """
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        enc_ch = enc_channels or ENC_CHANNELS
        freq_ch = freq_channels or FREQ_CHANNELS

        # ================================================================
        # 分类编码器 (ResDownBlock×4, 加宽通道)
        # 5→48→96→192→256, 100→50→25→13→7 步
        # ================================================================
        prev_ch = in_channels
        self.enc_blocks = nn.ModuleList()
        for i, ch in enumerate(enc_ch):
            ks = 7 if i == 0 else 5 if i < 3 else 3
            self.enc_blocks.append(ResDownBlock(prev_ch, ch, kernel_size=ks))
            prev_ch = ch

        self.bottleneck_conv = nn.Conv1d(enc_ch[-1], enc_ch[-1], kernel_size=3,
                                          padding=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm1d(enc_ch[-1])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.enc_fc = nn.Linear(enc_ch[-1], latent_dim)

        # ---- 分类器 ----
        self.cls_fc1 = nn.Linear(latent_dim, 128)
        self.cls_bn1 = nn.BatchNorm1d(128)
        self.cls_dropout1 = nn.Dropout(0.3)
        self.cls_fc2 = nn.Linear(128, 64)
        self.cls_bn2 = nn.BatchNorm1d(64)
        self.cls_dropout2 = nn.Dropout(0.2)
        self.cls_head = nn.Linear(64, num_classes)

        # ---- 分类嵌入 (用于 FiLM 调制) ----
        self.class_embedding = nn.Parameter(
            torch.randn(num_classes, class_embed_dim) * 0.02)

        # ---- DC 偏置支路: latent → 3 通道直流分量 ----
        self.dc_fc = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
        )

        # -------- latent_to_film: 编码器实例信息注入 FiLM 条件 ----------
        # 替代旧版 FM 的 dec_latent_adapter，将编码器潜在 z 投影到
        # 与 class_embed 相同的维度，通过加法统一注入所有 FiLM 层。
        self.latent_to_film = nn.Linear(latent_dim, class_embed_dim)  # 256→48

        # ================================================================
        # 频率保持路径: 保持 50 步分辨率 (Nyquist = 50/(2*5s) = 5Hz)
        # 5→freq_ch[0]→freq_ch[1], 100→50→50 步 (仅一次 stride-2)
        # ================================================================
        self.freq_conv1 = nn.Conv1d(in_channels, freq_ch[0],
                                     kernel_size=7, stride=2, padding=3, bias=False)
        self.freq_bn1 = nn.BatchNorm1d(freq_ch[0])
        self.freq_conv2 = nn.Conv1d(freq_ch[0], freq_ch[1],
                                     kernel_size=5, stride=1, padding=2, bias=False)
        self.freq_bn2 = nn.BatchNorm1d(freq_ch[1])
        self.freq_feat_size = (window_size + 1) // 2  # 50

        # ================================================================
        # 正弦位置编码: 为解码器提供绝对时间位置信息
        # ================================================================
        self.register_buffer('_pos_enc',
            self._build_sinusoidal_encoding(self.freq_feat_size, freq_ch[1]))

        # ================================================================
        # 多尺度膨胀卷积块: rate=1 (局部), rate=2 (中频), rate=4 (低频包络)
        # ================================================================
        self.dil_conv1 = nn.Conv1d(freq_ch[1], freq_ch[1], kernel_size=3,
                                    dilation=1, padding=1, bias=False)
        self.dil_conv2 = nn.Conv1d(freq_ch[1], freq_ch[1], kernel_size=3,
                                    dilation=2, padding=2, bias=False)
        self.dil_conv4 = nn.Conv1d(freq_ch[1], freq_ch[1], kernel_size=3,
                                    dilation=4, padding=4, bias=False)
        self.dil_bn = nn.BatchNorm1d(freq_ch[1])

        # ================================================================
        # FiLM 调制网络: film_input → (scale, shift) 每层解码器
        # film_input = class_embed + latent_to_film(z)，统一了类别信息
        # 与编码器实例信息。
        # ================================================================
        self.film_net1 = nn.Sequential(
            nn.Linear(class_embed_dim, freq_ch[1] * 2),
        )
        self.film_net2 = nn.Sequential(
            nn.Linear(class_embed_dim, 24 * 2),
        )
        self.film_net3 = nn.Sequential(
            nn.Linear(class_embed_dim, 16 * 2),
        )

        # ================================================================
        # [LEGACY] FM 子模块 — 仅用于兼容旧权重加载，前向路径不使用
        # ================================================================
        if use_fm:
            self.dec_latent_proj = nn.Linear(latent_dim, fm_latent_dim)

            fm_input_dim = fm_latent_dim + 64 + class_embed_dim  # z_t + t_emb + class_embed
            self.fm_vecfield = nn.Sequential(
                nn.Linear(fm_input_dim, fm_latent_dim),
                nn.SiLU(),
                nn.Linear(fm_latent_dim, fm_latent_dim),
            )

            self.dec_latent_adapter = nn.Sequential(
                nn.Linear(fm_latent_dim, 64),
                nn.SiLU(),
                nn.Linear(64, freq_ch[1] * 2),
            )

        # ================================================================
        # 解码器: 3 层 (50→100→100→100)
        # ================================================================
        self.dec_up1 = nn.ConvTranspose1d(freq_ch[1], 24,
                                           kernel_size=5, stride=2,
                                           padding=2, output_padding=1, bias=False)
        self.dec_bn1 = nn.BatchNorm1d(24)
        self.dec_conv2 = nn.Conv1d(24, 16, kernel_size=5, padding=2, bias=False)
        self.dec_bn2 = nn.BatchNorm1d(16)
        self.output_conv = nn.Conv1d(16, 3, kernel_size=5, padding=2)

        self._init_weights()

        # LEGACY: FM 近零初始化 (仅旧 FM 模型)
        if use_fm:
            last_linear = self.fm_vecfield[-1]
            nn.init.normal_(last_linear.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(last_linear.bias)
            adapter_last = self.dec_latent_adapter[-1]
            nn.init.normal_(adapter_last.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(adapter_last.bias)

    @staticmethod
    def _build_sinusoidal_encoding(steps: int, channels: int) -> torch.Tensor:
        """构建正弦位置编码 (Transformer 风格)"""
        position = torch.arange(steps).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float()
                           * -(math.log(10000.0) / channels))
        pe = torch.zeros(1, channels, steps)
        pe[0, 0::2, :] = torch.sin(position * div_term).T
        pe[0, 1::2, :] = torch.cos(position * div_term).T
        return pe

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # latent_to_film 近零初始化: 初始时不影响 class_embed，保留旧模型行为
        nn.init.normal_(self.latent_to_film.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.latent_to_film.bias)

    def enable_mc_dropout(self):
        """启用 MC Dropout 用于推理时不确定性估计。

        调用后所有 Dropout 层在 eval() 模式下保持激活，
        多次前向传播可产生贝叶斯近似后验。
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x: torch.Tensor):
        """前向传播

        Args:
            x: (B, W, C) 输入窗口 [internal_innovation(3) + u_cmd(2)]

        Returns:
            cls_logits:  (B, 9) 分类 logits
            attack_seq:  (B, W, 3) 攻击信号重建
            z:           (B, latent_dim) 潜在表示
        """
        B = x.shape[0]
        x_conv = x.permute(0, 2, 1)  # (B, C, W)

        # ================================================================
        # 1. 分类编码器 → latent → classifier
        # ================================================================
        h = x_conv
        for block in self.enc_blocks:
            h = block(h)

        h = F.relu(self.bottleneck_bn(self.bottleneck_conv(h)))
        h = self.global_pool(h).squeeze(-1)
        z = F.relu(self.enc_fc(h))  # (B, latent_dim)

        # 分类
        c = F.relu(self.cls_bn1(self.cls_fc1(z)))
        c = self.cls_dropout1(c)
        c = F.relu(self.cls_bn2(self.cls_fc2(c)))
        c = self.cls_dropout2(c)
        cls_logits = self.cls_head(c)

        # ================================================================
        # 2. 统一 FiLM 条件: 类别嵌入 + 编码器实例投影
        # ================================================================
        cls_probs = F.softmax(cls_logits, dim=1)
        class_embed = cls_probs @ self.class_embedding.to(cls_probs.dtype)  # (B, 48)
        film_input = class_embed + self.latent_to_film(z)  # (B, 48)

        # DC 偏置
        dc_offset = self.dc_fc(z)  # (B, 3)

        # ================================================================
        # 3. 频率保持路径: 50 步分辨率 (Nyquist=5Hz)
        # ================================================================
        f = F.leaky_relu(self.freq_bn1(self.freq_conv1(x_conv)), 0.2)  # (B, 24, 50)
        f = F.leaky_relu(self.freq_bn2(self.freq_conv2(f)), 0.2)        # (B, 32, 50)
        f = f + self._pos_enc  # 正弦位置编码

        # ================================================================
        # 4. 多尺度膨胀卷积 + 残差融合
        # ================================================================
        f_d1 = self.dil_conv1(f)  # rate=1: 局部高频
        f_d2 = self.dil_conv2(f)  # rate=2: 中频
        f_d4 = self.dil_conv4(f)  # rate=4: 低频包络
        f = f + F.leaky_relu(self.dil_bn(f_d1 + f_d2 + f_d4), 0.2)

        # ================================================================
        # 5. FiLM 调制 + 解码 (50→100→100→100)
        # ================================================================
        film1 = self.film_net1(film_input)
        gamma1, beta1 = film1.chunk(2, dim=1)
        f = gamma1.unsqueeze(-1) * f + beta1.unsqueeze(-1)
        f = F.leaky_relu(self.dec_bn1(self.dec_up1(f)), 0.2)  # (B, 24, 100)

        film2 = self.film_net2(film_input)
        gamma2, beta2 = film2.chunk(2, dim=1)
        f = gamma2.unsqueeze(-1) * f + beta2.unsqueeze(-1)
        f = F.leaky_relu(self.dec_bn2(self.dec_conv2(f)), 0.2)  # (B, 16, 100)

        film3 = self.film_net3(film_input)
        gamma3, beta3 = film3.chunk(2, dim=1)
        f = gamma3.unsqueeze(-1) * f + beta3.unsqueeze(-1)
        attack_seq = self.output_conv(f)  # (B, 3, 100)

        # 加直流分量
        attack_seq = attack_seq + dc_offset.unsqueeze(-1)
        attack_seq = attack_seq.permute(0, 2, 1)  # (B, W, 3)

        return cls_logits, attack_seq, z
