"""
attack.py — 传感器攻击注入器 (IEEE TIE 论文)
==========================================================================
7种攻击类型 + Normal基准，每种攻击具有严格的物理含义。

攻击分类:
  A0 — 正常 (Normal)                    — 无攻击，基准对照
  A1 — 恒定偏移 (Constant Bias)         — 加性：传感器校准误差 / 零漂
  A2 — 正弦注入 (Sinusoidal)             — 加性：工频 / PWM 电磁干扰
  A3 — 斜坡漂移 (Drift)                 — 加性：MEMS 温度漂移，线性累积
  A4 — 重放攻击 (Replay Attack)         — 非加性：录制后回放历史测量值
  A5 — 信号丢失 (Intermittent Dropout)  — 非加性：接线松动，传感器间歇开路
  A6 — 缩放攻击 (Scaling)               — 乘性：传感器增益被篡改
  A7 — 传感器冻结 (Sensor Freeze)       — 非加性：固件死锁 / ADC锁存

用法:
  attacker = SensorAttack(attack_type='A1', onset_time=5.0, seed=42)
  for t in simulation:
      y_meas = attacker.inject(t, y_clean)   # 统一接口
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# ============================================================================
# 攻击配置
# ============================================================================

@dataclass
class AttackConfig:
    """单种攻击的参数配置"""

    # —— A1 恒定偏移 (Constant Bias) ——
    # 来源: 传感器出厂校准后的残余零漂
    bias_xy: Tuple[float, float] = (0.15, -0.12)   # x,y 偏置 [m]
    bias_theta: float = 0.10                         # 角度偏置 [rad]

    # —— A2 正弦注入 (Sinusoidal) ——
    # 来源: 50Hz 工频或 PWM 开关噪声电磁耦合
    sin_amp: float = 0.12                            # 幅值 [m]
    sin_freq: float = 0.8                            # 干扰频率 [Hz]
    sin_phase: float = 0.0                            # 初始相位 [rad]

    # —— A3 斜坡漂移 (Drift) ——
    # 来源: MEMS 传感器温升导致信号基线缓慢漂移
    ramp_rate_xy: float = 0.100                       # 位置漂移率 [m/s] (5s→0.5m)
    ramp_rate_theta: float = 0.050                    # 角度漂移率 [rad/s] (5s→0.25rad)

    # —— A4 重放攻击 (Replay Attack) ——
    # 来源: 攻击者录制历史测量值后回放
    replay_record_gap: float = 0.0                     # 录制截止距攻击开始的间隔 [s]

    # —— A5 信号丢失 (Intermittent Dropout) ——
    # 来源: 接线松动、连接器氧化，传感器间歇性开路 → 输出为零
    # 固定模式: 在攻击窗口内等间隔插入 N 段固定长度的丢包
    dropout_burst_count: int = 4                       # 丢包段数
    dropout_burst_length: int = 10                     # 每段丢包步数 (0.5s)
    dropout_zero_output: bool = True                   # True=输出归零, False=保持末值

    # —— A6 缩放攻击 (Scaling) ——
    # 来源: 传感器信号调理电路被篡改，增益偏移或ADC参考电压被修改
    # 唯一乘性攻击: y_att = diag(s) @ y_clean
    scale_x: float = 0.6                                # x通道缩放因子 (<1=缩小)
    scale_y: float = 0.8                                # y通道缩放因子 (<1=轻微缩小)
    scale_theta: float = 0.5                            # θ通道缩放因子 (<1=缩小)

    # —— A7 传感器冻结 (Sensor Freeze) ——
    # 来源: 固件死锁、ADC锁存、I²C总线卡死导致传感器不再更新
    # y_att(t) = y_clean(t_freeze), t ≥ onset
    # 无需额外参数，冻结时刻 = onset_time

    # —— 攻击持续时间 (None = 永不结束) ——
    attack_duration: float = None


# ============================================================================
# 攻击注入器
# ============================================================================

class SensorAttack:
    """传感器攻击注入器

    统一 inject() 接口。每种攻击有明确的物理含义。
    """

    ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    ATTACK_NAMES = {
        'A0': 'Normal',
        'A1': 'Constant Bias',
        'A2': 'Sinusoidal',
        'A3': 'Drift',
        'A4': 'Replay Attack',
        'A5': 'Intermittent Dropout',
        'A6': 'Scaling',
        'A7': 'Sensor Freeze',
    }

    def __init__(self, attack_type: str = 'A0',
                 onset_time: float = 5.0,
                 config: AttackConfig = None,
                 seed: int = 42):
        if attack_type not in self.ATTACK_TYPES:
            raise ValueError(f"Unknown attack type: {attack_type}. "
                             f"Choose from {self.ATTACK_TYPES}")

        self.attack_type = attack_type
        self.onset_time = onset_time
        self.cfg = config if config is not None else AttackConfig()
        self.rng = np.random.RandomState(seed)

        if self.cfg.attack_duration is not None and self.cfg.attack_duration > 0:
            self.offset_time = onset_time + self.cfg.attack_duration
        else:
            self.offset_time = float('inf')

        # ---- 攻击类型特定状态 ----
        self._replay_buffer = []           # A4 录制缓冲
        self._replay_idx = 0               # A4 回放指针
        self._frozen_value = None          # A7 冻结值
        self._scaling_diag = None          # A6 缩放对角矩阵 (缓存)

    def reset(self):
        """重置攻击状态 (每次新仿真前调用)"""
        self._replay_buffer = []
        self._replay_idx = 0
        self._frozen_value = None
        self._scaling_diag = None

    def _is_active(self, t: float) -> bool:
        """检查攻击是否在生效期内"""
        if self.attack_type == 'A0':
            return False
        return self.onset_time <= t < self.offset_time

    # ==================================================================
    # 统一注入接口
    # ==================================================================

    def inject(self, t: float, y_clean: np.ndarray) -> np.ndarray:
        """将攻击施加于干净测量值

        Args:
            t:       当前仿真时间 [s]
            y_clean: 干净传感器测量值 (true_state + noise) (3,)

        Returns:
            y_attacked: 受攻击后的测量值 (3,)
        """
        # ---- A0 无攻击 或 攻击窗口外 ----
        if not self._is_active(t):
            # A4 特殊处理: 攻击前需要录制
            if self.attack_type == 'A4' and t < self.onset_time - self.cfg.replay_record_gap:
                self._replay_buffer.append(y_clean.copy())
            return y_clean.copy()

        # ---- 按攻击类型分发 ----
        if self.attack_type == 'A4':
            return self._inject_A4(t, y_clean)
        elif self.attack_type == 'A5':
            return self._inject_A5(t, y_clean)
        elif self.attack_type == 'A6':
            return self._inject_A6(t, y_clean)
        elif self.attack_type == 'A7':
            return self._inject_A7(t, y_clean)
        else:
            # 加性攻击: A1-A3
            a_k = self._step(t)
            return y_clean + a_k

    def _step(self, t: float) -> np.ndarray:
        """生成加性攻击向量 a(k) — 仅 A1-A3 使用"""
        if not self._is_active(t):
            return np.zeros(3)
        method = getattr(self, f'_attack_{self.attack_type}')
        return method(t)

    # ==================================================================
    # A1: 恒定偏移 — 传感器校准后残余零漂
    # ==================================================================
    def _attack_A1(self, t: float) -> np.ndarray:
        return np.array([
            self.cfg.bias_xy[0],
            self.cfg.bias_xy[1],
            self.cfg.bias_theta
        ])

    # ==================================================================
    # A2: 正弦注入 — 工频或PWM开关噪声电磁耦合
    # ==================================================================
    def _attack_A2(self, t: float) -> np.ndarray:
        val = self.cfg.sin_amp * np.sin(
            2 * np.pi * self.cfg.sin_freq * t + self.cfg.sin_phase
        )
        return np.array([val, val * 0.7, val * 0.3])

    # ==================================================================
    # A3: 斜坡漂移 — MEMS温升导致信号基线缓慢线性漂移
    # ==================================================================
    def _attack_A3(self, t: float) -> np.ndarray:
        dt = t - self.onset_time
        return np.array([
            self.cfg.ramp_rate_xy * dt,
            self.cfg.ramp_rate_xy * dt * 0.8,
            self.cfg.ramp_rate_theta * dt
        ])

    # ==================================================================
    # A4: 重放攻击 — 录制历史测量值后循环回放
    # ==================================================================
    def _inject_A4(self, t: float, y_clean: np.ndarray) -> np.ndarray:
        if self._replay_buffer:
            if self._replay_idx >= len(self._replay_buffer):
                self._replay_idx = 0
            y_replay = self._replay_buffer[self._replay_idx].copy()
            self._replay_idx += 1
            return y_replay
        else:
            return y_clean.copy()

    # ==================================================================
    # A5: 信号丢失 — 接线松动/连接器氧化，传感器间歇性开路
    #
    # 固定模式: 在攻击窗口内等间隔插入 N 段固定长度的丢包。
    #   每段丢包: 传感器输出归零 (open circuit = 0V)
    #   段间恢复: 正常输出
    #
    # 示例: 4 段 × 10 步 = 40 步丢包 / 100 步攻击 = 40% 丢包率
    # ==================================================================
    def _inject_A5(self, t: float, y_clean: np.ndarray) -> np.ndarray:
        step_in_attack = int(round((t - self.onset_time) / 0.05))
        burst_len = self.cfg.dropout_burst_length
        num_bursts = self.cfg.dropout_burst_count
        # 用缓存的 attack_duration 计算总攻击步数
        total_atk_steps = int(round(self.cfg.attack_duration / 0.05))
        # 等间隔排布丢包段
        total_gap = max(0, total_atk_steps - num_bursts * burst_len)
        gap = total_gap // (num_bursts + 1)
        extra = total_gap % (num_bursts + 1)

        pos = 0
        for i in range(num_bursts):
            eff_gap = gap + (1 if i < extra else 0)
            pos += eff_gap
            if pos <= step_in_attack < pos + burst_len:
                if self.cfg.dropout_zero_output:
                    return np.zeros(3)
                return y_clean.copy()  # 保持末值模式暂用 clean
            pos += burst_len

        return y_clean.copy()

    # ==================================================================
    # A6: 缩放攻击 — 传感器信号调理电路增益被篡改
    #
    # 唯一乘性攻击: y_att = diag(s_x, s_y, s_θ) · y_clean
    #
    # 物理场景:
    #   - 可编程增益放大器(PGA)被恶意重配置
    #   - ADC参考电压被修改
    #   - 数字信号链中乘法因子被篡改
    #
    # s > 1:  放大 → 机器人以为动了更多，实际控制减弱
    # 0<s<1:  缩小 → 机器人以为动了更少，实际控制过冲
    # s < 0:  反向 → 传感器读数符号反转，最危险(方向完全错误)
    # ==================================================================
    def _inject_A6(self, t: float, y_clean: np.ndarray) -> np.ndarray:
        if self._scaling_diag is None:
            self._scaling_diag = np.array([
                self.cfg.scale_x,
                self.cfg.scale_y,
                self.cfg.scale_theta
            ])
        return self._scaling_diag * y_clean

    # ==================================================================
    # A7: 传感器冻结 — 固件死锁/ADC锁存/I²C总线卡死
    #
    # y_att(t) = y_clean(t_freeze),  t ≥ t_onset
    #
    # 物理场景:
    #   - 传感器MCU固件进入死循环,不再更新输出寄存器
    #   - ADC采样保持电路锁存(单粒子翻转/闩锁效应)
    #   - I²C/SPI总线SCL被拉低,传感器无法响应主机请求
    #
    # 与重放(A4)的区别: 冻结是单一时刻的快照, 重放是历史序列的回放
    # 与恒定偏移(A1)的区别: 偏移是真实值+固定值, 冻结是与真实值无关的常数
    # ==================================================================
    def _inject_A7(self, t: float, y_clean: np.ndarray) -> np.ndarray:
        if self._frozen_value is None:
            self._frozen_value = y_clean.copy()
        return self._frozen_value.copy()


# ============================================================================
# 模块级攻击元数据 — 项目唯一数据源 (single source of truth)
# ============================================================================

# 攻击类型列表
ALL_ATTACK_TYPES = SensorAttack.ATTACK_TYPES

# 攻击英文名称 (论文用)
ATTACK_NAMES = SensorAttack.ATTACK_NAMES

# 攻击颜色方案 (8 类统一配色)
ATK_COLORS = {
    'A0': '#4CAF50', 'A1': '#E91E63', 'A2': '#FF9800', 'A3': '#2196F3',
    'A4': '#9C27B0', 'A5': '#795548', 'A6': '#00BCD4', 'A7': '#607D8B',
}
