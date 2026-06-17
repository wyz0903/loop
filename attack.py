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
    ramp_rate_xy: float = 0.020                       # 位置漂移率 [m/s]
    ramp_rate_theta: float = 0.010                    # 角度漂移率 [rad/s]

    # —— A4 重放攻击 (Replay Attack) ——
    # 来源: 攻击者录制历史测量值后回放
    replay_record_duration: float = 10.0              # 录制时长 [s] (攻击前)
    replay_record_gap: float = 0.0                     # 录制截止距攻击开始的间隔 [s]

    # —— A5 信号丢失 (Intermittent Dropout) ——
    # 来源: 接线松动、连接器氧化，传感器间歇性开路 → 输出为零
    # Gilbert-Elliott 双状态 Markov 模型
    dropout_p_gf: float = 0.02                        # P(good→fault) 每步转移概率 (稳态故障率~20%)
    dropout_p_fg: float = 0.08                        # P(fault→good) 每步转移概率 (平均故障持续~12.5步)
    dropout_zero_output: bool = True                   # True=输出归零, False=保持末值

    # —— A6 缩放攻击 (Scaling) ——
    # 来源: 传感器信号调理电路被篡改，增益偏移或ADC参考电压被修改
    # 唯一乘性攻击: y_att = diag(s) @ y_clean
    scale_x: float = 1.4                               # x通道缩放因子 (>1=放大)
    scale_y: float = 0.6                               # y通道缩放因子 (<1=缩小)
    scale_theta: float = -0.5                          # θ通道缩放因子 (<0=反向)

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

        # ---- A4 重放时长随机化 (打破固定周期过拟合) ----
        if attack_type == 'A4':
            self.cfg.replay_record_duration = self.rng.uniform(5.0, 15.0)

        # ---- 攻击类型特定状态 ----
        self._replay_buffer = []           # A4 录制缓冲
        self._replay_idx = 0               # A4 回放指针
        self._dropout_state = 0            # A5 Markov状态: 0=good, 1=fault
        self._last_good_value = None       # A5 故障前最后有效值 (用于保持末值模式)
        self._frozen_value = None          # A7 冻结值
        self._scaling_diag = None          # A6 缩放对角矩阵 (缓存)

    def reset(self):
        """重置攻击状态 (每次新仿真前调用)"""
        self._replay_buffer = []
        self._replay_idx = 0
        self._dropout_state = 0
        self._last_good_value = None
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
            a_k = self.step(t)
            return y_clean + a_k

    def step(self, t: float) -> np.ndarray:
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
    # Gilbert-Elliott 双状态 Markov 模型:
    #   State 0 (good):  正常输出
    #   State 1 (fault): 传感器输出归零 (open circuit = 0V)
    #
    # 转移概率:
    #   P(0→1) = dropout_p_gf : 突发故障概率 (小)
    #   P(1→0) = dropout_p_fg : 恢复概率     (大)
    #
    # 产生突发的、间歇性的信号丢失，具有工业传感器连接故障的特征
    # ==================================================================
    def _inject_A5(self, t: float, y_clean: np.ndarray) -> np.ndarray:
        # Markov 状态转移
        if self._dropout_state == 0:
            self._last_good_value = y_clean.copy()  # 记录故障前最后有效值
            if self.rng.rand() < self.cfg.dropout_p_gf:
                self._dropout_state = 1
        else:
            if self.rng.rand() < self.cfg.dropout_p_fg:
                self._dropout_state = 0

        if self._dropout_state == 1:
            if self.cfg.dropout_zero_output:
                return np.zeros(3)                       # 开路 → 零输出
            else:
                if self._last_good_value is not None:
                    return self._last_good_value.copy()  # 保持故障前末值
                return y_clean.copy()                    # 兜底 (无缓存时)
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

    # ==================================================================
    # 工具方法
    # ==================================================================

    @staticmethod
    def print_summary():
        """输出攻击类型汇总"""
        print("\n" + "=" * 65)
        print("Sensor Attack Catalog — 传感器攻击目录 (IEEE TIE)")
        print("=" * 65)
        for atype in ['A0','A1','A2','A3','A4','A5','A6','A7']:
            print(f"  {atype}: {SensorAttack.ATTACK_NAMES[atype]}")
        print("=" * 65)
        print("  A1-A3 : 加性攻击 (additive)")
        print("  A4-A5 : 非加性攻击 — 通过 inject() 接口")
        print("  A6    : 乘性攻击 (multiplicative) — 唯一非加性缩放类")
        print("  A7    : 非加性攻击 — 传感器输出冻结")
        print("=" * 65 + "\n")


# ============================================================================
# 模块级攻击元数据 — 项目唯一数据源 (single source of truth)
# ============================================================================

# 攻击类型列表
ALL_ATTACK_TYPES = SensorAttack.ATTACK_TYPES

# 攻击英文名称 (论文用)
ATTACK_NAMES = SensorAttack.ATTACK_NAMES

# 攻击中文名称 (图表标注用)
ATTACK_NAMES_CN = {
    'A0': '正常',
    'A1': '恒定偏移',
    'A2': '正弦注入',
    'A3': '斜坡漂移',
    'A4': '重放攻击',
    'A5': '信号丢失',
    'A6': '缩放攻击',
    'A7': '传感器冻结',
}

# 攻击颜色方案 (8 类统一配色)
ATK_COLORS = {
    'A0': '#4CAF50', 'A1': '#E91E63', 'A2': '#FF9800', 'A3': '#2196F3',
    'A4': '#9C27B0', 'A5': '#795548', 'A6': '#00BCD4', 'A7': '#607D8B',
}

# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    SensorAttack.print_summary()

    Ts = 0.05
    onset = 5.0

    # —— 测试加性攻击 A1-A3 ——
    print("Testing additive attacks (A1-A3, step interface)...\n")
    for atype in ['A1', 'A2', 'A3']:
        att = SensorAttack(attack_type=atype, onset_time=onset, seed=42)
        vals = []
        for t in np.arange(0.0, 8.0, Ts):
            vals.append(att.step(t))
        vals = np.array(vals)
        pre = vals[:int(onset/Ts)]
        post = vals[int(onset/Ts):]
        assert np.allclose(pre, 0), f"{atype}: pre-onset should be zero!"
        print(f"  {atype} ({SensorAttack.ATTACK_NAMES[atype]}):")
        print(f"    Post-onset range x: [{post[:,0].min():.4f}, {post[:,0].max():.4f}]")
        print(f"    Post-onset range y: [{post[:,1].min():.4f}, {post[:,1].max():.4f}]")
        print(f"    Post-onset range θ: [{post[:,2].min():.4f}, {post[:,2].max():.4f}]")

    # —— 测试 A4 重放攻击 ——
    print("\nTesting A4 Replay Attack...")
    att = SensorAttack(attack_type='A4', onset_time=onset, seed=42)
    y_clean_hist, y_att_hist = [], []
    for t in np.arange(0.0, 10.0, Ts):
        y_clean = np.array([np.sin(t), np.cos(t), 0.1 * t])
        y_att = att.inject(t, y_clean)
        y_clean_hist.append(y_clean)
        y_att_hist.append(y_att)
    y_clean_hist = np.array(y_clean_hist)
    y_att_hist = np.array(y_att_hist)
    pre_mask = np.arange(len(y_att_hist)) < int(onset / Ts)
    pre_ok = np.allclose(y_att_hist[pre_mask], y_clean_hist[pre_mask], atol=1e-10)
    post_dev = np.mean(np.abs(y_att_hist[~pre_mask] - y_clean_hist[~pre_mask]))
    print(f"  攻击前一致: {pre_ok}")
    print(f"  攻击后平均偏差: {post_dev:.4f}m")

    # —— 测试 A5 信号丢失 ——
    print("\nTesting A5 Intermittent Dropout...")
    att = SensorAttack(attack_type='A5', onset_time=onset, seed=42)
    dropout_count = 0
    y_att_hist, y_clean_hist = [], []
    for t in np.arange(0.0, 20.0, Ts):  # 长一点以看到间歇性
        y_clean = np.array([np.sin(t), np.cos(t), 0.1])
        y_att = att.inject(t, y_clean)
        y_att_hist.append(y_att)
        y_clean_hist.append(y_clean)
        if t >= onset and np.allclose(y_att, 0):
            dropout_count += 1
    y_att_hist = np.array(y_att_hist)
    post = y_att_hist[int(onset/Ts):]
    dropout_ratio = dropout_count / len(post) * 100
    print(f"  丢包率 (攻击后): {dropout_ratio:.1f}%")
    print(f"  故障态平均持续时间: {dropout_count / max(1, (post == 0).any(axis=1).sum() / max(1, np.sum(np.diff((post == 0).all(axis=1).astype(int)) == 1))):.1f} 步")
    print(f"  攻击前输出一致: {np.allclose(y_att_hist[:int(onset/Ts)], y_clean_hist[:int(onset/Ts)], atol=1e-10)}")

    # —— 测试 A6 缩放攻击 ——
    print("\nTesting A6 Scaling Attack...")
    att = SensorAttack(attack_type='A6', onset_time=onset, seed=42)
    y_clean_hist, y_att_hist = [], []
    for t in np.arange(0.0, 8.0, Ts):
        y_clean = np.array([1.0, 2.0, 0.5])  # 固定输入用于测试缩放
        y_att = att.inject(t, y_clean)
        y_clean_hist.append(y_clean)
        y_att_hist.append(y_att)
    y_att_hist = np.array(y_att_hist)
    post = y_att_hist[int(onset/Ts):]
    print(f"  输入 y_clean: [{1.0}, {2.0}, {0.5}]")
    print(f"  缩放因子:   s=[{att.cfg.scale_x}, {att.cfg.scale_y}, {att.cfg.scale_theta}]")
    print(f"  输出 y_att:  [{post[0,0]:.3f}, {post[0,1]:.3f}, {post[0,2]:.3f}]")
    print(f"  预期:        [{1.0*att.cfg.scale_x:.3f}, {2.0*att.cfg.scale_y:.3f}, {0.5*att.cfg.scale_theta:.3f}]")
    assert np.allclose(post[0], [1.0*att.cfg.scale_x, 2.0*att.cfg.scale_y, 0.5*att.cfg.scale_theta])
    pre_ok = np.allclose(y_att_hist[:int(onset/Ts)], y_clean_hist[:int(onset/Ts)], atol=1e-10)
    print(f"  攻击前一致: {pre_ok}")

    # —— 测试 A7 传感器冻结 ——
    print("\nTesting A7 Sensor Freeze...")
    att = SensorAttack(attack_type='A7', onset_time=onset, seed=42)
    y_att_hist = []
    y_clean_hist = []
    for t in np.arange(0.0, 8.0, Ts):
        y_clean = np.array([t, 2*t, 0.5*t])
        y_att = att.inject(t, y_clean)
        y_att_hist.append(y_att)
        y_clean_hist.append(y_clean)
    y_att_hist = np.array(y_att_hist)
    post = y_att_hist[int(onset/Ts):]
    # 冻结值应始终等于 onset 时刻的 y_clean
    frozen_expected = y_clean_hist[int(onset/Ts)]
    all_frozen = np.allclose(post, frozen_expected, atol=1e-10)
    print(f"  冻结时刻 (t={onset}s) y_clean: {frozen_expected}")
    print(f"  攻击后所有值 == 冻结值: {all_frozen}")
    # 确认冻结值不随时间变化
    assert all_frozen

    print("\n" + "=" * 65)
    print("  所有攻击自测通过！")
    print("=" * 65 + "\n")
