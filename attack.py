"""
attack.py — 传感器通道加性攻击注入器
==========================================================================
8种攻击类型 + Normal基准，全部作用于传感器测量值。

统一形式:
  y_meas(k) = X_true(k) + n(k) + a(k)

其中 n(k) 为正常传感器噪声，a(k) 为攻击注入信号。

攻击索引:
  A0 — Normal   (无攻击，基准)
  A1 — 恒定偏置 (Constant Bias)
  A2 — 正弦注入 (Sinusoidal Injection)
  A3 — 斜坡漂移 (Ramp Drift)
  A4 — 阶跃突变 (Step Attack)
  A5 — 重放攻击 (Replay Attack)
  A6 — 脉冲序列 (Pulse Train)
  A7 — 扫频攻击 (Chirp / Frequency Sweep)
  A8 — 多频叠加 (Multi-Tone Injection)

用法:
  attacker = SensorAttack(attack_type='A5', onset_time=5.0, seed=42)
  for t in simulation:
      a_k = attacker.step(t)
      y_meas = true_state + sensor_noise + a_k
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple


# ============================================================================
# 攻击配置
# ============================================================================

@dataclass
class AttackConfig:
    """单种攻击的参数配置"""
    # —— A1 恒定偏置 ——
    bias_xy: Tuple[float, float] = (0.15, -0.12)  # x,y 偏置 [m] (↑1.9x)
    bias_theta: float = 0.10                        # 角度偏置 [rad] (↑2x)

    # —— A2 正弦注入 ——
    sin_amp: float = 0.12                           # 幅值 [m] (↑1.5x)
    sin_freq: float = 0.8                           # 频率 [Hz]
    sin_phase: float = 0.0                          # 初始相位 [rad]

    # —— A3 斜坡漂移 ——
    ramp_rate_xy: float = 0.008                     # 位置漂移率 [m/s] (↑2x)
    ramp_rate_theta: float = 0.004                  # 角度漂移率 [rad/s] (↑2x)

    # —— A4 阶跃突变 ——
    step_amp_xy: float = 0.25                       # 位置阶跃 [m] (↑1.7x)
    step_amp_theta: float = 0.18                    # 角度阶跃 [rad] (↑1.8x)

    # —— A5 重放攻击 ——
    replay_record_duration: float = 10.0             # 录制时长 [s] (攻击前)
    replay_loop: bool = True                         # 是否循环回放

    # —— A6 脉冲序列 ——
    pulse_amp: float = 0.20                         # 脉冲幅值 [m] (↑1.7x)
    pulse_period: float = 1.0                       # 脉冲周期 [s]
    pulse_duty: float = 0.3                         # 占空比

    # —— A7 扫频攻击 ——
    chirp_amp: float = 0.10                         # 幅值 [m] (↑1.7x)
    chirp_f_start: float = 0.1                      # 起始频率 [Hz]
    chirp_f_end: float = 4.0                        # 终止频率 [Hz]
    chirp_duration: float = 8.0                     # 扫频周期 [s]

    # —— A8 多频叠加 ——
    multitone_freqs: Tuple[float, ...] = (0.5, 1.7, 3.3)  # 非谐波频率 [Hz]
    multitone_amps:  Tuple[float, ...] = (0.08, 0.06, 0.04)  # 各频率幅值 [m]

    # —— 攻击持续时间 (None = 永不结束) ——
    attack_duration: float = None  # 攻击持续时间 [s]，None 表示持续到仿真结束


# ============================================================================
# 攻击生成器基类
# ============================================================================

class SensorAttack:
    """传感器加性攻击注入器
    
    统一接口，根据 attack_type 选择对应的攻击生成策略。
    攻击仅在 onset_time 之后生效，之前输出零向量。
    """

    # 攻击类型注册表
    ATTACK_TYPES = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
    ATTACK_NAMES = {
        'A0': 'Normal',
        'A1': 'Constant Bias',
        'A2': 'Sinusoidal Injection',
        'A3': 'Ramp Drift',
        'A4': 'Step Attack',
        'A5': 'Replay Attack',
        'A6': 'Pulse Train',
        'A7': 'Chirp Sweep',
        'A8': 'Multi-Tone Injection',
    }

    def __init__(self, attack_type: str = 'A0',
                 onset_time: float = 5.0,
                 config: AttackConfig = None,
                 seed: int = 42):
        """
        Args:
            attack_type: 攻击类型 'A0'~'A8'
            onset_time:  攻击开始时间 [s]，Normal下无效
            config:      攻击参数配置
            seed:        随机种子 (保证可复现)
        """
        if attack_type not in self.ATTACK_TYPES:
            raise ValueError(f"Unknown attack type: {attack_type}. "
                             f"Choose from {self.ATTACK_TYPES}")

        self.attack_type = attack_type
        self.onset_time = onset_time
        self.cfg = config if config is not None else AttackConfig()
        self.rng = np.random.RandomState(seed)

        # 攻击结束时间
        if self.cfg.attack_duration is not None and self.cfg.attack_duration > 0:
            self.offset_time = onset_time + self.cfg.attack_duration
        else:
            self.offset_time = float('inf')  # 永不结束

        # 攻击状态 (类型特定)
        self._replay_buffer = []             # A5 录制缓冲
        self._replay_idx = 0                 # A5 回放指针
        self._phase_acc = 0.0               # A7 扫频相位累加器

    def reset(self):
        """重置攻击状态"""
        self._replay_buffer = []
        self._replay_idx = 0
        self._phase_acc = 0.0

    def step(self, t: float) -> np.ndarray:
        """生成当前时刻的攻击注入向量（仅适用于加性攻击 A1-A4,A6-A8）

        Args:
            t: 当前仿真时间 [s]

        Returns:
            a_k: 攻击注入向量 (3,) — [a_x, a_y, a_theta]
        """
        if self.attack_type == 'A0' or t < self.onset_time or t >= self.offset_time:
            return np.zeros(3)

        if self.attack_type == 'A5':
            raise RuntimeError(
                "A5 重放攻击不使用 step() 接口，请在仿真回路中调用 inject(t, y_clean)")

        method = getattr(self, f'_attack_{self.attack_type}')
        return method(t)

    def inject(self, t: float, y_clean: np.ndarray) -> np.ndarray:
        """将攻击施加于干净测量值，返回受攻击的测量值

        这是仿真回路的统一入口。加性攻击返回 y_clean + a(k)，
        重放攻击返回录制的历史测量值。

        Args:
            t:       当前仿真时间 [s]
            y_clean: 干净传感器测量值 (true_state + noise) (3,)

        Returns:
            y_attacked: 受攻击后的测量值 (3,)
        """
        # —— 重放攻击：独立处理（录制与回放跨越攻击边界） ——
        if self.attack_type == 'A5':
            record_end = self.onset_time - 0.5  # 录制截止：攻击前 0.5s
            if t < record_end:
                self._replay_buffer.append(y_clean.copy())
                return y_clean.copy()
            elif t >= self.onset_time and self._replay_buffer:
                if self._replay_idx >= len(self._replay_buffer):
                    self._replay_idx = 0
                y_replay = self._replay_buffer[self._replay_idx].copy()
                self._replay_idx += 1
                return y_replay
            else:
                # 过渡期 (record_end <= t < onset_time): 返回干净值
                return y_clean.copy()

        # —— 其余攻击 ——
        if self.attack_type == 'A0' or t < self.onset_time or t >= self.offset_time:
            return y_clean.copy()

        a_k = self.step(t)
        return y_clean + a_k

    # ------------------------------------------------------------------
    # A1: 恒定偏置
    # ------------------------------------------------------------------
    def _attack_A1(self, t: float) -> np.ndarray:
        """a = [b_x, b_y, b_theta]，恒定不变"""
        return np.array([
            self.cfg.bias_xy[0],
            self.cfg.bias_xy[1],
            self.cfg.bias_theta
        ])

    # ------------------------------------------------------------------
    # A2: 正弦注入
    # ------------------------------------------------------------------
    def _attack_A2(self, t: float) -> np.ndarray:
        """a = A · sin(2π·f·t + φ)，单频振荡"""
        val = self.cfg.sin_amp * np.sin(
            2 * np.pi * self.cfg.sin_freq * t + self.cfg.sin_phase
        )
        return np.array([val, val * 0.7, val * 0.3])  # 各通道比例缩放

    # ------------------------------------------------------------------
    # A3: 斜坡漂移
    # ------------------------------------------------------------------
    def _attack_A3(self, t: float) -> np.ndarray:
        """a = rate · (t - t_onset)，线性增长"""
        dt = t - self.onset_time
        return np.array([
            self.cfg.ramp_rate_xy * dt,
            self.cfg.ramp_rate_xy * dt * 0.8,
            self.cfg.ramp_rate_theta * dt
        ])

    # ------------------------------------------------------------------
    # A4: 阶跃突变
    # ------------------------------------------------------------------
    def _attack_A4(self, t: float) -> np.ndarray:
        """a = A · 1(t >= t_onset)，瞬时跳变后维持"""
        return np.array([
            self.cfg.step_amp_xy,
            -self.cfg.step_amp_xy * 0.6,
            self.cfg.step_amp_theta
        ])

    # ------------------------------------------------------------------
    # A5: 重放攻击 (通过 inject() 接口实现，不通过 step())
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # A6: 脉冲序列
    # ------------------------------------------------------------------
    def _attack_A6(self, t: float) -> np.ndarray:
        """周期方波，占空比 pulse_duty"""
        dt = t - self.onset_time
        phase_in_period = (dt % self.cfg.pulse_period) / self.cfg.pulse_period
        if phase_in_period < self.cfg.pulse_duty:
            amp = self.cfg.pulse_amp
            return np.array([amp, amp * 0.5, amp * 0.3])
        else:
            return np.zeros(3)

    # ------------------------------------------------------------------
    # A7: 扫频攻击 (线性调频)
    # ------------------------------------------------------------------
    def _attack_A7(self, t: float) -> np.ndarray:
        """a = A · sin(φ(t))，f(t) 从 f_start 线性扫描至 f_end"""
        dt = t - self.onset_time
        # 线性调频: 瞬时频率 f(t) = f0 + (f1-f0) * dt/T
        f_inst = (self.cfg.chirp_f_start +
                  (self.cfg.chirp_f_end - self.cfg.chirp_f_start)
                  * min(dt, self.cfg.chirp_duration) / self.cfg.chirp_duration)
        # 相位 = 积分频率
        self._phase_acc += 2 * np.pi * f_inst * 0.05  # Ts=0.05 积分
        val = self.cfg.chirp_amp * np.sin(self._phase_acc)
        return np.array([val, val * 0.6, val * 0.4])

    # ------------------------------------------------------------------
    # A8: 多频叠加注入
    # ------------------------------------------------------------------
    def _attack_A8(self, t: float) -> np.ndarray:
        """a = Σ A_i · sin(2π·f_i·(t-t_onset))，多个非谐波频率叠加"""
        dt = t - self.onset_time
        val = sum(
            amp * np.sin(2 * np.pi * freq * dt)
            for freq, amp in zip(self.cfg.multitone_freqs, self.cfg.multitone_amps)
        )
        return np.array([val, val * 0.6, val * 0.4])

    @staticmethod
    def print_summary():
        """输出攻击类型汇总"""
        print("\n" + "=" * 60)
        print("Sensor Attack Catalog (传感器加性攻击目录)")
        print("=" * 60)
        for atype, name in SensorAttack.ATTACK_NAMES.items():
            print(f"  {atype}: {name}")
        print("=" * 60 + "\n")


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    SensorAttack.print_summary()

    # 测试加性攻击 (A0-A4, A6-A8)
    print("Testing additive attacks (step interface)...\n")
    for atype in ['A0', 'A1', 'A2', 'A3', 'A4', 'A6', 'A7', 'A8']:
        att = SensorAttack(attack_type=atype, onset_time=5.0, seed=42)
        vals = []
        for t in np.arange(5.0, 7.0, 0.05):
            vals.append(att.step(t))
        vals = np.array(vals)
        print(f"  {atype} ({SensorAttack.ATTACK_NAMES[atype]}):")
        print(f"    Range x: [{vals[:,0].min():.4f}, {vals[:,0].max():.4f}]")
        print(f"    Range y: [{vals[:,1].min():.4f}, {vals[:,1].max():.4f}]")
        print(f"    Range θ: [{vals[:,2].min():.4f}, {vals[:,2].max():.4f}]")

    # 测试重放攻击 (inject 接口)
    print("\nTesting A5 Replay Attack (inject interface)...")
    att = SensorAttack(attack_type='A5', onset_time=5.0, seed=42)
    y_clean_hist = []
    y_att_hist = []
    for t in np.arange(0.0, 10.0, 0.05):
        y_clean = np.array([np.sin(t), np.cos(t), 0.1 * t])
        y_att = att.inject(t, y_clean)
        y_clean_hist.append(y_clean)
        y_att_hist.append(y_att)
    y_clean_hist = np.array(y_clean_hist)
    y_att_hist = np.array(y_att_hist)
    # 验证: 攻击前注入=干净值, 攻击后注入≠干净值
    pre_mask = np.arange(len(y_att_hist)) < 100  # t < 5s
    pre_match = np.allclose(y_att_hist[pre_mask], y_clean_hist[pre_mask], atol=1e-10)
    post_diff = np.mean(np.abs(y_att_hist[~pre_mask] - y_clean_hist[~pre_mask]))
    print(f"  攻击前注入==干净值: {pre_match}")
    print(f"  攻击后偏差均值:     {post_diff:.4f}m")
    print("\nAll attacks verified.")
