"""
control.py -- PD 제어 (실차판).

입력: 정규화 측방오차 err_norm ∈ [-1,1]
출력: 정규화 조향 명령 steer_norm ∈ [-1,1]  → serial_driver 가 전송

P항: 현재 오차에 비례해 꺾는다.
D항: 오차의 변화율. 코너 탈출에서 err가 빠르게 줄어들 때 미리 핸들을
     풀기 시작하게 해서, err=0을 지나 안쪽(중앙선)으로 파고드는
     오버슈트를 잡는다. (프레임 단위 미분 + EMA 필터로 노이즈 완화)
"""


class LaneFollowController:
    def __init__(self, kp=1.4, kd=6.0, steer_sign=1, right_gain=1.0):
        self.kp = float(kp)
        self.kd = float(kd)
        self.sign = int(steer_sign)
        self.right_gain = float(right_gain)  # 우회전(음수 조향)만 추가 배율
        self.prev_err = None
        self.d_filt = 0.0

    def compute(self, err_norm):
        d = 0.0 if self.prev_err is None else err_norm - self.prev_err
        self.prev_err = err_norm
        self.d_filt = 0.7 * self.d_filt + 0.3 * d   # 미분 노이즈 완화
        s = self.sign * (self.kp * err_norm + self.kd * self.d_filt)
        if s < 0:                # 최종 명령 기준 음수 = 우회전
            s *= self.right_gain
        return max(-1.0, min(1.0, s))

    def reset(self):
        self.prev_err = None
        self.d_filt = 0.0


class RecoverySpeedRamp:
    """Raise normal speed gradually after an avoidance maneuver."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.active = False
        self.elapsed_s = 0.0
        self.last_update_ts = None

    @property
    def duration_s(self):
        return max(0.0, float(self.cfg.get("recovery_ramp_duration_s", 3.0)))

    @property
    def start_pwm(self):
        return int(
            self.cfg.get(
                "recovery_ramp_start_pwm",
                self.cfg.get("change_pwm", 70),
            )
        )

    def start(self, now):
        self.active = self.duration_s > 0.0
        self.elapsed_s = 0.0
        self.last_update_ts = float(now)

    def reset(self):
        self.active = False
        self.elapsed_s = 0.0
        self.last_update_ts = None

    def speed(self, now, target_pwm, motion_enabled=True):
        target_pwm = int(target_pwm)
        if not self.active:
            return target_pwm

        now = float(now)
        if self.last_update_ts is None:
            self.last_update_ts = now
        dt = max(0.0, now - self.last_update_ts)
        self.last_update_ts = now
        if motion_enabled:
            self.elapsed_s += dt

        duration = self.duration_s
        if duration <= 0.0 or self.elapsed_s >= duration:
            self.active = False
            self.elapsed_s = duration
            return target_pwm

        start_pwm = min(self.start_pwm, target_pwm)
        progress = min(1.0, self.elapsed_s / duration)
        return int(round(start_pwm + (target_pwm - start_pwm) * progress))
