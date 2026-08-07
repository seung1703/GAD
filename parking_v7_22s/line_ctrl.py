"""
line_ctrl.py -- 선까지의 거리 → 조향 (P+D). 놓쳤을 때 유지 로직 포함.

I/O 가 없는 순수 상태기계라 하드웨어 없이 그대로 검증할 수 있다.

[동작]
  1) 처음 선을 찾으면 몇 프레임 모아 **중앙값을 기준각으로 잠근다(latch)**.
     그 전까지는 조향 0(직진) — 기준을 모르는 상태에서 꺾으면 안 되니까.
     한 번 잠그면 그 주행 내내 안 바뀐다. 선을 놓쳤다 다시 찾아도
     **다시 잡지 않는다** — 다른 선을 새 기준으로 삼아버리면 진행 방향이
     통째로 바뀐다.

  2) 선이 보이면 기준각과의 차이에 비례해 조향한다(P 제어).

  3) 선을 놓치면 **직전 조향값을 그대로 유지**하며 계속 간다.
     LOST_HOLD_S 동안은 그대로 두고, 그 뒤엔 서서히 0(직진)으로 되돌린다.
     ★멈추지 않는다★ — 선이 끝나서 안 보이는 게 정상 상황이기 때문이다.
     계속 꺾인 채로 두면 원을 그리므로 서서히 펴는 것이다.
"""

import collections

import numpy as np

import config as C


def apply_deadband(u):
    """★조향 데드밴드 보상★ 계산된 명령 u 를 **실제로 바퀴가 도는** 값으로.

    펌웨어는 |목표-현재| <= 데드밴드(8카운트) 면 조향 모터를 끊는다.
    steer_norm 0.09 는 pot 으로 6카운트뿐이라 바퀴가 아예 안 움직인다.
    작은 조향이 전부 무시되면 "선을 보고 있는데 조향을 안 한다" 가 된다.

    그래서 0 은 아니지만 작은 요구는 먹히는 최소값까지 밀어올린다.
    (조향 모터에 최소 기동 PWM 을 깔아주는 것과 같은 원리)
    아주 작은 요구는 0 으로 눌러서 최소값 근처에서 덜덜 떠는 걸 막는다.
    """
    if C.STEER_MIN_EFF <= 0:
        return u
    a = abs(u)
    if a < C.STEER_EPS:
        return 0.0
    if a < C.STEER_MIN_EFF:
        return float(np.sign(u) * C.STEER_MIN_EFF)
    return u


class LateralPD:
    """★선까지의 거리로 직진 유지★ (config.CONTROL = "lateral")

    왜 각도가 아니라 거리인가:
      차가 선에 대해 비스듬히 서 있어도 **직진하는 동안 선과의 각도는 안 변한다.**
      그래서 각도 오차가 계속 0 이고 조향도 0 이 된다 — 비스듬한 채로 곧게
      멀어져 가는데 "각도는 유지 중"이라 판단해 버린다. 각도만으로는 이 상황을
      원리적으로 감지할 수 없다.
      반면 **거리는 멀어지면 반드시 변한다.** 그래서 거리로 잡는다.

    P + D 를 쓰는 이유:
      거리에 P 만 걸면 목표를 지나칠 때 브레이크가 없어 좌우로 출렁인다.
      D(거리가 변하는 속도)가 그 감쇠를 맡는다. 선에 빠르게 다가가는 중이면
      아직 목표에 못 미쳤어도 미리 꺾음을 푼다.
      각도를 안 쓰므로 "몇 도가 나란한 것인지" 캘리브레이션이 필요 없다.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.xref = None         # 유지할 목표 횡위치(px)
        self.locked = False
        self._buf = []
        self._prev_x = None
        self._d = 0.0            # 거리 변화율 (EMA)
        self.last_steer = 0.0
        self.last_seen = None
        self.e_pos = 0.0
        self.diverged = False        # 발산 감지 래치
        self._best_e = None          # 지금까지 본 가장 작은 |오차|
        self._best_t = None
        self._announced = False
        if C.LAT_TARGET_PX > 0:  # ★고정 목표★ 시작 위치와 무관하게 이 거리로 수렴
            self.xref = float(C.LAT_TARGET_PX)
            self.locked = True

    # 기존 컨트롤러와 이름을 맞춰 둔다 (main 이 그대로 쓴다)
    @property
    def ref(self):
        return self.xref

    def relock(self):
        self.xref = None
        self.locked = False
        self._buf = []
        self._prev_x = None
        self._d = 0.0
        self.diverged = False
        self._best_e = None
        self._best_t = None
        self._announced = False

    def update(self, angle, now, x=None):
        if x is not None:
            self.last_seen = now

            if not self.locked:
                self._buf.append(float(x))
                if len(self._buf) < C.REF_LOCK_N:
                    return 0.0, f"locking {len(self._buf)}/{C.REF_LOCK_N}"
                self.xref = float(np.median(self._buf))
                self.locked = True
                print(f"[ref] 목표 횡위치 {self.xref:.0f}px 확정. "
                      "이 거리를 유지하며 직진합니다")

            # 고정 목표 모드면 첫 검출에서 "얼마나 벗어난 상태로 출발했는지"를
            # 한 번 알린다. 시작 위치별로 이 값이 다르게 나와야 정상이다.
            if not self._announced:
                self._announced = True
                if C.LAT_TARGET_PX > 0:
                    print(f"[ref] 목표 횡위치 {self.xref:.0f}px (고정). "
                          f"현재 {float(x):.0f}px → {float(x) - self.xref:+.0f}px "
                          "벗어남, 보정 시작")

            # P: 목표에서 얼마나 벗어났나
            self.e_pos = float(x) - self.xref
            # D: 얼마나 빠르게 벗어나는 중인가 (프레임간 변화, EMA 로 잡음 완화)
            raw_d = 0.0 if self._prev_x is None else float(x) - self._prev_x
            self._prev_x = float(x)
            self._d = 0.7 * self._d + 0.3 * raw_d

            # ── 발산 감지 ──
            # 부호(LAT_SIGN)가 반대면 오차가 커질수록 더 세게 잘못 꺾어서
            # 선으로 돌진한다. 오차가 줄어들 기미 없이 계속 커지면 멈춘다.
            a = abs(self.e_pos)
            if self._best_e is None or a < self._best_e - 2:
                self._best_e, self._best_t = a, now
            if (C.DIVERGE_PX > 0 and a > C.DIVERGE_PX
                    and self._best_t is not None
                    and now - self._best_t > C.DIVERGE_S):
                if not self.diverged:
                    self.diverged = True
                    print(f"[!] 발산 감지 — 오차 {self.e_pos:+.0f}px 가 "
                          f"{C.DIVERGE_S:.1f}s 동안 안 줄어듦. 정지합니다.\n"
                          f"    LAT_SIGN 을 {-C.LAT_SIGN} 로 뒤집어 보세요 "
                          f"(현재 {C.LAT_SIGN})")
                self.last_steer = 0.0
                return 0.0, f"DIVERGED err {self.e_pos:+.0f}px - flip LAT_SIGN"

            u = C.LAT_KP * self.e_pos + C.LAT_KD * self._d
            u = apply_deadband(u)          # ← 없으면 작은 조향이 다 무시된다
            steer = float(np.clip(u * C.LAT_SIGN, -C.STEER_MAX, C.STEER_MAX))
            self.last_steer = steer
            return steer, (f"err {self.e_pos:+.0f}px"
                           f"({C.LAT_KP * self.e_pos * C.LAT_SIGN:+.2f})"
                           f"  rate {self._d:+.1f}px/f"
                           f"({C.LAT_KD * self._d * C.LAT_SIGN:+.2f})")

        # ── 선이 안 보임 — 직전 조향 유지 후 서서히 직진으로 ──
        if self.last_seen is None:
            return 0.0, "no line yet - straight"
        lost = now - self.last_seen
        if lost <= C.LOST_HOLD_S:
            return self.last_steer, f"lost {lost:.1f}s - hold"
        f = max(0.0, 1.0 - (lost - C.LOST_HOLD_S) / max(1e-6, C.LOST_DECAY_S))
        return self.last_steer * f, f"lost {lost:.1f}s - fade {f:.0%}"


def make_controller():
    return LateralPD()
