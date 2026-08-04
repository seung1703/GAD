"""
arduino_iface.py -- 차량 명령 인터페이스 (스펙 7-5).

스펙은 "스텁"을 요구했지만, 이 차는 통합 펌웨어(firmware/firmware.ino)를 쓰므로
스텁 대신 obstacle 에서 가져온 serial_driver.MegaLink(1바이트 프로토콜)에
실제로 연결한다. 시그니처는 스펙대로 유지:
    set_steering(deg), motor_forward(speed), motor_reverse(speed), motor_stop()

프로토콜 요약: 파이썬은 정규화 조향(-1..1)만 보내고 pot 캘리브레이션은 펌웨어
전담. 조향+모터를 한 상태로 묶어 매 호출 전송(하트비트는 MegaLink가 담당).

--dry 면 시리얼 없이 명령 로그만.
"""

import serial_driver
import config as C


def pot_to_norm(pot):
    """pot 목표값 → 정규화 조향(-1..1).

    펌웨어 processByte() 의 구간별 map 을 역산한다:
        v >= 64 : map(v, 64, 127, NEUTRAL, LEFT)
        v <  64 : map(v,  0,  64, RIGHT,   NEUTRAL)
    그리고 serial_driver._steer_byte() 가 v = round((norm+1) * 63.5) 이므로
    norm = v / 63.5 - 1 로 되돌린다.
    """
    pot = max(C.POT_RIGHT, min(C.POT_LEFT, int(pot)))
    if pot >= C.POT_NEUTRAL:
        span = max(1, C.POT_LEFT - C.POT_NEUTRAL)
        v = 64.0 + (pot - C.POT_NEUTRAL) * 63.0 / span
    else:
        span = max(1, C.POT_NEUTRAL - C.POT_RIGHT)
        v = (pot - C.POT_RIGHT) * 64.0 / span
    return max(-1.0, min(1.0, v / 63.5 - 1.0))


class CarCommander:
    def __init__(self, dry_run=False):
        self.link = serial_driver.MegaLink(C.serial_cfg(), dry_run=dry_run)
        self._steer = 0.0      # 실제 전송값 (PARK_STEER_SIGN 적용 후)
        self._steer_cmd = 0.0  # 의도한 조향 (+=좌). 시뮬/디버그가 읽는다
        self._motor = 0        # +전진 / -후진 PWM

    # ── 조향 ──────────────────────────────────────────────────
    def set_steering_norm(self, norm):
        """정규화 조향(-1..1) 직접 명령. +=좌, -=우 (PARK_STEER_SIGN 적용 전 기준)."""
        self._steer_cmd = max(-1.0, min(1.0, norm))   # 부호보정 전 '의도한' 조향
        self._steer = self._steer_cmd * C.PARK_STEER_SIGN
        self._send()

    def steer_full_left(self):
        self.set_steering_norm(+1.0)

    def steer_full_right(self):
        self.set_steering_norm(-1.0)

    def steer_center(self):
        self.set_steering_norm(0.0)

    def steer_pot(self, pot):
        """조향을 **pot 목표값**으로 지정한다 (호밍·유격보정 경로 전용).

        같은 "중립"이라도 좌측에서 내려오느냐 우측에서 올라오느냐에 따라
        펌웨어 데드밴드 안 어디에 설지가 달라진다. 그 차이를 흡수하려면
        정규화값(-1..1)이 아니라 pot 으로 직접 겨눠야 한다.
        """
        self.steer_raw_norm(pot_to_norm(pot))

    def steer_raw_norm(self, norm):
        """★호밍 전용★ PARK_STEER_SIGN 을 적용하지 않는 직접 조향 명령.

        호밍 목표는 pot 값(= 펌웨어/하드웨어 좌표)에서 역산한 것이라, 의도↔
        하드웨어 부호 보정을 한 번 더 걸면 반대쪽으로 간다. 일반 주행 명령은
        반드시 set_steering_norm() 을 쓸 것.
        """
        self._steer_cmd = max(-1.0, min(1.0, float(norm)))
        self._steer = self._steer_cmd
        self._send()

    # ── 구동 ──────────────────────────────────────────────────
    def motor_forward(self, speed):
        self._motor = int(abs(speed))
        self._send()

    def motor_reverse(self, speed):
        self._motor = -int(abs(speed))
        self._send()

    def motor_stop(self):
        """모터만 정지, 조향은 유지 (기동 중 풀조향 홀드)."""
        self._motor = 0
        self._send()

    def brake(self):
        """완전 정지 + 조향 중립 (미션 종료/비상)."""
        self._motor = 0
        self._steer = 0.0
        self._steer_cmd = 0.0
        self.link.send_brake()

    def command(self):
        """(모터 PWM, 의도한 조향) — 시뮬레이터/디버그용."""
        return self._motor, self._steer_cmd

    # ── 하트비트/전송 ─────────────────────────────────────────
    def tick(self):
        """제어 루프 매 프레임 호출 — 명령 불변이어도 워치독 유지.

        같이 텔레메트리도 비운다. 펌웨어가 "P:<pot>" 을 10Hz 로 계속 보내는데
        미션 중엔 아무도 안 읽어서, 조향이 실제로 목표에 도달했는지 확인할
        방법이 없었고 시리얼 버퍼에 쌓이기만 했다.
        """
        self._send()
        self.link.read_telemetry()

    @property
    def pot(self):
        """조향 pot 실측값 (없으면 None). tick() 이 갱신한다."""
        return getattr(self.link, "pot", None)

    def _send(self):
        # MegaLink.send_drive(pwm, steer_norm): 음수 pwm=후진. 조향은 이미
        # PARK_STEER_SIGN 적용됨. send_drive 는 정규화값을 v바이트로 인코딩.
        self.link.send_drive(self._motor, self._steer)

    def close(self):
        self.brake()
        self.link.close()
