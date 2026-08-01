"""
serial_driver.py -- 네 아두이노 1바이트 프로토콜 전용 드라이버.

[펌웨어 프로토콜] (v5.x 통합 펌웨어)
  비트7=1 : 조향.  v=(b&0x7F) 0..127, 64=중립. pot 변환·캘리브레이션은 펌웨어 전담.
  비트7=0 : 모터.  0x00 정지, 0x01~0x3F 전진 b*4, 0x41~0x7F 후진 (b&0x3F)*4
  0x40    : 제어 이스케이프. 다음 1바이트가 명령 (0x01=초음파 ON, 0x00=OFF)
            맥은 정지로 0x40을 절대 안 쓰므로(0x00만 씀) 안전하게 재활용.

[초음파 스트리밍] 통합 펌웨어는 초음파가 기본 OFF다(주행 미션 루프 보호).
  장애물 미션은 시작 시 ON 명령을 보내고, 아두이노 리셋에 대비해 주기적으로
  재전송한다(_us_resend_s).

[두 가지 조향 모드]  (config "steer_mode")
  "continuous" : 7비트 전체 해상도. 펌웨어 v3.0(연속 매핑)과 세트. ← 현재 사용
  "bang3"      : 구 펌웨어(v2.1, 3단 양자화)용 레거시.
                 히스테리시스(bang_on/bang_off)로 채터링 방지.

[하트비트] 펌웨어 워치독(400ms)이 있으므로, 명령이 안 바뀌어도 150ms마다
  마지막 바이트를 재전송한다. 이게 없으면 직선 주행 중(명령 불변) 워치독이
  모터를 정지시키고, 중복전송 억제 때문에 영영 재시동이 안 되는 버그가 있었음.

main.py 와의 인터페이스는 이전과 동일: send_drive(pwm, steer_norm) / send_brake()
"""

import glob
import time

try:
    import serial
except ImportError:
    serial = None


class MegaLink:
    def __init__(self, cfg, dry_run=False):
        self.cfg = cfg
        self.dry = dry_run or serial is None
        self.ser = None
        self.mode = cfg.get("steer_mode", "bang3")
        self.bang_state = 'N'          # 히스테리시스 상태 기억
        self._last_steer_byte = None   # 같은 바이트 반복 전송 억제
        self._last_drive_byte = None
        # 하트비트: 펌웨어 워치독(400ms)이 통신 두절로 오인해 모터를 꺼버리지
        # 않도록, 명령이 안 바뀌어도 이 주기마다 마지막 바이트를 재전송한다.
        self._heartbeat_s = 0.15
        self._last_tx = 0.0
        # 텔레메트리 (펌웨어가 "U<idx>:<cm>" / "P:<pot>" 라인으로 보냄)
        self.us = {}        # {센서 인덱스: 최신 거리 cm}
        self.pot = None     # 조향 pot 실측값 (풀락 도달 확인용)
        self._rxbuf = ""
        # 초음파 스트리밍: 통합 펌웨어는 기본 OFF라 켜줘야 한다. 아두이노가
        # 도중에 리셋되면 다시 OFF로 돌아가므로 주기적으로 재전송한다.
        self._us_stream = False
        self._us_resend_s = 1.0
        self._last_us_enable = 0.0
        if not self.dry:
            port = cfg["serial_port"]
            if "*" in port:
                hits = glob.glob(port)
                if not hits:
                    raise RuntimeError(f"시리얼 포트 못 찾음: {port} "
                                       "(ls /dev/tty.usbmodem* 확인)")
                port = hits[0]
            self.ser = serial.Serial(port, int(cfg["baud"]), timeout=0)
            time.sleep(2.0)            # Mega 리셋 대기
            print(f"[serial] connected {port} mode={self.mode}")
        else:
            print(f"[serial] DRY-RUN mode={self.mode}")
        # 장애물 미션: 초음파 스트리밍 켜기 (통합 펌웨어 기본 OFF)
        self.set_ultrasonic(True)

    # ── 인코딩 ──────────────────────────────────────────────
    def _steer_byte(self, steer_norm):
        if self.mode == "continuous":
            # 정규화 조향(-1..1)을 v=0..127로 선형 변환 (64=중립).
            # pot 값 계산·비대칭 중립 보정은 전부 펌웨어가 담당.
            v = int(round((steer_norm + 1.0) * 63.5))
            v = max(0, min(127, v))
            return 0x80 | v

        # bang3 + 히스테리시스: on 임계 넘으면 꺾고, off 밑으로 와야 풀림
        on = float(self.cfg.get("bang_on", 0.35))
        off = float(self.cfg.get("bang_off", 0.15))
        s = steer_norm
        if self.bang_state == 'N':
            if s >= on:
                self.bang_state = 'L'
            elif s <= -on:
                self.bang_state = 'R'
        elif self.bang_state == 'L':
            if s < off:
                self.bang_state = 'N'
        elif self.bang_state == 'R':
            if s > -off:
                self.bang_state = 'N'
        v = {'R': 0x00, 'N': 0x40, 'L': 0x7F}[self.bang_state]
        return 0x80 | v

    @staticmethod
    def _drive_byte(pwm):
        pwm = int(pwm)
        if pwm == 0:
            return 0x00
        if pwm > 0:
            return min(0x3F, max(1, round(pwm / 4)))
        return 0x40 | min(0x3F, max(1, round(-pwm / 4)))

    # ── 전송 (main.py 인터페이스 동일) ───────────────────────
    def send_drive(self, drive_pwm, steer_norm):
        sb = self._steer_byte(steer_norm)
        db = self._drive_byte(drive_pwm)
        out = bytearray()
        if sb != self._last_steer_byte:
            out.append(sb)
            self._last_steer_byte = sb
        if db != self._last_drive_byte:
            out.append(db)
            self._last_drive_byte = db
        if out:
            self._write(bytes(out))
        elif time.time() - self._last_tx > self._heartbeat_s:
            self._write(bytes([sb, db]))   # 워치독 하트비트
        return sb, db

    def send_brake(self):
        # 중립 바이트는 모드에 따라 다름: bang3=0xC0, continuous=steer0 인코딩(pot 481)
        if self.mode == "continuous":
            nb = self._steer_byte(0.0)
        else:
            self.bang_state = 'N'
            nb = 0x80 | 0x40
        self._write(bytes([0x00, nb]))
        self._last_drive_byte = 0x00
        self._last_steer_byte = nb

    def set_ultrasonic(self, on):
        """통합 펌웨어에 초음파 스트리밍 ON/OFF 명령 (0x40 이스케이프)."""
        self._us_stream = bool(on)
        self._last_us_enable = time.time()
        self._write(bytes([0x40, 0x01 if on else 0x00]))

    def _write(self, data):
        self._last_tx = time.time()
        if self.dry:
            return
        try:
            self.ser.write(data)
        except Exception as e:
            print("[serial] write err:", e)

    def read_telemetry(self):
        """펌웨어 초음파 라인("U<idx>:<cm>") 파싱 → 최신값 dict {idx: cm}.
        비차단. dry-run에서는 self.us를 외부(시뮬)에서 채워도 됨."""
        if self.dry or not self.ser:
            return self.us
        # 아두이노 리셋 대비: 초음파 스트리밍 활성 명령 주기적 재전송
        if self._us_stream and (time.time() - self._last_us_enable > self._us_resend_s):
            self.set_ultrasonic(True)
        try:
            n = self.ser.in_waiting
            if n:
                self._rxbuf += self.ser.read(n).decode(errors="ignore")
                # 버퍼 폭주 방지 (줄바꿈 없는 쓰레기 데이터)
                if len(self._rxbuf) > 500:
                    self._rxbuf = self._rxbuf[-100:]
                while "\n" in self._rxbuf:
                    line, self._rxbuf = self._rxbuf.split("\n", 1)
                    line = line.strip()
                    if line.startswith("U") and ":" in line:
                        try:
                            i, cm = line[1:].split(":", 1)
                            self.us[int(i)] = int(cm)
                        except ValueError:
                            pass
                    elif line.startswith("P:"):
                        try:
                            self.pot = int(line[2:])
                        except ValueError:
                            pass
        except Exception as e:
            print("[serial] read err:", e)
        return self.us

    def close(self):
        if not self.dry and self.ser:
            self.send_brake()
            self.ser.close()
