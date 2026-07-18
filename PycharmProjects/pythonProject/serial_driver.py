"""
serial_driver.py -- 우리 아두이노 텍스트 프로토콜(S/D/R/X)용 드라이버.

[펌웨어 프로토콜]
  S<0~180>\n : 조향 (0=우, 90=중립, 180=좌)
  D<0~255>\n : 전진 PWM
  R<0~255>\n : 후진 PWM
  X\n        : 정지

[하트비트] 펌웨어 워치독(500ms)이 있으므로, 명령이 안 바뀌어도
  150ms마다 마지막 명령을 재전송한다.

main_model.py 와의 인터페이스: send_drive(pwm, steer_norm) / send_brake()
steer_norm(-1..1)을 S각도(0..180)로 변환해서 전송한다.
"""

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
        self._last_steer_cmd = None
        self._last_drive_cmd = None
        self._heartbeat_s = 0.15
        self._last_tx = 0.0
        if not self.dry:
            port = cfg["serial_port"]
            self.ser = serial.Serial(port, int(cfg["baud"]), timeout=0)
            time.sleep(2.0)            # Mega 리셋 대기
            print(f"[serial] connected {port}")
        else:
            print("[serial] DRY-RUN")

    # ── 변환 ──────────────────────────────────────────────
    @staticmethod
    def _steer_cmd(steer_norm):
        # steer_norm(-1..1) → S각도(0..180). +1=좌(180), -1=우(0)
        angle = int(round((steer_norm + 1.0) * 90))
        angle = max(0, min(180, angle))
        return f"S{angle}"

    @staticmethod
    def _drive_cmd(pwm):
        pwm = int(pwm)
        if pwm == 0:
            return "X"
        if pwm > 0:
            return f"D{min(255, pwm)}"
        return f"R{min(255, -pwm)}"

    # ── 전송 (main_model.py 인터페이스 동일) ───────────────
    def send_drive(self, drive_pwm, steer_norm):
        sc = self._steer_cmd(steer_norm)
        dc = self._drive_cmd(drive_pwm)
        out = []
        if sc != self._last_steer_cmd:
            out.append(sc)
            self._last_steer_cmd = sc
        if dc != self._last_drive_cmd:
            out.append(dc)
            self._last_drive_cmd = dc
        if out:
            self._write(out)
        elif time.time() - self._last_tx > self._heartbeat_s:
            self._write([sc, dc])   # 워치독 하트비트
        return sc, dc

    def send_brake(self):
        self._write(["X", "S90"])
        self._last_drive_cmd = "X"
        self._last_steer_cmd = "S90"

    def _write(self, cmds):
        self._last_tx = time.time()
        if self.dry:
            return
        try:
            for c in cmds:
                self.ser.write((c + "\n").encode())
        except Exception as e:
            print("[serial] write err:", e)

    def read_telemetry(self):
        return None

    def close(self):
        if not self.dry and self.ser:
            self.send_brake()
            self.ser.close()