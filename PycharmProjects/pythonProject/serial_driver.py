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
        self._heartbeat_s = 0.3
        self._last_tx = 0.0
        if not self.dry:
            port = cfg["serial_port"]
            self.ser = serial.Serial(port, int(cfg["baud"]),
                                     timeout=0.05, write_timeout=0.5)
            time.sleep(2.0)  # Mega 리셋 대기
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

        # 조향: 각도가 의미 있게(3 이상) 바뀔 때만 전송 → 시리얼 부하 감소
        if self._last_steer_cmd is None:
            send_steer = True
        else:
            try:
                prev_angle = int(self._last_steer_cmd[1:])
                new_angle = int(sc[1:])
                send_steer = abs(new_angle - prev_angle) >= 3
            except ValueError:
                send_steer = True

        if send_steer:
            out.append(sc)
            self._last_steer_cmd = sc
        if dc != self._last_drive_cmd:
            out.append(dc)
            self._last_drive_cmd = dc

        if out:
            self._write(out)
        elif time.time() - self._last_tx > self._heartbeat_s:
            self._write([sc, dc])  # 워치독 하트비트
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
        """아두이노가 보낸 'U,fL,fC,fR,bL,bC,bR\n' 줄을 파싱해서 반환.
           새 줄 없으면 None. 절대 블로킹되지 않도록 방어적으로 처리."""
        if self.dry or not self.ser:
            return None
        if not hasattr(self, '_rx_buf'):
            self._rx_buf = ""

        try:
            n = self.ser.in_waiting
        except Exception:
            return None
        if n <= 0:
            return None

        n = min(n, 512)   # 한 번에 너무 많이 안 읽음 (안전 상한)
        try:
            data = self.ser.read(n)
        except Exception:
            return None
        if not data:
            return None

        self._rx_buf += data.decode(errors="ignore")
        # 버퍼가 비정상적으로 커지면 (파싱 안 되는 쓰레기 누적) 리셋
        if len(self._rx_buf) > 4096:
            self._rx_buf = ""
            return None

        result = None
        while "\n" in self._rx_buf:
            line, self._rx_buf = self._rx_buf.split("\n", 1)
            line = line.strip()
            if line.startswith("U,"):
                parts = line.split(",")
                if len(parts) == 7:
                    try:
                        fL, fC, fR, bL, bC, bR = map(int, parts[1:])
                        result = {"fL": fL, "fC": fC, "fR": fR,
                                 "bL": bL, "bC": bC, "bR": bR}
                    except ValueError:
                        pass
        return result

    def close(self):
        if not self.dry and self.ser:
            self.send_brake()
            self.ser.close()