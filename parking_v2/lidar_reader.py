"""
lidar_reader.py -- 스캔 수신 스레드 + 로거 + 리플레이어 (스펙 7-1, 7-2).

iter_scans 는 버퍼가 밀리면 과거 프레임을 돌려주므로, 수신 스레드가 항상
"최신 프레임만" 공유 변수에 갱신하고 제어 루프는 그것만 읽는다. (안 그러면
이동 중 제어가 1초 전 세상을 봄)

rplidar 미설치/미연결이어도 import 되도록 가드. 리플레이는 라이다 없이 동작.
"""

import pickle
import threading
import time

try:
    from rplidar import RPLidar
except ImportError:
    RPLidar = None

import config as C


class LidarReader:
    """백그라운드 스레드로 최신 스캔만 유지. scan = [(quality, angle, dist), ...]."""

    def __init__(self, port=C.LIDAR_PORT, baud=C.LIDAR_BAUD, record_path=None):
        if RPLidar is None:
            raise ImportError("rplidar 필요: pip install rplidar-roboticia")
        self.port, self.baud = port, baud
        self._lidar = None
        self._latest = None          # (timestamp, scan)
        self._lock = threading.Lock()
        self._run = False
        self._thread = None
        self._log = open(record_path, "wb") if record_path else None

    def start(self):
        self._lidar = RPLidar(self.port, baudrate=self.baud)
        self._lidar.clean_input()    # ★ Descriptor length mismatch 예방 ★
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[lidar] started {self.port}@{self.baud}")

    def _loop(self):
        try:
            for scan in self._lidar.iter_scans():
                if not self._run:
                    break
                ts = time.time()
                with self._lock:
                    self._latest = (ts, scan)
                if self._log:
                    pickle.dump((ts, scan), self._log)
        except Exception as e:
            print("[lidar] loop err:", e)

    def get_latest(self):
        """(timestamp, scan) 또는 None. 항상 가장 최근 프레임."""
        with self._lock:
            return self._latest

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._lidar:
            try:
                self._lidar.stop()
                self._lidar.stop_motor()
                self._lidar.disconnect()
            except Exception:
                pass
        if self._log:
            self._log.close()
        print("[lidar] stopped")


class ReplayReader:
    """기록 pickle 파일을 재생 (라이다 없이 알고리즘만 오프라인 디버깅).
    LidarReader 와 같은 get_latest() 인터페이스."""

    def __init__(self, path, realtime=True, loop=False):
        self.frames = []
        with open(path, "rb") as f:
            while True:
                try:
                    self.frames.append(pickle.load(f))
                except EOFError:
                    break
        self.realtime = realtime      # True=원래 타임스탬프 간격대로 재생
        self.loop = loop
        self._i = 0
        self._t0_wall = None
        self._t0_data = self.frames[0][0] if self.frames else 0
        print(f"[replay] {len(self.frames)} frames from {path}")

    def start(self):
        self._t0_wall = time.time()

    def get_latest(self):
        if not self.frames:
            return None
        if self.realtime:
            elapsed = time.time() - self._t0_wall
            while (self._i + 1 < len(self.frames)
                   and self.frames[self._i + 1][0] - self._t0_data <= elapsed):
                self._i += 1
        else:
            self._i = min(self._i + 1, len(self.frames) - 1)
        if self._i >= len(self.frames) - 1 and self.loop:
            self._i = 0
            self._t0_wall = time.time()
        return self.frames[self._i]

    def stop(self):
        pass

    def __len__(self):
        return len(self.frames)
