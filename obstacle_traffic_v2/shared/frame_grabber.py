"""
frame_grabber.py -- 카메라를 별도 스레드에서 계속 읽어 '가장 최신 프레임'만 들고 있는다

왜 필요한가 (M3 맥북 실측):
  메인 루프에서 카메라 두 대를 `read()` 하면 **한 바퀴에 37ms** 를 카메라 대기에만
  쓴다. `read()` 는 다음 프레임이 도착할 때까지 블로킹하기 때문이다. 30fps 카메라
  두 대면 그 자체로 루프 상한이 정해져 버린다.

  같은 카메라를 스레드에서 계속 비우면서 최신 프레임만 남기면 메인 루프의 읽기
  비용은 **0.0ms** 가 된다. 제어 루프가 카메라 프레임 주기에 묶이지 않는다.

  실측: 동기 read 2대 36.9ms -> 스레드 grab 2대 0.0ms

부수 효과 하나가 더 중요하다. 드라이버 버퍼에 쌓인 오래된 프레임을 스레드가 계속
버려주므로 **영상 지연이 누적되지 않는다.** 제어에 쓰는 그림이 항상 지금 것이다.
"""

import glob
import sys
import threading
import time

import cv2

# 플랫폼별 기본 카메라 백엔드.
# CAP_DSHOW 는 Windows 전용이라 맥에서 그대로 쓰면 카메라가 안 열린다.
# 상수 자체는 어느 플랫폼에서나 존재해서 getattr 로는 걸러지지 않는다.
PLATFORM_CAMERA_BACKEND = {
    "darwin": "CAP_AVFOUNDATION",
    "win32": "CAP_DSHOW",
}


def resolve_camera_backend(backend_name):
    """'auto' 또는 현재 플랫폼에서 못 쓰는 값이면 플랫폼 기본값으로 바꾼다."""
    default = PLATFORM_CAMERA_BACKEND.get(sys.platform, "CAP_V4L2")
    name = str(backend_name or "auto")
    if name.lower() == "auto":
        return default
    for platform_key, platform_backend in PLATFORM_CAMERA_BACKEND.items():
        if name == platform_backend and sys.platform != platform_key:
            print(f"[camera] {name} is {platform_key}-only; using {default}")
            return default
    return name


def open_capture(index, width, height, backend_name="auto", fps=30,
                 fourcc="MJPG", buffer_size=1, quiet=False):
    """카메라 하나를 열고 설정까지 적용한다. 실패하면 CAP_ANY 로 한 번 더."""
    backend_name = resolve_camera_backend(backend_name)
    camera = cv2.VideoCapture(int(index), getattr(cv2, str(backend_name), cv2.CAP_ANY))
    if not camera.isOpened():
        if not quiet:
            print(f"[camera] index={index} failed with {backend_name}; retrying CAP_ANY")
        camera.release()
        backend_name = "CAP_ANY"
        camera = cv2.VideoCapture(int(index), cv2.CAP_ANY)
    codec = str(fourcc)[:4].ljust(4, " ")
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    camera.set(cv2.CAP_PROP_FPS, float(fps))
    camera.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(buffer_size)))
    if not quiet:
        print(
            f"[camera] index={index} backend={backend_name} "
            f"requested={int(width)}x{int(height)}@{float(fps):.1f} "
            f"actual={int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}@"
            f"{float(camera.get(cv2.CAP_PROP_FPS)):.1f} fourcc={codec.strip()}"
        )
    return camera, backend_name


def list_cameras(backend_name="auto", limit=8):
    """이 컴퓨터에서 열리는 카메라 인덱스를 훑는다.
    맥과 Windows 는 인덱스가 다르게 잡히므로 설정을 맞출 때 쓴다."""
    resolved = resolve_camera_backend(backend_name)
    backend = getattr(cv2, str(resolved), cv2.CAP_ANY)
    print(f"[camera] scanning 0..{limit - 1} with {resolved}")
    found = []
    for index in range(limit):
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened():
            ok, frame = camera.read()
            shape = f"{frame.shape[1]}x{frame.shape[0]}" if ok else "no frame"
            print(f"  index {index}: OPEN  {shape}")
            found.append(index)
        camera.release()
    if not found:
        print("  (열리는 카메라 없음)")
    return found


class FrameGrabber:
    """카메라 한 대를 전담해서 읽는 스레드. 최신 프레임 하나만 보관한다."""

    def __init__(self, camera, name="cam"):
        self.camera = camera
        self.name = name
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self._seq = 0
        self._fail_streak = 0
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name=f"grab-{name}", daemon=True
        )
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.camera.read()
            if not ok or frame is None:
                self._fail_streak += 1
                time.sleep(0.005)
                continue
            self._fail_streak = 0
            stamp = time.time()
            with self._lock:
                self._frame = frame
                self._stamp = stamp
                self._seq += 1

    def read(self):
        """(ok, frame, captured_at). 프레임은 복사하지 않는다 —
        읽는 쪽에서 덮어쓰지 말 것."""
        with self._lock:
            if self._frame is None:
                return False, None, 0.0
            return True, self._frame, self._stamp

    @property
    def sequence(self):
        with self._lock:
            return self._seq

    def healthy(self, max_fail=200):
        return self._fail_streak < max_fail

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.camera.release()


# ── 시리얼 포트 ─────────────────────────────────────────────
# 카메라와 같은 "플랫폼별 장치 경로" 문제라 여기에 같이 둔다.
PLATFORM_SERIAL_GLOBS = {
    "darwin": ["/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/cu.wchusbserial*"],
    "linux": ["/dev/ttyACM*", "/dev/ttyUSB*"],
}


def resolve_serial_port(port):
    """설정된 포트가 이 플랫폼에서 말이 안 되면 실제로 존재하는 포트로 바꾼다."""
    port = str(port or "")
    looks_windows = port.upper().startswith("COM")
    if sys.platform == "win32":
        return port or "COM3"
    if not looks_windows and (glob.glob(port) or port.startswith("/dev/")):
        return port
    for pattern in PLATFORM_SERIAL_GLOBS.get(sys.platform, []):
        hits = sorted(glob.glob(pattern))
        if hits:
            print(f"[serial] {port or '(unset)'} is not usable here; using {hits[0]}")
            return hits[0]
    print(f"[serial] no serial device found for this platform (configured: {port})")
    return port
