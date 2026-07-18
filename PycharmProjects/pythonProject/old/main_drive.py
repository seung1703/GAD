import cv2
import numpy as np
import torch
import json
import os
import time
import atexit
import threading
import queue
import Function_Library as fl

from ultralytics import YOLO

# ===== 설정 =====
CAM_PORT = 0
ARDUINO_PORT = 'COM7'
BAUD_RATE = 115200
IMG_W, IMG_H = 1280, 720
HEADING_OFFSET_X = 30
LANE_WIDTH_PX = 280          # 버드아이뷰에서 차선 폭(픽셀). 한쪽만 보일 때 추정에 사용
Kp_lat = 1.5                 # 직선 위치오차 게인 (작을수록 부드러움)
W_BALANCE = 1.0
SMOOTH = 0.7                 # lateral 부드럽게 (직전값 70% + 현재값 30%)
DRIVE_PWM = 35
STEER_NEUTRAL = 90
LOST_TIMEOUT = 0.5

# ===== YOLO 클래스 id (data.yaml 순서대로) =====
# names: 0=dash(점선), 1=lane(실제로는 횡단보도), 2=solid(실선)
DASH_ID = 0
CROSSWALK_ID = 1             # data.yaml엔 'lane'으로 돼있지만 실제 횡단보도
SOLID_ID = 2
LANE_IDS = (DASH_ID, SOLID_ID)   # 차선 경계 = 점선 + 실선

# ===== YOLOv8-seg 모델 로드 =====
try:
    model = YOLO("models/lane_seg_best.pt")
    model.to("cuda")         # GPU 메모리에 적재
    print("YOLOv8-seg 모델 로드 완료 (GPU 활성화)")
except Exception as e:
    print(f"모델 로드 실패! 경로를 확인하세요: {e}")

# ===== 저장된 ROI 파라미터 불러오기 =====
defaults = {
    'ROI_TOP_Y': 250, 'ROI_BOT_Y': 650,
    'ROI_TOP_L': 400, 'ROI_TOP_R': 900,
    'ROI_BOT_L': 150, 'ROI_BOT_R': 1150,
}
if os.path.exists('lane_params.json'):
    with open('lane_params.json', 'r') as f:
        saved = json.load(f)
    defaults.update(saved)   # white_L 등은 이제 안 쓰지만 ROI는 그대로 사용
    print("저장된 ROI 파라미터 불러옴!")


# ===== 카메라 스레드 =====
class CameraThread:
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        fail_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                fail_count += 1
                if fail_count > 30:  # 연속 실패 시 카메라 재오픈 시도
                    print("카메라 재연결 시도...")
                    self.cap.release()
                    time.sleep(0.5)
                    self.cap = cv2.VideoCapture(cv2.CAP_DSHOW + CAM_PORT)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    fail_count = 0
                time.sleep(0.01)
                continue
            fail_count = 0
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.005)   # 카메라 스레드에 작은 텀

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False


# ===== 시리얼 송신 스레드 =====
class SerialThread:
    def __init__(self, ser):
        self.ser = ser
        self.queue = queue.Queue(maxsize=3)
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                cmd = self.queue.get(timeout=0.1)
                self.ser.write((cmd + '\n').encode())
            except queue.Empty:
                continue
            except Exception as e:
                print(f"시리얼 에러: {e}")
                time.sleep(0.1)
                continue

    def send(self, cmd):
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except Exception:
                pass
        try:
            self.queue.put_nowait(cmd)
        except Exception:
            pass

    def send_urgent(self, cmd):
        try:
            self.ser.write((cmd + '\n').encode())
        except Exception:
            pass

    def stop(self):
        self.running = False


# ===== 하드웨어 초기화 =====
env = fl.libCAMERA()
ch0, _ = env.initial_setting(cam0port=CAM_PORT, capnum=1)
cap = ch0
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cam = CameraThread(cap)
print("카메라 스레드 시작!")

arduino = fl.libARDUINO()
ser = arduino.init(port=ARDUINO_PORT, baudrate=BAUD_RATE)
serial_thread = SerialThread(ser)
print("아두이노 연결 완료!")


# ===== 비상 정지 =====
def emergency_stop():
    try:
        serial_thread.send_urgent('X')
        serial_thread.send_urgent('S90')
        time.sleep(0.1)
        ser.close()
        print("비상 정지 완료!")
    except Exception:
        pass


atexit.register(emergency_stop)


def send(cmd):
    if cmd in ('X', 'S90'):
        serial_thread.send_urgent(cmd)
    else:
        serial_thread.send(cmd)


# 버드아이뷰 목적지 좌표 (워핑 후 펼쳐질 사각형)
dst = np.float32([
    [100, 0],
    [IMG_W - 100, 0],
    [IMG_W - 100, IMG_H],
    [100, IMG_H],
])


# ===== AI 비전 파이프라인 =====
def predict_and_warp(frame, M):
    combined = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    crosswalk_detected = False

    results = model.predict(frame, imgsz=640, device="cuda:0", verbose=False)
    r = results[0]
    if r.masks is None:
        del results, r
        return combined, crosswalk_detected

    fh, fw = frame.shape[:2]
    for mask, box in zip(r.masks.data, r.boxes):
        cls_id = int(box.cls[0].item())
        if cls_id == CROSSWALK_ID:
            crosswalk_detected = True
            continue
        if cls_id in LANE_IDS:
            m = mask.cpu().numpy()
            m = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_NEAREST)
            m = (m > 0.5).astype(np.uint8) * 255
            warped_m = cv2.warpPerspective(m, M, (IMG_W, IMG_H),
                                           flags=cv2.INTER_NEAREST)
            combined = cv2.bitwise_or(combined, warped_m)

    del results, r   # ← 추론 결과 메모리 해제
    return combined, crosswalk_detected


def calc_error_from_combined(combined):
    """버드아이뷰 합친 마스크 하단에서:
       - 신호1: 차선 중심 위치 오차 (직선 추종)
       - 신호2: 좌/우 픽셀 양 불균형 (커브 감지)
       두 신호를 함께 반환한다."""
    ref_x = IMG_W / 2.0 + HEADING_OFFSET_X
    half_lane = LANE_WIDTH_PX / 2.0

    # 하단 150px 밴드를 세로로 합쳐 안정적으로 샘플링 (점선 끊김 대응)
    band = combined[IMG_H - 150: IMG_H - 1, :]
    cols = np.where(band.sum(axis=0) > 0)[0]
    if len(cols) == 0:
        return None, None, None, 0.0

    # ref_x 기준으로 왼쪽/오른쪽 픽셀 그룹 분리
    left_cols = cols[cols < ref_x]
    right_cols = cols[cols >= ref_x]
    lx = np.mean(left_cols) if len(left_cols) > 0 else None
    rx = np.mean(right_cols) if len(right_cols) > 0 else None

    # --- 신호 1: 차선 중심 위치 오차 ---
    if lx is not None and rx is not None:
        center = (lx + rx) / 2.0
    elif lx is not None:
        center = lx + half_lane     # 오른쪽은 추정
    elif rx is not None:
        center = rx - half_lane     # 왼쪽은 추정
    else:
        return None, lx, rx, 0.0
    pos_error = (center - ref_x) / (IMG_W / 2.0)   # -1.0 ~ +1.0

    # --- 신호 2: 좌/우 픽셀 양 불균형 (양쪽 다 보일 때만 사용) ---
    nL = len(left_cols)
    nR = len(right_cols)
    if nL > 0 and nR > 0:
        balance = (nR - nL) / (nL + nR)  # 양쪽 다 보일 때만 정상 계산
    else:
        balance = 0.0  # 한쪽만 보이면 0 (위치오차만 사용, ±1 튐 방지)

    return pos_error, lx, rx, balance


def lateral_to_steer(lateral):
    steer = STEER_NEUTRAL + Kp_lat * lateral * 90
    return int(np.clip(steer, 30, 150))   # 급격한 조향 방지 위해 범위 제한


def draw_combined(combined):
    """버드아이뷰 차선 마스크를 반투명 초록으로 시각화"""
    vis = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    color = np.zeros_like(vis)
    color[combined > 0] = (0, 255, 0)
    return cv2.addWeighted(vis, 1.0, color, 0.5, 0)


# ===== 초기화 =====
print("=" * 40)
print("  Space: 출발/정지  |  q: 종료")
print("=" * 40)

send('S90')
time.sleep(0.5)

is_driving = False
last_lane_time = time.time()
frame_count = 0
prev_lateral = 0.0

try:
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            # 프레임 못 받은 지 0.5초 넘으면 안전 정지
            if is_driving and time.time() - last_lane_time > 1.0:
                send('X')
                send('S90')
                print("프레임 끊김 → 안전 정지")
            time.sleep(0.01)
            continue

        frame_count += 1

        if frame_count % 100 == 0:
            torch.cuda.empty_cache()

        # ROI 사다리꼴 → 버드아이뷰 변환 행렬
        src = np.float32([
            [defaults['ROI_TOP_L'], defaults['ROI_TOP_Y']],
            [defaults['ROI_TOP_R'], defaults['ROI_TOP_Y']],
            [defaults['ROI_BOT_R'], defaults['ROI_BOT_Y']],
            [defaults['ROI_BOT_L'], defaults['ROI_BOT_Y']],
        ])
        M = cv2.getPerspectiveTransform(src, dst)

        # --- 딥러닝 추론 (원본 frame 기준) → 마스크를 버드아이뷰로 워핑 ---
        t1 = time.time()
        combined_mask, crosswalk = predict_and_warp(frame, M)
        t2 = time.time()
        pos_error, lx, rx, balance = calc_error_from_combined(combined_mask)
        t3 = time.time()

        if t3 - t1 > 0.1:  # 10FPS 이하로 떨어지면 경고
            print(f"느린 처리: YOLO={t2 - t1:.3f}s  calc={t3 - t2:.3f}s")

        if pos_error is not None:
            last_lane_time = time.time()
            # 두 신호 결합: 위치오차(직선) + 커브(좌우 불균형)
            lateral = float(np.clip(pos_error + W_BALANCE * balance, -1.0, 1.0))
            # 직전 값과 섞어서 부드럽게 (튐 방지)
            lateral = SMOOTH * prev_lateral + (1 - SMOOTH) * lateral
            prev_lateral = lateral
            steer = lateral_to_steer(lateral)
            left_ok = lx is not None
            right_ok = rx is not None
        else:
            steer = STEER_NEUTRAL
            lateral = 0.0
            balance = 0.0
            left_ok, right_ok = False, False
            if is_driving and time.time() - last_lane_time > LOST_TIMEOUT:
                print("차선 소실 → 중립")

        if is_driving:
            send(f'S{steer}')

        if frame_count % 30 == 0:
            cw = 'O' if crosswalk else 'X'
            print(f"[{frame_count}] lateral={lateral:+.3f} balance={balance:+.2f} "
                  f"steer={steer} L={'O' if left_ok else 'X'} R={'O' if right_ok else 'X'} 횡단보도={cw}")

        # ===== 시각화 =====
        orig_vis = frame.copy()
        pts = src.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(orig_vis, [pts], True, (0, 255, 0), 2)

        status = 'DRIVING' if is_driving else 'STOPPED (Space:출발)'
        status_color = (0, 255, 0) if is_driving else (0, 0, 255)
        cv2.putText(orig_vis, f'lateral: {lateral:+.3f}',
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(orig_vis, f'balance: {balance:+.2f}',
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(orig_vis, f'steer: {steer}',
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(orig_vis, status,
                    (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        if crosswalk:
            cv2.putText(orig_vis, 'CROSSWALK',
                        (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2)

        lane_vis = draw_combined(combined_mask)
        ref_x = int(IMG_W // 2 + HEADING_OFFSET_X)
        cv2.line(lane_vis, (ref_x, 0), (ref_x, IMG_H), (0, 220, 220), 2)
        bottom_y = IMG_H - 150
        cv2.line(lane_vis, (0, bottom_y), (IMG_W, bottom_y), (255, 0, 0), 1)
        cv2.putText(lane_vis, f'lateral: {lateral:+.3f}',
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        h2, w2 = IMG_H // 2, IMG_W // 2
        debug = np.hstack([
            cv2.resize(orig_vis, (w2, h2)),
            cv2.resize(lane_vis, (w2, h2))
        ])
        cv2.imshow('Autonomous Drive - YOLOv8 Seg', debug)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if not is_driving:
                print("출발!")
                send(f'D{DRIVE_PWM}')
                is_driving = True
            else:
                print("정지!")
                send('X')
                send('S90')
                is_driving = False
        elif key == ord('q') or key == 27:
            print("종료 중...")
            break

except Exception as e:
    print(f"에러 발생: {e}")

finally:
    serial_thread.send_urgent('X')
    serial_thread.send_urgent('S90')
    time.sleep(0.3)
    serial_thread.stop()
    cam.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("종료 완료")