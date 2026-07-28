import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


ROOT_DIR = Path(__file__).resolve().parent
SHARED_DIR = ROOT_DIR / "shared"
INTRINSIC_DIR = ROOT_DIR / "camera_intrinsic"
TRAFFIC_MODEL_PATH = ROOT_DIR / "model" / "best_traffic_light_v2.pt"

sys.path.insert(0, str(SHARED_DIR))

from config import load, save
from control import LaneFollowController
from lane_model import ModelLaneDetector
from serial_driver import MegaLink
from undistort import Undistorter


TRAFFIC_CONFIDENCE = 0.25
TRAFFIC_IMAGE_SIZE = 512
VIEW_WIDTH = 640
VIEW_HEIGHT = 360
LANE_CONFIRM_COUNT = 2
STOP_SIGNAL_CONFIRM_COUNT = 2
GO_SIGNAL_CONFIRM_COUNT = 2
ROI_WINDOW_NAME = "Traffic ROI"
STATE_DRIVING = "DRIVING"
STATE_STOPPED_SIGNAL = "STOPPED_SIGNAL"
DEFAULT_TRAFFIC_CAM_INDEX = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic mission runner with lane follow + traffic-light stop/go logic."
    )
    parser.add_argument("--dry", action="store_true", help="Run vision only without serial commands.")
    parser.add_argument("--cam", type=int, help="Override lane camera index from calib.json.")
    parser.add_argument("--traffic-cam", type=int, help="Use a separate camera index for traffic detection.")
    parser.add_argument("--serial-port", help="Override serial port. Example: COM3")
    return parser.parse_args()


def _noop(_value):
    return None


def open_camera(index, width, height):
    camera = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    return camera


def detect_traffic_light(model, frame):
    signal = "unknown"
    best_stop_conf = 0.0
    best_red_conf = 0.0

    result = model.predict(
        source=frame,
        imgsz=TRAFFIC_IMAGE_SIZE,
        conf=TRAFFIC_CONFIDENCE,
        device="cpu",
        verbose=False,
    )[0]

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(model.names[class_id]).lower()
            confidence = float(box.conf[0].item())

            if class_name in {"green", "yellow"} and confidence >= best_stop_conf:
                signal = class_name
                best_stop_conf = confidence
            elif class_name == "red" and confidence >= best_red_conf:
                if best_stop_conf == 0.0:
                    signal = "red"
                best_red_conf = confidence

    return signal, result


class DetectionStabilizer:
    def __init__(self):
        self.lane_history = deque(maxlen=LANE_CONFIRM_COUNT)
        self.stop_history = deque(maxlen=STOP_SIGNAL_CONFIRM_COUNT)
        self.go_history = deque(maxlen=GO_SIGNAL_CONFIRM_COUNT)

    def update(self, lane_detected, traffic_signal):
        self.lane_history.append(bool(lane_detected))
        self.stop_history.append(traffic_signal in {"green", "yellow"})
        self.go_history.append(traffic_signal == "red")

    def lane_confirmed(self):
        return len(self.lane_history) == self.lane_history.maxlen and all(self.lane_history)

    def stop_signal_confirmed(self):
        return len(self.stop_history) == self.stop_history.maxlen and all(self.stop_history)

    def go_signal_confirmed(self):
        return len(self.go_history) == self.go_history.maxlen and all(self.go_history)


def create_roi_trackbars(cfg):
    cv2.namedWindow(ROI_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ROI_WINDOW_NAME, 520, 180)
    cv2.createTrackbar("roi_y_top", ROI_WINDOW_NAME, int(float(cfg["roi_y_top"]) * 100), 100, _noop)
    cv2.createTrackbar("roi_y_bot", ROI_WINDOW_NAME, int(float(cfg["roi_y_bot"]) * 100), 100, _noop)
    cv2.createTrackbar("roi_x_left", ROI_WINDOW_NAME, int(float(cfg["roi_x_left"]) * 100), 100, _noop)
    cv2.createTrackbar("roi_x_right", ROI_WINDOW_NAME, int(float(cfg["roi_x_right"]) * 100), 100, _noop)


def clamp_roi_cfg(cfg):
    roi_y_top = min(max(float(cfg.get("roi_y_top", 0.55)), 0.0), 0.97)
    roi_y_bot = min(max(float(cfg.get("roi_y_bot", 0.98)), roi_y_top + 0.01), 1.0)
    roi_x_left = min(max(float(cfg.get("roi_x_left", 0.0)), 0.0), 0.99)
    roi_x_right = min(max(float(cfg.get("roi_x_right", 1.0)), roi_x_left + 0.01), 1.0)
    cfg["roi_y_top"] = roi_y_top
    cfg["roi_y_bot"] = roi_y_bot
    cfg["roi_x_left"] = roi_x_left
    cfg["roi_x_right"] = roi_x_right


def sync_cfg_from_trackbars(cfg):
    cfg["roi_y_top"] = cv2.getTrackbarPos("roi_y_top", ROI_WINDOW_NAME) / 100.0
    cfg["roi_y_bot"] = cv2.getTrackbarPos("roi_y_bot", ROI_WINDOW_NAME) / 100.0
    cfg["roi_x_left"] = cv2.getTrackbarPos("roi_x_left", ROI_WINDOW_NAME) / 100.0
    cfg["roi_x_right"] = cv2.getTrackbarPos("roi_x_right", ROI_WINDOW_NAME) / 100.0
    clamp_roi_cfg(cfg)
    cv2.setTrackbarPos("roi_y_top", ROI_WINDOW_NAME, int(round(float(cfg["roi_y_top"]) * 100)))
    cv2.setTrackbarPos("roi_y_bot", ROI_WINDOW_NAME, int(round(float(cfg["roi_y_bot"]) * 100)))
    cv2.setTrackbarPos("roi_x_left", ROI_WINDOW_NAME, int(round(float(cfg["roi_x_left"]) * 100)))
    cv2.setTrackbarPos("roi_x_right", ROI_WINDOW_NAME, int(round(float(cfg["roi_x_right"]) * 100)))


def build_status_panel(width, lane_detected, traffic_signal, vehicle_state, stop_condition, fps, running):
    panel = np.full((150, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, 149), (40, 40, 40), 2)

    lines = [
        f"Lane(BEV): {lane_detected}",
        f"Traffic: {traffic_signal}",
        f"Vehicle: {vehicle_state}",
        f"Stop condition: {stop_condition}",
        f"Drive enabled: {running}",
        f"FPS: {fps:.1f}   s:start  x:stop  w:save ROI  q:quit",
    ]

    y = 26
    for idx, text in enumerate(lines):
        color = (30, 30, 30)
        if idx == 1 and traffic_signal in {"green", "yellow"}:
            color = (0, 0, 220)
        elif idx == 1 and traffic_signal == "red":
            color = (0, 150, 0)
        elif idx == 2 and vehicle_state == STATE_STOPPED_SIGNAL:
            color = (0, 0, 220)
        elif idx == 2 and vehicle_state == STATE_DRIVING:
            color = (0, 150, 0)
        elif idx == 3 and stop_condition:
            color = (0, 0, 220)
        cv2.putText(panel, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 2)
        y += 23

    return panel


def create_display(dbg_orig, dbg_bev, lane_detected, traffic_signal, vehicle_state, stop_condition, fps, running):
    orig_view = dbg_orig if dbg_orig.shape[1::-1] == (VIEW_WIDTH, VIEW_HEIGHT) else cv2.resize(dbg_orig, (VIEW_WIDTH, VIEW_HEIGHT))
    bev_view = dbg_bev if dbg_bev.shape[1::-1] == (VIEW_WIDTH, VIEW_HEIGHT) else cv2.resize(dbg_bev, (VIEW_WIDTH, VIEW_HEIGHT))
    stacked = cv2.vconcat([orig_view, bev_view])
    panel = build_status_panel(VIEW_WIDTH, lane_detected, traffic_signal, vehicle_state, stop_condition, fps, running)
    return cv2.vconcat([stacked, panel])


def main():
    if YOLO is None:
        raise ImportError("ultralytics is required. Install with: pip install ultralytics")

    args = parse_args()
    cfg = load()
    clamp_roi_cfg(cfg)

    if args.cam is not None:
        cfg["cam_index"] = args.cam
    if args.serial_port:
        cfg["serial_port"] = args.serial_port

    if not TRAFFIC_MODEL_PATH.exists():
        raise FileNotFoundError(f"Traffic model not found: {TRAFFIC_MODEL_PATH}")

    traffic_cam_index = args.traffic_cam if args.traffic_cam is not None else DEFAULT_TRAFFIC_CAM_INDEX

    lane_camera = open_camera(cfg["cam_index"], cfg["frame_w"], cfg["frame_h"])
    if not lane_camera.isOpened():
        raise RuntimeError(f"Failed to open lane camera index {cfg['cam_index']}")

    traffic_camera = lane_camera
    if int(traffic_cam_index) != int(cfg["cam_index"]):
        traffic_camera = open_camera(traffic_cam_index, 1280, 720)
        if not traffic_camera.isOpened():
            raise RuntimeError(f"Failed to open traffic camera index {traffic_cam_index}")

    print(f"[traffic] loading model: {TRAFFIC_MODEL_PATH}")
    traffic_model = YOLO(str(TRAFFIC_MODEL_PATH))
    lane_detector = ModelLaneDetector(cfg)
    controller = LaneFollowController(
        kp=cfg["kp"],
        kd=cfg.get("kd", 6.0),
        steer_sign=cfg["steer_sign"],
        right_gain=cfg.get("steer_right_gain", 1.0),
    )
    link = MegaLink(cfg, dry_run=args.dry)
    stabilizer = DetectionStabilizer()
    und = Undistorter(
        cfg.get("camera_id", cfg["cam_index"]),
        int(cfg["frame_w"]),
        int(cfg["frame_h"]),
        str(INTRINSIC_DIR),
        enabled=bool(cfg.get("undistort", 1)),
    )

    create_roi_trackbars(cfg)
    running = False
    vehicle_state = STATE_DRIVING
    lost_frames = 0
    last_steer = 0.0
    frame_count = 0
    started_at = time.time()

    print("[run] traffic mission final folder ready")
    print("[run] stop when BEV lane and green/yellow are confirmed together")
    print("[run] resume when red is confirmed")

    try:
        while True:
            ok_lane, lane_frame = lane_camera.read()
            if not ok_lane:
                print("[camera] failed to read lane frame")
                break

            ok_traffic, traffic_frame = traffic_camera.read()
            if not ok_traffic:
                traffic_frame = lane_frame

            sync_cfg_from_trackbars(cfg)
            lane_detector.cfg = cfg
            lane_frame = und(lane_frame)

            traffic_signal, _ = detect_traffic_light(traffic_model, traffic_frame)
            err, found, dbg_orig, dbg_bev = lane_detector.process(lane_frame)
            lane_detected = bool(getattr(lane_detector, "last_crosswalk_detected", False))

            stabilizer.update(lane_detected, traffic_signal)
            stop_condition = stabilizer.lane_confirmed() and stabilizer.stop_signal_confirmed()

            if vehicle_state == STATE_DRIVING and stop_condition:
                vehicle_state = STATE_STOPPED_SIGNAL
            elif vehicle_state == STATE_STOPPED_SIGNAL and stabilizer.go_signal_confirmed():
                vehicle_state = STATE_DRIVING

            if err is not None:
                steer = controller.compute(err)
                last_steer = steer
                lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames <= cfg["lost_hold_frames"]:
                    steer = last_steer
                else:
                    steer = last_steer * 0.5

            if not running:
                link.send_brake()
            elif vehicle_state == STATE_STOPPED_SIGNAL:
                link.send_brake()
            elif lost_frames > cfg["lost_stop_frames"]:
                link.send_brake()
            else:
                pwm = int(cfg["drive_pwm"])
                if abs(steer) > cfg["slow_steer_thresh"]:
                    pwm = int(cfg["slow_pwm"])
                link.send_drive(pwm, steer)

            frame_count += 1
            fps = frame_count / max(time.time() - started_at, 1e-6)
            display = create_display(
                dbg_orig,
                dbg_bev,
                lane_detected,
                traffic_signal,
                vehicle_state,
                stop_condition,
                fps,
                running,
            )
            cv2.imshow("Integrated Drive", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                running = True
            elif key in (ord("x"), 32):
                running = False
                link.send_brake()
            elif key == ord("w"):
                save(cfg)
                print("[config] saved ROI/settings")
    finally:
        link.send_brake()
        link.close()
        lane_camera.release()
        if traffic_camera is not lane_camera:
            traffic_camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
