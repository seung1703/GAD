"""Integrated runner for traffic_mission_v02_0804.

This loop merges:
- lane following from the GAD drive flow
- traffic-light stop/go from traffic_mission
- obstacle avoidance using ultrasonic telemetry
"""

import argparse
import json
import sys
import time
import tkinter as tk
from tkinter import ttk
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
LANE_MODEL_PATH = ROOT_DIR / "model" / "lane_best.pt"
CAR_MODEL_PATH = ROOT_DIR / "model" / "car_best.pt"

sys.path.insert(0, str(SHARED_DIR))

from car_detector import AsyncCarDetector
from config import load, save
from control import LaneFollowController, RecoverySpeedRamp
from lane_model import ModelLaneDetector
from obstacle_controller import ObstacleController
from run_logger import RunLogger
from serial_driver import MegaLink
from traffic_detector import AsyncTrafficDetector
from undistort import Undistorter


DISPLAY_WINDOW_NAME = "Integrated Drive"
TRAFFIC_ROI_WINDOW_NAME = "Traffic ROI"
DRIVE_CONTROL_WINDOW_NAME = "Lane ROI / Drive Controls"
STATE_DRIVING = "DRIVING"
STATE_AVOIDING_OBSTACLE = "AVOIDING_OBSTACLE"
STATE_STOPPED_RED = "STOPPED_RED"
STATE_SAFE_STOP = "SAFE_STOP"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic mission runner with lane follow + traffic-light stop/go logic."
    )
    parser.add_argument("--dry", action="store_true", help="Run vision only without serial commands.")
    parser.add_argument("--cam", type=int, help="Override lane camera index from calib.json.")
    parser.add_argument("--traffic-cam", type=int, help="Use a separate camera index for traffic detection.")
    parser.add_argument("--serial-port", help="Override serial port. Example: COM3")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --dry.")
    parser.add_argument("--mock-obstacle", action="store_true", help="Simulate periodic obstacle sensor input in dry mode.")
    return parser.parse_args()


def open_camera(
    index,
    width,
    height,
    backend_name="CAP_DSHOW",
    fps=30,
    fourcc="MJPG",
    buffer_size=1,
):
    backend = getattr(cv2, str(backend_name), cv2.CAP_DSHOW)
    camera = cv2.VideoCapture(int(index), backend)
    codec = str(fourcc)[:4].ljust(4, " ")
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    camera.set(cv2.CAP_PROP_FPS, float(fps))
    camera.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(buffer_size)))
    actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(camera.get(cv2.CAP_PROP_FPS))
    print(
        f"[camera] index={index} backend={backend_name} "
        f"requested={int(width)}x{int(height)}@{float(fps):.1f} "
        f"actual={actual_w}x{actual_h}@{actual_fps:.1f} fourcc={codec.strip()}"
    )
    return camera


def get_camera_size(camera, fallback_w, fallback_h):
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(fallback_w)
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(fallback_h)
    return width, height


def normalize_traffic_roi(cfg):
    left = min(max(float(cfg.get("traffic_roi_left", 0.20)), 0.0), 0.99)
    top = min(max(float(cfg.get("traffic_roi_top", 0.05)), 0.0), 0.99)
    right = min(max(float(cfg.get("traffic_roi_right", 0.80)), 0.01), 1.0)
    bottom = min(max(float(cfg.get("traffic_roi_bottom", 0.55)), 0.01), 1.0)

    # Keep at least one trackbar step between opposite edges.
    if right <= left:
        if left < 0.99:
            right = left + 0.01
        else:
            left, right = 0.98, 0.99
    if bottom <= top:
        if top < 0.99:
            bottom = top + 0.01
        else:
            top, bottom = 0.98, 0.99

    cfg["traffic_roi_left"] = left
    cfg["traffic_roi_top"] = top
    cfg["traffic_roi_right"] = right
    cfg["traffic_roi_bottom"] = bottom
    return left, top, right, bottom


def get_traffic_roi_pixels(frame, cfg):
    height, width = frame.shape[:2]
    left, top, right, bottom = normalize_traffic_roi(cfg)
    x1 = min(max(int(round(left * width)), 0), max(width - 1, 0))
    y1 = min(max(int(round(top * height)), 0), max(height - 1, 0))
    x2 = min(max(int(round(right * width)), x1 + 1), width)
    y2 = min(max(int(round(bottom * height)), y1 + 1), height)
    return x1, y1, x2, y2


def normalize_car_roi(cfg):
    left = min(max(float(cfg.get("car_roi_left", 0.0)), 0.0), 0.99)
    top = min(max(float(cfg.get("car_roi_top", 0.0)), 0.0), 0.99)
    right = min(max(float(cfg.get("car_roi_right", 1.0)), 0.01), 1.0)
    bottom = min(max(float(cfg.get("car_roi_bottom", 1.0)), 0.01), 1.0)

    if right <= left:
        if left < 0.99:
            right = left + 0.01
        else:
            left, right = 0.98, 0.99
    if bottom <= top:
        if top < 0.99:
            bottom = top + 0.01
        else:
            top, bottom = 0.98, 0.99

    cfg["car_roi_left"] = left
    cfg["car_roi_top"] = top
    cfg["car_roi_right"] = right
    cfg["car_roi_bottom"] = bottom
    return left, top, right, bottom


def get_car_roi_pixels(frame, cfg):
    height, width = frame.shape[:2]
    left, top, right, bottom = normalize_car_roi(cfg)
    x1 = min(max(int(round(left * width)), 0), max(width - 1, 0))
    y1 = min(max(int(round(top * height)), 0), max(height - 1, 0))
    x2 = min(max(int(round(right * width)), x1 + 1), width)
    y2 = min(max(int(round(bottom * height)), y1 + 1), height)
    return x1, y1, x2, y2


def draw_traffic_detection(frame, detection, cfg, now, valid_after=0.0):
    debug_frame = frame.copy()
    x1, y1, x2, y2 = get_traffic_roi_pixels(frame, cfg)
    age_s = None
    if detection.source_sequence >= 0:
        age_s = max(0.0, now - float(detection.frame_time))
    fresh = (
        age_s is not None
        and float(detection.frame_time) >= float(valid_after)
        and age_s <= float(cfg.get("traffic_result_stale_s", 0.50))
    )
    signal = detection.signal if fresh else "unknown"

    if fresh:
        for class_name, confidence, bbox in detection.boxes:
            box_x1, box_y1, box_x2, box_y2 = bbox
            colors = {
                "red": (0, 180, 0),
                "green": (0, 0, 220),
                "yellow": (0, 220, 220),
            }
            color = colors.get(class_name, (0, 180, 255))
            cv2.rectangle(
                debug_frame, (box_x1, box_y1), (box_x2, box_y2), color, 2
            )
            cv2.putText(
                debug_frame,
                f"{class_name} {'STOP' if class_name == 'green' else 'GO' if class_name == 'red' else ''} {confidence:.2f}",
                (box_x1, max(20, box_y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2,
            )

    cv2.rectangle(debug_frame, (x1, y1), (x2 - 1, y2 - 1), (255, 180, 0), 3)
    cv2.putText(
        debug_frame,
        (
            f"TRAFFIC ROI ({x1},{y1})-({x2},{y2}) detect={signal} "
            f"age={'--' if age_s is None else f'{age_s:.2f}s'} async"
        ),
        (max(8, x1), max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 180, 0),
        2,
    )
    return signal, debug_frame, fresh, age_s


def create_traffic_disabled_debug(frame, cfg):
    debug_frame = frame.copy()
    x1, y1, x2, y2 = get_traffic_roi_pixels(frame, cfg)
    cv2.rectangle(debug_frame, (x1, y1), (x2 - 1, y2 - 1), (140, 140, 140), 2)
    cv2.putText(
        debug_frame,
        "TRAFFIC MISSION COMPLETE - DETECTOR OFF",
        (max(8, x1), max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (140, 140, 140),
        2,
    )
    return debug_frame


def draw_car_detection(
    frame,
    detection,
    cfg,
    now,
    valid_after=0.0,
    expected_roi=None,
):
    debug = frame.copy()
    roi_x1, roi_y1, roi_x2, roi_y2 = get_car_roi_pixels(frame, cfg)
    cv2.rectangle(
        debug,
        (roi_x1, roi_y1),
        (roi_x2 - 1, roi_y2 - 1),
        (255, 0, 255),
        2,
    )
    cv2.putText(
        debug,
        f"CAR ROI ({roi_x1},{roi_y1})-({roi_x2},{roi_y2})",
        (max(8, roi_x1), min(debug.shape[0] - 8, roi_y1 + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 0, 255),
        2,
    )
    age_s = None
    if detection.source_sequence >= 0:
        age_s = max(0.0, now - float(detection.frame_time))
    fresh = (
        age_s is not None
        and float(detection.frame_time) >= float(valid_after)
        and (
            expected_roi is None
            or tuple(detection.roi or ()) == tuple(expected_roi)
        )
        and age_s <= float(cfg.get("car_result_stale_s", 0.8))
    )
    if detection.detected and fresh and detection.bbox is not None:
        bx1, by1, bx2, by2 = (int(value) for value in detection.bbox)
        cv2.rectangle(debug, (bx1, by1), (bx2, by2), (0, 140, 255), 3)
        cv2.putText(
            debug,
            f"CAR {detection.confidence:.2f} area={detection.area_ratio:.3f}",
            (bx1, max(22, by1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 140, 255),
            2,
        )
    age_text = "--" if age_s is None else f"{age_s:.2f}s"
    cv2.putText(
        debug,
        f"CAR ROI result_age={age_text} fresh={fresh}",
        (8, max(24, debug.shape[0] - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 110, 0),
        2,
    )
    return debug, fresh, age_s


def draw_car_disabled(frame, cfg):
    debug = frame.copy()
    roi_x1, roi_y1, roi_x2, roi_y2 = get_car_roi_pixels(frame, cfg)
    cv2.rectangle(
        debug,
        (roi_x1, roi_y1),
        (roi_x2 - 1, roi_y2 - 1),
        (140, 140, 140),
        2,
    )
    cv2.putText(
        debug,
        "CAR DETECTOR OFF WITH TRAFFIC MISSION",
        (8, max(24, debug.shape[0] - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (140, 140, 140),
        2,
    )
    return debug


def _noop(_value):
    return None


class DetectionStabilizer:
    def __init__(self, cfg):
        self.lane_history = deque(maxlen=int(cfg.get("lane_confirm_count", 2)))
        self.stop_history = deque(maxlen=int(cfg.get("stop_signal_confirm_count", 2)))
        self.go_history = deque(maxlen=int(cfg.get("go_signal_confirm_count", 2)))

    def update(self, lane_detected, traffic_signal):
        self.lane_history.append(bool(lane_detected))
        # This checkpoint was trained with physical red/green labels swapped.
        self.stop_history.append(traffic_signal == "green")
        self.go_history.append(traffic_signal == "red")

    def reset(self):
        self.lane_history.clear()
        self.stop_history.clear()
        self.go_history.clear()

    def invalidate_traffic(self):
        self.reset()

    def lane_confirmed(self):
        return len(self.lane_history) == self.lane_history.maxlen and all(self.lane_history)

    def stop_signal_confirmed(self):
        return len(self.stop_history) == self.stop_history.maxlen and all(self.stop_history)

    def go_signal_confirmed(self):
        return len(self.go_history) == self.go_history.maxlen and all(self.go_history)


class TrafficMissionGate:
    ACTIVE = "ACTIVE"
    HOLD_RED = "HOLD_RED"
    COMPLETE = "COMPLETE"

    def __init__(self):
        self.state = self.ACTIVE

    @property
    def detector_enabled(self):
        return self.state != self.COMPLETE

    @property
    def holding(self):
        return self.state == self.HOLD_RED

    def reset(self):
        self.state = self.ACTIVE

    def update(self, enabled, lane_confirmed, stop_confirmed, go_confirmed):
        stop_started = False
        resumed = False
        if enabled and self.state == self.ACTIVE and lane_confirmed and stop_confirmed:
            self.state = self.HOLD_RED
            stop_started = True
        elif enabled and self.state == self.HOLD_RED and go_confirmed:
            self.state = self.COMPLETE
            resumed = True
        return stop_started, resumed


class ScrollableDriveControls:
    """Separate scrollable control window; values match the old trackbars."""

    SPECS = (
        ("LANE ROI (BEV)", None, 0.0, 0.0, 0.0),
        ("TL X", "ipm_tl_x", -1.0, 2.0, 0.01),
        ("TL Y", "ipm_tl_y", 0.0, 1.0, 0.01),
        ("TR X", "ipm_tr_x", -1.0, 2.0, 0.01),
        ("TR Y", "ipm_tr_y", 0.0, 1.0, 0.01),
        ("BL X", "ipm_bl_x", -1.0, 2.0, 0.01),
        ("BL Y", "ipm_bl_y", 0.0, 1.0, 0.01),
        ("BR X", "ipm_br_x", -1.0, 2.0, 0.01),
        ("BR Y", "ipm_br_y", 0.0, 1.0, 0.01),
        ("Center px", "center_offset", -100, 100, 1),
        ("Lane width px", "bev_lane_width", 1, 640, 1),
        ("Bands", "n_bands", 2, 12, 1),
        ("DRIVE CONTROL", None, 0.0, 0.0, 0.0),
        ("Kp", "kp", 0.1, 5.0, 0.1),
        ("Kd", "kd", 0.0, 30.0, 0.1),
        ("Right gain", "steer_right_gain", 0.1, 3.0, 0.1),
        ("Heading gain", "heading_gain", 0.0, 3.0, 0.1),
        ("Drive PWM", "drive_pwm", 1, 255, 1),
    )
    INTEGER_KEYS = {"center_offset", "bev_lane_width", "n_bands", "drive_pwm"}

    def __init__(self, cfg):
        self.alive = True
        self.save_requested = False
        self.variables = {}
        self.value_labels = {}
        self.steps = {}

        self.root = tk.Tk()
        self.root.title(DRIVE_CONTROL_WINDOW_NAME)
        self.root.geometry("440x560+20+80")
        self.root.minsize(360, 300)
        self.root.protocol("WM_DELETE_WINDOW", self.hide)

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"), foreground="#165b72")
        style.configure("Value.TLabel", font=("Consolas", 9), foreground="#1f2933")

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, highlightthickness=0, background="#f3f6f8")
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas, padding=(10, 6))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_inner)
        self.root.bind("<MouseWheel>", self._on_mousewheel)

        row = 0
        for label, key, minimum, maximum, step in self.SPECS:
            if key is None:
                ttk.Label(self.inner, text=label, style="Section.TLabel").grid(
                    row=row, column=0, columnspan=3, sticky="w", pady=(8, 3)
                )
                row += 1
                continue
            initial = float(cfg.get(key, minimum))
            variable = tk.DoubleVar(value=initial)
            value_label = ttk.Label(
                self.inner, text=self._format_value(key, initial), style="Value.TLabel", width=7
            )
            ttk.Label(self.inner, text=label, width=14).grid(row=row, column=0, sticky="w")
            ttk.Scale(
                self.inner,
                from_=minimum,
                to=maximum,
                variable=variable,
                orient="horizontal",
                command=lambda raw, item=key: self._show_value(item, float(raw)),
            ).grid(row=row, column=1, sticky="ew", padx=(4, 8), pady=3)
            value_label.grid(row=row, column=2, sticky="e")
            self.variables[key] = variable
            self.value_labels[key] = value_label
            self.steps[key] = step
            row += 1

        self.inner.columnconfigure(1, weight=1)
        footer = ttk.Frame(self.root, padding=(10, 6))
        footer.pack(fill="x")
        ttk.Button(footer, text="Save config (W)", command=self._request_save).pack(side="left")
        ttk.Label(footer, text="Mouse wheel: scroll", foreground="#5b6870").pack(side="right")
        self.root.update_idletasks()
        self.show()

    @staticmethod
    def _format_value(key, value):
        return f"{int(round(value))}" if key in ScrollableDriveControls.INTEGER_KEYS else f"{value:.2f}"

    def _show_value(self, key, raw_value):
        step = self.steps[key]
        value = round(raw_value / step) * step
        self.value_labels[key].configure(text=self._format_value(key, value))

    def _update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_inner(self, event):
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-int(event.delta / 120), "units")

    def _request_save(self):
        self.save_requested = True

    def consume_save_request(self):
        requested = self.save_requested
        self.save_requested = False
        return requested

    def show(self):
        if not self.alive:
            return
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(250, lambda: self.root.attributes("-topmost", False))
        except tk.TclError:
            self.alive = False

    def hide(self):
        if not self.alive:
            return
        try:
            self.root.withdraw()
        except tk.TclError:
            self.alive = False

    def poll(self, cfg, controller=None):
        if not self.alive:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.alive = False
            return
        if not self.alive:
            return
        for key, variable in self.variables.items():
            step = self.steps[key]
            value = round(float(variable.get()) / step) * step
            cfg[key] = int(round(value)) if key in self.INTEGER_KEYS else float(value)
        if controller is not None:
            controller.kp = float(cfg["kp"])
            controller.kd = float(cfg["kd"])
            controller.right_gain = float(cfg["steer_right_gain"])

    def close(self):
        if not self.alive:
            return
        self.alive = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def create_traffic_roi_trackbars(cfg):
    left, top, right, bottom = normalize_traffic_roi(cfg)
    car_left, car_top, car_right, car_bottom = normalize_car_roi(cfg)
    cv2.namedWindow(TRAFFIC_ROI_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TRAFFIC_ROI_WINDOW_NAME, 900, 760)
    cv2.createTrackbar("Left %", TRAFFIC_ROI_WINDOW_NAME, round(left * 100), 100, _noop)
    cv2.createTrackbar("Top %", TRAFFIC_ROI_WINDOW_NAME, round(top * 100), 100, _noop)
    cv2.createTrackbar("Right %", TRAFFIC_ROI_WINDOW_NAME, round(right * 100), 100, _noop)
    cv2.createTrackbar("Bottom %", TRAFFIC_ROI_WINDOW_NAME, round(bottom * 100), 100, _noop)
    cv2.createTrackbar("Car Left %", TRAFFIC_ROI_WINDOW_NAME, round(car_left * 100), 100, _noop)
    cv2.createTrackbar("Car Top %", TRAFFIC_ROI_WINDOW_NAME, round(car_top * 100), 100, _noop)
    cv2.createTrackbar("Car Right %", TRAFFIC_ROI_WINDOW_NAME, round(car_right * 100), 100, _noop)
    cv2.createTrackbar("Car Bottom %", TRAFFIC_ROI_WINDOW_NAME, round(car_bottom * 100), 100, _noop)


def sync_traffic_roi_cfg_from_trackbars(cfg):
    cfg["traffic_roi_left"] = cv2.getTrackbarPos("Left %", TRAFFIC_ROI_WINDOW_NAME) / 100.0
    cfg["traffic_roi_top"] = cv2.getTrackbarPos("Top %", TRAFFIC_ROI_WINDOW_NAME) / 100.0
    cfg["traffic_roi_right"] = cv2.getTrackbarPos("Right %", TRAFFIC_ROI_WINDOW_NAME) / 100.0
    cfg["traffic_roi_bottom"] = cv2.getTrackbarPos("Bottom %", TRAFFIC_ROI_WINDOW_NAME) / 100.0
    normalized = normalize_traffic_roi(cfg)

    for name, value in zip(("Left %", "Top %", "Right %", "Bottom %"), normalized):
        position = round(value * 100)
        if cv2.getTrackbarPos(name, TRAFFIC_ROI_WINDOW_NAME) != position:
            cv2.setTrackbarPos(name, TRAFFIC_ROI_WINDOW_NAME, position)

    cfg["car_roi_left"] = cv2.getTrackbarPos("Car Left %", TRAFFIC_ROI_WINDOW_NAME) / 100.0
    cfg["car_roi_top"] = cv2.getTrackbarPos("Car Top %", TRAFFIC_ROI_WINDOW_NAME) / 100.0
    cfg["car_roi_right"] = cv2.getTrackbarPos("Car Right %", TRAFFIC_ROI_WINDOW_NAME) / 100.0
    cfg["car_roi_bottom"] = cv2.getTrackbarPos("Car Bottom %", TRAFFIC_ROI_WINDOW_NAME) / 100.0
    car_normalized = normalize_car_roi(cfg)
    car_names = ("Car Left %", "Car Top %", "Car Right %", "Car Bottom %")
    for name, value in zip(car_names, car_normalized):
        position = round(value * 100)
        if cv2.getTrackbarPos(name, TRAFFIC_ROI_WINDOW_NAME) != position:
            cv2.setTrackbarPos(name, TRAFFIC_ROI_WINDOW_NAME, position)


def _put_fitted_text(image, text, origin, max_width, color, scale=0.48, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fitted_scale = scale
    while fitted_scale > 0.32:
        text_width = cv2.getTextSize(text, font, fitted_scale, thickness)[0][0]
        if text_width <= max_width:
            break
        fitted_scale -= 0.02
    cv2.putText(image, text, origin, font, fitted_scale, color, thickness, cv2.LINE_AA)


def build_status_panel(width, height, lane_detected, traffic_signal, vehicle_state, stop_condition, obstacle_info, fps, running, cfg):
    panel = np.full((height, width, 3), (242, 245, 247), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (58, 72, 82), 1)
    cv2.rectangle(panel, (0, 0), (width - 1, 58), (36, 50, 59), -1)
    cv2.putText(panel, "MISSION TELEMETRY", (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)

    red = (38, 55, 220)
    green = (34, 145, 55)
    amber = (16, 132, 218)
    normal = (40, 49, 55)
    muted = (104, 113, 119)
    traffic_color = red if traffic_signal == "green" else green if traffic_signal == "red" else normal
    vehicle_color = normal
    if vehicle_state in {STATE_STOPPED_RED, STATE_SAFE_STOP}:
        vehicle_color = red
    elif vehicle_state == STATE_AVOIDING_OBSTACLE:
        vehicle_color = amber
    elif vehicle_state == STATE_DRIVING:
        vehicle_color = green

    sections = [
        (
            "MISSION",
            [
                ("LANE", str(lane_detected), green if lane_detected else normal),
                ("TRAFFIC", str(traffic_signal), traffic_color),
                ("VEHICLE", str(vehicle_state), vehicle_color),
                ("DRIVE", "ENABLED" if running else "STOPPED", green if running else red),
                ("STOP", str(stop_condition), red if stop_condition else normal),
                ("COMMAND", f"PWM {obstacle_info['speed']}   STEER {obstacle_info['steer']}", normal),
                ("FPS", f"{fps:.1f}", green if fps >= 15.0 else amber),
            ],
        ),
        (
            "OBSTACLE FUSION",
            [
                ("TRIGGER", f"{obstacle_info['detected']}   {obstacle_info['distance']}   {obstacle_info['phase']}", amber if obstacle_info["detected"] else normal),
                ("FRONT US", f"{obstacle_info['front_sensors']}   filtered:{obstacle_info['filtered_distance']}", normal),
                ("ALL US", obstacle_info["all_sensors"], normal),
                ("CAR", f"{obstacle_info['car_detected']}   conf:{obstacle_info['car_confidence']:.2f}   age:{obstacle_info['car_age']}", amber if obstacle_info["car_detected"] else normal),
                ("PIPELINE", f"detector:{'ON' if obstacle_info['car_detector_enabled'] else 'OFF'}   model:{obstacle_info['car_pipeline_ok']}   sensor:{obstacle_info['sensor_ok']}", red if not obstacle_info["sensor_ok"] else normal),
                ("FUSION", f"{obstacle_info['fusion_confirmed']}   hits:{obstacle_info['fusion_hits']}   skew:{obstacle_info['fusion_skew']}", amber if obstacle_info["fusion_confirmed"] else normal),
                ("REASON", str(obstacle_info["fusion_reason"]), muted),
            ],
        ),
        (
            "LANE CONTROL",
            [
                ("BOUNDARY", f"L:{obstacle_info['bottom_left_class']}   R:{obstacle_info['bottom_right_class']}", normal),
                ("CLASS", f"solid:{obstacle_info['solid_side']}   dash:{obstacle_info['dash_side']}   lock:{obstacle_info['locked_dash_side']}", normal),
                ("SOLID FILTER", f"{obstacle_info['solid_filter_angle']}   keep:{obstacle_info['solid_filter_kept']}   reject:{obstacle_info['solid_filter_rejected']}", normal),
                ("ERROR", f"P:{obstacle_info['position_error']}   C:{obstacle_info['curve_term']}   H:{obstacle_info['heading_term']}", normal),
                ("LOOKAHEAD", f"{obstacle_info['lookahead_mode']}   real:{obstacle_info['both_real_bands']}   warm:{obstacle_info['warmup_remaining']}", normal),
                ("GEOMETRY", f"meas:{obstacle_info['lane_width_measured']} fixed:{int(cfg['bev_lane_width'])} slope:{obstacle_info['left_slope_factor']}/{obstacle_info['right_slope_factor']}", normal),
                ("RECOVERY", f"{obstacle_info['recovery_ramp_active']}   PWM:{obstacle_info['recovery_ramp_pwm']}   {obstacle_info['recovery_ramp_elapsed']:.1f}/{obstacle_info['recovery_ramp_duration']:.1f}s", normal),
                ("SIGNAL LOCK", f"{obstacle_info['traffic_mission_state']}   avoid-block:{obstacle_info['avoidance_blocked_by_signal']}", red if obstacle_info["avoidance_blocked_by_signal"] else normal),
            ],
        ),
        (
            "LIVE TUNING",
            [
                ("GAINS", f"Kp:{cfg['kp']:.1f} Kd:{cfg['kd']:.1f} RG:{cfg['steer_right_gain']:.1f} HG:{cfg['heading_gain']:.1f}", normal),
                ("LANE", f"center:{int(cfg['center_offset']):+d}px width:{int(cfg['bev_lane_width'])} bands:{int(cfg['n_bands'])} PWM:{int(cfg['drive_pwm'])}", normal),
                ("IPM TOP", f"TL({cfg['ipm_tl_x']:+.2f},{cfg['ipm_tl_y']:.2f}) TR({cfg['ipm_tr_x']:+.2f},{cfg['ipm_tr_y']:.2f})", normal),
                ("IPM BOTTOM", f"BL({cfg['ipm_bl_x']:+.2f},{cfg['ipm_bl_y']:.2f}) BR({cfg['ipm_br_x']:+.2f},{cfg['ipm_br_y']:.2f})", normal),
                ("TRAFFIC ROI", f"L:{cfg['traffic_roi_left']:.2f} T:{cfg['traffic_roi_top']:.2f} R:{cfg['traffic_roi_right']:.2f} B:{cfg['traffic_roi_bottom']:.2f}", normal),
                ("CAR ROI", f"L:{cfg['car_roi_left']:.2f} T:{cfg['car_roi_top']:.2f} R:{cfg['car_roi_right']:.2f} B:{cfg['car_roi_bottom']:.2f}", normal),
                ("KEYS", "s:start  x/space:stop  r:reset  w:save  q:quit", muted),
            ],
        ),
    ]

    total_rows = sum(len(rows) for _title, rows in sections)
    card_header_height = 25
    outer_gap = 8
    available = height - 66 - outer_gap * len(sections) - card_header_height * len(sections) - 8
    row_height = max(18, min(25, available // total_rows))
    y = 66
    label_width = 112
    for title, rows in sections:
        card_height = card_header_height + row_height * len(rows) + 7
        cv2.rectangle(panel, (8, y), (width - 9, y + card_height), (255, 255, 255), -1)
        cv2.rectangle(panel, (8, y), (width - 9, y + card_height), (213, 220, 224), 1)
        cv2.rectangle(panel, (8, y), (width - 9, y + card_header_height), (226, 234, 238), -1)
        cv2.putText(panel, title, (17, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (39, 100, 124), 1, cv2.LINE_AA)
        row_y = y + card_header_height
        for label, value, color in rows:
            baseline = row_y + row_height - 6
            _put_fitted_text(panel, label, (17, baseline), label_width - 12, muted, 0.40, 1)
            _put_fitted_text(panel, value, (17 + label_width, baseline), width - label_width - 35, color, 0.47, 1)
            row_y += row_height
        y += card_height + outer_gap
    return panel


def create_display(dbg_orig, dbg_bev, lane_detected, traffic_signal, vehicle_state, stop_condition, obstacle_info, fps, running, cfg):
    view_width = int(cfg.get("view_width", 640))
    view_height = int(cfg.get("view_height", 360))
    panel_width = int(cfg.get("panel_width", 440))
    orig_view = dbg_orig if dbg_orig.shape[1::-1] == (view_width, view_height) else cv2.resize(dbg_orig, (view_width, view_height))
    bev_view = dbg_bev if dbg_bev.shape[1::-1] == (view_width, view_height) else cv2.resize(dbg_bev, (view_width, view_height))
    stacked = cv2.vconcat([orig_view, bev_view])
    panel = build_status_panel(panel_width, stacked.shape[0], lane_detected, traffic_signal, vehicle_state, stop_condition, obstacle_info, fps, running, cfg)
    display = cv2.hconcat([stacked, panel])
    x2 = display.shape[1] - 12
    x1 = x2 - 104
    cv2.rectangle(display, (x1, 10), (x2, 44), (30, 30, 210), -1)
    cv2.putText(
        display,
        "RESET",
        (x1 + 17, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )
    controls_x2 = x1 - 10
    controls_x1 = controls_x2 - 132
    cv2.rectangle(display, (controls_x1, 10), (controls_x2, 44), (124, 92, 31), -1)
    cv2.putText(
        display,
        "CONTROLS",
        (controls_x1 + 12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def obstacle_snapshot(
    decision,
    lane_detector,
    obstacle_controller,
    final_speed,
    final_steer,
    us,
    car_result,
    car_pipeline_ok,
):
    ignoring_distance = obstacle_controller.avoidance_in_progress()
    distance = "IGNORED" if ignoring_distance else "--" if decision.distance_cm is None else f"{decision.distance_cm}cm"
    front_ids = [
        int(sensor_id)
        for sensor_id in obstacle_controller.cfg.get("us_front_ids", [0, 1])
    ]
    if ignoring_distance:
        front_sensors = " ".join(f"U{sensor_id}:IGNORED" for sensor_id in front_ids)
        filtered_distance = "IGNORED"
    else:
        sensor_values = getattr(obstacle_controller, "front_sensor_values", {})
        front_sensors = " ".join(
            f"U{sensor_id}:{sensor_values[sensor_id]}cm"
            if sensor_id in sensor_values
            else f"U{sensor_id}:--"
            for sensor_id in front_ids
        )
        filtered_distance = distance
    def _signed_metric(name):
        value = getattr(lane_detector, name, None)
        return "--" if value is None else f"{value:+.3f}"

    lookahead_mode = getattr(lane_detector, "lookahead_mode", "--")
    effective_lookahead = getattr(lane_detector, "last_effective_lookahead", None)
    if effective_lookahead is not None:
        lookahead_mode = f"{lookahead_mode}({effective_lookahead:.1f})"
    all_sensors = " ".join(
        f"U{sensor_id}:{distance}cm" for sensor_id, distance in sorted(us.items())
    ) or "--"
    car_age = getattr(obstacle_controller, "car_result_age_s", None)
    fusion_skew = getattr(obstacle_controller, "fusion_skew_s", None)
    solid_filter_angle = getattr(lane_detector, "solid_angle_ema", None)
    return {
        "detected": decision.obstacle_detected,
        "distance": distance,
        "front_sensors": front_sensors,
        "all_sensors": all_sensors,
        "filtered_distance": filtered_distance,
        "phase": decision.phase,
        "sensor_ok": decision.sensor_ok,
        "solid_side": getattr(lane_detector, "solid_side", "--"),
        "dash_side": getattr(lane_detector, "dash_side", "--"),
        "locked_dash_side": getattr(obstacle_controller, "locked_dash_side", "--"),
        "bottom_left_class": getattr(lane_detector, "bottom_left_class", "--"),
        "bottom_right_class": getattr(lane_detector, "bottom_right_class", "--"),
        "solid_filter_angle": (
            "--" if solid_filter_angle is None else f"{solid_filter_angle:+.1f}deg"
        ),
        "solid_filter_kept": int(
            getattr(lane_detector, "last_solid_kept_pixels", 0)
        ),
        "solid_filter_rejected": int(
            getattr(lane_detector, "last_solid_rejected_pixels", 0)
        ),
        "position_error": _signed_metric("last_position_error"),
        "curve_term": _signed_metric("last_curve_term"),
        "heading": _signed_metric("last_heading"),
        "heading_term": _signed_metric("last_heading_term"),
        "lookahead_mode": lookahead_mode,
        "both_real_bands": getattr(lane_detector, "last_both_real_bands", 0),
        "warmup_remaining": getattr(
            lane_detector, "control_warmup_remaining", 0
        ),
        "lane_width_measured": int(
            getattr(lane_detector, "last_lane_width_measured", 0)
        ),
        "left_slope_factor": f"{float(getattr(lane_detector, 'last_left_slope_factor', 1.0)):.2f}",
        "right_slope_factor": f"{float(getattr(lane_detector, 'last_right_slope_factor', 1.0)):.2f}",
        "car_detected": bool(car_result.detected),
        "car_confidence": float(car_result.confidence),
        "car_age": "--" if car_age is None else f"{car_age:.2f}s",
        "car_pipeline_ok": bool(car_pipeline_ok),
        "ultrasonic_close": bool(
            getattr(obstacle_controller, "ultrasonic_close", False)
        ),
        "fusion_confirmed": bool(
            getattr(obstacle_controller, "fusion_confirmed", False)
        ),
        "fusion_hits": int(getattr(obstacle_controller, "front_hits", 0)),
        "fusion_skew": "--" if fusion_skew is None else f"{fusion_skew:.3f}s",
        "fusion_reason": getattr(obstacle_controller, "fusion_reason", "--"),
        "speed": final_speed,
        "steer": f"{final_steer:+.2f}",
    }


def main():
    args = parse_args()
    if YOLO is None:
        raise ImportError("ultralytics is required. Install with: pip install ultralytics")

    if args.dry_run:
        args.dry = True
    cfg = load()

    if args.cam is not None:
        cfg["cam_index"] = args.cam
    if args.serial_port:
        cfg["serial_port"] = args.serial_port

    if str(cfg.get("model_device", "cpu")).lower() == "cpu":
        import torch

        torch.set_num_threads(max(1, int(cfg.get("model_cpu_threads", 6))))
        try:
            torch.set_num_interop_threads(
                max(1, int(cfg.get("model_cpu_interop_threads", 2)))
            )
        except RuntimeError:
            pass
        print(
            f"[performance] torch CPU threads={torch.get_num_threads()} "
            f"interop={torch.get_num_interop_threads()}"
        )

    if not TRAFFIC_MODEL_PATH.exists():
        raise FileNotFoundError(f"Traffic model not found: {TRAFFIC_MODEL_PATH}")
    if not LANE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Lane model not found: {LANE_MODEL_PATH}")
    if not CAR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Obstacle car model not found: {CAR_MODEL_PATH}")

    traffic_cam_index = args.traffic_cam if args.traffic_cam is not None else int(cfg.get("traffic_cam_index", 2))

    lane_camera = open_camera(
        cfg["cam_index"],
        cfg["frame_w"],
        cfg["frame_h"],
        cfg.get("camera_backend", "CAP_DSHOW"),
        cfg.get("camera_fps", 30),
        cfg.get("camera_fourcc", "MJPG"),
        cfg.get("camera_buffer_size", 1),
    )
    if not lane_camera.isOpened():
        raise RuntimeError(f"Failed to open lane camera index {cfg['cam_index']}")

    traffic_camera = lane_camera
    if int(traffic_cam_index) != int(cfg["cam_index"]):
        traffic_camera = open_camera(
            traffic_cam_index,
            cfg.get("traffic_frame_w", 1280),
            cfg.get("traffic_frame_h", 720),
            cfg.get("camera_backend", "CAP_DSHOW"),
            cfg.get("camera_fps", 30),
            cfg.get("camera_fourcc", "MJPG"),
            cfg.get("camera_buffer_size", 1),
        )
        if not traffic_camera.isOpened():
            raise RuntimeError(f"Failed to open traffic camera index {traffic_cam_index}")

    print(f"[traffic] loading asynchronous model: {TRAFFIC_MODEL_PATH}")
    traffic_detector = AsyncTrafficDetector(TRAFFIC_MODEL_PATH, cfg)
    print(f"[obstacle] loading asynchronous car model: {CAR_MODEL_PATH}")
    car_detector = AsyncCarDetector(CAR_MODEL_PATH, cfg)
    actual_lane_w, actual_lane_h = get_camera_size(
        lane_camera, cfg["frame_w"], cfg["frame_h"]
    )
    lane_detector = ModelLaneDetector(cfg, model_path=str(LANE_MODEL_PATH))
    controller = LaneFollowController(
        kp=cfg["kp"],
        kd=cfg.get("kd", 6.0),
        steer_sign=cfg["steer_sign"],
        right_gain=cfg.get("steer_right_gain", 1.0),
    )
    recovery_ramp = RecoverySpeedRamp(cfg)
    link = MegaLink(cfg, dry_run=args.dry)
    stabilizer = DetectionStabilizer(cfg)
    traffic_gate = TrafficMissionGate()
    obstacle_controller = ObstacleController(cfg)
    und = Undistorter(
        cfg.get("camera_id", cfg["cam_index"]),
        actual_lane_w,
        actual_lane_h,
        str(INTRINSIC_DIR),
        enabled=bool(cfg.get("undistort", 1)),
    )

    display_width = int(cfg.get("view_width", actual_lane_w))
    display_height = int(cfg.get("view_height", actual_lane_h))
    panel_width = int(cfg.get("panel_width", 540))
    cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISPLAY_WINDOW_NAME, display_width + panel_width, display_height * 2)
    ui_actions = {"reset": False, "show_controls": False}
    reset_button_rect = (
        display_width + panel_width - 116,
        10,
        display_width + panel_width - 12,
        44,
    )
    controls_button_rect = (
        display_width + panel_width - 258,
        10,
        display_width + panel_width - 126,
        44,
    )

    def handle_display_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONUP:
            return
        x1, y1, x2, y2 = reset_button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            ui_actions["reset"] = True
            return
        x1, y1, x2, y2 = controls_button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            ui_actions["show_controls"] = True

    cv2.setMouseCallback(DISPLAY_WINDOW_NAME, handle_display_mouse)
    try:
        drive_controls = ScrollableDriveControls(cfg)
    except tk.TclError as exc:
        drive_controls = None
        print(f"[ui] Lane ROI / Drive Controls unavailable: {exc}")
    create_traffic_roi_trackbars(cfg)
    logger = RunLogger(
        ROOT_DIR,
        cfg,
        {
            "dry_run": bool(args.dry),
            "mock_obstacle": bool(args.mock_obstacle),
            "lane_camera": int(cfg["cam_index"]),
            "traffic_camera": int(traffic_cam_index),
            "car_camera": int(traffic_cam_index),
            "serial_port": str(cfg["serial_port"]),
            "lane_model": str(LANE_MODEL_PATH),
            "traffic_model": str(TRAFFIC_MODEL_PATH),
            "car_model": str(CAR_MODEL_PATH),
        },
    )
    running = False
    vehicle_state = STATE_DRIVING
    final_speed = 0
    lost_frames = 0
    last_steer = 0.0
    frame_count = 0
    started_at = time.time()
    runtime_reset_at = started_at
    fps_timestamps = deque(maxlen=31)
    last_log_state = None
    last_log_message = None
    last_recorded_state = None
    last_recorded_phase = None
    pending_runtime_event = ""
    last_traffic_result_sequence = -1
    video_stop_deadline = None

    def reset_runtime_state():
        nonlocal running, vehicle_state, final_speed, lost_frames, last_steer
        nonlocal runtime_reset_at
        nonlocal last_log_state, last_log_message, last_recorded_state
        nonlocal last_recorded_phase, pending_runtime_event
        nonlocal last_traffic_result_sequence
        nonlocal video_stop_deadline
        running = False
        vehicle_state = STATE_DRIVING
        final_speed = 0
        lost_frames = 0
        last_steer = 0.0
        runtime_reset_at = time.time()
        fps_timestamps.clear()
        last_log_state = None
        last_log_message = None
        last_recorded_state = None
        last_recorded_phase = None
        pending_runtime_event = "full-reset"
        last_traffic_result_sequence = -1
        video_stop_deadline = None
        stabilizer.reset()
        traffic_gate.reset()
        car_detector.set_enabled(True)
        obstacle_controller.reset()
        recovery_ramp.reset()
        lane_detector.reset_tracking(warmup_frames=0)
        controller.reset()
        link.reset_telemetry()
        link.send_brake()
        print("[reset] runtime state initialized; press s to start")

    print("[run] traffic_mission_v02_0804 ready")
    print("[run] stop once when BEV lane and model class green are confirmed together")
    print("[run] resume on model class red, then disable traffic and car detection until reset")
    print(
        "[run] all AVI recording stops "
        f"{float(cfg.get('video_stop_after_signal_resume_s', 3.0)):.1f}s after signal resume"
    )
    print("[run] avoidance requires car_best.pt AND synchronized front ultrasonic")
    print("[run] click RESET or press r for a full stopped-state initialization")
    print(
        f"[run] lane_cam={cfg['cam_index']} traffic_cam={traffic_cam_index} "
        f"serial={cfg['serial_port']} backend={cfg.get('camera_backend', 'CAP_DSHOW')}"
    )

    try:
        while True:
            ok_lane, lane_frame = lane_camera.read()
            if not ok_lane:
                print("[camera] failed to read lane frame")
                break
            lane_captured_at = time.time()

            ok_traffic, traffic_frame = traffic_camera.read()
            if not ok_traffic:
                traffic_frame = lane_frame
                traffic_captured_at = lane_captured_at
            else:
                traffic_captured_at = time.time()
            if drive_controls is not None:
                drive_controls.poll(cfg, controller)
                if ui_actions["show_controls"]:
                    ui_actions["show_controls"] = False
                    drive_controls.show()
                if drive_controls.consume_save_request():
                    save(cfg)
                    print("[config] saved -> shared/calib.json")
            sync_traffic_roi_cfg_from_trackbars(cfg)
            lane_detector.cfg = cfg
            obstacle_controller.cfg = cfg
            lane_frame = und(lane_frame)
            lane_clean_frame = lane_frame.copy()
            car_roi_pixels = get_car_roi_pixels(traffic_frame, cfg)
            if traffic_gate.detector_enabled:
                car_detector.submit(
                    traffic_frame,
                    frame_count,
                    traffic_captured_at,
                    car_roi_pixels,
                )

            if traffic_gate.detector_enabled:
                traffic_detector.submit(
                    traffic_frame,
                    frame_count,
                    traffic_captured_at,
                    get_traffic_roi_pixels(traffic_frame, cfg),
                )
                traffic_result = traffic_detector.latest()
                traffic_signal, traffic_debug, traffic_fresh, _traffic_age = (
                    draw_traffic_detection(
                        traffic_frame,
                        traffic_result,
                        cfg,
                        time.time(),
                        valid_after=runtime_reset_at,
                    )
                )
            else:
                traffic_signal = "disabled"
                traffic_debug = create_traffic_disabled_debug(traffic_frame, cfg)
                traffic_result = traffic_detector.latest()
                traffic_fresh = False
                _traffic_age = None
            tracking_enabled = obstacle_controller.lane_recognition_enabled()
            err, found, dbg_orig, dbg_bev = lane_detector.process(
                lane_frame,
                tracking_enabled=tracking_enabled,
            )
            lane_detected = bool(getattr(lane_detector, "last_crosswalk_detected", False))
            obstacle_controller.update_lane_context(
                getattr(lane_detector, "dash_side", None),
                getattr(lane_detector, "solid_side", None),
            )

            new_traffic_result = (
                traffic_gate.detector_enabled
                and traffic_fresh
                and traffic_result.source_sequence != last_traffic_result_sequence
            )
            if new_traffic_result:
                stabilizer.update(lane_detected, traffic_signal)
                last_traffic_result_sequence = traffic_result.source_sequence
            elif traffic_gate.detector_enabled and not traffic_fresh:
                stabilizer.invalidate_traffic()
            signal_stop_started, signal_resumed = traffic_gate.update(
                running,
                stabilizer.lane_confirmed(),
                stabilizer.stop_signal_confirmed(),
                stabilizer.go_signal_confirmed(),
            )
            if signal_stop_started:
                obstacle_controller.reset()
                controller.reset()
                lost_frames = 0
                last_steer = 0.0
                print("[traffic] model green+lane confirmed; avoidance blocked")
            elif signal_resumed:
                obstacle_controller.reset()
                car_detector.set_enabled(False)
                lane_detector.reset_control_state(
                    warmup_frames=int(cfg.get("control_warmup_frames", 3))
                )
                controller.reset()
                lost_frames = 0
                last_steer = 0.0
                stabilizer.reset()
                traffic_signal = "disabled"
                traffic_debug = create_traffic_disabled_debug(traffic_frame, cfg)
                traffic_fresh = False
                _traffic_age = None
                video_stop_deadline = time.time() + max(
                    0.0, float(cfg.get("video_stop_after_signal_resume_s", 3.0))
                )
                print("[traffic] model red confirmed; resumed and traffic/car detectors disabled")
            signal_holding = traffic_gate.holding
            stop_condition = signal_holding
            us = link.read_telemetry() or {}
            us_timestamps = link.ultrasonic_timestamps()
            now = time.time()
            car_detection_enabled = traffic_gate.detector_enabled
            car_result = car_detector.latest()
            car_pipeline_age = (
                None
                if car_result.processed_at <= 0
                else max(0.0, now - float(car_result.processed_at))
            )
            car_pipeline_ok = (
                not car_detection_enabled
                or (
                    car_detector.fault is None
                    and car_pipeline_age is not None
                    and car_pipeline_age <= float(cfg.get("car_pipeline_timeout_s", 2.0))
                )
            )
            within_car_startup_grace = (
                now - runtime_reset_at <= float(cfg.get("car_startup_grace_s", 5.0))
            )
            car_result_after_reset = (
                car_detection_enabled
                and car_result.source_sequence >= 0
                and float(car_result.frame_time) >= runtime_reset_at
                and tuple(car_result.roi or ()) == tuple(car_roi_pixels)
            )
            effective_car_detected = bool(car_result.detected and car_result_after_reset)
            effective_car_frame_time = (
                float(car_result.frame_time)
                if car_result_after_reset
                else None
            )
            effective_car_sequence = (
                int(car_result.source_sequence) if car_result_after_reset else -1
            )
            if car_detection_enabled and args.mock_obstacle and args.dry:
                cycle = int(now - runtime_reset_at) % 8
                us = {0: 35, 1: 38} if cycle in {2, 3, 4} else {0: 250, 1: 250}
                us_timestamps = {sensor_id: now for sensor_id in us}
                if cycle in {2, 3, 4}:
                    effective_car_detected = True
                    effective_car_frame_time = now
                    effective_car_sequence = frame_count
                    car_pipeline_ok = True

            maneuver_active = obstacle_controller.avoidance_in_progress()
            lane_motion_ready = (
                maneuver_active
                or err is not None
                or lost_frames <= cfg["lost_stop_frames"]
            )
            motion_enabled = running and lane_motion_ready and not signal_holding
            obstacle = obstacle_controller.decide(
                now,
                us,
                us_timestamps,
                motion_enabled=motion_enabled,
                car_detected=effective_car_detected,
                car_frame_time=effective_car_frame_time,
                car_sequence=effective_car_sequence,
            )

            if obstacle.avoidance_completed:
                recovery_ramp.start(now)
                lane_detector.reset_tracking()
                controller.reset()
                lost_frames = 0
                steer = 0.0
                last_steer = 0.0
            elif obstacle.avoidance_active:
                steer = 0.0
                lost_frames = 0
            elif signal_holding:
                steer = 0.0
                last_steer = 0.0
            elif signal_resumed:
                steer = 0.0
                last_steer = 0.0
            elif err is not None:
                steer = controller.compute(err)
                last_steer = steer
                lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames <= cfg["lost_hold_frames"]:
                    steer = last_steer
                else:
                    steer = last_steer * 0.5

            if signal_holding:
                vehicle_state = STATE_STOPPED_RED
            elif obstacle.avoidance_active:
                vehicle_state = STATE_AVOIDING_OBSTACLE
            elif not obstacle.sensor_ok and bool(cfg.get("sensor_required", 1)):
                vehicle_state = STATE_SAFE_STOP
            elif (
                car_detection_enabled
                and bool(cfg.get("car_model_required", 1))
                and not car_pipeline_ok
                and not within_car_startup_grace
            ):
                vehicle_state = STATE_SAFE_STOP
            else:
                vehicle_state = STATE_DRIVING

            final_steer = steer if obstacle.requested_steering is None else float(obstacle.requested_steering)
            ramp_motion_enabled = (
                running
                and vehicle_state == STATE_DRIVING
                and lost_frames <= cfg["lost_stop_frames"]
                and not obstacle.avoidance_active
                and obstacle.requested_speed is None
            )
            recovery_pwm = recovery_ramp.speed(
                now,
                int(cfg["drive_pwm"]),
                motion_enabled=ramp_motion_enabled,
            )

            if not running:
                final_speed = 0
                link.send_brake()
            elif vehicle_state in {STATE_STOPPED_RED, STATE_SAFE_STOP}:
                final_speed = 0
                link.send_brake()
            elif lost_frames > cfg["lost_stop_frames"]:
                final_speed = 0
                vehicle_state = STATE_SAFE_STOP
                link.send_brake()
            elif obstacle.requested_speed is not None and obstacle.requested_speed <= 0:
                final_speed = 0
                link.send_drive(0, final_steer)
            else:
                pwm = (
                    recovery_pwm
                    if obstacle.requested_speed is None
                    else min(recovery_pwm, int(obstacle.requested_speed))
                )
                if obstacle.requested_speed is None and abs(final_steer) > cfg["slow_steer_thresh"]:
                    pwm = min(pwm, int(cfg["slow_pwm"]))
                final_speed = pwm
                link.send_drive(pwm, final_steer)

            obstacle_info = obstacle_snapshot(
                obstacle,
                lane_detector,
                obstacle_controller,
                final_speed,
                final_steer,
                us,
                car_result,
                car_pipeline_ok or within_car_startup_grace,
            )
            obstacle_info["car_detected"] = effective_car_detected
            obstacle_info["car_detector_enabled"] = car_detection_enabled
            obstacle_info["car_confidence"] = (
                float(car_result.confidence) if car_result_after_reset else 0.0
            )
            obstacle_info["traffic_mission_state"] = traffic_gate.state
            obstacle_info["avoidance_blocked_by_signal"] = signal_holding
            obstacle_info["recovery_ramp_active"] = recovery_ramp.active
            obstacle_info["recovery_ramp_elapsed"] = recovery_ramp.elapsed_s
            obstacle_info["recovery_ramp_duration"] = recovery_ramp.duration_s
            obstacle_info["recovery_ramp_pwm"] = recovery_pwm
            log_message = (
                vehicle_state,
                traffic_signal,
                obstacle_info["phase"],
                obstacle_info["distance"],
                obstacle_info["sensor_ok"],
            )
            if log_message != last_log_message or vehicle_state != last_log_state:
                print(
                    f"[state] {vehicle_state} traffic={traffic_signal} obstacle={obstacle_info['phase']} "
                    f"dist={obstacle_info['distance']} sensor_ok={obstacle_info['sensor_ok']} "
                    f"cmd=({final_speed},{final_steer:+.2f})"
                )
                last_log_message = log_message
                last_log_state = vehicle_state

            frame_count += 1
            fps_timestamps.append(time.time())
            fps = (
                (len(fps_timestamps) - 1)
                / max(fps_timestamps[-1] - fps_timestamps[0], 1e-6)
                if len(fps_timestamps) >= 2
                else 0.0
            )
            if car_detection_enabled:
                traffic_debug, _car_fresh, car_result_age = draw_car_detection(
                    traffic_debug,
                    car_result,
                    cfg,
                    now,
                    valid_after=runtime_reset_at,
                    expected_roi=car_roi_pixels,
                )
            else:
                traffic_debug = draw_car_disabled(traffic_debug, cfg)
                _car_fresh = False
                car_result_age = None
            display = create_display(
                dbg_orig,
                dbg_bev,
                lane_detected,
                traffic_signal,
                vehicle_state,
                stop_condition,
                obstacle_info,
                fps,
                running,
                cfg,
            )
            cv2.imshow(DISPLAY_WINDOW_NAME, display)
            cv2.imshow(TRAFFIC_ROI_WINDOW_NAME, traffic_debug)

            video_recording_stopped = False
            if video_stop_deadline is not None and time.time() >= video_stop_deadline:
                video_recording_stopped = logger.stop_videos(
                    reason="signal resume delay elapsed"
                )
                video_stop_deadline = None

            event_parts = [pending_runtime_event] if pending_runtime_event else []
            pending_runtime_event = ""
            if signal_stop_started:
                event_parts.append("signal-stop-avoidance-blocked")
            if signal_resumed:
                event_parts.append("signal-resume-traffic-car-detectors-off")
            if video_recording_stopped:
                event_parts.append("signal-resume-video-recording-stopped")
            if obstacle.avoidance_completed:
                event_parts.append("avoidance-recovery-ramp-start")
            if vehicle_state != last_recorded_state:
                event_parts.append(f"state:{last_recorded_state}->{vehicle_state}")
                last_recorded_state = vehicle_state
            if obstacle.phase != last_recorded_phase:
                event_parts.append(f"phase:{last_recorded_phase}->{obstacle.phase}")
                last_recorded_phase = obstacle.phase
            sensor_ids = sorted(
                set(int(value) for value in cfg.get("log_ultrasonic_ids", []))
                | set(int(value) for value in us)
            )
            logged_us = {sensor_id: us.get(sensor_id) for sensor_id in sensor_ids}
            logged_us_age = {
                sensor_id: (
                    None
                    if sensor_id not in us_timestamps
                    else round(max(0.0, now - float(us_timestamps[sensor_id])), 3)
                )
                for sensor_id in sensor_ids
            }
            logger.write(
                {
                    "frame_index": frame_count,
                    "fps": f"{fps:.2f}",
                    "running": int(running),
                    "vehicle_state": vehicle_state,
                    "event": ";".join(event_parts),
                    "stop_condition": int(stop_condition),
                    "traffic_mission_enabled": int(traffic_gate.detector_enabled),
                    "car_detector_enabled": int(car_detection_enabled),
                    "signal_stop_latched": int(signal_holding),
                    "avoidance_blocked_by_signal": int(signal_holding),
                    "lane_found": int(bool(found)),
                    "stop_line_detected": int(lane_detected),
                    "lane_error": "" if err is None else f"{float(err):.6f}",
                    "solid_filter_angle_deg": (
                        ""
                        if lane_detector.solid_angle_ema is None
                        else f"{lane_detector.solid_angle_ema:.3f}"
                    ),
                    "solid_filter_kept_pixels": lane_detector.last_solid_kept_pixels,
                    "solid_filter_rejected_pixels": lane_detector.last_solid_rejected_pixels,
                    "lane_width_measured_px": lane_detector.last_lane_width_measured,
                    "left_slope_factor": f"{lane_detector.last_left_slope_factor:.6f}",
                    "right_slope_factor": f"{lane_detector.last_right_slope_factor:.6f}",
                    "raw_steer": f"{float(steer):.6f}",
                    "final_steer": f"{float(final_steer):.6f}",
                    "final_speed": int(final_speed),
                    "recovery_ramp_active": int(recovery_ramp.active),
                    "recovery_ramp_elapsed_s": f"{recovery_ramp.elapsed_s:.3f}",
                    "recovery_ramp_pwm": int(recovery_pwm),
                    "steering_pot": "" if link.pot is None else int(link.pot),
                    "traffic_signal": traffic_signal,
                    "traffic_result_age_s": "" if _traffic_age is None else f"{_traffic_age:.3f}",
                    "traffic_model_fault": traffic_detector.fault or "",
                    "car_detected": int(effective_car_detected),
                    "car_label": car_result.label,
                    "car_confidence": f"{float(car_result.confidence):.6f}",
                    "car_bbox": json.dumps(car_result.bbox if car_result_after_reset else None),
                    "car_area_ratio": f"{float(car_result.area_ratio if car_result_after_reset else 0.0):.6f}",
                    "car_result_age_s": "" if car_result_age is None or not car_result_after_reset else f"{car_result_age:.3f}",
                    "car_roi_normalized": json.dumps(
                        [
                            cfg["car_roi_left"],
                            cfg["car_roi_top"],
                            cfg["car_roi_right"],
                            cfg["car_roi_bottom"],
                        ]
                    ),
                    "car_roi_pixels": json.dumps(car_roi_pixels),
                    "car_model_fault": car_detector.fault or "",
                    "ultrasonic_json": json.dumps(logged_us, ensure_ascii=False),
                    "ultrasonic_age_json": json.dumps(logged_us_age, ensure_ascii=False),
                    "front_sensor_values": json.dumps(
                        obstacle_controller.front_sensor_values, ensure_ascii=False
                    ),
                    "front_distance_cm": "" if obstacle.distance_cm is None else obstacle.distance_cm,
                    "sensor_ok": int(obstacle.sensor_ok),
                    "ultrasonic_close": int(obstacle_info["ultrasonic_close"]),
                    "fusion_confirmed": int(obstacle_info["fusion_confirmed"]),
                    "fusion_hits": obstacle_info["fusion_hits"],
                    "fusion_skew_s": "" if obstacle_controller.fusion_skew_s is None else f"{obstacle_controller.fusion_skew_s:.3f}",
                    "fusion_reason": obstacle_info["fusion_reason"],
                    "obstacle_phase": obstacle.phase,
                    "dash_side": getattr(lane_detector, "dash_side", ""),
                    "solid_side": getattr(lane_detector, "solid_side", ""),
                    "locked_dash_side": obstacle_controller.locked_dash_side or "",
                    "lost_frames": lost_frames,
                },
                display,
                traffic_debug,
                traffic_clean=traffic_frame,
                lane_clean=lane_clean_frame,
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") or ui_actions["reset"]:
                ui_actions["reset"] = False
                reset_runtime_state()
            elif key == ord("s"):
                running = True
            elif key in (ord("x"), 32):
                running = False
                link.send_brake()
            elif key == ord("w"):
                save(cfg)
                print("[config] saved -> shared/calib.json")
    finally:
        if drive_controls is not None:
            drive_controls.close()
        link.send_brake()
        link.close()
        traffic_detector.close()
        car_detector.close()
        logger.close()
        lane_camera.release()
        if traffic_camera is not lane_camera:
            traffic_camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
