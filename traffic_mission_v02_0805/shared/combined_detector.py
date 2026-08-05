"""Single asynchronous full-frame detector for traffic lights and obstacle cars."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass(frozen=True)
class TrafficView:
    signal: str = "unknown"
    confidence: float = 0.0
    boxes: tuple = ()
    source_sequence: int = -1
    frame_time: float = 0.0
    processed_at: float = 0.0


@dataclass(frozen=True)
class CarView:
    detected: bool = False
    label: str = ""
    confidence: float = 0.0
    bbox: tuple | None = None
    area_ratio: float = 0.0
    source_sequence: int = -1
    frame_time: float = 0.0
    processed_at: float = 0.0


@dataclass(frozen=True)
class CombinedDetection:
    traffic: TrafficView = TrafficView()
    car: CarView = CarView()
    boxes: tuple = ()
    car_candidates: tuple = ()
    source_sequence: int = -1
    frame_time: float = 0.0
    processed_at: float = 0.0
    inference_ms: float = 0.0


def normalize_car_roi(cfg):
    """Return a valid normalized ROI without changing traffic-light coverage."""
    left = min(max(float(cfg.get("car_roi_left", 0.15)), 0.0), 0.99)
    top = min(max(float(cfg.get("car_roi_top", 0.15)), 0.0), 0.99)
    right = min(max(float(cfg.get("car_roi_right", 0.85)), 0.01), 1.0)
    bottom = min(max(float(cfg.get("car_roi_bottom", 1.0)), 0.01), 1.0)

    if right <= left:
        right = min(1.0, left + 0.01)
        if right <= left:
            left = max(0.0, right - 0.01)
    if bottom <= top:
        bottom = min(1.0, top + 0.01)
        if bottom <= top:
            top = max(0.0, bottom - 0.01)
    return left, top, right, bottom


def get_car_roi_pixels(frame, cfg):
    height, width = frame.shape[:2]
    left, top, right, bottom = normalize_car_roi(cfg)
    x1 = min(max(int(round(left * width)), 0), max(width - 1, 0))
    y1 = min(max(int(round(top * height)), 0), max(height - 1, 0))
    x2 = min(max(int(round(right * width)), x1 + 1), width)
    y2 = min(max(int(round(bottom * height)), y1 + 1), height)
    return x1, y1, x2, y2


def car_bbox_bottom_center_in_roi(bbox, frame, cfg):
    """Gate a car by its road-contact point, not by loose box overlap."""
    roi_x1, roi_y1, roi_x2, roi_y2 = get_car_roi_pixels(frame, cfg)
    frame_h, frame_w = frame.shape[:2]
    x1, _y1, x2, y2 = bbox
    contact_x = min(max(int(round((x1 + x2) / 2.0)), 0), max(frame_w - 1, 0))
    contact_y = min(max(int(y2) - 1, 0), max(frame_h - 1, 0))
    return (
        roi_x1 <= contact_x < roi_x2 and roi_y1 <= contact_y < roi_y2
    ), (contact_x, contact_y)


class AsyncCombinedDetector:
    """Infer the newest full camera frame once and split it into two views."""

    def __init__(self, model_path, cfg):
        if YOLO is None:
            raise ImportError("ultralytics is required for best_11n.pt")
        self.cfg = cfg
        self.model = YOLO(str(model_path))
        model_names = self.model.names
        available_names = {
            str(name).lower()
            for name in (
                model_names.values() if hasattr(model_names, "values") else model_names
            )
        }
        required_names = {"red", "green"} | {
            str(name).lower()
            for name in self.cfg.get("car_class_names", ["stroller"])
        }
        missing_names = required_names - available_names
        if missing_names:
            raise ValueError(
                "Combined model is missing required classes: "
                + ", ".join(sorted(missing_names))
            )
        self._condition = threading.Condition()
        self._pending = None
        self._latest = CombinedDetection()
        self._running = True
        self._enabled = True
        self.fault = None
        self._thread = threading.Thread(
            target=self._loop, name="combined-traffic-car-detector", daemon=True
        )
        self._thread.start()

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        with self._condition:
            self._enabled = enabled
            if not enabled:
                self._pending = None
                self._latest = CombinedDetection()
            self._condition.notify_all()

    @property
    def enabled(self):
        with self._condition:
            return self._enabled

    def submit(self, frame, sequence, captured_at):
        with self._condition:
            if not self._enabled:
                return
            self._pending = (frame.copy(), int(sequence), float(captured_at))
            self._condition.notify_all()

    def latest(self):
        with self._condition:
            return self._latest

    def _loop(self):
        next_allowed = 0.0
        while self._running:
            with self._condition:
                while self._running and (not self._enabled or self._pending is None):
                    self._condition.wait(timeout=0.2)
                if not self._running:
                    return

            wait_s = next_allowed - time.monotonic()
            if wait_s > 0:
                time.sleep(min(wait_s, 0.05))
                continue

            with self._condition:
                if not self._enabled or self._pending is None:
                    continue
                frame, sequence, captured_at = self._pending
                self._pending = None
            try:
                result = self._detect(frame, sequence, captured_at)
                with self._condition:
                    if self._enabled:
                        self._latest = result
                self.fault = None
            except Exception as exc:
                self.fault = str(exc)
            next_allowed = time.monotonic() + float(
                self.cfg.get("combined_inference_interval_s", 0.04)
            )

    def _detect(self, frame, sequence, captured_at):
        started = time.perf_counter()
        traffic_threshold = float(self.cfg.get("traffic_confidence", 0.25))
        car_threshold = float(self.cfg.get("car_confidence", 0.60))
        prediction = self.model.predict(
            source=frame,
            imgsz=int(self.cfg.get("combined_image_size", 640)),
            conf=min(traffic_threshold, car_threshold),
            device=str(self.cfg.get("model_device", "cpu")),
            verbose=False,
        )[0]
        processed_at = time.time()
        frame_area = max(1, frame.shape[0] * frame.shape[1])
        allowed_car_names = {
            str(name).lower() for name in self.cfg.get("car_class_names", ["stroller"])
        }
        min_car_area = float(self.cfg.get("car_min_area_ratio", 0.005))
        traffic_boxes = []
        combined_boxes = []
        car_candidates = []
        best_signal = "unknown"
        best_signal_confidence = 0.0
        best_car = None

        if prediction.boxes is not None:
            for box in prediction.boxes:
                class_id = int(box.cls[0].item())
                label = str(self.model.names[class_id]).lower()
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                bbox = (x1, y1, x2, y2)
                area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / frame_area

                if label in {"red", "green"} and confidence >= traffic_threshold:
                    traffic_boxes.append((label, confidence, bbox))
                    combined_boxes.append((label, confidence, bbox, area_ratio))
                    if confidence >= best_signal_confidence:
                        best_signal = label
                        best_signal_confidence = confidence
                elif (
                    label in allowed_car_names
                    and confidence >= car_threshold
                    and area_ratio >= min_car_area
                ):
                    roi_eligible, contact_point = car_bbox_bottom_center_in_roi(
                        bbox, frame, self.cfg
                    )
                    car_candidates.append(
                        (label, confidence, bbox, area_ratio, roi_eligible, contact_point)
                    )
                    if roi_eligible:
                        combined_boxes.append((label, confidence, bbox, area_ratio))
                        if best_car is None or confidence > best_car[0]:
                            best_car = (confidence, label, bbox, area_ratio)

        traffic = TrafficView(
            signal=best_signal,
            confidence=best_signal_confidence,
            boxes=tuple(traffic_boxes),
            source_sequence=sequence,
            frame_time=captured_at,
            processed_at=processed_at,
        )
        car = CarView(
            detected=best_car is not None,
            label="" if best_car is None else best_car[1],
            confidence=0.0 if best_car is None else best_car[0],
            bbox=None if best_car is None else best_car[2],
            area_ratio=0.0 if best_car is None else best_car[3],
            source_sequence=sequence,
            frame_time=captured_at,
            processed_at=processed_at,
        )
        return CombinedDetection(
            traffic=traffic,
            car=car,
            boxes=tuple(combined_boxes),
            car_candidates=tuple(car_candidates),
            source_sequence=sequence,
            frame_time=captured_at,
            processed_at=processed_at,
            inference_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self):
        self._running = False
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
