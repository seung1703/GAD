"""Latest-frame ROI detector for the obstacle car model."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass(frozen=True)
class CarDetection:
    detected: bool = False
    label: str = "unknown"
    confidence: float = 0.0
    bbox: object = None
    area_ratio: float = 0.0
    source_sequence: int = -1
    frame_time: float = 0.0
    processed_at: float = 0.0
    roi: object = None


class AsyncCarDetector:
    """Run YOLO off the driving loop and discard superseded pending frames."""

    def __init__(self, model_path, cfg):
        if YOLO is None:
            raise ImportError("ultralytics is required for car_best.pt")
        self.cfg = cfg
        self.model = YOLO(str(model_path))
        self._condition = threading.Condition()
        self._pending = None
        self._latest = CarDetection()
        self._running = True
        self._enabled = True
        self.fault = None
        self._thread = threading.Thread(
            target=self._loop, name="best-car-detector", daemon=True
        )
        self._thread.start()

    def submit(self, frame, sequence, captured_at, roi=None):
        if roi is None:
            roi = (0, 0, frame.shape[1], frame.shape[0])
        with self._condition:
            if not self._enabled:
                return
            self._pending = (
                frame.copy(),
                int(sequence),
                float(captured_at),
                tuple(int(value) for value in roi),
            )
            self._condition.notify_all()

    def latest(self):
        with self._condition:
            return self._latest

    @property
    def enabled(self):
        with self._condition:
            return self._enabled

    def set_enabled(self, enabled):
        with self._condition:
            self._enabled = bool(enabled)
            if not self._enabled:
                self._pending = None
                self._latest = CarDetection()
            self._condition.notify_all()

    def _loop(self):
        next_allowed = 0.0
        while self._running:
            with self._condition:
                while self._running and (not self._enabled or self._pending is None):
                    self._condition.wait(timeout=0.2)
                if not self._running:
                    return
                frame, sequence, captured_at, roi = self._pending
                self._pending = None

            wait_s = next_allowed - time.monotonic()
            if wait_s > 0:
                time.sleep(min(wait_s, 0.2))
            try:
                result = self._detect(frame, sequence, captured_at, roi)
                with self._condition:
                    if self._enabled:
                        self._latest = result
                self.fault = None
            except Exception as exc:
                self.fault = str(exc)
            next_allowed = time.monotonic() + float(
                self.cfg.get("car_inference_interval_s", 0.10)
            )

    def _detect(self, frame, sequence, captured_at, roi):
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        prediction = self.model.predict(
            source=roi_frame,
            imgsz=int(self.cfg.get("car_image_size", 512)),
            conf=float(self.cfg.get("car_confidence", 0.45)),
            device=str(self.cfg.get("model_device", "cpu")),
            verbose=False,
        )[0]
        accepted = {
            str(name).lower()
            for name in self.cfg.get("car_class_names", ["stroller"])
        }
        frame_area = max(1, frame.shape[0] * frame.shape[1])
        best = None
        if prediction.boxes is not None:
            for box in prediction.boxes:
                class_id = int(box.cls[0].item())
                label = str(self.model.names[class_id]).lower()
                confidence = float(box.conf[0].item())
                if label not in accepted:
                    continue
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / frame_area)
                if area_ratio < float(self.cfg.get("car_min_area_ratio", 0.01)):
                    continue
                candidate = (
                    confidence,
                    label,
                    (
                        int(x1) + roi_x1,
                        int(y1) + roi_y1,
                        int(x2) + roi_x1,
                        int(y2) + roi_y1,
                    ),
                    area_ratio,
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate

        processed_at = time.time()
        if best is None:
            return CarDetection(
                source_sequence=sequence,
                frame_time=captured_at,
                processed_at=processed_at,
                roi=roi,
            )
        return CarDetection(
            True,
            best[1],
            best[0],
            best[2],
            best[3],
            sequence,
            captured_at,
            processed_at,
            roi,
        )

    def close(self):
        self._running = False
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
