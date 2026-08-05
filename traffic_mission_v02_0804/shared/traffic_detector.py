"""Asynchronous latest-frame detector for the traffic-light ROI."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass(frozen=True)
class TrafficDetection:
    signal: str = "unknown"
    confidence: float = 0.0
    boxes: tuple = ()
    roi: tuple = (0, 0, 1, 1)
    source_sequence: int = -1
    frame_time: float = 0.0
    processed_at: float = 0.0


class AsyncTrafficDetector:
    """Run traffic YOLO off the control loop and discard superseded frames."""

    def __init__(self, model_path, cfg):
        if YOLO is None:
            raise ImportError("ultralytics is required for traffic-light detection")
        self.cfg = cfg
        self.model = YOLO(str(model_path))
        self._condition = threading.Condition()
        self._pending = None
        self._latest = TrafficDetection()
        self._running = True
        self.fault = None
        self._thread = threading.Thread(
            target=self._loop, name="traffic-light-detector", daemon=True
        )
        self._thread.start()

    def submit(self, frame, sequence, captured_at, roi):
        with self._condition:
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

    def _loop(self):
        next_allowed = 0.0
        while self._running:
            with self._condition:
                while self._running and self._pending is None:
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
                    self._latest = result
                self.fault = None
            except Exception as exc:
                self.fault = str(exc)
            next_allowed = time.monotonic() + float(
                self.cfg.get("traffic_inference_interval_s", 0.08)
            )

    def _detect(self, frame, sequence, captured_at, roi):
        x1, y1, x2, y2 = roi
        roi_frame = frame[y1:y2, x1:x2]
        prediction = self.model.predict(
            source=roi_frame,
            imgsz=int(self.cfg.get("traffic_image_size", 416)),
            conf=float(self.cfg.get("traffic_confidence", 0.25)),
            device=str(self.cfg.get("model_device", "cpu")),
            verbose=False,
        )[0]
        boxes = []
        best_signal = "unknown"
        best_confidence = 0.0
        if prediction.boxes is not None:
            for box in prediction.boxes:
                class_id = int(box.cls[0].item())
                label = str(self.model.names[class_id]).lower()
                confidence = float(box.conf[0].item())
                bx1, by1, bx2, by2 = (
                    int(value) for value in box.xyxy[0].tolist()
                )
                full_bbox = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
                boxes.append((label, confidence, full_bbox))
                if label in {"red", "green", "yellow"} and confidence >= best_confidence:
                    best_signal = label
                    best_confidence = confidence

        return TrafficDetection(
            signal=best_signal,
            confidence=best_confidence,
            boxes=tuple(boxes),
            roi=roi,
            source_sequence=sequence,
            frame_time=captured_at,
            processed_at=time.time(),
        )

    def close(self):
        self._running = False
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
