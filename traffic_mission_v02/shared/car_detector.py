"""Latest-frame full-camera detector for the obstacle car model."""

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


class AsyncCarDetector:
    """Run YOLO off the driving loop and discard superseded pending frames."""

    def __init__(self, model_path, cfg):
        if YOLO is None:
            raise ImportError("ultralytics is required for best_car.pt")
        self.cfg = cfg
        self.model = YOLO(str(model_path))
        self._condition = threading.Condition()
        self._pending = None
        self._latest = CarDetection()
        self._running = True
        self.fault = None
        self._thread = threading.Thread(
            target=self._loop, name="best-car-detector", daemon=True
        )
        self._thread.start()

    def submit(self, frame, sequence, captured_at):
        with self._condition:
            self._pending = (frame.copy(), int(sequence), float(captured_at))
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
                frame, sequence, captured_at = self._pending
                self._pending = None

            wait_s = next_allowed - time.monotonic()
            if wait_s > 0:
                time.sleep(min(wait_s, 0.2))
            try:
                result = self._detect(frame, sequence, captured_at)
                with self._condition:
                    self._latest = result
                self.fault = None
            except Exception as exc:
                self.fault = str(exc)
            next_allowed = time.monotonic() + float(
                self.cfg.get("car_inference_interval_s", 0.10)
            )

    def _detect(self, frame, sequence, captured_at):
        prediction = self.model.predict(
            source=frame,
            imgsz=int(self.cfg.get("car_image_size", 512)),
            conf=float(self.cfg.get("car_confidence", 0.45)),
            device=str(self.cfg.get("model_device", "cpu")),
            verbose=False,
        )[0]
        accepted = {
            str(name).lower()
            for name in self.cfg.get("car_class_names", ["obstacle-car"])
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
                    (int(x1), int(y1), int(x2), int(y2)),
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
        )

    def close(self):
        self._running = False
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
