"""CSV and rendered-UI recording for post-run diagnosis."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2


CSV_FIELDS = [
    "frame_index", "wall_time", "elapsed_s", "video_frame_index", "fps",
    "running", "vehicle_state", "event", "stop_condition", "lane_found",
    "traffic_mission_enabled", "signal_stop_latched", "avoidance_blocked_by_signal",
    "stop_line_detected", "lane_error", "solid_filter_angle_deg",
    "solid_filter_kept_pixels", "solid_filter_rejected_pixels", "raw_steer", "final_steer",
    "final_speed", "recovery_ramp_active", "recovery_ramp_elapsed_s",
    "recovery_ramp_pwm", "steering_pot", "traffic_signal", "traffic_result_age_s",
    "traffic_model_fault", "car_detected", "car_label",
    "car_confidence", "car_bbox", "car_area_ratio", "car_result_age_s",
    "car_roi_normalized", "car_roi_pixels", "car_model_fault",
    "ultrasonic_json", "ultrasonic_age_json",
    "front_sensor_values", "front_distance_cm", "sensor_ok",
    "ultrasonic_close", "fusion_confirmed", "fusion_hits", "fusion_skew_s",
    "fusion_reason", "obstacle_phase", "dash_side", "solid_side",
    "locked_dash_side", "lost_frames",
]


class RunLogger:
    def __init__(self, root_dir, cfg, run_context=None):
        self.enabled = bool(cfg.get("logging_enabled", 1))
        self.video_enabled = self.enabled and bool(cfg.get("log_video_enabled", 1))
        self.started_at = time.time()
        self.video_fps = max(1.0, float(cfg.get("log_video_fps", 15.0)))
        self.codec = str(cfg.get("log_video_codec", "MJPG"))[:4].ljust(4, " ")
        self.flush_interval = max(0.1, float(cfg.get("log_flush_interval_s", 1.0)))
        self.last_flush = self.started_at
        self.video_frame_index = -1
        self.csv_handle = None
        self.csv_writer = None
        self.drive_writer = None
        self.traffic_writer = None
        self.drive_size = None
        self.traffic_size = None
        self.drive_writer_attempted = False
        self.traffic_writer_attempted = False
        self.session_dir = None

        if not self.enabled:
            return
        log_dir = Path(str(cfg.get("log_dir", "logs")))
        if not log_dir.is_absolute():
            log_dir = Path(root_dir) / log_dir
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.session_dir = log_dir / stamp
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.csv_handle = (self.session_dir / "telemetry.csv").open(
            "w", newline="", encoding="utf-8-sig"
        )
        self.csv_writer = csv.DictWriter(
            self.csv_handle, fieldnames=CSV_FIELDS, extrasaction="ignore"
        )
        self.csv_writer.writeheader()
        metadata = {
            "created_at": datetime.now().astimezone().isoformat(),
            "config": cfg,
            "run_context": run_context or {},
            "files": {
                "csv": "telemetry.csv",
                "drive_video": "drive_ui.avi",
                "traffic_video": "traffic_roi.avi",
            },
        }
        with (self.session_dir / "session.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        print(f"[logger] session -> {self.session_dir}")

    def _open_writer(self, filename, frame):
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(self.session_dir / filename),
            cv2.VideoWriter_fourcc(*self.codec),
            self.video_fps,
            (width, height),
        )
        if not writer.isOpened():
            print(f"[logger] video disabled: failed to open {filename}")
            writer.release()
            return None
        return writer

    def write(self, row, drive_ui, traffic_ui):
        if not self.enabled:
            return
        now = time.time()
        if self.video_enabled:
            if not self.drive_writer_attempted:
                self.drive_writer_attempted = True
                self.drive_writer = self._open_writer("drive_ui.avi", drive_ui)
                self.drive_size = drive_ui.shape[1::-1]
            if not self.traffic_writer_attempted:
                self.traffic_writer_attempted = True
                self.traffic_writer = self._open_writer("traffic_roi.avi", traffic_ui)
                self.traffic_size = traffic_ui.shape[1::-1]
            if self.drive_writer is not None:
                drive_frame = (
                    drive_ui
                    if drive_ui.shape[1::-1] == self.drive_size
                    else cv2.resize(drive_ui, self.drive_size)
                )
                self.drive_writer.write(drive_frame)
            if self.traffic_writer is not None:
                traffic_frame = (
                    traffic_ui
                    if traffic_ui.shape[1::-1] == self.traffic_size
                    else cv2.resize(traffic_ui, self.traffic_size)
                )
                self.traffic_writer.write(traffic_frame)
            self.video_frame_index += 1

        values = {field: "" for field in CSV_FIELDS}
        values.update(row)
        values["wall_time"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        values["elapsed_s"] = f"{now - self.started_at:.3f}"
        values["video_frame_index"] = self.video_frame_index
        self.csv_writer.writerow(values)
        if now - self.last_flush >= self.flush_interval:
            self.csv_handle.flush()
            self.last_flush = now

    def close(self):
        for writer in (self.drive_writer, self.traffic_writer):
            if writer is not None:
                writer.release()
        if self.csv_handle is not None:
            self.csv_handle.flush()
            self.csv_handle.close()
        if self.session_dir is not None:
            print(f"[logger] saved -> {self.session_dir}")
