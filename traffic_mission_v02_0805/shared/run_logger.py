"""CSV, rendered UI, and clean camera recording for post-run diagnosis."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2


CSV_FIELDS = [
    "frame_index", "wall_time", "elapsed_s", "video_frame_index",
    "video_recording_active", "fps",
    "running", "vehicle_state", "event", "stop_condition", "lane_found",
    "traffic_mission_enabled", "car_detector_enabled", "signal_stop_latched",
    "avoidance_blocked_by_signal",
    "stop_line_detected", "lane_error", "solid_filter_angle_deg",
    "solid_filter_kept_pixels", "solid_filter_rejected_pixels",
    "lane_width_measured_px", "left_slope_factor", "right_slope_factor",
    "raw_steer", "final_steer",
    "final_speed", "recovery_ramp_active", "recovery_ramp_elapsed_s",
    "recovery_ramp_pwm", "steering_pot", "traffic_signal", "traffic_result_age_s",
    "combined_result_age_s", "combined_inference_ms", "combined_model_fault",
    "car_detected", "car_label",
    "car_confidence", "car_bbox", "car_area_ratio", "car_result_age_s",
    "car_roi_left", "car_roi_top", "car_roi_right", "car_roi_bottom",
    "car_roi_candidate_count", "car_roi_rejected_count",
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
        self.clean_traffic_video_enabled = self.video_enabled and bool(
            cfg.get("log_clean_traffic_video_enabled", 1)
        )
        self.clean_lane_video_enabled = self.video_enabled and bool(
            cfg.get("log_clean_lane_video_enabled", 1)
        )
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
        self.clean_traffic_writer = None
        self.clean_lane_writer = None
        self.drive_size = None
        self.traffic_size = None
        self.clean_traffic_size = None
        self.clean_lane_size = None
        self.drive_writer_attempted = False
        self.traffic_writer_attempted = False
        self.clean_traffic_writer_attempted = False
        self.clean_lane_writer_attempted = False
        self.videos_stopped = False
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
                "combined_video": "combined_detection.avi",
                "clean_traffic_video": "traffic_clean.avi",
                "clean_lane_video": "lane_clean.avi",
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

    def write(self, row, drive_ui, traffic_ui, traffic_clean=None, lane_clean=None):
        if not self.enabled:
            return
        now = time.time()
        video_written = False
        if self.video_enabled:
            if not self.drive_writer_attempted:
                self.drive_writer_attempted = True
                self.drive_writer = self._open_writer("drive_ui.avi", drive_ui)
                self.drive_size = drive_ui.shape[1::-1]
            if not self.traffic_writer_attempted:
                self.traffic_writer_attempted = True
                self.traffic_writer = self._open_writer("combined_detection.avi", traffic_ui)
                self.traffic_size = traffic_ui.shape[1::-1]
            if (
                self.clean_traffic_video_enabled
                and traffic_clean is not None
                and not self.clean_traffic_writer_attempted
            ):
                self.clean_traffic_writer_attempted = True
                self.clean_traffic_writer = self._open_writer(
                    "traffic_clean.avi", traffic_clean
                )
                self.clean_traffic_size = traffic_clean.shape[1::-1]
            if (
                self.clean_lane_video_enabled
                and lane_clean is not None
                and not self.clean_lane_writer_attempted
            ):
                self.clean_lane_writer_attempted = True
                self.clean_lane_writer = self._open_writer(
                    "lane_clean.avi", lane_clean
                )
                self.clean_lane_size = lane_clean.shape[1::-1]
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
            if self.clean_traffic_writer is not None and traffic_clean is not None:
                clean_traffic_frame = (
                    traffic_clean
                    if traffic_clean.shape[1::-1] == self.clean_traffic_size
                    else cv2.resize(traffic_clean, self.clean_traffic_size)
                )
                self.clean_traffic_writer.write(clean_traffic_frame)
            if self.clean_lane_writer is not None and lane_clean is not None:
                clean_lane_frame = (
                    lane_clean
                    if lane_clean.shape[1::-1] == self.clean_lane_size
                    else cv2.resize(lane_clean, self.clean_lane_size)
                )
                self.clean_lane_writer.write(clean_lane_frame)
            self.video_frame_index += 1
            video_written = True

        values = {field: "" for field in CSV_FIELDS}
        values.update(row)
        values["wall_time"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        values["elapsed_s"] = f"{now - self.started_at:.3f}"
        values["video_frame_index"] = self.video_frame_index if video_written else ""
        values["video_recording_active"] = int(self.video_enabled)
        self.csv_writer.writerow(values)
        if now - self.last_flush >= self.flush_interval:
            self.csv_handle.flush()
            self.last_flush = now

    def stop_videos(self, reason=""):
        if not self.video_enabled:
            return False
        for writer in (
            self.drive_writer,
            self.traffic_writer,
            self.clean_traffic_writer,
            self.clean_lane_writer,
        ):
            if writer is not None:
                writer.release()
        self.drive_writer = None
        self.traffic_writer = None
        self.clean_traffic_writer = None
        self.clean_lane_writer = None
        self.video_enabled = False
        self.videos_stopped = True
        suffix = f" ({reason})" if reason else ""
        print(f"[logger] all video recording stopped{suffix}")
        return True

    def close(self):
        for writer in (
            self.drive_writer,
            self.traffic_writer,
            self.clean_traffic_writer,
            self.clean_lane_writer,
        ):
            if writer is not None:
                writer.release()
        if self.csv_handle is not None:
            self.csv_handle.flush()
            self.csv_handle.close()
        if self.session_dir is not None:
            print(f"[logger] saved -> {self.session_dir}")
