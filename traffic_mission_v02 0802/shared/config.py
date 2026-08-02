"""Unified runtime configuration for traffic_mission_v02.

All tunable values live in ``shared/calib.json``. This module provides
defaults, loading, and saving so the runtime can stay code-stable while the
car is tuned from JSON or OpenCV trackbars.
"""

from __future__ import annotations

import json
import os


DEFAULTS = {
    "cam_index": 1,
    "traffic_cam_index": 2,
    "camera_id": 3,
    "frame_w": 640,
    "frame_h": 360,
    "traffic_frame_w": 1280,
    "traffic_frame_h": 720,
    "undistort": 1,
    "camera_backend": "CAP_DSHOW",
    "camera_fps": 30,
    "camera_fourcc": "MJPG",
    "camera_buffer_size": 1,
    "n_bands": 7,
    "kp": 1.8,
    "kd": 6.0,
    "steer_right_gain": 1.0,
    "steer_sign": -1,
    "lost_hold_frames": 8,
    "lost_stop_frames": 30,
    "drive_pwm": 70,
    "slow_pwm": 55,
    "slow_steer_thresh": 0.55,
    "model_imgsz": 320,
    "model_conf": 0.30,
    "model_mask_threshold": 0.40,
    "mask_close_kernel": 3,
    "model_device": "cpu",
    "model_cpu_threads": 6,
    "model_cpu_interop_threads": 2,
    "reacquire_frames": 8,
    "bev_lane_width": 328,
    "curve_gain": 0.3,
    "lookahead_gain": 5.0,
    "lookahead_exit_scale": 0.4,
    "lookahead_exit_delta_px": 1.0,
    "lookahead_exit_confirm_frames": 3,
    "heading_gain": 0.2,
    "heading_ema_alpha": 0.25,
    "heading_clamp": 0.25,
    "heading_min_bands": 4,
    "control_warmup_frames": 3,
    "traffic_confidence": 0.25,
    "traffic_image_size": 416,
    "traffic_inference_interval_s": 0.08,
    "traffic_result_stale_s": 0.50,
    "traffic_roi_left": 0.20,
    "traffic_roi_top": 0.05,
    "traffic_roi_right": 0.80,
    "traffic_roi_bottom": 0.55,
    "car_model_required": 1,
    "car_roi_left": 0.0,
    "car_roi_top": 0.0,
    "car_roi_right": 1.0,
    "car_roi_bottom": 1.0,
    "car_image_size": 512,
    "car_confidence": 0.60,
    "car_min_area_ratio": 0.005,
    "car_class_names": ["obstacle-car"],
    "car_inference_interval_s": 0.06,
    "car_result_stale_s": 0.80,
    "car_pipeline_timeout_s": 2.0,
    "car_startup_grace_s": 5.0,
    "fusion_max_skew_s": 0.60,
    "lane_confirm_count": 2,
    "stop_signal_confirm_count": 2,
    "go_signal_confirm_count": 2,
    "view_width": 800,
    "view_height": 520,
    "panel_width": 440,
    "inner_lane_solid_side": "left",
    "mask_roi_margin_x": 0.08,
    "mask_roi_margin_y": 0.03,
    "boundary_class_min_pixels": 2,
    "boundary_class_ratio": 1.10,
    "solid_orientation_filter_enabled": 1,
    "solid_min_component_pixels": 20,
    "solid_min_vertical_span_px": 30,
    "solid_max_angle_from_vertical_deg": 60.0,
    "solid_prev_angle_tolerance_deg": 35.0,
    "solid_fit_corridor_px": 18,
    "solid_angle_ema_alpha": 0.25,
    "solid_angle_reacquire_frames": 5,
    "ipm_tl_x": 0.21,
    "ipm_tl_y": 0.46,
    "ipm_tr_x": 0.89,
    "ipm_tr_y": 0.46,
    "ipm_bl_x": -0.01,
    "ipm_bl_y": 0.94,
    "ipm_br_x": 1.08,
    "ipm_br_y": 0.94,
    "center_offset": 0,
    "steer_mode": "continuous",
    "serial_port": "COM3",
    "baud": 115200,
    "us_front_ids": [0, 1],
    "obs_trigger_cm": 150,
    "obs_trigger_hits": 2,
    "obstacle_slow_cm": 80,
    "obstacle_slow_pwm": 50,
    "direction_confirm_frames": 3,
    "direction_hold_frames": 5,
    "change_steer": 1.0,
    "change_t_pre": 0.0,
    "change_duration_s": 2.0,
    "counter_steer_duration_s": 1.5,
    "change_pwm": 70,
    "recovery_ramp_start_pwm": 70,
    "recovery_ramp_duration_s": 3.0,
    "sensor_timeout_s": 0.8,
    "sensor_min_valid_cm": 2,
    "sensor_max_valid_cm": 400,
    "sensor_required": 1,
    "ultrasonic_enabled": 1,
    "fsm_max_step_s": 0.25,
    "logging_enabled": 1,
    "log_dir": "logs",
    "log_video_enabled": 1,
    "log_video_fps": 15.0,
    "log_video_codec": "MJPG",
    "log_flush_interval_s": 1.0,
    "log_ultrasonic_ids": [0, 1, 2, 3, 4, 5, 6, 7],
}


PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")


def load():
    cfg = dict(DEFAULTS)
    if os.path.exists(PATH):
        with open(PATH, encoding="utf-8") as handle:
            cfg.update(json.load(handle))
    return cfg


def save(cfg):
    with open(PATH, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, ensure_ascii=False)
    print(f"[config] saved -> {PATH}")
