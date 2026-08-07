"""Unified runtime configuration for traffic_mission_v02_0805.

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
    "traffic_frame_w": 640,
    "traffic_frame_h": 360,
    "undistort": 1,
    # "auto" 면 실행 플랫폼에 맞는 백엔드를 고른다
    # (macOS=CAP_AVFOUNDATION, Windows=CAP_DSHOW, Linux=CAP_V4L2)
    "camera_backend": "auto",
    # Tk 조절창 사용 여부. 0 이면 UI 없이 주행만 한다.
    "ui_controls_enabled": 1,
    # UI 는 제어 루프보다 느려도 된다. macOS 에서 imshow+waitKey 가 16ms,
    # waitKey 만 해도 11ms 다. 2 면 UI 를 절반 주기로만 그린다.
    "ui_render_every_n": 2,
    # Tk 조절창 폴링 주기. 한 번에 7.5ms 라 매 루프 돌 필요가 없다.
    "ui_controls_poll_every_n": 5,
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
    "drive_pwm": 255,
    "slow_pwm": 255,
    "slow_steer_thresh": 0.55,
    "model_imgsz": 320,
    "model_conf": 0.40,
    "model_mask_threshold": 0.50,
    "mask_close_kernel": 0,
    "model_device": "cpu",
    "model_cpu_threads": 6,
    "model_cpu_interop_threads": 2,
    "reacquire_frames": 8,
    "bev_lane_width": 328,
    "curve_gain": 0.6,
    "lookahead_gain": 5.0,
    "lookahead_exit_scale": 1.5,
    "lookahead_exit_delta_px": 1.0,
    "lookahead_exit_confirm_frames": 1,
    "heading_gain": 0.6,
    "heading_ema_alpha": 1.0,
    "heading_clamp": 1.0,
    "heading_min_bands": 2,
    "control_warmup_frames": 3,
    "combined_image_size": 640,
    "combined_inference_interval_s": 0.12,   # 약 8Hz.
    # 신호등/장애물은 25Hz 로 볼 이유가 없다. 예전 0.04(25Hz)는 M3 CPU 에서
    # best_11n(59ms) 이 따라잡지 못해 검출 스레드가 쉬지 않고 돌며 차선 모델의
    # CPU 를 뺏었다. traffic_result_stale_s=0.5, lane_confirm_count=2 라
    # 0.12 여도 신호 판정은 0.3초 안에 확정된다.
    "traffic_confidence": 0.25,
    "traffic_result_stale_s": 0.50,
    "car_model_required": 1,
    "car_roi_left": 0.15,
    "car_roi_top": 0.15,
    "car_roi_right": 0.85,
    "car_roi_bottom": 1.0,
    "car_confidence": 0.60,
    "car_min_area_ratio": 0.005,
    "car_class_names": ["stroller"],
    "car_result_stale_s": 0.80,
    "combined_pipeline_timeout_s": 2.0,
    "combined_startup_grace_s": 5.0,
    "fusion_max_skew_s": 0.60,
    "lane_confirm_count": 2,
    "stop_signal_confirm_count": 2,
    "go_signal_confirm_count": 2,
    "view_width": 896,
    "view_height": 504,
    "panel_width": 540,
    "inner_lane_solid_side": "left",
    "mask_roi_margin_x": 0.0,
    "mask_roi_margin_y": 0.0,
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
    "ipm_tl_x": 0.13,
    "ipm_tl_y": 0.46,
    "ipm_tr_x": 0.91,
    "ipm_tr_y": 0.46,
    "ipm_bl_x": -0.05,
    "ipm_bl_y": 0.94,
    "ipm_br_x": 1.09,
    "ipm_br_y": 0.94,
    "center_offset": 0,
    "steer_mode": "continuous",
    # Windows 는 COM3, macOS/Linux 는 실행 시 자동으로 실제 장치를 찾는다
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
    # ── 장애물 회피 v2 ────────────────────────────────────
    # 현재 차선. 1↔2 를 토글하며 회피 방향을 정한다. 출발 차선을 적는다.
    "start_lane_mode": 2,
    # 1차선이 차체 기준 어느 쪽인가. 회피가 반대로 나가면 이 값을 뒤집을 것.
    "lane1_side": "left",
    # 차선 모델의 INNER/OUTER 추정으로 lane_mode 를 따라가게 할지.
    # 기본 0(끔) — 시야가 흔들리면 lane_mode 가 멋대로 바뀌어 회피 방향이 뒤집힌다.
    "lane_mode_follow_vision": 0,
    # 초음파 시간창(초). 이 동안의 **최솟값**을 전방 거리로 쓴다.
    # 펌웨어는 에코를 못 받으면 250 을 반환하는데(못 읽음), 순간값을 그대로
    # 쓰면 그 250 이 연속감지 카운트를 리셋해 회피가 영영 발동하지 않는다.
    "us_window_s": 0.3,
    # 펌웨어 타임아웃 sentinel. 이 값 이상은 '측정 실패'로 보고 창에 안 넣는다.
    "us_timeout_cm": 250,
    # 노이즈 필터: 초음파가 이만큼 연속으로 가까워야 장애물로 인정
    "avoid_confirm_frames": 5,
    # 커브 필터: 조향 절대값이 이 이하(직선)일 때만 회피한다.
    # 커브를 돌 때 정면 벽을 장애물로 오인하는 걸 막는 핵심 값이다.
    "avoid_max_steer": 0.4,
    # 핸들을 꺾고 유지하는 시간과 그때의 강제 조향값
    "lane_change_seconds": 1.4,
    "lane_change_steer": 1.0,
    # 회피 직후 재발동 금지 시간 (잘못된 차선/센서 노이즈로 인한 2차 회피 방지)
    "avoid_cooldown_seconds": 2.0,
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
    "log_clean_traffic_video_enabled": 1,
    "log_clean_lane_video_enabled": 1,
    "video_stop_after_signal_resume_s": 3.0,
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
