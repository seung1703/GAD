"""
config.py -- 모든 튜닝 파라미터를 JSON 하나로 관리.

calibrate.py 가 저장하고 main.py 가 읽는다.
대회장에서 바꾸는 건 전부 여기(=calib.json)로 모인다. 코드 수정 없이 값만 교체.
"""

import json
import os

DEFAULTS = {
    # ── 카메라 ──────────────────────────────────────────────
    "cam_index": 1,          # Windows 기본 주행 카메라 인덱스
    "camera_id": 3,          # 실물 카메라 라벨 = 주행용 3번 → camera3_intrinsic.txt
    "frame_w": 640,          # 낮은 해상도 = 빠른 루프. 640x360이면 충분
    "frame_h": 360,
    "undistort": 1,          # 1=렌즈 왜곡 보정 (camera<camera_id>_intrinsic.txt 사용)
                             #   시작 시 remap맵 1회 생성 → 프레임당 ~0.2ms
                             # 0=끔 (오버헤드 0, 보정 전 원본)

    # ── 밴드 ─────────────────────────────────────────────────
    "n_bands": 5,            # BEV를 가로 밴드 N개로 나눠 스캔

    # ── 제어 ────────────────────────────────────────────────
    "kp": 1.4,               # 정규화 오차(-1..1) → 조향 명령(-1..1)
    "kd": 6.0,               # D게인: 코너 탈출 시 핸들 미리 풀기 (오버슈트 억제)
    "steer_right_gain": 1.0, # 우회전만 추가 배율 (1.0=좌우 대칭)
    "steer_sign": -1,         # 차가 반대로 꺾으면 -1로
    "lost_hold_frames": 8,   # 라인 로스트 시 마지막 조향 유지 프레임
    "lost_stop_frames": 30,  # 이 이상 로스트면 안전 정지

    # ── 주행 ────────────────────────────────────────────────
    "drive_pwm": 70,         # 뒷바퀴 PWM (네 DRIVE_SPEED)
    "slow_pwm": 55,          # 조향 클 때 감속 PWM
    "slow_steer_thresh": 0.55,  # |조향|이 이 이상이면 감속
    "brake_ramp_pwm": 6,     # 일반 정지(x키) 시 프레임당 PWM 감소량 (30fps 기준
                             # 135→0 약 0.75초). 비상정지/로스트 정지는 즉시

    # ── 딥러닝 모델 (lane_model.py) ─────────────────────────────
    "model_imgsz": 320,      # 추론 입력 크기. 320=10ms/f, 640=32ms/f (M3 CPU 실측)
    "model_conf": 0.4,       # 검출 신뢰도 임계값
    "model_device": "cpu",   # "cpu" 권장 (mps는 간헐 NMS 지연 있음)
    "reacquire_frames": 8,   # 한쪽 선을 이 프레임 수만큼 못 보면 추적 리셋(재획득)
    "bev_lane_width": 317,   # 단일 차선(한쪽 선만) 시 offset에 쓸 "고정" 차로폭(BEV px).
                             # 라이브 측정 폭은 커브서 팽창(폭/cosθ)·가장자리서 축소돼
                             # 불안정 → 고정값 사용. 직선 양선 구간의 meas 값(=수직
                             # 실측 폭)으로 맞출 것. 화면 하단 오버레이 meas 참고.
    "record_every": 3,       # --record 시 N프레임마다 1장 저장 (30fps→10fps)
    "curve_gain": 0.8,       # 곡률 피드포워드 게인 (커브 선회 부족→올리고, 지그재그→내리고)
    "lookahead_gain": 4.0,   # 먼 밴드 가중 배율. 긴 코너 안쪽 파고들면→내리고, 늦게 돌면→올리고
    "lookahead_exit_scale": 0.4,  # 코너 탈출(중앙 복귀) 국면에 룩어헤드를 이 배로 축소.
                             # 낮을수록 탈출 오버슈트↓(핸들 빨리 풀림), 1.0=비대칭 끔
    "heading_gain": 0.5,     # 헤딩(차선 대비 차체 틀어짐) 보정 게인. 0=끔 (Stanley식)

    # ── IPM (버드아이뷰 변환) : 사다리꼴 네 모서리 개별 지정 ────
    # 각 값은 프레임 크기 대비 비율. x는 1.0 초과/음수 허용(화면 밖 코너).
    "ipm_tl_x": 0.25,  "ipm_tl_y": 0.40,   # 좌상
    "ipm_tr_x": 0.75,  "ipm_tr_y": 0.40,   # 우상
    "ipm_bl_x": -0.40, "ipm_bl_y": 0.98,   # 좌하
    "ipm_br_x": 1.40,  "ipm_br_y": 0.98,   # 우하
    "center_offset": 0,      # 중심 보정(px). 양수→기준 오른쪽→차 왼쪽 보정

    # (포텐 캘리브레이션은 통합 펌웨어 firmware/firmware.ino의 STEER_* 가 전담.
    #  파이썬은 정규화 조향 -1..1 만 보낸다.)

    # ── 시리얼 ──────────────────────────────────────────────
    "steer_mode": "continuous",  # 조향 인코딩. "continuous"=7비트 연속(통합펌웨어 v4+),
                                 # "bang3"=구 3단 양자화(레거시). 반드시 continuous.
    "serial_port": "COM3",   # Windows 기본 아두이노 포트
    "baud": 115200,
}

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")


def load():
    cfg = dict(DEFAULTS)
    if os.path.exists(PATH):
        with open(PATH) as f:
            cfg.update(json.load(f))
    return cfg


def save(cfg):
    with open(PATH, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[config] saved -> {PATH}")
