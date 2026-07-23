"""
config.py -- 장애물 미션용 파라미터 관리 (lane 프로젝트에서 분리된 독립 사본).

main_obstacle.py 가 읽고, 'w' 키로 calib.json 에 저장.
"""

import json
import os

DEFAULTS = {
    # ══ 장애물 미션 (전방 초음파 2개 + 차선변경) ═══════════════
    # 초음파 인덱스: 펌웨어 US_TRIG/US_ECHO 배열 순서 (0=전방좌, 1=전방우)
    "us_front_ids": [0, 1],   # 전방 (장애물 감지, 둘 중 가까운 값 사용)
    "us_right_ids": [],       # 우측 센서 없음 → 복귀는 시간 기반(inner_hold_s)
                              # (나중에 우측 센서 달면 여기에 인덱스 추가 시
                              #  통과확인 로직이 자동으로 우선 적용됨)

    "obs_trigger_cm": 45,     # 전방 이 거리 미만이면 장애물로 판정
    "obs_trigger_hits": 3,    # 연속 이 횟수 이상 감지돼야 차선변경 시작 (노이즈 방지)
    "inner_hold_s": 3.0,      # 안쪽 차선 유지 시간(초) — 경과하면 복귀 시작.
                              # 장애물 길이/속도에 맞춰 현장 튜닝 (핵심 파라미터!)
    "side_block_cm": 50,      # (우측 센서 있을 때만) 통과 중 판정 거리
    "side_clear_cm": 65,      # (우측 센서 있을 때만) 통과 완료 판정 거리
    "side_clear_hits": 5,
    "min_inner_s": 1.2,       # 복귀 최소 대기 (센서 기반이든 시간 기반이든 공통)
    "cooldown_s": 1.5,        # 복귀 후 이 시간 동안은 새 장애물 감지 무시
    "us_log_s": 1.0,          # 터미널에 초음파 값 출력 주기(초). 0=끔
    "lane_switch_frames": 4,  # 차선 자동판정: 연속 이 프레임 일치해야 전환

    # 차선변경 기동 (오픈루프: 고정 조향 → 카운터 조향 → 비전 재개)
    # 기하 참고: 차로폭 850mm 횡이동, 풀락 회전반경 R≈1.5m 가정 시
    # 방향각 ~44도, 편도 호 ~1.1m → PWM 90 저속에서 1초대 필요.
    # 아래는 시작값 — 실측으로 "옆 차선 중앙 근처 도착"까지 조정할 것
    "change_steer": 1.0,      # 변경 시 조향 (1.0=풀락. 조향이 느린 차라 최대 사용)
    "change_t_pre": 0.4,      # 0단계: 정지 상태로 바퀴를 풀락까지 미리 꺾는 시간(초)
    # 1·2단계 종료는 시간이 아니라 조건 기반:
    #   꺾기(OUT)   종료 = 비전 차선판정(lane_est)이 목표 차선으로 플립
    #   정렬(ALIGN) 종료 = |헤딩| < align_heading_thresh (차선과 평행해짐)
    "change_t_min": 0.3,      # 꺾기 최소 시간(초) — 출발 직후 오판정 플립 무시
    "change_t_max": 2.5,      # 각 단계 타임아웃 폴백(초) — 조건 미달성 시 강제 진행
    "align_heading_thresh": 0.10,  # 정렬 완료 헤딩 임계 (작을수록 엄격)
    "change_pwm": 90,         # 변경 중 속도 (회전반경은 속도 무관 → 빨라도 경로 동일)

    # ── 카메라 ──────────────────────────────────────────────
    "cam_index": 0,          # 노트북이 인식하는 캡처 장치 인덱스 (0/1 실측 확인)
    "camera_id": 4,          # 실물 카메라 라벨 = 장애물용 4번 → camera4_intrinsic.txt
    "frame_w": 640,          # 낮은 해상도 = 빠른 루프. 640x360이면 충분
    "frame_h": 360,
    "undistort": 1,          # 1=렌즈 왜곡 보정 (시작 시 remap맵 1회, ~0.2ms/frame)

    # ── ROI (바닥 차선 영역만) : 비율로 지정 ─────────────────
    "roi_y_top": 0.55,       # 화면 위 55%는 버림 (배경/심판 차단)
    "roi_y_bot": 0.98,
    "roi_x_left": 0.0,
    "roi_x_right": 1.0,

    # ── 흰색 검출 (조명 강건) ────────────────────────────────
    "sat_max": 80,           # HSV S 상한: 흰색은 채도 낮음
    "use_otsu": 1,           # 1=Otsu 자동 임계 (V채널, CLAHE 후)
    "v_min_floor": 140,      # Otsu가 너무 낮게 잡는 것 방지 하한
    "use_adaptive": 1,       # 1=adaptiveThreshold 병행 (부분 햇빛 대응)
    "adaptive_block": 41,    # 홀수
    "adaptive_C": -12,
    "clahe_clip": 2.5,

    # ── centroid 밴드 ────────────────────────────────────────
    "n_bands": 5,            # ROI를 가로 밴드 5개로
    "min_pix_band": 25,      # 밴드당 이 픽셀 미만이면 미검출
    "track_win_px": 90,      # 이전 라인 위치 ±win 만 탐색 (점선/노이즈 배제)
    "line_side": "right",    # 반시계 2차선 → 바깥 실선은 차 오른쪽

    # ── 추종 목표 ────────────────────────────────────────────
    "offset_px": 150,        # 실선에서 왼쪽으로 이만큼 떨어진 지점이 목표
                             # (= 차로 중앙. 워핑 안 하므로 픽셀로 직접 튜닝)
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
    "record_every": 3,       # --record 시 N프레임마다 1장 저장 (30fps→10fps)
    "curve_gain": 0.8,       # 곡률 피드포워드 게인 (커브 선회 부족→올리고, 지그재그→내리고)
    "lookahead_gain": 4.0,   # 먼 밴드 가중 배율. 긴 코너 안쪽 파고들면→내리고, 늦게 돌면→올리고
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
    "serial_port": "/dev/tty.usbmodem*",   # 맥. 자동 글롭 탐색
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
