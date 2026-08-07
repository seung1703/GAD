"""
obstacle_controller.py -- 장애물 회피 v2

초음파(거리) + 비전(객체) 융합으로 오작동을 막고, 상태머신으로 차선을 바꾼다.
v1(traffic_mission_v02_0805)과 달라진 점:

  - 상태가 둘뿐이다: normal / changing. 반대조향(counter steer)도, 원래 차선으로
    되돌아오는 복귀 기동도 없다. 대신 **차선 설정(lane_mode)을 바꿔버리고**
    차선 추종이 새 차선을 잡게 한다.
  - 회피 방향을 설정 상수가 아니라 **현재 차선**에서 정한다 (1↔2 토글).
  - **직선 주행일 때만** 회피한다. 커브에서 정면 벽을 장애물로 오인하는 것을
    막는 필터다 (`avoid_max_steer`).
  - 회피 직후 **쿨타임**으로 2차 발동을 막는다.

YOLO 추론은 이 파일에 없다. `shared/combined_detector.py` 의 별도 스레드가 돌고
메인 루프는 결과를 폴링만 한다. 이 컨트롤러는 그 폴링 결과(`car_detected`,
`car_frame_time`)를 받기만 하므로 추론 때문에 프레임이 밀리지 않는다.
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ObstacleDecision:
    obstacle_detected: bool
    avoidance_active: bool
    requested_speed: Optional[int]
    requested_steering: Optional[float]
    avoidance_completed: bool
    sensor_ok: bool
    distance_cm: Optional[int]
    phase: str
    lane_mode: int = 2
    block_reason: str = ""


class ObstacleController:
    NORMAL = "normal"
    CHANGING = "changing"

    # ── 하이퍼파라미터 기본값 (calib.json 으로 덮어쓸 수 있다) ──
    OBSTACLE_CM = 140.0            # 장애물로 볼 초음파 거리
    AVOID_CONFIRM_FRAMES = 5       # 노이즈 필터: 최소 연속 감지 프레임
    AVOID_MAX_STEER = 0.4          # 이 이하로 직진 중일 때만 장애물로 인정
    LANE_CHANGE_SECONDS = 1.4      # 핸들을 꺾고 유지하는 시간
    LANE_CHANGE_STEER = 1.0        # 차선 변경 시 강제 조향값
    AVOID_COOLDOWN_SECONDS = 2.0   # 회피 직후 재발동 금지 시간

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    # ── 설정 읽기 ───────────────────────────────────────────
    def _p(self, key, default):
        try:
            return float(self.cfg.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _start_lane_mode(self):
        try:
            mode = int(self.cfg.get("start_lane_mode", 2))
        except (TypeError, ValueError):
            mode = 2
        return mode if mode in (1, 2) else 2

    def _lane1_is_left(self):
        """1차선이 차체 기준 왼쪽인가. 회피가 반대로 나가면 이 값을 뒤집을 것."""
        return str(self.cfg.get("lane1_side", "left")).lower() != "right"

    def _dir_toward(self, target_mode):
        """목표 차선으로 가려면 어느 쪽으로 꺾어야 하는가.
        시리얼 프로토콜은 **양수가 왼쪽**이다."""
        going_to_1 = target_mode == 1
        return 1.0 if (going_to_1 == self._lane1_is_left()) else -1.0

    def _change_steer(self):
        return self._p("lane_change_steer", self.LANE_CHANGE_STEER)

    def _change_pwm(self):
        """회피 기동 중 속도.

        원본(장애물회피_기공)과 **같은 키**를 쓴다. 원본은 커브 감속과 같은
        `slow_pwm` 을 회피에도 쓴다.

        주의: `obstacle_slow_pwm`(근접 크리프용, 실측 10)이 아니다. 그 값은
        장애물 코앞에서 기어가라는 뜻이라 차선 변경에 쓰면 모터가 스톨한다.

        주의2: `slow_pwm >= drive_pwm` 이면 **감속이 전혀 안 된다.** 원본의
        "감속" 의도는 그쪽 설정에서 slow_pwm < drive_pwm 이었기 때문에
        성립한 것이다. 실행부가 시작할 때 이 조건을 검사해 경고한다.
        """
        return int(self._p("slow_pwm", 90))

    # ── 센서 ────────────────────────────────────────────────
    def _fresh_front(self, now, us, timestamps):
        """전방 초음파 중 가장 가까운 유효값. → (거리, 토큰, 센서있음)

        ★ 짧은 시간창의 최솟값을 쓴다 ★
        펌웨어는 에코를 못 받으면 **250 을 반환한다**(`if (dur == 0) return 250`).
        그런데 이건 "250cm 에 뭔가 있다"가 아니라 "못 읽었다"는 뜻이다.
        실주행 로그에서 같은 센서가 `56, 250, 59, 250, 56 ...` 처럼 한 번 걸러
        타임아웃을 냈고, 그때마다 연속감지 카운트가 리셋돼 5회를 영원히 못
        채웠다 — **회피가 아예 발동할 수 없는 상태였다.**

        그래서 순간값이 아니라 `us_window_s` 동안의 최솟값을 본다. 에코를 한두
        번 놓쳐도 가까운 값이 창 안에 남아 있으면 계속 '가깝다'로 본다.
        """
        self.front_sensor_values = {}
        front_ids = [int(i) for i in self.cfg.get("us_front_ids", [0, 1])]
        timeout_s = self._p("sensor_timeout_s", 0.8)
        min_valid = int(self.cfg.get("sensor_min_valid_cm", 2))
        max_valid = int(self.cfg.get("sensor_max_valid_cm", 400))

        fresh = []
        for sensor_id in front_ids:
            if sensor_id not in us:
                continue
            try:
                distance = int(us[sensor_id])
            except (TypeError, ValueError):
                continue
            stamp = timestamps.get(sensor_id)
            if stamp is None or now - float(stamp) > timeout_s:
                continue
            if min_valid <= distance <= max_valid:
                fresh.append((distance, sensor_id, float(stamp)))
                self.front_sensor_values[sensor_id] = distance
        if not fresh:
            return None, None, bool(front_ids)
        distance, sensor_id, stamp = min(fresh)

        # 시간창 최솟값. 타임아웃 값(TIMEOUT_CM)은 창에 넣지 않는다.
        window_s = self._p("us_window_s", 0.3)
        timeout_cm = int(self._p("us_timeout_cm", 250))
        token = (sensor_id, stamp, distance)
        if token != self.last_sample_token and distance < timeout_cm:
            self._front_history.append((now, distance))
        while self._front_history and now - self._front_history[0][0] > window_s:
            self._front_history.pop(0)
        if self._front_history:
            distance = min(d for _, d in self._front_history)
        return distance, token, True

    # ── 외부에서 부르는 것들 ─────────────────────────────────
    def update_lane_estimate(self, label):
        """차선 모델이 본 INNER/OUTER 를 현재 차선에 반영한다.

        회피 중에는 무시한다 — 차선을 가로지르는 동안의 추정은 못 믿는다.
        기본은 꺼져 있다(`lane_mode_follow_vision`). 켜면 시야가 흔들릴 때
        lane_mode 가 멋대로 바뀌어 회피 방향이 뒤집힐 수 있다.
        """
        if self.avoid_state != self.NORMAL:
            return
        if not bool(self.cfg.get("lane_mode_follow_vision", 0)):
            return
        if label == "INNER":
            self.lane_mode = 1
        elif label == "OUTER":
            self.lane_mode = 2

    def avoidance_in_progress(self):
        return self.avoid_state == self.CHANGING

    def lane_recognition_enabled(self):
        """회피 중에는 차선을 가로지르므로 추적을 끈다."""
        return self.avoid_state == self.NORMAL

    def in_cooldown(self, now=None):
        return (now if now is not None else time.time()) < self.avoid_cooldown_until

    def cooldown_remaining(self, now=None):
        now = now if now is not None else time.time()
        return max(0.0, self.avoid_cooldown_until - now)

    def reset(self):
        self.avoid_state = self.NORMAL
        self.avoid_start_time = 0.0
        self.avoid_cooldown_until = 0.0
        self.avoid_dir = 0.0
        self.avoid_target_mode = None
        self.front_hit_count = 0
        self.lane_mode = self._start_lane_mode()
        self.front_cm = None
        self.front_sensor_values = {}
        self.last_sample_token = None
        self.car_detected = False
        self.car_result_age_s = None
        self.ultrasonic_close = False
        self.straight_ok = False
        self.front_blocked = False
        self.block_reason = "reset"
        self.last_steer = 0.0
        self._front_history = []

    def _decision(self, detected, active, speed=None, steering=None,
                  completed=False, sensor_ok=True, phase=None):
        return ObstacleDecision(
            detected, active, speed, steering, completed, sensor_ok,
            self.front_cm, phase or self.avoid_state,
            self.lane_mode, self.block_reason,
        )

    # ── 매 프레임 ───────────────────────────────────────────
    def decide(self, now, us, timestamps=None, motion_enabled=True,
               car_detected=False, car_frame_time=None, current_steer=0.0,
               allow_avoidance=True):
        """
        current_steer   : 차선 추종이 내놓은 조향값. 직선 주행 판정에 쓴다.
        allow_avoidance : 신호등 정지 같은 상위 미션이 회피를 막을 때 False.
        """
        timestamps = timestamps or {}
        self.last_steer = float(current_steer or 0.0)
        maneuver = self.avoidance_in_progress()

        sensor_required = (
            bool(self.cfg.get("sensor_required", 1))
            and bool(self.cfg.get("ultrasonic_enabled", 1))
            and bool(self.cfg.get("us_front_ids", [0, 1]))
        )
        if maneuver:
            # 기동 중에는 센서를 보지 않는다. 옆 차를 스쳐 지나가는 동안의
            # 초음파는 의미가 없고, 여기서 멈추면 차선 한복판에 선다.
            self.front_cm = None
            self.front_sensor_values = {}
            sample_token = None
            sensor_ok = True
        else:
            self.front_cm, sample_token, has_front = self._fresh_front(now, us, timestamps)
            sensor_ok = (not sensor_required) or (has_front and self.front_cm is not None)

        # ── 3중 검증 ────────────────────────────────────────
        trigger_cm = self._p("obs_trigger_cm", self.OBSTACLE_CM)
        max_steer = self._p("avoid_max_steer", self.AVOID_MAX_STEER)
        confirm = max(1, int(self._p("avoid_confirm_frames", self.AVOID_CONFIRM_FRAMES)))

        self.ultrasonic_close = self.front_cm is not None and self.front_cm < trigger_cm
        self.straight_ok = abs(self.last_steer) <= max_steer
        self.car_detected = bool(car_detected)
        self.car_result_age_s = (
            None if car_frame_time is None else max(0.0, now - float(car_frame_time))
        )
        car_required = bool(self.cfg.get("car_model_required", 1))
        car_fresh = (
            self.car_detected
            and self.car_result_age_s is not None
            and self.car_result_age_s <= self._p("car_result_stale_s", 0.8)
        )

        # ① 연속 감지: 같은 초음파 샘플을 두 번 세지 않는다
        new_sample = sample_token is not None and sample_token != self.last_sample_token
        if sample_token is not None:
            self.last_sample_token = sample_token
        if not maneuver and new_sample:
            self.front_hit_count = self.front_hit_count + 1 if self.ultrasonic_close else 0

        hits_ok = self.front_hit_count >= confirm
        vision_ok = car_fresh or not car_required
        # ② 직선 주행 ③ 비전 확인 — 셋을 모두 통과해야 장애물로 인정
        self.front_blocked = hits_ok and self.straight_ok and vision_ok

        if maneuver:
            self.block_reason = "changing"
        elif not self.ultrasonic_close:
            self.block_reason = "ultrasonic-far"
        elif not hits_ok:
            self.block_reason = f"hits {self.front_hit_count}/{confirm}"
        elif not self.straight_ok:
            self.block_reason = f"curve |steer|={abs(self.last_steer):.2f}>{max_steer:.2f}"
        elif not vision_ok:
            self.block_reason = "car-not-confirmed"
        elif self.in_cooldown(now):
            self.block_reason = f"cooldown {self.cooldown_remaining(now):.1f}s"
        elif not allow_avoidance:
            self.block_reason = "blocked-by-mission"
        else:
            self.block_reason = "ready"

        if not sensor_ok and not maneuver:
            self.front_hit_count = 0
            return self._decision(False, False, 0, 0.0, sensor_ok=False,
                                  phase="normal_SENSOR_TIMEOUT")

        if not motion_enabled and maneuver:
            # 원본과 동일하게 **벽시계 타이머를 그대로 흘려보낸다**. 멈춰 있는
            # 동안에도 lane_change_seconds 가 소진되므로, 오래 세워뒀다가
            # 재개하면 조향 없이 곧바로 차선변경이 끝난 것으로 처리된다.
            # (일시정지는 튜닝용이라 이 동작을 원본에 맞췄다)
            return self._decision(True, True, 0, 0.0, phase="changing_PAUSED")

        # ── Phase A: 진입 조건 ─────────────────────────────
        if self.avoid_state == self.NORMAL:
            if not motion_enabled:
                self.front_hit_count = 0
                return self._decision(False, False, phase="normal_PAUSED")

            if allow_avoidance and self.front_blocked and not self.in_cooldown(now):
                self.avoid_target_mode = 1 if self.lane_mode == 2 else 2
                self.avoid_dir = self._dir_toward(self.avoid_target_mode)
                self.avoid_state = self.CHANGING
                self.avoid_start_time = now
                self.front_hit_count = 0
                self.block_reason = "changing"
                return self._decision(True, True, self._change_pwm(),
                                      self.avoid_dir * self._change_steer())
            return self._decision(self.front_blocked, False, None, None)

        # ── Phase B: 회피 실행 (제어 덮어쓰기) ──────────────
        if now - self.avoid_start_time < self._p("lane_change_seconds",
                                                 self.LANE_CHANGE_SECONDS):
            return self._decision(True, True, self._change_pwm(),
                                  self.avoid_dir * self._change_steer())

        # ── Phase C: 복귀 및 쿨타임 ────────────────────────
        self.lane_mode = self.avoid_target_mode or self.lane_mode
        self.avoid_state = self.NORMAL
        self.avoid_target_mode = None
        self.avoid_dir = 0.0
        self.front_hit_count = 0
        self.front_cm = None
        self.avoid_cooldown_until = now + self._p("avoid_cooldown_seconds",
                                                  self.AVOID_COOLDOWN_SECONDS)
        self.block_reason = "cooldown"
        # completed=True 를 받은 실행부가 차선 추종 캐시를 초기화한다.
        return self._decision(False, False, self._change_pwm(), None,
                              completed=True, phase="normal")
