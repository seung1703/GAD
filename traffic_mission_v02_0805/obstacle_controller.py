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
    mission_completed: bool = False


class ObstacleController:
    """
    장애물 회피: outer 에서 출발 -> inner 로 피함 -> 무조건 outer 로 복귀 -> 종료.

    ★ 회피 방향을 차선 카메라에서 받지 않는다 ★
      예전에는 차선 모델이 알려준 dash_side 로 어느 쪽으로 피할지 정했다. 그래서
      점선을 못 찾으면 WAIT_DASH 에 머물러 **회피 자체를 시작하지 못하는** 실패
      모드가 있었다. 지금은 "항상 outer 에서 출발한다"를 전제로 깔았으므로
      첫 기동은 언제나 outer->inner, 복귀는 언제나 inner->outer 다. 방향은
      `inner_lane_side` 설정 하나로 결정되고 런타임 인식에 의존하지 않는다.

    상태 흐름:
      KEEP ─(stroller + 초음파 융합 확인)→ [CHG_PRE] → CHANGING → COUNTER_STEER
           → INNER_HOLD ─(inner_hold_s 경과)→ RETURN_CHANGING → RETURN_COUNTER
           → DONE (이후 장애물 무시)

    INNER_HOLD 는 회피 기동이 아니다. 이 구간에서는 조향을 넘기지 않고 평소대로
    차선 추종으로 달린다. 그래서 `avoidance_in_progress()` 에 포함되지 않는다.

    `avoidance_completed` 는 "기동이 방금 끝났다"는 뜻이라 **두 번** 뜬다
    (inner 진입 직후, outer 복귀 직후). 실행부는 이 신호로 차선 추종을 다시
    잠그고 속도 램프를 시작하므로, 차선을 갈아탄 두 시점 모두에서 필요하다.
    미션 자체가 끝나는 시점은 `mission_completed` 로 따로 알린다.
    """

    KEEP = "KEEP"
    CHG_PRE = "CHG_PRE"
    CHANGING = "CHANGING"
    COUNTER_STEER = "COUNTER_STEER"
    INNER_HOLD = "INNER_HOLD"
    RETURN_CHANGING = "RETURN_CHANGING"
    RETURN_COUNTER = "RETURN_COUNTER"
    DONE = "DONE"

    MANEUVER_STATES = frozenset(
        {CHG_PRE, CHANGING, COUNTER_STEER, RETURN_CHANGING, RETURN_COUNTER}
    )

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = self.KEEP
        self.front_hits = 0
        self.front_cm = None
        self.front_sensor_values = {}
        self.state_elapsed = 0.0
        self.last_update_ts = None
        self.last_sample_token = None
        self.last_fusion_token = None
        self.car_detected = False
        self.car_result_age_s = None
        self.ultrasonic_close = False
        self.fusion_confirmed = False
        self.fusion_skew_s = None
        self.fusion_reason = "waiting"

        # 회피 방향은 설정에서 나오지만, 어느 쪽으로 갔는지 기록은 남긴다(로그/UI용)
        self.dash_side = None
        self.locked_dash_side = None
        self.locked_change_direction = None
        self.locked_change_duration_s = None
        self.locked_maneuver_pwm = None
        self._direction_candidate = None
        self._direction_hits = 0
        self._direction_misses = 0

    # ── 방향 ────────────────────────────────────────────────
    def inner_side(self):
        """inner 차선이 차체 기준 어느 쪽인가. 'left' 또는 'right'."""
        side = self.cfg.get("inner_lane_side")
        if side is None:
            # 예전 키에서 유추: inner 차선의 실선이 왼쪽이면 inner 차선도 왼쪽이다
            side = self.cfg.get("inner_lane_solid_side", "left")
        side = str(side).lower()
        return side if side in {"left", "right"} else "left"

    def _steer_side(self, side):
        # 시리얼 프로토콜은 양수가 왼쪽, 음수가 오른쪽이다
        steer = abs(float(self.cfg.get("change_steer", 1.0)))
        return steer if side == "left" else -steer

    def _steer_to_inner(self):
        return self._steer_side(self.inner_side())

    def _steer_to_outer(self):
        return -self._steer_to_inner()

    # ── 시간 ────────────────────────────────────────────────
    def _start(self, state):
        self.state = state
        self.state_elapsed = 0.0

    def _advance_clock(self, now, motion_enabled, sensor_ok):
        if self.last_update_ts is None:
            self.last_update_ts = now
            return
        dt = max(0.0, now - self.last_update_ts)
        self.last_update_ts = now
        dt = min(dt, float(self.cfg.get("fsm_max_step_s", 0.25)))
        if motion_enabled and sensor_ok:
            self.state_elapsed += dt

    def _fresh_front(self, now, us, timestamps):
        self.front_sensor_values = {}
        front_ids = [int(sensor_id) for sensor_id in self.cfg.get("us_front_ids", [0, 1])]
        timeout_s = float(self.cfg.get("sensor_timeout_s", 0.8))
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
            sample_ts = timestamps.get(sensor_id)
            if sample_ts is None or now - float(sample_ts) > timeout_s:
                continue
            if min_valid <= distance <= max_valid:
                fresh.append((distance, sensor_id, float(sample_ts)))
                self.front_sensor_values[sensor_id] = distance

        if not fresh:
            return None, None, bool(front_ids)

        distance, sensor_id, sample_ts = min(fresh)
        token = (sensor_id, sample_ts, distance)
        return distance, token, True

    def update_lane_context(self, dash_side=None, solid_side=None):
        """회피 방향 결정에는 더 이상 쓰지 않는다. 화면/로그 표시용 기록만 남긴다."""
        candidate = dash_side if dash_side in {"left", "right"} else None
        if candidate is None and solid_side in {"left", "right"}:
            candidate = "right" if solid_side == "left" else "left"

        if candidate is None:
            self._direction_misses += 1
            if self._direction_misses >= int(self.cfg.get("direction_hold_frames", 5)):
                self.dash_side = None
                self._direction_candidate = None
                self._direction_hits = 0
            return

        self._direction_misses = 0
        if candidate == self._direction_candidate:
            self._direction_hits += 1
        else:
            self._direction_candidate = candidate
            self._direction_hits = 1

        if self._direction_hits >= int(self.cfg.get("direction_confirm_frames", 3)):
            self.dash_side = candidate

    def lane_recognition_enabled(self):
        """기동 중에는 차선을 가로지르므로 추적을 끈다. 그 외에는 켠다."""
        return not self.avoidance_in_progress()

    def avoidance_in_progress(self):
        return self.state in self.MANEUVER_STATES

    def mission_finished(self):
        return self.state == self.DONE

    def _reset_direction_context(self):
        self.dash_side = None
        self.locked_dash_side = None
        self.locked_change_direction = None
        self.locked_change_duration_s = None
        self._direction_candidate = None
        self._direction_hits = 0
        self._direction_misses = 0

    def _begin_avoid(self):
        """outer -> inner 기동 준비. 방향은 설정에서 나오므로 실패할 수 없다."""
        inner = self.inner_side()
        self.locked_dash_side = inner
        self.locked_change_direction = "outer_to_inner"
        legacy = float(self.cfg.get("change_duration_s", 2.0))
        self.locked_change_duration_s = max(
            0.0, float(self.cfg.get("change_duration_outer_to_inner_s", legacy))
        )
        self._lock_maneuver_speed()

    def _begin_return(self):
        """inner -> outer 복귀 기동 준비."""
        outer = "right" if self.inner_side() == "left" else "left"
        self.locked_dash_side = outer
        self.locked_change_direction = "inner_to_outer"
        legacy = float(self.cfg.get("change_duration_s", 2.0))
        self.locked_change_duration_s = max(
            0.0, float(self.cfg.get("change_duration_inner_to_outer_s", legacy))
        )
        self.locked_maneuver_pwm = None
        self._lock_maneuver_speed()

    def _approach_speed(self):
        if self.front_cm is None:
            return None
        limits = []
        if self.fusion_confirmed:
            limits.append(int(self.cfg.get("change_pwm", 70)))
        if self.front_cm <= float(self.cfg.get("obstacle_slow_cm", 80)):
            limits.append(int(self.cfg.get("obstacle_slow_pwm", 50)))
        return min(limits) if limits else None

    def _maneuver_speed(self):
        if self.locked_maneuver_pwm is not None:
            return self.locked_maneuver_pwm
        change_pwm = int(self.cfg.get("change_pwm", 70))
        approach_speed = self._approach_speed()
        return change_pwm if approach_speed is None else min(change_pwm, approach_speed)

    def _lock_maneuver_speed(self):
        self.locked_maneuver_pwm = self._maneuver_speed()

    def _decision(
        self,
        detected,
        active,
        speed=None,
        steering=None,
        completed=False,
        sensor_ok=True,
        phase=None,
        mission_completed=False,
    ):
        return ObstacleDecision(
            detected,
            active,
            speed,
            steering,
            completed,
            sensor_ok,
            self.front_cm,
            self.state if phase is None else phase,
            mission_completed,
        )

    def reset(self):
        self.state = self.KEEP
        self.front_hits = 0
        self.front_cm = None
        self.front_sensor_values = {}
        self.state_elapsed = 0.0
        self.last_update_ts = None
        self.last_sample_token = None
        self.last_fusion_token = None
        self.car_detected = False
        self.car_result_age_s = None
        self.ultrasonic_close = False
        self.fusion_confirmed = False
        self.fusion_skew_s = None
        self.fusion_reason = "reset"
        self._reset_direction_context()
        self.locked_maneuver_pwm = None

    def decide(
        self,
        now,
        us,
        timestamps=None,
        motion_enabled=True,
        car_detected=False,
        car_frame_time=None,
        car_sequence=-1,
    ):
        timestamps = timestamps or {}
        sensor_required = (
            bool(self.cfg.get("sensor_required", 1))
            and bool(self.cfg.get("ultrasonic_enabled", 1))
            and bool(self.cfg.get("us_front_ids", [0, 1]))
        )
        maneuver_active = self.avoidance_in_progress()
        if maneuver_active:
            self.front_cm = None
            self.front_sensor_values = {}
            sample_token = None
            sensor_ok = True
            self.fusion_confirmed = True
            self.fusion_reason = "maneuver-locked"
        else:
            self.front_cm, sample_token, has_front_sensor = self._fresh_front(now, us, timestamps)
            sensor_ok = (not sensor_required) or (has_front_sensor and self.front_cm is not None)
        self._advance_clock(now, motion_enabled, sensor_ok or maneuver_active)

        trigger_cm = float(self.cfg.get("obs_trigger_cm", 100))
        self.ultrasonic_close = self.front_cm is not None and self.front_cm < trigger_cm
        self.car_detected = bool(car_detected)
        self.car_result_age_s = (
            None if car_frame_time is None else max(0.0, now - float(car_frame_time))
        )
        car_fresh = (
            self.car_detected
            and self.car_result_age_s is not None
            and self.car_result_age_s <= float(self.cfg.get("car_result_stale_s", 0.8))
        )
        sample_time = None if sample_token is None else float(sample_token[1])
        self.fusion_skew_s = (
            None
            if sample_time is None or car_frame_time is None
            else abs(sample_time - float(car_frame_time))
        )
        car_required = bool(self.cfg.get("car_model_required", 1))
        skew_ok = (
            self.fusion_skew_s is not None
            and self.fusion_skew_s <= float(self.cfg.get("fusion_max_skew_s", 0.6))
        )
        if not maneuver_active:
            self.fusion_confirmed = self.ultrasonic_close and (
                (car_fresh and skew_ok) or not car_required
            )
            if self.state == self.DONE:
                self.fusion_reason = "mission-done-ignoring"
            elif self.fusion_confirmed:
                self.fusion_reason = "car+ultrasonic"
            elif not self.ultrasonic_close:
                self.fusion_reason = "ultrasonic-not-close"
            elif not car_fresh:
                self.fusion_reason = "car-not-detected-or-stale"
            else:
                self.fusion_reason = "timestamp-skew"
        obstacle_detected = maneuver_active or self.fusion_confirmed
        new_sample = sample_token is not None and sample_token != self.last_sample_token
        if sample_token is not None:
            self.last_sample_token = sample_token
        fusion_token = (sample_token, int(car_sequence))
        new_fusion_pair = new_sample and self.fusion_confirmed
        if new_fusion_pair:
            self.last_fusion_token = fusion_token

        if not sensor_ok and not maneuver_active:
            return self._decision(
                False,
                False,
                0,
                0.0,
                sensor_ok=False,
                phase=f"{self.state}_SENSOR_TIMEOUT",
            )

        if not motion_enabled and maneuver_active:
            return self._decision(
                obstacle_detected,
                True,
                0,
                0.0,
                phase=f"{self.state}_PAUSED",
            )

        pre_time = float(self.cfg.get("change_t_pre", 0.0))
        change_time = (
            float(self.locked_change_duration_s)
            if self.locked_change_duration_s is not None
            else float(self.cfg.get("change_duration_s", 2.0))
        )
        counter_time = float(self.cfg.get("counter_steer_duration_s", 0.4))
        return_counter_time = float(
            self.cfg.get("return_counter_steer_duration_s", counter_time)
        )
        hold_time = float(self.cfg.get("inner_hold_s", 3.0))

        # ── 회피 완료. 이후 장애물은 보지 않는다 ──────────────
        if self.state == self.DONE:
            return self._decision(False, False, phase=self.DONE)

        # ── outer 주행 중, 회피 트리거 대기 ────────────────────
        if self.state == self.KEEP:
            if not motion_enabled:
                self.front_hits = 0
                return self._decision(obstacle_detected, False, phase="KEEP_PAUSED")

            if new_sample:
                self.front_hits = self.front_hits + 1 if new_fusion_pair else 0

            if self.front_hits >= int(self.cfg.get("obs_trigger_hits", 3)):
                self.front_hits = 0
                self._begin_avoid()
                if pre_time > 0:
                    self._start(self.CHG_PRE)
                    return self._decision(True, True, self._maneuver_speed(), 0.0)
                self._start(self.CHANGING)
                return self._decision(
                    True, True, self._maneuver_speed(), self._steer_to_inner()
                )

            return self._decision(
                obstacle_detected,
                False,
                self._approach_speed(),
                None,
            )

        # ── 기동 전 정지/직진 (change_t_pre > 0 일 때만) ───────
        if self.state == self.CHG_PRE:
            if self.state_elapsed < pre_time:
                return self._decision(
                    True, True, self._maneuver_speed(), 0.0, sensor_ok=sensor_ok
                )
            self._start(self.CHANGING)
            return self._decision(
                True,
                True,
                self._maneuver_speed(),
                self._steer_to_inner(),
                sensor_ok=sensor_ok,
            )

        # ── outer -> inner ────────────────────────────────────
        if self.state == self.CHANGING:
            if self.state_elapsed >= change_time:
                self._start(self.COUNTER_STEER)
                return self._decision(
                    True,
                    True,
                    self._maneuver_speed(),
                    self._steer_to_outer(),
                    sensor_ok=sensor_ok,
                )
            return self._decision(
                True,
                True,
                self._maneuver_speed(),
                self._steer_to_inner(),
                sensor_ok=sensor_ok,
            )

        if self.state == self.COUNTER_STEER:
            if self.state_elapsed >= counter_time:
                settled_speed = self._maneuver_speed()
                self._start(self.INNER_HOLD)
                self.front_hits = 0
                self.front_cm = None
                self.locked_maneuver_pwm = None
                # inner 차선에 자리잡았다. 여기서 차선 추종을 다시 잠근다.
                return self._decision(
                    False,
                    False,
                    settled_speed,
                    None,
                    completed=True,
                    sensor_ok=sensor_ok,
                )
            return self._decision(
                True,
                True,
                self._maneuver_speed(),
                self._steer_to_outer(),
                sensor_ok=sensor_ok,
            )

        # ── inner 차선 주행 유지. 조향은 차선 추종에 맡긴다 ────
        if self.state == self.INNER_HOLD:
            if not motion_enabled:
                return self._decision(False, False, phase="INNER_HOLD_PAUSED")
            if self.state_elapsed >= hold_time:
                self._begin_return()
                self._start(self.RETURN_CHANGING)
                return self._decision(
                    False,
                    True,
                    self._maneuver_speed(),
                    self._steer_to_outer(),
                    sensor_ok=sensor_ok,
                )
            return self._decision(False, False, None, None, sensor_ok=sensor_ok)

        # ── inner -> outer 복귀 ───────────────────────────────
        if self.state == self.RETURN_CHANGING:
            if self.state_elapsed >= change_time:
                self._start(self.RETURN_COUNTER)
                return self._decision(
                    False,
                    True,
                    self._maneuver_speed(),
                    self._steer_to_inner(),
                    sensor_ok=sensor_ok,
                )
            return self._decision(
                False,
                True,
                self._maneuver_speed(),
                self._steer_to_outer(),
                sensor_ok=sensor_ok,
            )

        if self.state == self.RETURN_COUNTER:
            if self.state_elapsed >= return_counter_time:
                settled_speed = self._maneuver_speed()
                self._start(self.DONE)
                self.front_hits = 0
                self.front_cm = None
                self._reset_direction_context()
                self.locked_maneuver_pwm = None
                return self._decision(
                    False,
                    False,
                    settled_speed,
                    None,
                    completed=True,
                    sensor_ok=sensor_ok,
                    mission_completed=True,
                )
            return self._decision(
                False,
                True,
                self._maneuver_speed(),
                self._steer_to_inner(),
                sensor_ok=sensor_ok,
            )

        self.reset()
        return self._decision(obstacle_detected, False)
