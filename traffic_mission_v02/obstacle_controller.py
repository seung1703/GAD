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


class ObstacleController:
    KEEP = "KEEP"
    WAIT_DASH = "WAIT_DASH"
    CHG_PRE = "CHG_PRE"
    CHANGING = "CHANGING"
    COUNTER_STEER = "COUNTER_STEER"

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

        self.dash_side = None
        self.locked_dash_side = None
        self.locked_maneuver_pwm = None
        self._direction_candidate = None
        self._direction_hits = 0
        self._direction_misses = 0

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
        if self.state not in {self.KEEP, self.WAIT_DASH}:
            return

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
        return self.state in {self.KEEP, self.WAIT_DASH}

    def avoidance_in_progress(self):
        return self.state in {
            self.CHG_PRE,
            self.CHANGING,
            self.COUNTER_STEER,
        }

    def _reset_direction_context(self):
        self.dash_side = None
        self.locked_dash_side = None
        self._direction_candidate = None
        self._direction_hits = 0
        self._direction_misses = 0

    def _lock_direction(self):
        if self.dash_side not in {"left", "right"}:
            return False
        self.locked_dash_side = self.dash_side
        return True

    def _steer_toward_dash(self):
        steer = abs(float(self.cfg.get("change_steer", 1.0)))
        # The serial protocol uses positive steering for left and negative for right.
        return steer if self.locked_dash_side == "left" else -steer

    def _steer_away_from_dash(self):
        return -self._steer_toward_dash()

    def _approach_speed(self):
        if self.front_cm is None:
            return None
        if self.front_cm <= float(self.cfg.get("obstacle_slow_cm", 80)):
            return int(self.cfg.get("obstacle_slow_pwm", 50))
        return None

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
            if self.fusion_confirmed:
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
        new_fusion_pair = new_sample and (
            int(car_sequence) >= 0 or not car_required
        )
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
        change_time = float(self.cfg.get("change_duration_s", 2.0))
        counter_time = float(self.cfg.get("counter_steer_duration_s", 0.4))

        if self.state == self.KEEP:
            if not motion_enabled:
                self.front_hits = 0
                return self._decision(obstacle_detected, False, phase="KEEP_PAUSED")

            if new_fusion_pair:
                self.front_hits = self.front_hits + 1 if obstacle_detected else 0

            if self.front_hits >= int(self.cfg.get("obs_trigger_hits", 3)):
                self.front_hits = 0
                if not self._lock_direction():
                    self._start(self.WAIT_DASH)
                    return self._decision(
                        True,
                        False,
                        self._approach_speed(),
                        None,
                    )
                if pre_time > 0:
                    self._lock_maneuver_speed()
                    self._start(self.CHG_PRE)
                    return self._decision(
                        True,
                        True,
                        self._maneuver_speed(),
                        0.0,
                    )
                self._lock_maneuver_speed()
                self._start(self.CHANGING)
                return self._decision(
                    True,
                    True,
                    self._maneuver_speed(),
                    self._steer_toward_dash(),
                )

            return self._decision(
                obstacle_detected,
                False,
                self._approach_speed(),
                None,
            )

        if self.state == self.WAIT_DASH:
            if new_sample and not obstacle_detected:
                self._start(self.KEEP)
                return self._decision(False, False, self._approach_speed(), None)
            if self._lock_direction():
                if pre_time > 0:
                    self._lock_maneuver_speed()
                    self._start(self.CHG_PRE)
                    return self._decision(
                        True,
                        True,
                        self._maneuver_speed(),
                        0.0,
                    )
                self._lock_maneuver_speed()
                self._start(self.CHANGING)
                return self._decision(
                    True,
                    True,
                    self._maneuver_speed(),
                    self._steer_toward_dash(),
                )
            return self._decision(
                obstacle_detected,
                False,
                self._approach_speed(),
                None,
            )

        if self.state == self.CHG_PRE:
            if self.state_elapsed < pre_time:
                return self._decision(
                    True,
                    True,
                    self._maneuver_speed(),
                    0.0,
                    sensor_ok=sensor_ok,
                )
            self._start(self.CHANGING)
            return self._decision(
                True,
                True,
                self._maneuver_speed(),
                self._steer_toward_dash(),
                sensor_ok=sensor_ok,
            )

        if self.state == self.CHANGING:
            if self.state_elapsed >= change_time:
                self._start(self.COUNTER_STEER)
                return self._decision(
                    True,
                    True,
                    self._maneuver_speed(),
                    self._steer_away_from_dash(),
                    sensor_ok=sensor_ok,
                )
            return self._decision(
                True,
                True,
                self._maneuver_speed(),
                self._steer_toward_dash(),
                sensor_ok=sensor_ok,
            )

        if self.state == self.COUNTER_STEER:
            if self.state_elapsed >= counter_time:
                completed_speed = self._maneuver_speed()
                self._start(self.KEEP)
                self.front_hits = 0
                self.front_cm = None
                self._reset_direction_context()
                self.locked_maneuver_pwm = None
                return self._decision(
                    False,
                    False,
                    completed_speed,
                    None,
                    completed=True,
                    sensor_ok=sensor_ok,
                )
            return self._decision(
                True,
                True,
                self._maneuver_speed(),
                self._steer_away_from_dash(),
                sensor_ok=sensor_ok,
            )

        self.reset()
        return self._decision(obstacle_detected, False)
