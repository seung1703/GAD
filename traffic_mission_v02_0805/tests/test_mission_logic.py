import sys
import threading
import unittest
from pathlib import Path

import numpy as np

MISSION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MISSION_DIR))

from obstacle_controller import ObstacleController
from run_traffic_mission import (
    DetectionStabilizer,
    TrafficMissionGate,
)
from control import RecoverySpeedRamp
from shared.combined_detector import AsyncCombinedDetector, CombinedDetection
from lane_model import ModelLaneDetector


class DriveLaneGeometryTests(unittest.TestCase):
    def test_curve_slope_expands_single_boundary_offset(self):
        points = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
        self.assertAlmostEqual(
            ModelLaneDetector._slope_factor(points), np.sqrt(2.0), places=5
        )

    def test_virtual_boundary_fit_clamps_outside_observed_span(self):
        fit = ModelLaneDetector._fit_line([(0.0, 10.0), (100.0, 110.0)])
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit(50.0), 60.0, places=5)
        self.assertAlmostEqual(fit(-50.0), 10.0, places=5)
        self.assertAlmostEqual(fit(150.0), 110.0, places=5)


class CombinedDetectorTests(unittest.TestCase):
    def test_disabling_combined_detector_clears_pending_and_latest_results(self):
        detector = AsyncCombinedDetector.__new__(AsyncCombinedDetector)
        detector._condition = threading.Condition()
        detector._enabled = True
        detector._pending = ("frame", 1)
        detector._latest = CombinedDetection(source_sequence=1)

        detector.set_enabled(False)

        self.assertFalse(detector.enabled)
        self.assertIsNone(detector._pending)
        self.assertEqual(detector.latest().source_sequence, -1)

        detector.set_enabled(True)
        self.assertTrue(detector.enabled)

    def test_one_full_frame_inference_splits_signal_and_car(self):
        class Value:
            def __init__(self, values):
                self.values = np.array(values)

            def __getitem__(self, item):
                return self.values[item]

        class Box:
            def __init__(self, class_id, confidence, bbox):
                self.cls = Value([class_id])
                self.conf = Value([confidence])
                self.xyxy = Value([bbox])

        class Prediction:
            boxes = [
                Box(0, 0.82, [10.0, 20.0, 110.0, 120.0]),
                Box(1, 0.71, [140.0, 20.0, 240.0, 120.0]),
                Box(2, 0.91, [300.0, 200.0, 600.0, 500.0]),
            ]

        class Model:
            names = {0: "red", 1: "green", 2: "stroller"}

            def __init__(self):
                self.source_shape = None

            def predict(self, source, **_kwargs):
                self.source_shape = source.shape
                return [Prediction()]

        detector = AsyncCombinedDetector.__new__(AsyncCombinedDetector)
        detector.cfg = {
            "combined_image_size": 640,
            "traffic_confidence": 0.25,
            "car_confidence": 0.6,
            "model_device": "cpu",
            "car_class_names": ["stroller"],
            "car_min_area_ratio": 0.001,
        }
        detector.model = Model()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector._detect(frame, 3, 10.0)
        self.assertEqual(detector.model.source_shape, (720, 1280, 3))
        self.assertEqual(result.traffic.signal, "red")
        self.assertEqual(result.car.label, "stroller")
        self.assertEqual(result.car.bbox, (300, 200, 600, 500))
        self.assertEqual(result.traffic.source_sequence, result.car.source_sequence)
        self.assertEqual(result.traffic.frame_time, result.car.frame_time)

    def test_car_roi_filters_only_stroller_by_bbox_bottom_center(self):
        class Value:
            def __init__(self, values):
                self.values = np.array(values)

            def __getitem__(self, item):
                return self.values[item]

        class Box:
            def __init__(self, class_id, confidence, bbox):
                self.cls = Value([class_id])
                self.conf = Value([confidence])
                self.xyxy = Value([bbox])

        class Prediction:
            boxes = [
                Box(0, 0.82, [10.0, 10.0, 90.0, 80.0]),
                Box(2, 0.95, [20.0, 180.0, 160.0, 420.0]),
                Box(2, 0.80, [400.0, 180.0, 600.0, 420.0]),
            ]

        class Model:
            names = {0: "red", 1: "green", 2: "stroller"}

            def predict(self, source, **_kwargs):
                return [Prediction()]

        detector = AsyncCombinedDetector.__new__(AsyncCombinedDetector)
        detector.cfg = {
            "combined_image_size": 640,
            "traffic_confidence": 0.25,
            "car_confidence": 0.6,
            "model_device": "cpu",
            "car_class_names": ["stroller"],
            "car_min_area_ratio": 0.001,
            "car_roi_left": 0.25,
            "car_roi_top": 0.10,
            "car_roi_right": 0.75,
            "car_roi_bottom": 1.0,
        }
        detector.model = Model()

        result = detector._detect(np.zeros((500, 1000, 3), dtype=np.uint8), 9, 20.0)

        self.assertEqual(result.traffic.signal, "red")
        self.assertTrue(result.car.detected)
        self.assertEqual(result.car.confidence, 0.80)
        self.assertEqual(result.car.bbox, (400, 180, 600, 420))
        self.assertEqual(len(result.car_candidates), 2)
        self.assertFalse(result.car_candidates[0][4])
        self.assertTrue(result.car_candidates[1][4])


class RecoverySpeedRampTests(unittest.TestCase):
    def test_ramps_70_to_255_over_three_seconds_and_pauses(self):
        ramp = RecoverySpeedRamp(
            {
                "recovery_ramp_start_pwm": 70,
                "recovery_ramp_duration_s": 3.0,
            }
        )
        ramp.start(100.0)
        self.assertEqual(ramp.speed(100.0, 255), 70)
        halfway = ramp.speed(101.5, 255)
        self.assertTrue(161 <= halfway <= 163)
        self.assertEqual(ramp.speed(102.5, 255, motion_enabled=False), halfway)
        self.assertEqual(ramp.speed(104.0, 255), 255)
        self.assertFalse(ramp.active)


class TrafficMissionTests(unittest.TestCase):
    def test_stale_traffic_result_clears_partial_confirmation(self):
        stabilizer = DetectionStabilizer(
            {
                "lane_confirm_count": 2,
                "stop_signal_confirm_count": 2,
                "go_signal_confirm_count": 2,
            }
        )
        stabilizer.update(True, "red")
        stabilizer.invalidate_traffic()
        stabilizer.update(True, "red")
        self.assertFalse(stabilizer.stop_signal_confirmed())

    def test_model_red_and_lane_stop_then_model_green_completes(self):
        stabilizer = DetectionStabilizer(
            {
                "lane_confirm_count": 2,
                "stop_signal_confirm_count": 2,
                "go_signal_confirm_count": 2,
            }
        )
        gate = TrafficMissionGate()

        stabilizer.update(True, "red")
        self.assertEqual(gate.update(True, False, False, False), (False, False))
        stabilizer.update(True, "red")
        self.assertEqual(
            gate.update(
                True,
                stabilizer.lane_confirmed(),
                stabilizer.stop_signal_confirmed(),
                stabilizer.go_signal_confirmed(),
            ),
            (True, False),
        )
        self.assertTrue(gate.holding)

        stabilizer.update(True, "unknown")
        self.assertEqual(
            gate.update(True, True, False, stabilizer.go_signal_confirmed()),
            (False, False),
        )
        self.assertTrue(gate.holding)

        stabilizer.update(True, "green")
        stabilizer.update(True, "green")
        self.assertEqual(
            gate.update(True, True, False, stabilizer.go_signal_confirmed()),
            (False, True),
        )
        self.assertFalse(gate.detector_enabled)
        self.assertEqual(gate.update(True, True, True, True), (False, False))

    def test_reset_rearms_completed_traffic_mission(self):
        gate = TrafficMissionGate()
        gate.state = gate.COMPLETE
        gate.reset()
        self.assertEqual(gate.state, gate.ACTIVE)
        self.assertTrue(gate.detector_enabled)


class ObstacleFusionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "us_front_ids": [0, 1],
            "sensor_timeout_s": 1.0,
            "sensor_min_valid_cm": 2,
            "sensor_max_valid_cm": 400,
            "sensor_required": 1,
            "ultrasonic_enabled": 1,
            "obs_trigger_cm": 100,
            "obs_trigger_hits": 2,
            "obstacle_slow_cm": 80,
            "obstacle_slow_pwm": 100,
            "change_pwm": 70,
            "car_model_required": 1,
            "car_result_stale_s": 0.8,
            "fusion_max_skew_s": 0.6,
            "direction_confirm_frames": 1,
            "change_t_pre": 0.0,
        }
        self.controller = ObstacleController(self.cfg)
        self.controller.update_lane_context(dash_side="left")

    def decide(self, now, distance, car_detected, sequence):
        return self.controller.decide(
            now,
            {0: distance, 1: distance + 2},
            {0: now, 1: now},
            motion_enabled=True,
            car_detected=car_detected,
            car_frame_time=now if car_detected else None,
            car_sequence=sequence if car_detected else -1,
        )

    def test_ultrasonic_only_never_starts_lane_change(self):
        for index in range(5):
            decision = self.decide(10.0 + index * 0.1, 40, False, index)
            self.assertFalse(decision.avoidance_active)
        self.assertEqual(self.controller.state, self.controller.KEEP)
        self.assertEqual(self.controller.front_hits, 0)

    def test_car_only_never_starts_lane_change(self):
        for index in range(5):
            decision = self.decide(20.0 + index * 0.1, 180, True, index)
            self.assertFalse(decision.avoidance_active)
        self.assertEqual(self.controller.state, self.controller.KEEP)
        self.assertEqual(self.controller.front_hits, 0)

    def test_missing_car_sample_breaks_consecutive_fusion_hits(self):
        self.decide(25.0, 40, True, 1)
        self.assertEqual(self.controller.front_hits, 1)
        self.decide(25.1, 40, False, 2)
        self.assertEqual(self.controller.front_hits, 0)

        next_pair = self.decide(25.2, 40, True, 3)
        self.assertFalse(next_pair.avoidance_active)
        self.assertEqual(self.controller.front_hits, 1)

    def test_car_and_close_ultrasonic_start_then_reset_blocks(self):
        first = self.decide(30.0, 40, True, 1)
        second = self.decide(30.1, 40, True, 2)
        self.assertFalse(first.avoidance_active)
        self.assertEqual(first.requested_speed, 70)
        self.assertTrue(second.avoidance_active)
        self.assertEqual(second.requested_speed, 70)
        self.assertEqual(self.controller.state, self.controller.CHANGING)

        self.controller.reset()
        blocked = self.controller.decide(
            30.2,
            {0: 40, 1: 42},
            {0: 30.2, 1: 30.2},
            motion_enabled=False,
            car_detected=True,
            car_frame_time=30.2,
            car_sequence=3,
        )
        self.assertFalse(blocked.avoidance_active)
        self.assertEqual(blocked.phase, "KEEP_PAUSED")
        self.assertEqual(self.controller.front_hits, 0)

    def test_avoidance_goes_to_inner_then_always_returns_to_outer(self):
        """회피 1회 = inner 로 나갔다가 반드시 outer 로 돌아오고 끝난다."""
        cfg = dict(self.cfg)
        cfg.update(
            {
                "inner_lane_side": "left",
                "change_duration_outer_to_inner_s": 0.8,
                "change_duration_inner_to_outer_s": 0.6,
                "counter_steer_duration_s": 0.2,
                "return_counter_steer_duration_s": 0.2,
                "inner_hold_s": 1.0,
                "fsm_max_step_s": 1.0,
                "obs_trigger_hits": 1,
            }
        )
        ctl = ObstacleController(cfg)

        def step(now, distance=40, car=True, seq=1):
            return ctl.decide(
                now,
                {0: distance, 1: distance + 2},
                {0: now, 1: now},
                motion_enabled=True,
                car_detected=car,
                car_frame_time=now if car else None,
                car_sequence=seq if car else -1,
            )

        # 융합 확인 즉시 inner 로 조향 (왼쪽 = 양수)
        first = step(0.0)
        self.assertEqual(ctl.state, ctl.CHANGING)
        self.assertGreater(first.requested_steering, 0)

        # 기동 시간이 지나면 반대조향으로 정렬
        step(0.9)
        self.assertEqual(ctl.state, ctl.COUNTER_STEER)

        # 정렬이 끝나면 inner 차선 유지 구간. 조향은 차선 추종에 넘긴다
        settled = step(1.2)
        self.assertEqual(ctl.state, ctl.INNER_HOLD)
        self.assertTrue(settled.avoidance_completed)
        self.assertFalse(settled.mission_completed)
        self.assertIsNone(settled.requested_steering)
        self.assertFalse(settled.avoidance_active)

        # 유지 시간이 지나면 장애물이 있든 없든 outer 로 복귀 시작
        step(2.3)
        self.assertEqual(ctl.state, ctl.RETURN_CHANGING)

        step(3.0)
        self.assertEqual(ctl.state, ctl.RETURN_COUNTER)

        done = step(3.3)
        self.assertEqual(ctl.state, ctl.DONE)
        self.assertTrue(done.avoidance_completed)
        self.assertTrue(done.mission_completed)

    def test_inner_hold_returns_even_while_obstacle_still_close(self):
        """복귀는 시간 기준이다. 초음파가 계속 가깝다고 inner 에 갇히면 안 된다."""
        cfg = dict(self.cfg)
        cfg.update({"inner_hold_s": 0.5, "fsm_max_step_s": 1.0})
        ctl = ObstacleController(cfg)
        ctl._start(ctl.INNER_HOLD)
        ctl.last_update_ts = 0.0
        ctl.decide(0.6, {0: 20, 1: 20}, {0: 0.6, 1: 0.6}, car_detected=True, car_frame_time=0.6)
        self.assertEqual(ctl.state, ctl.RETURN_CHANGING)

    def test_done_ignores_further_obstacles(self):
        """1회 회피 후에는 장애물+초음파가 다시 맞아도 회피하지 않는다."""
        ctl = ObstacleController(dict(self.cfg))
        ctl._start(ctl.DONE)
        for index in range(5):
            now = 50.0 + index * 0.1
            decision = ctl.decide(
                now,
                {0: 20, 1: 22},
                {0: now, 1: now},
                car_detected=True,
                car_frame_time=now,
                car_sequence=index,
            )
            self.assertFalse(decision.avoidance_active)
            self.assertFalse(decision.obstacle_detected)
        self.assertEqual(ctl.state, ctl.DONE)

    def test_reset_rearms_obstacle_mission_from_done(self):
        """정지 후 s(전체 리셋)를 누르면 회피 미션도 처음 상태로 돌아온다."""
        ctl = ObstacleController(dict(self.cfg))
        ctl._start(ctl.DONE)
        ctl.reset()
        self.assertEqual(ctl.state, ctl.KEEP)
        self.assertFalse(ctl.mission_finished())

    def test_inner_side_flips_every_steering_sign(self):
        """inner_lane_side 하나만 뒤집으면 회피/복귀 방향이 통째로 뒤집힌다."""
        left = ObstacleController(dict(self.cfg, inner_lane_side="left"))
        right = ObstacleController(dict(self.cfg, inner_lane_side="right"))
        self.assertGreater(left._steer_to_inner(), 0)
        self.assertLess(left._steer_to_outer(), 0)
        self.assertLess(right._steer_to_inner(), 0)
        self.assertGreater(right._steer_to_outer(), 0)

    def test_lane_tracking_runs_except_during_maneuvers(self):
        ctl = ObstacleController(dict(self.cfg))
        for state in (ctl.KEEP, ctl.INNER_HOLD, ctl.DONE):
            ctl._start(state)
            self.assertTrue(ctl.lane_recognition_enabled(), state)
        for state in (ctl.CHANGING, ctl.COUNTER_STEER, ctl.RETURN_CHANGING, ctl.RETURN_COUNTER):
            ctl._start(state)
            self.assertFalse(ctl.lane_recognition_enabled(), state)


if __name__ == "__main__":
    unittest.main()
