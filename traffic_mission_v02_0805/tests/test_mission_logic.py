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

    def test_direction_specific_change_durations_are_locked_and_used(self):
        cfg = dict(self.cfg)
        cfg.update(
            {
                "inner_lane_solid_side": "left",
                "change_duration_inner_to_outer_s": 0.4,
                "change_duration_outer_to_inner_s": 0.8,
                "fsm_max_step_s": 1.0,
            }
        )

        inner_to_outer = ObstacleController(cfg)
        inner_to_outer.update_lane_context(dash_side="right")
        self.assertTrue(inner_to_outer._lock_direction())
        self.assertEqual(inner_to_outer.locked_change_direction, "inner_to_outer")
        self.assertEqual(inner_to_outer.locked_change_duration_s, 0.4)
        inner_to_outer._start(inner_to_outer.CHANGING)
        inner_to_outer.last_update_ts = 0.0
        inner_to_outer.decide(0.5, {}, {})
        self.assertEqual(inner_to_outer.state, inner_to_outer.COUNTER_STEER)

        outer_to_inner = ObstacleController(cfg)
        outer_to_inner.update_lane_context(dash_side="left")
        self.assertTrue(outer_to_inner._lock_direction())
        self.assertEqual(outer_to_inner.locked_change_direction, "outer_to_inner")
        self.assertEqual(outer_to_inner.locked_change_duration_s, 0.8)
        outer_to_inner._start(outer_to_inner.CHANGING)
        outer_to_inner.last_update_ts = 0.0
        outer_to_inner.decide(0.5, {}, {})
        self.assertEqual(outer_to_inner.state, outer_to_inner.CHANGING)
        outer_to_inner.decide(0.9, {}, {})
        self.assertEqual(outer_to_inner.state, outer_to_inner.COUNTER_STEER)


if __name__ == "__main__":
    unittest.main()
