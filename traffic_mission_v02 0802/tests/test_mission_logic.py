import sys
import unittest
from pathlib import Path

import numpy as np

MISSION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MISSION_DIR))

from obstacle_controller import ObstacleController
from run_traffic_mission import (
    DetectionStabilizer,
    TrafficMissionGate,
    get_car_roi_pixels,
    normalize_car_roi,
)
from control import RecoverySpeedRamp
from car_detector import AsyncCarDetector


class CarRoiTests(unittest.TestCase):
    def test_car_roi_normalization_and_pixel_conversion(self):
        cfg = {
            "car_roi_left": 0.25,
            "car_roi_top": 0.20,
            "car_roi_right": 0.75,
            "car_roi_bottom": 0.80,
        }
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertEqual(get_car_roi_pixels(frame, cfg), (320, 144, 960, 576))

        cfg.update({"car_roi_left": 0.9, "car_roi_right": 0.2})
        left, _top, right, _bottom = normalize_car_roi(cfg)
        self.assertGreater(right, left)

    def test_car_detector_offsets_roi_box_to_full_frame(self):
        class Value:
            def __init__(self, values):
                self.values = np.array(values)

            def __getitem__(self, item):
                return self.values[item]

        class Box:
            cls = Value([0])
            conf = Value([0.9])
            xyxy = Value([[10.0, 20.0, 110.0, 120.0]])

        class Prediction:
            boxes = [Box()]

        class Model:
            names = {0: "obstacle-car"}

            def __init__(self):
                self.source_shape = None

            def predict(self, source, **_kwargs):
                self.source_shape = source.shape
                return [Prediction()]

        detector = AsyncCarDetector.__new__(AsyncCarDetector)
        detector.cfg = {
            "car_image_size": 512,
            "car_confidence": 0.6,
            "model_device": "cpu",
            "car_class_names": ["obstacle-car"],
            "car_min_area_ratio": 0.001,
        }
        detector.model = Model()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector._detect(frame, 3, 10.0, (100, 50, 500, 300))
        self.assertEqual(detector.model.source_shape, (250, 400, 3))
        self.assertEqual(result.bbox, (110, 70, 210, 170))
        self.assertEqual(result.roi, (100, 50, 500, 300))


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
        stabilizer.update(True, "green")
        stabilizer.invalidate_traffic()
        stabilizer.update(True, "green")
        self.assertFalse(stabilizer.stop_signal_confirmed())

    def test_model_green_and_lane_stop_then_model_red_completes(self):
        stabilizer = DetectionStabilizer(
            {
                "lane_confirm_count": 2,
                "stop_signal_confirm_count": 2,
                "go_signal_confirm_count": 2,
            }
        )
        gate = TrafficMissionGate()

        stabilizer.update(True, "green")
        self.assertEqual(gate.update(True, False, False, False), (False, False))
        stabilizer.update(True, "green")
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

        stabilizer.update(True, "red")
        stabilizer.update(True, "red")
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


if __name__ == "__main__":
    unittest.main()
