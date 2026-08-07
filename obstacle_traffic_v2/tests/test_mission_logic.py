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


class ObstacleAvoidV2Tests(unittest.TestCase):
    """장애물 회피 v2: normal <-> changing, 3중 검증, 쿨타임"""

    def setUp(self):
        self.cfg = {
            "us_front_ids": [0, 1],
            "sensor_timeout_s": 1.0,
            "sensor_min_valid_cm": 2,
            "sensor_max_valid_cm": 400,
            "sensor_required": 1,
            "ultrasonic_enabled": 1,
            "obs_trigger_cm": 140,
            "avoid_confirm_frames": 3,
            "avoid_max_steer": 0.4,
            "lane_change_seconds": 1.0,
            "lane_change_steer": 1.0,
            "avoid_cooldown_seconds": 2.0,
            "obstacle_slow_pwm": 10,   # 근접 크리프용 (기동에 쓰면 스톨)
            "slow_pwm": 90,            # 회피 기동 속도 (원본과 같은 키)
            "drive_pwm": 220,
            "car_model_required": 1,
            "car_result_stale_s": 0.8,
            "us_window_s": 0.3,
            "us_timeout_cm": 250,
            "start_lane_mode": 2,
            "lane1_side": "left",
        }
        self.ctl = ObstacleController(self.cfg)

    def step(self, now, distance=100, car=True, steer=0.0):
        return self.ctl.decide(
            now,
            {0: distance, 1: distance + 2},
            {0: now, 1: now},
            motion_enabled=True,
            car_detected=car,
            car_frame_time=now if car else None,
            current_steer=steer,
        )

    def run_until_change(self, t0=10.0, **kw):
        for i in range(6):
            d = self.step(t0 + i * 0.1, **kw)
            if d.avoidance_active:
                return t0 + i * 0.1, d
        return None, d

    # ── 3중 검증 ────────────────────────────────────────
    def test_ultrasonic_only_never_triggers(self):
        for i in range(6):
            d = self.step(10.0 + i * 0.1, distance=50, car=False)
            self.assertFalse(d.avoidance_active)
        self.assertEqual(self.ctl.avoid_state, self.ctl.NORMAL)

    def test_car_only_never_triggers(self):
        for i in range(6):
            d = self.step(20.0 + i * 0.1, distance=300, car=True)
            self.assertFalse(d.avoidance_active)
        self.assertEqual(self.ctl.front_hit_count, 0)

    def test_needs_consecutive_frames(self):
        self.step(30.0, distance=50)
        self.step(30.1, distance=50)
        self.assertEqual(self.ctl.front_hit_count, 2)
        self.assertEqual(self.ctl.avoid_state, self.ctl.NORMAL)
        d = self.step(30.2, distance=50)
        self.assertTrue(d.avoidance_active)

    def test_one_far_reading_no_longer_resets_the_count(self):
        """단발 타임아웃은 무시한다 (us_window_s 시간창 최솟값).

        예전에는 멀리 한 번만 읽혀도 카운트가 0 이 됐는데, 실제 센서가 에코를
        한 번 걸러 놓치는 바람에 회피가 영영 발동하지 않았다.
        """
        self.step(40.0, distance=50)
        self.step(40.05, distance=50)
        self.step(40.10, distance=300)         # 단발 -> 창에 50 이 남아 있다
        self.assertTrue(self.ctl.ultrasonic_close)
        # 창에 50 이 남아 있으니 3회째에 그대로 회피가 발동한다
        self.assertEqual(self.ctl.avoid_state, self.ctl.CHANGING)

    def test_sustained_far_readings_do_reset(self):
        """창(0.3s)을 넘겨 계속 멀면 그때는 해제된다"""
        self.step(45.0, distance=50)
        for i in range(6):
            self.step(45.4 + i * 0.1, distance=300)
        self.assertFalse(self.ctl.ultrasonic_close)
        self.assertEqual(self.ctl.front_hit_count, 0)

    def test_curve_suppresses_avoidance(self):
        """커브에서 정면 벽을 장애물로 오인하지 않아야 한다"""
        for i in range(6):
            d = self.step(50.0 + i * 0.1, distance=50, steer=0.9)
            self.assertFalse(d.avoidance_active)
        self.assertIn("curve", self.ctl.block_reason)
        # 직선으로 돌아오면 다시 쌓여서 발동한다
        _, d = self.run_until_change(t0=51.0, distance=50, steer=0.0)
        self.assertTrue(d.avoidance_active)

    def test_signal_hold_blocks_avoidance(self):
        for i in range(6):
            d = self.ctl.decide(60.0 + i * 0.1, {0: 50, 1: 52},
                                {0: 60.0 + i * 0.1, 1: 60.0 + i * 0.1},
                                car_detected=True, car_frame_time=60.0 + i * 0.1,
                                current_steer=0.0, allow_avoidance=False)
            self.assertFalse(d.avoidance_active)
        self.assertEqual(self.ctl.block_reason, "blocked-by-mission")

    # ── 상태 전이 ───────────────────────────────────────
    def test_full_cycle_switches_lane_mode_and_sets_cooldown(self):
        t0, d = self.run_until_change(distance=50)
        self.assertIsNotNone(t0)
        self.assertEqual(self.ctl.avoid_state, self.ctl.CHANGING)
        self.assertEqual(self.ctl.lane_mode, 2)          # 아직 안 바뀜
        self.assertEqual(self.ctl.avoid_target_mode, 1)
        self.assertGreater(d.requested_steering, 0)      # 2->1 은 왼쪽(양수)
        self.assertEqual(d.requested_speed, 90)   # slow_pwm (크리프 10 아님)

        mid = self.step(t0 + 0.5, distance=50)
        self.assertTrue(mid.avoidance_active)

        done = self.step(t0 + 1.1, distance=50)
        self.assertFalse(done.avoidance_active)
        self.assertTrue(done.avoidance_completed)
        self.assertEqual(self.ctl.lane_mode, 1)          # 여기서 바뀐다
        self.assertTrue(self.ctl.in_cooldown(t0 + 1.1))

    def test_cooldown_blocks_second_avoidance(self):
        t0, _ = self.run_until_change(distance=50)
        self.step(t0 + 1.1, distance=50)                 # 완료
        for i in range(8):
            d = self.step(t0 + 1.2 + i * 0.1, distance=50)
            self.assertFalse(d.avoidance_active)
        self.assertIn("cooldown", self.ctl.block_reason)
        # 쿨타임이 지나면 다시 가능하다
        _, d = self.run_until_change(t0=t0 + 3.5, distance=50)
        self.assertTrue(d.avoidance_active)

    def test_direction_flips_with_lane_mode(self):
        self.ctl.lane_mode = 1
        _, d = self.run_until_change(distance=50)
        self.assertEqual(self.ctl.avoid_target_mode, 2)
        self.assertLess(d.requested_steering, 0)         # 1->2 는 오른쪽(음수)

    def test_lane1_side_right_flips_every_sign(self):
        right = ObstacleController(dict(self.cfg, lane1_side="right"))
        self.assertLess(right._dir_toward(1), 0)
        self.assertGreater(right._dir_toward(2), 0)

    def test_lane_tracking_off_only_while_changing(self):
        self.assertTrue(self.ctl.lane_recognition_enabled())
        self.run_until_change(distance=50)
        self.assertFalse(self.ctl.lane_recognition_enabled())

    def test_reset_clears_state_and_cooldown(self):
        t0, _ = self.run_until_change(distance=50)
        self.step(t0 + 1.1, distance=50)
        self.ctl.reset()
        self.assertEqual(self.ctl.avoid_state, self.ctl.NORMAL)
        self.assertEqual(self.ctl.lane_mode, 2)
        self.assertFalse(self.ctl.in_cooldown(t0 + 1.2))

    def test_pause_lets_the_maneuver_timer_keep_running(self):
        """원본과 같은 동작: 일시정지 중에도 벽시계가 흐른다"""
        t0, _ = self.run_until_change(distance=50)
        paused = self.ctl.decide(t0 + 0.3, {0: 50, 1: 52}, {0: t0, 1: t0},
                                 motion_enabled=False, car_detected=True,
                                 car_frame_time=t0, current_steer=0.0)
        self.assertEqual(paused.requested_speed, 0)
        self.assertIn("PAUSED", paused.phase)
        # 멈춘 사이에 lane_change_seconds 가 지나가면 재개 즉시 완료된다
        done = self.step(t0 + 1.5, distance=50)
        self.assertTrue(done.avoidance_completed)
        self.assertEqual(self.ctl.lane_mode, 1)

    def test_alternating_timeout_readings_still_trigger(self):
        """실주행 로그 재현: 56, 250, 59, 250, 56 ... 처럼 에코를 한 번 걸러
        놓쳐도 회피가 발동해야 한다.

        펌웨어는 에코 실패 시 250 을 반환한다. 이걸 정상값으로 받으면
        연속감지 카운트가 매번 리셋돼 5회를 영영 못 채운다 — 실제로 그래서
        차가 장애물로 그대로 돌진했다.
        """
        pattern = [56, 250, 59, 250, 56, 250, 58, 250, 57, 250]
        fired = False
        for i, dist in enumerate(pattern):
            d = self.step(80.0 + i * 0.05, distance=dist, steer=0.05)
            if d.avoidance_active:
                fired = True
                break
        self.assertTrue(fired, f"교대 타임아웃에서 회피 실패 (hits={self.ctl.front_hit_count})")

    def test_timeout_only_readings_do_not_trigger(self):
        """전부 250(측정 실패)이면 가깝다고 보면 안 된다"""
        for i in range(10):
            d = self.step(90.0 + i * 0.05, distance=250, steer=0.0)
            self.assertFalse(d.avoidance_active)

    def test_window_expires_so_a_passed_obstacle_stops_counting(self):
        """장애물을 지나쳐 멀어지면 창이 비워져 더는 안 잡혀야 한다"""
        self.step(95.0, distance=50)
        self.assertTrue(self.ctl.ultrasonic_close)
        # 창(0.3s)보다 오래 멀리 있으면 해제
        for i in range(6):
            self.step(95.5 + i * 0.1, distance=300)
        self.assertFalse(self.ctl.ultrasonic_close)

    def test_sensor_timeout_stops_and_clears_hits(self):
        self.step(70.0, distance=50)
        d = self.ctl.decide(71.5, {}, {}, car_detected=True, car_frame_time=71.5)
        self.assertFalse(d.sensor_ok)
        self.assertEqual(d.requested_speed, 0)
        self.assertEqual(self.ctl.front_hit_count, 0)


if __name__ == "__main__":
    unittest.main()
