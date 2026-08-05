"""
lane_model.py -- lane following segmentation model based on the latest GAD drive flow.

Classes:
  0 = dash
  1 = crosswalk
  2 = solid

Steering still uses dash/solid. The crosswalk class is additionally projected into
bird's-eye view and exposed through `last_crosswalk_detected` so the traffic
runner can decide when to stop.
"""

import os
import time

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


CLS_DASH = 0
CLS_LANE = 1
CLS_SOLID = 2

COLOR_DASH = (0, 165, 255)
COLOR_LANE = (0, 255, 0)
COLOR_SOLID = (255, 80, 0)
COLOR_TARGET = (0, 255, 255)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model",
    "lane_best.pt",
)


class ModelLaneDetector:
    def __init__(self, cfg, model_path=DEFAULT_MODEL_PATH):
        self.cfg = cfg
        if YOLO is None:
            raise ImportError("ultralytics is required: pip install ultralytics")

        print(f"[ModelLaneDetector] loading: {model_path}")
        self.model = YOLO(model_path)
        self.device = str(cfg.get("model_device", "cpu"))
        self.imgsz = int(cfg.get("model_imgsz", 320))
        self.conf = float(cfg.get("model_conf", 0.4))

        self.prev_left_x = None
        self.prev_right_x = None
        self.prev_width = None
        self.miss_left = 0
        self.miss_right = 0
        self.reacquire_frames = int(cfg.get("reacquire_frames", 8))
        self.infer_ms = 0.0
        self._mask_cache = None
        self.last_crosswalk_detected = False
        self.last_heading = 0.0
        self.bottom_left_class = None
        self.bottom_right_class = None
        self.solid_side = None
        self.dash_side = None
        self.debug_lane_label = "UNKNOWN"
        self.solid_contact_detected = False
        self.solid_contact_pixels = 0
        self.reset_solid_orientation_state()
        self.reset_control_state(warmup_frames=0)

        self.model.predict(
            np.zeros((360, 640, 3), dtype=np.uint8),
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf,
            verbose=False,
        )

    def _infer(self, frame_bgr, reuse=False):
        if reuse and self._mask_cache is not None:
            return self._mask_cache

        height, width = frame_bgr.shape[:2]
        t0 = time.time()
        result = self.model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf,
            verbose=False,
        )[0]
        self.infer_ms = (time.time() - t0) * 1000.0

        dash = np.zeros((height, width), dtype=np.uint8)
        lane = np.zeros((height, width), dtype=np.uint8)
        solid = np.zeros((height, width), dtype=np.uint8)

        if result.masks is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for mask, class_id in zip(result.masks.data.cpu().numpy(), classes):
                resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_bin = resized > float(self.cfg.get("model_mask_threshold", 0.4))
                if class_id == CLS_DASH:
                    dash[mask_bin] = 255
                elif class_id == CLS_LANE:
                    lane[mask_bin] = 255
                elif class_id == CLS_SOLID:
                    solid[mask_bin] = 255

        close_kernel = max(0, int(self.cfg.get("mask_close_kernel", 3)))
        if close_kernel >= 2:
            if close_kernel % 2 == 0:
                close_kernel += 1
            kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
            dash = cv2.morphologyEx(dash, cv2.MORPH_CLOSE, kernel)
            solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel)

        self._mask_cache = (dash, lane, solid)
        return dash, lane, solid

    def reset_control_state(self, warmup_frames=0):
        self.prev_near_off = None
        self.exit_confirm_count = 0
        self.lookahead_mode = "FULL"
        self.heading_ema = 0.0
        self.control_warmup_remaining = max(0, int(warmup_frames))
        self.last_position_error = 0.0
        self.last_curve_term = 0.0
        self.last_heading = 0.0
        self.last_heading_term = 0.0
        self.last_effective_lookahead = 0.0
        self.last_both_real_bands = 0
        self.last_lane_width_measured = 0
        self.last_left_slope_factor = 1.0
        self.last_right_slope_factor = 1.0

    def reset_solid_orientation_state(self):
        self.solid_angle_ema = None
        self.solid_angle_misses = 0
        self.last_solid_filter_angle = None
        self.last_solid_kept_pixels = 0
        self.last_solid_rejected_pixels = 0

    @staticmethod
    def _line_angle_difference(angle_a, angle_b):
        difference = abs(float(angle_a) - float(angle_b)) % 180.0
        return min(difference, 180.0 - difference)

    def _filter_solid_orientation(self, solid_mask, update_state=True):
        if not bool(self.cfg.get("solid_orientation_filter_enabled", 1)):
            self.last_solid_filter_angle = self.solid_angle_ema
            self.last_solid_kept_pixels = int(np.count_nonzero(solid_mask))
            self.last_solid_rejected_pixels = 0
            return solid_mask, np.zeros_like(solid_mask)

        binary = (solid_mask > 0).astype(np.uint8)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        filtered = np.zeros_like(solid_mask)
        min_pixels = max(1, int(self.cfg.get("solid_min_component_pixels", 20)))
        min_span = max(2, int(self.cfg.get("solid_min_vertical_span_px", 30)))
        max_angle = max(
            0.0,
            min(89.0, float(self.cfg.get("solid_max_angle_from_vertical_deg", 60.0))),
        )
        angle_tolerance = max(
            0.0, float(self.cfg.get("solid_prev_angle_tolerance_deg", 35.0))
        )
        corridor = max(1.0, float(self.cfg.get("solid_fit_corridor_px", 18)))
        accepted_angles = []

        for component_id in range(1, component_count):
            pixel_count = int(stats[component_id, cv2.CC_STAT_AREA])
            vertical_span = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            if pixel_count < min_pixels or vertical_span < min_span:
                continue

            component_left = int(stats[component_id, cv2.CC_STAT_LEFT])
            component_top = int(stats[component_id, cv2.CC_STAT_TOP])
            component_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            component_labels = labels[
                component_top:component_top + component_height,
                component_left:component_left + component_width,
            ]
            ys, xs = np.where(component_labels == component_id)
            ys = ys + component_top
            xs = xs + component_left
            unique_y = np.unique(ys)
            if unique_y.size < min_span:
                continue

            # One median x per row prevents a long horizontal branch from
            # outweighing the actual longitudinal solid boundary.
            row_x = np.array(
                [np.median(xs[ys == row]) for row in unique_y], dtype=np.float64
            )
            degree = 2 if unique_y.size >= 20 else 1
            coefficients = np.polyfit(unique_y.astype(np.float64), row_x, degree)
            reference_y = float(np.percentile(unique_y, 70))
            derivative = float(np.polyval(np.polyder(coefficients), reference_y))
            angle = float(np.degrees(np.arctan(derivative)))
            if abs(angle) > max_angle:
                continue
            if (
                self.solid_angle_ema is not None
                and self._line_angle_difference(angle, self.solid_angle_ema)
                > angle_tolerance
            ):
                continue

            predicted_x = np.polyval(coefficients, ys.astype(np.float64))
            near_path = np.abs(xs.astype(np.float64) - predicted_x) <= corridor
            kept_y = ys[near_path]
            kept_x = xs[near_path]
            if kept_x.size < min_pixels:
                continue
            filtered[kept_y, kept_x] = 255
            accepted_angles.append((angle, int(kept_x.size), vertical_span))

        original_pixels = int(np.count_nonzero(solid_mask))
        kept_pixels = int(np.count_nonzero(filtered))
        rejected = cv2.bitwise_and(solid_mask, cv2.bitwise_not(filtered))
        self.last_solid_kept_pixels = kept_pixels
        self.last_solid_rejected_pixels = original_pixels - kept_pixels

        if update_state:
            if accepted_angles:
                best_angle = max(
                    accepted_angles, key=lambda item: (item[2], item[1])
                )[0]
                alpha = max(
                    0.0,
                    min(1.0, float(self.cfg.get("solid_angle_ema_alpha", 0.25))),
                )
                if self.solid_angle_ema is None:
                    self.solid_angle_ema = best_angle
                else:
                    self.solid_angle_ema = (
                        (1.0 - alpha) * self.solid_angle_ema + alpha * best_angle
                    )
                self.solid_angle_misses = 0
                self.last_solid_filter_angle = best_angle
            else:
                self.solid_angle_misses += 1
                if self.solid_angle_misses >= int(
                    self.cfg.get("solid_angle_reacquire_frames", 5)
                ):
                    self.solid_angle_ema = None
                self.last_solid_filter_angle = None

        return filtered, rejected

    def reset_tracking(self, warmup_frames=None):
        self.prev_left_x = None
        self.prev_right_x = None
        self.prev_width = None
        self.miss_left = 0
        self.miss_right = 0
        self.last_heading = None
        self.bottom_left_class = None
        self.bottom_right_class = None
        self.solid_side = None
        self.dash_side = None
        self.debug_lane_label = "UNKNOWN"
        self.solid_contact_detected = False
        self.solid_contact_pixels = 0
        self.reset_solid_orientation_state()
        if warmup_frames is None:
            warmup_frames = int(self.cfg.get("control_warmup_frames", 3))
        self.reset_control_state(warmup_frames=warmup_frames)

    @staticmethod
    def _nearest_blob(hist, ref_x, max_jump):
        xs = np.where(hist > 0)[0]
        if len(xs) == 0:
            return None

        splits = np.where(np.diff(xs) > 5)[0]
        blobs = np.split(xs, splits + 1)
        centers = np.array([blob.mean() for blob in blobs])

        if ref_x is None:
            return None, centers

        idx = int(np.argmin(np.abs(centers - ref_x)))
        if abs(centers[idx] - ref_x) > max_jump:
            return None, centers
        return float(centers[idx]), centers

    @staticmethod
    def _slope_factor(points):
        """Convert dx/dy into the fixed-width correction used by GAD/drive."""
        if len(points) < 2:
            return 1.0
        ys = np.array([point[0] for point in points], dtype=float)
        xs = np.array([point[1] for point in points], dtype=float)
        if np.ptp(ys) < 1.0:
            return 1.0
        slope = float(np.polyfit(ys, xs, 1)[0])
        return float(min(1.8, np.sqrt(1.0 + slope * slope)))

    @staticmethod
    def _fit_line(points):
        """Fit one boundary and clamp prediction to its observed vertical span."""
        if len(points) < 2:
            return None
        ys = np.array([point[0] for point in points], dtype=float)
        xs = np.array([point[1] for point in points], dtype=float)
        y_min = float(ys.min())
        y_max = float(ys.max())
        if y_max - y_min < 1.0:
            x_mean = float(xs.mean())
            return lambda _cy: x_mean
        slope, intercept = np.polyfit(ys, xs, 1)

        def predict(cy):
            bounded_y = min(y_max, max(y_min, float(cy)))
            return float(slope * bounded_y + intercept)

        return predict

    def process(self, frame_bgr, reuse_masks=False, tracking_enabled=True):
        height, width = frame_bgr.shape[:2]
        raw_dash, raw_lane, raw_solid = self._infer(frame_bgr, reuse=reuse_masks)

        cfg = self.cfg
        tl = [width * float(cfg.get("ipm_tl_x", 0.25)), height * float(cfg.get("ipm_tl_y", 0.40))]
        tr = [width * float(cfg.get("ipm_tr_x", 0.75)), height * float(cfg.get("ipm_tr_y", 0.40))]
        bl = [width * float(cfg.get("ipm_bl_x", -0.40)), height * float(cfg.get("ipm_bl_y", 0.98))]
        br = [width * float(cfg.get("ipm_br_x", 1.40)), height * float(cfg.get("ipm_br_y", 0.98))]

        tr[0] = max(tr[0], tl[0] + 10)
        br[0] = max(br[0], bl[0] + 10)
        bl[1] = max(bl[1], tl[1] + 10)
        br[1] = max(br[1], tr[1] + 10)
        src = np.float32([tl, tr, bl, br])

        mask_src = src.copy()
        margin_x = width * float(cfg.get("mask_roi_margin_x", 0.08))
        margin_y = height * float(cfg.get("mask_roi_margin_y", 0.03))
        mask_src[[0, 2], 0] -= margin_x
        mask_src[[1, 3], 0] += margin_x
        mask_src[[0, 1], 1] -= margin_y
        mask_src[[2, 3], 1] += margin_y

        roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [np.int32(mask_src[[0, 1, 3, 2]])], 255)
        dash_mask = cv2.bitwise_and(raw_dash, roi_mask)
        lane_mask = cv2.bitwise_and(raw_lane, roi_mask)
        solid_mask = cv2.bitwise_and(raw_solid, roi_mask)

        dbg = frame_bgr.copy()
        overlay = np.zeros_like(dbg)
        overlay[raw_dash == 255] = COLOR_DASH
        overlay[raw_lane == 255] = COLOR_LANE
        overlay[raw_solid == 255] = COLOR_SOLID
        dbg = cv2.addWeighted(dbg, 1.0, overlay, 0.5, 0)
        cv2.polylines(dbg, [np.int32(src[[0, 1, 3, 2]])], True, (255, 0, 255), 2)
        cv2.polylines(dbg, [np.int32(mask_src[[0, 1, 3, 2]])], True, (255, 255, 0), 1)
        cv2.putText(
            dbg,
            f"infer {self.infer_ms:.0f}ms raw D/L/S "
            f"{int(np.count_nonzero(raw_dash))}/"
            f"{int(np.count_nonzero(raw_lane))}/"
            f"{int(np.count_nonzero(raw_solid))} "
            f"roi S {int(np.count_nonzero(solid_mask))}",
            (8, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
        )

        dst = np.float32([
            [width * 0.2, 0],
            [width * 0.8, 0],
            [width * 0.2, height],
            [width * 0.8, height],
        ])
        matrix = cv2.getPerspectiveTransform(src, dst)

        w_dash = cv2.warpPerspective(dash_mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
        w_lane = cv2.warpPerspective(lane_mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
        w_solid_raw = cv2.warpPerspective(solid_mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
        w_solid, w_solid_rejected = self._filter_solid_orientation(
            w_solid_raw, update_state=tracking_enabled
        )
        w_any = cv2.bitwise_or(w_dash, w_solid)
        w_roi = cv2.warpPerspective(roi_mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
        self.last_crosswalk_detected = bool(np.any(w_lane))

        contact_top_ratio = max(
            0.0, min(0.95, float(cfg.get("solid_contact_roi_top", 0.62)))
        )
        contact_top = int(height * contact_top_ratio)
        contact_center = width // 2 + int(cfg.get("center_offset", 0))
        contact_half_width = max(
            2, int(cfg.get("solid_contact_corridor_px", 35))
        )
        contact_left = max(0, contact_center - contact_half_width)
        contact_right = min(width, contact_center + contact_half_width + 1)
        contact_mask = w_solid[contact_top:height, contact_left:contact_right]
        self.solid_contact_pixels = int(np.count_nonzero(contact_mask))
        self.solid_contact_detected = self.solid_contact_pixels >= max(
            1, int(cfg.get("solid_contact_min_pixels", 120))
        )

        warped_dbg = cv2.warpPerspective(frame_bgr, matrix, (width, height))
        warped_dbg[w_roi == 0] = (warped_dbg[w_roi == 0] * 0.25).astype(np.uint8)
        warped_overlay = np.zeros_like(warped_dbg)
        warped_overlay[w_dash == 255] = COLOR_DASH
        warped_overlay[w_lane == 255] = COLOR_LANE
        warped_overlay[w_solid_rejected == 255] = (0, 0, 255)
        warped_overlay[w_solid == 255] = COLOR_SOLID
        warped_dbg = cv2.addWeighted(warped_dbg, 0.7, warped_overlay, 0.5, 0)
        contact_color = (0, 0, 255) if self.solid_contact_detected else (120, 120, 120)
        cv2.rectangle(
            warped_dbg,
            (contact_left, contact_top),
            (max(contact_left, contact_right - 1), height - 1),
            contact_color,
            2,
        )
        cv2.putText(
            warped_dbg,
            f"solid contact {self.solid_contact_pixels}",
            (max(5, contact_left - 20), max(20, contact_top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            contact_color,
            1,
        )

        if not tracking_enabled:
            cv2.putText(
                dbg,
                "LANE CONTROL PAUSED",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                warped_dbg,
                "AVOIDANCE - TRACKING PAUSED",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return None, 0, dbg, warped_dbg

        n_bands = max(2, int(cfg.get("n_bands", 5)))
        band_h = max(1, height // n_bands)
        center_offset = int(cfg.get("center_offset", 0))
        center_x = width // 2 + center_offset
        max_jump = int(width * 0.15)

        left_x = self.prev_left_x
        right_x = self.prev_right_x
        lane_w_fixed = int(cfg.get("bev_lane_width", width * 0.6))

        bands = []
        found = 0
        bottom_left = None
        bottom_right = None
        bottom_width = None
        left_real = False
        right_real = False
        left_class_votes = []
        right_class_votes = []

        for band in range(n_bands - 1, -1, -1):
            by0 = band * band_h
            by1 = min(height, (band + 1) * band_h)
            cy = (by0 + by1) // 2

            h_any = np.sum(w_any[by0:by1, :], axis=0)
            h_dash = np.sum(w_dash[by0:by1, :], axis=0)
            h_solid = np.sum(w_solid[by0:by1, :], axis=0)

            def _cls(x):
                a = int(max(0, x - 12))
                z = int(min(width, x + 12))
                solid_score = float(h_solid[a:z].sum())
                dash_score = float(h_dash[a:z].sum())
                min_score = 255.0 * float(cfg.get("boundary_class_min_pixels", 2))
                ratio = float(cfg.get("boundary_class_ratio", 1.10))
                if solid_score < min_score and dash_score < min_score:
                    return None
                if solid_score > dash_score * ratio:
                    return "solid"
                if dash_score > solid_score * ratio:
                    return "dash"
                return None

            cx_r = None
            res = self._nearest_blob(h_any, right_x, max_jump)
            if res is not None:
                cand, centers = res
                if cand is not None:
                    cx_r = cand
                elif right_x is None:
                    right_candidates = centers[centers > center_x]
                    if len(right_candidates) > 0:
                        cx_r = float(right_candidates.min())

            cx_l = None
            res = self._nearest_blob(h_any, left_x, max_jump)
            if res is not None:
                cand, centers = res
                if cand is not None:
                    cx_l = cand
                elif left_x is None:
                    left_candidates = centers[centers < center_x]
                    if len(left_candidates) > 0:
                        cx_l = float(left_candidates.max())

            if cx_l is not None and cx_r is not None and cx_l >= cx_r:
                cx_l = None
                cx_r = None

            cls_l = _cls(cx_l) if cx_l is not None else None
            cls_r = _cls(cx_r) if cx_r is not None else None
            vote_weight = band + 1
            if cls_l is not None:
                left_class_votes.append((cls_l, vote_weight))
            if cls_r is not None:
                right_class_votes.append((cls_r, vote_weight))

            if cx_l is not None and cx_r is not None:
                left_x, right_x = cx_l, cx_r
                width_now = cx_r - cx_l
                if width * 0.2 < width_now < width * 0.8:
                    if bottom_width is None:
                        bottom_width = width_now
                found += 1
            elif cx_r is not None:
                right_x = cx_r
                left_x = cx_r - lane_w_fixed
                found += 1
            elif cx_l is not None:
                left_x = cx_l
                right_x = cx_l + lane_w_fixed
                found += 1
            else:
                if left_x is None or right_x is None:
                    continue

            if cx_l is not None:
                left_real = True
            if cx_r is not None:
                right_real = True
            if bottom_left is None:
                bottom_left, bottom_right = left_x, right_x

            bands.append((cy, cx_l, cx_r, left_x, right_x, cls_l, cls_r))

        if found == 0:
            self.prev_left_x = None
            self.prev_right_x = None
            self.miss_left = 0
            self.miss_right = 0
            self.last_heading = None
            self.bottom_left_class = None
            self.bottom_right_class = None
            self.solid_side = None
            self.dash_side = None
            self.debug_lane_label = "UNKNOWN"
            self.reset_solid_orientation_state()
            self.reset_control_state(
                warmup_frames=int(cfg.get("control_warmup_frames", 3))
            )
            cv2.putText(
                warped_dbg,
                "NO LANE",
                (width // 2 - 70, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            return None, 0, dbg, warped_dbg

        left_points = [
            (cy, cx_l) for cy, cx_l, _cx_r, _lx, _rx, _cl, _cr in bands
            if cx_l is not None
        ]
        right_points = [
            (cy, cx_r) for cy, _cx_l, cx_r, _lx, _rx, _cl, _cr in bands
            if cx_r is not None
        ]
        left_factor = self._slope_factor(left_points)
        right_factor = self._slope_factor(right_points)
        left_fit = self._fit_line(left_points)
        right_fit = self._fit_line(right_points)
        half_lane_width = lane_w_fixed / 2.0
        pts = []
        both_real_pts = []

        def _class_color(class_name):
            if class_name == "solid":
                return COLOR_SOLID
            if class_name == "dash":
                return COLOR_DASH
            return (160, 160, 160)

        for cy, cx_l, cx_r, phantom_l, phantom_r, cls_l, cls_r in bands:
            predicted_l = cx_l if cx_l is not None else (
                left_fit(cy) if left_fit is not None else None
            )
            predicted_r = cx_r if cx_r is not None else (
                right_fit(cy) if right_fit is not None else None
            )
            if (
                predicted_l is not None
                and predicted_r is not None
                and predicted_l < predicted_r
            ):
                target = (predicted_l + predicted_r) / 2.0
                if cx_l is not None and cx_r is not None:
                    confidence = 1.0
                    both_real_pts.append((cy, target))
                elif cx_l is not None or cx_r is not None:
                    confidence = 0.85
                else:
                    confidence = 0.7
            elif cx_r is not None:
                target = cx_r - half_lane_width * right_factor
                confidence = 0.6
            elif cx_l is not None:
                target = cx_l + half_lane_width * left_factor
                confidence = 0.6
            else:
                target = (phantom_l + phantom_r) / 2.0
                confidence = 0.3

            target = max(0.0, min(float(width - 1), float(target)))
            pts.append((cy, target, confidence))
            if cx_l is not None:
                cv2.circle(
                    warped_dbg, (int(cx_l), cy), 5, _class_color(cls_l), -1
                )
            elif predicted_l is not None:
                cv2.circle(
                    warped_dbg, (int(predicted_l), cy), 3, (160, 160, 160), 1
                )
            if cx_r is not None:
                cv2.circle(
                    warped_dbg, (int(cx_r), cy), 5, _class_color(cls_r), -1
                )
            elif predicted_r is not None:
                cv2.circle(
                    warped_dbg, (int(predicted_r), cy), 3, (160, 160, 160), 1
                )
            cv2.circle(warped_dbg, (int(target), cy), 7, COLOR_TARGET, -1)

        self.last_lane_width_measured = int(bottom_width or 0)
        self.last_left_slope_factor = left_factor
        self.last_right_slope_factor = right_factor

        self.prev_left_x = bottom_left
        self.prev_right_x = bottom_right
        if bottom_width is not None:
            self.prev_width = bottom_width

        self.miss_left = 0 if left_real else self.miss_left + 1
        self.miss_right = 0 if right_real else self.miss_right + 1
        if self.miss_left >= self.reacquire_frames:
            self.prev_left_x = None
        if self.miss_right >= self.reacquire_frames:
            self.prev_right_x = None

        warmup_active = self.control_warmup_remaining > 0
        near_targets = [cx for cy, cx, _ in pts if cy >= height * 0.6]
        if not near_targets:
            near_targets = [cx for _, cx, _ in pts]
        current_near_offset = abs(
            sum(near_targets) / max(len(near_targets), 1) - center_x
        )

        exit_delta_px = float(cfg.get("lookahead_exit_delta_px", 1.0))
        exit_confirm_frames = max(
            1, int(cfg.get("lookahead_exit_confirm_frames", 3))
        )
        if warmup_active or self.prev_near_off is None:
            self.lookahead_mode = "FULL"
            self.exit_confirm_count = 0
        elif current_near_offset < self.prev_near_off - exit_delta_px:
            self.exit_confirm_count += 1
            if self.exit_confirm_count >= exit_confirm_frames:
                self.lookahead_mode = "EXIT"
        elif current_near_offset > self.prev_near_off + exit_delta_px:
            self.lookahead_mode = "FULL"
            self.exit_confirm_count = 0
        self.prev_near_off = current_near_offset

        lookahead_gain = float(cfg.get("lookahead_gain", 4.0))
        if warmup_active:
            effective_lookahead = 0.0
        elif self.lookahead_mode == "EXIT":
            effective_lookahead = lookahead_gain * float(
                cfg.get("lookahead_exit_scale", 0.4)
            )
        else:
            effective_lookahead = lookahead_gain

        x_sum = 0.0
        w_sum = 0.0
        for cy, cx, conf in pts:
            weight = (
                conf
                if conf < 1.0
                else 1.0 + ((height - cy) / height) * effective_lookahead
            )
            x_sum += cx * weight
            w_sum += weight
        line_x = x_sum / max(w_sum, 1e-6)

        ys = [cy for cy, _, _ in pts]
        mid_y = (min(ys) + max(ys)) / 2.0
        near = [(cx, conf) for cy, cx, conf in pts if cy >= mid_y]
        far = [(cx, conf) for cy, cx, conf in pts if cy < mid_y]
        curve = 0.0
        if near and far:
            near_x = sum(cx * conf for cx, conf in near) / sum(conf for _, conf in near)
            far_x = sum(cx * conf for cx, conf in far) / sum(conf for _, conf in far)
            curve = (far_x - near_x) / max(width / 2.0, 1.0)
            cv2.line(
                warped_dbg,
                (int(near_x), height - 20),
                (int(far_x), 20),
                (255, 255, 0),
                2,
            )

        heading_used = 0.0
        heading_min_bands = max(2, int(cfg.get("heading_min_bands", 2)))
        if not warmup_active and len(pts) >= heading_min_bands:
            heading_ys = np.array([cy for cy, _cx, _conf in pts], dtype=float)
            heading_xs = np.array([cx for _cy, cx, _conf in pts], dtype=float)
            heading_weights = np.array(
                [conf for _cy, _cx, conf in pts], dtype=float
            )
            heading_coef = np.polyfit(
                heading_ys, heading_xs, 1, w=heading_weights
            )
            heading_slope = float(heading_coef[0])
            heading_raw = -heading_slope
            heading_alpha = max(
                0.0, min(1.0, float(cfg.get("heading_ema_alpha", 1.0)))
            )
            self.heading_ema = (
                (1.0 - heading_alpha) * self.heading_ema
                + heading_alpha * heading_raw
            )
            heading_clamp = abs(float(cfg.get("heading_clamp", 1.0)))
            heading_used = max(
                -heading_clamp, min(heading_clamp, self.heading_ema)
            )
            heading_y0 = height - 10
            heading_y1 = max(0, height - 130)
            heading_x0 = float(np.polyval(heading_coef, heading_y0))
            heading_x1 = heading_x0 + heading_slope * (heading_y1 - heading_y0)
            cv2.line(
                warped_dbg,
                (int(heading_x0), heading_y0),
                (int(heading_x1), heading_y1),
                (255, 0, 255),
                2,
            )
        else:
            self.heading_ema *= 0.5

        position_error = (line_x - center_x) / max(width / 2.0, 1.0)
        curve_term = (
            0.0
            if warmup_active
            else curve * float(cfg.get("curve_gain", 0.8))
        )
        heading_term = (
            0.0
            if warmup_active
            else heading_used * float(cfg.get("heading_gain", 0.2))
        )
        err = position_error + curve_term + heading_term
        err = max(-1.0, min(1.0, err))
        self.last_position_error = position_error
        self.last_curve_term = curve_term
        self.last_heading = heading_used
        self.last_heading_term = heading_term
        self.last_effective_lookahead = effective_lookahead
        self.last_both_real_bands = len(both_real_pts)
        if warmup_active:
            self.control_warmup_remaining -= 1

        def _vote_class(votes):
            scores = {"dash": 0, "solid": 0}
            for class_name, weight in votes:
                scores[class_name] += weight
            if scores["solid"] == scores["dash"]:
                return None
            return "solid" if scores["solid"] > scores["dash"] else "dash"

        bot_cls_L = _vote_class(left_class_votes)
        bot_cls_R = _vote_class(right_class_votes)
        raw = None
        self.bottom_left_class = bot_cls_L
        self.bottom_right_class = bot_cls_R
        if bot_cls_L == "solid" and bot_cls_R != "solid":
            self.solid_side = "left"
        elif bot_cls_R == "solid" and bot_cls_L != "solid":
            self.solid_side = "right"
        else:
            self.solid_side = None
        if bot_cls_L == "dash" and bot_cls_R != "dash":
            self.dash_side = "left"
        elif bot_cls_R == "dash" and bot_cls_L != "dash":
            self.dash_side = "right"
        elif self.solid_side == "left":
            self.dash_side = "right"
        elif self.solid_side == "right":
            self.dash_side = "left"
        else:
            self.dash_side = None
        inner_lane_solid_side = str(cfg.get("inner_lane_solid_side", "left")).lower()
        if inner_lane_solid_side == "right":
            if bot_cls_R == "solid":
                raw = "inner"
            elif bot_cls_L == "solid":
                raw = "outer"
            elif bot_cls_L == "dash" and bot_cls_R == "dash":
                raw = "inner"
        else:
            if bot_cls_L == "solid":
                raw = "inner"
            elif bot_cls_R == "solid":
                raw = "outer"
            elif bot_cls_L == "dash" and bot_cls_R == "dash":
                raw = "inner"

        self.debug_lane_label = raw.upper() if raw is not None else "UNKNOWN"

        cv2.line(warped_dbg, (center_x, 0), (center_x, height - 1), (255, 255, 255), 1)
        cv2.line(warped_dbg, (int(line_x), 0), (int(line_x), height - 1), COLOR_TARGET, 2)
        cv2.putText(
            warped_dbg,
            (
                f"err {err:+.3f} P {position_error:+.3f} "
                f"C {curve_term:+.3f} H {heading_term:+.3f}"
            ),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            warped_dbg,
            (
                f"LA={self.lookahead_mode}:{effective_lookahead:.1f} "
                f"both={len(both_real_pts)} warm={self.control_warmup_remaining} "
                f"L={bot_cls_L} R={bot_cls_R}"
            ),
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 0),
            2,
        )
        angle_text = (
            "--"
            if self.solid_angle_ema is None
            else f"{self.solid_angle_ema:+.1f}deg"
        )
        cv2.putText(
            warped_dbg,
            (
                f"solid filter keep={self.last_solid_kept_pixels} "
                f"reject={self.last_solid_rejected_pixels} angle={angle_text}"
            ),
            (10, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            warped_dbg,
            (
                f"laneW meas={self.last_lane_width_measured} fixed={lane_w_fixed} "
                f"cos L{left_factor:.2f} R{right_factor:.2f}"
            ),
            (10, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 255, 120),
            2,
        )
        lane_color = (0, 255, 255) if self.debug_lane_label == "UNKNOWN" else (0, 255, 0) if self.debug_lane_label == "INNER" else (0, 140, 255)
        cv2.putText(
            dbg,
            self.debug_lane_label,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            lane_color,
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            warped_dbg,
            self.debug_lane_label,
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            lane_color,
            4,
            cv2.LINE_AA,
        )

        return err, found, dbg, warped_dbg
