"""
lane_model.py -- lane following segmentation model based on the latest GAD drive flow.

Classes:
  0 = dash
  1 = lane
  2 = solid

Steering still uses dash/solid. The lane class is additionally projected into
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
    "lane_seg_best.pt",
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
                resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
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
        w_solid = cv2.warpPerspective(solid_mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
        w_any = cv2.bitwise_or(w_dash, w_solid)
        w_roi = cv2.warpPerspective(roi_mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
        self.last_crosswalk_detected = bool(np.any(w_lane))

        warped_dbg = cv2.warpPerspective(frame_bgr, matrix, (width, height))
        warped_dbg[w_roi == 0] = (warped_dbg[w_roi == 0] * 0.25).astype(np.uint8)
        warped_overlay = np.zeros_like(warped_dbg)
        warped_overlay[w_dash == 255] = COLOR_DASH
        warped_overlay[w_lane == 255] = COLOR_LANE
        warped_overlay[w_solid == 255] = COLOR_SOLID
        warped_dbg = cv2.addWeighted(warped_dbg, 0.7, warped_overlay, 0.5, 0)

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
        lane_w = self.prev_width if self.prev_width else int(cfg.get("bev_lane_width", width * 0.6))

        pts = []
        found = 0
        bottom_left = None
        bottom_right = None
        bottom_width = None
        left_real = False
        right_real = False
        left_class_votes = []
        right_class_votes = []
        both_real_pts = []

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
                    lane_w = width_now
                    if bottom_width is None:
                        bottom_width = width_now
                target = (cx_l + cx_r) / 2.0
                both_real_pts.append((cy, target))
                conf = 1.0
                found += 1
            elif cx_r is not None:
                right_x = cx_r
                left_x = cx_r - int(cfg.get("bev_lane_width", width * 0.6))
                target = cx_r - lane_w / 2.0
                conf = 0.6
                found += 1
            elif cx_l is not None:
                left_x = cx_l
                right_x = cx_l + int(cfg.get("bev_lane_width", width * 0.6))
                target = cx_l + lane_w / 2.0
                conf = 0.6
                found += 1
            else:
                if left_x is None or right_x is None:
                    continue
                target = (left_x + right_x) / 2.0
                conf = 0.3

            if cx_l is not None:
                left_real = True
            if cx_r is not None:
                right_real = True
            if bottom_left is None:
                bottom_left, bottom_right = left_x, right_x

            pts.append((cy, target, conf))
            if cx_l is not None:
                left_color = (
                    COLOR_SOLID
                    if cls_l == "solid"
                    else COLOR_DASH if cls_l == "dash" else (160, 160, 160)
                )
                cv2.circle(warped_dbg, (int(cx_l), cy), 5, left_color, -1)
            if cx_r is not None:
                right_color = (
                    COLOR_SOLID
                    if cls_r == "solid"
                    else COLOR_DASH if cls_r == "dash" else (160, 160, 160)
                )
                cv2.circle(warped_dbg, (int(cx_r), cy), 5, right_color, -1)
            cv2.circle(warped_dbg, (int(target), cy), 5, COLOR_TARGET, -1)

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
            curve = (far_x - near_x) / max(width, 1)

        heading_used = 0.0
        heading_min_bands = max(2, int(cfg.get("heading_min_bands", 4)))
        if not warmup_active and len(both_real_pts) >= heading_min_bands:
            heading_slope = float(
                np.polyfit(
                    [cy for cy, _ in both_real_pts],
                    [cx for _, cx in both_real_pts],
                    1,
                )[0]
            )
            heading_raw = -heading_slope
            heading_alpha = max(
                0.0, min(1.0, float(cfg.get("heading_ema_alpha", 0.25)))
            )
            self.heading_ema = (
                (1.0 - heading_alpha) * self.heading_ema
                + heading_alpha * heading_raw
            )
            heading_clamp = abs(float(cfg.get("heading_clamp", 0.25)))
            heading_used = max(
                -heading_clamp, min(heading_clamp, self.heading_ema)
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
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            lane_color,
            4,
            cv2.LINE_AA,
        )

        return err, found, dbg, warped_dbg
