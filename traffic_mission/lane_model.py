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
                mask_bin = resized > 0.5
                if class_id == CLS_DASH:
                    dash[mask_bin] = 255
                elif class_id == CLS_LANE:
                    lane[mask_bin] = 255
                elif class_id == CLS_SOLID:
                    solid[mask_bin] = 255

        self._mask_cache = (dash, lane, solid)
        return dash, lane, solid

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

    def process(self, frame_bgr, reuse_masks=False):
        height, width = frame_bgr.shape[:2]
        dash_mask, lane_mask, solid_mask = self._infer(frame_bgr, reuse=reuse_masks)

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

        roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [np.int32(src[[0, 1, 3, 2]])], 255)
        dash_mask = cv2.bitwise_and(dash_mask, roi_mask)
        lane_mask = cv2.bitwise_and(lane_mask, roi_mask)
        solid_mask = cv2.bitwise_and(solid_mask, roi_mask)

        dbg = frame_bgr.copy()
        overlay = np.zeros_like(dbg)
        overlay[dash_mask == 255] = COLOR_DASH
        overlay[lane_mask == 255] = COLOR_LANE
        overlay[solid_mask == 255] = COLOR_SOLID
        dbg = cv2.addWeighted(dbg, 1.0, overlay, 0.5, 0)
        cv2.polylines(dbg, [np.int32(src[[0, 1, 3, 2]])], True, (255, 0, 255), 2)
        cv2.putText(
            dbg,
            f"infer {self.infer_ms:.0f}ms dash/lane/solid "
            f"{int(np.count_nonzero(dash_mask))}/"
            f"{int(np.count_nonzero(lane_mask))}/"
            f"{int(np.count_nonzero(solid_mask))}",
            (8, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
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
        w_roi = cv2.warpPerspective(roi_mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
        self.last_crosswalk_detected = bool(np.any(w_lane))

        warped_dbg = cv2.warpPerspective(frame_bgr, matrix, (width, height))
        warped_dbg[w_roi == 0] = (warped_dbg[w_roi == 0] * 0.25).astype(np.uint8)
        warped_overlay = np.zeros_like(warped_dbg)
        warped_overlay[w_dash == 255] = COLOR_DASH
        warped_overlay[w_lane == 255] = COLOR_LANE
        warped_overlay[w_solid == 255] = COLOR_SOLID
        warped_dbg = cv2.addWeighted(warped_dbg, 0.7, warped_overlay, 0.5, 0)

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

        for band in range(n_bands - 1, -1, -1):
            by0 = band * band_h
            by1 = min(height, (band + 1) * band_h)
            cy = (by0 + by1) // 2

            h_dash = np.sum(w_dash[by0:by1, :], axis=0)
            h_solid = np.sum(w_solid[by0:by1, :], axis=0)

            cx_r = None
            res = self._nearest_blob(h_solid, right_x, max_jump)
            if res is not None:
                cand, centers = res
                if cand is not None:
                    cx_r = cand
                elif right_x is None:
                    right_candidates = centers[centers > center_x]
                    if len(right_candidates) > 0:
                        cx_r = float(right_candidates.min())

            cx_l = None
            res = self._nearest_blob(h_dash, left_x, max_jump)
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

            if cx_l is not None and cx_r is not None:
                left_x, right_x = cx_l, cx_r
                width_now = cx_r - cx_l
                if width * 0.2 < width_now < width * 0.8:
                    lane_w = width_now
                    if bottom_width is None:
                        bottom_width = width_now
                target = (cx_l + cx_r) / 2.0
                conf = 1.0
                found += 1
            elif cx_r is not None:
                right_x = cx_r
                left_x = cx_r - lane_w
                target = cx_r - lane_w / 2.0
                conf = 0.6
                found += 1
            elif cx_l is not None:
                left_x = cx_l
                right_x = cx_l + lane_w
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
                cv2.circle(warped_dbg, (int(cx_l), cy), 5, COLOR_DASH, -1)
            if cx_r is not None:
                cv2.circle(warped_dbg, (int(cx_r), cy), 5, COLOR_SOLID, -1)
            cv2.circle(warped_dbg, (int(target), cy), 5, COLOR_TARGET, -1)

        if found == 0:
            self.prev_left_x = None
            self.prev_right_x = None
            self.miss_left = 0
            self.miss_right = 0
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

        lookahead_gain = float(cfg.get("lookahead_gain", 4.0))
        x_sum = 0.0
        w_sum = 0.0
        for cy, cx, conf in pts:
            weight = conf if conf < 1.0 else 1.0 + ((height - cy) / height) * lookahead_gain
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

        heading = (line_x - center_x) / max(width / 2.0, 1.0)
        err = heading + curve * float(cfg.get("curve_gain", 0.8))
        err = max(-1.0, min(1.0, err))

        cv2.line(warped_dbg, (center_x, 0), (center_x, height - 1), (255, 255, 255), 1)
        cv2.line(warped_dbg, (int(line_x), 0), (int(line_x), height - 1), COLOR_TARGET, 2)
        cv2.putText(
            warped_dbg,
            f"err {err:+.3f} found {found} lane {self.last_crosswalk_detected}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        return err, found, dbg, warped_dbg
