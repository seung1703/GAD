"""
lane_model.py -- YOLOv8n-seg 기반 차선 검출, BEV(IPM) 없이 원근 화면에서 직접 처리.

dl_model/lane_model.py(BEV 버전)과 검출·밴드추적·제어항 구조는 동일하되,
warpPerspective를 아예 안 쓰고 원본 카메라 좌표에서 바로 밴드 스캔한다.

장점: IPM 사다리꼴 캘리브레이션이 필요 없다 (카메라 각도가 바뀌어도 무조건 동작).
      단순 위/아래 y범위(persp_top_y~persp_bot_y)만 있으면 됨.

**알려진 한계 (이론적으로 못 피함)**: 원근 화면에서는 멀리 있는 점일수록 실제
위치와 무관하게 화면 중앙(소실점)으로 압축되어 보인다. 그래서 "가까운 밴드와
먼 밴드의 x차이"로 곡률/헤딩을 추정하는 계산이, 직선에서 단순 좌우 오프셋만
있어도 마치 커브가 있는 것처럼 오염된다 (BEV에서는 이 문제가 없음 — 그게
IPM을 쓰는 이유). curve_gain/heading_gain은 그대로 이식했지만 이 오염을
안고 가는 것이니, 실차에서 직선 흔들림이 커지면 이 두 게인부터 낮출 것.

인터페이스: process(frame) -> (err_norm, found, dbg, dbg)  (dl_model과 동일,
  BEV 뷰가 없으므로 두 번째 반환값도 원본 뷰와 같은 이미지)
"""

import os
import time

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

CLS_DASH, CLS_LANE, CLS_SOLID = 0, 1, 2

COLOR_DASH = (0, 165, 255)
COLOR_SOLID = (255, 80, 0)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model", "lane_seg_best.pt")


class ModelLaneDetector:
    def __init__(self, cfg, model_path=DEFAULT_MODEL_PATH):
        self.cfg = cfg
        if YOLO is None:
            raise ImportError("ultralytics 필요: pip install ultralytics")

        print(f"[ModelLaneDetector-perspective] 모델 로드: {model_path}")
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

        self.model.predict(np.zeros((360, 640, 3), dtype=np.uint8),
                           imgsz=self.imgsz, device=self.device,
                           conf=self.conf, verbose=False)

    def _infer(self, frame_bgr, reuse=False):
        if reuse and self._mask_cache is not None:
            return self._mask_cache
        H, W = frame_bgr.shape[:2]
        t0 = time.time()
        r = self.model.predict(frame_bgr, imgsz=self.imgsz,
                               device=self.device, conf=self.conf,
                               verbose=False)[0]
        self.infer_ms = (time.time() - t0) * 1000

        dash = np.zeros((H, W), dtype=np.uint8)
        solid = np.zeros((H, W), dtype=np.uint8)
        if r.masks is not None:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for m, c in zip(r.masks.data.cpu().numpy(), cls):
                if c == CLS_LANE:
                    continue
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                if c == CLS_DASH:
                    dash[m > 0.5] = 255
                else:
                    solid[m > 0.5] = 255
        self._mask_cache = (dash, solid)
        return dash, solid

    @staticmethod
    def _nearest_blob(hist, ref_x, max_jump):
        xs = np.where(hist > 0)[0]
        if len(xs) == 0:
            return None
        splits = np.where(np.diff(xs) > 5)[0]
        blobs = np.split(xs, splits + 1)
        centers = np.array([b.mean() for b in blobs])
        if ref_x is None:
            return None, centers
        i = int(np.argmin(np.abs(centers - ref_x)))
        if abs(centers[i] - ref_x) > max_jump:
            return None, centers
        return float(centers[i]), centers

    def process(self, frame_bgr, reuse_masks=False):
        H, W = frame_bgr.shape[:2]
        dash_mask, solid_mask = self._infer(frame_bgr, reuse=reuse_masks)

        dbg = frame_bgr.copy()
        ov = np.zeros_like(dbg)
        ov[dash_mask == 255] = COLOR_DASH
        ov[solid_mask == 255] = COLOR_SOLID
        dbg = cv2.addWeighted(dbg, 1.0, ov, 0.5, 0)

        # ── 밴드 스캔: BEV 없이 원본 화면 y범위를 그대로 씀 ──
        top_y = int(H * float(self.cfg.get("persp_top_y", 0.42)))
        bot_y = int(H * float(self.cfg.get("persp_bot_y", 0.98)))
        cv2.line(dbg, (0, top_y), (W, top_y), (255, 0, 255), 1)
        cv2.line(dbg, (0, bot_y), (W, bot_y), (255, 0, 255), 1)

        n = int(self.cfg["n_bands"])
        band_h = max(1, (bot_y - top_y) // n)
        center_offset = int(self.cfg.get("center_offset", 0))
        center_x = W // 2 + center_offset
        max_jump = int(W * 0.15)

        left_x = self.prev_left_x
        right_x = self.prev_right_x
        lane_w = self.prev_width if self.prev_width else int(W * 0.5)

        pts = []
        found = 0
        bot_left = bot_right = None
        bot_width = None
        left_real = right_real = False
        for b in range(n - 1, -1, -1):   # 차 바로 앞(화면 하단)부터
            by0 = top_y + b * band_h
            by1 = top_y + (b + 1) * band_h
            cy = (by0 + by1) // 2
            h_dash = np.sum(dash_mask[by0:by1, :], axis=0)
            h_solid = np.sum(solid_mask[by0:by1, :], axis=0)

            cx_R = None
            res = self._nearest_blob(h_solid, right_x, max_jump)
            if res is not None:
                cand, centers = res
                if cand is not None:
                    cx_R = cand
                elif right_x is None:
                    rs = centers[centers > center_x]
                    if len(rs) > 0:
                        cx_R = float(rs.min())

            cx_L = None
            res = self._nearest_blob(h_dash, left_x, max_jump)
            if res is not None:
                cand, centers = res
                if cand is not None:
                    cx_L = cand
                elif left_x is None:
                    ls = centers[centers < center_x]
                    if len(ls) > 0:
                        cx_L = float(ls.max())

            if cx_L is not None and cx_R is not None and cx_L >= cx_R:
                cx_L = cx_R = None

            if cx_L is not None and cx_R is not None:
                left_x, right_x = cx_L, cx_R
                w = cx_R - cx_L
                if W * 0.03 < w < W * 0.8:   # 원근상 먼 밴드는 폭이 훨씬 작음
                    lane_w = w
                    if bot_width is None:
                        bot_width = w
                target = (cx_L + cx_R) / 2
                conf = 1.0
                found += 1
            elif cx_R is not None:
                right_x = cx_R
                left_x = cx_R - lane_w
                target = cx_R - lane_w / 2
                conf = 0.6
                found += 1
            elif cx_L is not None:
                left_x = cx_L
                right_x = cx_L + lane_w
                target = cx_L + lane_w / 2
                conf = 0.6
                found += 1
            else:
                if left_x is None or right_x is None:
                    continue
                target = (left_x + right_x) / 2
                conf = 0.3

            if cx_L is not None:
                left_real = True
            if cx_R is not None:
                right_real = True
            if bot_left is None:
                bot_left, bot_right = left_x, right_x

            pts.append((cy, target, conf))
            if cx_L is not None:
                cv2.circle(dbg, (int(cx_L), cy), 5, COLOR_DASH, -1)
            if cx_R is not None:
                cv2.circle(dbg, (int(cx_R), cy), 5, COLOR_SOLID, -1)
            cv2.circle(dbg, (int(target), cy), 5, (0, 255, 255), -1)

        if found == 0:
            self.prev_left_x = self.prev_right_x = None
            self.miss_left = self.miss_right = 0
            cv2.putText(dbg, "NO LANE", (W // 2 - 70, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return None, 0, dbg, dbg

        self.prev_left_x, self.prev_right_x = bot_left, bot_right
        if bot_width is not None:
            self.prev_width = bot_width

        self.miss_left = 0 if left_real else self.miss_left + 1
        self.miss_right = 0 if right_real else self.miss_right + 1
        if self.miss_left >= self.reacquire_frames:
            self.prev_left_x = None
        if self.miss_right >= self.reacquire_frames:
            self.prev_right_x = None

        # ── 룩어헤드 가중 평균 ──
        la_gain = float(self.cfg.get("lookahead_gain", 4.0))
        band_span = max(1, bot_y - top_y)
        wsum, xsum = 0.0, 0.0
        for cy, cx, conf in pts:
            if conf >= 1.0:
                wgt = 1.0 + ((bot_y - cy) / band_span) * la_gain
            else:
                wgt = conf
            wsum += wgt
            xsum += cx * wgt
        line_x = xsum / wsum

        # ── 곡률/헤딩 (원근 오염 있음 — 모듈 docstring 참고) ──
        ys = [p[0] for p in pts]
        mid_y = (min(ys) + max(ys)) / 2.0
        near = [(cx, cf) for cy, cx, cf in pts if cy >= mid_y]
        far = [(cx, cf) for cy, cx, cf in pts if cy < mid_y]
        curve = 0.0
        if near and far:
            near_x = sum(cx * cf for cx, cf in near) / sum(cf for _, cf in near)
            far_x = sum(cx * cf for cx, cf in far) / sum(cf for _, cf in far)
            curve = (far_x - near_x) / (W / 2.0)
            cv2.line(dbg, (int(near_x), bot_y), (int(far_x), top_y),
                     (255, 255, 0), 2)

        heading_gain = float(self.cfg.get("heading_gain", 0.5))
        heading = 0.0
        if len(pts) >= 2:
            ys_ = np.array([p[0] for p in pts], dtype=float)
            xs_ = np.array([p[1] for p in pts], dtype=float)
            ws_ = np.array([p[2] for p in pts], dtype=float)
            if np.ptp(ys_) > 1:
                coef = np.polyfit(ys_, xs_, 1, w=ws_)
                slope = float(coef[0])
                heading = -slope
                y0, y1 = bot_y, max(top_y, bot_y - 120)
                x0 = float(np.polyval(coef, y0))
                x1 = x0 + slope * (y1 - y0)
                cv2.line(dbg, (int(x0), y0), (int(x1), y1), (255, 0, 255), 2)

        curve_gain = float(self.cfg.get("curve_gain", 0.5))
        pos_err = (line_x - center_x) / (W / 2.0)
        err_norm = float(np.clip(pos_err + curve_gain * curve
                                 + heading_gain * heading, -1, 1))

        cv2.line(dbg, (int(line_x), bot_y), (int(line_x), top_y),
                 (0, 255, 255), 2)
        cv2.line(dbg, (center_x, bot_y), (center_x, top_y), (0, 200, 200), 1)
        cv2.putText(dbg,
                    f"infer {self.infer_ms:.0f}ms  NO-BEV  bands {found}/{n}  "
                    f"miss L{self.miss_left} R{self.miss_right}",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)

        return err_norm, found, dbg, dbg
