"""
lane_model.py -- YOLOv8n-seg 기반 차선 검출 + 자동 IPM 캘리브레이션 (ROI 트랙바 없음)

dl_model/lane_model.py 와 동일한 클래스 인지형 검출·밴드스캔·헤딩/곡률 피드포워드를
그대로 쓰되, **BEV 사다리꼴을 사람이 트랙바로 맞추는 대신 매 프레임 스스로 계산**한다.

핵심 아이디어 (소실점 기반 자동 캘리브레이션):
  dash(왼쪽 경계), solid(오른쪽 경계) 마스크 픽셀에 각각 직선 x=a*y+b 를 피팅하면,
  두 직선의 교차점 = 카메라가 보는 "소실점"(원근 상 두 평행선이 만나는 점)이다.
  이 소실점은 카메라 틸트/장착각이 바뀌면 그에 따라 자동으로 이동하므로,
  소실점 바로 아래를 사다리꼴 상단으로, 화면 하단을 사다리꼴 하단으로 잡고
  두 직선 위의 대응 x좌표를 그대로 사다리꼴 네 꼭짓점으로 쓰면
  장소/카메라 각도가 바뀌어도 IPM이 스스로 따라간다.

안전장치:
  - 각 선의 픽셀 수가 너무 적으면(신뢰 불가) 이전 프레임 값 유지, 그마저 없으면
    calib.json의 고정 사다리꼴(ipm_tl_x 등)로 폴백
  - 소실점이 비정상 범위(화면 밖 크게 벗어남 등)면 마찬가지로 폴백
  - EMA 스무딩으로 프레임간 지터 억제

인터페이스: process(frame) -> (err_norm, found, dbg, warped_dbg)  (dl_model과 동일)
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

        print(f"[ModelLaneDetector-autocal] 모델 로드: {model_path}")
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

        # 자동 캘리브레이션 상태
        self._auto_src = None       # EMA로 스무딩된 사다리꼴 (4x2)
        self.calib_mode = "INIT"    # 디버그 표시용: AUTO / FALLBACK / INIT

        self.model.predict(np.zeros((360, 640, 3), dtype=np.uint8),
                           imgsz=self.imgsz, device=self.device,
                           conf=self.conf, verbose=False)

    # ── 추론: 클래스별 마스크 분리 ──────────────────────────────
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

    # ── 소실점 기반 자동 사다리꼴 계산 ──────────────────────────
    def _fallback_src(self, W, H):
        c = self.cfg
        tl = [W * float(c.get("ipm_tl_x", 0.25)), H * float(c.get("ipm_tl_y", 0.40))]
        tr = [W * float(c.get("ipm_tr_x", 0.75)), H * float(c.get("ipm_tr_y", 0.40))]
        bl = [W * float(c.get("ipm_bl_x", -0.40)), H * float(c.get("ipm_bl_y", 0.98))]
        br = [W * float(c.get("ipm_br_x", 1.40)), H * float(c.get("ipm_br_y", 0.98))]
        return np.float32([tl, tr, bl, br])

    def _auto_trapezoid(self, dash_mask, solid_mask, W, H):
        min_px = int(self.cfg.get("auto_calib_min_px", 150))
        alpha = float(self.cfg.get("auto_calib_alpha", 0.15))
        vp_margin = float(self.cfg.get("auto_calib_vp_margin", 0.14)) * H
        y_bot = H * float(self.cfg.get("auto_calib_bot_y", 0.98))

        ys_l, xs_l = np.where(dash_mask > 0)
        ys_r, xs_r = np.where(solid_mask > 0)

        computed = None
        if len(xs_l) >= min_px and len(xs_r) >= min_px:
            # x = a*y + b  (y를 독립변수로: 선이 수직에 가까워 y에 대해 안정적)
            a_l, b_l = np.polyfit(ys_l, xs_l, 1)
            a_r, b_r = np.polyfit(ys_r, xs_r, 1)

            if abs(a_l - a_r) > 1e-4:   # 평행에 가까우면 소실점 계산 불안정 → 폐기
                y_vp = (b_r - b_l) / (a_l - a_r)
                y_top = y_vp + vp_margin
                # 상단이 하단과 너무 가깝거나(사다리꼴 퇴화) 화면 밖으로 튀면 클램프
                y_top = max(H * 0.05, min(y_top, y_bot - H * 0.15))

                bl_x = a_l * y_bot + b_l
                br_x = a_r * y_bot + b_r
                tl_x = a_l * y_top + b_l
                tr_x = a_r * y_top + b_r

                # 상식 밖 좌표(화면 폭의 몇 배로 튐) 방지
                if all(-2 * W < x < 3 * W for x in (bl_x, br_x, tl_x, tr_x)) \
                        and br_x - bl_x > W * 0.05:
                    computed = np.float32([[tl_x, y_top], [tr_x, y_top],
                                           [bl_x, y_bot], [br_x, y_bot]])

        if computed is not None:
            if self._auto_src is None:
                self._auto_src = computed
            else:
                self._auto_src = alpha * computed + (1 - alpha) * self._auto_src
            self.calib_mode = "AUTO"
            return self._auto_src
        elif self._auto_src is not None:
            self.calib_mode = "AUTO(유지)"
            return self._auto_src
        else:
            self.calib_mode = "FALLBACK"
            return self._fallback_src(W, H)

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

        # ── IPM 사다리꼴: 자동 계산 (트랙바 없음) ──
        src = self._auto_trapezoid(dash_mask, solid_mask, W, H)
        tr = list(src[1]); br = list(src[3])
        tr[0] = max(tr[0], src[0][0] + 10)
        br[0] = max(br[0], src[2][0] + 10)
        src = np.float32([src[0], tr, src[2], br])

        dbg = frame_bgr.copy()
        ov = np.zeros_like(dbg)
        ov[dash_mask == 255] = COLOR_DASH
        ov[solid_mask == 255] = COLOR_SOLID
        dbg = cv2.addWeighted(dbg, 1.0, ov, 0.5, 0)
        mode_color = (0, 255, 0) if self.calib_mode == "AUTO" else (0, 165, 255)
        cv2.putText(dbg, f"infer {self.infer_ms:.0f}ms  calib={self.calib_mode}",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1)

        dst = np.float32([[W * 0.2, 0], [W * 0.8, 0],
                          [W * 0.2, H], [W * 0.8, H]])
        M = cv2.getPerspectiveTransform(src, dst)
        cv2.polylines(dbg, [np.int32(src[[0, 1, 3, 2]])], True,
                      (255, 0, 255), 2)

        w_dash = cv2.warpPerspective(dash_mask, M, (W, H),
                                     flags=cv2.INTER_NEAREST)
        w_solid = cv2.warpPerspective(solid_mask, M, (W, H),
                                      flags=cv2.INTER_NEAREST)
        warped_dbg = cv2.warpPerspective(frame_bgr, M, (W, H))
        ovw = np.zeros_like(warped_dbg)
        ovw[w_dash == 255] = COLOR_DASH
        ovw[w_solid == 255] = COLOR_SOLID
        warped_dbg = cv2.addWeighted(warped_dbg, 0.7, ovw, 0.5, 0)

        # ── 밴드 스캔: 왼쪽=dash, 오른쪽=solid (dl_model과 동일) ──
        n = int(self.cfg["n_bands"])
        band_h = H // n
        center_offset = int(self.cfg.get("center_offset", 0))
        center_x = W // 2 + center_offset
        max_jump = int(W * 0.15)

        left_x = self.prev_left_x
        right_x = self.prev_right_x
        lane_w = self.prev_width if self.prev_width else int(W * 0.6)

        pts = []
        found = 0
        bot_left = bot_right = None
        bot_width = None
        left_real = right_real = False
        for b in range(n - 1, -1, -1):
            by0, by1 = b * band_h, (b + 1) * band_h
            cy = (by0 + by1) // 2
            h_dash = np.sum(w_dash[by0:by1, :], axis=0)
            h_solid = np.sum(w_solid[by0:by1, :], axis=0)

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
                if W * 0.2 < w < W * 0.8:
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
                cv2.circle(warped_dbg, (int(cx_L), cy), 5, COLOR_DASH, -1)
            if cx_R is not None:
                cv2.circle(warped_dbg, (int(cx_R), cy), 5, COLOR_SOLID, -1)
            cv2.circle(warped_dbg, (int(target), cy), 5, (0, 255, 255), -1)

        if found == 0:
            self.prev_left_x = self.prev_right_x = None
            self.miss_left = self.miss_right = 0
            cv2.putText(warped_dbg, "NO LANE", (W // 2 - 70, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return None, 0, dbg, warped_dbg

        self.prev_left_x, self.prev_right_x = bot_left, bot_right
        if bot_width is not None:
            self.prev_width = bot_width

        self.miss_left = 0 if left_real else self.miss_left + 1
        self.miss_right = 0 if right_real else self.miss_right + 1
        if self.miss_left >= self.reacquire_frames:
            self.prev_left_x = None
        if self.miss_right >= self.reacquire_frames:
            self.prev_right_x = None

        la_gain = float(self.cfg.get("lookahead_gain", 4.0))
        wsum, xsum = 0.0, 0.0
        for cy, cx, conf in pts:
            if conf >= 1.0:
                wgt = 1.0 + ((H - cy) / H) * la_gain
            else:
                wgt = conf
            wsum += wgt
            xsum += cx * wgt
        line_x = xsum / wsum

        ys = [p[0] for p in pts]
        mid_y = (min(ys) + max(ys)) / 2.0
        near = [(cx, cf) for cy, cx, cf in pts if cy >= mid_y]
        far = [(cx, cf) for cy, cx, cf in pts if cy < mid_y]
        curve = 0.0
        if near and far:
            near_x = sum(cx * cf for cx, cf in near) / sum(cf for _, cf in near)
            far_x = sum(cx * cf for cx, cf in far) / sum(cf for _, cf in far)
            curve = (far_x - near_x) / (W / 2.0)
            cv2.line(warped_dbg, (int(near_x), H - 20), (int(far_x), 20),
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
                y0, y1 = H - 10, H - 130
                x0 = float(np.polyval(coef, y0))
                x1 = x0 + slope * (y1 - y0)
                cv2.line(warped_dbg, (int(x0), y0), (int(x1), y1),
                         (255, 0, 255), 2)

        curve_gain = float(self.cfg.get("curve_gain", 0.5))
        pos_err = (line_x - center_x) / (W / 2.0)
        err_norm = float(np.clip(pos_err + curve_gain * curve
                                 + heading_gain * heading, -1, 1))

        cv2.line(warped_dbg, (int(line_x), 0), (int(line_x), H),
                 (0, 255, 255), 2)
        cv2.line(warped_dbg, (center_x, 0), (center_x, H), (0, 200, 200), 2)
        cv2.putText(warped_dbg,
                    f"L=dash R=solid  bands {found}/{n}  "
                    f"miss L{self.miss_left} R{self.miss_right}  "
                    f"calib={self.calib_mode}",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)

        return err_norm, found, dbg, warped_dbg
