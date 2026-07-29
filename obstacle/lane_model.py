"""
lane_model.py -- YOLOv8n-seg 기반 차선 검출 (장애물 미션용: 차선 전환 지원).

lane 프로젝트 dl_model/lane_model.py 에서 분리·확장한 독립 사본.
클래스: 0=dash(중앙 점선) 1=lane(횡단보도, 무시) 2=solid(실선)

핵심 확장 — 비전 기반 차선 자동 판정 (lane_est):
  좌우 경계는 클래스 무관하게 "차에서 가장 가까운 선"으로 잡고,
  잡힌 경계의 클래스로 현재 차선을 스스로 판정한다:
    왼쪽 경계=solid → inner(1차선)  /  오른쪽 경계=solid → outer(2차선)
  수동 모드 전환(set_lane)이 필요 없음 — 차선변경 기동이 어긋나도
  비전이 실제 위치를 판정하므로 상태가 저절로 진실에 수렴한다.
  연속 lane_switch_frames 프레임 일치해야 전환(깜빡임 방지).

중앙 추정(2026-07-25, drive와 동일한 개선 이식):
  · ROI 크롭 — 사다리꼴 밖 검출을 워핑 전에 제거
  · 가상 양선 — 좌/우 경계를 각각 직선피팅해, 한쪽이 끊긴 밴드는 예측으로
    채워 실측+예측 중점을 잡음(폭 상수 의존 제거)
  · 고정 차로폭 + cosθ 보정 — 가상 양선 불가 시 폴백
  · 비대칭 룩어헤드 — 코너 탈출 국면엔 룩어헤드 축소(오버슈트 방지)

인터페이스: process(frame) -> (err_norm, found, dbg, warped_dbg)
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

# 디버그 색 (BGR): dash=주황, solid=파랑, target=밝은초록
COLOR_DASH = (0, 165, 255)
COLOR_SOLID = (255, 80, 0)
COLOR_TARGET = (0, 255, 0)       # 차가 향하는 목표(밴드 점 + line_x 세로선)
COLOR_PRED = (150, 150, 150)     # 예측(가상) 경계점 — 실측과 구분

# 실행 위치(CWD)와 무관하게 항상 model/ 을 가리키도록 이 파일 기준 경로 사용
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model", "lane_seg_best.pt")


class ModelLaneDetector:
    def __init__(self, cfg, model_path=DEFAULT_MODEL_PATH):
        self.cfg = cfg
        if YOLO is None:
            raise ImportError("ultralytics 필요: pip install ultralytics")

        print(f"[ModelLaneDetector] 모델 로드: {model_path}")
        self.model = YOLO(model_path)
        self.device = str(cfg.get("model_device", "cpu"))
        self.imgsz = int(cfg.get("model_imgsz", 320))
        self.conf = float(cfg.get("model_conf", 0.4))

        # 프레임 간 추적 상태 (BEV 좌표계)
        self.prev_left_x = None
        self.prev_right_x = None
        self.prev_width = None
        self.prev_near_off = 0.0   # 비대칭 룩어헤드용: 직전 프레임 횡오프셋(px)
        # 한쪽 선을 연속으로 못 본 프레임 수 → 임계 초과 시 그쪽 추적 리셋
        self.miss_left = 0
        self.miss_right = 0
        self.reacquire_frames = int(cfg.get("reacquire_frames", 8))
        self.infer_ms = 0.0
        self._mask_cache = None   # (dash, solid) — 튠 모드 재사용용

        # ── 비전 기반 차선 자동 판정 ──
        #   왼쪽 경계=solid → inner(1차선, 인덱스0) / 오른쪽 경계=solid → outer(2차선, 인덱스1)
        #   연속 lane_switch_frames 프레임 일치해야 전환 (변경 중 깜빡임 방지)
        self.lane_est = "outer"
        self._lane_raw_prev = None
        self._lane_votes = 0
        self.lane_switch_frames = int(cfg.get("lane_switch_frames", 4))
        self.last_heading = 0.0   # 마지막 프레임 헤딩 (미검출 시 None)

        # 워밍업 (첫 추론 지연 제거)
        self.model.predict(np.zeros((360, 640, 3), dtype=np.uint8),
                           imgsz=self.imgsz, device=self.device,
                           conf=self.conf, verbose=False)

    # ── 추론: 클래스별 마스크 분리 ──────────────────────────────
    def _infer(self, frame_bgr, reuse=False):
        # reuse=True: 마지막 추론 마스크 재사용 (IPM 튜닝 중 트랙바 반응 ↑)
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
                    continue  # 기타 노면표시는 조향에 사용 안 함
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                if c == CLS_DASH:
                    dash[m > 0.5] = 255
                else:
                    solid[m > 0.5] = 255
        self._mask_cache = (dash, solid)
        return dash, solid

    # ── 밴드 히스토그램에서 이전 위치와 가장 가까운 덩어리 중심 ──
    @staticmethod
    def _nearest_blob(hist, ref_x, max_jump):
        xs = np.where(hist > 0)[0]
        if len(xs) == 0:
            return None
        splits = np.where(np.diff(xs) > 5)[0]
        blobs = np.split(xs, splits + 1)
        centers = np.array([b.mean() for b in blobs])
        if ref_x is None:
            return None, centers  # 호출부에서 초기 선택
        i = int(np.argmin(np.abs(centers - ref_x)))
        if abs(centers[i] - ref_x) > max_jump:
            return None, centers
        return float(centers[i]), centers

    # ── 좌/우 경계 직선 피팅 (가상 양선용) ──────────────────────
    @staticmethod
    def _fit_line(pts_xy):
        """검출점 (cy,cx)들로 x=a*cy+b 피팅. 스팬 밖 cy는 끝값 유지(외삽폭주 방지).
        점 2개 미만이면 None(피팅 불가 → 폭 offset 폴백)."""
        if len(pts_xy) < 2:
            return None
        ys = np.array([p[0] for p in pts_xy], dtype=float)
        xs = np.array([p[1] for p in pts_xy], dtype=float)
        ymin, ymax = float(ys.min()), float(ys.max())
        if ymax - ymin < 1:
            xm = float(xs.mean())
            return lambda cy: xm
        a, b = np.polyfit(ys, xs, 1)

        def pred(cy, a=a, b=b, ymin=ymin, ymax=ymax):
            cyc = ymin if cy < ymin else (ymax if cy > ymax else cy)
            return a * cyc + b
        return pred

    @staticmethod
    def _slope_factor(pts_xy):
        """cosθ 보정계수 √(1+slope²), 상한 1.8. 폭 offset 폴백에서만 사용."""
        if len(pts_xy) < 2:
            return 1.0
        ys = np.array([p[0] for p in pts_xy], dtype=float)
        xs = np.array([p[1] for p in pts_xy], dtype=float)
        if np.ptp(ys) < 1:
            return 1.0
        slope = float(np.polyfit(ys, xs, 1)[0])
        return float(min(1.8, np.sqrt(1.0 + slope * slope)))

    def process(self, frame_bgr, reuse_masks=False):
        H, W = frame_bgr.shape[:2]
        dash_mask, solid_mask = self._infer(frame_bgr, reuse=reuse_masks)

        # ── IPM 사다리꼴(=ROI) 네 모서리 (비율 → 픽셀) ──
        c = self.cfg
        tl = [W * float(c.get("ipm_tl_x", 0.25)), H * float(c.get("ipm_tl_y", 0.40))]
        tr = [W * float(c.get("ipm_tr_x", 0.75)), H * float(c.get("ipm_tr_y", 0.40))]
        bl = [W * float(c.get("ipm_bl_x", -0.40)), H * float(c.get("ipm_bl_y", 0.98))]
        br = [W * float(c.get("ipm_br_x", 1.40)), H * float(c.get("ipm_br_y", 0.98))]
        tr[0] = max(tr[0], tl[0] + 10)
        br[0] = max(br[0], bl[0] + 10)
        bl[1] = max(bl[1], tl[1] + 10)
        br[1] = max(br[1], tr[1] + 10)
        src = np.float32([tl, tr, bl, br])

        # ── ROI 크롭: 사다리꼴 밖 검출은 버림 (밴드 스캔 오염 방지) ──
        roi_poly = np.int32(src[[0, 1, 3, 2]])
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(roi_mask, roi_poly, 255)
        dash_mask = cv2.bitwise_and(dash_mask, roi_mask)
        solid_mask = cv2.bitwise_and(solid_mask, roi_mask)

        # ── 원본 뷰 디버그: 클래스별 색 오버레이 ──
        dbg = frame_bgr.copy()
        ov = np.zeros_like(dbg)
        ov[dash_mask == 255] = COLOR_DASH
        ov[solid_mask == 255] = COLOR_SOLID
        dbg = cv2.addWeighted(dbg, 1.0, ov, 0.5, 0)
        cv2.putText(dbg, f"infer {self.infer_ms:.0f}ms  "
                    f"dash/solid px {int(np.count_nonzero(dash_mask))}"
                    f"/{int(np.count_nonzero(solid_mask))}",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)

        # ── 버드아이뷰 변환 ──
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
        # BEV 배경도 ROI 밖은 어둡게 → 밴드 스캔이 실제 쓰는 영역만 밝게
        w_roi = cv2.warpPerspective(roi_mask, M, (W, H), flags=cv2.INTER_NEAREST)
        warped_dbg[w_roi == 0] = (warped_dbg[w_roi == 0] * 0.25).astype(np.uint8)
        ovw = np.zeros_like(warped_dbg)
        ovw[w_dash == 255] = COLOR_DASH
        ovw[w_solid == 255] = COLOR_SOLID
        warped_dbg = cv2.addWeighted(warped_dbg, 0.7, ovw, 0.5, 0)

        # ── 밴드 스캔 파라미터 ──
        n = int(self.cfg["n_bands"])
        band_h = H // n
        center_offset = int(self.cfg.get("center_offset", 0))
        center_x = W // 2 + center_offset
        max_jump = int(W * 0.15)
        left_x = self.prev_left_x
        right_x = self.prev_right_x
        lane_w_fixed = int(self.cfg.get("bev_lane_width", int(W * 0.6)))
        lane_w = self.prev_width if self.prev_width else lane_w_fixed

        # 클래스 무관 결합 마스크: 좌우 경계는 최근접으로 잡고 클래스는 사후 분류
        w_any = cv2.bitwise_or(w_dash, w_solid)

        # ── Pass 1: 밴드별 좌우 경계 검출 + 클래스 분류 (목표는 Pass 2에서) ──
        bands = []        # (cy, cx_L, cx_R, lx, rx, kind, cls_L, cls_R)
        left_all, right_all = [], []   # 경계별 검출점 (가상 양선 피팅용)
        found = 0
        bot_left = bot_right = None
        bot_width = None
        left_real = right_real = False
        bot_cls_L = bot_cls_R = None   # 최하단 실검출 경계의 클래스 (lane_est용)
        for b in range(n - 1, -1, -1):   # 차 바로 앞(하단)부터
            by0, by1 = b * band_h, (b + 1) * band_h
            cy = (by0 + by1) // 2
            h_any = np.sum(w_any[by0:by1, :], axis=0)
            h_d = np.sum(w_dash[by0:by1, :], axis=0)
            h_s = np.sum(w_solid[by0:by1, :], axis=0)

            def _cls(x):
                """경계 x 주변 ±12px에서 dash/solid 픽셀 다수결.
                한쪽이 뚜렷하게 우세할 때만 판정, 애매하면 None(불확실)."""
                a, z = int(max(0, x - 12)), int(min(W, x + 12))
                s_sum, d_sum = float(h_s[a:z].sum()), float(h_d[a:z].sum())
                total = s_sum + d_sum
                if total < 1 or abs(s_sum - d_sum) / total < 0.3:
                    return None
                return "solid" if s_sum > d_sum else "dash"

            # 오른쪽 경계: 이전 x 근처, 초기엔 중앙 오른쪽 최근접
            cx_R = None
            res = self._nearest_blob(h_any, right_x, max_jump)
            if res is not None:
                cand, centers = res
                if cand is not None:
                    cx_R = cand
                elif right_x is None:
                    rs = centers[centers > center_x]
                    if len(rs) > 0:
                        cx_R = float(rs.min())

            # 왼쪽 경계: 이전 x 근처, 초기엔 중앙 왼쪽 최근접
            cx_L = None
            res = self._nearest_blob(h_any, left_x, max_jump)
            if res is not None:
                cand, centers = res
                if cand is not None:
                    cx_L = cand
                elif left_x is None:
                    ls = centers[centers < center_x]
                    if len(ls) > 0:
                        cx_L = float(ls.max())

            cls_L = _cls(cx_L) if cx_L is not None else None
            cls_R = _cls(cx_R) if cx_R is not None else None

            # 좌우가 뒤집히면(교차) 신뢰 불가 → 이 밴드 버림
            if cx_L is not None and cx_R is not None and cx_L >= cx_R:
                cx_L = cx_R = None
                cls_L = cls_R = None

            # lane_est용 클래스 확정은 교차 무효화 이후 값으로만 (오염 방지)
            if bot_cls_L is None and cls_L is not None:
                bot_cls_L = cls_L
            if bot_cls_R is None and cls_R is not None:
                bot_cls_R = cls_R

            if cx_L is not None and cx_R is not None:
                left_x, right_x = cx_L, cx_R
                w = cx_R - cx_L
                if W * 0.2 < w < W * 0.8:
                    lane_w = w
                    if bot_width is None:
                        bot_width = w
                kind = "both"
                found += 1
            elif cx_R is not None:          # 오른쪽 경계만
                right_x = cx_R
                left_x = cx_R - lane_w_fixed   # 팬텀 좌측(추적/추정용, 대략)
                kind = "right"
                found += 1
            elif cx_L is not None:          # 왼쪽 경계만
                left_x = cx_L
                right_x = cx_L + lane_w_fixed
                kind = "left"
                found += 1
            else:
                if left_x is None or right_x is None:
                    continue
                kind = "est"

            if cx_L is not None:
                left_real = True
                left_all.append((cy, cx_L))
            if cx_R is not None:
                right_real = True
                right_all.append((cy, cx_R))
            if bot_left is None:
                bot_left, bot_right = left_x, right_x
            bands.append((cy, cx_L, cx_R, left_x, right_x, kind, cls_L, cls_R))

        # ── 미검출 처리: 실검출 0밴드면 err=None (main lost 로직 발동) ──
        if found == 0:
            self.prev_left_x = self.prev_right_x = None
            self.miss_left = self.miss_right = 0
            self.last_heading = None   # 헤딩 정보 없음 (FSM 정렬 판단용)
            cv2.putText(warped_dbg, "NO LANE", (W // 2 - 70, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return None, 0, dbg, warped_dbg

        # 다음 프레임 추적 기준은 "차 바로 앞(최하단) 밴드" 값
        self.prev_left_x, self.prev_right_x = bot_left, bot_right
        if bot_width is not None:
            self.prev_width = bot_width

        # ── 차선 자동 판정(lane_est): 최하단 경계 클래스 + 연속 프레임 다수결 ──
        # 왼쪽=solid → inner / 오른쪽=solid → outer (점선은 끊겨서 solid 우선)
        raw = None
        if bot_cls_L == "solid":
            raw = "inner"
        elif bot_cls_R == "solid":
            raw = "outer"
        elif bot_cls_L == "dash":
            raw = "outer"
        elif bot_cls_R == "dash":
            raw = "inner"
        self.lane_switch_frames = int(self.cfg.get("lane_switch_frames", 4))
        if raw is not None and raw != self.lane_est:
            self._lane_votes = self._lane_votes + 1 \
                if raw == self._lane_raw_prev else 1
            if self._lane_votes >= self.lane_switch_frames:
                self.lane_est = raw
                self._lane_votes = 0
                print(f"[ModelLaneDetector] 차선 판정 -> {raw}")
        else:
            self._lane_votes = 0
        self._lane_raw_prev = raw

        # 한쪽 선을 오래 못 보면 그쪽 추적 리셋 → 초기 탐색으로 재획득
        self.miss_left = 0 if left_real else self.miss_left + 1
        self.miss_right = 0 if right_real else self.miss_right + 1
        if self.miss_left >= self.reacquire_frames:
            self.prev_left_x = None
        if self.miss_right >= self.reacquire_frames:
            self.prev_right_x = None

        # ── 가상 양선: 좌/우 경계 각각 직선피팅 → 끊긴 밴드 위치 예측 ──
        left_fit = self._fit_line(left_all)
        right_fit = self._fit_line(right_all)
        f_left = self._slope_factor(left_all)
        f_right = self._slope_factor(right_all)
        half = lane_w_fixed / 2.0

        # ── Pass 2: 목표 계산 (가상 양선 우선, 불가 시 폭 offset 폴백) + 그리기 ──
        pts = []          # (cy, target_cx, conf)
        for cy, cx_L, cx_R, lx, rx, kind, cls_L, cls_R in bands:
            d = cx_L if cx_L is not None else (left_fit(cy) if left_fit else None)
            s = cx_R if cx_R is not None else (right_fit(cy) if right_fit else None)
            if d is not None and s is not None and d < s:
                target = (d + s) / 2                # 가상 양선: 폭 상수 불필요
                if cx_L is not None and cx_R is not None:
                    conf = 1.0
                elif cx_L is not None or cx_R is not None:
                    conf = 0.85
                else:
                    conf = 0.7
            elif cx_R is not None:                   # 오른쪽만 + 왼쪽 정보 없음
                target = cx_R - half * f_right       # 폭 offset(cosθ 보정) 폴백
                conf = 0.6
            elif cx_L is not None:                   # 왼쪽만 + 오른쪽 정보 없음
                target = cx_L + half * f_left
                conf = 0.6
            else:                                    # est: 팬텀 중점
                target = (lx + rx) / 2
                conf = 0.3
            pts.append((cy, target, conf))
            # 실측 경계=클래스색 꽉찬원 / 예측 경계=회색 작은원
            if cx_L is not None:
                col = COLOR_SOLID if cls_L == "solid" else COLOR_DASH
                cv2.circle(warped_dbg, (int(cx_L), cy), 5, col, -1)
            elif d is not None:
                cv2.circle(warped_dbg, (int(d), cy), 3, COLOR_PRED, 1)
            if cx_R is not None:
                col = COLOR_SOLID if cls_R == "solid" else COLOR_DASH
                cv2.circle(warped_dbg, (int(cx_R), cy), 5, col, -1)
            elif s is not None:
                cv2.circle(warped_dbg, (int(s), cy), 3, COLOR_PRED, 1)
            cv2.circle(warped_dbg, (int(target), cy), 7, COLOR_TARGET, -1)

        # ── 비대칭 룩어헤드 (진입엔 풀, 탈출엔 축소) ──
        la_gain = float(self.cfg.get("lookahead_gain", 4.0))
        la_exit = float(self.cfg.get("lookahead_exit_scale", 0.4))
        near_t = [t for cy, t, cf in pts if cy >= H * 0.6]
        cur_off = abs(sum(near_t) / len(near_t) - center_x) if near_t else 0.0
        exiting = cur_off < self.prev_near_off - 1.0
        self.prev_near_off = cur_off
        eff_la = la_gain * (la_exit if exiting else 1.0)

        wsum, xsum = 0.0, 0.0
        for cy, cx, conf in pts:
            if conf >= 1.0:
                wgt = 1.0 + ((H - cy) / H) * eff_la
            else:
                wgt = conf
            wsum += wgt
            xsum += cx * wgt
        line_x = xsum / wsum

        # ── 곡률 피드포워드 ──
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

        # ── 헤딩 항 (Stanley식) ──
        heading_gain = float(self.cfg.get("heading_gain", 0.5))
        heading = 0.0
        if len(pts) >= 2:
            ys = np.array([p[0] for p in pts], dtype=float)
            xs = np.array([p[1] for p in pts], dtype=float)
            ws = np.array([p[2] for p in pts], dtype=float)
            if np.ptp(ys) > 1:
                coef = np.polyfit(ys, xs, 1, w=ws)
                slope = float(coef[0])
                heading = -slope
                y0, y1 = H - 10, H - 130
                x0 = float(np.polyval(coef, y0))
                x1 = x0 + slope * (y1 - y0)
                cv2.line(warped_dbg, (int(x0), y0), (int(x1), y1),
                         (255, 0, 255), 2)
        self.last_heading = heading   # FSM(정렬 단계 종료 판단)에서 참조

        curve_gain = float(self.cfg.get("curve_gain", 0.5))
        pos_err = (line_x - center_x) / (W / 2.0)
        err_norm = float(np.clip(pos_err + curve_gain * curve
                                 + heading_gain * heading, -1, 1))

        cv2.line(warped_dbg, (int(line_x), 0), (int(line_x), H), COLOR_TARGET, 3)
        cv2.line(warped_dbg, (center_x, 0), (center_x, H), (0, 200, 200), 2)
        cv2.putText(warped_dbg,
                    f"[{self.lane_est.upper()}] L={bot_cls_L} R={bot_cls_R}  "
                    f"bands {found}/{n}  LA={'EXIT' if exiting else 'full'}",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)

        return err_norm, found, dbg, warped_dbg
