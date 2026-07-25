"""
lane_model.py -- YOLOv8n-seg 기반 차선 검출 (클래스 인지형).

모델: model/lane_seg_best.pt (yolov8n-seg, imgsz 640으로 학습)
클래스: 0=dash(중앙 점선) 1=lane(기타 노면표시, 무시) 2=solid(바깥 실선)

핵심 아이디어 — 반시계 2차선에서 내 차로는
  왼쪽 경계 = 중앙 점선(dash), 오른쪽 경계 = 바깥 실선(solid).
클래스별 마스크를 분리해서 왼쪽은 dash에서만, 오른쪽은 solid에서만 찾으므로
centroid 방식의 고질병이던 "점선을 실선으로 오인" 문제가 구조적으로 안 생긴다.

속도 (MacBook Air M3 실측): cpu+imgsz320 ≈ 10ms/frame (97fps),
imgsz320에서도 solid는 67/67장 전부 검출됨. calib.json으로 조정 가능:
  model_imgsz(320), model_conf(0.4), model_device("cpu")

인터페이스: process(frame) -> (err_norm, found, dbg, warped_dbg)
  err_norm: 정규화 측방오차 -1..1, 미검출이면 None (main의 lost 로직이 처리)
  found:    실제 차선이 검출된 밴드 수 (추정으로 채운 밴드는 제외)
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

# 디버그 색 (BGR): dash=주황, solid=파랑, target=밝은초록(눈에 잘 띄게)
COLOR_DASH = (0, 165, 255)
COLOR_SOLID = (255, 80, 0)
COLOR_TARGET = (0, 255, 0)      # 차가 향하는 목표(밴드 점 + line_x 세로선)

# 실행 위치(CWD)와 무관하게 항상 dl_model/model/ 을 가리키도록 이 파일 기준 경로 사용
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
        # (리셋해야 "중앙 기준 최근접" 초기 탐색이 다시 발동해 재획득 가능)
        self.miss_left = 0
        self.miss_right = 0
        self.reacquire_frames = int(cfg.get("reacquire_frames", 8))
        self.infer_ms = 0.0
        self._mask_cache = None   # (dash, solid) — 튠 모드 재사용용

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
        # 연속 구간(덩어리)으로 묶기
        splits = np.where(np.diff(xs) > 5)[0]
        blobs = np.split(xs, splits + 1)
        centers = np.array([b.mean() for b in blobs])
        if ref_x is None:
            return None, centers  # 호출부에서 초기 선택
        i = int(np.argmin(np.abs(centers - ref_x)))
        if abs(centers[i] - ref_x) > max_jump:
            return None, centers
        return float(centers[i]), centers

    def process(self, frame_bgr, reuse_masks=False):
        H, W = frame_bgr.shape[:2]
        dash_mask, solid_mask = self._infer(frame_bgr, reuse=reuse_masks)

        # ── IPM 사다리꼴(=ROI) 네 모서리 (비율 → 픽셀) ──
        c = self.cfg
        tl = [W * float(c.get("ipm_tl_x", 0.25)), H * float(c.get("ipm_tl_y", 0.40))]
        tr = [W * float(c.get("ipm_tr_x", 0.75)), H * float(c.get("ipm_tr_y", 0.40))]
        bl = [W * float(c.get("ipm_bl_x", -0.40)), H * float(c.get("ipm_bl_y", 0.98))]
        br = [W * float(c.get("ipm_br_x", 1.40)), H * float(c.get("ipm_br_y", 0.98))]
        # 퇴화 방지: 상/하단이 뒤집히거나 좌/우가 겹치면 변환이 깨짐
        tr[0] = max(tr[0], tl[0] + 10)
        br[0] = max(br[0], bl[0] + 10)
        bl[1] = max(bl[1], tl[1] + 10)
        br[1] = max(br[1], tr[1] + 10)
        src = np.float32([tl, tr, bl, br])

        # ── ROI 크롭: 사다리꼴(=자홍 박스) 밖 검출은 버린다 ──
        # 추론은 화면 전체에서 돌고 warpPerspective도 전체를 변환하므로, 크롭이
        # 없으면 박스 밖(배경/옆 트랙)의 선도 BEV로 매핑돼 밴드 스캔을 오염시킴.
        # 워핑 전에 마스크를 사다리꼴로 잘라 "박스 안"만 남긴다.
        roi_poly = np.int32(src[[0, 1, 3, 2]])       # tl, tr, br, bl 순 링
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
        # BEV 배경도 ROI(사다리꼴) 밖은 어둡게 → 밴드 스캔이 실제 쓰는 영역만 밝게.
        # (검출 마스크는 이미 ROI로 잘렸고, 여기선 배경 그림만 시각적으로 일치시킴)
        w_roi = cv2.warpPerspective(roi_mask, M, (W, H), flags=cv2.INTER_NEAREST)
        warped_dbg[w_roi == 0] = (warped_dbg[w_roi == 0] * 0.25).astype(np.uint8)
        ovw = np.zeros_like(warped_dbg)
        ovw[w_dash == 255] = COLOR_DASH
        ovw[w_solid == 255] = COLOR_SOLID
        warped_dbg = cv2.addWeighted(warped_dbg, 0.7, ovw, 0.5, 0)

        # ── 밴드 스캔: 왼쪽=dash, 오른쪽=solid ──
        n = int(self.cfg["n_bands"])
        band_h = H // n
        center_offset = int(self.cfg.get("center_offset", 0))
        center_x = W // 2 + center_offset
        max_jump = int(W * 0.15)

        # 프레임 간 추적: 이전 프레임 값에서 시작 (없으면 화면 기준 초기화)
        left_x = self.prev_left_x
        right_x = self.prev_right_x
        # 단일 차선 offset에 쓸 "고정" 차로폭(BEV px). 라이브 측정 폭(cx_R-cx_L)은
        # 커브에서 팽창(가로거리=폭/cosθ), 프레임 가장자리 조각남에선 축소돼
        # 양방향으로 불안정하므로, offset에는 이 고정값을 쓴다. (트랙바 LaneW px =
        # 직선 양선 구간의 실측 수직 폭에 맞출 것). 라이브 측정은 표시/기록용만.
        lane_w_fixed = int(self.cfg.get("bev_lane_width", int(W * 0.6)))
        lane_w = self.prev_width if self.prev_width else lane_w_fixed

        # ── Pass 1: 각 밴드의 선 위치만 확정 (목표는 기울기 안 뒤 Pass 2에서) ──
        bands = []        # (cy, cx_L, cx_R, lx, rx, kind)  kind: both/solid/dash/est
        found = 0         # 실검출 밴드 수
        bot_left = bot_right = None   # 최하단 유효 밴드의 좌/우 (다음 프레임 기준)
        bot_width = None              # 최하단 양선 밴드의 차로폭 (왜곡 최소)
        left_real = right_real = False
        for b in range(n - 1, -1, -1):   # 차 바로 앞(하단)부터
            by0, by1 = b * band_h, (b + 1) * band_h
            cy = (by0 + by1) // 2
            h_dash = np.sum(w_dash[by0:by1, :], axis=0)
            h_solid = np.sum(w_solid[by0:by1, :], axis=0)

            # solid(오른쪽 경계): 이전 x 근처, 초기엔 중앙 오른쪽 최근접
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

            # dash(왼쪽 경계): 이전 x 근처, 초기엔 중앙 왼쪽 최근접
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

            # 좌우가 뒤집히면(교차) 신뢰 불가 → 이 밴드 버림
            if cx_L is not None and cx_R is not None and cx_L >= cx_R:
                cx_L = cx_R = None

            if cx_L is not None and cx_R is not None:
                left_x, right_x = cx_L, cx_R
                w = cx_R - cx_L
                if W * 0.2 < w < W * 0.8:    # 비상식적 폭은 기억하지 않음
                    lane_w = w
                    if bot_width is None:
                        bot_width = w        # 최하단 양선 밴드 폭만 저장
                kind = "both"
                found += 1
            elif cx_R is not None:          # 실선만 (점선 끊긴 구간)
                right_x = cx_R
                left_x = cx_R - lane_w_fixed   # 팬텀 좌측(추적/추정용, 대략)
                kind = "solid"
                found += 1
            elif cx_L is not None:          # 점선만
                left_x = cx_L
                right_x = cx_L + lane_w_fixed
                kind = "dash"
                found += 1
            else:
                if left_x is None or right_x is None:
                    continue                # 아직 아무 기준 없음: 밴드 스킵
                kind = "est"                # 추정 (found에 안 셈)

            if cx_L is not None:
                left_real = True
            if cx_R is not None:
                right_real = True
            if bot_left is None:             # 첫(=최하단) 유효 밴드 기록
                bot_left, bot_right = left_x, right_x

            bands.append((cy, cx_L, cx_R, left_x, right_x, kind))

        # ── 미검출 처리: 실검출 0밴드면 err=None (main lost 로직 발동) ──
        if found == 0:
            self.prev_left_x = self.prev_right_x = None  # 추적 리셋
            self.miss_left = self.miss_right = 0
            cv2.putText(warped_dbg, "NO LANE", (W // 2 - 70, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return None, 0, dbg, warped_dbg

        # ── 선 기울기 → cosθ 보정계수 (한쪽 선만 볼 때 커브 중앙 정확도) ──
        # 수직 폭을 가로 offset으로 쓰면 실제로는 폭/cosθ 만큼 벌어져야 하므로
        # (선이 기울수록 커짐), 각 선의 라이브 기울기 slope=dx/dy로 √(1+slope²)를
        # 곱한다. 점 2개 미만이면 보정 없음(1.0). 급기울기 폭주 방지로 상한 1.8.
        def _slope_factor(pts_xy):
            if len(pts_xy) < 2:
                return 1.0
            ys = np.array([p[0] for p in pts_xy], dtype=float)
            xs = np.array([p[1] for p in pts_xy], dtype=float)
            if np.ptp(ys) < 1:
                return 1.0
            slope = float(np.polyfit(ys, xs, 1)[0])   # dx/dy
            return float(min(1.8, np.sqrt(1.0 + slope * slope)))

        dash_all = [(cy, cxl) for cy, cxl, cxr, _, _, _ in bands if cxl is not None]
        solid_all = [(cy, cxr) for cy, cxl, cxr, _, _, _ in bands if cxr is not None]
        f_dash = _slope_factor(dash_all)
        f_solid = _slope_factor(solid_all)
        half = lane_w_fixed / 2.0

        # ── 각 선을 프레임 내에서 직선 피팅 → 끊긴 밴드의 위치를 "예측" ──
        # 점선은 끊겨도 여러 밴드에서 잡히므로, 그 점들로 선을 그어 빈 밴드의
        # 점선 위치를 채운다(실선도 동일). 그러면 한쪽만 실측돼도 실측+예측으로
        # 중점을 잡을 수 있어(=가상 양선) 폭 상수에 의존하지 않는다.
        # 점 2개 미만이면 피팅 불가(None) → 그 선은 폭 offset 폴백을 쓴다.
        # 스팬 밖 cy는 끝값을 유지해 폭주(과한 외삽)를 막는다.
        def _fit_line(pts_xy):
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

        dash_fit = _fit_line(dash_all)
        solid_fit = _fit_line(solid_all)

        # ── Pass 2: 목표 계산 (가상 양선 우선, 불가 시 폭 offset 폴백) + 그리기 ──
        pts = []          # (cy, target_cx, conf)
        for cy, cx_L, cx_R, lx, rx, kind in bands:
            # 실측 우선, 없으면 그 선의 피팅으로 예측
            d = cx_L if cx_L is not None else (dash_fit(cy) if dash_fit else None)
            s = cx_R if cx_R is not None else (solid_fit(cy) if solid_fit else None)
            if d is not None and s is not None and d < s:
                target = (d + s) / 2                # 가상 양선: 폭 상수 불필요
                if cx_L is not None and cx_R is not None:
                    conf = 1.0                       # 둘 다 실측
                elif cx_L is not None or cx_R is not None:
                    conf = 0.85                      # 한쪽 실측 + 한쪽 예측
                else:
                    conf = 0.7                       # 둘 다 예측
            elif cx_R is not None:                   # 실선만 + 점선 정보 없음
                target = cx_R - half * f_solid       # 폭 offset(cosθ 보정) 폴백
                conf = 0.6
            elif cx_L is not None:                   # 점선만 + 실선 정보 없음
                target = cx_L + half * f_dash
                conf = 0.6
            else:                                    # est: 팬텀 중점
                target = (lx + rx) / 2
                conf = 0.3
            pts.append((cy, target, conf))
            # 실측 점(꽉 찬 원) / 예측 점(작은 원)로 구분 표시
            if cx_L is not None:
                cv2.circle(warped_dbg, (int(cx_L), cy), 5, COLOR_DASH, -1)
            elif d is not None:
                cv2.circle(warped_dbg, (int(d), cy), 3, COLOR_DASH, 1)
            if cx_R is not None:
                cv2.circle(warped_dbg, (int(cx_R), cy), 5, COLOR_SOLID, -1)
            elif s is not None:
                cv2.circle(warped_dbg, (int(s), cy), 3, COLOR_SOLID, 1)
            cv2.circle(warped_dbg, (int(target), cy), 7, COLOR_TARGET, -1)

        # 다음 프레임 추적 기준은 "차 바로 앞(최하단) 밴드" 값으로 저장.
        # (상단 밴드 값으로 저장하면 곡선에서 다음 프레임 하단 탐색이 어긋나
        #  탐색창 밖으로 선을 놓치고, 한 번 놓치면 못 되찾는 문제가 있었음)
        self.prev_left_x, self.prev_right_x = bot_left, bot_right
        if bot_width is not None:
            self.prev_width = bot_width

        # 한쪽 선을 오래 못 보면 그쪽 추적을 리셋 → 초기 탐색으로 재획득
        self.miss_left = 0 if left_real else self.miss_left + 1
        self.miss_right = 0 if right_real else self.miss_right + 1
        if self.miss_left >= self.reacquire_frames:
            self.prev_left_x = None
        if self.miss_right >= self.reacquire_frames:
            self.prev_right_x = None

        # ── 룩어헤드 가중 평균 (위쪽=먼 곳 가중 ↑ → 코너 선진입) ──
        # 룩어헤드 부스트(최대 5배)는 "양쪽 선이 다 보이는" 밴드에만 준다.
        # 한쪽 선 기반 추정 밴드는 곡선에서 그 선 쪽으로 치우치는데,
        # 여기에 5배 가중까지 주면 정확한 가까운 밴드들이 평균에서 묻혀서
        # 차가 보이는 차선 쪽으로 붙는 문제가 있었음.
        # conf: 양선 1.0(+룩어헤드) / 한쪽 추정 0.6 / 무검출 추정 0.3
        #
        # ── 비대칭 룩어헤드 (진입엔 풀, 탈출엔 축소) ──
        # 룩어헤드는 먼(위쪽) 밴드를 크게 가중해 코너에 미리 진입하게 한다. 그런데
        # 코너 탈출 국면엔 가까운 길은 이미 직선인데 먼 밴드엔 코너 꼬리가 남아,
        # 목표선이 계속 코너 쪽으로 물려 오버슈트("탈출 관성")가 난다.
        # 그래서: 가까운(하단) 밴드로 잰 "현재 횡오프셋"이 중앙 쪽으로 줄어드는
        # 중이면(=복귀/탈출) 룩어헤드를 lookahead_exit_scale 배로 줄여 목표선이
        # 가까운 직선을 빨리 따라가게 한다. 진입/직선/코너 유지 중엔 풀 룩어헤드.
        la_gain = float(self.cfg.get("lookahead_gain", 4.0))
        la_exit = float(self.cfg.get("lookahead_exit_scale", 0.4))
        near_t = [t for cy, t, cf in pts if cy >= H * 0.6]   # 하단(가까운) 밴드
        cur_off = abs(sum(near_t) / len(near_t) - center_x) if near_t else 0.0
        exiting = cur_off < self.prev_near_off - 1.0         # 1px↑ 중앙 복귀 중
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

        # ── 곡률 피드포워드: "점들이 가리키는 방향"을 조향에 직접 반영 ──
        # P제어는 오차가 있어야만 꺾으므로, 위치 오차(line_x)만 주면 커브에서
        # 바깥 선에 붙은 상태가 평형이 된다. 먼 점들과 가까운 점들의 가로 차이
        # (=경로 굽음)를 더해서 밀려나기 전에 미리 꺾게 한다.
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
                     (255, 255, 0), 2)   # 하늘색: 경로 방향(피드포워드)

        # ── 헤딩 항 (Stanley식): 차선 대비 차체의 틀어짐 보정 ──
        # 밴드 점들에 가중 직선 피팅 → 기울기 = 경로가 화면 수직축과 이루는 방향.
        # 차가 중앙에 있어도 삐딱하게 서 있으면 미리 반대 조향 → 코너 진입/탈출 부드러움.
        heading_gain = float(self.cfg.get("heading_gain", 0.5))
        heading = 0.0
        if len(pts) >= 2:
            ys = np.array([p[0] for p in pts], dtype=float)
            xs = np.array([p[1] for p in pts], dtype=float)
            ws = np.array([p[2] for p in pts], dtype=float)
            if np.ptp(ys) > 1:
                coef = np.polyfit(ys, xs, 1, w=ws)
                slope = float(coef[0])      # d(x)/d(y): +면 경로가 왼쪽을 향함
                heading = -slope            # 왼쪽 향함 → err 음수(좌조향) 기여
                # 시각화(자홍): 차 앞에서 피팅 방향으로 뻗는 헤딩 선
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
                 COLOR_TARGET, 3)
        cv2.line(warped_dbg, (center_x, 0), (center_x, H), (0, 200, 200), 2)
        cv2.putText(warped_dbg,
                    f"L=dash R=solid  bands {found}/{n}  "
                    f"miss L{self.miss_left} R{self.miss_right}  "
                    f"LA={'EXIT' if exiting else 'full'}({eff_la:.1f})",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)
        # 측정 차로폭 표시 — 직선 양선 구간의 meas 값을 LaneW px(고정 offset)에
        # 넣으면 점선만/실선만 두 구간 모두 중앙을 정확히 잡는다. 커브의 meas는
        # 팽창하므로 무시하고 "직선에서의 최소 meas"를 기준으로.
        meas_w = int(self.prev_width) if self.prev_width else 0
        cv2.putText(warped_dbg,
                    f"laneW meas={meas_w}(dir)  fixed={lane_w_fixed}  "
                    f"cos-corr D{f_dash:.2f} S{f_solid:.2f}",
                    (8, H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 120), 1)

        return err_norm, found, dbg, warped_dbg
