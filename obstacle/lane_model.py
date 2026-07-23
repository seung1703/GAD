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

# 디버그 색 (BGR): dash=주황, solid=파랑
COLOR_DASH = (0, 165, 255)
COLOR_SOLID = (255, 80, 0)

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
        # 한쪽 선을 연속으로 못 본 프레임 수 → 임계 초과 시 그쪽 추적 리셋
        # (리셋해야 "중앙 기준 최근접" 초기 탐색이 다시 발동해 재획득 가능)
        self.miss_left = 0
        self.miss_right = 0
        self.reacquire_frames = int(cfg.get("reacquire_frames", 8))
        self.infer_ms = 0.0
        self._mask_cache = None   # (dash, solid) — 튠 모드 재사용용

        # ── 비전 기반 차선 자동 판정 ──
        # 좌우 경계는 클래스 무관하게 "가장 가까운 선"으로 잡고, 잡힌 경계가
        # dash인지 solid인지로 현재 차선을 추정한다:
        #   왼쪽 경계=solid → inner(1차선, 인덱스0) / 오른쪽 경계=solid → outer(2차선, 인덱스1)
        # 연속 lane_switch_frames 프레임 일치해야 전환 (점선 끊김/변경 중 깜빡임 방지)
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
        lane_w = self.prev_width if self.prev_width else int(W * 0.6)

        pts = []          # (cy, target_cx, conf)
        found = 0         # 실검출 밴드 수
        bot_left = bot_right = None   # 최하단 유효 밴드의 좌/우 (다음 프레임 기준)
        bot_width = None              # 최하단 양선 밴드의 차로폭 (왜곡 최소)
        left_real = right_real = False
        # 클래스 무관 결합 마스크에서 좌우 최근접 경계를 잡고,
        # 각 경계의 클래스(dash/solid)는 사후 분류 → 차선 자동 판정에 사용
        w_any = cv2.bitwise_or(w_dash, w_solid)
        bot_cls_L = bot_cls_R = None   # 최하단 실검출 경계의 클래스

        for b in range(n - 1, -1, -1):   # 차 바로 앞(하단)부터
            by0, by1 = b * band_h, (b + 1) * band_h
            cy = (by0 + by1) // 2
            h_any = np.sum(w_any[by0:by1, :], axis=0)
            h_d = np.sum(w_dash[by0:by1, :], axis=0)
            h_s = np.sum(w_solid[by0:by1, :], axis=0)

            def _cls(x):
                """경계 x 주변 ±12px에서 dash/solid 픽셀 다수결"""
                a, z = int(max(0, x - 12)), int(min(W, x + 12))
                return "solid" if h_s[a:z].sum() > h_d[a:z].sum() else "dash"

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
            if bot_cls_L is None and cls_L is not None:
                bot_cls_L = cls_L
            if bot_cls_R is None and cls_R is not None:
                bot_cls_R = cls_R
            col_left = COLOR_SOLID if cls_L == "solid" else COLOR_DASH
            col_right = COLOR_SOLID if cls_R == "solid" else COLOR_DASH

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
                target = (cx_L + cx_R) / 2
                conf = 1.0
                found += 1
            elif cx_R is not None:          # 실선만 (점선 끊긴 구간)
                right_x = cx_R
                left_x = cx_R - lane_w
                target = cx_R - lane_w / 2
                conf = 0.6                   # 한쪽 추정: 가중치 낮춤
                found += 1
            elif cx_L is not None:          # 점선만
                left_x = cx_L
                right_x = cx_L + lane_w
                target = cx_L + lane_w / 2
                conf = 0.6
                found += 1
            else:
                if left_x is None or right_x is None:
                    continue                # 아직 아무 기준 없음: 밴드 스킵
                target = (left_x + right_x) / 2  # 추정 (found에 안 셈)
                conf = 0.3

            if cx_L is not None:
                left_real = True
            if cx_R is not None:
                right_real = True
            if bot_left is None:             # 첫(=최하단) 유효 밴드 기록
                bot_left, bot_right = left_x, right_x

            pts.append((cy, target, conf))
            if cx_L is not None:
                cv2.circle(warped_dbg, (int(cx_L), cy), 5, col_left, -1)
            if cx_R is not None:
                cv2.circle(warped_dbg, (int(cx_R), cy), 5, col_right, -1)
            cv2.circle(warped_dbg, (int(target), cy), 5, (0, 255, 255), -1)

        # ── 미검출 처리: 실검출 0밴드면 err=None (main lost 로직 발동) ──
        if found == 0:
            self.prev_left_x = self.prev_right_x = None  # 추적 리셋
            self.miss_left = self.miss_right = 0
            self.last_heading = None   # 헤딩 정보 없음 (FSM 정렬 판단용)
            cv2.putText(warped_dbg, "NO LANE", (W // 2 - 70, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return None, 0, dbg, warped_dbg

        # 다음 프레임 추적 기준은 "차 바로 앞(최하단) 밴드" 값으로 저장.
        # (상단 밴드 값으로 저장하면 곡선에서 다음 프레임 하단 탐색이 어긋나
        #  탐색창 밖으로 선을 놓치고, 한 번 놓치면 못 되찾는 문제가 있었음)
        self.prev_left_x, self.prev_right_x = bot_left, bot_right
        if bot_width is not None:
            self.prev_width = bot_width

        # ── 차선 자동 판정: 최하단 경계 클래스 기준 + 연속 프레임 다수결 ──
        # 왼쪽=solid → inner(1차선) / 오른쪽=solid → outer(2차선)
        # (solid 판정을 dash보다 우선: 점선은 끊겨서 미검출이 잦으므로)
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

        self.last_heading = heading   # FSM(정렬 단계 종료 판단)에서 참조

        curve_gain = float(self.cfg.get("curve_gain", 0.5))
        pos_err = (line_x - center_x) / (W / 2.0)
        err_norm = float(np.clip(pos_err + curve_gain * curve
                                 + heading_gain * heading, -1, 1))

        cv2.line(warped_dbg, (int(line_x), 0), (int(line_x), H),
                 (0, 255, 255), 2)
        cv2.line(warped_dbg, (center_x, 0), (center_x, H), (0, 200, 200), 2)
        cv2.putText(warped_dbg,
                    f"[{self.lane_est.upper()}] L={bot_cls_L} R={bot_cls_R}  "
                    f"bands {found}/{n}  "
                    f"miss L{self.miss_left} R{self.miss_right}",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)

        return err_norm, found, dbg, warped_dbg
