#!/usr/bin/env python3
"""
main_obstacle.py -- 장애물+신호등 미션: 초음파 감지 → 차선변경 → 복귀

  카메라 → YOLO 차선검출(차선전환 지원) → PD제어 → 시리얼(Mega)
  Mega → 초음파 6개 텔레메트리("U<idx>:<cm>") → 상태머신

상태머신:
  KEEP(바깥 2차선) ──전방<임계 N연속──▶ CHG_OUT(좌조향 t1초)
      ▲                                     │
      │                               CHG_ALIGN(카운터 t2초)
      │                                     ▼
  RET_ALIGN ◀── RET_OUT ◀──우측 통과확인── KEEP(안쪽 1차선)

차선변경 자체는 오픈루프(고정 조향+시간): 점선을 가로지르는 동안은 차선
추적이 무의미하므로, 건너간 뒤 detector.set_lane()으로 좌우 경계 클래스를
스왑하고 추적을 리셋해 새 차선에서 비전 주행을 재개한다.

실행:
  python3 main_obstacle.py             # 실차
  python3 main_obstacle.py --dry       # 시리얼 없이 (키보드 시뮬로 화면 테스트)
  python3 main_obstacle.py --steeronly # 조향 전용 테스트: 뒷바퀴 정지 상태로
                                       # 초음파/차선변경 기동이 조향으로만 재생됨
                                       # (차 거치대에 올려두고 손으로 센서 가려보기)
  python3 main_obstacle.py --cam 1

키:
  s=출발  x/Space=정지  q=종료  w=설정저장  d=디버그  t=튠모드
  o=수동 차선변경(장애물 감지 시뮬)  p=수동 복귀(통과 시뮬)
  ESC=비상정지(전역)  F12=종료(전역)
"""
import os
import sys
import time

import cv2
import numpy as np

from config import load, save
from lane_model import ModelLaneDetector
from control import LaneFollowController
from serial_driver import MegaLink
from undistort import Undistorter

# camera_intrinsic 폴더: self_drive/camera_intrinsic (이 파일 기준 ../)
INTRINSIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "camera_intrinsic")

# ── 전역 비상정지 ──
try:
    from pynput import keyboard as _kb
    _ESTOP = {"stop": False, "quit": False}

    def _on_press(key):
        if key == _kb.Key.esc:
            _ESTOP["stop"] = True
        elif key == _kb.Key.f12:
            _ESTOP["quit"] = True

    _kb.Listener(on_press=_on_press, daemon=True).start()
    print("[estop] ESC=정지, F12=종료")
except ImportError:
    _ESTOP = {"stop": False, "quit": False}
    print("[estop] pynput 없음")


# ══ 장애물 회피 상태머신 ══════════════════════════════════════

class ObstacleFSM:
    KEEP = "KEEP"
    CHG_PRE = "CHG_PRE"       # 정지 상태로 바퀴 먼저 풀락 (조향모터 대기)
    CHG_OUT = "CHG_OUT"       # 안쪽으로 꺾는 중
    CHG_ALIGN = "CHG_ALIGN"   # 카운터 조향으로 정렬 중
    RET_PRE = "RET_PRE"
    RET_OUT = "RET_OUT"       # 바깥으로 꺾는 중
    RET_ALIGN = "RET_ALIGN"

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = self.KEEP
        self.lane = "outer"
        self.t_state = time.time()
        self.front_hits = 0
        self.side_seen = False
        self.side_clear = 0
        self.t_inner = 0.0
        self.t_cooldown = 0.0
        self.front_cm = None   # 표시용
        self.side_cm = None

    def _min_dist(self, us, ids):
        vals = [us[i] for i in ids if i in us]
        return min(vals) if vals else None

    def force_change(self):
        """수동 트리거('o'): 바깥 KEEP에서 즉시 차선변경 시작"""
        if self.state == self.KEEP and self.lane == "outer":
            self._start(self.CHG_PRE)
            return True
        return False

    def force_return(self):
        """수동 트리거('p'): 안쪽 KEEP에서 즉시 복귀 시작"""
        if self.state == self.KEEP and self.lane == "inner":
            self._start(self.RET_PRE)
            return True
        return False

    def _start(self, state):
        self.state = state
        self.t_state = time.time()

    def update(self, now, us, lane_est=None, heading=None):
        """매 프레임 호출. lane_est/heading = 비전 판정 (조건 기반 기동 전환용).
        반환: (steer_override, pwm_override, lane_changed)
          steer_override None이면 평소 비전 주행, 값이 있으면 오픈루프 기동 중.
        기동 단계 종료는 조건 기반:
          꺾기(OUT)   종료 = 비전 차선 판정(lane_est)이 목표 차선으로 플립
          정렬(ALIGN) 종료 = |헤딩| < align_heading_thresh (차선과 평행해짐)
          각 단계 change_t_max 초과 시 타임아웃 폴백 (안전)
        """
        c = self.cfg
        self.front_cm = self._min_dist(us, c.get("us_front_ids", [0, 1]))
        self.side_cm = self._min_dist(us, c.get("us_right_ids", [2]))
        el = now - self.t_state
        mag = float(c.get("change_steer", 0.75))
        t_min = float(c.get("change_t_min", 0.3))
        t_max = float(c.get("change_t_max", 2.5))
        h_th = float(c.get("align_heading_thresh", 0.10))

        if self.state == self.KEEP:
            if self.lane == "outer":
                # 전방 장애물 감지 (쿨다운 중엔 무시)
                if now > self.t_cooldown and self.front_cm is not None:
                    if self.front_cm < float(c.get("obs_trigger_cm", 45)):
                        self.front_hits += 1
                    else:
                        self.front_hits = 0
                    if self.front_hits >= int(c.get("obs_trigger_hits", 3)):
                        self.front_hits = 0
                        self._start(self.CHG_PRE)
            else:   # inner: 복귀 시점 판단
                past_min = now - self.t_inner > float(c.get("min_inner_s", 1.2))
                # 1) 우측 센서가 있으면: 장애물을 봤다가 사라진 것 확인 (우선)
                if self.side_cm is not None:
                    if self.side_cm < float(c.get("side_block_cm", 50)):
                        self.side_seen = True
                        self.side_clear = 0
                    elif self.side_cm > float(c.get("side_clear_cm", 65)):
                        if self.side_seen:
                            self.side_clear += 1
                    if (self.side_seen
                            and self.side_clear >= int(c.get("side_clear_hits", 5))
                            and past_min):
                        self._start(self.RET_OUT)
                # 2) 우측 센서 없음(전방 2개 구성): 시간 기반 복귀
                elif past_min and \
                        now - self.t_inner > float(c.get("inner_hold_s", 3.0)):
                    self._start(self.RET_PRE)
            return None, None, None

        elif self.state == self.CHG_PRE:      # 정지+풀락 대기 (미리 확 꺾기)
            if el > float(c.get("change_t_pre", 0.4)):
                self._start(self.CHG_OUT)
            return +mag, 0, None

        elif self.state == self.CHG_OUT:      # 왼쪽(안쪽)으로 — 비전 플립까지
            arrived = el > t_min and lane_est == "inner"
            if arrived or el > t_max:
                if not arrived:
                    print("[fsm] CHG_OUT 타임아웃 — 비전 플립 미확인, 정렬 진행")
                self._start(self.CHG_ALIGN)
            return +mag, int(c.get("change_pwm", 70)), None

        elif self.state == self.CHG_ALIGN:    # 카운터 — 헤딩 수렴까지
            aligned = (el > 0.15 and heading is not None
                       and abs(heading) < h_th)
            if aligned or el > t_max:
                self.lane = "inner"
                self.side_seen = False
                self.side_clear = 0
                self.t_inner = time.time()
                self._start(self.KEEP)
                return None, None, "inner"
            return -mag * 0.8, int(c.get("change_pwm", 70)), None

        elif self.state == self.RET_PRE:      # 정지+풀락 대기
            if el > float(c.get("change_t_pre", 0.4)):
                self._start(self.RET_OUT)
            return -mag, 0, None

        elif self.state == self.RET_OUT:      # 오른쪽(바깥)으로 — 비전 플립까지
            arrived = el > t_min and lane_est == "outer"
            if arrived or el > t_max:
                if not arrived:
                    print("[fsm] RET_OUT 타임아웃 — 비전 플립 미확인, 정렬 진행")
                self._start(self.RET_ALIGN)
            return -mag, int(c.get("change_pwm", 70)), None

        elif self.state == self.RET_ALIGN:
            aligned = (el > 0.15 and heading is not None
                       and abs(heading) < h_th)
            if aligned or el > t_max:
                self.lane = "outer"
                self.t_cooldown = time.time() + float(c.get("cooldown_s", 1.5))
                self._start(self.KEEP)
                return None, None, "outer"
            return +mag * 0.8, int(c.get("change_pwm", 70)), None

        return None, None, None


# ── UI (lane 프로젝트와 동일 팔레트) ──────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG      = (28, 26, 24)
PANEL   = (46, 42, 38)
OUTLINE = (70, 63, 57)
TXT     = (232, 227, 220)
DIM     = (145, 137, 128)
GREEN   = (105, 205, 95)
AMBER   = (55, 175, 255)
RED     = (80, 80, 245)


def _text(img, s, xy, scale=0.45, color=TXT, thick=1):
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, val, label, flip=False):
    f = -val if flip else val
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), OUTLINE, 1)
    cx = x + w // 2
    mag = min(abs(f), 1.0)
    fill = int(mag * (w // 2 - 2))
    color = GREEN if mag < 0.3 else AMBER if mag < 0.6 else RED
    if f > 0:
        cv2.rectangle(img, (cx + 1, y + 2), (cx + 1 + fill, y + h - 2),
                      color, -1)
    elif f < 0:
        cv2.rectangle(img, (cx - 1 - fill, y + 2), (cx - 1, y + h - 2),
                      color, -1)
    cv2.line(img, (cx, y + 1), (cx, y + h - 1), (105, 98, 90), 1)
    arrow = "R >" if f > 0.05 else "< L" if f < -0.05 else "="
    _text(img, label, (x, y - 6), 0.42, DIM)
    _text(img, f"{val:+.2f}  {arrow}", (x + w + 10, y + h - 5), 0.5, TXT)


def _title(img, text):
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (img.shape[1], 26), BG, -1)
    cv2.addWeighted(ov, 0.6, img, 0.4, 0, dst=img)
    _text(img, text, (10, 18), 0.48, TXT)


def _dashboard(total_w, running, steer, err, lost, fps, pwm, fsm, tune=False,
               steer_only=False):
    H = 168
    d = np.full((H, total_w, 3), BG, dtype=np.uint8)
    cv2.line(d, (0, 0), (total_w, 0), OUTLINE, 1)

    fill = (52, 130, 52) if running else (46, 46, 140)
    edge = GREEN if running else RED
    cv2.rectangle(d, (14, 14), (124, 56), fill, -1)
    cv2.rectangle(d, (14, 14), (124, 56), edge, 1)
    label = "RUN" if running else "STOP"
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.8, 2)
    _text(d, label, (14 + (110 - tw) // 2, 44), 0.8, (255, 255, 255), 2)
    _text(d, f"PWM {pwm}", (20, 82), 0.45, DIM)

    bx = 150
    bw = max(120, total_w - bx - 122)
    _bar(d, bx, 24, bw, 20, steer, "STEER", flip=True)
    if err is not None:
        _bar(d, bx, 68, bw, 20, err, "ERROR")
    else:
        _text(d, f"LANE LOST ({lost})", (bx, 82), 0.55, RED, 2)

    # ── 미션 상태 라인 ──
    cv2.line(d, (0, 98), (total_w, 98), OUTLINE, 1)
    st_c = GREEN if fsm.state == fsm.KEEP else AMBER
    lane_c = TXT if fsm.lane == "outer" else AMBER
    f_cm = "--" if fsm.front_cm is None else str(fsm.front_cm)
    s_cm = "--" if fsm.side_cm is None else str(fsm.side_cm)
    f_c = RED if (fsm.front_cm or 999) < 50 else TXT
    items = [("STATE", fsm.state, st_c, 14), ("LANE", fsm.lane, lane_c, 155),
             ("FRONT", f_cm + "cm", f_c, 285), ("SIDE", s_cm + "cm", TXT, 420)]
    for k, v, c, x in items:
        _text(d, k, (x, 120), 0.42, DIM)
        _text(d, v, (x + 52, 120), 0.42, c)

    cv2.line(d, (0, 130), (total_w, 130), OUTLINE, 1)
    lost_c = RED if lost > 5 else AMBER if lost > 0 else TXT
    _text(d, f"FPS {fps:.0f}   LOST", (14, 150), 0.4, DIM)
    _text(d, str(lost), (152, 150), 0.4, lost_c)
    if tune:
        _text(d, "TUNE (t)", (total_w - 110, 150), 0.45, AMBER, 2)
    if steer_only:
        _text(d, "STEER-ONLY", (total_w - 250, 150), 0.45, AMBER, 2)
    _text(d, "s go  x stop  o change  p return  w save  q quit",
          (14, 164), 0.38, DIM)
    return d


def _settings_panel(cfg):
    w, h = 500, 136
    p = np.full((h, w, 3), BG, dtype=np.uint8)
    _text(p, "Drag sliders  |  w: save -> calib.json", (10, 20), 0.42, DIM)
    cv2.line(p, (0, 28), (w, 28), OUTLINE, 1)
    _text(p, f"OBS  trigger {int(cfg.get('obs_trigger_cm', 45))}cm"
             f"   clear {int(cfg.get('side_clear_cm', 65))}cm"
             f"   cooldown {cfg.get('cooldown_s', 1.5):.1f}s", (10, 50), 0.42)
    _text(p, f"CHG  steer {cfg.get('change_steer', 0.75):.2f}"
             f"   t_max {cfg.get('change_t_max', 2.5):.1f}s"
             f"   alignTh {cfg.get('align_heading_thresh', 0.10):.2f}"
             f"   pwm {int(cfg.get('change_pwm', 70))}", (10, 72), 0.42)
    _text(p, f"CTRL Kp {cfg.get('kp', 0):.1f}  Kd {cfg.get('kd', 0):.1f}"
             f"  HGain {cfg.get('heading_gain', 0.5):.1f}"
             f"  Ctr {int(cfg.get('center_offset', 0)):+d}px", (10, 94), 0.42)
    _text(p, f"DRIVE PWM {int(cfg.get('drive_pwm', 0))}"
             f"   Bands {int(cfg.get('n_bands', 5))}", (10, 118), 0.42)
    return p


# ── 메인 ──────────────────────────────────────────────────────

def main():
    dry = "--dry" in sys.argv
    steer_only = "--steeronly" in sys.argv
    cfg = load()

    if "--cam" in sys.argv:
        cfg["cam_index"] = int(sys.argv[sys.argv.index("--cam") + 1])

    cap = cv2.VideoCapture(int(cfg["cam_index"]), cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg["frame_w"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg["frame_h"]))
    if not cap.isOpened():
        print("카메라 열기 실패. cam_index 확인")
        return

    und = Undistorter(cfg.get("camera_id", cfg["cam_index"]),
                      int(cfg["frame_w"]), int(cfg["frame_h"]), INTRINSIC_DIR,
                      enabled=bool(cfg.get("undistort", 1)))

    det = ModelLaneDetector(cfg)
    ctrl = LaneFollowController(kp=cfg["kp"], kd=cfg.get("kd", 6.0),
                                steer_sign=cfg["steer_sign"],
                                right_gain=cfg.get("steer_right_gain", 1.0))
    link = MegaLink(cfg, dry_run=dry)
    fsm = ObstacleFSM(cfg)

    # ── Settings 트랙바 ──
    SW = "Settings"
    cv2.namedWindow(SW, cv2.WINDOW_AUTOSIZE)
    nop = lambda v: None
    cv2.createTrackbar("ObsTrig cm", SW,
                       int(cfg.get("obs_trigger_cm", 45)), 150, nop)
    cv2.createTrackbar("ChgSteer x100", SW,
                       int(float(cfg.get("change_steer", 0.75)) * 100), 100, nop)
    cv2.createTrackbar("Tmax x10", SW,
                       int(float(cfg.get("change_t_max", 2.5)) * 10), 50, nop)
    cv2.createTrackbar("AlignTh x100", SW,
                       int(float(cfg.get("align_heading_thresh", 0.10)) * 100),
                       40, nop)
    cv2.createTrackbar("Center+100", SW,
                       int(cfg.get("center_offset", 0)) + 100, 200, nop)
    cv2.createTrackbar("Kp x10", SW,
                       int(float(cfg.get("kp", 1.4)) * 10), 50, nop)
    cv2.createTrackbar("Kd x10", SW,
                       int(float(cfg.get("kd", 6.0)) * 10), 300, nop)
    cv2.createTrackbar("HGain x10", SW,
                       int(float(cfg.get("heading_gain", 0.5)) * 10), 30, nop)
    cv2.createTrackbar("Drive PWM", SW,
                       int(cfg.get("drive_pwm", 70)), 255, nop)

    cv2.imshow(SW, _settings_panel(cfg))

    running = False
    show_debug = True
    tune_mode = False
    last_steer = 0.0
    cur_pwm = 0.0
    lost_frames = 0
    last_us_log = 0.0
    t0, frames = time.time(), 0
    VW, VH = 640, 360

    print(f"[obstacle] cam={cfg['cam_index']} 준비 완료! "
          f"(o/p 키로 차선변경 수동 트리거)")
    if steer_only:
        print("[obstacle] ★ STEER-ONLY 모드: 뒷바퀴 정지, 조향만 동작 ★")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = und(frame)   # 렌즈 왜곡 보정 (비활성 시 원본 그대로)

            try:
                cfg["obs_trigger_cm"] = cv2.getTrackbarPos("ObsTrig cm", SW)
                cfg["change_steer"] = max(10, cv2.getTrackbarPos(
                    "ChgSteer x100", SW)) / 100.0
                cfg["change_t_max"] = max(5, cv2.getTrackbarPos(
                    "Tmax x10", SW)) / 10.0
                cfg["align_heading_thresh"] = max(2, cv2.getTrackbarPos(
                    "AlignTh x100", SW)) / 100.0
                cfg["center_offset"] = cv2.getTrackbarPos(
                    "Center+100", SW) - 100
                kp = max(1, cv2.getTrackbarPos("Kp x10", SW)) / 10.0
                cfg["kp"] = kp
                ctrl.kp = kp
                kd = cv2.getTrackbarPos("Kd x10", SW) / 10.0
                cfg["kd"] = kd
                ctrl.kd = kd
                cfg["heading_gain"] = cv2.getTrackbarPos("HGain x10", SW) / 10.0
                cfg["drive_pwm"] = max(0, cv2.getTrackbarPos("Drive PWM", SW))
            except cv2.error:
                pass
            det.cfg = cfg
            fsm.cfg = cfg

            # ── 초음파 텔레메트리 + 상태머신 ──
            us = link.read_telemetry()

            # 초음파 수신 확인용 주기 출력 (us_log_s 간격, 0이면 끔)
            log_s = float(cfg.get("us_log_s", 1.0))
            if log_s > 0 and time.time() - last_us_log >= log_s:
                last_us_log = time.time()
                if us:
                    pot_s = f"  pot:{link.pot}" if link.pot is not None else ""
                    print("[us] " + "  ".join(
                        f"U{i}:{us[i]}cm" for i in sorted(us)) + pot_s)
                elif not dry:
                    print("[us] 수신 없음 — 펌웨어 업로드/배선/포트 확인")
            # ── 차선 검출 먼저 (경계 클래스로 현재 차선 자동 판정) ──
            err, found, dbg_orig, dbg_bev = det.process(
                frame, reuse_masks=tune_mode)

            # ── 상태머신 (비전 판정을 조건으로 기동 단계 전환) ──
            steer_ov, pwm_ov, lane_changed = (None, None, None)
            if running:   # 정지 중엔 상태머신 동결 (오발동 방지)
                steer_ov, pwm_ov, lane_changed = fsm.update(
                    time.time(), us,
                    lane_est=det.lane_est, heading=det.last_heading)
            if lane_changed:
                print(f"[obstacle] 기동 완료 -> {lane_changed} "
                      f"(비전 플립+헤딩 수렴 확인됨)")

            # 비전 판정을 상태머신에 동기화 (KEEP 상태에서만 — 기동 중엔
            # 점선을 가로지르는 중이라 판정이 흔들리는 게 정상)
            if fsm.state == fsm.KEEP and det.lane_est in ("outer", "inner"):
                if fsm.lane != det.lane_est:
                    print(f"[obstacle] 비전 차선 판정: {fsm.lane} -> {det.lane_est}")
                    if det.lane_est == "inner":
                        fsm.t_inner = time.time()   # 안쪽 도착 시점 재기준
                fsm.lane = det.lane_est

            # ── 조향 결정 (오픈루프 기동 중엔 override) ──
            if steer_ov is not None:
                steer = steer_ov
                last_steer = steer
                lost_frames = 0   # 기동 중 로스트 카운트 정지
            elif err is not None:
                steer = ctrl.compute(err)
                last_steer = steer
                lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames <= cfg["lost_hold_frames"]:
                    steer = last_steer
                else:
                    steer = last_steer * 0.5

            # ── 구동 (soft-brake 램프 포함) ──
            if steer_only:
                # 조향 전용 테스트: 뒷바퀴 항상 정지, 조향만 실시간 전송
                pwm = 0
                cur_pwm = 0.0
                if running:
                    link.send_drive(0, steer)   # 모터 0x00 + 조향 바이트
                else:
                    link.send_brake()
            else:
                if running and (steer_ov is not None
                                or lost_frames <= cfg["lost_stop_frames"]):
                    if pwm_ov is not None:
                        target_pwm = pwm_ov
                        cur_pwm = float(pwm_ov)   # 기동 중엔 램프 없이 정확히
                    else:
                        target_pwm = int(cfg["drive_pwm"])
                        if abs(steer) > cfg["slow_steer_thresh"]:
                            target_pwm = int(cfg["slow_pwm"])
                else:
                    target_pwm = 0
                    if running:
                        cur_pwm = 0.0   # 로스트 정지는 즉시

                ramp = float(cfg.get("brake_ramp_pwm", 6))
                if target_pwm < cur_pwm:
                    cur_pwm = max(target_pwm, cur_pwm - ramp)
                else:
                    cur_pwm = float(target_pwm)

                pwm = int(round(cur_pwm))
                if pwm > 0:
                    link.send_drive(pwm, steer)
                elif running and steer_ov is not None:
                    # PRE 단계: 모터만 정지, 조향은 계속 전송 (바퀴 미리 꺾기)
                    link.send_drive(0, steer)
                else:
                    link.send_brake()

            # ── 비상정지 ──
            if _ESTOP["stop"]:
                _ESTOP["stop"] = False
                running = False
                cur_pwm = 0.0
                link.send_brake()
                print("[estop] 비상정지!")
            if _ESTOP["quit"]:
                break

            # ── 디스플레이 ──
            frames += 1
            if show_debug:
                fps = frames / max(1e-3, time.time() - t0)
                v_orig = dbg_orig if dbg_orig.shape[1::-1] == (VW, VH) \
                    else cv2.resize(dbg_orig, (VW, VH))
                v_bev = dbg_bev if dbg_bev.shape[1::-1] == (VW, VH) \
                    else cv2.resize(dbg_bev, (VW, VH))
                _title(v_orig, "CAMERA + MASK")
                _title(v_bev, f"BIRD'S EYE VIEW  [{det.lane_est}]")
                # 카메라 뷰 타이틀 우측에 센서 원시값 상시 표시
                if us:
                    _text(v_orig,
                          "  ".join(f"U{i} {us[i]}" for i in sorted(us)),
                          (VW - 165, 18), 0.42, AMBER)
                else:
                    _text(v_orig, "US: no data", (VW - 120, 18), 0.42, RED)
                views = np.vstack((v_orig, v_bev))
                dash = _dashboard(VW, running, steer, err, lost_frames,
                                  fps, pwm, fsm, tune=tune_mode,
                                  steer_only=steer_only)
                display = np.vstack((views, dash))
                cv2.imshow("Obstacle Mission", display)

            if frames % 3 == 0:
                cv2.imshow(SW, _settings_panel(cfg))

            # ── 키 입력 ──
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                running = True
                t0, frames = time.time(), 0
                print("[obstacle] GO")
            elif k in (ord('x'), ord(' ')):
                running = False
                print("[obstacle] STOP")
            elif k == ord('o'):
                if fsm.force_change():
                    print("[obstacle] 수동: 차선변경 시작")
            elif k == ord('p'):
                if fsm.force_return():
                    print("[obstacle] 수동: 복귀 시작")
            elif k == ord('d'):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow("Obstacle Mission")
            elif k == ord('t'):
                tune_mode = not tune_mode
                if tune_mode:
                    running = False
                    cur_pwm = 0.0
                    link.send_brake()
                print(f"[obstacle] TUNE {'ON' if tune_mode else 'OFF'}")
            elif k == ord('w'):
                save(cfg)
                print("[obstacle] 설정 저장 완료")
    finally:
        link.send_brake()
        link.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[obstacle] 종료")


if __name__ == "__main__":
    main()
