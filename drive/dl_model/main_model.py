#!/usr/bin/env python3
"""
main_model.py -- AI 모델 차선추종 + ROI 실시간 설정 + 대시보드

  카메라 → YOLO 차선 검출 → P제어 → 시리얼(Mega)

실행:
  python3 main_model.py            # 실제 주행
  python3 main_model.py --dry      # 시리얼 없이 비전만
  python3 main_model.py --cam 1    # 카메라 지정
  python3 main_model.py --record   # 주행하며 데이터셋 녹화
                                   # (records/run_*/ 에 원본 JPG + labels.csv)

키:
  s=출발  x/Space=정지  q=종료  w=설정저장  d=디버그토글
  ESC=비상정지(전역)  F12=종료(전역)
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "shared"))

from config import load, save
from lane_model import ModelLaneDetector
from control import LaneFollowController
from serial_driver import MegaLink
from undistort import Undistorter

# camera_intrinsic 폴더: self_drive/camera_intrinsic (이 파일 기준 ../../)
INTRINSIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "camera_intrinsic")

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


# ── UI 팔레트 (BGR) ──────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG      = (28, 26, 24)     # 배경
PANEL   = (46, 42, 38)     # 바/패널
OUTLINE = (70, 63, 57)     # 테두리
TXT     = (232, 227, 220)  # 본문
DIM     = (145, 137, 128)  # 보조 텍스트
GREEN   = (105, 205, 95)
AMBER   = (55, 175, 255)
RED     = (80, 80, 245)


def _text(img, s, xy, scale=0.45, color=TXT, thick=1):
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, val, label, flip=False):
    """중앙 기준 수평 바 (-1~1). flip=True면 화면 방향=실제 진행 방향으로 뒤집음."""
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


# ── 방향키 미세조정 ────────────────────────────────────────────
# 위/아래로 트랙바를 고르고, 좌/우로 1스텝씩 움직인다. 1스텝은 트랙바
# 해상도 그대로라 IPM은 0.01, Center는 1px, 게인은 0.1 단위가 된다.
# 순서 = 튜닝하는 순서(ROI 먼저).
TUNE_BARS = ["TL x+100", "TL y%", "TR x+100", "TR y%",
             "BL x+100", "BL y%", "BR x+100", "BR y%",
             "Center+100", "Bands", "Kp x10", "Kd x10",
             "RGain x10", "HGain x10", "Drive PWM", "LaneW px"]

# 방향키 코드는 OS/백엔드마다 다르다 (waitKeyEx 원값: macOS, GTK, Windows).
# ★ & 0xFF 로 마스킹하면 안 된다 — GTK의 Left(65361)가 'q'(113)와 겹쳐서
#   왼쪽 방향키를 누르는 순간 프로그램이 종료된다.
_ARROW_UP    = (63232, 65362, 2490368)
_ARROW_DOWN  = (63233, 65364, 2621440)
_ARROW_LEFT  = (63234, 65361, 2424832)
_ARROW_RIGHT = (63235, 65363, 2555904)


def _nudge_bar(win, bar, delta):
    """선택된 트랙바를 delta(±1)만큼 이동. 상한은 OpenCV가 알아서 자른다."""
    try:
        cv2.setTrackbarPos(bar, win,
                           max(0, cv2.getTrackbarPos(bar, win) + delta))
    except cv2.error:
        pass


def _title(img, text):
    """뷰 상단 반투명 타이틀 스트립"""
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (img.shape[1], 26), BG, -1)
    cv2.addWeighted(ov, 0.6, img, 0.4, 0, dst=img)
    _text(img, text, (10, 18), 0.48, TXT)


def _dashboard(total_w, running, steer, err, found, lost, fps, pwm, offset,
               tune=False):
    """상태 대시보드 (세로 배치용 컴팩트 레이아웃, 폭 640 기준)"""
    H = 146
    d = np.full((H, total_w, 3), BG, dtype=np.uint8)
    cv2.line(d, (0, 0), (total_w, 0), OUTLINE, 1)

    # ── 상태 배지 ──
    fill = (52, 130, 52) if running else (46, 46, 140)
    edge = GREEN if running else RED
    cv2.rectangle(d, (14, 14), (124, 56), fill, -1)
    cv2.rectangle(d, (14, 14), (124, 56), edge, 1)
    label = "RUN" if running else "STOP"
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.8, 2)
    _text(d, label, (14 + (110 - tw) // 2, 44), 0.8, (255, 255, 255), 2)
    _text(d, f"PWM {pwm}", (20, 82), 0.45, DIM)

    # ── 조향/오차 바 (화면 오른쪽 = 실제 오른쪽) ──
    bx = 150
    bw = max(120, total_w - bx - 122)   # 오른쪽 값 텍스트 공간 확보
    _bar(d, bx, 24, bw, 20, steer, "STEER", flip=True)   # 조향 +는 왼쪽
    if err is not None:
        _bar(d, bx, 68, bw, 20, err, "ERROR")            # 오차 +는 오른쪽
    else:
        _text(d, f"LANE LOST ({lost})", (bx, 82), 0.55, RED, 2)

    # ── 하단 정보 라인 ──
    cv2.line(d, (0, 98), (total_w, 98), OUTLINE, 1)
    lost_c = RED if lost > 5 else AMBER if lost > 0 else TXT
    items = [("FPS", f"{fps:.0f}", TXT, 14), ("BANDS", str(found), TXT, 110),
             ("LOST", str(lost), lost_c, 215),
             ("OFS", f"{offset:+d}px", TXT, 315)]
    for k, v, c, x in items:
        _text(d, k, (x, 120), 0.42, DIM)
        _text(d, v, (x + 46, 120), 0.42, c)
    if tune:
        _text(d, "TUNE MODE (t)", (total_w - 175, 120), 0.5, AMBER, 2)

    _text(d, "s go   x stop   t tune   w save   d debug   q quit",
          (14, 140), 0.4, DIM)
    return d


def _settings_panel(cfg, sel_bar=None, win=None):
    """Settings 창 하단: 현재 적용값 + 방향키로 선택 중인 트랙바 표시"""
    w, h = 480, 160
    p = np.full((h, w, 3), BG, dtype=np.uint8)
    _text(p, "Sliders or arrow keys  |  w: save -> calib.json",
          (10, 20), 0.42, DIM)
    cv2.line(p, (0, 28), (w, 28), OUTLINE, 1)
    _text(p, f"IPM   TL {cfg.get('ipm_tl_x', 0):+.2f},{cfg.get('ipm_tl_y', 0):.2f}"
             f"   TR {cfg.get('ipm_tr_x', 0):+.2f},{cfg.get('ipm_tr_y', 0):.2f}",
          (10, 50), 0.42)
    _text(p, f"      BL {cfg.get('ipm_bl_x', 0):+.2f},{cfg.get('ipm_bl_y', 0):.2f}"
             f"   BR {cfg.get('ipm_br_x', 0):+.2f},{cfg.get('ipm_br_y', 0):.2f}",
          (10, 70), 0.42)
    _text(p, f"CTRL  Kp {cfg.get('kp', 0):.1f}   Kd {cfg.get('kd', 0):.1f}"
             f"   RGain {cfg.get('steer_right_gain', 1.0):.1f}"
             f"   HGain {cfg.get('heading_gain', 0.5):.1f}"
             f"   Ctr {int(cfg.get('center_offset', 0)):+d}px",
          (10, 94), 0.42)
    _text(p, f"DRIVE PWM {int(cfg.get('drive_pwm', 0))}"
             f"   Bands {int(cfg.get('n_bands', 5))}", (10, 118), 0.42)

    # ── 방향키 선택 항목 ──
    cv2.line(p, (0, 128), (w, 128), OUTLINE, 1)
    if sel_bar is None:
        _text(p, "^ v pick bar   < > step +-1   (or n/m , .)",
              (10, 148), 0.4, DIM)
    else:
        name = TUNE_BARS[sel_bar]
        pos = ""
        if win is not None:
            try:
                pos = f" = {cv2.getTrackbarPos(name, win)}"
            except cv2.error:
                pass
        _text(p, "SEL", (10, 148), 0.42, DIM)
        _text(p, f"{name}{pos}", (46, 148), 0.45, AMBER, 1)
        _text(p, f"[{sel_bar + 1}/{len(TUNE_BARS)}]  ^v pick  <> +-1",
              (250, 148), 0.4, DIM)
    return p


# ── 메인 ──────────────────────────────────────────────────────

def main():
    dry = "--dry" in sys.argv
    cfg = load()

    if "--cam" in sys.argv:
        cfg["cam_index"] = int(sys.argv[sys.argv.index("--cam") + 1])

    # ── --record : 주행 중 원본 프레임을 데이터셋용으로 저장 ──
    # records/run_날짜시각/ 에 오버레이 없는 JPG + labels.csv(err,steer,pwm)
    record = "--record" in sys.argv
    rec_dir, rec_csv, rec_count = None, None, 0
    if record:
        rec_dir = time.strftime("records/run_%Y%m%d_%H%M%S")
        os.makedirs(rec_dir, exist_ok=True)
        rec_csv = open(os.path.join(rec_dir, "labels.csv"), "w")
        rec_csv.write("file,err,steer,pwm\n")
        print(f"[record] 저장 폴더: {rec_dir} "
              f"(RUN 중에만, {int(cfg.get('record_every', 3))}프레임마다 1장)")

    cap = cv2.VideoCapture(int(cfg["cam_index"]), cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg["frame_w"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg["frame_h"]))
    if not cap.isOpened():
        print("카메라 열기 실패. cam_index 확인")
        return

    # 렌즈 왜곡 보정기 (시작 시 remap맵 1회 생성; 파일 없으면 pass-through)
    und = Undistorter(cfg.get("camera_id", cfg["cam_index"]),
                      int(cfg["frame_w"]), int(cfg["frame_h"]), INTRINSIC_DIR,
                      enabled=bool(cfg.get("undistort", 1)))

    det = ModelLaneDetector(cfg)   # 모델 경로는 lane_model.py 기본값(자기 폴더 기준) 사용
    ctrl = LaneFollowController(kp=cfg["kp"], kd=cfg.get("kd", 6.0),
                                steer_sign=cfg["steer_sign"],
                                right_gain=cfg.get("steer_right_gain", 1.0))
    link = MegaLink(cfg, dry_run=dry)

    # ── Settings 트랙바 창 ──
    SW = "Settings"
    cv2.namedWindow(SW, cv2.WINDOW_AUTOSIZE)
    nop = lambda v: None
    # ── IPM 사다리꼴 네 모서리 (x는 +100 오프셋: 0~300 → -100%~200%) ──
    cv2.createTrackbar("TL x+100", SW,
                       int(float(cfg.get("ipm_tl_x", 0.25)) * 100) + 100, 300, nop)
    cv2.createTrackbar("TL y%",    SW,
                       int(float(cfg.get("ipm_tl_y", 0.40)) * 100), 100, nop)
    cv2.createTrackbar("TR x+100", SW,
                       int(float(cfg.get("ipm_tr_x", 0.75)) * 100) + 100, 300, nop)
    cv2.createTrackbar("TR y%",    SW,
                       int(float(cfg.get("ipm_tr_y", 0.40)) * 100), 100, nop)
    cv2.createTrackbar("BL x+100", SW,
                       int(float(cfg.get("ipm_bl_x", -0.40)) * 100) + 100, 300, nop)
    cv2.createTrackbar("BL y%",    SW,
                       int(float(cfg.get("ipm_bl_y", 0.98)) * 100), 100, nop)
    cv2.createTrackbar("BR x+100", SW,
                       int(float(cfg.get("ipm_br_x", 1.40)) * 100) + 100, 300, nop)
    cv2.createTrackbar("BR y%",    SW,
                       int(float(cfg.get("ipm_br_y", 0.98)) * 100), 100, nop)
    cv2.createTrackbar("Center+100", SW,
                       int(cfg.get("center_offset", 0)) + 100, 200, nop)
    cv2.createTrackbar("Bands",      SW,
                       int(cfg.get("n_bands", 5)), 10, nop)
    cv2.createTrackbar("Kp x10",     SW,
                       int(float(cfg.get("kp", 1.4)) * 10), 50, nop)
    cv2.createTrackbar("Kd x10",     SW,
                       int(float(cfg.get("kd", 6.0)) * 10), 300, nop)
    cv2.createTrackbar("RGain x10",  SW,
                       int(float(cfg.get("steer_right_gain", 1.0)) * 10), 30, nop)
    cv2.createTrackbar("HGain x10",  SW,
                       int(float(cfg.get("heading_gain", 0.5)) * 10), 30, nop)
    cv2.createTrackbar("Drive PWM",  SW,
                       int(cfg.get("drive_pwm", 70)), 255, nop)
    cv2.createTrackbar("LaneW px",   SW,
                       int(cfg.get("bev_lane_width", 384)), 640, nop)

    sel_bar = 0          # 방향키로 선택 중인 트랙바 인덱스
    cv2.imshow(SW, _settings_panel(cfg, sel_bar, SW))

    running = False
    show_debug = True
    tune_mode = False   # 't': 추론 일시정지 + 마지막 마스크 재사용 (트랙바 반응속도 ↑)
    last_steer = 0.0
    cur_pwm = 0.0       # 현재 출력 PWM (정지 시 램프 감속용)
    lost_frames = 0
    t0, frames = time.time(), 0
    VW, VH = 640, 360

    print(f"[main_model] cam={cfg['cam_index']} 준비 완료!")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = und(frame)   # 렌즈 왜곡 보정 (비활성 시 원본 그대로)

            # ── 트랙바에서 설정 읽기 ──
            try:
                for key, bar in (("ipm_tl_x", "TL x+100"),
                                 ("ipm_tr_x", "TR x+100"),
                                 ("ipm_bl_x", "BL x+100"),
                                 ("ipm_br_x", "BR x+100")):
                    cfg[key] = (cv2.getTrackbarPos(bar, SW) - 100) / 100.0
                for key, bar in (("ipm_tl_y", "TL y%"), ("ipm_tr_y", "TR y%"),
                                 ("ipm_bl_y", "BL y%"), ("ipm_br_y", "BR y%")):
                    cfg[key] = cv2.getTrackbarPos(bar, SW) / 100.0
                cfg["center_offset"] = cv2.getTrackbarPos(
                    "Center+100", SW) - 100
                cfg["n_bands"] = max(2, cv2.getTrackbarPos("Bands", SW))
                kp = max(1, cv2.getTrackbarPos("Kp x10", SW)) / 10.0
                cfg["kp"] = kp
                ctrl.kp = kp
                kd = cv2.getTrackbarPos("Kd x10", SW) / 10.0
                cfg["kd"] = kd
                ctrl.kd = kd
                rg = max(1, cv2.getTrackbarPos("RGain x10", SW)) / 10.0
                cfg["steer_right_gain"] = rg
                ctrl.right_gain = rg
                cfg["heading_gain"] = cv2.getTrackbarPos("HGain x10", SW) / 10.0
                cfg["drive_pwm"] = max(0, cv2.getTrackbarPos(
                    "Drive PWM", SW))
                cfg["bev_lane_width"] = max(1, cv2.getTrackbarPos(
                    "LaneW px", SW))
            except cv2.error:
                pass
            det.cfg = cfg

            # ── 차선 검출 (튠 모드에선 추론 생략, 마지막 마스크 재사용) ──
            err, found, dbg_orig, dbg_bev = det.process(
                frame, reuse_masks=tune_mode)

            # ── 조향 결정 ──
            if err is not None:
                steer = ctrl.compute(err)
                last_steer = steer
                lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames <= cfg["lost_hold_frames"]:
                    steer = last_steer
                else:
                    steer = last_steer * 0.5

            # ── 구동 명령 (일반 정지는 램프 감속, 비상/로스트 정지는 즉시) ──
            if running and lost_frames <= cfg["lost_stop_frames"]:
                target_pwm = int(cfg["drive_pwm"])
                if abs(steer) > cfg["slow_steer_thresh"]:
                    target_pwm = int(cfg["slow_pwm"])
            else:
                target_pwm = 0
                if running:          # 로스트 초과 정지는 안전상 즉시
                    cur_pwm = 0.0

            ramp = float(cfg.get("brake_ramp_pwm", 6))
            if target_pwm < cur_pwm:
                cur_pwm = max(target_pwm, cur_pwm - ramp)  # 서서히 감속
            else:
                cur_pwm = float(target_pwm)                # 가속/유지는 즉시

            pwm = int(round(cur_pwm))
            if pwm > 0:
                link.send_drive(pwm, steer)  # 감속 중에도 조향은 계속 유지
            else:
                link.send_brake()

            # ── 데이터셋 녹화 (RUN 중에만, N프레임마다 원본 저장) ──
            if record and running and \
                    frames % max(1, int(cfg.get("record_every", 3))) == 0:
                fn = f"f{frames:06d}.jpg"
                cv2.imwrite(os.path.join(rec_dir, fn), frame)
                rec_csv.write(f"{fn},"
                              f"{'' if err is None else round(err, 4)},"
                              f"{round(steer, 4)},{pwm}\n")
                rec_count += 1

            # ── 비상정지 ──
            if _ESTOP["stop"]:
                _ESTOP["stop"] = False
                running = False
                cur_pwm = 0.0        # 비상정지는 램프 없이 즉시
                link.send_brake()
                print("[estop] 비상정지!")
            if _ESTOP["quit"]:
                break

            # ── 디스플레이 ──
            frames += 1
            if show_debug:
                fps = frames / max(1e-3, time.time() - t0)

                # 이미 640x360이면 resize 생략 (프레임당 몇 ms 절약)
                v_orig = dbg_orig if dbg_orig.shape[1::-1] == (VW, VH) \
                    else cv2.resize(dbg_orig, (VW, VH))
                v_bev = dbg_bev if dbg_bev.shape[1::-1] == (VW, VH) \
                    else cv2.resize(dbg_bev, (VW, VH))

                _title(v_orig, "CAMERA + MASK")
                _title(v_bev, "BIRD'S EYE VIEW")

                views = np.vstack((v_orig, v_bev))   # 카메라 위, BEV 아래
                dash = _dashboard(
                    VW, running, steer, err, found,
                    lost_frames, fps, pwm,
                    int(cfg.get("center_offset", 0)), tune=tune_mode)
                display = np.vstack((views, dash))
                if record and running:
                    cv2.circle(display, (VW - 96, 16), 7, RED, -1)
                    _text(display, f"REC {rec_count}",
                          (VW - 82, 21), 0.5, RED, 2)
                cv2.imshow("AI Lane", display)

            # Settings 창 현재값 패널 (3프레임마다 갱신)
            if frames % 3 == 0:
                cv2.imshow(SW, _settings_panel(cfg, sel_bar, SW))

            # ── 키 입력 ──
            # 방향키를 받으려면 waitKeyEx 원값이 필요하다. 방향키 판정을
            # 반드시 먼저 하고, 그 뒤에만 & 0xFF 로 ASCII 키를 본다.
            kx = cv2.waitKeyEx(1)
            k = kx & 0xFF

            # 방향키(또는 대체키 n/m , .): 트랙바 선택 + 1스텝 미세조정
            if kx in _ARROW_UP or k == ord('n'):
                sel_bar = (sel_bar - 1) % len(TUNE_BARS)
            elif kx in _ARROW_DOWN or k == ord('m'):
                sel_bar = (sel_bar + 1) % len(TUNE_BARS)
            elif kx in _ARROW_LEFT or k == ord(','):
                _nudge_bar(SW, TUNE_BARS[sel_bar], -1)
            elif kx in _ARROW_RIGHT or k == ord('.'):
                _nudge_bar(SW, TUNE_BARS[sel_bar], +1)
            elif k == ord('q'):
                break
            elif k == ord('s'):
                running = True
                t0, frames = time.time(), 0
                print("[main_model] GO")
            elif k in (ord('x'), ord(' ')):
                running = False
                print("[main_model] STOP")
            elif k == ord('d'):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow("AI Lane")
            elif k == ord('t'):
                tune_mode = not tune_mode
                if tune_mode:
                    running = False
                    cur_pwm = 0.0
                    link.send_brake()   # 튜닝 중 주행 금지 (오래된 마스크)
                print(f"[main_model] TUNE {'ON (추론 정지)' if tune_mode else 'OFF'}")
            elif k == ord('w'):
                save(cfg)
                print("[main_model] 설정 저장 완료")
    finally:
        link.send_brake()
        link.close()
        cap.release()
        cv2.destroyAllWindows()
        if rec_csv:
            rec_csv.close()
            print(f"[record] {rec_count}장 저장됨 -> {rec_dir}")
        print("[main_model] 종료")


if __name__ == "__main__":
    main()
