#!/usr/bin/env python3
"""
main_model.py -- AI 모델 차선추종 (자동 IPM 캘리브레이션, ROI 트랙바 없음)

dl_model/main_model.py 와 동일하되, IPM 사다리꼴을 사람이 트랙바로 맞추는 대신
lane_model.py가 매 프레임 소실점을 계산해서 스스로 잡는다 (자세한 원리는
lane_model.py 상단 docstring 참고). 그래서 IPM 관련 트랙바 8개가 사라지고,
대신 자동 캘리브레이션 민감도(VP margin)와 스무딩(Smooth) 트랙바가 생겼다.

실행:
  python3 main_model.py            # 실제 주행
  python3 main_model.py --dry      # 시리얼 없이 비전만
  python3 main_model.py --cam 1    # 카메라 지정
  python3 main_model.py --record   # 주행하며 데이터셋 녹화

키:
  s=출발  x/Space=정지  q=종료  w=설정저장  d=디버그토글  t=튠모드
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


def _dashboard(total_w, running, steer, err, found, lost, fps, pwm, offset,
               calib_mode="INIT", tune=False):
    H = 146
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

    cv2.line(d, (0, 98), (total_w, 98), OUTLINE, 1)
    lost_c = RED if lost > 5 else AMBER if lost > 0 else TXT
    calib_c = GREEN if calib_mode == "AUTO" else AMBER
    items = [("FPS", f"{fps:.0f}", TXT, 14), ("BANDS", str(found), TXT, 100),
             ("LOST", str(lost), lost_c, 195),
             ("CALIB", calib_mode, calib_c, 280)]
    for k, v, c, x in items:
        _text(d, k, (x, 120), 0.42, DIM)
        _text(d, v, (x + 50, 120), 0.42, c)
    if tune:
        _text(d, "TUNE MODE (t)", (total_w - 175, 120), 0.5, AMBER, 2)

    _text(d, "s go   x stop   t tune   w save   d debug   q quit",
          (14, 140), 0.4, DIM)
    return d


def _settings_panel(cfg):
    """Settings 창 하단: 현재 적용값 실시간 표시 (IPM은 자동이라 여기 없음)"""
    w, h = 480, 118
    p = np.full((h, w, 3), BG, dtype=np.uint8)
    _text(p, "Drag sliders  |  w: save -> calib.json", (10, 20), 0.42, DIM)
    cv2.line(p, (0, 28), (w, 28), OUTLINE, 1)
    _text(p, f"AUTO-CAL  VP margin {cfg.get('auto_calib_vp_margin', 0.14):.2f}"
             f"   smooth a={cfg.get('auto_calib_alpha', 0.15):.2f}"
             f"   min_px {int(cfg.get('auto_calib_min_px', 150))}",
          (10, 50), 0.42)
    _text(p, f"CTRL  Kp {cfg.get('kp', 0):.1f}   Kd {cfg.get('kd', 0):.1f}"
             f"   RGain {cfg.get('steer_right_gain', 1.0):.1f}"
             f"   HGain {cfg.get('heading_gain', 0.5):.1f}"
             f"   Ctr {int(cfg.get('center_offset', 0)):+d}px",
          (10, 72), 0.42)
    _text(p, f"DRIVE PWM {int(cfg.get('drive_pwm', 0))}"
             f"   Bands {int(cfg.get('n_bands', 5))}", (10, 96), 0.42)
    return p


# ── 메인 ──────────────────────────────────────────────────────

def main():
    dry = "--dry" in sys.argv
    cfg = load()

    if "--cam" in sys.argv:
        cfg["cam_index"] = int(sys.argv[sys.argv.index("--cam") + 1])

    record = "--record" in sys.argv
    rec_dir, rec_csv, rec_count = None, None, 0
    if record:
        rec_dir = time.strftime("records/run_%Y%m%d_%H%M%S")
        os.makedirs(rec_dir, exist_ok=True)
        rec_csv = open(os.path.join(rec_dir, "labels.csv"), "w")
        rec_csv.write("file,err,steer,pwm\n")
        print(f"[record] 저장 폴더: {rec_dir}")

    cap = cv2.VideoCapture(int(cfg["cam_index"]), cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg["frame_w"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg["frame_h"]))
    if not cap.isOpened():
        print("카메라 열기 실패. cam_index 확인")
        return

    det = ModelLaneDetector(cfg)
    ctrl = LaneFollowController(kp=cfg["kp"], kd=cfg.get("kd", 6.0),
                                steer_sign=cfg["steer_sign"],
                                right_gain=cfg.get("steer_right_gain", 1.3))
    link = MegaLink(cfg, dry_run=dry)

    # ── Settings 트랙바 창 (IPM 트랙바 없음 — 자동 계산됨) ──
    SW = "Settings"
    cv2.namedWindow(SW, cv2.WINDOW_AUTOSIZE)
    nop = lambda v: None
    cv2.createTrackbar("VPmargin x100", SW,
                       int(float(cfg.get("auto_calib_vp_margin", 0.14)) * 100),
                       50, nop)
    cv2.createTrackbar("Smooth x100",   SW,
                       int(float(cfg.get("auto_calib_alpha", 0.15)) * 100),
                       100, nop)
    cv2.createTrackbar("Center+100", SW,
                       int(cfg.get("center_offset", 0)) + 100, 200, nop)
    cv2.createTrackbar("Bands",      SW,
                       int(cfg.get("n_bands", 5)), 10, nop)
    cv2.createTrackbar("Kp x10",     SW,
                       int(float(cfg.get("kp", 1.4)) * 10), 50, nop)
    cv2.createTrackbar("Kd x10",     SW,
                       int(float(cfg.get("kd", 6.0)) * 10), 300, nop)
    cv2.createTrackbar("RGain x10",  SW,
                       int(float(cfg.get("steer_right_gain", 1.3)) * 10), 30, nop)
    cv2.createTrackbar("HGain x10",  SW,
                       int(float(cfg.get("heading_gain", 0.5)) * 10), 30, nop)
    cv2.createTrackbar("Drive PWM",  SW,
                       int(cfg.get("drive_pwm", 70)), 255, nop)

    cv2.imshow(SW, _settings_panel(cfg))

    running = False
    show_debug = True
    tune_mode = False
    last_steer = 0.0
    lost_frames = 0
    t0, frames = time.time(), 0
    VW, VH = 640, 360

    print(f"[main_model-autocal] cam={cfg['cam_index']} 준비 완료!")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            try:
                cfg["auto_calib_vp_margin"] = cv2.getTrackbarPos(
                    "VPmargin x100", SW) / 100.0
                cfg["auto_calib_alpha"] = max(0.01, cv2.getTrackbarPos(
                    "Smooth x100", SW) / 100.0)
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
            except cv2.error:
                pass
            det.cfg = cfg

            err, found, dbg_orig, dbg_bev = det.process(
                frame, reuse_masks=tune_mode)

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

            pwm = 0
            if running:
                if lost_frames > cfg["lost_stop_frames"]:
                    link.send_brake()
                else:
                    pwm = int(cfg["drive_pwm"])
                    if abs(steer) > cfg["slow_steer_thresh"]:
                        pwm = int(cfg["slow_pwm"])
                    link.send_drive(pwm, steer)
            else:
                link.send_brake()

            if record and running and \
                    frames % max(1, int(cfg.get("record_every", 3))) == 0:
                fn = f"f{frames:06d}.jpg"
                cv2.imwrite(os.path.join(rec_dir, fn), frame)
                rec_csv.write(f"{fn},"
                              f"{'' if err is None else round(err, 4)},"
                              f"{round(steer, 4)},{pwm}\n")
                rec_count += 1

            if _ESTOP["stop"]:
                _ESTOP["stop"] = False
                running = False
                link.send_brake()
                print("[estop] 비상정지!")
            if _ESTOP["quit"]:
                break

            frames += 1
            if show_debug:
                fps = frames / max(1e-3, time.time() - t0)

                v_orig = dbg_orig if dbg_orig.shape[1::-1] == (VW, VH) \
                    else cv2.resize(dbg_orig, (VW, VH))
                v_bev = dbg_bev if dbg_bev.shape[1::-1] == (VW, VH) \
                    else cv2.resize(dbg_bev, (VW, VH))

                _title(v_orig, "CAMERA + MASK")
                _title(v_bev, "BIRD'S EYE VIEW (auto-cal)")

                views = np.vstack((v_orig, v_bev))
                dash = _dashboard(
                    VW, running, steer, err, found,
                    lost_frames, fps, pwm,
                    int(cfg.get("center_offset", 0)),
                    calib_mode=det.calib_mode, tune=tune_mode)
                display = np.vstack((views, dash))
                if record and running:
                    cv2.circle(display, (VW - 96, 16), 7, RED, -1)
                    _text(display, f"REC {rec_count}",
                          (VW - 82, 21), 0.5, RED, 2)
                cv2.imshow("AI Lane", display)

            if frames % 3 == 0:
                cv2.imshow(SW, _settings_panel(cfg))

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
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
                    link.send_brake()
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
