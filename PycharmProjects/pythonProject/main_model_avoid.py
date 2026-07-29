#!/usr/bin/env python3
"""main_model.py -- lane following + IPM tuning + dashboard
   + lane 1/2 mode switch + ultrasonic front-center obstacle avoidance

  camera(thread) -> YOLO lane detection -> P control -> serial(Mega)

run:
  python3 main_model.py            # normal drive
  python3 main_model.py --dry      # vision only, no serial
  python3 main_model.py --cam 1    # choose camera index
  python3 main_model.py --record   # record dataset while driving

keys:
  s=go  x/Space=stop  q=quit  w=save settings  d=toggle debug view
  1=lane1 mode (L=solid R=dash)  2=lane2 mode (L=dash R=solid)
  ESC=global emergency stop  F12=global quit
"""
import os
import sys
import time
import threading

import cv2
import numpy as np

from config import load, save
from lane_model import ModelLaneDetector
from control import LaneFollowController
from serial_driver import MegaLink


# ── camera thread (avoid buffer buildup / blocking, auto reconnect) ──
class CameraThread:
    def __init__(self, cam_index, w, h):
        self.cam_index = cam_index
        self.w, self.h = w, h
        self.cap = self._open()
        self.frame = None
        self.ok = False
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _update(self):
        fail = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                fail += 1
                if fail > 30:
                    print("[camera] reconnecting...", flush=True)
                    self.cap.release()
                    time.sleep(0.5)
                    self.cap = self._open()
                    fail = 0
                time.sleep(0.01)
                continue
            fail = 0
            with self.lock:
                self.ok = ok
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


# ── global emergency stop ──
try:
    from pynput import keyboard as _kb

    _ESTOP = {"stop": False, "quit": False}


    def _on_press(key):
        if key == _kb.Key.esc:
            _ESTOP["stop"] = True
        elif key == _kb.Key.f12:
            _ESTOP["quit"] = True


    _kb.Listener(on_press=_on_press, daemon=True).start()
    print("[estop] ESC=stop, F12=quit")
except ImportError:
    _ESTOP = {"stop": False, "quit": False}
    print("[estop] pynput not installed")

# ── UI palette (BGR) ──
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (28, 26, 24)
PANEL = (46, 42, 38)
OUTLINE = (70, 63, 57)
TXT = (232, 227, 220)
DIM = (145, 137, 128)
GREEN = (105, 205, 95)
AMBER = (55, 175, 255)
RED = (80, 80, 245)


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
               tune=False, obstacle=False, obstacle_cm=None, lane_mode=2,
               avoiding=False):
    H = 146
    d = np.full((H, total_w, 3), BG, dtype=np.uint8)
    cv2.line(d, (0, 0), (total_w, 0), OUTLINE, 1)

    if avoiding:
        fill, edge, label = (55, 120, 200), AMBER, "AVOID"
    elif obstacle:
        fill, edge, label = (46, 46, 140), RED, "STOP"
    elif running:
        fill, edge, label = (52, 130, 52), GREEN, "RUN"
    else:
        fill, edge, label = (46, 46, 140), RED, "STOP"
    cv2.rectangle(d, (14, 14), (124, 56), fill, -1)
    cv2.rectangle(d, (14, 14), (124, 56), edge, 1)
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.7, 2)
    _text(d, label, (14 + (110 - tw) // 2, 44), 0.7, (255, 255, 255), 2)
    _text(d, f"PWM {pwm}  [lane {lane_mode}]", (20, 82), 0.42, DIM)

    bx = 150
    bw = max(120, total_w - bx - 122)
    _bar(d, bx, 24, bw, 20, steer, "STEER", flip=True)
    if err is not None:
        _bar(d, bx, 68, bw, 20, err, "ERROR")
    else:
        _text(d, f"LANE LOST ({lost})", (bx, 82), 0.55, RED, 2)

    if avoiding:
        _text(d, "LANE CHANGE IN PROGRESS", (bx, 82), 0.5, AMBER, 2)
    elif obstacle:
        _text(d, f"OBSTACLE {obstacle_cm}cm", (bx, 82), 0.55, RED, 2)

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

    _text(d, "s go  x stop  1/2 lane  w save  d debug  q quit",
          (14, 140), 0.4, DIM)
    return d


def _settings_panel(cfg, lane_mode=2, avoid_state="normal"):
    w, h = 480, 150
    p = np.full((h, w, 3), BG, dtype=np.uint8)
    _text(p, "Drag sliders  |  w: save -> calib.json", (10, 20), 0.42, DIM)
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
    _text(p, f"LANE MODE: {lane_mode}   AVOID: {avoid_state}",
          (10, 140), 0.42, AMBER)
    return p


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
        print(f"[record] output folder: {rec_dir} "
              f"(only while RUN, 1 frame per {int(cfg.get('record_every', 3))})")

    cap = CameraThread(int(cfg["cam_index"]),
                       int(cfg["frame_w"]), int(cfg["frame_h"]))
    if not cap.isOpened():
        print("camera open failed. check cam_index")
        return

    det = ModelLaneDetector(cfg)
    ctrl = LaneFollowController(kp=cfg["kp"], kd=cfg.get("kd", 6.0),
                                steer_sign=cfg["steer_sign"],
                                right_gain=cfg.get("steer_right_gain", 1.3))
    link = MegaLink(cfg, dry_run=dry)

    if "lane_mode" not in cfg:
        cfg["lane_mode"] = 2

    SW = "Settings"
    cv2.namedWindow(SW, cv2.WINDOW_AUTOSIZE)
    nop = lambda v: None
    cv2.createTrackbar("TL x+100", SW,
                       int(float(cfg.get("ipm_tl_x", 0.25)) * 100) + 100, 300, nop)
    cv2.createTrackbar("TL y%", SW,
                       int(float(cfg.get("ipm_tl_y", 0.40)) * 100), 100, nop)
    cv2.createTrackbar("TR x+100", SW,
                       int(float(cfg.get("ipm_tr_x", 0.75)) * 100) + 100, 300, nop)
    cv2.createTrackbar("TR y%", SW,
                       int(float(cfg.get("ipm_tr_y", 0.40)) * 100), 100, nop)
    cv2.createTrackbar("BL x+100", SW,
                       int(float(cfg.get("ipm_bl_x", -0.40)) * 100) + 100, 300, nop)
    cv2.createTrackbar("BL y%", SW,
                       int(float(cfg.get("ipm_bl_y", 0.98)) * 100), 100, nop)
    cv2.createTrackbar("BR x+100", SW,
                       int(float(cfg.get("ipm_br_x", 1.40)) * 100) + 100, 300, nop)
    cv2.createTrackbar("BR y%", SW,
                       int(float(cfg.get("ipm_br_y", 0.98)) * 100), 100, nop)
    cv2.createTrackbar("Center+100", SW,
                       int(cfg.get("center_offset", 0)) + 100, 200, nop)
    cv2.createTrackbar("Bands", SW,
                       int(cfg.get("n_bands", 5)), 10, nop)
    cv2.createTrackbar("Kp x10", SW,
                       int(float(cfg.get("kp", 1.4)) * 10), 50, nop)
    cv2.createTrackbar("Kd x10", SW,
                       int(float(cfg.get("kd", 6.0)) * 10), 300, nop)
    cv2.createTrackbar("RGain x10", SW,
                       int(float(cfg.get("steer_right_gain", 1.3)) * 10), 30, nop)
    cv2.createTrackbar("HGain x10", SW,
                       int(float(cfg.get("heading_gain", 0.5)) * 10), 30, nop)
    cv2.createTrackbar("Drive PWM", SW,
                       int(cfg.get("drive_pwm", 70)), 255, nop)

    cv2.imshow(SW, _settings_panel(cfg, cfg["lane_mode"], "normal"))

    running = False
    show_debug = True
    tune_mode = False
    last_steer = 0.0
    lost_frames = 0
    t0, frames = time.time(), 0
    VW, VH = 640, 360

    # ── obstacle avoidance settings ──
    OBSTACLE_CM = 100  # front-center distance threshold (cm)
    last_tele = None

    LANE_CHANGE_FRAMES = 60  # forced-steer duration during lane change
    LANE_CHANGE_STEER = 0.8  # forced steer magnitude (-1..1)
    AVOID_COOLDOWN_FRAMES = 60  # frames to ignore re-detection after a change
    avoid_state = "normal"  # "normal" | "changing"
    avoid_dir = 0.0
    avoid_target_mode = None
    avoid_frame_count = 0
    avoid_cooldown = 0

    print(f"[main_model] cam={cfg['cam_index']} ready! (start lane_mode={cfg['lane_mode']})")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if running:
                    link.send_brake()
                k = cv2.waitKey(10) & 0xFF
                if k == ord('q'):
                    break
                continue

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
            except cv2.error:
                pass
            det.cfg = cfg

            err, found, dbg_orig, dbg_bev = det.process(
                frame, reuse_masks=tune_mode)

            # ── ultrasonic telemetry: front-center sensor only ──
            tele = link.read_telemetry()
            if tele:
                last_tele = tele
            front_blocked = False
            front_min = None
            if last_tele:
                front_min = last_tele["fC"]
                front_blocked = front_min < OBSTACLE_CM

            # ── avoidance state machine ──
            # steer toward the side where solid line is NOT:
            #   lane 2 (R=solid) -> steer left ; lane 1 (L=solid) -> steer right
            if avoid_state == "normal":
                if avoid_cooldown > 0:
                    avoid_cooldown -= 1
                elif front_blocked and running:
                    cur_mode = int(cfg.get("lane_mode", 2))
                    if cur_mode == 2:
                        avoid_dir, avoid_target_mode = 1.0, 1
                    else:
                        avoid_dir, avoid_target_mode = -1.0, 2
                    avoid_state = "changing"
                    avoid_frame_count = 0
                    print(f"[avoid] front {front_min}cm detected -> "
                          f"changing to lane {avoid_target_mode}")
            elif avoid_state == "changing":
                avoid_frame_count += 1
                if avoid_frame_count >= LANE_CHANGE_FRAMES:
                    cfg["lane_mode"] = avoid_target_mode
                    det.prev_left_x = det.prev_right_x = None
                    det.miss_left = det.miss_right = 0
                    ctrl.prev_err = None
                    ctrl.d_filt = 0.0
                    avoid_state = "normal"
                    avoid_cooldown = AVOID_COOLDOWN_FRAMES
                    print(f"[avoid] lane change to {avoid_target_mode} complete")

            # ── steering decision ──
            if avoid_state == "changing":
                steer = avoid_dir * LANE_CHANGE_STEER
                last_steer = steer
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

            # ── drive command ──
            pwm = 0
            if running:
                if avoid_state == "changing":
                    pwm = int(cfg["slow_pwm"])
                    link.send_drive(pwm, steer)
                elif lost_frames > cfg["lost_stop_frames"]:
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
                print("[estop] EMERGENCY STOP!")
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
                _title(v_bev, "BIRD'S EYE VIEW")

                views = np.vstack((v_orig, v_bev))
                dash = _dashboard(
                    VW, running, steer, err, found,
                    lost_frames, fps, pwm,
                    int(cfg.get("center_offset", 0)), tune=tune_mode,
                    obstacle=front_blocked, obstacle_cm=front_min,
                    lane_mode=int(cfg.get("lane_mode", 2)),
                    avoiding=(avoid_state == "changing"))
                display = np.vstack((views, dash))
                if record and running:
                    cv2.circle(display, (VW - 96, 16), 7, RED, -1)
                    _text(display, f"REC {rec_count}",
                          (VW - 82, 21), 0.5, RED, 2)
                cv2.imshow("AI Lane", display)

            if frames % 3 == 0:
                cv2.imshow(SW, _settings_panel(
                    cfg, int(cfg.get("lane_mode", 2)), avoid_state))

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
                print(f"[main_model] TUNE {'ON (inference paused)' if tune_mode else 'OFF'}")
            elif k == ord('w'):
                save(cfg)
                print("[main_model] settings saved")
            elif k == ord('1'):
                cfg["lane_mode"] = 1
                det.prev_left_x = det.prev_right_x = None
                det.miss_left = det.miss_right = 0
                print("[main_model] lane_mode=1 (L=solid R=dash)")
            elif k == ord('2'):
                cfg["lane_mode"] = 2
                det.prev_left_x = det.prev_right_x = None
                det.miss_left = det.miss_right = 0
                print("[main_model] lane_mode=2 (L=dash R=solid)")
    finally:
        link.send_brake()
        link.close()
        cap.release()
        cv2.destroyAllWindows()
        if rec_csv:
            rec_csv.close()
            print(f"[record] {rec_count} images saved -> {rec_dir}")
        print("[main_model] shutdown")


if __name__ == "__main__":
    main()