"""
line_guide.py -- 카메라로 좌측 주차선을 보고 **직진을 유지**한다.

SEARCH / APPROACH 구간에서만 조향을 넘겨받고, 스윙부터는 손을 뗀다.
(어느 상태에서 쓸지는 config.LINE_GUIDE_STATES)

★왜 별도 스레드인가★
  모델 추론이 한 장에 40~50ms 다. 미션 루프는 20Hz(50ms)라, 같은 루프에서
  돌리면 라이다 판정 주기가 통째로 무너진다. *_HOLD_S 같은 시간 조건이 전부
  영향을 받는다.
  그래서 카메라는 따로 돌고, 미션 루프는 **가장 최근 조향값만 읽어 간다.**
  추론이 느려도 미션 루프는 원래 주기를 지킨다.

★못 봐도 미션을 막지 않는다★
  카메라가 안 열리거나 선을 놓치면 조향 0(직진)을 돌려준다. 그러면 v4 와
  같은 동작(조향중립 직진)이 되고, 주차는 그대로 진행된다.
"""

import threading
import time

import cv2
import numpy as np

import config as C
from line_ctrl import make_controller
from model_detect import make_finder


def _overlay(frame, res, steer, note, state, hits, frames):
    """카메라 화면에 무엇을 보고 무엇을 시키는지 겹쳐 그린다."""
    v = frame.copy()
    x0, y0, x1, y1 = res["roi_box"]
    cv2.rectangle(v, (x0, y0), (x1, y1), (0, 200, 200), 2)          # ROI

    if res["mask"] is not None:                                      # 검출 픽셀
        m = np.zeros(v.shape[:2], np.uint8)
        m[y0:y1, x0:x1] = res["mask"]
        v[m > 0] = (0, 255, 0)

    if res["found"] and res["seg"] is not None:                      # 피팅된 직선
        (ax, ay), (bx, by) = res["seg"]
        cv2.line(v, (int(ax), int(ay)), (int(bx), int(by)), (0, 0, 255), 3)
        cv2.circle(v, (int(res["x_bot"]), y1), 6, (0, 0, 255), -1)   # 제어 기준점

    bar = np.zeros((58, v.shape[1], 3), np.uint8)
    on = state in C.LINE_GUIDE_STATES
    cv2.putText(bar, f"{state}{'' if on else '  (guide off)'}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0) if on else (120, 120, 120), 1)
    cv2.putText(bar, f"steer {steer:+.2f}   {note}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    rate = hits / max(1, frames)
    cv2.putText(bar, f"det {hits}/{frames} ({rate:.0%})",
                (v.shape[1] - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255) if rate > 0.8 else (0, 165, 255), 1)
    return np.vstack([v, bar])


class LineGuide:
    def __init__(self):
        self.steer = 0.0          # 미션 루프가 읽어 가는 값
        self.status = "off"
        self.alive = False
        self.frames = 0
        self.hits = 0
        self.ex_frames = 0        # 탈출 구간(EXIT_TURN/CRUISE)은 따로 센다.
        self.ex_hits = 0          #   섞으면 직진 구간 검출률이 왜곡된다
        # ★"INIT" 로 시작한다★ FSM 이 상태를 넣어주기 전(= 호밍 중)에는
        # 처리도 집계도 하지 않기 위함. 빈 문자열로 두면 호밍 4초 동안의
        # 프레임이 검출률에 섞여 숫자가 실제보다 낮게 보인다.
        self.state = "INIT"       # FSM 이 매 프레임 갱신한다
        self._stop = threading.Event()
        self._t = None
        self._cap = None          # start() 에서 미리 열어 확인해 둔 카메라
        self.view = None          # 디버그용 오버레이 프레임 (메인 스레드가 띄운다)
        self.angle = None         # 최근 프레임의 선 기울기(도)
        self.straight_angle = None  # 직진 구간에서 본 기울기의 중앙값.
                                    #   탈출 우회전을 "이 각도로 돌아오면 끝"
                                    #   으로 판정하는 기준이 된다
        self._ang_buf = []

    # ── 미션 루프에서 쓰는 것 ────────────────────────────────
    def steer_for(self, state):
        """이 상태에서 넘겨받을 조향. 해당 없으면 None (FSM 이 알아서 중립).

        상태는 FSM.update() 가 매 프레임 넣어주므로 여기서는 읽기만 한다.
        """
        if not self.alive or state not in C.LINE_GUIDE_STATES:
            return None
        return self.steer

    def start(self):
        if not C.LINE_GUIDE:
            print("[guide] 비활성 (config.LINE_GUIDE=0)")
            return self
        # ★카메라를 미리 열어 확인한다★ 's' 를 누르기 **전에** 알아야 한다.
        # 다른 스택(main_model / main_obstacle / main_line)이 떠 있으면 그쪽이
        # 카메라를 잡고 있어 여기서 실패한다. 미션이 시작된 뒤에 알면 늦다.
        self._cap = cv2.VideoCapture(C.CAM_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
        ok, frame = (False, None)
        if self._cap.isOpened():
            ok, frame = self._cap.read()
        if not ok:
            self._cap.release()
            self._cap = None
            print("=" * 64)
            print(f"[guide] ★★ 카메라 {C.CAM_INDEX} 를 못 엽니다 ★★")
            print("   다른 프로그램이 카메라를 잡고 있을 가능성이 큽니다.")
            print("   확인:  ps aux | grep main_")
            print("   (main_model.py / main_obstacle.py / main_line.py 를 먼저 끄세요)")
            print("   이대로 진행하면 직진 보조 없이 달립니다.")
            print("   호밍도 꺼져 있으면 바퀴를 바로잡을 수단이 없습니다 —")
            print("   calib.json 의 park_home_enable 을 1 로 켜세요.")
            print("=" * 64)
        else:
            print(f"[guide] 카메라 {C.CAM_INDEX} OK "
                  f"({frame.shape[1]}x{frame.shape[0]})")

        # ★모델도 미리 읽어둔다★ 스레드 안에서 처음 쓸 때 읽으면 몇 초가
        # 걸리는데, 그 동안 조향이 0 이라 출발 직후 보정이 안 들어간다.
        # (실측: SEARCH 초반 내내 cam +0.00 이었다)
        try:
            from model_detect import _load
            _load()
        except Exception as e:
            print(f"[guide] 모델 사전 로딩 실패({e}) — 스레드에서 다시 시도합니다")
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)

    # ── 스레드 ───────────────────────────────────────────────
    def _run(self):
        cap = self._cap          # start() 에서 이미 열고 확인했다
        if cap is None:
            self.status = "no camera"
            return

        finder = make_finder()
        ctrl = make_controller()
        self.alive = True
        print(f"[guide] 시작 — 세그멘테이션 모델 + 거리 P/D 제어, "
              f"적용 구간 {C.LINE_GUIDE_STATES}")

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            # 직진 구간이 아니면 판정도 기준잡기도 하지 않는다.
            # 스윙 중에 기준이 잠기면 다음 직진 구간에서 엉뚱한 값을 쓴다.
            # 집계(frames/hits)도 여기서 걸러지므로, 종료 시 찍히는 검출률은
            # **직진 구간만의 값**이 된다.
            if self.state not in C.LINE_GUIDE_STATES:
                self.steer = 0.0
                time.sleep(0.03)
                continue

            res = finder.process(frame)
            straight = self.state in ("SEARCH", "APPROACH")
            if straight:
                self.frames += 1
            else:
                self.ex_frames += 1
            if res["found"]:
                if straight:
                    self.hits += 1
                else:
                    self.ex_hits += 1
                self.angle = res["angle"]
                # 직진 구간의 각도를 모아 기준을 만든다. 탈출 회전에서
                # "그때 그 각도로 돌아왔나"를 판정할 때 쓴다.
                if self.state in ("SEARCH", "APPROACH"):
                    self._ang_buf.append(res["angle"])
                    self.straight_angle = float(np.median(self._ang_buf[-40:]))
            else:
                self.angle = None
            s, note = ctrl.update(res["angle"] if res["found"] else None,
                                  time.time(),
                                  res["x_bot"] if res["found"] else None)
            self.steer = float(s)
            self.status = note
            if C.GUIDE_SHOW:
                # ★그리기만 하고 띄우지는 않는다★ macOS 에서 cv2.imshow 는
                # 메인 스레드에서만 안전하다. 메인 루프가 self.view 를 가져다 띄운다.
                self.view = _overlay(frame, res, s, note, self.state,
                                     self.hits, self.frames)

        self.alive = False
        self.steer = 0.0
        cap.release()
        print(f"[guide] 종료 — 직진구간 {self.frames}프레임 중 검출 "
              f"{self.hits}({self.hits / max(1, self.frames):.0%})"
              f"   |  탈출구간 {self.ex_frames}중 {self.ex_hits}"
              f"({self.ex_hits / max(1, self.ex_frames):.0%})")
