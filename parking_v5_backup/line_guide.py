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

import config as C
from line_ctrl import make_controller
from model_detect import make_finder


class LineGuide:
    def __init__(self):
        self.steer = 0.0          # 미션 루프가 읽어 가는 값
        self.status = "off"
        self.alive = False
        self.frames = 0
        self.hits = 0
        # ★"INIT" 로 시작한다★ FSM 이 상태를 넣어주기 전(= 호밍 중)에는
        # 처리도 집계도 하지 않기 위함. 빈 문자열로 두면 호밍 4초 동안의
        # 프레임이 검출률에 섞여 숫자가 실제보다 낮게 보인다.
        self.state = "INIT"       # FSM 이 매 프레임 갱신한다
        self._stop = threading.Event()
        self._t = None

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
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)

    # ── 스레드 ───────────────────────────────────────────────
    def _run(self):
        cap = cv2.VideoCapture(C.CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
        if not cap.isOpened():
            print(f"[guide] ★★ 카메라 {C.CAM_INDEX} 를 못 엽니다 ★★")
            print("        직진 보조 없이 진행합니다. 그런데 호밍도 꺼져 있으면")
            print("        (calib.json park_home_enable=0) 출발 시 바퀴를 바로잡는")
            print("        수단이 하나도 없습니다 — 차가 휠 수 있습니다.")
            print("        ESC 로 멈추고 park_home_enable=1 로 켜세요.")
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
            self.frames += 1
            if res["found"]:
                self.hits += 1
            s, note = ctrl.update(res["angle"] if res["found"] else None,
                                  time.time(),
                                  res["x_bot"] if res["found"] else None)
            self.steer = float(s)
            self.status = note

        self.alive = False
        self.steer = 0.0
        cap.release()
        print(f"[guide] 종료 — 직진구간 {self.frames}프레임 중 검출 "
              f"{self.hits}({self.hits / max(1, self.frames):.0%})")
