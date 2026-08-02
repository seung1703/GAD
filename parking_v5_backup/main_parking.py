#!/usr/bin/env python3
"""
main_parking.py -- RPLidar 직각 후진 주차 미션 엔트리.

drive/obstacle 와 완전 분리된 독립 패키지. 통합 펌웨어(1바이트 프로토콜)를
serial_driver 로 재사용. 라이다는 노트북 직결.

모드:
  python3 main_parking.py --run            # 실주행 (라이다+아두이노+FSM)
  python3 main_parking.py --run --dry      # 아두이노 없이 (명령 로그만)
  python3 main_parking.py --record log.pkl # 실주행하며 스캔 기록
  python3 main_parking.py --replay log.pkl # 기록 재생 (라이다 없이 알고리즘/FSM)
  python3 main_parking.py --calib          # 섹터별 점유/벽면각 실시간 표시
  python3 main_parking.py --sim            # 합성 장면으로 무하드웨어 상태전이 확인
  python3 main_parking.py --straight       # 디버그: FSM 없이 조향중립+직진만

기동 순서 (parking_fsm.py 참고):
  SEARCH → SWING_OUT → REVERSE_IN → WAIT → EXIT_FWD → EXIT_TURN → CRUISE
마지막 CRUISE 는 계속 직진한다 — **ESC 로 직접 세운다**. 완료 상태는 없다.

키(전역, 창 포커스 무관): s=시작  ESC=정지  Space=일시정지/재개
  일시정지 중엔 차를 세우고 라이다뷰를 마지막 프레임에 고정한다 —
  '현재 위치'에서 인식 결과를 차분히 들여다볼 수 있다.
  멈춰 있던 시간은 FSM 타임아웃/유지시간에서 빼준다(shift_clock).

디버깅 중 디스플레이 절전 방지: 실행 내내 macOS caffeinate (종료 시 자동 해제).
"""
import atexit
import subprocess
import sys
import time

import numpy as np

import config as C
import frames as F
import perception as P
from parking_fsm import ParkingFSM

_PERIOD = 1.0 / C.LOOP_HZ

# ── 디스플레이 절전 방지 (macOS) ──────────────────────────────
try:
    _caffeine = subprocess.Popen(["caffeinate", "-d"])
    atexit.register(_caffeine.terminate)
except FileNotFoundError:
    pass   # macOS 아니면 무시

# ── 시작/정지/일시정지 키 (전역) ──────────────────────────────
try:
    from pynput import keyboard as _kb
    _CTRL = {"start": False, "stop": False, "pause": False}

    def _on_press(key):
        try:
            if key.char == "s":
                _CTRL["start"] = True
        except AttributeError:
            if key == _kb.Key.esc:
                _CTRL["stop"] = True
            elif key == _kb.Key.space:
                _CTRL["pause"] = not _CTRL["pause"]

    _kb.Listener(on_press=_on_press, daemon=True).start()
    print("[parking] s=시작, ESC=정지, Space=일시정지/재개")
except ImportError:
    _CTRL = {"start": True, "stop": False, "pause": False}   # 없으면 즉시 시작
    print("[parking] pynput 없음 — 즉시 시작(정지/일시정지 키 비활성)")


def _pause_gate(car, fsm=None):
    """일시정지 중이면 차를 세우고 화면을 마지막 프레임에 고정한 채 대기."""
    if not _CTRL["pause"]:
        return
    car.motor_stop()
    print("[pause] 일시정지 — 화면 고정, Space로 재개")
    t0 = time.time()
    while _CTRL["pause"] and not _CTRL["stop"]:
        time.sleep(0.05)
    if not _CTRL["stop"]:
        paused_s = time.time() - t0
        if fsm is not None:
            fsm.shift_clock(paused_s)   # 멈춘 시간은 타임아웃에서 제외
        print(f"[pause] 재개 ({paused_s:.1f}s 일시정지)")


def _wait_start(tag):
    """'s' 대기. 시작 전 ESC 면 False."""
    print(f"[{tag}] 's' 누르면 시작 (ESC=정지, Space=일시정지, Ctrl-C=종료)")
    while not _CTRL["start"]:
        if _CTRL["stop"]:
            print(f"[{tag}] 시작 전 취소")
            return False
        time.sleep(0.05)
    print(f"[{tag}] 시작!")
    return True


def _arg(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else True
    return default


def _try_viz():
    try:
        from visualize import ParkingViz
        return ParkingViz()
    except Exception as e:
        print(f"[viz] 비활성({e}) — 텍스트 모드")
        return None


# ── 합성 장면 (하드웨어 없이 상태전이만 확인) ──────────────────
def _wall(center_car_deg, dist, wall_dir_car_deg, n=14, span=500.0):
    """차체방위 center 쪽 dist(mm) 에 방향이 wall_dir 인 직선 벽 → 원시 스캔 조각."""
    cx = dist * np.sin(np.radians(center_car_deg))
    cy = dist * np.cos(np.radians(center_car_deg))
    ux = np.sin(np.radians(wall_dir_car_deg))
    uy = np.cos(np.radians(wall_dir_car_deg))
    out = []
    for s in np.linspace(-span / 2, span / 2, n):
        x, y = cx + s * ux, cy + s * uy
        out.append((30, float(F.to_raw(F.car_deg_of_xy(x, y))),
                    float(np.hypot(x, y))))
    return out


class SimScene:
    """상태에 맞춰 장면을 만들어주는 **로직 검증용** 스텁.

    물리(동역학·슬립·조향지연) 없음 — "상태가 순서대로 넘어가나"만 본다.
    실제 기하 검증은 --record 로 실차 로그를 받아 --replay 로 한다.
      SEARCH     : 우측에 주차차량 앞면
      SWING_OUT  : 시간에 비례해 벽면이 돌아 보이게 (자차 회전 흉내)
      REVERSE_IN : 처음엔 양옆 벽 → 잠시 후 사라짐 (완전 진입 흉내)
      그 외      : 빈 장면
    """

    def __init__(self):
        self.t_swing = None
        self.t_rev = None

    def scan(self, state):
        if state == ParkingFSM.SEARCH:
            return _wall(F.RIGHT_CAR_DEG, 1100.0, 0.0)
        if state == ParkingFSM.SWING_OUT:
            if self.t_swing is None:
                self.t_swing = time.time()
            rot = min(80.0, 45.0 * (time.time() - self.t_swing))   # 45도/초
            return _wall(F.RIGHT_CAR_DEG, 1000.0, rot)
        if state == ParkingFSM.REVERSE_IN:
            if self.t_rev is None:
                self.t_rev = time.time()
            if time.time() - self.t_rev < 1.2:      # 양옆 벽이 보이는 구간
                return (_wall(F.RIGHT_CAR_DEG, 420.0, 0.0)
                        + _wall(F.LEFT_CAR_DEG, 420.0, 0.0))
            return []                               # 완전 진입 → 양옆 소멸
        return []


# ── 캘리브레이션 모드 ──────────────────────────────────────────
def run_calib(reader, viz):
    print("[calib] 섹터별 점유 / 벽면각 표시. Ctrl-C 종료.")
    print(F.describe())
    reader.start()
    try:
        while True:
            latest = reader.get_latest()
            if latest is None:
                time.sleep(0.05)
                continue
            per = P.perceive(latest[1])
            cl = per["clusters"]
            parts = [f"clusters={len(cl)}"]
            for name in ("detect_right", "flank_left", "flank_right", "rear_fan"):
                lim = C.FLANK_NEAR_DIST if name.startswith("flank") else None
                st = P.sector_stats(cl, [F.SECTORS[name]], lim)
                d = "--" if st["dmin"] is None else f"{st['dmin']:.0f}"
                parts.append(f"{name} n={st['n']} d={d}")
            c = P.largest_near_cluster(cl, [F.DETECT_RIGHT, F.FLANK_RIGHT], 1500)
            wa = P.wall_car_deg(c) if c is not None else None
            parts.append(f"우측벽면각={'--' if wa is None else f'{wa:.0f}도'}")
            line = "  ".join(parts)
            print("[calib] " + line)
            if viz:
                viz.update(per, "CALIB", line, sectors=tuple(F.SECTORS))
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()


# ── 디버그: FSM 없이 조향중립+직진 + 라이다 인식 화면 ──────────
def run_straight(reader, car, viz, sim=False):
    if reader:
        reader.start()
    print(F.describe())
    if not _wait_start("straight"):
        return
    print("[straight] steer=중립, 직진 — FSM 없이 라이다 인식만 표시")
    try:
        while not _CTRL["stop"]:
            _pause_gate(car)
            if _CTRL["stop"]:
                break
            if _CTRL["pause"]:
                time.sleep(_PERIOD)
                continue
            if sim:
                scan = _wall(F.RIGHT_CAR_DEG, 1100.0, 0.0)
            else:
                latest = reader.get_latest()
                if latest is None:
                    time.sleep(0.03)
                    continue
                scan = latest[1]
            per = P.perceive(scan)
            car.steer_center()
            car.motor_forward(C.FWD_SPEED)
            car.tick()
            if viz:
                viz.update(per, "STRAIGHT", "steer=0(중립) 직진중 "
                           "(Space=일시정지)", sectors=tuple(F.SECTORS))
            time.sleep(_PERIOD)
        print("[straight] 정지 요청 — 중단")
    except KeyboardInterrupt:
        print("\n[straight] 중단")
    finally:
        car.close()
        if reader:
            reader.stop()


STATUS_ECHO_S = 0.5      # 상태줄 콘솔 출력 주기(초). 0 이면 끔


def _echo_status(fsm, _last=[0.0]):
    """fsm.status 를 콘솔에도 주기적으로 찍는다.

    status 는 원래 viz.update() 로만 전달돼서, 시각화가 꺼진 '텍스트 모드'
    에서는 아무 데도 안 나왔다. 그러면 REVERSE_IN 의 dev 궤적 같은 걸 볼
    수가 없어 현장 튜닝이 감으로 흐른다. viz 유무와 무관하게 찍는다.
    """
    if STATUS_ECHO_S <= 0 or not fsm.status:
        return
    now = time.time()
    if now - _last[0] < STATUS_ECHO_S:
        return
    _last[0] = now
    print(f"  [{fsm.state}] {fsm.status}")


# ── 미션 루프 (run/record/replay/sim 공용) ─────────────────────
def run_mission(reader, car, viz, sim=False, back=False):
    # 카메라 직진 보조 — 별도 스레드. 실패해도 미션은 그대로 진행된다.
    from line_guide import LineGuide
    guide = LineGuide().start() if not sim else LineGuide()
    fsm = ParkingFSM(car, guide=guide)
    fsm.back_mode = back
    if back:
        print("[mission] --back : 마지막 CRUISE 에서 전진 대신 후진합니다 "
              "(후방 비상정지는 그대로 동작)")
    scene = SimScene() if sim else None
    if reader:
        reader.start()
    # 이번 실행이 어느 방향/섹터를 봤는지 로그에 남긴다 (나중에 로그만 보고 판단 가능)
    print(F.describe())
    if not _wait_start("mission"):
        return

    # 조향 유격 정렬 — 좌 풀락 → 중립. SEARCH 가 steer_center() 로 직진하며
    # 차를 찾으므로, 출발 시 바퀴가 실제로 똑바로 서 있어야 한다.
    # (calib.json park_home_enable=0 으로 끌 수 있다. 2~3초 소요)
    from homing import run_homing
    run_homing(car)

    # 라이다 워밍업 — ★제자리에서 기다리지 않는다★
    # 곧바로 출발해서 달리되, 첫 LIDAR_WARMUP_S 동안만 SEARCH 의 "차 발견"
    # 판정을 막는다. 호밍으로 바퀴가 방금 크게 움직였고 라이다도 막 도는
    # 참이라 첫 프레임에 헛것이 섞이는데, 그걸로 스윙에 들어가는 걸 막는 것.
    # 대회 시계가 도는 중이므로 서 있는 시간은 없다.
    fsm.arm_after(C.LIDAR_WARMUP_S)

    try:
        while not _CTRL["stop"]:
            _pause_gate(car, fsm)
            if _CTRL["stop"]:
                break
            if _CTRL["pause"]:
                time.sleep(_PERIOD)
                continue

            if sim:
                scan = scene.scan(fsm.state)
            else:
                latest = reader.get_latest()
                if latest is None:
                    time.sleep(0.03)
                    continue
                scan = latest[1]
            per = P.perceive(scan)

            # 후방 비상정지: FSM 과 독립, 최우선. 원시 스캔 직독 —
            # 차체마스크(MIN_VALID_DIST)에 안 걸리게 (후방엔 차체가 없다)
            if fsm.rear_guard(scan):
                if viz:
                    viz.update(per, fsm.state, fsm.status,
                               sectors=fsm.sector_names())
                _echo_status(fsm)
                time.sleep(_PERIOD)
                continue

            fsm.update(per)
            if viz:
                viz.update(per, fsm.state, fsm.status,
                           sectors=fsm.sector_names())
            _echo_status(fsm)
            time.sleep(_PERIOD)
        print("[mission] 정지 요청 — 중단")
    except KeyboardInterrupt:
        print("\n[mission] 중단")
    finally:
        guide.stop()
        if car:
            car.close()
        if reader:
            reader.stop()


def main():
    replay = _arg("--replay")
    record = _arg("--record")
    dry = "--dry" in sys.argv
    sim = "--sim" in sys.argv
    calib = "--calib" in sys.argv
    straight = "--straight" in sys.argv
    back = "--back" in sys.argv        # 마지막 크루즈를 후진으로 (디버깅용)

    viz = _try_viz()

    # 리더 선택
    reader = None
    if replay:
        from lidar_reader import ReplayReader
        reader = ReplayReader(replay, realtime=True)
    elif not sim:
        from lidar_reader import LidarReader
        reader = LidarReader(record_path=record if isinstance(record, str) else None)

    if calib:
        if reader is None:
            print("[calib] --sim 과 함께 쓸 수 없음 (라이다나 --replay 필요)")
            return
        run_calib(reader, viz)
        return

    # 차량 커맨더 (replay/sim 은 dry 강제 — 실차 명령 방지)
    from arduino_iface import CarCommander
    car = CarCommander(dry_run=dry or bool(replay) or sim)

    if straight:
        run_straight(reader, car, viz, sim=sim)
        return

    run_mission(reader, car, viz, sim=sim, back=back)


if __name__ == "__main__":
    main()
