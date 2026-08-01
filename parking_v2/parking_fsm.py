"""
parking_fsm.py -- 직각 후진 주차 상태머신.

사양(사용자 확정) — 주차차량은 **우측에만** 있고, 슬롯에 들어가면 이웃 차량이
좌·우 양옆에 온다:

  0 SEARCH      조향중립 직진        → 우측 섹터에 클러스터 발견
  2 SWING_OUT   좌 풀조향 전진       → 처음 발견한 그 차와 SWING_STOP_DIST 만큼 벌어짐
  3 REVERSE_IN  우 풀조향 후진       → 좌·우 섹터가 (봤다가) 둘 다 비면 정지
  5 WAIT        정지                 → WAIT_S 경과
  6 EXIT_FWD    조향중립 직진        → EXIT_FWD_S 경과
  8 EXIT_TURN   우 조향 전진         → EXIT_TURN_S 경과
    CRUISE      조향중립 직진(계속)  → 없음. 사용자가 ESC 로 정지

완료(DONE) 상태는 없다 — 마지막은 계속 직진하며 ESC 를 기다린다.
카메라는 쓰지 않는다 (전부 라이다).

시간 조건은 프레임 수가 아니라 **초**로 센다(`_hold`). 루프 주기(LOOP_HZ)를
바꿔도 의미가 안 변한다. 예전 TRIGGER_FRAMES/DISAPPEAR_FRAMES 는 main 의
sleep(0.05) 리터럴에 몰래 묶여 있었다.
"""

import time

import numpy as np

import config as C
import frames as F
import perception as P

# ── 내부 상수 (튜닝 대상 아님) ────────────────────────────────
_REAR_SELF_NOISE_MM = 30     # 이보다 가까운 반사는 라이다 자기노이즈
_REAR_MAX_MM = 1500          # 후방 감시는 이 거리까지만 본다
_WALL_NEAR_MM = 1000         # 정렬 판정(3단계)에서 벽면각을 잴 근거리 한계.
                             #   ☆ 넓히면 안 된다 ☆ 2000 으로 실측해보니 후진
                             #   초반에 dev 가 6/40/21/35/22/37/43 처럼 추세
                             #   없이 튀었다. 그 구간은 L n=0, R n=0 (900mm 안에
                             #   아무것도 없음) 이었으니, 재고 있던 건 옆차가
                             #   아니라 900~2000mm 의 산발 클러스터였다.
                             #   실제 기준 옆차는 590~850mm 에서 잡힌다.
_WALL_MIN_PTS = 5            # 벽면각 SVD 에 쓸 최소 점 개수. wall_car_deg 는
                             #   2점이면 계산은 되지만, 그 각도는 의미가 없다
_TRACK_GATE_MM = 500         # 프레임간 연결: 직전 중심에서 이 안이면 같은 물체
_HEAD_NEAR_MM = 1800         # [계측] 차체 틀어짐을 잴 때 볼 최대 거리(mm).
                             #   첫 차가 1400mm 근처에 있으니 여유를 조금 둔다


def _mm(v):
    """거리 표시용 — 미검출(None)이면 '--'."""
    return "--" if v is None else f"{v:.0f}"


# 설정 함정 조기 경고 (실차에서 원인 찾느라 시간 버리지 않게)
if C.SWING_STOP_DIST >= C.MAX_DISTANCE:
    print(f"[config] ⚠ SWING_STOP_DIST({C.SWING_STOP_DIST}) >= "
          f"MAX_DISTANCE({C.MAX_DISTANCE}) — 그 거리에선 차가 안 보여서 "
          f"2단계가 영영 안 끝나고 타임아웃만 납니다")
if C.REVERSE_ALIGN_TOL > 90:
    print(f"[config] ⚠ REVERSE_ALIGN_TOL({C.REVERSE_ALIGN_TOL}) > 90 — "
          f"dev 는 0~90 이라 첫 프레임에 즉시 참이 됩니다 "
          f"(풀락 후진이 사라짐). 12 근처를 쓰세요")


# ── 후방 비상정지 (FSM 과 독립, 원시 스캔 직독) ────────────────
def rear_min_dist(scan):
    """원시 스캔에서 후진 부채꼴(frames.REAR_FAN) 최소거리(mm). 없으면 None.

    ★차체마스크(MIN_VALID_DIST=150)를 일부러 적용하지 않는다★
    비상거리(120mm)가 마스크보다 가까워서, 마스크를 태우면 정작 위험한 물체를
    "차체 반사"로 버린다. 후방엔 차체가 없으니 안전하다.
    """
    best = None
    for q, ang, dist in scan:
        if dist <= _REAR_SELF_NOISE_MM or dist > _REAR_MAX_MM:
            continue
        if q < P.MIN_QUALITY:
            continue
        if not F.in_sectors(F.to_car(ang), [F.REAR_FAN]):
            continue
        d = float(dist)
        if best is None or d < best:
            best = d
    return best


# ── 처음 발견한 주차차량 추적 (스윙 종료 판정용) ───────────────
class TrackedCar:
    """0단계에서 처음 발견한 주차차량 **하나**를 계속 따라가며 최근접 거리를 준다.

    ★왜 거리인가 (각도가 아니라)★
      이 차엔 엔코더·IMU 가 없어 자기 회전각을 직접 못 잰다. 예전엔 벽면각의
      프레임간 변화를 누적해 회전량을 추정했는데, 그건 **누적값**이라 중간에
      벽을 한 번 놓치면 그 자리에서 멈춰 목표에 영영 도달하지 못했다
      (실차에서 매번 SWING_TIMEOUT 으로 빠지던 원인).
      거리는 매 프레임 새로 재는 **절대값**이라 몇 프레임 놓쳤다 다시 잡아도
      즉시 정상값으로 복구된다. 풀락+정속이면 원호가 고정이라 거리는 회전량의
      단조 대리지표이기도 하다. 덤으로 속도·배터리·노면 변화에 영향받지 않는다.

    ★왜 후진~우측 사분면인가★
      좌로 꺾으면 그 차의 방위가 우측(90°)에서 후진(0°) 쪽으로 밀린다
      (90° - 회전량). frames.SWING_TRACK 이 그 경로를 통째로 덮는다.
      우측 섹터(±35°)만 보면 35°쯤에서 벗어나 놓친다.

    같은 물체인지는 직전 중심과의 거리(_TRACK_GATE_MM)로 판단한다. 사분면 안에
    후보가 여럿이어도(다음 차, 벽 등) 엉뚱한 걸로 갈아타지 않는다.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.dist = None          # 최근접 거리(mm). 못 보면 None
        self.bearing = None       # 차체기준 방위각(-180~180). 0=정후방, 90=정우측
        self._prev_c = None       # 추적 중인 클러스터 중심 (xy)
        self.tracking = False
        self.reacquired = 0

    @staticmethod
    def _center(c):
        return P.cxy(c).mean(0)

    def update(self, clusters):
        """→ 추적 중인 차와의 최근접 거리(mm) 또는 None."""
        # 사분면 안에 점이 있는 클러스터만 후보
        cand = [c for c in clusters if len(c) >= 2
                and P.sector_stats([c], [F.SWING_TRACK])["n"] > 0]
        if not cand:
            self.tracking = False
            self.dist = None
            self.bearing = None
            return None

        pick = None
        if self._prev_c is not None:                 # 같은 물체 이어가기
            best_d = _TRACK_GATE_MM
            for c in cand:
                d = float(np.hypot(*(self._center(c) - self._prev_c)))
                if d < best_d:
                    pick, best_d = c, d
        if pick is None:                             # 최초 획득 또는 재획득
            pick = min(cand, key=lambda c: float(c[:, 3].min()))
            if self._prev_c is not None:
                self.reacquired += 1

        self._prev_c = self._center(pick)
        self.tracking = True
        self.dist = float(pick[:, 3].min())          # 최근접 점까지의 거리

        # 중심의 차체기준 방위각. 좌표 규약이 x=d·sin, y=d·cos 이라 atan2(x, y).
        # -180~180 으로 감싸 둔다 — 0~360 그대로 두면 정후방을 막 지난 순간
        # 359° 가 되어 "각도가 작아지면 지나간 것" 판정이 뒤집힌다.
        cx, cy = self._prev_c
        self.bearing = float((np.degrees(np.arctan2(cx, cy)) + 180.0) % 360.0
                             - 180.0)
        return self.dist


class ParkingFSM:
    STATES = (SEARCH, APPROACH, SWING_OUT, REVERSE_IN, WAIT, EXIT_FWD,
              EXIT_TURN, CRUISE) = ("SEARCH", "APPROACH", "SWING_OUT",
                                    "REVERSE_IN", "WAIT", "EXIT_FWD",
                                    "EXIT_TURN", "CRUISE")

    # 그 상태가 **실제로 판정에 쓰는** 섹터. 화면은 이 표만 보고 강조한다
    # → "보이는 것 == 인식에 쓰는 것" 이 구조적으로 보장된다.
    ACTIVE_SECTORS = {
        SEARCH: ("detect_right",),
        APPROACH: ("detect_right",),
        SWING_OUT: ("swing_track",),
        REVERSE_IN: ("flank_left", "flank_right", "rear_fan"),
        WAIT: (),
        EXIT_FWD: (),
        EXIT_TURN: (),
        CRUISE: (),
    }

    def __init__(self, car):
        self.car = car
        self.state = self.SEARCH
        self.t_state = time.time()
        self._hold_t0 = None
        self._align_t0 = None        # 정렬 판정 전용 홀드 타이머
        self.car_track = TrackedCar()
        # 3단계 종료 판정: 좌·우 옆차를 **각각** 봤는지 따로 래치한다.
        # 하나로 합치면(예전 _flank_seen) 한쪽만 보고도 "봤음"이 켜져서,
        # 좌→우 인계 순간의 공백에 조기 정지한다.
        self._seen_left = False
        self._seen_right = False
        self._rear_t0 = None         # 후방 한계 접촉 지속 시간 타이머
        self.pass_dist = None        # ★캘리브레이션 기준량★ 첫 차 옆을
                                     #   지날 때의 최소거리(mm)
        self.head_dev = None         # [계측] 차체 틀어짐(도). 판정엔 미사용
        self._head_samples = []      # 위 값의 표본 (중앙값용)
        self._wheels_centered = False
        self.status = ""
        self.emergency = False

    # ── 상태 전이 ─────────────────────────────────────────────
    def _go(self, state):
        self.state = state
        self.t_state = time.time()
        self._hold_t0 = None
        self._align_t0 = None
        print(f"[fsm] -> {state}")

    def _elapsed(self):
        return time.time() - self.t_state

    def sector_names(self):
        return self.ACTIVE_SECTORS.get(self.state, ())

    def shift_clock(self, dt):
        """일시정지한 시간(dt)만큼 내부 시계를 밀어, 멈춰 있던 동안이
        타임아웃/유지시간에 카운트되지 않게 한다. main 의 _pause_gate 가 호출."""
        self.t_state += dt
        if self._hold_t0 is not None:
            self._hold_t0 += dt
        if self._align_t0 is not None:
            self._align_t0 += dt

    def _start_comp(self):
        """시작 위치에 따른 기동값 → (통과각[도], 스윙정지거리[mm]).

        config 의 CAL_A_* / CAL_B_* 두 지점을 실측 거리로 **선형 보간**한다.
        두 DIST 중 하나라도 0 이면 보간을 끄고 PASSED_CAR_DEG /
        SWING_STOP_DIST 를 그대로 쓴다.

        ★왜 2점 실측인가★
          한 지점 + 물리 모델(tan 기하)로 계산해봤더니 실차에서 안 맞았다.
          모델이 맞는지 아닌지를 검증할 방법이 없어서다. 두 지점에서 각각
          "실제로 잘 되는 값"을 재두면 그 사이 기울기가 **측정에서** 나오므로
          가정이 필요 없다. 양 끝을 시작점의 양 극단으로 잡고 사이만 보간하면
          외삽도 없다.

        ★왜 분기(if)가 아니라 보간인가★
          분기는 경계 근처에서 거리를 조금만 잘못 재도 값이 통째로 점프한다.
          보간은 오차가 비례해서만 커진다. 실측 노이즈가 40mm 남짓인데
          분기 경계에서 틀리면 스윙거리가 300mm 튀는 반면, 보간은 50mm 수준.
        """
        deg, stop = C.PASSED_CAR_DEG, C.SWING_STOP_DIST
        da, db = C.CAL_A_DIST, C.CAL_B_DIST
        d = self.pass_dist
        if da <= 0 or db <= 0 or d is None or abs(db - da) < 1.0:
            return deg, stop, False

        # 0~1 로 클램프 = 두 캘리브레이션 지점 밖으로는 안 나간다(외삽 금지)
        t = (d - da) / (db - da)
        t = max(0.0, min(1.0, t))
        deg = C.CAL_A_PASSED_DEG + t * (C.CAL_B_PASSED_DEG - C.CAL_A_PASSED_DEG)
        stop = C.CAL_A_SWING_STOP + t * (C.CAL_B_SWING_STOP - C.CAL_A_SWING_STOP)

        # 이상값 방어 — 라이다가 헛것을 봐도 말이 안 되는 기동은 안 나가게
        deg = max(30.0, min(88.0, float(deg)))
        stop = max(d + 100.0, min(C.MAX_DISTANCE - 200.0, float(stop)))
        return deg, stop, True

    def _measure_heading(self, clusters):
        """★계측 전용★ 첫 차의 면 각도로 내 차가 얼마나 틀어졌는지 잰다.

        판정에는 **일절 쓰지 않는다.** 화면·로그에만 띄운다.

        원리:
          주차된 차의 앞면은 차량 열과 나란하다. 내 차가 열과 평행하게 가고
          있으면 그 면은 내 진행축과도 나란해서 차체기준 0도로 읽힌다.
          내가 5도 틀어져 있으면 5도로 읽힌다. 그 값이 곧 헤딩 오차다.

        왜 계측만 하나:
          이 거리(1400mm 근처)에서 각도 피팅이 믿을 만한지 아직 모른다.
          후진 정렬에서 2000mm 로 재봤을 때 6/40/21/35/22/37/43 처럼 난수가
          나온 전례가 있다(점이 적으면 SVD 주축은 방향이 아니라 잡음이다).
          그래서 먼저 눈으로 확인한다 —
            · 차를 똑바로 놓고 돌렸을 때 값이 한 곳에 모이면  → 신뢰 가능
            · 프레임마다 튀면                                  → 이 방법은 포기
        """
        c = P.largest_near_cluster(clusters, [F.DETECT_RIGHT, F.SWING_TRACK],
                                   _HEAD_NEAR_MM)
        if c is None or len(c) < _WALL_MIN_PTS:
            self.head_dev = None
            return
        a = P.wall_car_deg(c)
        if a is None:
            self.head_dev = None
            return
        # 직선은 180도 주기 → -90~90 으로 접어서 0(=열과 나란함) 기준 편차로
        dev = ((a - C.HEADING_REF_DEG) + 90.0) % 180.0 - 90.0
        self.head_dev = float(dev)
        self._head_samples.append(self.head_dev)

    def _report_calib(self, bearing):
        """스윙 직전에 캘리브레이션용 숫자를 한 줄로 찍는다.

        ★현장에서 이 줄만 보고 config 를 채우면 된다★
        """
        deg, stop, interp = self._start_comp()
        s = sorted(self._head_samples)
        head = s[len(s) // 2] if s else None       # 중앙값 (튀는 값에 강함)
        print(f"[fsm] 첫 차 통과(방위 {bearing:+.0f}도) → 좌 스윙 시작")
        print(f"[fsm] ★캘리브레이션★ 옆을 지날 때 거리 = "
              f"{_mm(self.pass_dist)}mm"
              f"   (이 숫자를 CAL_A_DIST 또는 CAL_B_DIST 에 적으세요)")
        print(f"[fsm]   적용값: 통과각 {deg:.0f}도 / 스윙정지 {stop:.0f}mm"
              f"   보간={'ON' if interp else 'OFF (CAL_*_DIST=0)'}")
        print(f"[fsm]   [계측] 차체 틀어짐 = "
              f"{'측정불가' if head is None else f'{head:+.0f}도'}"
              f"  (표본 {len(s)}개, 판정엔 미사용)")

    def _align_hold(self, cond, hold_s):
        """정렬 판정 전용 홀드. _hold() 는 REVERSE_IN 에서 이미 '양옆 소멸'
        판정에 쓰고 있어서 타이머를 공유하면 서로 리셋시킨다."""
        if not cond:
            self._align_t0 = None
            return False
        now = time.time()
        if self._align_t0 is None:
            self._align_t0 = now
        return now - self._align_t0 >= hold_s

    def _hold(self, cond, hold_s):
        """cond 가 hold_s 초 동안 연속 참이면 True. 끊기면 처음부터 다시."""
        if not cond:
            self._hold_t0 = None
            return False
        now = time.time()
        if self._hold_t0 is None:
            self._hold_t0 = now
        return now - self._hold_t0 >= hold_s

    # ── 후방 비상정지 (매 프레임 최우선) ──────────────────────
    def rear_guard(self, scan):
        if self.state != self.REVERSE_IN:
            self._rear_t0 = None
            return False
        rd = rear_min_dist(scan)
        if rd is not None and rd < C.REAR_EMERGENCY_DIST:
            self.car.motor_stop()
            self.emergency = True
            now = time.time()
            if self._rear_t0 is None:
                self._rear_t0 = now
            elif now - self._rear_t0 >= C.REAR_EMERGENCY_HOLD_S:
                # ★탈출구★ main 은 rear_guard 가 True 면 update() 를 통째로
                # 건너뛴다. 그래서 이 상태가 계속되면 REVERSE_TIMEOUT 도 안
                # 돌아 미션이 영원히 멈춰 선다(모터만 정지한 채).
                # 후방 한계에 계속 닿아 있다 = 더는 못 들어간다 = 진입 완료로
                # 보고 다음 단계로 넘긴다.
                print(f"[fsm] 후방 한계 {rd:.0f}mm 가 "
                      f"{C.REAR_EMERGENCY_HOLD_S:.1f}s 지속 → 진입 완료로 판정")
                self.emergency = False
                self._rear_t0 = None
                self._go(self.WAIT)
                return False
            self.status = f"REAR EMERGENCY {rd:.0f}mm ({now - self._rear_t0:.1f}s)"
            return True
        self._rear_t0 = None
        self.emergency = False
        return False

    # ── 메인 스텝 ─────────────────────────────────────────────
    def update(self, percep):
        clusters = percep["clusters"]

        if self.state == self.SEARCH:
            self.car.steer_center()
            self.car.motor_forward(C.FWD_SPEED)
            st = P.sector_stats(clusters, [F.DETECT_RIGHT])
            self.status = f"search  우측 n={st['n']} dmin={_mm(st['dmin'])}"
            if self._hold(st["n"] > 0, C.TRIGGER_HOLD_S):
                # 트리거 지점에서 바로 꺾으면 진입각이 얕아 슬롯에 삐뚤게 들어간다.
                # 그 차를 완전히 지나칠 때까지 직진한 뒤 스윙을 시작한다.
                #
                # pass_dist(옆을 지날 때의 최소거리)를 여기서부터 재기 시작한다.
                # 이게 캘리브레이션의 기준량이다. 발견 순간의 거리를 그냥 쓰면
                # 트리거가 걸린 각도에 따라 d=L/sin(θ) 로 최대 6% 흔들리는데,
                # 정측면 최소거리는 그 각도 의존성이 없는 순수 횡거리다.
                self.pass_dist = st["dmin"]
                self.car_track.reset()           # 여기서 그 차를 물고 시작
                self._go(self.APPROACH)          # 멈추지 않고 그대로 직진 유지
            elif self._elapsed() > C.SEARCH_TIMEOUT_S:
                print("[fsm] SEARCH 타임아웃 — 우측 클러스터 못 봄. 정지")
                self.car.motor_stop()
                self._go(self.CRUISE)

        elif self.state == self.APPROACH:
            # 1.5단계: 처음 발견한 그 차를 **완전히 지나칠 때까지** 직진한다.
            #
            # 판정은 그 차의 차체기준 방위각으로 한다. 직진하며 우측의 물체를
            # 지나가면 방위각이 90°(정우측) → 0°(정후방) 으로 내려온다.
            # PASSED_CAR_DEG 아래로 내려오면 "다 지나갔다".
            #
            # 시간(초)이 아니라 각도로 재는 이유: 시간은 속도·배터리·노면에
            # 따라 이동거리가 달라져 매번 다른 지점에서 꺾는다. 방위각은 매
            # 프레임 새로 재는 절대값이라 몇 프레임 놓쳐도 즉시 복구된다.
            # (SWING_STOP_DIST 를 거리로 잰 것과 같은 이유)
            self.car.steer_center()
            self.car.motor_forward(C.FWD_SPEED)
            d = self.car_track.update(clusters)
            b = self.car_track.bearing

            # ★캘리브레이션 기준량★ 옆을 지나며 가장 가까웠던 거리를 계속 갱신.
            # 정측면(방위 90도)을 지날 때가 최소이고, 통과각(77도 등)은 그보다
            # 뒤라서 스윙을 시작할 때쯤이면 이미 확정돼 있다.
            if d is not None and (self.pass_dist is None or d < self.pass_dist):
                self.pass_dist = d
            self._measure_heading(clusters)   # 계측만 — 판정엔 안 씀

            deg_eff, _, interp = self._start_comp()
            if C.PASSED_CAR_DEG > 0:
                passed = b is not None and b <= deg_eff
            else:
                passed = False        # 각도 조건 끔 → 타임아웃 시간만큼 직진
            self.status = (
                f"approach  방위 {'--' if b is None else f'{b:+.0f}'}"
                f"→{deg_eff:.0f}도{'*' if interp else ''}  dist {_mm(d)}"
                f"  최소 {_mm(self.pass_dist)}"
                f"  면각 {'--' if self.head_dev is None else f'{self.head_dev:+.0f}'}"
                f"  추적={'O' if self.car_track.tracking else 'X'}"
                f"  {self._elapsed():.1f}/{C.APPROACH_TIMEOUT_S:.1f}s")

            if self._hold(passed, C.TRIGGER_HOLD_S):
                self._report_calib(b)
                self.car.motor_stop()
                self._go(self.SWING_OUT)
            elif self._elapsed() > C.APPROACH_TIMEOUT_S:
                print(f"[fsm] APPROACH 타임아웃 — 방위 "
                      f"{'--' if b is None else f'{b:+.0f}'}도 "
                      f"(목표 {deg_eff:.0f} 이하), 스윙 진행")
                self._report_calib(b)
                self.car.motor_stop()
                self._go(self.SWING_OUT)

        elif self.state == self.SWING_OUT:
            # 좌 풀조향 전진 → 처음 발견한 그 차와 SWING_STOP_DIST 만큼 벌어지면 정지
            self.car.steer_full_left()
            self.car.motor_forward(C.FWD_SPEED)
            d = self.car_track.update(clusters)
            _, stop_eff, _ = self._start_comp()
            self.status = (f"swing_out  dist {_mm(d)}/{stop_eff:.0f}mm"
                           f"  추적={'O' if self.car_track.tracking else 'X'}"
                           f"  재획득 {self.car_track.reacquired}")
            if d is not None and d >= stop_eff:
                self.car.motor_stop()
                self._seen_left = False
                self._seen_right = False
                self._wheels_centered = False
                self._go(self.REVERSE_IN)
            elif self._elapsed() > C.SWING_TIMEOUT_S:
                print(f"[fsm] SWING_OUT 타임아웃 — dist {_mm(d)}mm "
                      f"(목표 {stop_eff:.0f}), 후진 진행")
                self.car.motor_stop()
                self._seen_left = False
                self._seen_right = False
                self._wheels_centered = False
                self._go(self.REVERSE_IN)

        elif self.state == self.REVERSE_IN:
            # 정렬되면 바퀴만 중립으로 (과회전 방지). REVERSE_ALIGN_TOL=0 이면
            # 끝까지 풀락 후진 = 사양 원문 동작.
            # dev 는 상태줄에도 띄운다. 정렬 판정이 "늦게 걸리는" 게 문턱값
            # 때문인지, 애초에 각도를 못 재고 있어서인지 구분하려면 값의
            # 궤적이 보여야 한다. (TOL=12 인데 dev=6 에서 걸리면 문턱이
            # 병목이 아니라 측정이 늦은 것 — TOL 을 올려도 안 바뀐다)
            dev = None
            wall_c = None
            if not self._wheels_centered and C.REVERSE_ALIGN_TOL > 0:
                wall_c = P.largest_near_cluster(
                    clusters, [F.FLANK_RIGHT, F.FLANK_LEFT], _WALL_NEAR_MM)
                # 점이 몇 개뿐인 클러스터의 주축은 방향이 아니라 잡음이다.
                # 여기서 안 거르면 산발 점 2~3개가 그럴듯한 각도로 둔갑한다.
                if wall_c is not None and len(wall_c) >= _WALL_MIN_PTS:
                    dev = P.wall_dev(wall_c, F.REVERSE_CAR_DEG)
                # ★단발 판정 금지★ dev 는 프레임마다 다른 클러스터를 잡아
                # 45 → 60 → 34 처럼 튄다(실측). 한 프레임만 보고 래치를 걸면
                # 노이즈가 우연히 TOL 밑을 찍는 순간 확정돼서, 발화 지점이
                # 실행마다 dev=2/6/10 으로 제각각이 된다. 연속 유지를 본다.
                if self._align_hold(dev is not None and dev < C.REVERSE_ALIGN_TOL,
                                    C.REVERSE_ALIGN_HOLD_S):
                    self._wheels_centered = True
                    print(f"[fsm] 슬롯과 정렬(dev={dev:.0f}도, "
                          f"{C.REVERSE_ALIGN_HOLD_S:.2f}s 유지) → 바퀴 중립")
            if self._wheels_centered:
                # 그냥 steer_center() 를 쓰면 안 된다 — 여기선 우 풀락에서
                # **올라오는** 접근이라 펌웨어 데드밴드에 걸려 중립보다 우측
                # (실측 pot=740, 호밍 기준 752)에서 서버리고, 그 상태로 계속
                # 후진해 슬롯에 비스듬히 들어간다. pot 으로 직접 겨눈다.
                if C.REVERSE_CENTER_POT > 0:
                    self.car.steer_pot(C.REVERSE_CENTER_POT)
                else:
                    self.car.steer_center()
            else:
                self.car.steer_full_right()
            self.car.motor_reverse(C.REV_SPEED)

            left = P.sector_stats(clusters, [F.FLANK_LEFT], C.FLANK_NEAR_DIST)
            right = P.sector_stats(clusters, [F.FLANK_RIGHT], C.FLANK_NEAR_DIST)
            if left["n"] > 0:
                self._seen_left = True
            if right["n"] > 0:
                self._seen_right = True
            # ★양쪽을 다 봤어야 한다★ 차 두 대 사이로 들어가는 기동이므로
            # 좌·우 옆차가 각각 한 번씩은 flank 섹터를 지나야 정상이다.
            # 예전엔 (좌 or 우) 하나만 봐도 '봤음'이 켜졌는데, 실측 로그가
            # "왼쪽만 보임 → 오른쪽만 보임" 순서라서 그 인계 공백(양쪽 0)
            # 두 프레임이면 오른쪽 차를 보기도 전에 완전진입으로 오판했다.
            clear = (self._seen_left and self._seen_right
                     and left["n"] == 0 and right["n"] == 0)
            seen_txt = ("L" + ("O" if self._seen_left else "-")
                        + " R" + ("O" if self._seen_right else "-"))
            # 조향이 명령대로 실제로 갔는지. 중립 복귀는 "우측→중앙" 접근이라
            # 유격이 가장 크게 남는 방향이다(실측 보정량 45 vs 좌측 13).
            # 명령만 보고는 알 수 없어서 pot 실측을 같이 띄운다.
            pot = getattr(self.car, "pot", None)
            ctr_txt = ""
            if self._wheels_centered:
                ctr_txt = " 바퀴중립"
                if pot is not None:
                    ctr_txt += f"(pot={pot})"
            # dev: 숫자=측정됨 / '-'=벽 클러스터 없음 / '?'=점 부족해 각도 실패
            if self._wheels_centered:
                dev_txt = "-"
            elif dev is not None:
                dev_txt = f"{dev:.0f}"
            elif wall_c is None:
                dev_txt = "-"
            else:
                dev_txt = "?"
            self.status = (
                f"reverse_in  dev={dev_txt}/{C.REVERSE_ALIGN_TOL:.0f}"
                f" | L n={left['n']} d={_mm(left['dmin'])}"
                f" | R n={right['n']} d={_mm(right['dmin'])}"
                f" | {seen_txt}{ctr_txt}")
            if self._hold(clear, C.FLANK_CLEAR_HOLD_S):
                self.car.motor_stop()
                self._go(self.WAIT)
            elif self._elapsed() > C.REVERSE_TIMEOUT_S:
                print("[fsm] REVERSE_IN 타임아웃 — 양옆 소멸 미확인, 정지")
                self.car.motor_stop()
                self._go(self.WAIT)

        elif self.state == self.WAIT:
            self.car.motor_stop()
            self.status = f"wait {self._elapsed():.1f}/{C.WAIT_S:.1f}s"
            if self._elapsed() >= C.WAIT_S:
                self._go(self.EXIT_FWD)

        elif self.state == self.EXIT_FWD:
            self.car.steer_center()
            self.car.motor_forward(C.FWD_SPEED)
            self.status = f"exit_fwd {self._elapsed():.1f}/{C.EXIT_FWD_S:.1f}s"
            if self._elapsed() >= C.EXIT_FWD_S:
                self._go(self.EXIT_TURN)

        elif self.state == self.EXIT_TURN:
            self.car.steer_full_right()
            self.car.motor_forward(C.FWD_SPEED)
            self.status = f"exit_turn {self._elapsed():.1f}/{C.EXIT_TURN_S:.1f}s"
            if self._elapsed() >= C.EXIT_TURN_S:
                self._go(self.CRUISE)

        elif self.state == self.CRUISE:
            self.car.steer_center()
            self.car.motor_forward(C.FWD_SPEED)
            self.status = "cruise — 직진 유지 (ESC 로 정지)"

        self.car.tick()                      # 워치독 하트비트
        return self.state
