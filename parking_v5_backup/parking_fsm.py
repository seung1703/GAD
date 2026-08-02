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
_CAR2_HALF_DEG = 60.0        # 2번차 탐색 섹터 반폭. 우측 ±this.
                             #   SWING_TRACK(345~105)은 **앞쪽 우측을 못 덮는다**.
                             #   2번차는 처음에 방위 120~140 쯤(앞-우)에 있다가
                             #   전진하며 90 → 그 아래로 내려오므로 더 넓게 본다
_SWING_JUMP_MAX = 25         # 스윙 회전량 측정: 프레임간 이보다 튀면 다른 면을
                             #   잡은 것으로 보고 버린다 (앞면↔옆면은 90도 점프)
_HEAD_NEAR_MM = 1800         # [계측] 차체 틀어짐을 잴 때 볼 최대 거리(mm).
                             #   첫 차가 1400mm 근처에 있으니 여유를 조금 둔다


CAR2_SECTOR = F.sector(F.RIGHT_CAR_DEG, _CAR2_HALF_DEG)


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
        self.cluster = None       # 이번 프레임에 고른 클러스터 (벽면각 측정용)
        self._prev_c = None       # 추적 중인 클러스터 중심 (xy)
        self.tracking = False
        self.reacquired = 0

    @staticmethod
    def _center(c):
        return P.cxy(c).mean(0)

    def update(self, clusters, exclude_center=None, prefer="near",
               sectors=None):
        """→ 추적 중인 차와의 최근접 거리(mm) 또는 None.

        exclude_center: 이 xy 근처(_TRACK_GATE_MM) 클러스터는 후보에서 뺀다.
                        두 번째 차를 추적할 때 첫 차를 걸러내는 용도.
        prefer: 최초 획득 규칙. "near"=가장 가까운 것,
                "front"=방위각이 가장 큰 것(=더 앞쪽/우측. 두 번째 차용)
        """
        # 사분면 안에 점이 있는 클러스터만 후보
        sec = sectors if sectors is not None else [F.SWING_TRACK]
        cand = [c for c in clusters if len(c) >= 2
                and P.sector_stats([c], sec)["n"] > 0]
        if exclude_center is not None:
            cand = [c for c in cand
                    if float(np.hypot(*(self._center(c) - exclude_center)))
                    > _TRACK_GATE_MM]
        if not cand:
            self.tracking = False
            self.dist = None
            self.bearing = None
            self.cluster = None
            return None

        pick = None
        if self._prev_c is not None:                 # 같은 물체 이어가기
            best_d = _TRACK_GATE_MM
            for c in cand:
                d = float(np.hypot(*(self._center(c) - self._prev_c)))
                if d < best_d:
                    pick, best_d = c, d
        if pick is None:                             # 최초 획득 또는 재획득
            if prefer == "front":
                # 두 번째 차 = 첫 차보다 앞(우측)에 있으므로 방위각이 더 크다
                def _bear(c):
                    cx, cy = self._center(c)
                    return float((np.degrees(np.arctan2(cx, cy)) + 180.0)
                                 % 360.0 - 180.0)
                pick = max(cand, key=_bear)
            else:
                pick = min(cand, key=lambda c: float(c[:, 3].min()))
            if self._prev_c is not None:
                self.reacquired += 1

        self._prev_c = self._center(pick)
        self.tracking = True
        self.cluster = pick
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

    def __init__(self, car, guide=None):
        self.car = car
        self.guide = guide           # 카메라 직진 보조 (없으면 None)
        self.state = self.SEARCH
        self.t_state = time.time()
        self._hold_t0 = None
        self._align_t0 = None        # 정렬 판정 전용 홀드 타이머
        self.car_track = TrackedCar()
        # 두 번째 차(슬롯 반대편). 스윙 종료를 "얼마나 앞으로 갔나"로 재기 위함.
        # 첫 차와의 거리 하나로는 옆으로 밀린 건지 앞으로 간 건지 구분이 안 된다.
        self.car2_track = TrackedCar()
        self.car2_bear_min = None    # 2번차 방위각의 최솟값(래치).
                                     #   전진할수록 내려가므로 최솟값이 곧
                                     #   "가장 많이 전진했던 시점"이다.
                                     #   섹터 밖으로 나가 놓쳐도 조건이 안 풀린다
        self.car2_seen = False       # 스윙 중 2번차를 한 번이라도 봤나
        # 3단계 종료 판정: 좌·우 옆차를 **각각** 봤는지 따로 래치한다.
        # 하나로 합치면(예전 _flank_seen) 한쪽만 보고도 "봤음"이 켜져서,
        # 좌→우 인계 순간의 공백에 조기 정지한다.
        self._seen_left = False
        self._seen_right = False
        self._rear_t0 = None         # 후방 한계 접촉 지속 시간 타이머
        self._center_t = None        # 바퀴가 중립으로 풀린 시각
        self.pass_dist = None        # ★캘리브레이션 기준량★ 첫 차 옆을
                                     #   지날 때의 최소거리(mm)
        self.head_dev = None         # [계측] 차체 틀어짐(도). 판정엔 미사용
        self._head_samples = []      # 위 값의 표본 (중앙값용)
        self.swing_rot = None        # 스윙 시작 후 회전량(도)
        self._swing_a0 = None        # 스윙 시작 시점의 1번차 면 방향
        self._swing_aprev = None     # 직전 프레임 면 방향 (급변 감지용)
        # 2번차로 이어 재기 — 1번차가 멀어져 안 보이면 회전량이 얼어붙는다.
        # 실측: 1번차가 2600~3000mm(MAX_DISTANCE 경계)에서 들락날락하며
        # 회전량이 32에서 멈춰 목표 35를 영영 못 넘고 타임아웃까지 갔다.
        self._swing2_a0 = None       # 2번차 기준 면 방향
        self._swing2_aprev = None
        self._swing2_off = 0.0       # 2번차 기준을 잡던 시점의 회전량
        self.rot_src = "-"           # 지금 어느 차로 재고 있나 (표시용)
        self._wheels_centered = False
        self.status = ""
        self.emergency = False
        self.back_mode = False       # --back: 마지막 CRUISE 를 후진으로
        self.arm_at = 0.0            # 이 시각 전에는 SEARCH 판정을 안 한다
                                     #   (달리기는 한다 — 제자리 대기 아님)

    def arm_after(self, sec):
        """지금부터 sec 초 동안 SEARCH 의 '차 발견' 판정을 보류한다.

        ★차는 그동안에도 직진으로 달린다★ 제자리에서 기다리는 게 아니다.
        호밍으로 바퀴가 방금 크게 움직였고 라이다도 막 회전을 시작한 참이라
        첫 프레임에 헛것이 섞인다. 그걸 "차 발견" 으로 오판해 곧바로 스윙에
        들어가는 것만 막으면 되고, 그동안 앞으로 나아가는 건 손해가 아니다.
        """
        self.arm_at = time.time() + max(0.0, float(sec))

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

    def _drive_straight(self):
        """직진 구간의 조향. 카메라 보조가 살아 있으면 그 값을, 아니면 중립.

        ★스윙부터는 절대 안 쓴다★ SWING_OUT 이후는 일부러 크게 꺾는 기동이라
        직진 유지가 방해만 된다. guide.steer_for() 가 상태를 보고 걸러 준다.
        """
        s = None if self.guide is None else self.guide.steer_for(self.state)
        if s is None:
            self.car.steer_center()
            return ""
        self.car.steer_raw_norm(s)   # 부호 보정 없이 그대로 (LAT_SIGN 으로 이미 맞춤)
        return f"  cam {s:+.2f}"

    def _start_comp(self):
        """시작 위치에 따른 스윙정지거리 → (미사용, 거리[mm], 보간여부).

        ※ 통과각 자리는 예전에 각도 기반 APPROACH 를 쓰던 흔적이다. 지금은
          시간 기반(APPROACH_S)이라 안 쓰지만, CAL_*_PASSED_DEG 를 그대로
          보간해 두면 나중에 되살릴 때 캘리브레이션을 다시 안 해도 된다.

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
        deg, stop = 0.0, C.SWING_STOP_DIST
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

    def _swing_rotation(self):
        """스윙을 시작한 뒤 **차가 얼마나 돌았는지**(도). 못 재면 None.

        ★왜 이 값인가★
          기존 종료 조건(SWING_STOP_DIST)은 첫 차와의 **거리**라서, 시작 위치가
          옆으로 30cm 밀리면 시작 거리부터 달라져 회전량이 통째로 바뀐다.
          "얼마나 돌았나"는 위치와 무관한 순수 회전량이라 시작점이 어디든
          같은 자세에서 스윙이 끝난다.

        ★어떻게 재나★
          추적 중인 첫 차의 **면 방향**을 스윙 시작 시점과 비교한다. 그 차는
          가만히 있으므로, 면 방향이 바뀐 만큼이 곧 내 차가 돌아간 각도다.
          절대 기준이 필요 없어서 어느 면을 보고 있든(앞면/옆면) 상관없다 —
          스윙 내내 **같은 면**을 보고 있기만 하면 된다.

        ★안전장치★
          · 점이 적은 클러스터의 주축은 방향이 아니라 잡음이다 → 최소 점수
          · 앞면↔옆면으로 바뀌면 90도가 튄다 → 프레임간 급변 거부
          · 회전은 한 방향으로만 늘어난다 → 줄어드는 값 거부
        """
        def face(tr):
            c = tr.cluster
            if c is None or len(c) < _WALL_MIN_PTS:
                return None
            return P.wall_car_deg(c)

        a1, a2 = face(self.car_track), face(self.car2_track)

        # ── 2번차 기준선은 볼 수 있을 때 미리 잡아둔다 ──
        # 1번차가 아직 멀쩡할 때 잡아야 offset(그 시점의 회전량)이 정확하다.
        if a2 is not None and self._swing2_a0 is None:
            self._swing2_a0 = a2
            self._swing2_aprev = a2
            self._swing2_off = self.swing_rot or 0.0

        def advance(a, a0_attr, aprev_attr, off):
            """면 방향 a 로 회전량을 계산. 급변(다른 면)이면 None."""
            aprev = getattr(self, aprev_attr)
            if P.angle_between(a, aprev) > _SWING_JUMP_MAX:
                return None
            setattr(self, aprev_attr, a)
            return off + P.angle_between(a, getattr(self, a0_attr))

        rot = None
        if a1 is not None:
            if self._swing_a0 is None:          # 스윙 시작 기준
                self._swing_a0 = a1
                self._swing_aprev = a1
                self.swing_rot = 0.0
                self.rot_src = "1"
                return 0.0
            rot = advance(a1, "_swing_a0", "_swing_aprev", 0.0)
            if rot is not None:
                self.rot_src = "1"

        # 1번차가 안 되면 2번차로 이어 잰다. 둘 다 가만히 있는 물체라
        # 내가 돌아간 각도는 어느 쪽으로 재든 같다.
        if rot is None and a2 is not None and self._swing2_a0 is not None:
            rot = advance(a2, "_swing2_a0", "_swing2_aprev", self._swing2_off)
            if rot is not None:
                self.rot_src = "2"

        if rot is None:                         # 이번 프레임엔 못 잼
            return self.swing_rot

        # 회전은 되돌아가지 않는다. 줄어드는 건 잡음이므로 최댓값을 유지
        self.swing_rot = max(self.swing_rot or 0.0, float(rot))
        return self.swing_rot

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
        stop, interp = self._start_comp()[1], self._start_comp()[2]
        s = sorted(self._head_samples)
        head = s[len(s) // 2] if s else None       # 중앙값 (튀는 값에 강함)
        print(f"[fsm] 첫 차 통과(직진 {C.APPROACH_S:.1f}s) → 좌 스윙 시작")
        print(f"[fsm] ★캘리브레이션★ 옆을 지날 때 거리 = "
              f"{_mm(self.pass_dist)}mm"
              f"   (이 숫자를 CAL_A_DIST 또는 CAL_B_DIST 에 적으세요)")
        print(f"[fsm]   적용값: 스윙정지(폴백) {stop:.0f}mm"
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
        # --back 크루즈도 후진이므로 후방 감시를 켠다. 안 그러면 눈 감고
        # 뒤로 가는 셈이다. (다만 그땐 "진입 완료" 판정 없이 멈추기만 한다)
        cruising_back = self.back_mode and self.state == self.CRUISE
        if self.state != self.REVERSE_IN and not cruising_back:
            self._rear_t0 = None
            return False
        rd = rear_min_dist(scan)
        if rd is not None and rd < C.REAR_EMERGENCY_DIST:
            self.car.motor_stop()
            self.emergency = True
            now = time.time()
            if self._rear_t0 is None:
                self._rear_t0 = now
            elif cruising_back:
                # 크루즈 후진은 그냥 멈춰 있는다 (다음 단계가 없다)
                self.status = f"cruise ← 후방 {rd:.0f}mm — 정지"
                return True
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
        # ★매 프레임 상태를 알려준다★ steer_for() 는 직진 구간에서만 불리므로,
        # 그것만으로는 스윙에 들어가도 가이드가 자기 상태를 APPROACH 로 알고
        # 계속 제어를 돌린다(실측: 스윙 중 발산 감지가 떴다).
        if self.guide is not None:
            self.guide.state = self.state

        if self.state == self.SEARCH:
            cam = self._drive_straight()
            self.car.motor_forward(C.FWD_SPEED)
            # ★너무 가까운 것은 차가 아니다★ 첫 차는 차량 열 옆을 지나며
            # 보는 것이라 항상 1300~1500mm 쯤에 있다. 그보다 훨씬 가까운
            # 검출은 자기 차체 반사·옆에 선 사람·라이다 초기 잡음이다.
            # (실측: 출발 직후 723mm 에 점 1개짜리가 1초쯤 떴다 사라졌다.
            #  n>0 조건만으론 그게 "첫 차 발견"으로 통과한다)
            st = P.sector_stats(clusters, [F.DETECT_RIGHT],
                                min_dist=C.TRIGGER_MIN_DIST)

            # 워밍업 구간: 달리기는 하되 발견 판정만 보류한다.
            warm = self.arm_at - time.time()
            if warm > 0:
                self._hold_t0 = None      # 워밍업 중 홀드가 쌓이지 않게
                self.status = (f"search  [워밍업 {warm:.1f}s — 라이다 무시] "
                               f"우측 n={st['n']} dmin={_mm(st['dmin'])}{cam}")
            elif self._hold(st["n"] > 0, C.TRIGGER_HOLD_S):
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
            elif warm <= 0 and self._elapsed() > C.SEARCH_TIMEOUT_S:
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
            cam = self._drive_straight()
            self.car.motor_forward(C.FWD_SPEED)
            d = self.car_track.update(clusters)
            b = self.car_track.bearing

            # ★캘리브레이션 기준량★ 옆을 지나며 가장 가까웠던 거리를 계속 갱신.
            # 정측면(방위 90도)을 지날 때가 최소이고, 통과각(77도 등)은 그보다
            # 뒤라서 스윙을 시작할 때쯤이면 이미 확정돼 있다.
            if d is not None and (self.pass_dist is None or d < self.pass_dist):
                self.pass_dist = d
            self._measure_heading(clusters)   # 계측만 — 판정엔 안 씀

            common = (f"  최소 {_mm(self.pass_dist)}"
                      f"  면각 {'--' if self.head_dev is None else f'{self.head_dev:+.0f}'}"
                      f"  추적={'O' if self.car_track.tracking else 'X'}")
            self.status = (f"approach {self._elapsed():.1f}/{C.APPROACH_S:.1f}s"
                           f"  방위 {'--' if b is None else f'{b:+.0f}'}도"
                           f"{common}{cam}")
            if self._elapsed() >= C.APPROACH_S:
                self._report_calib(b)
                self.car.motor_stop()
                self._swing_a0 = None      # 회전량 기준을 여기서 새로 잡는다
                self.swing_rot = None
                self.car2_track.reset()
                self.car2_bear_min = None
                self.car2_seen = False
                self._go(self.SWING_OUT)

        elif self.state == self.SWING_OUT:
            # 좌 풀조향 전진 → 처음 발견한 그 차와 SWING_STOP_DIST 만큼 벌어지면 정지
            self.car.steer_full_left()
            self.car.motor_forward(C.FWD_SPEED)
            # ★상호 배제★ 두 추적기가 같은 물체를 잡으면 안 된다.
            # 실측 로그에서 1번차 dist 가 2338 → 1708 로 급락했는데, 멀어지던
            # 차가 가까워질 리 없다. 1번차 추적기가 2번차를 잡아챈 것이고,
            # 그 바람에 2번차는 실종되고 회전량도 얼어붙었다.
            d = self.car_track.update(
                clusters, exclude_center=self.car2_track._prev_c)
            _, stop_eff, _ = self._start_comp()
            rot = self._swing_rotation()

            # ── 두 번째 차 (슬롯 반대편) ──
            # 첫 차를 제외하고, 방위각이 더 큰(=앞/우측) 클러스터를 문다.
            d2 = self.car2_track.update(
                clusters, exclude_center=self.car_track._prev_c,
                prefer="front", sectors=[CAR2_SECTOR])
            b1 = self.car_track.bearing
            b2 = self.car2_track.bearing
            if b2 is not None:
                self.car2_seen = True
                self.car2_bear_min = (b2 if self.car2_bear_min is None
                                      else min(self.car2_bear_min, b2))
            # 두 차가 벌어져 보이는 각도. 슬롯 입구를 얼마나 정면으로 보고
            # 있는지의 지표라 참고로 같이 띄운다.
            gap_deg = None if (b1 is None or b2 is None) else abs(b2 - b1)
            two_txt = (f"  2번차 {'--' if b2 is None else f'{b2:+.0f}도'}"
                       f" d={_mm(d2)}"
                       f" 사이 {'--' if gap_deg is None else f'{gap_deg:.0f}도'}")

            # ★두 조건을 항상 같이 표시한다★ 거리 모드로 달리면서 "그때 회전량이
            # 몇 도였나"를 로그로 모으면, 그 값이 곧 각도 모드의 목표치가 된다.
            rot_txt = ("--" if rot is None
                       else f"{rot:.0f}({self.rot_src}번)")

            # ★자세 + 위치★ 회전량으로 자세를 맞추고, 두 번째 차(슬롯 반대편)의
            # 방위각으로 "얼마나 앞으로 갔나"를 잡는다. 첫 차와의 거리 하나로는
            # 옆으로 밀린 건지 앞으로 간 건지 구분이 안 된다(거리엔 방향이 없다).
            turned = rot is not None and rot >= C.SWING_STOP_DEG
            if self.car2_seen:
                # 래치된 최솟값으로 판정 — 2번차가 섹터 밖으로 나가 안 보이게
                # 돼도 이미 지나온 사실은 그대로다
                advanced = (self.car2_bear_min is not None
                            and self.car2_bear_min <= C.SWING_CAR2_DEG)
                src = ""
            else:
                # 2번차를 한 번도 못 봄 → 거리 조건으로 폴백.
                # 안 그러면 조건이 영영 안 걸려 타임아웃까지 간다
                advanced = d is not None and d >= stop_eff
                src = "  [2번차 미검출 → 거리폴백]"
            done = turned and advanced
            wait = src
            if turned and not advanced:
                wait += "  [전진 대기]"
            elif advanced and not turned:
                wait += "  [회전 대기]"
            self.status = (f"swing_out  회전 {rot_txt}/{C.SWING_STOP_DEG:.0f}도"
                           f"  2번차 "
                           f"{'--' if self.car2_bear_min is None else f'{self.car2_bear_min:+.0f}'}"
                           f"/{C.SWING_CAR2_DEG:.0f}도"
                           f"  dist {_mm(d)}{two_txt}"
                           f"  재획득 {self.car_track.reacquired}"
                           f"/{self.car2_track.reacquired}{wait}")
            if done:
                print(f"[fsm] ★스윙 종료★ 회전 {rot_txt} / dist {_mm(d)}mm"
                      f" / 2번차 {'--' if b2 is None else f'{b2:+.0f}도'}"
                      f" d={_mm(d2)} / 두 차 사이 "
                      f"{'--' if gap_deg is None else f'{gap_deg:.0f}도'}")
                self.car.motor_stop()
                self._seen_left = False
                self._seen_right = False
                self._wheels_centered = False
                self._center_t = None
                self._go(self.REVERSE_IN)
            elif self._elapsed() > C.SWING_TIMEOUT_S:
                print(f"[fsm] SWING_OUT 타임아웃 — 회전 {rot_txt} "
                      f"(목표 {C.SWING_STOP_DEG:.0f}) / 2번차 "
                      f"{'--' if self.car2_bear_min is None else f'{self.car2_bear_min:+.0f}'}"
                      f" (목표 {C.SWING_CAR2_DEG:.0f}) / dist {_mm(d)}, 후진 진행")
                self.car.motor_stop()
                self._seen_left = False
                self._seen_right = False
                self._wheels_centered = False
                self._center_t = None
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
                    self._center_t = time.time()
                    extra = ("" if C.REVERSE_AFTER_CENTER_S <= 0
                             else f" → 이후 {C.REVERSE_AFTER_CENTER_S:.1f}s 더 후진")
                    print(f"[fsm] 슬롯과 정렬(dev={dev:.0f}도, "
                          f"{C.REVERSE_ALIGN_HOLD_S:.2f}s 유지) → 바퀴 중립{extra}")
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

            # ★시간 기반 종료★ REVERSE_AFTER_CENTER_S > 0 이면, 바퀴가 중립으로
            # 풀린 시점부터 그 시간만큼만 더 후진하고 멈춘다. 양옆 소멸 판정보다
            # 우선한다 — 얼마나 깊이 넣을지를 직접 정하고 싶을 때 쓴다.
            # (후방 비상정지와 REVERSE_TIMEOUT_S 는 그대로 살아 있다)
            timed = None
            if C.REVERSE_AFTER_CENTER_S > 0 and self._center_t is not None:
                timed = time.time() - self._center_t
                clear = timed >= C.REVERSE_AFTER_CENTER_S
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
                if timed is not None:
                    ctr_txt += f" 후진 {timed:.1f}/{C.REVERSE_AFTER_CENTER_S:.1f}s"
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
            # --back 이면 전진 대신 **후진**한다. 디버깅용 — 한 판 끝나고
            # 차를 출발 지점 쪽으로 되돌려 다음 시도를 바로 할 수 있다.
            self.car.steer_center()
            if self.back_mode:
                self.car.motor_reverse(C.REV_SPEED)
                self.status = "cruise ← 후진 (--back, ESC 로 정지)"
            else:
                self.car.motor_forward(C.FWD_SPEED)
                self.status = "cruise — 직진 유지 (ESC 로 정지)"

        self.car.tick()                      # 워치독 하트비트
        return self.state
