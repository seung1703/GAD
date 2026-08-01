"""
homing.py -- 미션 시작 시 조향 유격 정렬 (호밍).

[왜 필요한가]
  조향 링키지에 유격이 있어서, 같은 "중립" 명령이라도 어느 쪽에서 진입하느냐에
  따라 바퀴가 실제로 멈추는 각도가 다르다. 실차 정지 테스트 결과:

      우측 → 중앙 : 바퀴가 덜 따라옴 (중앙에 못 미침)
      좌측 → 중앙 : 바퀴가 중앙에 정렬됨

  straight_trim_test_v5 의 튜닝 결과와도 일치한다 — 우측 접근 보정량이 45,
  좌측 접근이 13으로 우측이 3.5배 더 필요했다. 즉 좌측 진입이 "보정이 가장
  덜 필요한 경로"다.

  주차는 FSM 이 시퀀스/타이밍 기반이라 매 시도의 초기 자세가 같아야 튜닝이
  된다. 특히 SEARCH 단계는 steer_center() 로 직진하며 차를 찾는데(:214),
  여기서 바퀴가 틀어져 있으면 발견 위치·각도가 매번 달라지고, 그게 SWING_OUT
  시작 자세와 후진 진입 각도까지 전부 흔든다.

  그래서 미션 시작 직후 **좌 풀락 → 중립** 을 1회 돌려서, 항상 같은 방향에서
  중립에 진입한 상태로 출발한다.

[한계]
  t=0 의 상태만 결정론적으로 만든다. 주행 중 방향이 바뀔 때마다 유격은 다시
  생긴다 (SWING_OUT 좌 → REVERSE_IN 우 전환 등). 그건 기구 조정이나 펌웨어
  보정의 몫이고 호밍으로는 못 잡는다.

  SWING_OUT 은 풀락(포화) 명령이라 유격을 먹어도 결국 스토퍼까지 가서 붙는다.
  손해가 도착 지연뿐이라, SEARCH 의 직진 정확도를 얻는 쪽이 이득이다.

[튜닝]
  calib.json 의 park_home_center_pot 하나만 만지면 된다. 바퀴가 덜 오면
  낮추고, 지나치면 올린다. 나머지 값은 그대로 둬도 된다.
"""

import time

import config as C

_POLL_S = 0.02        # 폴링 주기 (50Hz). 펌웨어 pot 텔레메트리는 10Hz다
_STILL_DELTA = 1      # 이 이하로만 변하면 "멈춘 것"으로 본다 (ADC 카운트)


def _move_to(car, pot_target, label, verbose):
    """pot_target 으로 조향을 보내고 '멈출 때까지' 기다린다.

    고정 sleep 이 아니라 pot 텔레메트리로 실제 정지를 확인한다. 모터가 멈춘
    뒤 감김이 되풀리며 pot 이 조금 되돌아오는 구간이 있어서, "목표 근처 도달"
    만으로 끊으면 최종 위치를 잘못 읽는다. 그래서 값이 settle_s 동안 변하지
    않을 때까지 본다.

    반환: (최종 pot 또는 None, 성공 여부)
    """
    car.steer_pot(pot_target)   # pot→정규화 역산은 arduino_iface 가 담당

    t0 = time.time()
    last_pot = None
    stable_since = None

    while True:
        car.tick()                  # 워치독(400ms) 유지 + 명령 재전송
        car.link.read_telemetry()   # "P:<pot>" 파싱 → car.link.pot
        pot = car.link.pot
        now = time.time()

        if pot is not None:
            if last_pot is not None and abs(pot - last_pot) <= _STILL_DELTA:
                if stable_since is None:
                    stable_since = now
                elif (now - stable_since >= C.HOME_SETTLE_S
                      and abs(pot - pot_target) <= C.HOME_TOL):
                    # 멈췄고 목표 범위 안 → 도달
                    if verbose:
                        print(f"[homing] {label} 도달  pot={pot} "
                              f"(목표 {pot_target}, {now - t0:.1f}s)")
                    return pot, True
            else:
                stable_since = None
            last_pot = pot

        if now - t0 > C.HOME_TIMEOUT_S:
            if pot is None:
                print(f"[homing] ★{label} 실패★ pot 텔레메트리가 안 옴. "
                      "아두이노 연결과 펌웨어 'P:' 송신을 확인할 것")
            else:
                print(f"[homing] ★{label} 타임아웃★ pot={pot} "
                      f"(목표 {pot_target}, 허용 ±{C.HOME_TOL}). "
                      "링키지 걸림이나 pot 캘리브레이션 확인")
            return pot, False

        time.sleep(_POLL_S)


def run_homing(car, verbose=True):
    """좌 풀락 → 중립. 미션 시작 직후 1회 호출한다.

    반환: 최종 pot 값 (실패하거나 건너뛰면 None)
    """
    if not C.HOME_ENABLE:
        if verbose:
            print("[homing] 비활성 (calib.json park_home_enable=0) — 건너뜀")
        return None
    if getattr(car.link, "dry", False):
        if verbose:
            print("[homing] dry-run — 건너뜀")
        return None

    if verbose:
        print(f"[homing] 시작: 좌 풀락({C.HOME_LEFT_POT}) "
              f"→ 중립({C.HOME_CENTER_POT})")

    car.motor_stop()   # 조향만 움직인다. 구동은 반드시 정지 상태

    _, ok_left = _move_to(car, C.HOME_LEFT_POT, "좌 풀락", verbose)
    pot, ok_center = _move_to(car, C.HOME_CENTER_POT, "중립", verbose)

    # 실패해도 미션은 계속 간다 — 호밍은 정밀도 개선이지 안전 요건이 아니다.
    # 다만 초기 자세가 안 맞은 채로 출발한다는 걸 로그에 남긴다.
    if ok_left and ok_center:
        if verbose:
            print(f"[homing] 완료. 이 pot({pot})에서 바퀴가 똑바르지 않으면 "
                  "calib.json 의 park_home_center_pot 을 조정할 것")
    else:
        print("[homing] ☆ 호밍 미완료 — 초기 조향 자세가 평소와 다를 수 있음 ☆")

    return pot if ok_center else None
