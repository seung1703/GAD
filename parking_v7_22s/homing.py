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


def _land_on(car, target, approach, verbose):
    """중립 목표 pot 에 **정확히** 착지시킨다.

    왜 한 번에 안 되나:
      펌웨어는 |오차| <= 데드밴드(8) 면 모터를 끊는다. 그래서 아래(우측)에서
      올라오면 목표보다 6~8 **모자란** 곳에서 서버린다. 실측으로도 명령 758 →
      착지 752 였다. 한 번 명령해서는 원하는 값에 못 선다.

    어떻게 하나:
      서고 나면 실제 pot 을 읽어서 모자란 만큼 명령을 더 밀어준다. 목표에
      들어올 때까지 최대 HOME_RETRY 번 반복한다.

    ★반드시 접근 방향으로만 밀어준다★
      넘어갔다고 반대로 되돌리면 유격이 반대쪽으로 눌리면서 물리적 바퀴각이
      확 달라진다(실측 보정량이 방향에 따라 45 vs 13 이었다). 호밍의 목적이
      "항상 같은 방향에서 진입해 같은 상태로 시작"하는 것이므로, 넘어간 경우엔
      되돌리지 않고 그대로 두고 경고만 남긴다.

    approach: +1 = 아래(우측)에서 위로 올라감 / -1 = 위(좌측)에서 아래로 내려감
    """
    cmd = target
    pot = None
    for attempt in range(C.HOME_RETRY + 1):
        pot, _ = _move_to(car, cmd, "중립", verbose and attempt == 0)
        if pot is None:
            return None, False
        err = target - pot                    # + 면 아직 덜 온 것(목표가 위)
        if abs(err) <= C.HOME_LAND_TOL:
            if verbose:
                print(f"[homing] 중립 착지  pot={pot} "
                      f"(목표 {target}, 시도 {attempt + 1}회)")
            return pot, True
        # 접근 방향으로 모자랄 때만 더 민다. 반대면 되돌리지 않는다.
        if err * approach <= 0:
            print(f"[homing] ☆ 중립 pot={pot} 이 목표 {target} 를 지나침 "
                  f"— 되돌리면 유격 상태가 바뀌므로 그대로 둡니다 ☆")
            return pot, False
        cmd += err
        cmd = max(C.POT_RIGHT, min(C.POT_LEFT, cmd))
        if verbose:
            print(f"[homing]   {pot} → 목표 {target} 까지 {err:+d} 부족, "
                  f"명령을 {cmd} 로 올려 재시도")
    print(f"[homing] ★중립 착지 실패★ pot={pot} (목표 {target}). "
          f"park_home_retry 를 늘리거나 목표를 조정하세요")
    return pot, False


def run_homing(car, verbose=True):
    """좌 풀락 → 중립. 미션 시작 직후 1회 호출한다.

    반환: 최종 pot 값 (실패하거나 건너뛰면 None)
    """
    if not C.HOME_ENABLE:
        if verbose:
            print("[homing] 비활성 — 건너뜀 (카메라 직진 보조가 대신합니다). "
                  "카메라가 안 되면 calib.json park_home_enable=1 로 켜세요")
        return None
    if getattr(car.link, "dry", False):
        if verbose:
            print("[homing] dry-run — 건너뜀")
        return None

    # ── 순서 결정 ──────────────────────────────────────────────
    # park_home_dir = 중립으로 **마지막에 진입하는 방향**.
    # park_home_sweep = 1 이면 그 전에 반대쪽 끝까지 한 번 털고 온다.
    #
    #   sweep=1, dir="left"  →  우 풀락 → 좌 풀락 → 중립
    #   sweep=1, dir="right" →  좌 풀락 → 우 풀락 → 중립
    #   sweep=0              →  dir 쪽 풀락 → 중립
    #
    # ★양 끝을 터는 이유★ 링키지·유격을 양방향으로 한 번씩 밀어놓으면
    #   전원을 켠 직후의 어중간한 상태가 지워지고, 매번 같은 조건에서
    #   중립 진입이 시작된다.
    #
    # ★★ 진입 방향과 목표 pot 은 반드시 한 세트다 ★★
    #   같은 pot 값이라도 어느 쪽에서 왔느냐에 따라 실제 바퀴각이 다르다
    #   (실측 보정량 우측접근 45 vs 좌측접근 13, 차이 32카운트).
    #   실측: 좌측 진입 직진 ≈ pot 740 / 우측 진입 직진 ≈ pot 771.
    #   방향을 바꾸면 park_home_center_pot 도 같이 바꿔야 한다.
    if C.HOME_DIR == "right":
        first_pot, first_label = C.HOME_LEFT_POT, "좌 풀락"
        last_pot, last_label, approach = C.HOME_RIGHT_POT, "우 풀락", +1
    else:
        first_pot, first_label = C.HOME_RIGHT_POT, "우 풀락"
        last_pot, last_label, approach = C.HOME_LEFT_POT, "좌 풀락", -1

    if verbose:
        seq = (f"{first_label}({first_pot}) → " if C.HOME_SWEEP else "")
        print(f"[homing] 시작: {seq}{last_label}({last_pot}) "
              f"→ 중립 {C.HOME_CENTER_POT} 에 착지")

    car.motor_stop()   # 조향만 움직인다. 구동은 반드시 정지 상태

    ok_lock = True
    if C.HOME_SWEEP:
        _, ok_first = _move_to(car, first_pot, first_label, verbose)
        ok_lock = ok_lock and ok_first
    _, ok_last = _move_to(car, last_pot, last_label, verbose)
    ok_lock = ok_lock and ok_last
    pot, ok_center = _land_on(car, C.HOME_CENTER_POT, approach, verbose)

    # 실패해도 미션은 계속 간다 — 호밍은 정밀도 개선이지 안전 요건이 아니다.
    # 다만 초기 자세가 안 맞은 채로 출발한다는 걸 로그에 남긴다.
    if ok_lock and ok_center:
        if verbose:
            print(f"[homing] 완료. 이 pot({pot})에서 바퀴가 똑바르지 않으면 "
                  "calib.json 의 park_home_center_pot 을 조정할 것")
    else:
        print("[homing] ☆ 호밍 미완료 — 초기 조향 자세가 평소와 다를 수 있음 ☆")

    return pot if ok_center else None
