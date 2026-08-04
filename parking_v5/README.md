# 주차 미션 v2 — RPLidar 직각 후진 주차 (parking_v2/)

> **v1(`parking/`) 의 복사본에서 갈라진 작업 폴더입니다.** v1 은 손대지 않았으니
> 잘 안 되면 그냥 v1 을 쓰면 됩니다.
>
> v2 에서 추가된 것은 **두 가지뿐**이고, 둘 다 기본값에서는 꺼져 있어서
> **처음 돌리면 v1 과 완전히 똑같이 동작합니다.**
>
> 1. **시작 위치 2점 보간** — 횡으로 30cm 떨어진 두 시작점에서 각각 최적값을
>    재두면, 실제 거리에 따라 그 사이를 자동으로 이어 씁니다.
>    켜는 법은 `config.py` 의 "★★★ 시작 위치 캘리브레이션 ★★★" 박스 참고.
> 2. **차체 틀어짐 계측** — 첫 차의 면 각도로 내 차가 얼마나 틀어졌는지
>    로그에만 찍습니다. **판정에는 일절 안 씁니다.** 값이 믿을 만한지
>    먼저 눈으로 확인하려는 용도입니다.

## 내일 현장에서 할 일 (순서대로)

```
0) 아무것도 안 건드리고 한 번 돌린다        → v1 과 같은지 확인 (회귀 없음)

1) 오른쪽 시작점                            2) 왼쪽 시작점 (30cm 옆)
   · 돌린다                                    · 같은 걸 반복
   · 로그에서 이 줄을 본다:                     · CAL_B_* 에 적는다
       ★캘리브레이션★ 옆을 지날 때 거리 = ____mm
   · 잘 들어갈 때까지 조정
       SWING_STOP_DIST 먼저  (영향 큼)
       PASSED_CAR_DEG 나중
   · 3~5번 더 돌려 거리 평균
   · CAL_A_* 세 칸에 적는다

3) 끝. 두 CAL_*_DIST 가 0 이 아니면 자동으로 켜진다.
   로그에 `보간=ON` 이 뜨면 정상.
```

**같이 봐둘 것 — 차체 틀어짐 계측**
차를 똑바로 놓고 돌렸을 때 `[계측] 차체 틀어짐` 이 매번 ±3도 안으로 모이면
그 각도를 믿어도 된다는 뜻입니다. 값이 제각각이면 그 방법은 접고 2점 보간만
쓰면 됩니다. (판정에 안 쓰니 어느 쪽이든 주차에는 영향 없습니다)

---

## 구조
```
frames.py         ★각도 프레임 단일 진실원천★ raw↔car 변환 + 모든 섹터 정의
config.py         사람이 실제로 만지는 값만 (실측·튜닝값). 내부 상수는 여기 없음
perception.py     라이다 인지: 필터→클러스터→벽면각/섹터질의
lidar_reader.py   수신 스레드(최신 프레임만) + 스캔 로거 + 리플레이어
arduino_iface.py  CarCommander → 통합펌웨어 프로토콜(MegaLink)
parking_fsm.py    미션 상태머신 (조건 기반 + 안전 타임아웃 폴백)
visualize.py      미션 실시간 뷰 (라이다 원시각 극좌표) + 공용 극좌표 장식 함수
lidar_scope.py    각도별 관측 점검 도구 (가림구간 자동추출)
main_parking.py   엔트리 (run/record/replay/calib/sim/straight)
serial_driver.py  obstacle에서 가져온 통합 펌웨어 드라이버
```

## 실행
```bash
python3 main_parking.py --sim            # 무하드웨어 상태전이 확인 (합성 장면)
python3 main_parking.py --calib          # 섹터별 점유/벽면각 실시간
python3 main_parking.py --straight       # FSM 없이 조향중립+직진, 인식 화면만
python3 main_parking.py --record log.pkl # 실주행하며 스캔 기록
python3 main_parking.py --replay log.pkl # 기록 재생 (라이다 없이 알고리즘/FSM 디버깅)
python3 main_parking.py --run            # 실주행 (라이다+아두이노)
python3 main_parking.py --run --dry      # 아두이노 없이 (명령 로그만)
```
실행 환경은 conda env `drive` (rplidar / matplotlib / pynput 설치돼 있음):
`conda run -n drive python main_parking.py --run`

**키 (전역, 창 포커스 무관)**: `s`=시작, `ESC`=정지, `Space`=일시정지/재개
일시정지하면 차를 세우고 라이다 화면을 마지막 프레임에 고정한다 — 그 자리에서
인식 결과를 들여다볼 수 있다. 멈춘 시간은 FSM 타임아웃에서 빼준다.

## 의존성
- `pip install rplidar-roboticia` (라이브 라이다 — 리플레이/sim엔 불필요)
- `pip install matplotlib` (시각화 — 없으면 텍스트 모드로 자동 폴백)
- `pip install pynput` (시작/정지 키 — 없으면 즉시 시작)
- numpy

## ★ 각도 규약 — 헷갈리면 여기부터 ★
두 프레임만 있다. **말할 때는 원시각(raw)으로 말한다** (lidar_scope 화면 기준).

| 원시각 | 차체 방향 | 차체각(car) |
|---|---|---|
| raw 0° | 좌측 | 270° |
| raw 90° | 전진 | 180° |
| raw 180° | 우측 | 90° |
| raw 270° | 후진 | 0° |

`car = D·raw + M`, `raw = D·(car - M)`  (M=`MOUNT_OFFSET_DEG`, D=`LIDAR_ANGLE_DIR`).
**D가 필요한 이유**: M은 회전이라 좌우를 못 바꾼다. 라이다가 각도를 반대로 세면
좌우가 뒤집혀 보이는데 그건 거울 반전이라 M으로는 안 고쳐진다.

**섹터는 전부 차체 기준으로 정의하고, 표시할 때만 원시각으로 되돌린다.**
원시각 상수를 코드에 적어두면 라이다를 재장착할 때 같이 안 고쳐져서 썩는다
(실제로 `REAR_GATE_SECTORS`가 후방 대신 좌측을 가리는 버그가 있었다).
그래서 원시각으로 적는 상수는 `BLOCKED_SECTORS_RAW` 하나뿐이고, 이름에 `_RAW`를 박았다.

현재 장착의 방향·섹터 표는 언제든 확인할 수 있고, 미션 시작 시 로그에도 찍힌다:
```bash
python3 frames.py
```

## 상태머신
```
0 SEARCH      조향중립 직진        → 우측 섹터에 클러스터 (TRIGGER_HOLD_S 유지)
2 SWING_OUT   좌 풀조향 전진       → 자차 회전량 ≥ SWING_TARGET_DEG
3 REVERSE_IN  우 풀조향 후진       → 좌·우 섹터를 봤다가 둘 다 비면 정지
              (정렬되면 바퀴만 중립으로 — REVERSE_ALIGN_TOL)
              (후방 비상정지는 이 상태에서 매 프레임 최우선, FSM 독립)
5 WAIT        정지                 → WAIT_S (3~5초)
6 EXIT_FWD    조향중립 직진        → EXIT_FWD_S (2초)
8 EXIT_TURN   우 조향 전진         → EXIT_TURN_S
  CRUISE      조향중립 직진(계속)  → 없음. ★ESC로 직접 세운다★
```
완료(DONE) 상태는 없다. 마지막은 계속 직진하며 ESC를 기다린다.

### 왜 2단계 종료를 "회전량"으로 재는가 (중요)
직각 주차에서 이웃 차량은 **90° 어긋난 두 면**을 동시에 보여준다 — 차선과 평행한
앞면, 그리고 수직인 슬롯쪽 측면. 어느 면을 잡았는지는 알 수 없으므로 벽면
**절대 각도**로 "45도 됐나"를 판정하면, 측면을 잡은 프레임에서는 시작값이 이미
90°라 **첫 프레임에 조건이 참**이 된다(확인된 버그).
`RotationTracker`가 벽면각의 프레임간 **변화량**을 누적한다 — 어느 면을 보든
변화량은 자차 회전량과 같다. 면이 바뀌면 한 프레임에 ~90° 튀므로 걸러낸다.
엔코더/IMU가 없어 이게 유일한 회전량 관측이라, 실패하면 `SWING_TIMEOUT_S`가 일한다.

### 왜 3단계에 "봤다가 사라짐" 래치가 있는가
"양옆이 비면 정지"를 그대로 구현하면, 후진을 시작하는 시점엔 이웃 차량이 아직
`FLANK_NEAR_DIST` 밖이라 양옆이 **이미 비어 있어** 첫 프레임에 멈춘다.
그래서 한 번 본 적이 있어야 "사라짐"을 종료로 인정한다.

## 라이다 각도 점검 (`lidar_scope.py`)
"몇 도에 뭐가 있나"를 그대로 보는 도구. `MOUNT_OFFSET_DEG` / `BLOCKED_SECTORS_RAW` 실측용.
```bash
python3 lidar_scope.py                 # 라이브 (극좌표 + 각도→거리 2분할)
python3 lidar_scope.py --replay a.pkl  # 기록 재생, --record a.pkl 로 보며 기록
python3 lidar_scope.py --text          # matplotlib 없이 터미널 막대표
python3 lidar_scope.py --bin 5         # 각도 구간 폭(도, 360의 약수)
```
- 필터를 걸지 않은 **원시 스캔 전부**를 그리고, `MIN_VALID_DIST`/`MAX_DISTANCE`/
  `BLOCKED_SECTORS_RAW`와 **FSM이 쓰는 섹터 전부**를 음영으로 겹쳐 표시.
- 미션 화면(`visualize.py`)과 **같은 극좌표 장식 함수**를 쓴다 → 같은 물체가
  두 화면에서 같은 각도에 보인다.
- 종료 시 구간별 유효율 표 + `BLOCKED_SECTORS_RAW = [...]` 후보 줄 출력 →
  config.py 에 그대로 붙여넣기.

## 실차 전 캘리브레이션
**→ 재는 방법·순서는 `CALIBRATION.md` 참조.** 순서가 중요하다(앞 항목에 뒤가 의존).
가장 먼저 `MOUNT_OFFSET_DEG`, 그 다음 `PARK_STEER_SIGN`. 이 둘이 틀리면 나머지는 무의미.

## 검증 상태
무하드웨어로 검증 완료: 프레임 변환 왕복 및 실측 방향표 일치, 0°를 걸치는 섹터 래핑,
벽면각의 점순서 불변성, `--sim` 전 상태 전이(0→2→3→5→6→8→CRUISE), 일시정지/재개,
`--replay` 경로, 후방 비상정지의 차체마스크 우회, 두 화면의 각도 일치.

**실차에서 정해야 하는 것**: `SWING_TARGET_DEG`, `EXIT_TURN_S`, `FLANK_NEAR_DIST`,
`BLOCKED_SECTORS_RAW`, 그리고 3단계 "양옆 소멸"이 실제로 성립하는지
(슬롯 폭·차폭에 따라 벽이 `MIN_VALID_DIST` 안으로 들어와 사라지는 원리에 의존한다 —
차가 좁으면 영영 안 사라지고 `REVERSE_TIMEOUT_S`가 실제 종료 조건이 된다).
