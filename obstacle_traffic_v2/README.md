# obstacle_traffic_v2 — 장애물 회피 v2 + 신호등 미션

`traffic_mission_v02_0805` 를 복사해 **장애물 회피만 새로 짠** 버전입니다.
신호등 미션, 차선 추종, UI, 성능 최적화는 그대로입니다. 원본 폴더는 안 건드렸습니다.

## [장애물 회피 v2 — 무엇이 달라졌나]

| | v1 (0805) | **v2 (이 폴더)** |
|---|---|---|
| 상태 | KEEP → CHG_PRE → CHANGING → COUNTER_STEER → INNER_HOLD → RETURN_* → DONE (7개) | **normal ↔ changing (2개)** |
| 복귀 | 시간 지나면 원래 차선으로 되돌아옴 | **안 돌아옴.** 차선 설정(`lane_mode`)을 바꾸고 새 차선을 유지 |
| 방향 결정 | 설정 상수 `inner_lane_side` 고정 | **현재 차선에서 토글** (1↔2) |
| 커브 오인식 | 대책 없음 | **`avoid_max_steer`** — 직선일 때만 회피 |
| 재발동 방지 | 1회만 하고 `DONE` | **쿨타임** (`avoid_cooldown_seconds`) 후 다시 가능 |
| 반대조향 | 있음 | 없음 (차선 추종이 알아서 잡음) |

### 3중 검증 — 이걸 다 통과해야 회피한다

```text
① 연속 감지   초음파 < obs_trigger_cm 가 avoid_confirm_frames 회 연속
              (같은 센서 샘플을 두 번 세지 않는다. 새 값일 때만 카운트)
② 직선 주행   abs(steer) <= avoid_max_steer
              ★커브를 돌 때 정면 벽을 장애물로 착각하는 걸 막는 핵심 필터★
③ 비전 확인   best_ob_v2.pt 가 stroller 를 확정 (car_result_stale_s 이내)
=> front_blocked = True
```

셋 중 하나라도 빠지면 회피하지 않고, **왜 안 하는지**가 UI `REASON` 과 CSV
`block_reason` 에 그대로 찍힙니다 (`hits 2/5`, `curve |steer|=0.62>0.40`,
`car-not-confirmed`, `cooldown 1.3s`, `blocked-by-mission`).

### 상태 흐름

```text
normal ──(3중 검증 통과 + 쿨타임 아님)──> changing
   ^                                          │
   │                                          │ lane_change_seconds 동안
   │                                          │ steer = avoid_dir * lane_change_steer
   │                                          │ speed = lane_change_pwm  (제어 덮어쓰기)
   └──(lane_mode 를 목표 차선으로 교체 ────────┘
       + 차선 추종 캐시 초기화 + 쿨타임 시작)
```

방향은 `lane_mode` 에서 정합니다. 2차선이면 1차선으로(왼쪽), 1차선이면
2차선으로(오른쪽). **회피가 반대로 나가면 `lane1_side` 만 뒤집으면** 부호가
통째로 뒤집힙니다.

### 구동 명령 우선순위

```python
if not running or paused:            브레이크
elif STOPPED_RED or SAFE_STOP:       브레이크
elif avoid_state == "changing":      send_drive(slow_pwm, 강제조향)          # ★
elif 차선 로스트:                     브레이크
else:                                send_drive(drive_pwm, lane_steer)
```

**회피가 차선 로스트 정지보다 위**에 있어야 합니다. 차선을 가로지르는 동안
차선이 안 보이는 건 정상인데, 로스트 정지가 먼저 걸리면 회피 도중에 차선
한복판에 서버립니다.

### YOLO 는 메인 루프 밖에서 돈다

`shared/combined_detector.py` 가 별도 스레드에서 추론하고 메인 루프는
`latest()` 로 **폴링만** 합니다. 대기 중 가장 최신 프레임만 처리하므로 오래된
프레임이 쌓이지 않습니다. 컨트롤러는 그 결과를 인자로 받기만 하니
추론 때문에 제어 주기가 밀리지 않습니다.

### 회피가 늦게 느껴질 때 — 어디를 볼 것인가

회피 시작까지의 지연은 네 군데에서 나옵니다. 큰 것부터입니다.

| | 값 | 기여 |
|---|---|---|
| **트리거 거리** `obs_trigger_cm` | 140cm | **가장 큰 레버.** 거리 자체를 늘리면 그만큼 일찍 시작 |
| **비전 확인 주기** `combined_inference_interval_s` | 0.12s (8.3Hz) | 최대 166ms (주기 120 + 추론 46) |
| **연속 감지** `avoid_confirm_frames` | 3 | 75~150ms (1회당 25~50ms) |
| 비전 신선도 `car_result_stale_s` | 0.8s | 오래된 결과 무효화 |

`avoid_confirm_frames` 를 5→3 으로 줄이면 **50~100ms** 빨라집니다. 체감이
부족하면 `obs_trigger_cm` 을 올리는 게 훨씬 효과가 큽니다.

**진짜 병목은 로그로 확인하세요.** `[state]` 줄의 `why=` 가 부딪히기 직전에
무엇이었는지가 답입니다.

| `why=` | 의미 | 손볼 곳 |
|---|---|---|
| `hits 2/3` | 초음파 연속감지가 모자람 | `avoid_confirm_frames` |
| `ultrasonic-far` | 아직 트리거 거리 밖 | `obs_trigger_cm` |
| `car-not-confirmed` | **모델이 차를 확정 못 함** | `car_confidence`, `combined_inference_interval_s` |
| `curve \|steer\|=..` | 커브라 억제됨 | `avoid_max_steer` |
| `cooldown 1.3s` | 직전 회피 쿨타임 | `avoid_cooldown_seconds` |

`car-not-confirmed` 가 자주 보이면 초음파가 아니라 **비전이 늦은 것**입니다.
그때는 `avoid_confirm_frames` 를 아무리 줄여도 안 빨라집니다.

### ★ 초음파 250cm 함정 (2026-08-06 실주행에서 발견) ★

펌웨어는 에코를 못 받으면 **250 을 반환합니다** (`if (dur == 0) return 250;`).
이건 "250cm 에 뭔가 있다"가 아니라 **"못 읽었다"** 는 뜻입니다. 그런데
`sensor_max_valid_cm = 400` 이라 예전 코드는 이걸 정상 측정값으로 받았습니다.

실주행 로그에서 거리가 이렇게 찍혔습니다.

```text
dist=56  cmd=(220,+0.01)   <- 직선, 가까움. 회피 조건 성립
dist=250 cmd=(220,+0.02)   <- 카운트 리셋
dist=59  cmd=(220,-0.04)
dist=250                   <- 리셋
dist=56  cmd=(220,-0.08)
dist=250                   <- 리셋
```

`avoid_confirm_frames = 5` 는 **연속** 5회를 요구하는데 한 번 걸러 250 이 끼니
카운트가 `1→0→1→0` 으로 영영 5 에 못 갑니다. **회피가 발동할 수 없는
상태였고, 차가 장애물로 그대로 돌진했습니다.**

이때 조향은 +0.01 ~ +0.21 로 전부 `avoid_max_steer=0.4` 안쪽이었습니다.
**커브 필터는 아무것도 막지 않았습니다.**

#### 해결: 순간값이 아니라 시간창 최솟값

`us_window_s`(기본 0.3초) 동안의 **최솟값**을 전방 거리로 씁니다. 250 이상
(`us_timeout_cm`)은 측정 실패로 보고 창에 넣지 않습니다. 에코를 한두 번 놓쳐도
가까운 값이 창에 남아 있으면 계속 '가깝다'로 봅니다.

실제 로그 패턴으로 재현한 결과:

| | 결과 |
|---|---|
| 예전 (순간값) | 끝까지 발동 안 함 (`hits 1/5` 에서 멈춤) |
| 수정 (시간창 0.3s) | **5번째 샘플에서 회피 발동** |

#### 센서 2개는 원인이 아닙니다

전방 센서들의 **최솟값**을 쓰므로, 한 센서가 250 을 내도 다른 센서가 56 을
보면 56 이 채택됩니다. 로그에 250 이 찍혔다는 건 **그 순간 두 센서가 모두
250** 이었다는 뜻입니다 — 장애물을 보는 센서가 사실상 하나뿐이고 그게 한 번
걸러 놓치고 있다는 신호입니다. 센서를 줄이면 더 나빠집니다.

진단하려면 `[state]` 로그의 `us={...}` 를 보세요. 센서별 값이 찍힙니다.

#### 커브 필터(`avoid_max_steer`)를 지우지 않은 이유

같은 로그 뒷부분에서 조향 +0.58~+0.67(코너 한복판)일 때 거리가 52·71·82cm 로
찍힙니다. 그건 **코너 벽** 입니다. 필터를 빼면 이 지점에서 차선변경을 시도해
벽으로 꺾습니다.

그래도 끄고 싶으면 코드를 고칠 필요 없이 **`avoid_max_steer: 1.0`** 으로 두면
사실상 무제한이 됩니다 (조향 범위가 -1..1 이므로).

### 원본(`장애물회피_기공.txt`)과의 대조

이 회피 로직은 다른 팀 코드를 기반으로 했습니다. 상수는 **전부 원본과 동일**하게
맞췄고, 원본이 하드코딩한 값만 calib 으로 뺐습니다.

| 항목 | 원본 | 이 폴더 | |
|---|---|---|---|
| `OBSTACLE_CM` | 140 (하드코딩) | `obs_trigger_cm` 140 | 동일 |
| `AVOID_CONFIRM_FRAMES` | 5 | `avoid_confirm_frames` 5 | 동일 |
| `AVOID_MAX_STEER` | 0.4 | `avoid_max_steer` 0.4 | 동일 |
| `LANE_CHANGE_SECONDS` | 1.4 (하드코딩) | `lane_change_seconds` 1.4 | 동일 |
| `LANE_CHANGE_STEER` | 1.0 (하드코딩) | `lane_change_steer` 1.0 | 동일 |
| `AVOID_COOLDOWN_SECONDS` | 2.0 (하드코딩) | `avoid_cooldown_seconds` 2.0 | 동일 |
| 회피 중 속도 | `cfg["slow_pwm"]` | `slow_pwm` | 동일 |
| 일시정지 중 타이머 | 벽시계 계속 흐름 | 벽시계 계속 흐름 | 동일 |

**★ `slow_pwm` 함정 ★** 원본은 회피 속도로 `slow_pwm` 을 씁니다. 그런데 우리
calib 은 `slow_pwm = 220 = drive_pwm` 이라 **차선 변경 중 감속이 전혀 안 됩니다.**
원본의 "감속" 의도는 그쪽 설정에서 `slow_pwm < drive_pwm` 이었기 때문에 성립한
것입니다. 실행하면 시작할 때 이 조건을 검사해 경고를 냅니다.

```text
[avoid] ! slow_pwm(220) >= drive_pwm(220) — 차선 변경 중에 감속이 전혀 안 됩니다
```

1.4초 동안 최대 조향을 거는 기동이라 속도가 높으면 반대 차선까지 넘어가거나
거스를 수 있습니다. 실차에서 그런 증상이 보이면 `slow_pwm` 을 낮추세요.
다만 `slow_pwm` 은 커브 감속(`slow_steer_thresh` 0.55 초과)에도 쓰이므로
주행 전반이 같이 느려집니다.

### 연속감지 카운트는 방식이 다릅니다 (의도한 차이)

원본은 **메인 루프 프레임마다** 세고, 여기서는 **새 초음파 샘플일 때만** 셉니다.
펌웨어가 `US_PERIOD_MS=25` 로 센서를 번갈아 재므로(센서당 20Hz) 실제 지연은
거의 같습니다.

| | 5회 채우는 데 걸리는 시간 |
|---|---|
| 원본 (프레임 기준, 24fps) | 약 0.21초 |
| 여기 (샘플 기준) | 약 0.13~0.25초 |

원본 방식은 **fps 가 바뀌면 트리거 시점도 같이 바뀝니다**(60fps 면 0.08초).
그리고 텔레메트리가 끊겨 같은 값이 반복돼도 카운트가 올라가 노이즈 필터가
무력화됩니다. 그래서 이쪽만 원본과 다르게 뒀습니다.

### 튜닝 값

| 키 | 기본 | 설명 |
|---|---|---|
| `obs_trigger_cm` | 140 | 장애물로 볼 초음파 거리 |
| `avoid_confirm_frames` | 3 | 노이즈 필터. 최소 연속 감지 (1회당 25~50ms) |
| `avoid_max_steer` | 0.4 | **커브 필터.** 이 이하로 직진 중일 때만 회피 |
| `lane_change_seconds` | 1.4 | 핸들을 꺾고 유지하는 시간 |
| `lane_change_steer` | 1.0 | 그때의 강제 조향값 |
| `slow_pwm` | 220 | 그때의 속도. **`drive_pwm` 과 같으면 감속이 안 된다** |
| `avoid_cooldown_seconds` | 2.0 | 회피 직후 재발동 금지 |
| `start_lane_mode` | 2 | 출발 차선 |
| `lane1_side` | left | 1차선이 어느 쪽인가. **회피가 반대면 이걸 뒤집을 것** |
| `lane_mode_follow_vision` | 0 | 차선모델 INNER/OUTER 로 `lane_mode` 를 따라갈지. 기본 끔 |

## [모델]

통합 모델은 **`model/best_ob_v2.pt`** 입니다. 2026-08-06 학습본이 들어가 있습니다.
없으면 예전 `model/best_11n.pt` 로 돌아가되 시작할 때 경고를 냅니다.

| | 값 |
|---|---|
| 베이스 | yolo11s (9.4M) |
| 학습 | 150 epochs, imgsz 640, `fliplr=0.5` |
| 학습셋 | 대회장 1591 + 연습장 green2 547 + red2 395 = 2533장 |
| mAP50 / mAP50-95 | 0.9905 / 0.7957 |

### 추가 학습의 효과 (val 431장, imgsz 640 / conf 0.4)

| 출처 | 구모델 best_11n | **best_ob_v2** |
|---|---|---|
| green2 (연습장 초록불) | 0.018 | **1.000** |
| red2 (연습장 빨간불) | 0.000 | **1.000** |
| 대회장 red / green | 1.000 / 1.000 | 1.000 / 1.000 |
| 대회장 stroller | 0.985 | 0.985 |
| **전체 재현율** | 0.503 | **0.995** |
| 오검출 | 57 | **8** |

연습장에서 통째로 무너지던 게 해결됐고 **대회장 성능은 그대로**입니다.
red/green 뒤바뀜도 정상입니다 (일치 234 / 반대 0).

### imgsz 는 640 을 유지할 것

`combined_image_size` 를 320 으로 낮추면 빨라지지만 연습장 초록불을 놓칩니다.

| | imgsz 640 | imgsz 320 |
|---|---|---|
| green2 재현율 | **1.000** | 0.835 |
| 추론 시간 (M3 CPU) | 46.1ms | 15.8ms |

통합 검출기는 별도 스레드에서 `combined_inference_interval_s=0.12`(약 8Hz)로 돌아
46ms 를 감당합니다. 실측 루프 속도는 **24.3 fps** 로, 구모델(27.3) 대비 3fps
줄어드는 정도입니다.

참고로 구모델 계열에서는 11n 이 11s 와 성능이 같으면서 2배 빨랐습니다.
이번 모델은 11s 라 M3 CPU 에서 640 기준 46ms 로 두 배 무겁습니다. 속도가
문제되면 같은 데이터로 **11n 을 학습해 비교**해 보는 게 다음 카드입니다.

## [ROI 결론]

신호등 ROI는 없습니다. `best_11n.pt`가 통합 카메라 전체 프레임을 한 번 추론하며 `red/green`은 화면 위치와 관계없이 사용합니다.

자동차 ROI는 모델 입력을 자르는 영역이 아니라 `stroller` 결과에만 적용하는 후처리 영역입니다. confidence와 최소 면적을 통과한 자동차 박스의 **아래쪽 중앙점**이 ROI 안에 있어야 초음파 융합 후보가 됩니다. 따라서 신호등 검출 범위와 모델 연산량에는 영향을 주지 않으면서 화면 옆 자동차가 코너 벽 초음파와 잘못 결합되는 경우를 줄입니다.

현재 자동차 ROI 기본 범위는 전체 영상 비율 기준 `L=0.15, T=0.15, R=0.85, B=1.00`입니다. 통합 화면의 자홍색 사각형과 접점을 보고 실제 두 차선이 포함되도록 조절합니다.

## [폴더 구조]

```text
traffic_mission_v02_0805/
|-- integrated_traffic_obstacle.py   실행 진입점
|-- run_traffic_mission.py           메인 루프, UI, 상태 통합
|-- lane_model.py                    차선 분할과 BEV 차선 판단
|-- obstacle_controller.py           초음파 융합과 회피 상태기계
|-- run_windows.ps1                  Windows 실행 스크립트
|-- README.md                        전체 사용 설명
|-- VARIABLES.md                     변수별 상세 설명
|-- model/
|   |-- best_11n.pt                  신호등+자동차 통합 검출 모델
|   `-- lane_best.pt                 차선 검출 모델
|-- shared/
|   |-- combined_detector.py         비동기 전체 프레임 통합 추론
|   |-- calib.json                   실제 실행 설정
|   |-- config.py                    기본 설정과 저장
|   |-- control.py                   차선 PD와 회피 후 속도 복구
|   |-- serial_driver.py             Arduino 통신과 초음파 수신
|   |-- run_logger.py                CSV와 AVI 기록
|   `-- undistort.py                 차선 카메라 왜곡 보정
|-- camera_intrinsic/                카메라 보정값
`-- tests/                           미션 로직 단위 테스트
```

## [모델 파일]

실행 코드는 루트의 `traffic_car/best_11n.pt`를 직접 참조하지 않습니다. 복사본인 `traffic_mission_v02_0805/model/best_11n.pt`만 사용합니다.

복사본의 SHA-256은 다음과 같으며 루트 모델과 동일합니다.

```text
160FD440028041E4929147F6C8E87FB7BCAA7BE2AA5683B7D5CFD2210D9EA5C1
```

`model` 폴더에는 실행에 필요한 통합 모델과 차선 모델만 두었습니다. 구형 `best_car.pt`, `car_best.pt`, `best_traffic_light_v2.pt`, `lane_seg_best.pt`는 0805에서 사용하지 않습니다.

## [전체 주행 로직]

1. 차선 카메라 프레임을 왜곡 보정한 뒤 `lane_best.pt`로 `dash`, `crosswalk`, `solid`를 검출합니다.
2. 신호등 카메라 원본 전체 프레임을 비동기 통합 검출기에 제출합니다.
3. `best_11n.pt` 추론은 한 번만 수행하고 `red/green` 결과와 `stroller` 결과를 같은 프레임 번호와 촬영 시각으로 제공합니다.
4. 일반 상태에서는 차선 중심 오차, 곡선 보정, 차선 기울기 보정을 합쳐 조향합니다.
5. `crosswalk + red`가 각각 설정 프레임 수만큼 연속 확인되면 즉시 정지하고 장애물 회피 상태를 초기화합니다.
6. 정지 상태에서 `green`이 연속 확인되면 다시 출발합니다.
7. 출발이 확정되면 통합 검출기 전체를 끕니다. 따라서 이후에는 신호등과 자동차 검출 및 회피가 모두 비활성화됩니다.
8. 다시 신호등과 자동차 검출을 쓰려면 `RESET` 또는 `r`로 초기화한 뒤 `s`로 출발합니다.

## [통합 검출기]

`shared/combined_detector.py`는 별도 스레드에서 동작합니다. 메인 주행 루프는 YOLO 추론을 기다리지 않으며, 검출기는 대기 중 가장 최신 프레임만 처리합니다. 오래된 프레임을 순서대로 쌓지 않으므로 카메라 지연 누적을 줄입니다.

한 번의 추론 결과에서 다음 기준을 따로 적용합니다.

| 대상 | 클래스 | 판정 기준 |
|---|---|---|
| 정지 신호 | `red` | `traffic_confidence` 이상 |
| 출발 신호 | `green` | `traffic_confidence` 이상 |
| 자동차 장애물 | `stroller` | `car_confidence`와 `car_min_area_ratio`를 통과하고 박스 아래쪽 중앙점이 Car ROI 내부 |

`traffic_result_stale_s` 또는 `car_result_stale_s`보다 오래된 결과는 화면에 남아 있어도 제어에는 사용하지 않습니다. `combined_pipeline_timeout_s` 동안 새 결과가 없거나 모델 오류가 발생하면 필수 자동차 검출 파이프라인 이상으로 보고 안전 정지합니다.

## [신호등 미션]

정지 조건은 단순히 빨간불만 보는 것이 아닙니다.

```text
lane 모델의 crosswalk 연속 확인
AND
통합 모델의 red 연속 확인
=> STOPPED_RED
```

`lane_confirm_count=2`이면 서로 다른 두 번의 새 통합 검출 결과에서 횡단보도가 확인되어야 합니다. `red`는 현재 `stop_signal_confirm_count` 설정값만큼 연속 확인되어야 합니다. 오래된 결과는 연속 프레임 횟수를 올리지 않습니다.

정지와 동시에 장애물 회피 상태, 방향 잠금, 누적 융합 횟수를 초기화합니다. 정지 중에는 자동차와 초음파 조건이 맞아도 회피를 시작하지 않습니다.

정지 상태에서 `green`이 `go_signal_confirm_count`번 연속 확인되면 출발하고 신호 미션은 `COMPLETE`가 됩니다. 출발 직후 통합 검출기가 꺼지며, `video_stop_after_signal_resume_s`초 뒤 네 AVI 기록도 모두 종료됩니다. CSV 기록과 차선 주행은 프로그램 종료 전까지 계속됩니다.

## [장애물 융합]

위 **[장애물 회피 v2]** 의 3중 검증을 보세요. 요약하면 자동차만 보이거나
초음파만 가까워서는 회피하지 않고, 커브 중에도 회피하지 않습니다.

같은 초음파 측정값을 여러 루프에서 반복 사용해 연속 감지 횟수를 올리지 않습니다.
**새로운 센서 샘플일 때만** 카운트합니다.

## [초음파 사용]

`us_front_ids=[0,1]`의 센서만 회피 시작 거리 판단에 사용하고, 두 센서 중 유효한 최솟값을 전방 거리로 사용합니다. 값이 `sensor_min_valid_cm` 미만이거나 `sensor_max_valid_cm` 초과이면 무효입니다. 마지막 수신 후 `sensor_timeout_s`가 지나도 무효입니다.

센서 값 `250cm`은 코드가 강제로 만드는 값이 아닙니다. 실제 펌웨어가 250을 보내면 그대로 표시됩니다. 단, `--dry-run --mock-obstacle`에서는 테스트를 위해 250과 근거리 값을 의도적으로 만듭니다.

좌우 초음파가 ID 2 이상으로 연결되어 있으면 CSV의 `ultrasonic_json`에는 기록되지만 현재 회피 시작과 방향 결정에는 사용하지 않습니다. 실제 센서 배치와 ID가 확인되지 않은 상태에서 좌우 센서를 조향 조건으로 넣으면 벽이나 옆 차선을 오판할 위험이 있어, 0805는 전방 융합만 제어에 사용합니다.

## [차선 검출]

차선 검출용 카메라는 `cam_index`, 통합 검출용 카메라는 `traffic_cam_index`입니다. 두 인덱스가 같으면 한 카메라 프레임을 공유하고, 다르면 각각 엽니다. 기본 설정은 차선 카메라 `0`, 통합 카메라 `3`, Arduino `COM3`입니다.

차선 원본은 `640x360`으로 캡처하고 모델 입력은 `model_imgsz=320`을 사용합니다. 화면 표시 크기를 키워도 모델 입력 해상도나 카메라 종횡비는 바뀌지 않습니다.

solid 가로선 제거 필터는 연결 요소의 세로 길이, 수직축 기준 각도, 이전 정상 solid 기울기 차이, fitted 경로와의 거리를 검사합니다. 필터를 통과한 solid만 차선 경계 판단에 사용합니다. 원본 화면의 모델 마스크에는 필터 전 결과가 보일 수 있고 BEV와 로그에는 필터 통과/제외 픽셀이 표시됩니다.

## [UI 화면]

`Integrated Drive` 창은 위쪽 원본 차선 화면, 아래쪽 BEV, 오른쪽 상태 패널로 구성됩니다. 오른쪽 패널의 핵심 항목은 다음과 같습니다.

| UI 항목 | 의미 |
|---|---|
| `SIGNAL` | 현재 유효한 `red`, `green`, `unknown`, `disabled` |
| `MISSION` | `ACTIVE`, `HOLD_RED`, `COMPLETE` |
| `CAR` | 유효한 `stroller` 검출과 confidence |
| `FUSION` | 자동차+초음파 동시 조건, 누적 횟수, 시각 차이, 실패 이유 |
| `FRONT US` | 전방 센서별 거리와 실제 사용한 최솟값 |
| `PIPELINE` | 통합 모델과 초음파 수신 정상 여부 |
| `OBSTACLE` | 회피 상태와 잠긴 방향 |
| `FPS` | 메인 주행 루프 처리 FPS |
| `COMMAND` | 최종 PWM과 조향값 |

`Combined Detection` 창은 통합 카메라 전체를 보여줍니다. 빨간 박스는 `red STOP`, 초록 박스는 `green GO`, 주황 박스는 ROI 안에서 인정된 `stroller CAR`입니다. 자홍색 사각형은 자동차 전용 ROI이고 자홍색 점은 박스 아래쪽 중앙 접점입니다. 회색 `OUTSIDE CAR ROI` 박스는 모델이 자동차로 검출했지만 회피에서는 제외한 대상입니다. 신호등에는 자홍색 ROI가 적용되지 않습니다.

`Lane ROI / Drive Controls`는 스크롤 가능한 별도 창입니다. BEV 네 꼭짓점, 차선 폭, 중심 보정, 밴드 수, `Kp`, `Kd`, 우회전 gain, heading gain, 주행 PWM과 `Car ROI left/top/right/bottom`을 실시간 조절합니다. `Save config (W)` 또는 `w`를 눌러야 `shared/calib.json`에 저장됩니다.

## [키와 버튼]

| 입력 | 동작 |
|---|---|
| `s` | **전체 리셋 후 출발.** 언제 눌러도 "처음부터"입니다 |
| `p` | 일시정지 / 이어가기. 미션 상태를 그대로 두고 모터만 멈춥니다 |
| `x` 또는 `Space` | 정지. 브레이크를 걸고 주행을 멈춥니다 |
| `r` 또는 `RESET` | 브레이크 후 실행 직후 상태처럼 전체 런타임 초기화 (출발은 안 함) |
| `w` | 현재 UI 조절값을 `shared/calib.json`에 저장 |
| `CONTROLS` | 닫거나 숨긴 조절 창 다시 표시 |
| `q` | 브레이크, 로그 저장, 프로그램 종료 |

`s`는 신호등 미션(`traffic_gate`)과 회피 상태(`DONE` 포함)를 모두 초기화하고 통합
검출기를 다시 켠 뒤 출발합니다. 그래서 한 번 주행을 마치고 차를 출발선에 놓은 다음
`s`만 누르면 됩니다. 주행 중 잠깐 멈췄다 이어가려면 `s`가 아니라 `p`를 쓰세요.

`RESET`(`r`) 후에는 자동 출발하지 않습니다.

## [macOS 실행]

같은 코드가 맥에서도 돌아갑니다. 플랫폼 차이는 실행 시 자동으로 처리하므로
`calib.json` 을 고칠 필요가 없습니다.

| 항목 | Windows | macOS |
|---|---|---|
| 카메라 백엔드 | `CAP_DSHOW` | `CAP_AVFOUNDATION` (자동) |
| 시리얼 포트 | `COM3` | `/dev/cu.usbmodem*` 등을 자동 탐색 |

`camera_backend` 를 `auto` 로 두면 실행 플랫폼에 맞는 백엔드를 고릅니다. 설정에
다른 OS 전용 값(`CAP_DSHOW` 등)이 남아 있어도 조용히 갈아끼웁니다. 백엔드로 열기가
실패하면 `CAP_ANY` 로 한 번 더 시도합니다.

`serial_port` 가 `COM3` 처럼 맥에서 말이 안 되는 값이면 `/dev/cu.usbmodem*`,
`/dev/cu.usbserial*` 를 훑어 실제로 붙어 있는 장치를 씁니다.

### macOS 함정: Tk 조절창은 cv2 창보다 먼저 만들어야 한다

`ScrollableDriveControls`(Tk)를 `cv2.namedWindow` **뒤에** 만들면 macOS 에서
프로세스가 통째로 죽습니다.

```text
*** NSInvalidArgumentException: -[NSApplication macOSVersion]: unrecognized selector
```

둘 다 NSApplication 을 소유하려 들기 때문입니다. cv2 가 먼저 Cocoa 창을 띄우면
Tk 가 자기 것이 아닌 NSApplication 에 대고 자기 카테고리 메서드를 호출해서 터집니다.
**Objective-C 예외라서 `try/except tk.TclError` 로는 못 막습니다.**

실측 결과는 이렇습니다.

| 순서 | 결과 |
|---|---|
| Tk 단독 | OK |
| cv2 창 → Tk | **크래시** |
| Tk → cv2 창 | OK |

그래서 `main()` 에서 조절창을 `cv2.namedWindow` 앞으로 옮겨놨습니다. 이 순서를
되돌리지 마세요. 조절창 없이 돌리려면 `ui_controls_enabled=0` 으로 두면 됩니다.

```bash
./run_mac.sh --list-cameras     # 이 맥에서 열리는 카메라 인덱스 확인
./run_mac.sh --dry-run          # 시리얼 없이 영상만
./run_mac.sh                    # 실제 주행
```

**카메라 인덱스는 자동으로 못 맞춥니다.** 맥과 Windows 는 같은 카메라라도 인덱스가
다르게 잡힙니다. 기본 설정은 차선 `0` / 통합 `3` 인데 맥에서는 보통 `3` 이 없습니다.
`--list-cameras` 로 확인한 뒤 `--cam` / `--traffic-cam` 으로 넘기거나 `calib.json` 의
`cam_index` / `traffic_cam_index` 를 고치세요.

## [Windows 실행]

VS Code에서 `C:\Users\dmsal\Desktop\traffic_car` 폴더를 열고 PowerShell 터미널에서 실행합니다.

```powershell
.\traffic_mission_v02_0805\run_windows.ps1
```

모터 명령 없이 카메라와 모델을 확인하려면 다음을 사용합니다.

```powershell
.\traffic_mission_v02_0805\run_windows.ps1 -DryRun
```

모의 자동차+초음파 회피까지 확인하려면 다음을 사용합니다.

```powershell
.\traffic_mission_v02_0805\run_windows.ps1 -DryRun -MockObstacle
```

카메라와 포트를 일시적으로 덮어쓸 수 있습니다.

```powershell
.\traffic_mission_v02_0805\run_windows.ps1 -Cam 0 -TrafficCam 3 -SerialPort COM3
```

가상환경이 없으면 저장소 루트에서 생성하고 의존성을 설치합니다.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\traffic_mission_v02_0805\requirements.txt
```

## [성능 — 맥북에서 실시간으로 돌리기]

M3 맥북에서 처음 돌렸을 때 **약 7fps** 였습니다. 프로파일로 원인을 찾아 세 군데를
고쳐 **27fps** 가 됐습니다. 짐작이 아니라 실측으로 정한 값들입니다.

### 원래 한 바퀴 140ms 의 내역

| 항목 | 반복당 |
|---|---|
| 카메라 `read()` ×2 | **43ms** |
| 차선 모델 (추론 15 + 후처리 11) | 26ms |
| `cv2.waitKey` | 16ms |
| Tk 조절창 poll | 7.5ms |

### 고친 것

**1. 카메라를 스레드로 (`shared/frame_grabber.py`)** — 가장 큰 이득입니다.
`read()` 는 다음 프레임이 올 때까지 블로킹해서, 30fps 카메라 두 대면 그 자체로
루프 상한이 정해집니다. 스레드가 계속 비우고 최신 프레임만 남기게 하면
**36.9ms → 0.0ms** 입니다. 덤으로 드라이버 버퍼에 프레임이 쌓이지 않아
영상 지연도 누적되지 않습니다.

**2. UI 를 제어 루프에서 분리 (`ui_render_every_n`, 기본 2)** — macOS 에서
`imshow`+`waitKey` 는 창 하나만 있어도 16.6ms, `waitKey` 만 해도 11.0ms 입니다.
이건 Cocoa 이벤트 펌프 고정비용이라 줄일 수 없고, **덜 자주 부르는 수밖에** 없습니다.
2 로 두면 화면은 절반 주기로 갱신되고 제어 루프는 그만큼 빨라집니다.

**3. 통합 검출기 주기 (`combined_inference_interval_s` 0.04 → 0.12)** — 25Hz 는
신호등·장애물에 과합니다. best_11n 이 59ms 라 따라잡지도 못하면서 검출 스레드가
쉬지 않고 돌았습니다. `traffic_result_stale_s=0.5`, `lane_confirm_count=2` 라
0.12 여도 신호 판정은 0.3초 안에 끝납니다.

### 더 줄이려면

지금은 차선 모델이 26ms 로 예산의 70% 입니다. 확인해본 것들:

- **torch 스레드 수는 거의 무관** (2/4/6/8 모두 차선 추론 10~12ms, 통합검출 동시에는 15~16ms)
- **디버그 오버레이 그리기는 전부 합쳐 0.6ms** 라 걷어내도 소용없습니다
- 남은 후처리 11ms 는 밴드 루프 같은 파이썬 연산이라, 줄이려면 `lane_model.py`
  알고리즘을 다시 짜야 합니다. 제어 로직 한복판이라 위험 대비 이득이 적습니다

화면이 느리게 느껴지면 `ui_render_every_n` 을 3~4 로 올리세요. 제어는 그대로
빨라지고 화면만 느려집니다. 반대로 튜닝 중이라 화면이 부드러워야 하면 1 로 내리되
제어 주기가 느려지는 걸 감수해야 합니다.

## [영상 녹화 — record_camera.py]

원하는 카메라 인덱스로 영상을 찍는 단독 도구입니다. 주행 코드와 분리돼 있고
ultralytics/torch 를 안 불러오므로 바로 뜹니다. 카메라 백엔드/인덱스 처리는
주행 코드와 같은 `shared/frame_grabber.py` 를 씁니다.

```bash
python record_camera.py --list                 # 이 컴퓨터에서 열리는 인덱스 확인
python record_camera.py --cam 1                # 1번 카메라로 녹화 (Space 로 시작)
python record_camera.py --cam 1 --seconds 30 --auto
python record_camera.py --cam 1 --size 1280x720 --fps 30
python record_camera.py --cam 1 --out ~/Desktop/test.avi
python record_camera.py --cam 1 --no-preview   # 화면 없이
```

| 키 | 동작 |
|---|---|
| `Space` | 녹화 시작 / 일시정지 (`--auto` 면 시작하자마자 녹화) |
| `s` | 스냅샷 한 장 저장 (녹화와 별개로 `.jpg`) |
| `q` / `ESC` | 저장하고 종료 |

저장되는 영상은 **오버레이가 없는 원본 프레임**입니다. 화면의 빨간 테두리와 글자는
미리보기에만 그리므로 학습 데이터로 바로 쓸 수 있습니다. 기본 저장 경로는
`records/cam<인덱스>_<날짜시각>.avi` 입니다.

녹화도 `FrameGrabber` 를 쓰므로 드라이버 버퍼에 프레임이 쌓이지 않습니다. 같은
프레임을 두 번 쓰지 않고 저장 fps 간격을 지키므로 재생 시간이 실제 시간과 맞습니다.

## [실행 전 점검]

1. `model/best_11n.pt`와 `model/lane_best.pt`가 있는지 확인합니다.
2. Windows 장치 관리자와 카메라 앱을 닫고 카메라 인덱스를 확인합니다.
3. Arduino 포트가 `shared/calib.json`의 `serial_port`와 같은지 확인합니다.
4. 바퀴를 공중에 띄운 상태에서 먼저 `-DryRun`으로 박스, 차선, FPS를 확인합니다.
5. 통합 창에서 실제 빨간불이 `red STOP`, 초록불이 `green GO`로 나오는지 확인합니다.
6. 앞차 검출이 `stroller CAR`로 나오고 confidence가 0.60 이상인지 확인합니다.
7. `FRONT US`의 U0/U1이 실제 거리 변화에 반응하는지 확인합니다.
8. 실제 주행 전 낮은 `drive_pwm`과 짧은 회피 시간으로 방향부터 확인합니다.

## [로그 위치]

`logging_enabled=1`일 때 실행마다 다음 폴더가 생성됩니다. 현재 설정이 `0`이면 CSV와 AVI가 모두 생성되지 않습니다.

```text
traffic_mission_v02_0805/logs/YYYYMMDD_HHMMSS_mmm/
|-- session.json
|-- telemetry.csv
|-- drive_ui.avi
|-- combined_detection.avi
|-- traffic_clean.avi
`-- lane_clean.avi
```

`drive_ui.avi`는 차선 화면, BEV, 오른쪽 상태 패널을 함께 기록합니다. `combined_detection.avi`는 통합 박스와 상태 글자가 있는 전체 신호등 카메라 화면입니다. `traffic_clean.avi`는 어떤 ROI 선이나 검출 박스도 없는 통합 카메라 원본입니다. `lane_clean.avi`는 왜곡 보정 후 프레임이 아니라 차선 카메라에서 읽은 원본 프레임을 기록합니다.

`telemetry.csv`는 Excel에서 열 수 있도록 UTF-8 BOM으로 저장합니다. 주요 열은 `final_speed`, `final_steer`, `traffic_signal`, `car_confidence`, `car_roi_left/top/right/bottom`, `car_roi_candidate_count`, `car_roi_rejected_count`, `front_distance_cm`, `fusion_reason`, `obstacle_phase`, `combined_inference_ms`, `fps`, `event`입니다.

신호등 정지 후 `green`으로 출발하면 `video_stop_after_signal_resume_s=3.0`초 후 네 AVI가 모두 닫힙니다. 이후 CSV는 계속 기록됩니다. 영상 파일이 재생되지 않으면 프로그램이 정상 종료되어 VideoWriter가 닫혔는지 먼저 확인합니다.

## [문제 찾기]

| 증상 | 먼저 확인할 항목 |
|---|---|
| 빨간불인데 늦게 정지 | `combined_inference_ms`, `traffic_result_age_s`, `lane_confirm_count`, `stop_signal_confirm_count`, crosswalk 검출 |
| 초록불인데 늦게 출발 | `go_signal_confirm_count`, `traffic_result_age_s`, 모델 confidence, 실제 클래스가 `green`인지 |
| 차를 늦게 발견 | `combined_image_size`, `combined_inference_interval_s`, `car_confidence`, `car_min_area_ratio`, 통합 카메라 해상도/FPS |
| 화면에는 차가 보이지만 회피 후보가 아님 | 회색 `OUTSIDE CAR ROI`인지, 자홍색 접점과 `car_roi_*` 확인 |
| 코너 벽과 화면 옆 자동차가 결합됨 | Car ROI 좌우 범위를 좁히고 `car_roi_rejected_count`, `fusion_reason` 확인 |
| 차만 보고 회피하지 않음 | 정상 동작입니다. `front_distance_cm`, `fusion_reason`, `fusion_skew_s` 확인 |
| 초음파만 가까운데 회피하지 않음 | 정상 동작입니다. `car_detected`, `car_confidence` 확인 |
| 갑자기 SAFE_STOP | `sensor_ok`, `combined_model_fault`, `PIPELINE`, 카메라/시리얼 연결 확인 |
| 250cm가 고정 | `--MockObstacle` 여부, Arduino 원시 출력, 센서 배선/각도/echo 확인 |
| 가로 흰 선을 solid로 사용 | `solid_filter_angle_deg`, kept/rejected pixels와 `VARIABLES.md`의 solid 필터 변수 확인 |
| UI FPS가 낮음 | CPU 사용률, `combined_image_size`, `model_imgsz`, 카메라 해상도, AVI 기록 부하 확인 |

## [변수 수정]

실제 실행값은 `shared/calib.json`에 있습니다. 파일을 수정할 때는 프로그램을 종료한 상태에서 수정하는 것이 안전합니다. UI에서 바꿀 수 있는 값은 실행 중 적용되지만 `w`로 저장해야 다음 실행에도 유지됩니다.

모든 변수의 단위, 증가/감소 효과, 위험 요소는 [VARIABLES.md](VARIABLES.md)에 정리되어 있습니다. Ctrl+F로 변수 이름을 그대로 검색하면 됩니다.

## [테스트]

```powershell
Set-Location .\traffic_mission_v02_0805
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트는 `red/green` 신호 의미, 연속 프레임 확정, 자동차+초음파 동시 조건, 방향별 차선 변경 시간, 회피 후 속도 복구, 통합 모델의 한 프레임 공유, 자동차 ROI 밖에서도 신호등이 검출되는지를 확인합니다.
