# traffic_mission_v02_0804

최상위 `drive`의 최신 차선 주행과 `traffic_mission_v02_0802`의 신호등, 자동차+초음파 장애물 회피, 차선변경, 녹화 기능을 결합한 Windows용 새 버전입니다. 두 원본 폴더는 수정하지 않습니다.

## 결합 기준

- `drive`에서 가져온 항목: 차선 주행 구조, 가상 좌우 경계 피팅, 단일 경계 기울기 보정, confidence 가중 중심선, 곡률·헤딩 제어, 현재 IPM/PD/PWM 값
- `0802`에서 유지한 항목: 두 카메라 구조, 신호등 정지·재출발, 자동차 ROI, 자동차+전방 초음파 동시 조건, dash 방향 차선변경, 회피 후 3초 재가속, 로그·영상 녹화
- 신호 재출발 완료 후: 신호등 YOLO와 자동차 YOLO를 함께 끄고 일반 차선 주행만 유지
- 하드웨어 설정: 주행 카메라 `3`, 신호 카메라 `1`, Arduino `COM3`

## 전체 실행

저장소 최상위 폴더 `C:\Users\dmsal\Desktop\traffic_car`에서 실행합니다.

```powershell
.\traffic_mission_v02_0804\run_windows.ps1
```

하드웨어 명령 없이 카메라와 알고리즘만 확인하려면 다음 명령을 사용합니다.

```powershell
.\traffic_mission_v02_0804\run_windows.ps1 -DryRun -MockObstacle
```

직접 실행하려면 `.\.venv\Scripts\python.exe .\traffic_mission_v02_0804\integrated_traffic_obstacle.py --dry-run`을 사용합니다.

실제 메인 루프는 `run_traffic_mission.py`에 있습니다.

## 현재 수행 알고리즘

### 차선 주행

1. `traffic_mission_v02_0804/model/lane_best.pt`로 전체 주행 카메라 프레임에서 `dash`, `crosswalk`, `solid`를 분할합니다. 클래스 인덱스 1인 `crosswalk`가 신호 정지 구역 판정에 사용됩니다.
2. 카메라 원본은 `640×360`으로 유지하고, 제어 주기를 높이기 위해 차선 모델 추론 입력은 `320`을 사용합니다.
3. 현재 `drive` IPM 사다리꼴 안의 마스크만 남긴 뒤 버드아이뷰로 변환합니다.
4. 여러 높이 구간에서 좌우 경계를 찾고, 한 구간이 아닌 전체 구간 투표로 각 경계를 `dash` 또는 `solid`로 분류합니다.
5. 차량은 `dash`와 `solid` 경계의 중앙을 목표로 주행합니다. 실제 제어에는 `inner/outer` 구분을 사용하지 않습니다.
6. 끊긴 밴드는 다른 높이의 같은 경계점에 맞춘 직선으로 가상 경계를 복원합니다. 피팅도 불가능한 단일 경계에서는 `bev_lane_width/2 × sqrt(1+slope²)`로 중앙을 추정합니다.
7. 일반 코너에서는 먼 구간의 차선 중심에 `lookahead_gain` 가중치를 주어 코너를 미리 따라갑니다.
8. 가까운 구간의 중앙 오차가 감소하면 코너 탈출로 판단하고 현재 `drive` 값 `lookahead_exit_scale=1.5`를 적용합니다.
9. 실측·가상 목표점 전체를 confidence 가중 직선 피팅해 헤딩을 계산하고 `heading_gain=0.6`으로 조향에 더합니다.
10. 장애물 회피 완료 또는 신호 정지 후 재출발 시 3개의 정상 검출 프레임 동안 곡률과 기울기 보정을 끄고 차선 중앙 위치만으로 조향합니다.

최종 차선 오차는 `중앙 위치 오차(P) + 곡률 보정(C) + 차선 기울기 보정(H)`입니다. 가상 경계는 실측 양선보다 낮은 confidence를 사용하므로 끊긴 차선을 이어가면서도 실제 검출을 우선합니다.

원본 화면의 마스크 색상은 다음과 같습니다.

- 주황색: `dash`
- 초록색: `lane`
- 파란색 계열: `solid`
- 자홍색 사다리꼴: 실제 BEV 변환 기준
- 청록색 사다리꼴: 마스크 처리 영역. 현재 여백이 0이라 자홍색 IPM과 겹침

화면 아래의 `raw D/L/S`는 ROI로 자르기 전 전체 프레임의 픽셀 수이고, `roi S`는 확장 처리 영역 안에 남은 solid 픽셀 수입니다.

### solid 가로선 제외

분할 모델의 원본 `solid` 출력은 그대로 보존하지만, 경계 판단에는 BEV 방향 필터를 통과한 solid만 사용합니다. 연결 영역마다 행별 중앙점으로 solid 경로를 맞추고 다음 항목을 확인합니다.

1. 세로로 `solid_min_vertical_span_px` 이상 이어지는지 확인합니다.
2. 세로축 기준 각도가 `solid_max_angle_from_vertical_deg` 이하인지 확인합니다. 완전한 가로선은 90도입니다.
3. 이전 정상 solid의 EMA 기울기와 차이가 `solid_prev_angle_tolerance_deg` 이하인지 확인합니다.
4. fitted 경로에서 `solid_fit_corridor_px`보다 멀리 뻗은 가로 가지를 제거합니다.

필터를 통과한 solid만 차선 위치 탐색과 solid/dash 분류에 들어갑니다. `dash`와 신호 정지 구역용 `lane` 마스크에는 이 필터를 적용하지 않습니다. 원본 화면의 파란색은 모델이 출력한 전체 solid이고, BEV의 파란색은 필터를 통과한 solid, BEV의 빨간색은 모델이 solid로 검출했지만 경계 판단에서 제외된 픽셀입니다.

BEV의 `solid filter keep=... reject=... angle=...`과 상태 패널의 `Solid filter` 줄에서 동작을 확인할 수 있습니다. CSV에는 `solid_filter_angle_deg`, `solid_filter_kept_pixels`, `solid_filter_rejected_pixels`가 기록됩니다. 이전 기울기와 맞는 후보가 없으면 `solid_angle_reacquire_frames` 후 기준을 초기화해 새로운 곡선 기울기를 다시 잡습니다.

오른쪽 상태 패널의 `Front US`에는 `U0`, `U1`의 최신 유효 거리가 각각 표시됩니다. `filtered`는 실제 장애물 판단에 사용한 두 센서의 최솟값입니다. 회피 중에는 초음파를 사용하지 않으므로 모두 `IGNORED`로 표시됩니다.

### 장애물 회피

장애물 회피는 다음 순서로 진행됩니다.

1. `traffic_mission_v02_0804/model/car_best.pt`가 신호등 카메라의 `car_roi_*` 영역에서 장애물 자동차를 검출하고, 같은 시점의 새 전방 초음파 측정값이 `obs_trigger_cm`보다 가까우면 첫 융합 프레임부터 속도를 `change_pwm` 이하로 제한합니다. 이 모델의 클래스 이름은 `stroller`이지만 코드에서는 장애물 자동차 의미로 사용합니다. 이 상태를 `obs_trigger_hits`회 확인하면 회피에 진입합니다.
2. 차선 모델이 판단한 dash 방향을 `direction_confirm_frames` 프레임 동안 확인합니다.
3. dash 방향이 불명확하면 `WAIT_DASH` 상태에서 모터를 정지하지 않고 기존 차선 중앙 주행을 계속하며 방향을 확인합니다.
4. 방향이 확인되면 해당 방향을 잠급니다. 회피 중 solid 오검출이 발생해도 방향은 바뀌지 않습니다.
5. 회피 시작 직전 거리가 `obstacle_slow_cm` 이하인지 한 번 확인하고 회피 속도를 잠급니다.
6. `change_steer` 조향으로 dash 방향을 향해 이동합니다. INNER→OUTER는 `change_duration_inner_to_outer_s`, OUTER→INNER는 `change_duration_outer_to_inner_s`만큼 유지합니다.
7. `counter_steer_duration_s` 동안 반대 방향으로 조향합니다.
8. 회피가 끝나면 기존 추적값과 PD 미분값을 초기화하고, 속도를 `70`에서 시작해 실제 주행 시간 3초 동안 `255`까지 선형으로 올립니다.
9. 회복 가속 중에도 즉시 장애물 감지를 다시 시작합니다. 새 자동차+근거리 초음파 융합 측정 1회와 dash 방향이 확인되면 가속 램프보다 새 회피를 우선합니다.

회피 시작 조건은 다음 진리표를 따릅니다.

| `car_best.pt` 자동차 | 전방 초음파 `< obs_trigger_cm` | 동작 |
|---|---|---|
| X | X | 일반 차선 주행 |
| O | X | 회피하지 않음 |
| X | O | 근접 감속은 가능하지만 회피하지 않음 |
| O | O | 시각 동기와 연속 횟수 확인 후 회피 |

자동차 결과는 별도 스레드가 최신 프레임만 처리합니다. 자동차와 신호등은 모두 `traffic_cam_index` 카메라의 같은 원본 프레임을 공유하며, 자동차 모델은 `car_roi_*`, 신호등 모델은 `traffic_roi_*` 범위를 각각 잘라 사용합니다. 자동차 confidence가 `0.60` 이상이어야 회피 후보로 인정합니다. 자동차 원본 프레임 시각과 새 초음파 패킷 수신 시각 차이가 `fusion_max_skew_s` 이하여야 한 쌍으로 인정하며, 같은 초음파 패킷은 한 번만 셉니다. 자동차 결과가 `car_result_stale_s`보다 오래되면 검출로 인정하지 않습니다. 모델 스레드가 `car_pipeline_timeout_s` 동안 새 결과를 만들지 못하거나 오류가 발생하면 시작 유예 이후 `SAFE_STOP`으로 정지합니다.

회피 조향이 시작된 뒤에는 일반 차선 중앙 PD 조향, 차선 소실 안전정지, 초음파 거리값을 모두 무시합니다. UI의 거리는 `IGNORED`로 표시됩니다. 초음파값이 흔들려도 `CHANGING -> COUNTER_STEER` 동작은 잠긴 방향과 속도로 수행합니다. 단, `lane + 모델 green 클래스` 정지가 확정되면 안전을 위해 진행 중 회피를 즉시 취소하고 브레이크하며, 모델 `red` 클래스 재출발 전까지 새 회피도 막습니다. 사용자가 `x` 또는 `Space`로 정지하면 회피 시간을 일시정지하고, `RESET` 또는 `r`은 회피 상태 자체를 초기화합니다.

현재 주요 회피 설정은 다음과 같습니다.

- 감지 거리: `obs_trigger_cm = 150`
- 새 측정 확인 횟수: `obs_trigger_hits = 1`
- 근접 감속 거리: `obstacle_slow_cm = 30`
- 근접 감속 속도: `obstacle_slow_pwm = 10`
- INNER→OUTER dash 방향 조향 시간: `change_duration_inner_to_outer_s = 2.4`
- OUTER→INNER dash 방향 조향 시간: `change_duration_outer_to_inner_s = 2.4`
- 반대 조향 시간: `counter_steer_duration_s = 1.2`
- 회피 조향: `change_steer = 1.0`
- 회피 속도: `change_pwm = 70`
- 회피 후 재가속: `70 → 255`, 실제 주행 시간 `3.0초`

### 초음파 안전 처리

각 초음파 센서의 마지막 실제 패킷 수신 시각을 별도로 기록합니다. 과거 거리값이 딕셔너리에 남아 있어도 `sensor_timeout_s`보다 오래된 값은 사용하지 않습니다. `sensor_min_valid_cm`보다 작거나 `sensor_max_valid_cm`보다 큰 값도 무효 처리합니다.

일반 주행 또는 `WAIT_DASH`에서 유효한 최신 측정값이 없으면 `SAFE_STOP`으로 정지합니다. 이미 회피 조향이 시작된 뒤에는 초음파 거리와 센서 상태를 사용하지 않고 잠긴 방향, 속도, 시간으로 회피를 끝까지 수행합니다. 회피 완료 다음 프레임부터 센서 유효성을 다시 검사합니다.

### 신호등

신호등 모델은 신호등 카메라 전체 프레임이 아니라 `traffic_roi_left/top/right/bottom`으로 지정한 사각형의 픽셀만 사용합니다. 실행 중 `Traffic ROI` 창에는 전체 신호등 영상, 현재 ROI 테두리, ROI 안에서 검출된 객체가 함께 표시됩니다. 이 체크포인트는 실제 신호 색상과 클래스 이름이 반대로 학습되어 있으므로 **모델 출력 `green`은 정지**, **모델 출력 `red`는 출발**로 사용합니다. `lane` 구역과 모델 `green` 클래스가 각각 설정된 확인 프레임 수만큼 함께 유지되면 한 번 정지하고, 정지 상태에서 모델 `red` 클래스가 확인되면 다시 출발합니다. `yellow`는 현재 모델 클래스에 없으며 상태 전이에 사용하지 않습니다.

신호등 YOLO는 제어 루프를 멈추지 않는 비동기 스레드에서 최신 프레임만 처리합니다. 입력 크기는 `416`, 최소 추론 간격은 `0.08초`이며 같은 추론 결과를 여러 제어 프레임에서 표시하더라도 확인 횟수는 한 번만 증가합니다. 결과 나이가 `0.5초`를 넘으면 `unknown`으로 무효화하고 부분 확인 이력을 지웁니다. `Traffic ROI` 상단의 `age=... async`가 결과 나이이며, 카메라는 `MJPG`, `30 FPS`, 버퍼 1을 요청해 오래된 프레임 적체를 줄입니다.

코드 기본값은 전체 영상 너비의 20~80%, 높이의 5~55%이고, 현재 `calib.json` 저장값은 너비의 6~95%, 높이의 11~41%입니다. 카메라가 다른 해상도로 열려도 비율을 실제 픽셀 좌표로 다시 계산하므로 같은 상대 위치를 유지합니다.

신호 정지가 확정되는 프레임에 장애물 회피 상태기와 PD를 즉시 초기화합니다. 정지 중에는 상태 패널의 `Traffic mission`이 `HOLD_RED`, `Avoidance blocked by signal`이 `True`가 되며 자동차+초음파 조건이 맞아도 회피 누적이나 차선변경을 시작하지 않습니다. 모델 `red` 클래스 재출발 시 PD의 이전 오차, 기울기 필터, 룩어헤드 판정 상태를 초기화하고 3개의 정상 검출 프레임 동안 중앙 위치만 사용합니다.

모델 `red` 클래스 재출발이 한 번 완료되면 신호 미션 상태는 `COMPLETE`가 되고 신호등 YOLO와 자동차 후면 YOLO를 동시에 끕니다. 자동차 검출기의 대기 프레임과 최신 결과도 즉시 지우므로 완료 직전 결과가 초음파와 융합되어 회피를 다시 시작하지 않습니다. `Traffic ROI` 창에는 `TRAFFIC MISSION COMPLETE - DETECTOR OFF`와 `CAR DETECTOR OFF WITH TRAFFIC MISSION`이 표시되며, 상태 패널은 `Car[OFF]`로 바뀝니다. 재출발 확정 시각부터 `video_stop_after_signal_resume_s=3.0`초 동안 네 AVI를 더 기록한 뒤 모든 VideoWriter를 닫습니다. 이후 카메라 화면, 일반 차선 주행과 CSV 텔레메트리는 계속되지만 AVI에는 새 프레임이 추가되지 않습니다. 신호·자동차 검출을 다시 사용하려면 `RESET` 버튼 또는 `r` 키로 전체 런타임 상태를 초기화한 뒤 `s`를 누릅니다.

## ROI UI

메인 `Integrated Drive` 창과 분리된 `Lane ROI / Drive Controls` 창에서 BEV 사다리꼴, 중앙값, 조향 게인을 실시간 조절합니다. 조절 창은 세로 스크롤바와 마우스 휠을 지원하므로 메인 주행 영상을 가리지 않습니다. 창의 X 버튼은 조절 창만 숨기며 메인 화면 오른쪽 위 `CONTROLS` 버튼을 누르면 다시 표시됩니다. 별도의 `Traffic ROI` 창에서는 신호등 ROI와 자동차 ROI를 함께 실시간 조절합니다.

- `TL`, `TR`, `BL`, `BR`: BEV 사다리꼴 네 꼭짓점
- `Center`: 목표 중앙 보정
- `LaneW`: 한쪽 경계만 보일 때 사용할 차선 폭
- `Bands`: 차선을 읽는 세로 구간 수
- `Kp x10`, `Kd x10`: PD 조향 게인
- `RGain x10`, `HGain x10`: 우회전·헤딩 보정 게인
- `Drive PWM`: 일반 주행과 회피 후 회복 목표 속도
- `Left %`, `Top %`: 신호등 ROI의 왼쪽·위쪽 시작 위치
- `Right %`, `Bottom %`: 신호등 ROI의 오른쪽·아래쪽 끝 위치
- `Car Left %`, `Car Top %`: 자동차 ROI의 왼쪽·위쪽 시작 위치
- `Car Right %`, `Car Bottom %`: 자동차 ROI의 오른쪽·아래쪽 끝 위치
- `Save config (W)` 버튼 또는 `w`: 현재 값을 `shared/calib.json`에 저장
- `s`: 주행 시작
- `x` 또는 `Space`: 정지
- `CONTROLS`: 숨긴 차선/주행 조절 창 다시 표시
- `RESET` 버튼 또는 `r`: 브레이크 후 시작 직후 상태로 전체 런타임 초기화
- `q`: 종료

`x`/`Space`는 모터만 멈추며 현재 신호·회피 상태를 보존합니다. `RESET`/`r`은 주행을 끄고 브레이크한 뒤 신호 미션을 `ACTIVE`로 재활성화하고, 신호 누적 프레임, 회피 상태기와 방향 잠금, 차선 추적과 PD, 초음파 캐시, 자동차 결과 유효 시작 시각, FPS 계산을 모두 초기화합니다. 모델과 카메라를 다시 로드하지 않고 로그 세션도 유지하며, 다음 CSV 프레임의 `event`에 `full-reset`을 기록합니다. 초기화 뒤 실제 출발은 반드시 `s`를 눌러야 합니다.

신호등과 자동차 ROI 트랙바 값은 같은 카메라 전체 프레임에 대한 백분율입니다. 두 ROI 모두 오른쪽은 왼쪽보다, 아래쪽은 위쪽보다 최소 1% 크게 자동 보정되므로 경계가 뒤집히거나 빈 이미지가 YOLO에 전달되지 않습니다. 조절값은 즉시 다음 최신 프레임 추론에 반영되며 `w`를 누르기 전에는 파일에 저장되지 않습니다. 자동차 ROI를 너무 좁히면 차량이 경계 밖에 있는 동안 절대 검출되지 않으므로 실제 주행 경로 전체가 자홍색 사각형 안에 들어오도록 설정합니다.

메인 창의 원본 주행 화면과 BEV 화면은 각각 `896x504`의 16:9 표시 비율이며, 카메라 캡처 해상도 `640x360`과 모델 입력 크기는 바꾸지 않습니다. 오른쪽 상태 패널은 확대 렌더링 후 축소하지 않고 최종 높이에 직접 그리며 `MISSION`, `OBSTACLE FUSION`, `LANE CONTROL`, `LIVE TUNING` 카드로 구분합니다. 따라서 영상과 로그 배치만 달라지고 차선 판단, 조향, 신호등 또는 회피 로직은 바뀌지 않습니다.

오른쪽 상태 패널의 `ERROR`와 `LOOKAHEAD` 줄은 현재 제어값을 보여줍니다.

- `P`: 차선 중앙의 좌우 위치 오차
- `C`: 곡률 보정값
- `H`: 실제 차선 기울기 보정값
- `LA`: `FULL` 또는 코너 탈출용 `EXIT` 룩어헤드와 현재 가중치
- `B`: 좌우 경계가 모두 실제 검출된 밴드 수
- `W`: 위치만 사용하는 유예 프레임 수

`Traffic ROI` 화면에서 신호등 ROI는 청록색, 자동차 ROI는 자홍색 사각형으로 표시됩니다. 자동차 ROI 안에서 찾은 후면은 주황색 박스와 confidence/전체 프레임 면적 비율로 표시됩니다. 상태 패널의 `Car`, `US all`, `Fusion`, `Car ROI` 줄에서 모델 결과 나이, 전체 수신 센서값, 융합 시각 차이, 누적 횟수와 현재 ROI 비율을 확인할 수 있습니다.

## 자동 로깅과 영상 녹화

프로그램을 실행할 때마다 다음 폴더가 자동 생성됩니다.

```text
traffic_mission_v02_0804/logs/YYYYMMDD_HHMMSS_mmm/
├─ session.json       실행 당시 전체 calib 설정, 카메라·포트·모델 경로
├─ telemetry.csv      매 처리 프레임의 제어·검출·센서·상태 값
├─ drive_ui.avi       Integrated Drive에 렌더링된 원본/BEV/상태 패널
├─ traffic_roi.avi    같은 카메라의 신호등·자동차 ROI와 검출 결과
├─ traffic_clean.avi  ROI 선과 검출 표시가 없는 신호등 카메라 원본
└─ lane_clean.avi     ROI 선과 검출 표시가 없는 차선 모델 입력 영상
```

`log_video_enabled=1`, `log_clean_traffic_video_enabled=1`, `log_clean_lane_video_enabled=1`인 기본 설정에서는 AVI가 총 4개 생성됩니다. 네 영상은 신호 재출발 후 3초가 지나면 동시에 종료되며 파일 자체는 세션 폴더에 유지됩니다. `lane_clean.avi`는 왜곡 보정 후 차선 모델에 전달되는 프레임을 차선 마스크, BEV 사다리꼴, ROI 선 없이 기록합니다. AVI는 Windows OpenCV에서 호환성이 높은 `MJPG`를 기본 사용합니다. 녹화되는 것은 OpenCV가 렌더링한 영상과 상태 패널이며 Windows 창 테두리와 조절 창 자체 모양은 포함되지 않습니다. 대신 모든 BEV/ROI 값이 상태 패널과 `session.json`에 저장됩니다. `q`로 정상 종료하거나 오류로 `finally` 정리가 실행되면 CSV를 flush하고 아직 열린 동영상을 닫습니다.

`telemetry.csv`의 주요 진단 열은 다음과 같습니다.

| 확인 대상 | 열 |
|---|---|
| 시간·영상 대응 | `frame_index`, `video_frame_index`, `video_recording_active`, `wall_time`, `elapsed_s`, `event` |
| 조향·속도 | `lane_error`, `raw_steer`, `final_steer`, `final_speed`, `recovery_ramp_active`, `recovery_ramp_elapsed_s`, `recovery_ramp_pwm`, `steering_pot` |
| 자동차 모델 | `car_detector_enabled`, `car_detected`, `car_confidence`, `car_bbox`, `car_area_ratio`, `car_result_age_s`, `car_roi_normalized`, `car_roi_pixels`, `car_model_fault` |
| 초음파 | `ultrasonic_json`, `ultrasonic_age_json`, `front_sensor_values`, `front_distance_cm`, `sensor_ok` |
| 융합·회피 | `ultrasonic_close`, `fusion_confirmed`, `fusion_hits`, `fusion_skew_s`, `fusion_reason`, `obstacle_phase` |
| 차선·신호 | `lane_found`, `lane_width_measured_px`, `left_slope_factor`, `right_slope_factor`, `stop_line_detected`, `traffic_signal`, `traffic_result_age_s`, `traffic_model_fault`, `traffic_mission_enabled`, `signal_stop_latched`, `avoidance_blocked_by_signal`, `lost_frames`, `vehicle_state` |

문제가 발생한 시각은 `drive_ui.avi`에서 확인한 뒤 같은 `video_frame_index` 행을 CSV에서 찾습니다. 신호등·자동차 ROI 표시가 판단에 방해되면 같은 인덱스의 `traffic_clean.avi`, 차선 검출 오버레이가 방해되면 `lane_clean.avi`를 확인합니다. 영상 종료 이후 CSV 행은 `video_recording_active=0`, 빈 `video_frame_index`로 기록되고 종료 프레임의 `event`에는 `signal-resume-video-recording-stopped`가 남습니다. 좌우로 흔들리면 `lane_error/raw_steer/final_steer/steering_pot`, 회피가 시작되지 않으면 `car_detected/ultrasonic_close/fusion_reason/fusion_skew_s`, 갑자기 멈추면 `vehicle_state/event/sensor_ok/car_model_fault` 순서로 확인합니다. AVI 용량이 부담되면 `log_clean_traffic_video_enabled=0` 또는 `log_clean_lane_video_enabled=0`으로 각 깨끗한 원본 영상을 따로 끄거나, `log_video_enabled=0`으로 모든 영상을 끄고 CSV는 계속 기록할 수 있습니다. 전체 기록을 끄려면 `logging_enabled=0`으로 설정합니다. 녹화 파일 크기 제한과 자동 삭제는 하지 않으므로 실차 시험 전에 디스크 여유 공간을 확인하고, 필요한 세션을 백업한 뒤 오래된 `logs` 세션은 사용자가 정리해야 합니다.

## 파일 구조

```text
traffic_mission_v02_0804/
├─ integrated_traffic_obstacle.py  전체 실행 진입점
├─ run_traffic_mission.py          카메라, 신호등, 차선, 장애물 통합 루프
├─ lane_model.py                   차선 분할, BEV 변환, 중앙 및 경계 클래스 계산
├─ obstacle_controller.py          장애물 회피 상태기와 안전 조건
├─ README.md                       실행법과 전체 메커니즘
├─ VARIABLES.md                    calib.json 변수 설명
├─ model/
│  ├─ lane_best.pt                 현재 사용하는 차선 분할 모델
│  ├─ car_best.pt                  현재 사용하는 장애물 자동차 모델
│  └─ best_traffic_light_v2.pt     신호등 검출 모델
├─ shared/
│  ├─ calib.json                   실제 실행 조절값
│  ├─ config.py                    기본값, 불러오기, 저장
│  ├─ car_detector.py              비동기 car_best.pt 자동차 ROI 추론
│  ├─ traffic_detector.py          비동기 신호등 ROI 최신 프레임 추론
│  ├─ control.py                   차선 중앙 PD 조향
│  ├─ run_logger.py                CSV와 UI/ROI/원본 영상 세션 기록
│  ├─ serial_driver.py             Windows COM 통신과 초음파 수신 시각 관리
│  └─ undistort.py                 카메라 왜곡 보정
└─ camera_intrinsic/               카메라별 내부 파라미터
```

모든 실행 모델은 `traffic_mission_v02_0804/model/` 안에서 읽습니다. 차선은 `lane_best.pt`, 장애물 자동차는 `car_best.pt`, 신호등은 `best_traffic_light_v2.pt`를 사용합니다. 세 파일 중 하나라도 없으면 카메라 및 주행 초기화 전에 오류로 실행을 중단합니다. 루트의 모델 파일을 교체하더라도 `0804`에는 자동 반영되지 않으므로, 새 모델을 적용하려면 해당 파일을 `0804/model/`에도 복사해야 합니다.

## 원본 보존

이 폴더는 새로 생성되었으며 원본 `drive`와 `traffic_mission_v02_0802`는 변경하지 않습니다. 주행 쪽을 다시 비교할 때는 `drive/shared/calib.json`, 미션 쪽을 다시 비교할 때는 `traffic_mission_v02_0802`를 기준으로 확인합니다.

## 조절 시 주의사항

`drive_pwm=255`, `slow_pwm=255`는 최고 속도 설정입니다. 또한 `change_steer=1.0`, 방향별 `change_duration_* = 2.4`, `counter_steer_duration_s=1.2`, `change_pwm=70`은 강한 개방루프 회피입니다. 자동차+초음파가 처음 융합되는 순간부터 속도를 `70` 이하로 제한하고, 완료 후에는 `70 → 255`로 3초 동안 올립니다. 반드시 바퀴를 띄운 dry-run과 낮춘 `Drive PWM`으로 공간 시험을 끝낸 뒤 실제 코스에 투입해야 합니다.
