# traffic_mission_v02

차선 주행, 신호등 검출, `best_car.pt` 자동차 후면 검출과 초음파 융합 장애물 회피를 동시에 수행하는 Windows용 통합 실행 폴더입니다. `GAD` 폴더는 참고만 하며 실행 중 수정하지 않습니다.

## 전체 실행

저장소 최상위 폴더 `C:\Users\dmsal\Desktop\traffic_car`에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe traffic_mission_v02\integrated_traffic_obstacle.py
```

현재 위치가 `GAD` 폴더라면 상위 폴더의 실행 파일을 지정합니다.

```powershell
.\..\.venv\Scripts\python.exe ..\traffic_mission_v02\integrated_traffic_obstacle.py
```

하드웨어 명령 없이 카메라와 알고리즘만 확인하려면 다음 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe traffic_mission_v02\integrated_traffic_obstacle.py --dry-run --mock-obstacle
```

실제 메인 루프는 `run_traffic_mission.py`에 있습니다.

## 현재 수행 알고리즘

### 차선 주행

1. `model/lane_seg_best.pt`로 전체 주행 카메라 프레임에서 `dash`, `lane`, `solid`를 분할합니다.
2. 카메라 원본은 `640×360`으로 유지하고, 제어 주기를 높이기 위해 차선 모델 추론 입력은 `320`을 사용합니다.
3. 원본 마스크에 확장된 BEV 처리 영역을 적용한 뒤 버드아이뷰로 변환합니다.
4. 여러 높이 구간에서 좌우 경계를 찾고, 한 구간이 아닌 전체 구간 투표로 각 경계를 `dash` 또는 `solid`로 분류합니다.
5. 차량은 `dash`와 `solid` 경계의 중앙을 목표로 주행합니다. 실제 제어에는 `inner/outer` 구분을 사용하지 않습니다.
6. 한쪽 경계만 보이면 `bev_lane_width`를 이용해 반대 경계를 추정합니다.
7. 일반 코너에서는 먼 구간의 차선 중심에 `lookahead_gain` 가중치를 주어 코너를 미리 따라갑니다.
8. 가까운 구간의 중앙 오차가 3프레임 연속 감소하면 코너 탈출로 판단하고 룩어헤드를 `lookahead_exit_scale`만큼 줄여 과도한 반대 조향을 억제합니다.
9. 좌우 경계가 모두 실제로 검출된 밴드가 4개 이상일 때만 차선 중심선의 기울기를 계산합니다. 이 값은 EMA 필터와 상한 제한을 거쳐 조향에 더합니다.
10. 장애물 회피 완료 또는 신호 정지 후 재출발 시 3개의 정상 검출 프레임 동안 곡률과 기울기 보정을 끄고 차선 중앙 위치만으로 조향합니다.

최종 차선 오차는 `중앙 위치 오차(P) + 곡률 보정(C) + 차선 기울기 보정(H)`입니다. 한쪽 경계를 추정한 프레임은 기울기 계산에서 제외되므로 solid 또는 dash가 일시적으로 사라졌을 때 잘못된 기울기 보정이 들어가지 않습니다.

원본 화면의 마스크 색상은 다음과 같습니다.

- 주황색: `dash`
- 초록색: `lane`
- 파란색 계열: `solid`
- 자홍색 사다리꼴: 실제 BEV 변환 기준
- 청록색 사다리꼴: 마스크를 보존하는 확장 처리 영역

화면 아래의 `raw D/L/S`는 ROI로 자르기 전 전체 프레임의 픽셀 수이고, `roi S`는 확장 처리 영역 안에 남은 solid 픽셀 수입니다.

오른쪽 상태 패널의 `Front US`에는 `U0`, `U1`의 최신 유효 거리가 각각 표시됩니다. `filtered`는 실제 장애물 판단에 사용한 두 센서의 최솟값입니다. 회피 중에는 초음파를 사용하지 않으므로 모두 `IGNORED`로 표시됩니다.

### 장애물 회피

장애물 회피는 다음 순서로 진행됩니다.

1. `model/best_car.pt`가 신호등 카메라 전체 프레임에서 `obstacle-car`를 검출하고, 같은 시점의 새 전방 초음파 측정값이 `obs_trigger_cm`보다 가까운 상태를 `obs_trigger_hits`회 확인합니다.
2. 차선 모델이 판단한 dash 방향을 `direction_confirm_frames` 프레임 동안 확인합니다.
3. dash 방향이 불명확하면 `WAIT_DASH` 상태에서 모터를 정지하지 않고 기존 차선 중앙 주행을 계속하며 방향을 확인합니다.
4. 방향이 확인되면 해당 방향을 잠급니다. 회피 중 solid 오검출이 발생해도 방향은 바뀌지 않습니다.
5. 회피 시작 직전 거리가 `obstacle_slow_cm` 이하인지 한 번 확인하고 회피 속도를 잠급니다.
6. `change_steer` 조향으로 dash 방향을 향해 `change_duration_s` 동안 이동합니다.
7. `counter_steer_duration_s` 동안 반대 방향으로 조향합니다.
8. 회피가 끝나면 기존 추적값과 PD 미분값을 초기화하고, 3개의 정상 검출 프레임 동안 중앙 위치만 사용한 뒤 곡률·기울기 보정을 다시 켭니다.
9. 회피가 끝나는 즉시 다시 장애물 감지를 시작합니다. 이후 새로 들어온 자동차+근거리 초음파 융합 측정 3회와 새 dash 방향이 확인되면 바로 다음 회피를 시작합니다.

회피 시작 조건은 다음 진리표를 따릅니다.

| `best_car.pt` 자동차 | 전방 초음파 `< obs_trigger_cm` | 동작 |
|---|---|---|
| X | X | 일반 차선 주행 |
| O | X | 회피하지 않음 |
| X | O | 근접 감속은 가능하지만 회피하지 않음 |
| O | O | 시각 동기와 연속 횟수 확인 후 회피 |

자동차 결과는 별도 스레드가 최신 프레임만 처리합니다. 자동차와 신호등은 모두 `traffic_cam_index` 카메라의 같은 원본 프레임을 공유하지만, 자동차 모델은 전체 프레임을 사용하고 신호등 모델만 `traffic_roi_*` 범위를 잘라 사용합니다. 자동차 원본 프레임 시각과 새 초음파 패킷 수신 시각 차이가 `fusion_max_skew_s` 이하여야 한 쌍으로 인정하며, 같은 초음파 패킷은 한 번만 셉니다. 자동차 결과가 `car_result_stale_s`보다 오래되면 검출로 인정하지 않습니다. 모델 스레드가 `car_pipeline_timeout_s` 동안 새 결과를 만들지 못하거나 오류가 발생하면 시작 유예 이후 `SAFE_STOP`으로 정지합니다.

회피 조향이 시작된 뒤에는 일반 차선 중앙 PD 조향, 차선 소실 안전정지, 초음파 거리값을 모두 무시합니다. UI의 거리는 `IGNORED`로 표시됩니다. 신호 상태가 바뀌거나 초음파값이 흔들려도 `CHANGING -> COUNTER_STEER` 동작은 잠긴 방향과 속도로 중간 정지 없이 끝까지 수행합니다. 사용자가 `x` 또는 `Space`로 정지한 경우에만 즉시 멈추고 회피 시간을 일시정지합니다. 회피가 끝난 다음 프레임부터 초음파 거리 사용을 다시 시작합니다.

현재 주요 회피 설정은 다음과 같습니다.

- 감지 거리: `obs_trigger_cm = 150`
- 새 측정 확인 횟수: `obs_trigger_hits = 3`
- 근접 감속 거리: `obstacle_slow_cm = 80`
- 근접 감속 속도: `obstacle_slow_pwm = 50`
- dash 방향 조향 시간: `change_duration_s = 2.2`
- 반대 조향 시간: `counter_steer_duration_s = 1.0`
- 회피 조향: `change_steer = 1.0`
- 회피 속도: `change_pwm = 70`

### 초음파 안전 처리

각 초음파 센서의 마지막 실제 패킷 수신 시각을 별도로 기록합니다. 과거 거리값이 딕셔너리에 남아 있어도 `sensor_timeout_s`보다 오래된 값은 사용하지 않습니다. `sensor_min_valid_cm`보다 작거나 `sensor_max_valid_cm`보다 큰 값도 무효 처리합니다.

일반 주행 또는 `WAIT_DASH`에서 유효한 최신 측정값이 없으면 `SAFE_STOP`으로 정지합니다. 이미 회피 조향이 시작된 뒤에는 초음파 거리와 센서 상태를 사용하지 않고 잠긴 방향, 속도, 시간으로 회피를 끝까지 수행합니다. 회피 완료 다음 프레임부터 센서 유효성을 다시 검사합니다.

### 신호등

신호등 모델은 신호등 카메라 전체 프레임이 아니라 `traffic_roi_left/top/right/bottom`으로 지정한 사각형의 픽셀만 사용합니다. 실행 중 `Traffic ROI` 창에는 전체 신호등 영상, 현재 ROI 테두리, ROI 안에서 검출된 객체가 함께 표시됩니다. 현재 코드 기준으로 `lane` 구역과 `green/yellow`가 함께 확인되면 정지하고, `red`가 확인되면 다시 출발합니다.

코드 기본값은 전체 영상 너비의 20~80%, 높이의 5~55%이고, 현재 `calib.json` 저장값은 너비의 6~95%, 높이의 5~55%입니다. 카메라가 다른 해상도로 열려도 비율을 실제 픽셀 좌표로 다시 계산하므로 같은 상대 위치를 유지합니다.

신호 정지 중에는 PD 계산을 갱신하지 않습니다. 재출발 시 PD의 이전 오차, 기울기 필터, 룩어헤드 판정 상태를 초기화하고 3개의 정상 검출 프레임 동안 중앙 위치만 사용합니다.

## ROI UI

`Integrated Drive` 창의 트랙바에서 BEV 사다리꼴과 중앙값을 실시간 조절할 수 있습니다. 별도의 `Traffic ROI` 창에서는 신호등 검출 범위를 실시간 조절합니다.

- `TL`, `TR`, `BL`, `BR`: BEV 사다리꼴 네 꼭짓점
- `Center`: 목표 중앙 보정
- `LaneW`: 한쪽 경계만 보일 때 사용할 차선 폭
- `Left %`, `Top %`: 신호등 ROI의 왼쪽·위쪽 시작 위치
- `Right %`, `Bottom %`: 신호등 ROI의 오른쪽·아래쪽 끝 위치
- `w`: 현재 값을 `shared/calib.json`에 저장
- `s`: 주행 시작
- `x` 또는 `Space`: 정지
- `q`: 종료

신호등 ROI 트랙바 값은 전체 프레임에 대한 백분율입니다. 오른쪽은 왼쪽보다, 아래쪽은 위쪽보다 최소 1% 크게 자동 보정되므로 경계가 뒤집히거나 빈 이미지가 YOLO에 전달되지 않습니다. 조절값은 즉시 다음 프레임 추론에 반영되며 `w`를 누르기 전에는 파일에 저장되지 않습니다.

오른쪽 상태 패널의 `Control`과 `Lookahead` 줄은 현재 제어값을 보여줍니다.

- `P`: 차선 중앙의 좌우 위치 오차
- `C`: 곡률 보정값
- `H`: 실제 차선 기울기 보정값
- `LA`: `FULL` 또는 코너 탈출용 `EXIT` 룩어헤드와 현재 가중치
- `B`: 좌우 경계가 모두 실제 검출된 밴드 수
- `W`: 위치만 사용하는 유예 프레임 수

`Traffic ROI` 화면에는 신호등 ROI와 함께 전체 프레임에서 찾은 자동차 후면이 주황색 박스와 confidence/전체 프레임 면적 비율로 표시됩니다. 자동차 ROI는 사용하지 않습니다. 상태 패널의 `Car`, `US all`, `Fusion` 줄에서 모델 결과 나이, 전체 수신 센서값, 융합 시각 차이와 누적 횟수를 확인할 수 있습니다.

## 자동 로깅과 영상 녹화

프로그램을 실행할 때마다 다음 폴더가 자동 생성됩니다.

```text
traffic_mission_v02/logs/YYYYMMDD_HHMMSS_mmm/
├─ session.json       실행 당시 전체 calib 설정, 카메라·포트·모델 경로
├─ telemetry.csv      매 처리 프레임의 제어·검출·센서·상태 값
├─ drive_ui.avi       Integrated Drive에 렌더링된 원본/BEV/상태 패널
└─ traffic_roi.avi    같은 카메라의 신호등 ROI와 전체 프레임 자동차 검출
```

AVI는 Windows OpenCV에서 호환성이 높은 `MJPG`를 기본 사용합니다. 녹화되는 것은 OpenCV가 렌더링한 영상과 상태 패널이며 Windows 창 테두리와 트랙바 자체 모양은 포함되지 않습니다. 대신 모든 BEV/ROI 값이 상태 패널과 `session.json`에 저장됩니다. `q`로 정상 종료하거나 오류로 `finally` 정리가 실행되면 CSV를 flush하고 동영상을 닫습니다.

`telemetry.csv`의 주요 진단 열은 다음과 같습니다.

| 확인 대상 | 열 |
|---|---|
| 시간·영상 대응 | `frame_index`, `video_frame_index`, `wall_time`, `elapsed_s`, `event` |
| 조향·속도 | `lane_error`, `raw_steer`, `final_steer`, `final_speed`, `steering_pot` |
| 자동차 모델 | `car_detected`, `car_confidence`, `car_bbox`, `car_area_ratio`, `car_result_age_s`, `car_model_fault` |
| 초음파 | `ultrasonic_json`, `ultrasonic_age_json`, `front_sensor_values`, `front_distance_cm`, `sensor_ok` |
| 융합·회피 | `ultrasonic_close`, `fusion_confirmed`, `fusion_hits`, `fusion_skew_s`, `fusion_reason`, `obstacle_phase` |
| 차선·신호 | `lane_found`, `stop_line_detected`, `traffic_signal`, `lost_frames`, `vehicle_state` |

문제가 발생한 시각은 `drive_ui.avi`에서 확인한 뒤 같은 `video_frame_index` 행을 CSV에서 찾습니다. 좌우로 흔들리면 `lane_error/raw_steer/final_steer/steering_pot`, 회피가 시작되지 않으면 `car_detected/ultrasonic_close/fusion_reason/fusion_skew_s`, 갑자기 멈추면 `vehicle_state/event/sensor_ok/car_model_fault` 순서로 확인합니다. AVI 용량이 부담되면 `log_video_enabled=0`으로 영상만 끄고 CSV는 계속 기록할 수 있으며, 전체 기록을 끄려면 `logging_enabled=0`으로 설정합니다. 녹화 파일 크기 제한과 자동 삭제는 하지 않으므로 실차 시험 전에 디스크 여유 공간을 확인하고, 필요한 세션을 백업한 뒤 오래된 `logs` 세션은 사용자가 정리해야 합니다.

## 파일 구조

```text
traffic_mission_v02/
├─ integrated_traffic_obstacle.py  전체 실행 진입점
├─ run_traffic_mission.py          카메라, 신호등, 차선, 장애물 통합 루프
├─ lane_model.py                   차선 분할, BEV 변환, 중앙 및 경계 클래스 계산
├─ obstacle_controller.py          장애물 회피 상태기와 안전 조건
├─ README.md                       실행법과 전체 메커니즘
├─ VARIABLES.md                    calib.json 변수 설명
├─ model/
│  ├─ best_car.pt                  자동차 후면 검출 모델
│  ├─ lane_seg_best.pt             차선 분할 모델
│  └─ best_traffic_light_v2.pt     신호등 검출 모델
├─ shared/
│  ├─ calib.json                   실제 실행 조절값
│  ├─ config.py                    기본값, 불러오기, 저장
│  ├─ car_detector.py              비동기 best_car.pt 전체 프레임 추론
│  ├─ control.py                   차선 중앙 PD 조향
│  ├─ run_logger.py                CSV와 두 UI 영상 세션 기록
│  ├─ serial_driver.py             Windows COM 통신과 초음파 수신 시각 관리
│  └─ undistort.py                 카메라 왜곡 보정
└─ camera_intrinsic/               카메라별 내부 파라미터
```

자동차 모델은 `traffic_mission_v02/model/best_car.pt`를 읽습니다. 모델 파일을 이동하거나 이름을 바꾸면 안전하게 실행을 중단합니다.

## 롤백 지점

이번 차선 제어 보정을 넣기 직전 상태는 `archive_review_nonruntime/shot1/traffic_mission_v02`에 파일 단위로 보존되어 있습니다. 이후 `shot1 상태로 돌려줘`라고 요청하면 이 복사본을 기준으로 `traffic_mission_v02`를 복원할 수 있습니다. `GAD`는 백업과 수정 대상에 포함하지 않았습니다.

## 조절 시 주의사항

`change_steer = 1.0`, `change_duration_s = 2.2`, `counter_steer_duration_s = 1.0`, `change_pwm = 70`은 강한 개방루프 조향입니다. 회피 시작 직전 장애물이 `80cm` 이하이면 회피 속도를 `50`으로 잠그고, 회피 중에는 추가 거리 변화로 속도를 바꾸지 않습니다. 반드시 바퀴를 띄운 dry-run 확인과 저속 공간 시험 후 실제 코스에서 사용해야 합니다. solid가 약하면 우선 원본 화면의 `raw S`와 `roi S`를 비교한 뒤 `model_conf`, `model_mask_threshold`, `mask_roi_margin_x` 순서로 조절합니다.
