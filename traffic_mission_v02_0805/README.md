# traffic_mission_v02_0805

`traffic_mission_v02_0804`의 차선 주행, 신호등 미션, 자동차 장애물 회피를 유지하면서 신호등과 자동차를 `best_11n.pt` 한 개로 동시에 검출하도록 바꾼 Windows용 버전입니다. 0804와 루트의 원본 모델은 수정하지 않았습니다.

## [핵심 변경]

- 통합 모델: `model/best_11n.pt`
- 통합 모델 클래스: `red`, `green`, `stroller`
- 의미: `red=정지`, `green=출발`, `stroller=장애물 자동차 뒷면`
- 신호등 카메라의 전체 프레임을 한 번 추론하고 같은 결과에서 신호등과 자동차를 분리합니다.
- 신호등 `red/green`은 ROI 없이 전체 화면에서 검출합니다.
- 자동차 `stroller`에만 별도 ROI를 적용하며 UI에서 실시간 조절할 수 있습니다.
- 자동차 회피는 여전히 `stroller 검출 + 전방 초음파 근거리`가 함께 만족될 때만 시작합니다.
- 차선 검출은 별도 `model/lane_best.pt`와 차선 카메라를 그대로 사용합니다.

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

회피 시작에는 두 조건이 모두 필요합니다.

```text
best_11n.pt의 stroller 검출
AND
자동차 박스 아래쪽 중앙점이 Car ROI 내부
AND
전방 초음파 최솟값 < obs_trigger_cm
AND
두 측정 시각 차이 <= fusion_max_skew_s
AND
위 조합이 obs_trigger_hits회 확인
=> 회피 방향 확정 및 차선 변경
```

자동차만 보이면 회피하지 않습니다. 초음파만 가까우면 회피하지 않습니다. 같은 초음파 측정값을 여러 루프에서 반복 사용해 확인 횟수를 올리지 않으며, 새로운 센서 샘플과 새로운 모델 결과 조합만 융합 횟수로 인정합니다.

현재 자동차 confidence는 `0.60`, 최소 면적 비율은 `0.01`입니다. `0.46` 같은 검출이나 너무 작은 박스는 회피 조건에 들어가지 않습니다. 기준을 통과했지만 ROI 밖인 자동차는 통합 화면에 회색 `OUTSIDE CAR ROI`로 표시되며 초음파와 융합되지 않습니다.

## [초음파 사용]

`us_front_ids=[0,1]`의 센서만 회피 시작 거리 판단에 사용하고, 두 센서 중 유효한 최솟값을 전방 거리로 사용합니다. 값이 `sensor_min_valid_cm` 미만이거나 `sensor_max_valid_cm` 초과이면 무효입니다. 마지막 수신 후 `sensor_timeout_s`가 지나도 무효입니다.

센서 값 `250cm`은 코드가 강제로 만드는 값이 아닙니다. 실제 펌웨어가 250을 보내면 그대로 표시됩니다. 단, `--dry-run --mock-obstacle`에서는 테스트를 위해 250과 근거리 값을 의도적으로 만듭니다.

좌우 초음파가 ID 2 이상으로 연결되어 있으면 CSV의 `ultrasonic_json`에는 기록되지만 현재 회피 시작과 방향 결정에는 사용하지 않습니다. 실제 센서 배치와 ID가 확인되지 않은 상태에서 좌우 센서를 조향 조건으로 넣으면 벽이나 옆 차선을 오판할 위험이 있어, 0805는 전방 융합만 제어에 사용합니다.

## [차선 변경]

회피 직전 차선 모델의 `dash_side`를 여러 프레임 확인해 이동 방향을 정합니다. 실제 조향을 시작할 때 방향과 시간을 잠그며, 회피 중 카메라가 흔들려도 방향을 토글하지 않습니다.

| 상태 | 동작 |
|---|---|
| `KEEP` | 일반 차선 추종, 자동차+초음파 융합 확인 |
| `WAIT_DASH` | 장애물은 확정됐지만 dash 방향이 불충분하여 일반 조향 유지 |
| `PREPARE` | 잠근 방향과 속도로 회피 준비 |
| `CHANGING` | `change_steer`로 잠근 방향 조향 |
| `COUNTER_STEER` | 반대 조향으로 차체 자세 복원 |
| `RECOVER` | 추적값을 초기화하고 일반 차선 주행 복귀 |

INNER에서 OUTER로 갈 때는 `change_duration_inner_to_outer_s`, OUTER에서 INNER로 갈 때는 `change_duration_outer_to_inner_s`를 사용합니다. 두 값을 따로 조정할 수 있습니다. `counter_steer_duration_s`는 반대 조향 시간입니다.

회피가 끝난 뒤 속도는 `recovery_ramp_start_pwm`에서 시작해 `recovery_ramp_duration_s` 동안 `drive_pwm`까지 선형으로 증가합니다. 복구 중에도 자동차+초음파 융합이 다시 만족되면 속도 복구보다 새 회피가 우선입니다.

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
| `s` | 모터 주행 시작 |
| `x` 또는 `Space` | 주행 일시 정지, 현재 미션 상태는 유지 |
| `r` 또는 `RESET` | 브레이크 후 실행 직후 상태처럼 전체 런타임 초기화 |
| `w` | 현재 UI 조절값을 `shared/calib.json`에 저장 |
| `CONTROLS` | 닫거나 숨긴 조절 창 다시 표시 |
| `q` | 브레이크, 로그 저장, 프로그램 종료 |

`RESET` 후에는 자동 출발하지 않습니다. 안전하게 초기화된 뒤 `s`를 눌러야 움직입니다.

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
