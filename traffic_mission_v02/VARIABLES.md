# traffic_mission_v02 변수 정리

실제 실행값은 `shared/calib.json`에서 조절합니다. 파일에 없는 값은 `shared/config.py`의 기본값을 사용합니다.

## 카메라와 통신

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `cam_index` | `3` | 주행 카메라 번호 |
| `traffic_cam_index` | `2` | 신호등 카메라 번호 |
| `camera_id` | `3` | 왜곡 보정 파일 번호 |
| `frame_w`, `frame_h` | `640`, `360` | 요청할 주행 카메라 해상도 |
| `traffic_frame_w`, `traffic_frame_h` | `1280`, `720` | 요청할 신호등 카메라 해상도 |
| `undistort` | `1` | 주행 카메라 왜곡 보정 사용 여부 |
| `camera_backend` | `CAP_DSHOW` | Windows OpenCV 카메라 백엔드 |
| `serial_port` | `COM3` | 아두이노 연결 포트 |
| `baud` | `115200` | 시리얼 통신 속도 |
| `steer_mode` | `continuous` | 연속 조향 프로토콜 사용 |

카메라가 요청 해상도를 지원하지 않으면 콘솔에 실제 해상도가 출력되며, 왜곡 보정은 실제 해상도에 맞춰 생성됩니다.

## 차선 모델과 마스크

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `model_imgsz` | `320` | 차선 모델 추론 입력 크기. 카메라 원본 해상도와 별개 |
| `model_conf` | `0.3` | 차선 객체 confidence 하한 |
| `model_mask_threshold` | `0.4` | 분할 마스크 픽셀 채택 기준 |
| `mask_close_kernel` | `3` | 끊긴 dash/solid 마스크를 연결하는 커널 크기 |
| `model_device` | `cpu` | 차선 모델 실행 장치 |
| `mask_roi_margin_x` | `0.08` | BEV 기준 영역 좌우에 추가할 마스크 보존 폭 |
| `mask_roi_margin_y` | `0.03` | BEV 기준 영역 위아래에 추가할 마스크 보존 높이 |
| `boundary_class_min_pixels` | `2` | 경계 클래스 판정에 필요한 최소 픽셀 증거 |
| `boundary_class_ratio` | `1.1` | dash/solid 중 한 클래스를 확정할 최소 우세 비율 |
| `reacquire_frames` | `8` | 경계 소실 후 기존 추적 위치를 버릴 프레임 수 |
| `inner_lane_solid_side` | `left` | 화면의 INNER/OUTER 디버그 문구 해석 방향이며 실제 제어에는 사용하지 않음 |

solid가 약하면 `model_conf`를 먼저 낮추고, 그다음 `model_mask_threshold`, `mask_roi_margin_x`를 조금씩 조절합니다. 너무 낮추면 노면 반사와 배경 오검출이 늘어날 수 있습니다.

## BEV와 차선 중앙

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `ipm_tl_x`, `ipm_tl_y` | `0.19`, `0.46` | BEV 사다리꼴 왼쪽 위 좌표 비율 |
| `ipm_tr_x`, `ipm_tr_y` | `0.85`, `0.46` | BEV 사다리꼴 오른쪽 위 좌표 비율 |
| `ipm_bl_x`, `ipm_bl_y` | `-0.05`, `0.94` | BEV 사다리꼴 왼쪽 아래 좌표 비율 |
| `ipm_br_x`, `ipm_br_y` | `1.03`, `0.94` | BEV 사다리꼴 오른쪽 아래 좌표 비율 |
| `center_offset` | `0` | 영상 중앙에 더할 목표 위치 보정 픽셀 |
| `bev_lane_width` | `328` | 한쪽 경계만 보일 때 사용할 BEV 차선 폭 |
| `n_bands` | `7` | BEV를 나눌 세로 구간 수 |
| `curve_gain` | `0.3` | 원근 구간 차이로 계산한 곡률 반영량 |
| `lookahead_gain` | `5.0` | 먼 구간에 추가할 조향 가중치 |
| `lookahead_exit_scale` | `0.4` | 코너 탈출 중 `lookahead_gain`에 곱할 비율 |
| `lookahead_exit_delta_px` | `1.0` | 코너 탈출 판정에 필요한 가까운 중앙 오차 감소량 |
| `lookahead_exit_confirm_frames` | `3` | 오차 감소를 연속 확인할 프레임 수 |
| `heading_gain` | `0.2` | 실제 차선 기울기 보정 반영량 |
| `heading_ema_alpha` | `0.25` | 차선 기울기 EMA 필터의 새 값 반영 비율 |
| `heading_clamp` | `0.25` | 필터링된 차선 기울기 절댓값 상한 |
| `heading_min_bands` | `4` | 기울기 계산에 필요한 양쪽 실제 경계 밴드 수 |
| `control_warmup_frames` | `3` | 회피 완료·신호 재출발 후 위치만 사용할 정상 검출 프레임 수 |

`ipm_*`, `center_offset`, `bev_lane_width`는 UI 트랙바로 실시간 변경하고 `w` 키로 저장할 수 있습니다.

`FULL` 모드는 `lookahead_gain` 전체를 사용하고, 가까운 중앙 오차가 연속 감소하면 `EXIT` 모드로 전환해 룩어헤드를 줄입니다. 기울기는 좌우 경계가 모두 실제 검출된 밴드만 사용하며, 한쪽 경계를 `bev_lane_width`로 추정한 밴드는 제외합니다. 최종 입력 오차는 중앙 위치, 곡률, 기울기 보정의 합입니다.

## 일반 주행 제어

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `kp` | `1.8` | 차선 중앙 오차 비례 게인 |
| `kd` | `6.0` | 오차 변화량 미분 게인 |
| `steer_right_gain` | `1.0` | 음수 조향 방향의 추가 배율 |
| `steer_sign` | `-1` | 영상 오차와 실제 조향 방향 변환 부호 |
| `drive_pwm` | `70` | 일반 주행 속도 |
| `slow_pwm` | `70` | 큰 조향 시 감속 속도 |
| `slow_steer_thresh` | `0.55` | `slow_pwm` 적용 조향 절댓값 |
| `lost_hold_frames` | `8` | 차선 소실 시 마지막 조향을 유지할 프레임 수 |
| `lost_stop_frames` | `30` | 일반 주행 중 차선 소실 안전정지 프레임 수 |

회피 중에는 일반 PD 제어와 `lost_stop_frames` 카운터를 사용하지 않습니다. 회피 완료 시 PD의 이전 오차를 초기화하고 `control_warmup_frames` 동안 중앙 위치만 사용합니다. 신호 정지 중에도 PD를 갱신하지 않으며 재출발 시 같은 초기화를 수행합니다.

## 신호등

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `traffic_confidence` | `0.25` | 신호등 confidence 하한 |
| `traffic_image_size` | `512` | 신호등 모델 추론 입력 크기 |
| `traffic_roi_left` | `0.06` | 전체 프레임 너비 대비 ROI 왼쪽 경계 |
| `traffic_roi_top` | `0.05` | 전체 프레임 높이 대비 ROI 위쪽 경계 |
| `traffic_roi_right` | `0.95` | 전체 프레임 너비 대비 ROI 오른쪽 경계 |
| `traffic_roi_bottom` | `0.55` | 전체 프레임 높이 대비 ROI 아래쪽 경계 |
| `lane_confirm_count` | `2` | 정지 구역 `lane` 확인 프레임 수 |
| `stop_signal_confirm_count` | `2` | `green/yellow` 정지 신호 확인 프레임 수 |
| `go_signal_confirm_count` | `2` | `red` 출발 신호 확인 프레임 수 |

네 `traffic_roi_*` 값은 0.0~1.0 비율이며, `Traffic ROI` 창의 `Left %`, `Top %`, `Right %`, `Bottom %` 트랙바로 실시간 변경합니다. 실제 YOLO 입력에는 이 사각형으로 자른 픽셀만 들어갑니다. 오른쪽/아래쪽 경계는 왼쪽/위쪽 경계보다 최소 0.01 크게 자동 보정됩니다. `w` 키를 누르면 BEV 값과 함께 `shared/calib.json`에 저장됩니다.

ROI를 작게 줄이면 배경 오검출과 추론할 픽셀을 줄일 수 있지만, 신호등이 ROI 밖으로 벗어나면 절대 검출되지 않습니다. 실제 차량을 움직이지 않는 `--dry-run` 상태에서 카메라 위치와 주행 코스를 확인하며 먼저 맞추는 것이 안전합니다.

## 장애물 감지와 즉시 재감지

### 자동차 후면 모델과 초음파 융합

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `car_model_required` | `1` | 자동차+초음파 이중 조건 강제. 실차에서는 끄지 않음 |
| `car_image_size` | `512` | `best_car.pt` 추론 입력 크기 |
| `car_confidence` | `0.45` | `obstacle-car` confidence 하한 |
| `car_min_area_ratio` | `0.01` | ROI 면적 대비 자동차 박스 최소 비율 |
| `car_class_names` | `["obstacle-car"]` | 회피 후보로 허용하는 모델 클래스 |
| `car_inference_interval_s` | `0.10` | 비동기 자동차 추론 최소 간격 |
| `car_result_stale_s` | `0.80` | 융합에 사용할 자동차 원본 프레임 최대 나이 |
| `car_pipeline_timeout_s` | `2.0` | 새 모델 결과가 없을 때 파이프라인 실패 판정 시간 |
| `car_startup_grace_s` | `5.0` | 모델 첫 결과를 기다리는 실행 직후 유예 시간 |
| `fusion_max_skew_s` | `0.60` | 자동차 프레임과 초음파 패킷의 최대 시각 차이 |

`model/best_car.pt`의 실제 클래스는 `obstacle-car`입니다. 자동차 모델은 별도 ROI 없이 `traffic_cam_index` 신호등 카메라의 전체 프레임을 사용하며, 신호등 모델만 `traffic_roi_*`로 잘라서 사용합니다. `obs_trigger_hits`는 초음파 단독 횟수가 아니라 새 초음파 패킷과 유효한 자동차 검출이 동시에 만족된 횟수입니다. 자동차만 보이거나 초음파만 가까운 경우 회피 상태로 전환하지 않습니다.

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `us_front_ids` | `[0, 1]` | 전방 거리로 사용할 초음파 센서 ID |
| `obs_trigger_cm` | `150` | 이 거리보다 가까우면 장애물 후보 |
| `obs_trigger_hits` | `3` | 서로 다른 새 측정값에서 연속 확인할 횟수 |
| `obstacle_slow_cm` | `80` | 일반 주행과 회피 시작 직전 속도를 제한하는 기준 |
| `obstacle_slow_pwm` | `50` | 근접 장애물 감지 시 잠글 제한 속도 |
| `direction_confirm_frames` | `3` | dash 방향을 확정할 영상 프레임 수 |
| `direction_hold_frames` | `5` | 일시적으로 방향을 못 봤을 때 기존 판정을 유지할 프레임 수 |

회피 완료 후 clear 거리나 cooldown을 기다리지 않고 즉시 감지를 다시 시작합니다. 회피 전에 사용한 측정값은 다시 세지 않으며, 완료 이후 들어온 새 근거리 측정값이 `obs_trigger_hits`회 확인되어야 다음 회피를 시작합니다. `WAIT_DASH` 중에는 모터를 정지하지 않고 차선 중앙 주행을 계속합니다.

## 장애물 회피 동작

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `change_steer` | `1.0` | dash 방향 조향 절댓값 |
| `change_t_pre` | `0.0` | 회피 조향 전 직진 대기 시간 |
| `change_duration_s` | `2.2` | dash 방향 조향 유지 시간 |
| `counter_steer_duration_s` | `1.0` | 반대 방향 조향 유지 시간 |
| `change_pwm` | `70` | 회피 중 주행 속도 |
| `fsm_max_step_s` | `0.25` | 한 루프에서 회피 시간에 반영할 최대 시간 |

dash가 왼쪽이면 양수 조향, 오른쪽이면 음수 조향을 사용합니다. 회피 시작 시 방향과 속도를 잠그므로 회피 도중 차선 클래스나 초음파 거리가 변해도 조향 방향과 속도는 바뀌지 않습니다. 회피 중 UI 거리는 `IGNORED`로 표시하며, 회피 완료 다음 프레임부터 초음파 거리를 다시 사용합니다.

## 초음파 안전

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `sensor_timeout_s` | `0.8` | 마지막 실제 초음파 패킷을 유효하게 볼 시간 |
| `sensor_min_valid_cm` | `2` | 사용할 수 있는 최소 거리 |
| `sensor_max_valid_cm` | `400` | 사용할 수 있는 최대 거리 |
| `sensor_required` | `1` | 유효한 전방 센서가 없을 때 안전정지 |
| `ultrasonic_enabled` | `1` | 아두이노 초음파 스트림 요청 여부 |

거리값은 캐시 존재 여부가 아니라 센서별 실제 패킷 수신 시각으로 유효성을 판정합니다.

UI의 `Front US`에는 `U0`, `U1`의 최신 유효값을 따로 표시하고, `filtered`에는 실제 판단에 사용한 최솟값을 표시합니다. 회피 중에는 모든 초음파 표시가 `IGNORED`로 바뀝니다.

## UI

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `view_width`, `view_height` | `800`, `520` | 원본 및 BEV 화면 각각의 표시 크기 |
| `panel_width` | `440` | 오른쪽 상태 정보 패널 너비 |

## 로깅과 녹화

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `logging_enabled` | `1` | 실행별 `session.json`과 `telemetry.csv` 기록 |
| `log_dir` | `logs` | v02 기준 세션 로그 상위 폴더 |
| `log_video_enabled` | `1` | `drive_ui.avi`, `traffic_roi.avi` 녹화 |
| `log_video_fps` | `15.0` | AVI 재생 프레임률 |
| `log_video_codec` | `MJPG` | Windows OpenCV 녹화 FourCC |
| `log_flush_interval_s` | `1.0` | CSV 디스크 flush 주기 |
| `log_ultrasonic_ids` | `[0..7]` | CSV에 항상 열거할 센서 ID. 실제 수신 ID도 자동 추가 |

CSV는 매 처리 프레임 기록됩니다. `raw_steer`는 차선/정지 처리 후 회피 명령을 덮어쓰기 전 값이고, `final_steer`는 실제 `MegaLink`로 전달한 값입니다. `steering_pot`는 펌웨어가 `P:` telemetry를 보내는 경우의 실제 조향 potentiometer 값입니다. `ultrasonic_json`은 현재 캐시값, `ultrasonic_age_json`은 센서별 마지막 실제 패킷 나이이므로 값이 남아 있어도 오래된 센서를 구분할 수 있습니다.
