# traffic_mission_v02_0804 변수 정리

실제 실행값은 `shared/calib.json`에서 조절합니다. 파일에 없는 값은 `shared/config.py`의 기본값을 사용합니다.

## 카메라와 통신

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `cam_index` | `3` | `drive`에서 가져온 주행 카메라 번호 |
| `traffic_cam_index` | `1` | 0802에서 유지한 신호등·자동차 카메라 번호 |
| `camera_id` | `3` | 왜곡 보정 파일 번호 |
| `frame_w`, `frame_h` | `640`, `360` | 요청할 주행 카메라 해상도 |
| `traffic_frame_w`, `traffic_frame_h` | `1280`, `720` | 요청할 신호등 카메라 해상도 |
| `undistort` | `1` | 주행 카메라 왜곡 보정 사용 여부 |
| `camera_backend` | `CAP_DSHOW` | Windows OpenCV 카메라 백엔드 |
| `camera_fps` | `30` | 카메라 드라이버에 요청할 FPS |
| `camera_fourcc` | `MJPG` | USB 전송량을 줄이기 위한 카메라 압축 형식 |
| `camera_buffer_size` | `1` | 오래된 프레임 지연을 줄이기 위한 캡처 버퍼 크기 |
| `serial_port` | `COM3` | 아두이노 연결 포트 |
| `baud` | `115200` | 시리얼 통신 속도 |
| `steer_mode` | `continuous` | 연속 조향 프로토콜 사용 |

카메라가 요청 해상도를 지원하지 않으면 콘솔에 실제 해상도가 출력되며, 왜곡 보정은 실제 해상도에 맞춰 생성됩니다.

## 차선 모델과 마스크

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `model_imgsz` | `320` | 차선 모델 추론 입력 크기. 카메라 원본 해상도와 별개 |
| `model_conf` | `0.4` | `drive` 차선 객체 confidence 하한 |
| `model_mask_threshold` | `0.5` | `drive` 분할 마스크 픽셀 채택 기준 |
| `mask_close_kernel` | `0` | `drive`와 같이 추가 마스크 닫기 연산을 사용하지 않음 |
| `model_device` | `cpu` | 차선 모델 실행 장치 |
| `model_cpu_threads` | `6` | CPU 모델들이 공유하는 PyTorch 연산 스레드 수 |
| `model_cpu_interop_threads` | `2` | PyTorch 연산 간 병렬 스레드 수 |
| `mask_roi_margin_x` | `0.0` | `drive` IPM과 같은 좌우 처리 영역 |
| `mask_roi_margin_y` | `0.0` | `drive` IPM과 같은 상하 처리 영역 |
| `boundary_class_min_pixels` | `2` | 경계 클래스 판정에 필요한 최소 픽셀 증거 |
| `boundary_class_ratio` | `1.1` | dash/solid 중 한 클래스를 확정할 최소 우세 비율 |
| `solid_orientation_filter_enabled` | `1` | BEV solid 방향 필터 사용 |
| `solid_min_component_pixels` | `20` | 분석할 solid 연결 영역의 최소 픽셀 수 |
| `solid_min_vertical_span_px` | `30` | 경계 후보에 필요한 최소 세로 길이 |
| `solid_max_angle_from_vertical_deg` | `60.0` | 세로축에서 허용할 최대 각도. 90도는 완전한 가로선 |
| `solid_prev_angle_tolerance_deg` | `35.0` | 이전 정상 solid 기울기와 허용할 최대 차이 |
| `solid_fit_corridor_px` | `18` | fitted solid 경로 주변에 보존할 반폭 |
| `solid_angle_ema_alpha` | `0.25` | 정상 solid 기울기의 새 값 반영 비율 |
| `solid_angle_reacquire_frames` | `5` | 기울기 불일치 후 이전 기준을 버릴 프레임 수 |
| `reacquire_frames` | `8` | 경계 소실 후 기존 추적 위치를 버릴 프레임 수 |
| `inner_lane_solid_side` | `left` | INNER 차선에서 solid가 보이는 화면 방향. INNER/OUTER 표시와 방향별 차선변경 시간 매핑에 사용 |

solid가 약하면 `model_conf`를 먼저 낮추고, 그다음 `model_mask_threshold`, `mask_roi_margin_x`를 조금씩 조절합니다. 너무 낮추면 노면 반사와 배경 오검출이 늘어날 수 있습니다.

solid 방향 필터는 BEV의 solid 연결 영역에서 행별 중앙점을 구해 곡선 경로와 기울기를 계산합니다. 세로 길이가 짧거나 수평에 가까운 영역, 이전 정상 solid와 각도 차이가 큰 영역은 `w_solid`와 경계 탐색용 `w_any`에서 제외합니다. 정상 경로 주변 `solid_fit_corridor_px` 밖의 픽셀도 제거하므로 실제 solid에 붙은 가로선 가지도 판단에 들어가지 않습니다. 원본 모델 오버레이는 바꾸지 않고 BEV에서 제외 픽셀만 빨간색으로 표시합니다.

실제 solid가 빨간색으로 빠지면 먼저 `solid_prev_angle_tolerance_deg`를 `40~45`로 올리고, 곡선 바깥쪽이 잘리면 `solid_fit_corridor_px`를 `22~26`으로 올립니다. 가로선이 계속 파란색으로 남으면 `solid_max_angle_from_vertical_deg`를 `50~55`로 내리거나 `solid_min_vertical_span_px`를 올립니다. 너무 엄격하게 설정하면 좌회전 진입의 실제 solid도 사라져 차선 소실 안전정지가 발생할 수 있습니다.

## BEV와 차선 중앙

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `ipm_tl_x`, `ipm_tl_y` | `0.13`, `0.46` | 현재 `drive` BEV 사다리꼴 왼쪽 위 좌표 비율 |
| `ipm_tr_x`, `ipm_tr_y` | `0.91`, `0.46` | 현재 `drive` BEV 사다리꼴 오른쪽 위 좌표 비율 |
| `ipm_bl_x`, `ipm_bl_y` | `-0.05`, `0.94` | BEV 사다리꼴 왼쪽 아래 좌표 비율 |
| `ipm_br_x`, `ipm_br_y` | `1.09`, `0.94` | 현재 `drive` BEV 사다리꼴 오른쪽 아래 좌표 비율 |
| `center_offset` | `0` | 영상 중앙에 더할 목표 위치 보정 픽셀 |
| `bev_lane_width` | `328` | 한쪽 경계만 보일 때 사용할 BEV 차선 폭 |
| `n_bands` | `7` | BEV를 나눌 세로 구간 수 |
| `curve_gain` | `0.6` | `drive` 원근 구간 차이 곡률 반영량 |
| `lookahead_gain` | `5.0` | 먼 구간에 추가할 조향 가중치 |
| `lookahead_exit_scale` | `1.5` | 코너 탈출 중 `lookahead_gain`에 곱할 `drive` 비율 |
| `lookahead_exit_delta_px` | `1.0` | 코너 탈출 판정에 필요한 가까운 중앙 오차 감소량 |
| `lookahead_exit_confirm_frames` | `1` | `drive`처럼 오차 감소를 즉시 반영 |
| `heading_gain` | `0.6` | `drive` 차선 기울기 보정 반영량 |
| `heading_ema_alpha` | `1.0` | 현재 헤딩을 즉시 반영 |
| `heading_clamp` | `1.0` | 헤딩 절댓값 안전 상한 |
| `heading_min_bands` | `2` | 가중 헤딩 피팅에 필요한 최소 목표점 수 |
| `control_warmup_frames` | `3` | 회피 완료·신호 재출발 후 위치만 사용할 정상 검출 프레임 수 |

`ipm_*`, `center_offset`, `bev_lane_width`, `n_bands`, `kp`, `kd`, `steer_right_gain`, `heading_gain`, `drive_pwm`은 별도 `Lane ROI / Drive Controls` 창에서 실시간 변경합니다. 세로 스크롤바 또는 마우스 휠로 모든 항목을 볼 수 있으며 `Save config (W)` 버튼이나 메인 창의 `w` 키로 저장할 수 있습니다. 조절 창을 닫아 숨겼다면 메인 화면 오른쪽 위 `CONTROLS` 버튼으로 다시 띄웁니다.

`FULL` 모드는 `lookahead_gain` 전체를 사용하고, 가까운 중앙 오차가 연속 감소하면 `EXIT` 모드로 전환해 룩어헤드를 줄입니다. 기울기는 좌우 경계가 모두 실제 검출된 밴드만 사용하며, 한쪽 경계를 `bev_lane_width`로 추정한 밴드는 제외합니다. 최종 입력 오차는 중앙 위치, 곡률, 기울기 보정의 합입니다.

## 일반 주행 제어

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `kp` | `1.8` | 차선 중앙 오차 비례 게인 |
| `kd` | `6.0` | 오차 변화량 미분 게인 |
| `steer_right_gain` | `1.0` | 음수 조향 방향의 추가 배율 |
| `steer_sign` | `-1` | 영상 오차와 실제 조향 방향 변환 부호 |
| `drive_pwm` | `255` | 일반 주행 목표 속도 |
| `slow_pwm` | `255` | 큰 조향 시 적용할 속도 상한 |
| `slow_steer_thresh` | `0.55` | `slow_pwm` 적용 조향 절댓값 |
| `lost_hold_frames` | `8` | 차선 소실 시 마지막 조향을 유지할 프레임 수 |
| `lost_stop_frames` | `30` | 일반 주행 중 차선 소실 안전정지 프레임 수 |

회피 중에는 일반 PD 제어와 `lost_stop_frames` 카운터를 사용하지 않습니다. 회피 완료 시 PD의 이전 오차를 초기화하고 `control_warmup_frames` 동안 중앙 위치만 사용합니다. 신호 정지 중에도 PD를 갱신하지 않으며 재출발 시 같은 초기화를 수행합니다.

## 신호등

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `traffic_confidence` | `0.25` | 신호등 confidence 하한 |
| `traffic_image_size` | `416` | 신호등 모델 추론 입력 크기. 512보다 빠른 균형값 |
| `traffic_inference_interval_s` | `0.08` | 비동기 신호등 추론 사이의 최소 간격 |
| `traffic_result_stale_s` | `0.50` | 신호 판정에 사용할 수 있는 결과의 최대 나이 |
| `traffic_roi_left` | `0.06` | 전체 프레임 너비 대비 ROI 왼쪽 경계 |
| `traffic_roi_top` | `0.11` | 전체 프레임 높이 대비 ROI 위쪽 경계 |
| `traffic_roi_right` | `0.95` | 전체 프레임 너비 대비 ROI 오른쪽 경계 |
| `traffic_roi_bottom` | `0.41` | 전체 프레임 높이 대비 ROI 아래쪽 경계 |
| `lane_confirm_count` | `2` | 정지 구역 `lane` 확인 프레임 수 |
| `stop_signal_confirm_count` | `2` | 모델 `green` 클래스 정지 확인 프레임 수 |
| `go_signal_confirm_count` | `2` | 정지 후 모델 `red` 클래스 출발 확인 프레임 수 |

네 `traffic_roi_*` 값은 0.0~1.0 비율이며, `Traffic ROI` 창의 `Left %`, `Top %`, `Right %`, `Bottom %` 트랙바로 실시간 변경합니다. 실제 YOLO 입력에는 이 사각형으로 자른 픽셀만 들어갑니다. 오른쪽/아래쪽 경계는 왼쪽/위쪽 경계보다 최소 0.01 크게 자동 보정됩니다. `w` 키를 누르면 BEV 값과 함께 `shared/calib.json`에 저장됩니다.

ROI를 작게 줄이면 배경 오검출과 추론할 픽셀을 줄일 수 있지만, 신호등이 ROI 밖으로 벗어나면 절대 검출되지 않습니다. 실제 차량을 움직이지 않는 `--dry-run` 상태에서 카메라 위치와 주행 코스를 확인하며 먼저 맞추는 것이 안전합니다.

신호등 모델은 제어 루프와 별도 스레드에서 최신 프레임만 처리합니다. 제어 루프가 같은 결과를 여러 번 읽어도 확인 프레임 수는 증가하지 않으며, 새 추론 결과가 도착했을 때만 `lane/stop/go` 누적을 한 번 갱신합니다. 결과가 `traffic_result_stale_s`를 넘으면 `unknown`으로 보고 부분 누적을 지우므로 FPS 향상을 위해 안전 확인 횟수가 약해지지 않습니다. `Traffic ROI` 영상의 `age=... async`에서 현재 결과 나이를 확인합니다.

이 모델은 실제 색상과 클래스 이름이 반대로 학습되어 모델 `green`을 정지, 모델 `red`를 출발로 해석합니다. 신호 미션 상태는 설정 파일 변수가 아니라 런타임 상태기 `ACTIVE -> HOLD_RED -> COMPLETE`로 관리합니다. `HOLD_RED` 진입과 동시에 회피 상태기를 초기화하고 회피 판단을 잠급니다. `COMPLETE`에서는 신호등 모델과 자동차 모델 추론을 동시에 생략하고 자동차+초음파 회피도 비활성화합니다. 비활성 자동차 모델은 파이프라인 타임아웃 안전정지 조건에서도 제외됩니다. UI의 `RESET` 또는 `r`은 상태를 `ACTIVE`로 되돌리고 두 검출기를 다시 활성화한 뒤 모든 누적 상태와 센서 캐시를 초기화하며, `x`/`Space`는 상태를 유지한 채 모터만 정지합니다.

## 장애물 감지와 즉시 재감지

### 자동차 후면 모델과 초음파 융합

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `car_model_required` | `1` | 자동차+초음파 이중 조건 강제. 실차에서는 끄지 않음 |
| `car_roi_left` | `0.00` | 전체 프레임 대비 자동차 ROI 왼쪽 경계 |
| `car_roi_top` | `0.00` | 전체 프레임 대비 자동차 ROI 위쪽 경계 |
| `car_roi_right` | `1.00` | 전체 프레임 대비 자동차 ROI 오른쪽 경계 |
| `car_roi_bottom` | `1.00` | 전체 프레임 대비 자동차 ROI 아래쪽 경계 |
| `car_image_size` | `512` | `traffic_mission_v02_0804/model/car_best.pt` 추론 입력 크기 |
| `car_confidence` | `0.60` | 회피 후보로 인정할 `stroller` confidence 하한. 이 라벨을 장애물 자동차 의미로 사용 |
| `car_min_area_ratio` | `0.005` | 전체 프레임 대비 자동차 박스 최소 비율 |
| `car_class_names` | `["stroller"]` | `car_best.pt`에서 회피 후보로 허용하는 모델 클래스 |
| `car_inference_interval_s` | `0.06` | 비동기 자동차 추론 최소 간격 |
| `car_result_stale_s` | `0.80` | 융합에 사용할 자동차 원본 프레임 최대 나이 |
| `car_pipeline_timeout_s` | `2.0` | 새 모델 결과가 없을 때 파이프라인 실패 판정 시간 |
| `car_startup_grace_s` | `5.0` | 모델 첫 결과를 기다리는 실행 직후 유예 시간 |
| `fusion_max_skew_s` | `0.60` | 자동차 프레임과 초음파 패킷의 최대 시각 차이 |

`traffic_mission_v02_0804/model/car_best.pt`를 장애물 자동차 모델로 사용합니다. 체크포인트의 실제 클래스 이름은 `stroller`이며, 이 프로젝트에서는 해당 라벨을 장애물 자동차로 해석합니다. 자동차 모델은 `traffic_cam_index` 카메라에서 `car_roi_*`로 자른 픽셀만 사용하며, 검출 박스 좌표는 다시 전체 화면 좌표로 변환합니다. 박스 면적 비율은 ROI 면적이 아니라 전체 프레임 면적을 기준으로 유지합니다. 네 값은 `Traffic ROI` 창의 `Car Left/Top/Right/Bottom %`로 즉시 조절하고 `w`로 저장합니다. `obs_trigger_hits`는 초음파 단독 횟수가 아니라 새 초음파 패킷과 유효한 자동차 검출이 동시에 만족된 횟수입니다. 자동차만 보이거나 초음파만 가까운 경우 회피 상태로 전환하지 않습니다.

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `us_front_ids` | `[0, 1]` | 전방 거리로 사용할 초음파 센서 ID |
| `obs_trigger_cm` | `150` | 이 거리보다 가까우면 장애물 후보 |
| `obs_trigger_hits` | `1` | 서로 다른 새 융합 측정값에서 확인할 횟수 |
| `obstacle_slow_cm` | `30` | 일반 주행과 회피 시작 직전 속도를 제한하는 기준 |
| `obstacle_slow_pwm` | `10` | 초근접 장애물 감지 시 적용할 속도 상한 |
| `direction_confirm_frames` | `3` | dash 방향을 확정할 영상 프레임 수 |
| `direction_hold_frames` | `5` | 일시적으로 방향을 못 봤을 때 기존 판정을 유지할 프레임 수 |

회피 완료 후 clear 거리나 cooldown을 기다리지 않고 즉시 감지를 다시 시작합니다. 회피 전에 사용한 측정값은 다시 세지 않으며, 완료 이후 들어온 새 근거리 측정값이 `obs_trigger_hits`회 확인되어야 다음 회피를 시작합니다. `WAIT_DASH` 중에는 모터를 정지하지 않고 차선 중앙 주행을 계속합니다.

## 장애물 회피 동작

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `change_steer` | `1.0` | dash 방향 조향 절댓값 |
| `change_t_pre` | `0.0` | 회피 조향 전 직진 대기 시간 |
| `change_duration_inner_to_outer_s` | `2.4` | INNER에서 OUTER로 갈 때 dash 방향 조향 유지 시간 |
| `change_duration_outer_to_inner_s` | `2.4` | OUTER에서 INNER로 갈 때 dash 방향 조향 유지 시간 |
| `counter_steer_duration_s` | `1.2` | 반대 방향 조향 유지 시간 |
| `change_pwm` | `70` | 회피 중 주행 속도 |
| `recovery_ramp_start_pwm` | `70` | 회피 완료 직후 재가속 시작 속도 |
| `recovery_ramp_duration_s` | `3.0` | 시작 속도에서 `drive_pwm`까지 올리는 실제 주행 시간 |
| `fsm_max_step_s` | `0.25` | 한 루프에서 회피 시간에 반영할 최대 시간 |

dash가 왼쪽이면 양수 조향, 오른쪽이면 음수 조향을 사용합니다. `inner_lane_solid_side=left`인 현재 설정에서는 오른쪽 dash로 조향하면 INNER→OUTER, 왼쪽 dash로 조향하면 OUTER→INNER입니다. `inner_lane_solid_side=right`로 바꾸면 이 매핑도 자동으로 반대가 됩니다. 회피 시작 시 방향, 방향별 조향 시간과 속도를 함께 잠그므로 회피 도중 차선 클래스나 초음파 거리가 변해도 바뀌지 않습니다. 이전 설정 파일에 두 새 변수가 없을 때만 기존 `change_duration_s`를 호환용 fallback으로 읽습니다. 회피 중 UI 거리는 `IGNORED`로 표시하며, 회피 완료 다음 프레임부터 초음파 거리를 다시 사용합니다.

회피 완료 시 속도는 `70`에서 시작해 실제로 움직인 시간 3초 동안 `255`까지 선형 증가합니다. 수동 정지, 신호 정지 또는 안전정지 중에는 램프 시간이 흐르지 않습니다. 램프 중에도 자동차+근거리 초음파 융합이 확인되면 즉시 `change_pwm` 이하로 제한하고 새 회피가 램프보다 우선합니다.

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
| `view_width`, `view_height` | `896`, `504` | 원본 및 BEV 화면 각각의 16:9 표시 크기. 실제 카메라 입력 해상도는 변경하지 않음 |
| `panel_width` | `540` | 오른쪽 카드형 상태 정보 패널 너비 |

메인 `Integrated Drive` 창에는 위쪽 원본 주행 영상과 아래쪽 BEV 영상을 16:9 비율로 표시합니다. 오른쪽 로그는 `MISSION`, `OBSTACLE FUSION`, `LANE CONTROL`, `LIVE TUNING` 카드로 나뉘며 패널을 크게 그렸다가 축소하지 않으므로 작은 글자가 이전보다 선명합니다. `Traffic ROI` 창은 신호등/자동차 ROI 조절용으로 계속 별도 표시됩니다. 이 UI 변수들은 화면 크기만 바꾸며 카메라 캡처 해상도, 모델 입력, 조향 또는 미션 판단에는 영향을 주지 않습니다.

## 로깅과 녹화

| 변수 | 현재값 | 설명 |
|---|---:|---|
| `logging_enabled` | `1` | 실행별 `session.json`과 `telemetry.csv` 기록 |
| `log_dir` | `logs` | 0804 폴더 기준 세션 로그 상위 폴더 |
| `log_video_enabled` | `1` | 네 AVI 전체 녹화의 상위 스위치 |
| `log_clean_traffic_video_enabled` | `1` | ROI 선과 검출 표시가 없는 신호등 카메라 원본 `traffic_clean.avi` 녹화 |
| `log_clean_lane_video_enabled` | `1` | ROI 선과 검출 표시가 없는 차선 모델 입력 `lane_clean.avi` 녹화 |
| `video_stop_after_signal_resume_s` | `3.0` | 신호 재출발 확정 후 네 AVI를 더 녹화할 시간. 경과하면 영상만 종료 |
| `log_video_fps` | `15.0` | AVI 재생 프레임률 |
| `log_video_codec` | `MJPG` | Windows OpenCV 녹화 FourCC |
| `log_flush_interval_s` | `1.0` | CSV 디스크 flush 주기 |
| `log_ultrasonic_ids` | `[0..7]` | CSV에 항상 열거할 센서 ID. 실제 수신 ID도 자동 추가 |

신호 재출발 후 3초가 지나면 네 AVI writer를 모두 닫습니다. 프로그램, 카메라 표시, 주행 및 `telemetry.csv` 기록은 종료하지 않습니다. 영상 종료 뒤 CSV의 `video_recording_active`는 `0`, `video_frame_index`는 빈 값이 됩니다. 같은 실행 세션에서 `RESET`을 눌러도 이미 닫힌 AVI를 다시 열거나 덮어쓰지 않습니다.

CSV는 매 처리 프레임 기록됩니다. `lane_width_measured_px`, `left_slope_factor`, `right_slope_factor`는 `drive` 경계 피팅 상태를 보여줍니다. `raw_steer`는 차선/정지 처리 후 회피 명령을 덮어쓰기 전 값이고, `final_steer`는 실제 `MegaLink`로 전달한 값입니다. `steering_pot`는 펌웨어가 `P:` telemetry를 보내는 경우의 실제 조향 potentiometer 값입니다. `ultrasonic_json`은 현재 캐시값, `ultrasonic_age_json`은 센서별 마지막 실제 패킷 나이이므로 값이 남아 있어도 오래된 센서를 구분할 수 있습니다.
