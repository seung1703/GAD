# Traffic Mission Final

이 폴더만 가지고 차량 주행 + 신호등 미션을 실행할 수 있도록 정리한 최종 실행본입니다.

## 구성

- `run_traffic_mission.py`: 실제 실행 메인 파일
- `traffic_green_control.py`: 동일 실행용 간단한 진입 파일
- `lane_model.py`: 차선/정지선 판단용 모델 처리
- `shared/`: 주행 파라미터, 시리얼 제어, 왜곡보정
- `model/lane_seg_best.pt`: 차선 세그멘테이션 모델
- `model/best_traffic_light_v2.pt`: 신호등 검출 모델
- `camera_intrinsic/`: 카메라 보정 파일

## 실행 환경

Python 3.9 기준 권장 패키지:

```bash
pip install ultralytics opencv-python pyserial numpy
```

## 실행 방법

기본 실행:

```bash
python run_traffic_mission.py --serial-port COM3 --cam 1 --traffic-cam 2
```

모터/조향 없이 화면만 확인:

```bash
python run_traffic_mission.py --dry --serial-port COM3 --cam 1 --traffic-cam 2
```

호환용 실행:

```bash
python traffic_green_control.py --serial-port COM3 --cam 1 --traffic-cam 2
```

## 현재 미션 로직

- 차선 주행은 `lane_seg_best.pt` 기반으로 수행
- 신호등은 `best_traffic_light_v2.pt`로 판별
- `lane_model.py`에서 BEV 기준 `lane` 검출이 확인되고
- 동시에 신호등이 `green` 또는 `yellow`로 연속 확인되면 정지
- 정지 상태에서 `red`가 연속 확인되면 다시 출발

## 조작 키

- `s`: 주행 시작
- `x` 또는 `Space`: 정지
- `w`: 현재 ROI/설정 저장
- `q`: 종료

## 설정 파일

- `shared/calib.json`에서 카메라 번호, PWM, 게인, ROI 관련 값을 관리
- 실행 중 `w`를 누르면 현재 값이 `shared/calib.json`에 저장

## 참고

- 기본 시리얼 포트는 `COM3`로 설정되어 있으니 실제 포트에 맞게 `--serial-port` 또는 `shared/calib.json`을 수정하면 됩니다.
- 카메라 보정은 `shared/calib.json`의 `camera_id`와 `camera_intrinsic/` 파일명을 기준으로 동작합니다.
- 기존 학습 데이터, 테스트 코드, 예전 버전, 중간 산출물은 루트의 `archive_review_nonruntime/`로 모아 확인할 수 있게 정리했습니다.
