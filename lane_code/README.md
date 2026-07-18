# 자율주행 시간측정 미션 — 차선검출 주행 코드

## 폴더 구조
```
shared/        공용 모듈: config.py(파라미터), control.py(PD제어),
               serial_driver.py(아두이노 통신), calib.json(현재 튜닝값)
dl_model/      딥러닝 주행 (메인): main_model.py 실행
               model/lane_seg_best.pt = YOLO11n-seg (dash/lane/solid)
cv_classic/    구버전 OpenCV(centroid) 방식 백업: main.py
car_firmware/  아두이노 Mega 펌웨어 (.ino) — 업로드 필요
tools/         카메라 인덱스 확인 등 진단 툴
```

## 환경 설치
Python 3.9+ 에서:
```
pip install ultralytics opencv-python pyserial pynput
```
(torch는 ultralytics가 같이 설치함. Apple Silicon이면 그대로 OK)

## 실행
```
python dl_model/main_model.py --dry      # 시리얼 없이 비전만 (첫 테스트)
python dl_model/main_model.py            # 실차 주행
python dl_model/main_model.py --cam 1    # 카메라 인덱스 지정
python dl_model/main_model.py --record   # 주행하며 데이터셋 녹화
```

키: `s`=출발 `x`/Space=정지 `t`=튠모드(추론정지+트랙바 빠릿) `w`=설정저장
`d`=디버그토글 `q`=종료 | 전역: ESC=비상정지 F12=종료 (pynput, 손쉬운사용 권한 필요)

## 주의
- 시리얼 포트는 한 프로세스만: 아두이노 IDE 시리얼 모니터 켜져 있으면 열기 실패
- 펌웨어 워치독 400ms ↔ 파이썬 150ms 하트비트 세트 — 펌웨어 맘대로 바꾸지 말 것
- 조향 프로토콜: 파이썬은 -1..1 정규화만 보냄. 포텐 캘리브레이션(STEER_*)은
  전부 펌웨어 담당 (중립 틀어지면 .ino의 STEER_NEUTRAL 수정 후 재업로드)
- 장소 바뀌면: 차를 차로 중앙에 정면으로 놓고 't' 튠모드 → BEV에서 차선이
  수직으로 서도록 IPM 트랙바 조정 → 'w' 저장

## 튜닝값 역할 (calib.json / Settings 트랙바)
- Kp: 오차 비례 조향 (복귀 속도) / Kd: 코너 탈출 시 핸들 미리 풀기
- curve_gain: 경로 굽음 선반영 (커브 안쪽 붙으면 ↓) / heading_gain: 차체 틀어짐 보정
- lookahead_gain: 먼 밴드 가중 / center_offset: 좌우 치우침 보정
- drive_pwm / slow_pwm: 직선/급조향 속도
