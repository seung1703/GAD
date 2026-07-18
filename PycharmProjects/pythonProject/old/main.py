import Function_Library as fl
import time
import numpy as np

EPOCH = 500000

# ===== 설정 =====
ARDUINO_PORT = 'COM7'
LIDAR_PORT   = 'COM6'
BAUD_RATE    = 115200
GO_PWM       = 40        # 초록불 전진 속도
STOP_DIST_MM = 700       # 정지 거리 (70cm = 700mm)

# 전방 각도 범위 (라이다 정면 기준, 필요시 조정)
FRONT_MIN_A  = 350       # 350도 ~ 360도
FRONT_MAX_A  = 360
FRONT_MIN_A2 = 0         # 0도 ~ 10도
FRONT_MAX_A2 = 10

if __name__ == "__main__":
    env = fl.libCAMERA()

    # ===== 아두이노 연결 =====
    arduino = fl.libARDUINO()
    ser = arduino.init(port=ARDUINO_PORT, baudrate=BAUD_RATE)
    print("아두이노 연결 완료!")

    # ===== 라이다 연결 =====
    lidar = fl.libLIDAR(LIDAR_PORT)
    lidar.init()
    lidar.getState()
    scan_generator = lidar.scanning()
    print("라이다 연결 완료!")

    def send(cmd):
        try:
            ser.write((cmd + '\n').encode())
        except:
            pass

    def check_front_obstacle(scan):
        """전방에 STOP_DIST_MM 이내 물체가 있으면 True"""
        # 전방 두 구간 (350~360, 0~10)
        front1 = lidar.getAngleDistanceRange(scan, FRONT_MIN_A, FRONT_MAX_A, 1, STOP_DIST_MM)
        front2 = lidar.getAngleDistanceRange(scan, FRONT_MIN_A2, FRONT_MAX_A2, 1, STOP_DIST_MM)
        # 둘 중 하나라도 점이 있으면 장애물
        return len(front1) > 0 or len(front2) > 0

    # Camera Initial Setting
    ch0, ch1 = env.initial_setting(cam0port=1, capnum=1)

    send('S90')
    send('X')
    time.sleep(0.5)
    print("주행 시작! (q로 종료)")

    last_drive = 'X'

    try:
        for i in range(EPOCH):
            _, frame0 = env.camera_read(ch0, ch1)
            env.image_show(frame0)

            # ===== 라이다 스캔 (최신 1회) =====
            obstacle = False
            try:
                scan = next(scan_generator)
                obstacle = check_front_obstacle(scan)
            except StopIteration:
                pass
            except Exception as e:
                print(f"라이다 에러: {e}")

            # ===== 신호등 인식 =====
            color = env.object_detection(frame0, sample=16, print_enable=True)

            # ===== 제어 로직 (우선순위) =====
            if obstacle:
                last_drive = 'X'                  # 1순위: 장애물 → 정지
                print("전방 70cm 이내 장애물 → 정지!")
            elif color == "RED":
                last_drive = 'X'                  # 2순위: 빨간불 → 정지
            elif color == "GREEN":
                last_drive = f'D{GO_PWM}'         # 초록불 → 전진
            # 그 외 → 마지막 명령 유지

            send(last_drive)

            if env.loop_break():
                break

    finally:
        send('X')
        send('S90')
        time.sleep(0.2)
        ser.close()
        lidar.stop()
        print("종료 완료")