import Function_Library as fl
import time

# 아두이노 연결 (COM7)
arduino = fl.libARDUINO()
ser = arduino.init(port="COM7", baudrate=115200)
print("아두이노 연결 완료")

def send(cmd):
    """명령 문자열에 개행 붙여서 전송"""
    ser.write((cmd + "\n").encode())
    print(f"전송: {cmd}")
    time.sleep(0.05)

# 테스트 시퀀스 (바퀴 띄워둔 상태에서)
print("=== 조향 테스트 ===")
send("S90")   # 중립
time.sleep(1)
send("S0")    # 우
time.sleep(1)
send("S180")  # 좌
time.sleep(1)
send("S90")   # 중립
time.sleep(1)

print("=== 구동 테스트 ===")
send("D80")   # 전진
time.sleep(1.5)
send("X")     # 정지
time.sleep(1)

print("테스트 완료")
ser.close()