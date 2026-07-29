"""
test_ultrasonic.py -- 초음파 센서 값만 받아서 확인하는 테스트 스크립트.
아두이노가 보내는 "U,fL,fC,fR,bL,bC,bR\n" 줄을 파싱해서 출력한다.
"""
import serial
import time

PORT = "COM7"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=0)
time.sleep(2.0)   # 아두이노 리셋 대기
print(f"연결됨: {PORT}")

buf = ""
try:
    while True:
        if ser.in_waiting:
            buf += ser.read(ser.in_waiting).decode(errors="ignore")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if line.startswith("U,"):
                    parts = line.split(",")
                    if len(parts) == 7:
                        fL, fC, fR, bL, bC, bR = map(int, parts[1:])
                        print(f"전방 L={fL:3d} C={fC:3d} R={fR:3d}  |  "
                              f"후방 L={bL:3d} C={bC:3d} R={bR:3d}  (cm)")
        time.sleep(0.01)
except KeyboardInterrupt:
    print("종료")
finally:
    ser.close()