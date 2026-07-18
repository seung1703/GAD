import cv2
import os
import Function_Library as fl

# ===== 설정 =====
CAM_PORT = 1
IMG_W, IMG_H = 1280, 720
SAVE_DIR = "dataset/images"

# 저장 폴더가 없으면 자동 생성
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"'{SAVE_DIR}' 폴더를 생성했습니다.")

# ===== 카메라 초기화 =====
print("카메라 연결 중...")
env = fl.libCAMERA()
ch0, _ = env.initial_setting(cam0port=CAM_PORT, capnum=1)
ch0.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 현재 저장된 사진 개수 확인 (이어서 찍기 방지)
existing_files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')]
count = len(existing_files)

print("=" * 50)
print(f"📸 현재 수집된 사진: {count}장")
print(" [Spacebar] : 사진 촬영 및 저장")
print(" [q] 또는 [ESC] : 종료")
print("=" * 50)

while True:
    # 버퍼 비우기 (최신 프레임 유지)
    for _ in range(3):
        ch0.grab()

    ret, frame = ch0.retrieve()
    if not ret or frame is None:
        continue

    # 화면에 가이드 텍스트 표시 (저장되는 원본 사진에는 안 나옴)
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Saved: {count} / Goal: 250", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(display_frame, "Press SPACE to save, Q to quit", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Data Collector", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # 스페이스바를 누르면 원본 frame 저장
    if key == ord(' '):
        count += 1
        filename = os.path.join(SAVE_DIR, f"track_data_{count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[찰칵] {filename} 저장 완료! (총 {count}장)")

    # q 또는 ESC 누르면 종료
    elif key == ord('q') or key == 27:
        print("촬영을 종료합니다.")
        break

ch0.release()
cv2.destroyAllWindows()