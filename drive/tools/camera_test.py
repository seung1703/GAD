import cv2

def main():
    # 0번 카메라는 기본 내장 웹캠 또는 첫 번째로 연결된 USB 카메라를 의미합니다.
    # 만약 화면이 안 나오거나 다른 카메라를 쓰고 싶다면 1 또는 2로 변경해 보세요.
    camera_index = 0
    
    cap = cv2.VideoCapture(camera_index)

    # 카메라가 정상적으로 열렸는지 확인
    if not cap.isOpened():
        print(f"Error: {camera_index}번 카메라를 열 수 없습니다.")
        return

    print("카메라 화면 출력을 시작합니다. 종료하려면 'q' 키를 누르세요.")

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        # 프레임을 성공적으로 읽지 못했다면 루프 탈출
        if not ret:
            print("화면을 가져오는 데 실패했습니다.")
            break

        # 화면에 카메라 영상 표시
        cv2.imshow("Camera View", frame)

        # 키 입력 대기 (1ms 동안 대기하며 'q' 키가 눌렸는지 확인)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("카메라를 안전하게 종료합니다.")
            break

    # 사용한 자원 해제 및 윈도우 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()