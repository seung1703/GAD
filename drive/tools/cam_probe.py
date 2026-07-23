#!/usr/bin/env python3
"""
cam_probe.py -- 맥 카메라 인덱스 확인 툴.

인덱스 0~5를 전부 열어보고, 열리는 카메라마다 창을 띄워서
"몇 번이 어떤 캠인지" 눈으로 확인한다.

키:
  0~5 : 해당 인덱스를 캠1(바닥, cam_index)로 calib.json에 저장
  q   : 종료

팁: 아이폰이 근처에 있으면 Continuity Camera가 인덱스를 하나 차지한다.
    (제어 센터에서 끄거나 아이폰을 멀리 두기)
"""

import cv2

from config import load, save

MAX_INDEX = 6


def main():
    cfg = load()
    caps = {}
    for i in range(MAX_INDEX):
        # 맥은 AVFoundation 백엔드 명시가 안정적
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                caps[i] = cap
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[probe] index {i}: OPEN  ({w}x{h})")
                continue
        cap.release()
        print(f"[probe] index {i}: 없음")

    if not caps:
        print("열리는 카메라가 없음. 카메라 권한(시스템 설정→개인정보 보호) 확인")
        return

    print(f"\n열린 카메라 {len(caps)}대. 각 창을 보고 바닥캠 인덱스 숫자키를 누르면 저장. q=종료")

    while True:
        for i, cap in caps.items():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (480, 270))
            cv2.putText(frame, f"INDEX {i}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, "press this number to set as cam1",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
            cv2.imshow(f"camera index {i}", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if ord('0') <= k <= ord('5'):
            idx = k - ord('0')
            if idx in caps:
                cfg["cam_index"] = idx
                save(cfg)
                print(f"[probe] cam_index = {idx} 저장 완료 (main.py가 이 캠 사용)")
            else:
                print(f"[probe] index {idx}는 안 열린 카메라")

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
