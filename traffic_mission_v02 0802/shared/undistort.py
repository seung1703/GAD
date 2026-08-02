"""
undistort.py -- 카메라 렌즈 왜곡 보정 (오버헤드 최소화).

camera_intrinsic/camera<N>_intrinsic.txt (ROS camera_calibration 형식)에서
K(카메라행렬)·D(왜곡계수)를 읽어, 캡처 해상도에 맞춰 스케일한 뒤
initUndistortRectifyMap 으로 remap 맵을 '한 번만' 계산해둔다.
매 프레임은 cv2.remap 만 호출 → 640x360에서 ~0.5ms (undistort 통짜 호출의 5~10배 빠름).

사용:
    und = Undistorter(cam_index, frame_w, frame_h, intrinsic_dir)
    frame = und(frame)          # 보정 (파일 없거나 비활성이면 원본 그대로 반환)
"""

import os
import re

import cv2
import numpy as np


def _parse_intrinsic(path):
    """ROS 형식 txt → (K 3x3, D 1x5, calib_w, calib_h). 실패 시 예외."""
    txt = open(path).read()

    def _nums(after):
        m = re.search(after + r"\s*=\s*\[([^\]]+)\]", txt)
        return [float(x) for x in m.group(1).split(",")] if m else None

    kv = _nums("K")
    dv = _nums("D")
    if kv is None or dv is None:
        raise ValueError(f"K/D 파싱 실패: {path}")
    K = np.array(kv, dtype=np.float64).reshape(3, 3)
    D = np.array(dv, dtype=np.float64).reshape(1, -1)
    w = int(re.search(r"width\s+(\d+)", txt).group(1))
    h = int(re.search(r"height\s+(\d+)", txt).group(1))
    return K, D, w, h


class Undistorter:
    def __init__(self, cam_index, frame_w, frame_h, intrinsic_dir,
                 enabled=True):
        self.enabled = False
        self.map1 = self.map2 = None
        if not enabled:
            print("[undistort] 비활성 (undistort=0)")
            return

        path = os.path.join(intrinsic_dir,
                            f"camera{int(cam_index)}_intrinsic.txt")
        if not os.path.exists(path):
            print(f"[undistort] 내부파라미터 없음 → 보정 생략: {path}")
            return
        try:
            K, D, cw, ch = _parse_intrinsic(path)
        except Exception as e:
            print(f"[undistort] 파싱 실패 → 보정 생략: {e}")
            return

        # 캘리브레이션 해상도(cw x ch)와 캡처 해상도(frame_w x frame_h)가
        # 다르면 K의 초점거리/주점을 비율만큼 스케일 (D는 무차원이라 그대로)
        sx = frame_w / cw
        sy = frame_h / ch
        Ks = K.copy()
        Ks[0, 0] *= sx; Ks[0, 2] *= sx     # fx, cx
        Ks[1, 1] *= sy; Ks[1, 2] *= sy     # fy, cy

        # newCameraMatrix = Ks (동일 초점거리 유지) → 확대/축소 없이 왜곡만 편다.
        # 기존 IPM/ROI 튜닝과 시야가 최대한 가깝게 유지됨.
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            Ks, D, None, Ks, (frame_w, frame_h), cv2.CV_16SC2)
        self.enabled = True
        print(f"[undistort] camera{cam_index} 보정 활성 "
              f"(calib {cw}x{ch} → capture {frame_w}x{frame_h})")

    def __call__(self, frame):
        if not self.enabled:
            return frame
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
