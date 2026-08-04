"""
line_detect.py -- 마스크를 **직선**으로 바꾸는 부분.

parking_v5 에서는 세그멘테이션 모델이 마스크를 만들고(model_detect.py),
여기서는 그 마스크를 선 단위로 나눠 가장 가까운 것 하나에 직선을 맞춘다.
(고전 CV 이진화 경로는 line_straight/ 쪽에만 남겨두고 여기선 뺐다)
"""

import cv2
import numpy as np

import config as C


def _split_lines(mask, roi_h):
    """이진 마스크를 **선 단위로 분리**해 각각 직선을 맞춘다.

    → [(각도, 아래쪽 x, 픽셀수, 선분), ...]  쓸 만한 것만

    끊긴 선(점선·가림)이 여러 조각으로 쪼개지지 않도록 살짝 부풀린 뒤
    연결요소를 찾는다. 라벨은 부풀린 마스크에서 얻고, 직선 맞추기는
    **원본 마스크 픽셀**로 해서 굵어진 만큼 각도가 흐려지지 않게 한다.
    """
    # ★세로 방향으로만 부풀린다★ 정사각 커널을 쓰면 나란히 있는 다른 선까지
    # 가로로 붙어서 하나로 합쳐진다. 선은 대체로 세로로 뻗으므로, 세로로만
    # 이으면 끊긴 조각은 붙이면서 옆 선과는 안 섞인다.
    k = np.ones((C.LINK_KSIZE, 1), np.uint8)
    linked = cv2.dilate(mask, k)
    n, lab = cv2.connectedComponents(linked)

    out = []
    for i in range(1, n):
        sel = (lab == i) & (mask > 0)          # 원본 픽셀만
        ys, xs = np.nonzero(sel)
        if ys.size < C.MIN_PIXELS:
            continue
        # 한 덩어리(얼룩)를 직선으로 착각하지 않게 세로 퍼짐을 본다
        if (ys.max() - ys.min()) / max(1, roi_h) < C.MIN_Y_SPAN:
            continue
        ang, xb, seg = _fit_angle(ys.astype(np.float32),
                                  xs.astype(np.float32), roi_h)
        if abs(ang) > C.MAX_ABS_ANGLE:
            continue
        out.append((ang, xb, int(ys.size), seg))
    return out



def _fit_angle(ys, xs, roi_h):
    """점들에 직선을 맞춘다.

    → (기울기[도], ROI 바닥에서의 x, 선분 양끝 ((x,y),(x,y)) )   전부 ROI 로컬

    cv2.fitLine + DIST_HUBER: 최소제곱과 달리 이상치(잡티 몇 점)에 덜 끌린다.

    ★끝점까지 여기서 계산해 돌려준다★
      예전엔 그리는 쪽에서 각도로 끝점을 다시 만들었는데, 거기서 부호가
      뒤집혀 빨간 선이 거울처럼 반대로 그려졌다. 계산은 한 곳에서만 한다.

    ★x 는 항상 'ROI 바닥'에서 잰다★
      마스크의 최하단 픽셀에서 재면, 선이 짧게 잡힌 프레임마다 기준 높이가
      달라져 횡위치가 들쭉날쭉해진다. 제어 입력이 흔들리면 안 된다.
    """
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    vx, vy, px, py = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    # 방향을 "화면 위쪽(=전방)" 으로 통일한다. y 는 아래로 증가하므로 vy<0.
    if vy > 0:
        vx, vy = -vx, -vy
    angle = float(np.degrees(np.arctan2(vx, -vy)))   # 0=수직, +=전방이 우측으로

    def x_at(y):
        return float(px + vx * (y - py) / (vy if abs(vy) > 1e-6 else -1e-6))

    x_bot = x_at(float(roi_h))                       # 기준 높이 = ROI 바닥, 항상 동일
    y_lo, y_hi = float(ys.min()), float(ys.max())    # 실제 픽셀이 있는 구간만 그린다
    return angle, x_bot, ((x_at(y_hi), y_hi), (x_at(y_lo), y_lo))


def _split_lines(mask, roi_h):
    """이진 마스크를 **선 단위로 분리**해 각각 직선을 맞춘다.

    → [(각도, 아래쪽 x, 픽셀수, 선분), ...]  쓸 만한 것만

    끊긴 선(점선·가림)이 여러 조각으로 쪼개지지 않도록 살짝 부풀린 뒤
    연결요소를 찾는다. 라벨은 부풀린 마스크에서 얻고, 직선 맞추기는
    **원본 마스크 픽셀**로 해서 굵어진 만큼 각도가 흐려지지 않게 한다.
    """
    # ★세로 방향으로만 부풀린다★ 정사각 커널을 쓰면 나란히 있는 다른 선까지
    # 가로로 붙어서 하나로 합쳐진다. 선은 대체로 세로로 뻗으므로, 세로로만
    # 이으면 끊긴 조각은 붙이면서 옆 선과는 안 섞인다.
    k = np.ones((C.LINK_KSIZE, 1), np.uint8)
    linked = cv2.dilate(mask, k)
    n, lab = cv2.connectedComponents(linked)

    out = []
    for i in range(1, n):
        sel = (lab == i) & (mask > 0)          # 원본 픽셀만
        ys, xs = np.nonzero(sel)
        if ys.size < C.MIN_PIXELS:
            continue
        # 한 덩어리(얼룩)를 직선으로 착각하지 않게 세로 퍼짐을 본다
        if (ys.max() - ys.min()) / max(1, roi_h) < C.MIN_Y_SPAN:
            continue
        ang, xb, seg = _fit_angle(ys.astype(np.float32),
                                  xs.astype(np.float32), roi_h)
        if abs(ang) > C.MAX_ABS_ANGLE:
            continue
        out.append((ang, xb, int(ys.size), seg))
    return out


