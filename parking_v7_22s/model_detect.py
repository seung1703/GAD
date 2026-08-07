"""
model_detect.py -- 학습한 세그멘테이션 모델로 선을 찾는다 (고전 CV 대체).

line_detect.LineFinder 와 **완전히 같은 인터페이스**다:
    finder.process(frame) -> dict(found, angle, x_bot, npix, nlines, ...)
그래서 컨트롤러(controller.py)도 화면 그리기(line_detect.draw)도 그대로 쓴다.
config.DETECTOR 한 줄로 갈아끼울 수 있다.

★고전 CV 와의 차이★
  CV  : 밝기 임계 → 흰 픽셀. 조명·바닥 얼룩에 흔들리지만 1~2ms 로 빠르다.
  모델: 학습한 "주차선"만 골라낸다. 얼룩·다른 흰 물체를 안 잡지만 30~40ms.

★공통점★
  마스크를 얻은 뒤는 똑같다 — 선 단위로 나눠 가장 가까운 것 하나를 고르고,
  직선을 맞춰 (기울기, 아래쪽 x) 를 낸다. 그래야 제어부가 검출 방식을 몰라도 된다.
"""

import collections

import cv2
import numpy as np

import config as C
from line_detect import _split_lines

_model = None


def _load():
    """모델은 처음 쓸 때만 읽는다 (CV 모드에서는 ultralytics 를 import 도 안 함)."""
    global _model
    if _model is None:
        from ultralytics import YOLO       # 여기서만 import
        print(f"[model] 로딩 {C.MODEL_PATH}  (device={C.MODEL_DEVICE})")
        _model = YOLO(C.MODEL_PATH)
    return _model


class ModelFinder:
    """프레임 → 각도/횡위치. LineFinder 와 같은 출력."""

    def __init__(self):
        self.hist = collections.deque(maxlen=max(1, C.SMOOTH_N))
        self.xhist = collections.deque(maxlen=max(1, C.SMOOTH_N))

    def reset(self):
        self.hist.clear()
        self.xhist.clear()

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        work, src = frame_bgr, None
        x0, x1 = int(w * C.ROI_X0), int(w * C.ROI_X1)
        y0, y1 = int(h * C.ROI_Y0), int(h * C.ROI_Y1)
        out = {"found": False, "angle": None, "angle_raw": None,
               "x_bot": None, "npix": 0, "nlines": 0, "reason": "",
               "work": work, "src": src, "seg": None,
               "roi_box": (x0, y0, x1, y1), "mask": None}

        r = _load().predict(work, imgsz=C.MODEL_IMGSZ, conf=C.MODEL_CONF,
                            device=C.MODEL_DEVICE, verbose=False)[0]
        if r.masks is None or len(r.masks) == 0:
            out["reason"] = "모델이 선을 못 찾음"
            return out

        # 폴리곤을 채워 마스크로. 고전 CV 의 이진 마스크와 같은 형태로 맞춘다.
        full = np.zeros((h, w), np.uint8)
        for poly in r.masks.xy:
            if len(poly) >= 3:
                cv2.fillPoly(full, [poly.astype(np.int32)], 255)

        # ROI 로 자른다 — 화면 밖/반대편 선을 배제하는 건 CV 모드와 동일하게.
        roi_mask = full[y0:y1, x0:x1]
        if roi_mask.size == 0 or not roi_mask.any():
            out["reason"] = "검출은 됐으나 ROI 밖"
            return out
        out["mask"] = roi_mask

        # 여기서부터는 CV 모드와 완전히 같은 처리
        cands = _split_lines(roi_mask, y1 - y0)
        out["nlines"] = len(cands)
        if not cands:
            out["reason"] = "직선으로 맞출 만한 덩어리 없음"
            return out

        angle, x_bot, npix, seg = max(cands, key=lambda t: t[1])  # 가장 가까운 선
        out["npix"] = npix
        out["seg"] = (((seg[0][0] + x0), (seg[0][1] + y0)),
                      ((seg[1][0] + x0), (seg[1][1] + y0)))
        out["angle_raw"] = angle

        self.hist.append(angle)
        self.xhist.append(x_bot + x0)
        out["angle"] = float(np.median(self.hist))
        out["x_bot"] = float(np.median(self.xhist))
        out["found"] = True
        return out


def make_finder():
    return ModelFinder()
