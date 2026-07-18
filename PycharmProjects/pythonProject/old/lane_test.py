import cv2
import numpy as np
import json
import os
import Function_Library as fl

# ===== 설정 =====
CAM_PORT = 0
IMG_W, IMG_H = 1280, 720
HEADING_OFFSET_X = 38
LANE_WIDTH_PX = 280
N_WINDOWS = 9
MINPIX = 50
MIN_PIXELS = 300

# ===== 저장된 파라미터 불러오기 =====
defaults = {
    'white_L': 185, 'white_S_max': 90, 'margin': 80,
    'ROI_TOP_Y': 250, 'ROI_BOT_Y': 650,
    'ROI_TOP_L': 400, 'ROI_TOP_R': 900,
    'ROI_BOT_L': 150, 'ROI_BOT_R': 1150,
}
if os.path.exists('lane_params.json'):
    with open('lane_params.json', 'r') as f:
        saved = json.load(f)
    defaults.update(saved)
    print("저장된 파라미터 불러옴!")
else:
    print("기본 파라미터 사용")

# ===== 카메라 초기화 =====
env = fl.libCAMERA()
ch0, _ = env.initial_setting(cam0port=CAM_PORT, capnum=1)
ch0.set(cv2.CAP_PROP_BUFFERSIZE, 1)

dst = np.float32([
    [100,         0],
    [IMG_W - 100, 0],
    [IMG_W - 100, IMG_H],
    [100,         IMG_H],
])

# ===== 슬라이더 창 =====
cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Settings', 500, 400)
cv2.createTrackbar('white_L',    'Settings', defaults['white_L'],    255, lambda x: None)
cv2.createTrackbar('white_S_max','Settings', defaults['white_S_max'],255, lambda x: None)
cv2.createTrackbar('margin',     'Settings', defaults['margin'],     200, lambda x: None)
cv2.createTrackbar('ROI_TOP_Y',  'Settings', defaults['ROI_TOP_Y'],  720, lambda x: None)
cv2.createTrackbar('ROI_BOT_Y',  'Settings', defaults['ROI_BOT_Y'],  720, lambda x: None)
cv2.createTrackbar('ROI_TOP_L',  'Settings', defaults['ROI_TOP_L'], 1280, lambda x: None)
cv2.createTrackbar('ROI_TOP_R',  'Settings', defaults['ROI_TOP_R'], 1280, lambda x: None)
cv2.createTrackbar('ROI_BOT_L',  'Settings', defaults['ROI_BOT_L'], 1280, lambda x: None)
cv2.createTrackbar('ROI_BOT_R',  'Settings', defaults['ROI_BOT_R'], 1280, lambda x: None)

def make_binary(warped):
    hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
    H, L, S = cv2.split(hls)
    wL = cv2.getTrackbarPos('white_L',     'Settings')
    # 기본 노이즈 제거
    wS = cv2.getTrackbarPos('white_S_max', 'Settings')
    binary = np.zeros_like(L, dtype=np.uint8)
    binary[(L >= wL) & (S <= wS)] = 255
    binary[:, :100]  = 0
    binary[:, -100:] = 0

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ★ 세로형 커널로 가로 방향 얇은 선 제거 ★
    # (2, 15): 가로 2픽셀, 세로 15픽셀 → 세로 성분만 살아남음
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    return binary

def sliding_window(binary):
    h, w = binary.shape
    margin = cv2.getTrackbarPos('margin', 'Settings')
    hist   = np.sum(binary[h//2:, :], axis=0)
    mid    = w // 2
    leftx  = np.argmax(hist[:mid])
    rightx = np.argmax(hist[mid:]) + mid

    nonzero  = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    lx_cur, rx_cur = leftx, rightx
    left_inds, right_inds = [], []
    win_h = h // N_WINDOWS
    out_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for w_idx in range(N_WINDOWS):
        y_lo = h - (w_idx + 1) * win_h
        y_hi = h - w_idx * win_h
        cv2.rectangle(out_img, (lx_cur-margin, y_lo), (lx_cur+margin, y_hi), (0,255,0), 2)
        cv2.rectangle(out_img, (rx_cur-margin, y_lo), (rx_cur+margin, y_hi), (0,255,0), 2)
        gl = ((nonzeroy >= y_lo) & (nonzeroy < y_hi) &
              (nonzerox >= lx_cur-margin) & (nonzerox < lx_cur+margin)).nonzero()[0]
        gr = ((nonzeroy >= y_lo) & (nonzeroy < y_hi) &
              (nonzerox >= rx_cur-margin) & (nonzerox < rx_cur+margin)).nonzero()[0]
        left_inds.append(gl)
        right_inds.append(gr)
        if len(gl) > MINPIX: lx_cur = int(np.mean(nonzerox[gl]))
        if len(gr) > MINPIX: rx_cur = int(np.mean(nonzerox[gr]))

    left_inds  = np.concatenate(left_inds)  if left_inds  else np.array([])
    right_inds = np.concatenate(right_inds) if right_inds else np.array([])

    left_fit = right_fit = None
    if len(left_inds)  >= MIN_PIXELS:
        left_fit  = np.polyfit(nonzeroy[left_inds],  nonzerox[left_inds],  2)
    if len(right_inds) >= MIN_PIXELS:
        right_fit = np.polyfit(nonzeroy[right_inds], nonzerox[right_inds], 2)

    return left_fit, right_fit, out_img

def calc_error(left_fit, right_fit, h, w):
    ref_x     = w / 2.0 + HEADING_OFFSET_X
    bottom_y  = h - 1
    half_lane = LANE_WIDTH_PX / 2.0

    def x_at_y(fit, y):
        return fit[0]*y*y + fit[1]*y + fit[2]

    if left_fit is not None and right_fit is not None:
        center = (x_at_y(left_fit, bottom_y) + x_at_y(right_fit, bottom_y)) / 2.0
    elif left_fit is not None:
        center = x_at_y(left_fit, bottom_y) + half_lane
    elif right_fit is not None:
        center = x_at_y(right_fit, bottom_y) - half_lane
    else:
        return 0.0

    return float(np.clip((center - ref_x) / (w / 2.0), -1.0, 1.0))

def draw_lanes(binary, left_fit, right_fit):
    vis  = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    h, w = binary.shape
    ys   = np.linspace(0, h-1, 80)

    def draw_poly(fit, color):
        pts = [[int(fit[0]*y*y + fit[1]*y + fit[2]), int(y)] for y in ys
               if 0 <= fit[0]*y*y + fit[1]*y + fit[2] < w]
        if len(pts) > 2:
            cv2.polylines(vis, [np.array(pts, dtype=np.int32).reshape(-1,1,2)],
                          False, color, 4)

    if left_fit  is not None: draw_poly(left_fit,  (255, 100, 0))
    if right_fit is not None: draw_poly(right_fit, (0,   165, 255))
    return vis

# ===== 메인 루프 =====
print("실행 중... q 누르면 저장 후 종료")

while True:
    for _ in range(3):
        ch0.grab()
    ret, frame = ch0.retrieve()
    if not ret or frame is None:
        continue

    roi_top_y = cv2.getTrackbarPos('ROI_TOP_Y', 'Settings')
    roi_bot_y = cv2.getTrackbarPos('ROI_BOT_Y', 'Settings')
    roi_top_l = cv2.getTrackbarPos('ROI_TOP_L', 'Settings')
    roi_top_r = cv2.getTrackbarPos('ROI_TOP_R', 'Settings')
    roi_bot_l = cv2.getTrackbarPos('ROI_BOT_L', 'Settings')
    roi_bot_r = cv2.getTrackbarPos('ROI_BOT_R', 'Settings')

    src = np.float32([
        [roi_top_l, roi_top_y],
        [roi_top_r, roi_top_y],
        [roi_bot_r, roi_bot_y],
        [roi_bot_l, roi_bot_y],
    ])
    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(frame, M, (IMG_W, IMG_H))
    binary = make_binary(warped)
    left_fit, right_fit, win_img = sliding_window(binary)
    lateral = calc_error(left_fit, right_fit, IMG_H, IMG_W)
    lane_vis = draw_lanes(binary, left_fit, right_fit)

    orig_vis = frame.copy()
    pts = src.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(orig_vis, [pts], True, (0, 255, 0), 2)

    ref_x = IMG_W // 2 + HEADING_OFFSET_X
    cv2.line(lane_vis, (ref_x, 0), (ref_x, IMG_H), (0, 220, 220), 2)

    cv2.putText(orig_vis, f'lateral: {lateral:+.3f}',
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)
    cv2.putText(lane_vis, f'lateral: {lateral:+.3f}',
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)

    h2, w2 = IMG_H // 2, IMG_W // 2
    debug = np.hstack([
        cv2.resize(orig_vis,  (w2, h2)),
        cv2.resize(lane_vis,  (w2, h2))
    ])

    cv2.imshow('Lane Detection', debug)
    cv2.imshow('Settings', np.zeros((1, 400, 3), dtype=np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        params = {
            'white_L':    cv2.getTrackbarPos('white_L',     'Settings'),
            'white_S_max':cv2.getTrackbarPos('white_S_max', 'Settings'),
            'margin':     cv2.getTrackbarPos('margin',      'Settings'),
            'ROI_TOP_Y':  cv2.getTrackbarPos('ROI_TOP_Y',  'Settings'),
            'ROI_BOT_Y':  cv2.getTrackbarPos('ROI_BOT_Y',  'Settings'),
            'ROI_TOP_L':  cv2.getTrackbarPos('ROI_TOP_L',  'Settings'),
            'ROI_TOP_R':  cv2.getTrackbarPos('ROI_TOP_R',  'Settings'),
            'ROI_BOT_L':  cv2.getTrackbarPos('ROI_BOT_L',  'Settings'),
            'ROI_BOT_R':  cv2.getTrackbarPos('ROI_BOT_R',  'Settings'),
        }
        with open('lane_params.json', 'w') as f:
            json.dump(params, f, indent=2)
        print("파라미터 저장 완료! → lane_params.json")
        break

ch0.release()
cv2.destroyAllWindows()