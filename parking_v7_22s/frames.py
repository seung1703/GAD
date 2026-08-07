"""
frames.py -- 각도 프레임 단일 진실원천. MOUNT_OFFSET_DEG 하나에서 전부 파생된다.

두 가지 프레임만 존재한다:

  raw : 라이다 원시각. lidar_scope.py 화면에 찍히는 각도이고, 사람이 대화할 때
        쓰는 기준이다.
  car : 차체 기준 방위각 = atan2(x, y).  후진 0°, 우측 90°, 전진 180°, 좌측 270°.
        라이다를 재장착해도 "차체 우측"은 여전히 90°다.

        car = (D·raw + M) % 360      raw = (D·(car - M)) % 360
            M = MOUNT_OFFSET_DEG (회전),  D = LIDAR_ANGLE_DIR (±1, 각도 증가 방향)

        ★ D 가 필요한 이유 ★ M 은 회전이라 좌우를 못 바꾼다. 라이다가 각도를
        반대 방향으로 세면 좌우가 뒤집혀 보이는데, 이건 회전이 아니라 거울 반전이라
        M 을 아무리 돌려도 안 고쳐진다. 전진·후진은 맞는데 좌우만 반대면 D 를 뒤집는다.

★ 섹터는 전부 car 로 정의하고, 표시하거나 원시각과 비교할 때만 to_raw_sector()
  로 되돌린다. 원시각 상수를 코드에 적어두면 재장착 때 같이 안 고쳐져서 썩는다
  (실제로 REAR_GATE_SECTORS 가 후방 대신 좌측을 가리는 버그가 있었다). ★

섹터가 0°를 걸치는지는 프레임에 따라 바뀐다:
  우측 ±20°  → car (70, 110)   래핑 없음  /  raw (340, 20)   0° 걸침
  후진 ±35°  → car (325, 35)   0° 걸침    /  raw (235, 305)  래핑 없음
그래서 `lo <= a <= hi` 같은 단순 비교는 절반의 경우 틀린다. in_sectors() 를 쓴다.
"""

import numpy as np

import config as C

# ── 차체 기준 방위 (절대 변하지 않음) ─────────────────────────
REVERSE_CAR_DEG = 0.0
RIGHT_CAR_DEG = 90.0
FORWARD_CAR_DEG = 180.0
LEFT_CAR_DEG = 270.0

_DIR_NAMES = ((RIGHT_CAR_DEG, "우측"), (FORWARD_CAR_DEG, "전진"),
              (LEFT_CAR_DEG, "좌측"), (REVERSE_CAR_DEG, "후진"))


# ── 변환 ─────────────────────────────────────────────────────
def to_car(raw_deg):
    """라이다 원시각 → 차체 기준 방위각(0~360). 스칼라/ndarray 모두."""
    return (C.LIDAR_ANGLE_DIR * np.asarray(raw_deg, dtype=float)
            + C.MOUNT_OFFSET_DEG) % 360.0


def to_raw(car_deg):
    """차체 기준 방위각 → 라이다 원시각(0~360). 스칼라/ndarray 모두.
    D=±1 이라 역변환도 같은 형태다 (D⁻¹ = D)."""
    return (C.LIDAR_ANGLE_DIR
            * (np.asarray(car_deg, dtype=float) - C.MOUNT_OFFSET_DEG)) % 360.0


def xy(raw_deg, dist_mm):
    """원시각+거리 → 차체 좌표 (x=우측+, y=후진+). 전진은 -y.
    to_car 를 거치므로 각도 규약과 절대 어긋나지 않는다."""
    a = np.radians(to_car(raw_deg))
    d = np.asarray(dist_mm, dtype=float)
    return d * np.sin(a), d * np.cos(a)


def car_deg_of_xy(x, y):
    """차체 좌표 → 차체 방위각(0~360). atan2(x, y) 규약."""
    return np.degrees(np.arctan2(x, y)) % 360.0


# ── 섹터 ─────────────────────────────────────────────────────
def sector(center_car_deg, half_deg):
    """차체각 중심 ±half → (lo, hi) 차체각 섹터. 0° 걸침은 lo>hi 로 표현된다."""
    return ((center_car_deg - half_deg) % 360.0,
            (center_car_deg + half_deg) % 360.0)


def to_raw_sector(car_sector):
    """차체각 섹터 → 원시각 섹터.

    in_sectors 는 (lo, hi) 를 "lo 에서 **증가 방향**으로 hi 까지의 호"로 해석한다.
    회전(D=+1)은 그 방향을 보존하지만, **반전(D=-1)은 호의 방향을 뒤집으므로
    lo/hi 를 서로 바꿔야** 같은 물리적 부채꼴이 된다. 안 바꾸면 섹터가
    "여집합"이 되어 정반대 영역을 가리킨다.
    """
    lo, hi = car_sector
    if C.LIDAR_ANGLE_DIR < 0:
        lo, hi = hi, lo
    return (float(to_raw(lo)), float(to_raw(hi)))


def to_raw_sectors(car_sectors):
    return [to_raw_sector(s) for s in car_sectors]


def in_sectors(ang, sectors):
    """ang 이 sectors 중 하나에 들어가는지. lo>hi(0° 걸침)를 올바르게 처리한다.
    스칼라면 bool, ndarray 면 bool 마스크. ang 과 sectors 는 같은 프레임이어야 한다."""
    a = np.asarray(ang, dtype=float) % 360.0
    hit = np.zeros(a.shape, dtype=bool)
    for lo, hi in sectors:
        if lo <= hi:
            hit |= (a >= lo) & (a <= hi)
        else:                       # 0° 걸침
            hit |= (a >= lo) | (a <= hi)
    return hit if a.ndim else bool(hit)


# ── FSM·뷰가 공유하는 섹터 (config 는 반폭만 갖는다) ──────────
DETECT_RIGHT = sector(RIGHT_CAR_DEG, C.DETECT_HALF_DEG)
FLANK_RIGHT = sector(RIGHT_CAR_DEG, C.FLANK_HALF_DEG)
FLANK_LEFT = sector(LEFT_CAR_DEG, C.FLANK_HALF_DEG)
REAR_FAN = sector(REVERSE_CAR_DEG, C.REAR_FAN_HALF_DEG)

# 스윙 중 첫 주차차량을 따라갈 범위: **후진(0°) ~ 우측(90°) 사분면**.
# 좌로 꺾으면 그 차의 방위가 우측 90° 에서 후진 0° 쪽으로 밀리므로
# (90° - 회전량), 이 사분면이 스윙 내내 그 차를 담는다.
# 우측 섹터(±35°)만 쓰면 35°쯤에서 벗어나 관측이 끊겼다.
_SWING_MARGIN_DEG = 15.0     # 양 끝 여유 (획득 시점·오버슛 대비)
SWING_TRACK = ((REVERSE_CAR_DEG - _SWING_MARGIN_DEG) % 360.0,
               (RIGHT_CAR_DEG + _SWING_MARGIN_DEG) % 360.0)

SECTORS = {"detect_right": DETECT_RIGHT, "swing_track": SWING_TRACK,
           "flank_right": FLANK_RIGHT, "flank_left": FLANK_LEFT,
           "rear_fan": REAR_FAN}

# 화면 색 (visualize / lidar_scope 공용)
SECTOR_COLORS = {"detect_right": "#ffee58", "swing_track": "#ba68c8",
                 "flank_right": "#4fc3f7", "flank_left": "#81c784",
                 "rear_fan": "#ff5252"}


# ── 자기점검 / 로그 증거 ──────────────────────────────────────
def direction_marks():
    """[(원시각, "우측 (raw 0°)"), ...] — 화면 방향 라벨용."""
    out = []
    for car, name in _DIR_NAMES:
        raw = float(to_raw(car))
        out.append((raw, f"{name} (raw {raw:.0f}°)"))
    return out


def describe():
    """현재 장착 기준의 방향·섹터 표를 문자열로. 미션 시작 시 로그에 남긴다 —
    나중에 로그만 보고 '그때 어느 섹터를 봤나'를 알 수 있게."""
    lines = [f"[frames] MOUNT_OFFSET_DEG={C.MOUNT_OFFSET_DEG:+.0f}"
             f"  LIDAR_ANGLE_DIR={C.LIDAR_ANGLE_DIR:+d}"
             f"   (car = {C.LIDAR_ANGLE_DIR:+d}·raw {C.MOUNT_OFFSET_DEG:+.0f})"]
    lines.append("  방향:  " + "   ".join(
        f"{name} car{car:.0f}=raw{to_raw(car):.0f}" for car, name in _DIR_NAMES))
    for name, sec in SECTORS.items():
        raw = to_raw_sector(sec)
        wrap = " (raw 0° 걸침)" if raw[0] > raw[1] else ""
        lines.append(f"  섹터 {name:12s} car({sec[0]:5.0f},{sec[1]:5.0f})"
                     f" → raw({raw[0]:5.0f},{raw[1]:5.0f}){wrap}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
