"""
perception.py -- 라이다 인지 파이프라인.

입력: scan = [(quality, angle_deg, distance_mm), ...]  (RPLidar 원시, angle=원시각)
좌표/각도 변환은 전부 frames.py 가 담당한다 (단일 진실원천).

순수 numpy — 라이다/카메라 없이 리플레이·유닛테스트로 검증 가능.

클러스터 표현: (M, 4) = [원시각, x, y, 거리]
  filter_points 의 행을 그대로 잘라 담는다. 각도·거리를 다시 계산할 필요가 없어서
  극좌표 화면(visualize/lidar_scope)과 섹터 판정이 같은 숫자를 쓴다.
  xy 만 필요하면 cxy(c) 를 쓴다.
"""

import numpy as np

import config as C
import frames as F

# ── 알고리즘 내부 상수 (튜닝 대상이 아니라서 config 에 두지 않는다) ──
MIN_QUALITY = 10           # RPLidar quality 미만은 버림


# ── 좌표 변환 + 필터 ──────────────────────────────────────────
def filter_points(scan):
    """→ np.array (N,4): [원시각(0~360), x, y, 거리]. 원시각순 정렬.

    버리는 것: 거리 0/초과, 저품질, 차체 반사(MIN_VALID_DIST 미만),
    그리고 차체가 영구히 가리는 원시각 구간(BLOCKED_SECTORS_RAW).
    ★차체 가림은 좌표 변환 전에 원시각으로 판정한다 — 라이다 장착에 붙는 성질★
    """
    out = []
    blocked = C.BLOCKED_SECTORS_RAW
    for q, ang, dist in scan:
        if dist <= 0 or dist > C.MAX_DISTANCE:
            continue
        if q < MIN_QUALITY or dist < C.MIN_VALID_DIST:
            continue
        if blocked and F.in_sectors(ang, blocked):
            continue
        x, y = F.xy(ang, dist)
        out.append((ang % 360.0, float(x), float(y), float(dist)))
    if not out:
        return np.empty((0, 4))
    arr = np.array(out, dtype=float)
    return arr[np.argsort(arr[:, 0])]


# ── 클러스터링 (인접 포인트 xy거리 기준) ───────────────────────
def cluster_points(pts):
    """pts: filter_points 출력. → list[np.array(M,4)] (행 구성은 pts 와 동일)."""
    if len(pts) == 0:
        return []
    xy_ = pts[:, 1:3]
    clusters, start = [], 0
    for i in range(1, len(xy_)):
        if np.hypot(*(xy_[i] - xy_[i - 1])) > C.CLUSTER_GAP_THRESH:
            clusters.append(pts[start:i])
            start = i
    clusters.append(pts[start:])
    return clusters


def cxy(c):
    """클러스터의 xy 부분 (M,2)."""
    return c[:, 1:3]


# ── 벽면 각도 ─────────────────────────────────────────────────
def wall_car_deg(c):
    """클러스터가 이루는 직선(벽면)의 방향을 **차체기준 0~180°** 로. 점<2면 None.

    구 wall_angle 은 PCA 주축의 atan2(dy,dx) 를 그대로 돌려줬는데, SVD 주축의
    부호가 임의여서 같은 벽 하나가 점 순서에 따라 +10° / -170° 로 나왔다.
    직선은 180° 주기이므로 % 180 으로 그 모호성을 원천 제거한다.
    """
    P = cxy(c) if c.ndim == 2 and c.shape[1] == 4 else np.asarray(c, float)
    if len(P) < 2:
        return None
    _, _, vt = np.linalg.svd(P - P.mean(0))
    dx, dy = vt[0]
    # atan2(dx, dy): 차체 방위각 규약(atan2(x, y))과 축을 맞춘다
    return float(np.degrees(np.arctan2(dx, dy)) % 180.0)


def angle_between(a_deg, b_deg):
    """두 직선 방향(180° 주기) 사이 예각차 0~90. None 이 섞이면 None."""
    if a_deg is None or b_deg is None:
        return None
    return float(abs((a_deg - b_deg + 90.0) % 180.0 - 90.0))


def wall_dev(c, ref_car_deg):
    """벽면이 기준 방향과 이루는 예각차 0~90. 180° 주기 안전."""
    return angle_between(wall_car_deg(c), ref_car_deg)


# ── 섹터 질의 (트리거·양옆 소멸 판정 공용) ────────────────────
def sector_stats(clusters, sectors, max_dist=None, min_dist=None):
    """sectors(차체기준) 안, [min_dist, max_dist] 안 점들의 통계.
    → dict(n=점수, dmin=최소거리|None, n_clusters=해당 클러스터 수)"""
    n = 0
    dmin = None
    n_cl = 0
    for c in clusters:
        if len(c) == 0:
            continue
        m = F.in_sectors(F.to_car(c[:, 0]), sectors)
        if max_dist is not None:
            m &= c[:, 3] <= max_dist
        if min_dist is not None:
            m &= c[:, 3] >= min_dist
        k = int(m.sum())
        if k == 0:
            continue
        n += k
        n_cl += 1
        d = float(c[m, 3].min())
        dmin = d if dmin is None else min(dmin, d)
    return {"n": n, "dmin": dmin, "n_clusters": n_cl}


def largest_near_cluster(clusters, sectors, max_dist=None):
    """sectors 안에 점이 있는 클러스터 중 (섹터 안) 점이 가장 많은 것. 없으면 None.
    벽면각을 프레임마다 같은 물체에서 재려고 쓴다."""
    best, best_n = None, 0
    for c in clusters:
        if len(c) == 0:
            continue
        m = F.in_sectors(F.to_car(c[:, 0]), sectors)
        if max_dist is not None:
            m &= c[:, 3] <= max_dist
        k = int(m.sum())
        if k > best_n:
            best, best_n = c, k
    return best


def perceive(scan):
    """한 스캔 → 인지 결과 묶음 (FSM/시각화 공용)."""
    pts = filter_points(scan)
    return {"points": pts, "clusters": cluster_points(pts)}
