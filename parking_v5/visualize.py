"""
visualize.py -- 미션 실시간 시각화 (라이다 원시각 극좌표).

★ 이 화면은 lidar_scope.py 와 **완전히 같은 각도 규약**을 쓴다 ★
  0°가 위, 시계방향(theta_zero_location="N", theta_direction=-1).
  같은 주차차량이 두 화면에서 같은 각도에 보인다 — 예전엔 이 화면이 차체 기준
  직교좌표였고 lidar_scope 는 원시각 극좌표여서, 보이는 각도와 말하는 각도가
  달라 대화와 디버깅이 계속 엇갈렸다.

방향 라벨과 섹터 음영은 전부 frames.py 가 MOUNT_OFFSET_DEG 에서 계산한다.
라이다를 재장착하면 라벨이 자동으로 따라 돈다 — 화면에 각도를 하드코딩하지 않는다.

FSM 이 그 순간 **실제로 판정에 쓰는** 섹터만 진하게 칠한다(`sectors` 인자).
목록은 ParkingFSM.ACTIVE_SECTORS 에서 오므로 "보이는 것 == 인식에 쓰는 것"이
구조적으로 보장된다.

matplotlib 미설치여도 import 가능하도록 가드 (헤드리스 실행 대비).
"""

try:
    import matplotlib
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

import numpy as np

import config as C
import frames as F

_CLUSTER_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ba68c8",
                   "#f06292", "#4db6ac", "#fff176", "#a1887f"]

_KR_FONTS = ("AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic",
             "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR",
             "Arial Unicode MS")


def setup_font():
    """한글 렌더 가능한 폰트 지정. → 사용 가능 여부."""
    if not _HAS_MPL:
        return False
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for name in _KR_FONTS:
        if name in have:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


# ── 공용 극좌표 장식 (lidar_scope.py 와 공유) ──────────────────
def decorate_polar(ax, max_mm, label_dirs=True):
    """극좌표축을 라이다 원시각 규약으로 세팅하고 방향 라벨을 붙인다.

    0°=위, 시계방향 — perception 의 x=d·sin, y=d·cos 규약과 일치한다.
    방향 라벨(우측/전진/좌측/후진)은 MOUNT_OFFSET_DEG 에서 계산된다.
    """
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(range(0, 360, 30), [f"{a}°" for a in range(0, 360, 30)])
    ax.set_ylim(0, max_mm)
    ax.set_rlabel_position(157)      # 반경 눈금이 0°/90° 라벨과 겹치지 않게
    ax.grid(alpha=0.25)
    # config 임계값 링
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(th, [C.MIN_VALID_DIST] * 200, color="#ff5252", lw=1, ls="--")
    ax.plot(th, [min(C.MAX_DISTANCE, max_mm)] * 200, color="#666", lw=1, ls=":")
    if label_dirs:
        for raw, text in F.direction_marks():
            t = np.radians(raw)
            ax.plot([t, t], [0, max_mm], color="#4fc3f7", lw=1.2, alpha=0.55)
            # 눈금 라벨과 겹치지 않게 살짝 안쪽에
            ax.text(t, max_mm * 0.86, text, color="#4fc3f7", fontsize=8,
                    ha="center", va="center",
                    bbox=dict(fc="#101010", ec="none", alpha=0.65, pad=1.5))


def _sector_spans(raw_sector):
    """원시각 섹터 → [(lo, hi), ...] 0° 걸침이면 두 조각으로 쪼갠다."""
    lo, hi = raw_sector
    return [(lo, hi)] if lo <= hi else [(lo, 360.0), (0.0, hi)]


def draw_sectors(ax, names, max_mm, active=(), alpha_on=0.22, alpha_off=0.07):
    """frames.SECTORS 를 원시각으로 되돌려 음영. → 생성된 artist 리스트.

    active 에 든 이름만 진하게 = 지금 FSM 이 실제로 보는 섹터.
    """
    arts = []
    for name in names:
        sec = F.SECTORS.get(name)
        if sec is None:
            continue
        color = F.SECTOR_COLORS.get(name, "#888888")
        a = alpha_on if name in active else alpha_off
        for lo, hi in _sector_spans(F.to_raw_sector(sec)):
            th = np.radians(np.linspace(lo, hi, 40))
            arts.append(ax.fill_between(th, 0, max_mm, color=color, alpha=a))
    # 차체 가림 구간 (원시각으로 직접 지정된 유일한 상수)
    for lo, hi in C.BLOCKED_SECTORS_RAW:
        for a, b in _sector_spans((float(lo), float(hi))):
            th = np.radians(np.linspace(a, b, 30))
            arts.append(ax.fill_between(th, 0, max_mm, color="#555555",
                                        alpha=0.35))
    return arts


class ParkingViz:
    def __init__(self, max_mm=None):
        if not _HAS_MPL:
            raise ImportError("matplotlib 필요: pip install matplotlib")
        self.max_mm = C.MAX_DISTANCE if max_mm is None else max_mm
        plt.style.use("dark_background")
        setup_font()
        self.fig = plt.figure(figsize=(7.5, 7.8))
        self.ax = self.fig.add_subplot(1, 1, 1, projection="polar")
        self.ax.set_title("주차 미션 — 라이다 원시각 (0°=위, 시계방향)",
                          fontsize=10, pad=24)
        decorate_polar(self.ax, self.max_mm)
        # 자차(원점)
        self.ax.plot(0, 0, "o", color="#ffffff", ms=7)
        self.txt = self.fig.text(0.02, 0.975, "", va="top", ha="left",
                                 color="#eee", fontsize=9)
        self._artists = []
        plt.ion()
        plt.show(block=False)

    def update(self, percep, state="", status="", sectors=()):
        for a in self._artists:
            a.remove()
        self._artists = []
        arts = []

        # 섹터 음영 (활성 섹터만 진하게) — 점보다 먼저 깔아야 가리지 않는다
        arts += draw_sectors(self.ax, list(F.SECTORS), self.max_mm,
                             active=tuple(sectors))

        # 클러스터: 행이 [원시각, x, y, 거리] 라 극좌표로 바로 그린다
        for i, c in enumerate(percep["clusters"]):
            if len(c) == 0:
                continue
            col = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
            ln, = self.ax.plot(np.radians(c[:, 0]), c[:, 3], ".",
                               color=col, ms=5)
            arts.append(ln)

        act = ",".join(sectors) if sectors else "-"
        self.txt.set_text(f"STATE: {state}    보는 섹터: {act}\n{status}")
        self._artists = arts
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        return tuple(arts) + (self.txt,)

    @staticmethod
    def alive():
        return _HAS_MPL and len(plt.get_fignums()) > 0
