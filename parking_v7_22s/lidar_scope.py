"""
lidar_scope.py -- 라이다가 "각 각도에서 무엇을 보고 있는지" 확인하는 점검 도구.

목적 3가지
  1) MOUNT_OFFSET_DEG 실측: 후진 진행 방향이 라이다 원시각 몇 도인지 눈으로 확인
  2) BLOCKED_SECTORS_RAW 실측: 차체에 가려 항상 비거나 근거리 반사만 나오는 각도 구간
  3) 일반 디버깅: 지금 이 각도에 벽/차가 몇 mm에 있나

화면 2분할
  좌: 극좌표 스캐터 (위에서 본 그림). 원시각 기준, 30도마다 눈금.
  우: 각도(0~360) → 거리(mm) 라인. "각도별로 뭐가 있는지"를 바로 읽는 패널.

필터는 걸지 않고 **원시 스캔 전부**를 그린다. 다만 config 임계값(MIN_VALID_DIST /
MAX_DISTANCE / BLOCKED_SECTORS_RAW)을 배경 음영으로 겹쳐 그려서, 지금 설정이 어떤 점을
버리고 있는지 함께 보이게 했다. (perception.filter_points 는 버린 뒤라 못 봄)

실행
  python3 lidar_scope.py                 # 라이브
  python3 lidar_scope.py --replay a.pkl  # 기록 재생 (라이다 없이)
  python3 lidar_scope.py --text          # matplotlib 없이 터미널 표
  python3 lidar_scope.py --bin 5         # 각도 구간 폭(도)
종료(Ctrl-C 또는 창 닫기) 시 각도 구간별 통계 요약을 출력한다 → 캘리브레이션용.
"""

import argparse
import math
import sys
import time
from collections import defaultdict

import numpy as np

import config as C
import frames as F

try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as _fm
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# 한글 폰트 (없으면 라벨을 영문으로 폴백 — 네모박스 방지)
_KR_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic",
                  "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR",
                  "Arial Unicode MS"]


def _setup_font():
    """한글 렌더 가능한 폰트를 rcParams 에 지정. → 사용 가능 여부."""
    have = {f.name for f in _fm.fontManager.ttflist}
    for name in _KR_CANDIDATES:
        if name in have:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


_L_KR = {
    "win": "LiDAR Scope — 각도별 관측",
    "polar": "원시각 기준 상면도 (0°=라이다 기준방향)",
    "line": "각도별 거리 — 점=원시, 계단=구간 중앙값",
    "xlab": "라이다 원시각 (deg)",
    "ylab": "거리 (mm)",
    "fwd": " +y(후진)",
    "none": "관측 없음",
    "stat": "pts {n}  유효 {v}  빈구간 {e}/{nb}",
    "near": "최근접 {d:.0f}mm @ {a:.1f}°",
}
_L_EN = {
    "win": "LiDAR Scope - per-angle view",
    "polar": "top view, raw angle (0deg = lidar ref)",
    "line": "distance by angle - dots=raw, steps=bin median",
    "xlab": "raw lidar angle (deg)",
    "ylab": "distance (mm)",
    "fwd": " +y(reverse)",
    "none": "no return",
    "stat": "pts {n}  valid {v}  empty {e}/{nb}",
    "near": "nearest {d:.0f}mm @ {a:.1f}deg",
}


# ── 스캔 → 정렬된 배열 (필터 없음, 원시) ───────────────────────
def scan_array(scan):
    """→ np.array (N,3): [angle 0~360, dist mm, quality]. 각도순."""
    if not scan:
        return np.empty((0, 3))
    a = np.array([(ang % 360.0, dist, q) for q, ang, dist in scan], dtype=float)
    return a[np.argsort(a[:, 0])]


def bin_stats(arr, bin_deg):
    """각도 구간별 통계. → dict[bin_index] = (n, dmin, dmed, dmax)."""
    out = {}
    if len(arr) == 0:
        return out
    idx = (arr[:, 0] // bin_deg).astype(int)
    for b in np.unique(idx):
        d = arr[idx == b, 1]
        d = d[d > 0]
        if len(d) == 0:
            continue
        out[int(b)] = (len(d), float(d.min()), float(np.median(d)), float(d.max()))
    return out


def classify(dmin):
    """이 구간이 지금 config 기준으로 어떻게 취급되는지."""
    if dmin < C.MIN_VALID_DIST:
        return "차체?"          # 근거리 반사 → filter_points 가 버림
    if dmin > C.MAX_DISTANCE:
        return "원거리"         # 역시 버림
    return "유효"


def in_blocked(ang):
    """차체 가림 구간(원시각)인지. 래핑 처리는 frames.in_sectors 하나만 쓴다."""
    return bool(C.BLOCKED_SECTORS_RAW) and F.in_sectors(
        ang, C.BLOCKED_SECTORS_RAW)


# ── 누적 통계 (종료 시 캘리브레이션 요약) ──────────────────────
class Accumulator:
    def __init__(self, bin_deg):
        self.bin_deg = bin_deg
        self.frames = 0
        self.hit = defaultdict(int)      # 구간에 유효점(MIN_VALID_DIST 이상)이 있던 프레임
        self.near = defaultdict(int)     # 근거리 반사만 있던 프레임
        self.dmin = defaultdict(lambda: 1e9)
        self.dsum = defaultdict(float)
        self.dcnt = defaultdict(int)

    def add(self, arr):
        self.frames += 1
        st = bin_stats(arr, self.bin_deg)
        for b, (n, dmn, dmed, dmx) in st.items():
            if dmn < C.MIN_VALID_DIST:
                self.near[b] += 1
            if dmx >= C.MIN_VALID_DIST:
                self.hit[b] += 1
                self.dmin[b] = min(self.dmin[b], dmn)
                self.dsum[b] += dmed
                self.dcnt[b] += 1

    def report(self):
        if self.frames == 0:
            print("[scope] 수집된 프레임 없음")
            return
        nb = int(360 // self.bin_deg)
        print(f"\n=== 각도 구간별 요약 ({self.frames} 프레임, {self.bin_deg}도 단위) ===")
        print(" 각도구간   유효율   근거리율   최소mm   평균mm   판정")
        dead = []
        for b in range(nb):
            lo, hi = b * self.bin_deg, (b + 1) * self.bin_deg
            hr = self.hit[b] / self.frames
            nr = self.near[b] / self.frames
            dmn = self.dmin[b] if self.dcnt[b] else float("nan")
            avg = self.dsum[b] / self.dcnt[b] if self.dcnt[b] else float("nan")
            # 항상 비거나 근거리 반사만 = 차체 가림 후보
            verdict = ""
            if hr < 0.2:
                verdict = "★가림후보★"
                dead.append((lo, hi))
            elif nr > 0.5:
                verdict = "근거리반사"
            print(f" {lo:3.0f}-{hi:3.0f}   {hr*100:5.1f}%   {nr*100:6.1f}%  "
                  f"{dmn:7.0f}  {avg:7.0f}   {verdict}")
        if dead:
            merged = []
            for lo, hi in dead:
                if merged and merged[-1][1] == lo:
                    merged[-1][1] = hi
                else:
                    merged.append([lo, hi])
            print("\nBLOCKED_SECTORS_RAW 후보 (config.py 에 기입 — 원시각):")
            print("  BLOCKED_SECTORS_RAW = [" +
                  ", ".join(f"({lo:.0f}, {hi:.0f})" for lo, hi in merged) + "]")
        print("\n※ MOUNT_OFFSET_DEG: 차 뒤(후진 진행 방향)에 물체를 두고, 그 물체가"
              "\n  보이는 원시각을 A라 하면  MOUNT_OFFSET_DEG = -A  로 넣으면 됨.")
        print(F.describe())


# ── 텍스트 모드 ────────────────────────────────────────────────
def run_text(reader, acc, bin_deg, max_mm, hz):
    nb = int(360 // bin_deg)
    period = 1.0 / hz
    while True:
        f = reader.get_latest()
        if f is None:
            time.sleep(0.05)
            continue
        arr = scan_array(f[1])
        acc.add(arr)
        st = bin_stats(arr, bin_deg)
        lines = [f"\n--- {time.strftime('%H:%M:%S')}  pts={len(arr)}  "
                 f"(bar: 0~{max_mm}mm, 좌=가까움) ---"]
        for b in range(nb):
            lo = b * bin_deg
            mark = "B" if in_blocked(lo) else " "
            if b not in st:
                lines.append(f"{mark}{lo:3.0f}도 |{'':30s}|      -   (없음)")
                continue
            n, dmn, dmed, dmx = st[b]
            fill = int(30 * min(dmed, max_mm) / max_mm)
            bar = "#" * max(1, fill)
            lines.append(f"{mark}{lo:3.0f}도 |{bar:<30s}| {dmed:6.0f}mm "
                         f"n={n:3d} min={dmn:5.0f} {classify(dmn)}")
        print("\n".join(lines))
        time.sleep(period)


# ── 그래프 모드 ────────────────────────────────────────────────
def run_plot(reader, acc, bin_deg, max_mm, hz):
    plt.style.use("dark_background")
    L = _L_KR if _setup_font() else _L_EN
    fig = plt.figure(figsize=(13, 6.5))
    axp = fig.add_subplot(1, 2, 1, projection="polar")
    axl = fig.add_subplot(1, 2, 2)
    if fig.canvas.manager is not None:      # Agg 등 헤드리스 백엔드 가드
        fig.canvas.manager.set_window_title(L["win"])

    # 좌: 극좌표. 축 규약·방향 라벨·섹터 음영은 visualize 와 **같은 함수**를 쓴다
    # → 미션 화면과 이 화면에서 같은 물체가 같은 각도에 보인다.
    from visualize import decorate_polar, draw_sectors
    decorate_polar(axp, max_mm)
    axp.set_title(L["polar"], fontsize=10, pad=22)
    sc = axp.scatter([], [], s=6, c=[], cmap="viridis", vmin=0, vmax=max_mm)
    # FSM 이 쓰는 섹터를 흐리게 전부 표시 (손으로 차를 밀며 어느 섹터에 들어가는지 확인)
    draw_sectors(axp, list(F.SECTORS), max_mm)

    fwd = float(F.to_raw(F.REVERSE_CAR_DEG))   # 후진 방향 원시각 (우 패널용)

    # 우: 각도 → 거리
    axl.set_xlim(0, 360)
    axl.set_ylim(0, max_mm)
    axl.set_xticks(range(0, 361, 30))
    axl.set_xlabel(L["xlab"])
    axl.set_ylabel(L["ylab"])
    axl.set_title(L["line"], fontsize=10)
    axl.grid(alpha=0.15)
    axl.axhline(C.MIN_VALID_DIST, color="#ff5252", ls="--", lw=1)
    axl.text(2, C.MIN_VALID_DIST + 30, f"MIN_VALID_DIST {C.MIN_VALID_DIST}",
             color="#ff5252", fontsize=8)
    axl.axhline(C.MAX_DISTANCE, color="#666", ls=":", lw=1)
    # 네 방향 표시 (원시각 기준) — 어느 각도가 어느 방향인지 바로 읽히게
    for raw, text in F.direction_marks():
        axl.axvline(raw, color="#4fc3f7", lw=1.2, alpha=0.7)
        axl.text(raw + 3, max_mm * 0.93, text.split(" ")[0], color="#4fc3f7",
                 fontsize=8)
    for lo, hi in C.BLOCKED_SECTORS_RAW:
        axl.axvspan(lo, hi, color="#555555", alpha=0.35)
    # FSM 섹터 (0도 걸치는 건 두 조각으로)
    for name, sec in F.SECTORS.items():
        lo, hi = F.to_raw_sector(sec)
        spans = [(lo, hi)] if lo <= hi else [(lo, 360.0), (0.0, hi)]
        for a, b in spans:
            axl.axvspan(a, b, color=F.SECTOR_COLORS[name], alpha=0.07)
    raw, = axl.plot([], [], ".", color="#81c784", ms=3, alpha=0.7)
    med, = axl.plot([], [], drawstyle="steps-post", color="#ffee58", lw=1.5)
    # monospace 계열엔 한글 글리프가 없어 네모박스가 뜬다 → 한글일 땐 본문 폰트 사용
    info = axl.text(0.02, 0.97, "", transform=axl.transAxes, va="top",
                    color="#eee", fontsize=9,
                    family=("monospace" if L is _L_EN else None))

    # 마우스로 각도 읽기
    cursor = {"txt": axl.text(0, 0, "", color="#fff", fontsize=9,
                              bbox=dict(fc="#222", ec="#888", alpha=0.9))}
    state = {"arr": np.empty((0, 3))}

    def on_move(ev):
        if ev.inaxes is not axl or ev.xdata is None:
            cursor["txt"].set_text("")
            return
        a = ev.xdata % 360
        arr = state["arr"]
        t = cursor["txt"]
        if len(arr):
            near = arr[np.abs(((arr[:, 0] - a + 180) % 360) - 180) <= bin_deg / 2]
            if len(near):
                t.set_position((a, ev.ydata))
                t.set_text(f"{a:.0f}°  d={np.median(near[:, 1]):.0f}mm "
                           f"(n={len(near)})")
                return
        t.set_position((a, ev.ydata))
        t.set_text(f"{a:.0f}°  ({L['none']})")

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, wspace=0.22)   # 극좌표 제목이 0° 라벨/상단에 안 걸리게
    plt.ion()
    plt.show(block=False)

    period = 1.0 / hz
    nb = int(360 // bin_deg)
    while plt.get_fignums():
        f = reader.get_latest()
        if f is None:
            plt.pause(0.05)
            continue
        arr = scan_array(f[1])
        state["arr"] = arr
        acc.add(arr)

        if len(arr):
            sc.set_offsets(np.c_[np.radians(arr[:, 0]), arr[:, 1]])
            sc.set_array(arr[:, 1])
            raw.set_data(arr[:, 0], arr[:, 1])
            st = bin_stats(arr, bin_deg)
            xs = [b * bin_deg for b in range(nb)]
            ys = [st[b][2] if b in st else np.nan for b in range(nb)]
            med.set_data(xs, ys)
            valid = arr[(arr[:, 1] >= C.MIN_VALID_DIST) &
                        (arr[:, 1] <= C.MAX_DISTANCE)]
            nearest = arr[arr[:, 1] > 0]
            n_txt = ""
            if len(nearest):
                i = int(np.argmin(nearest[:, 1]))
                n_txt = L["near"].format(d=nearest[i, 1], a=nearest[i, 0])
            info.set_text(L["stat"].format(n=len(arr), v=len(valid),
                                           e=nb - len(st), nb=nb) + "\n" + n_txt)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(period)


# ── 엔트리 ─────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="라이다 각도별 관측 확인")
    ap.add_argument("--replay", metavar="PKL", help="기록 파일 재생 (라이다 불필요)")
    ap.add_argument("--record", metavar="PKL", help="보면서 스캔 기록")
    ap.add_argument("--port", default=C.LIDAR_PORT)
    ap.add_argument("--baud", type=int, default=C.LIDAR_BAUD)
    ap.add_argument("--bin", type=float, default=10.0, help="각도 구간 폭(도)")
    ap.add_argument("--max", type=int, default=C.MAX_DISTANCE, help="표시 최대 거리 mm")
    ap.add_argument("--hz", type=float, default=8.0, help="화면 갱신 주기")
    ap.add_argument("--text", action="store_true", help="matplotlib 없이 터미널 표")
    args = ap.parse_args()

    if 360 % args.bin != 0:
        print(f"[scope] --bin 은 360의 약수여야 함 (입력 {args.bin})")
        return 2

    from lidar_reader import LidarReader, ReplayReader
    if args.replay:
        reader = ReplayReader(args.replay, realtime=True, loop=True)
    else:
        reader = LidarReader(args.port, args.baud, record_path=args.record)
    reader.start()

    acc = Accumulator(args.bin)
    try:
        if args.text or not _HAS_MPL:
            if not _HAS_MPL and not args.text:
                print("[scope] matplotlib 없음 → 텍스트 모드")
            run_text(reader, acc, args.bin, args.max, args.hz)
        else:
            run_plot(reader, acc, args.bin, args.max, args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        acc.report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
