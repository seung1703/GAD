#!/usr/bin/env python3
"""
record_camera.py -- 원하는 카메라 인덱스로 영상을 찍는다

주행 코드와 완전히 분리된 단독 도구다. ultralytics/torch 를 안 불러오므로
바로 뜬다. 카메라 백엔드와 인덱스 처리는 주행 코드와 같은 것을 쓴다
(`shared/frame_grabber.py`).

  python record_camera.py --list                 # 이 컴퓨터에서 열리는 인덱스 확인
  python record_camera.py --cam 1                # 1번 카메라로 녹화
  python record_camera.py --cam 1 --seconds 30   # 30초만 찍고 자동 종료
  python record_camera.py --cam 1 --size 1280x720 --fps 30
  python record_camera.py --cam 1 --out ~/Desktop/test.avi
  python record_camera.py --cam 1 --no-preview   # 화면 없이 (원격/느린 기기)

키:
  Space   녹화 시작 / 일시정지   (--auto 를 주면 시작하자마자 녹화)
  s       스냅샷 한 장 저장 (녹화와 별개로 .jpg)
  q/ESC   저장하고 종료

기록되는 영상은 **오버레이가 없는 원본 프레임**이다. 화면에 보이는 빨간 테두리와
글자는 미리보기에만 그린다. 학습 데이터로 바로 쓸 수 있다.

파일은 기본적으로 `records/cam<인덱스>_<날짜시각>.avi` 로 저장된다.
"""

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared"))

from frame_grabber import FrameGrabber, list_cameras, open_capture  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW = "record_camera"


def parse_size(text):
    try:
        width, height = str(text).lower().split("x")
        return int(width), int(height)
    except Exception:
        raise argparse.ArgumentTypeError(f"--size 는 640x360 형식이어야 합니다: {text}")


def parse_args():
    parser = argparse.ArgumentParser(description="Record video from a chosen camera index.")
    parser.add_argument("--cam", type=int, default=0, help="카메라 인덱스 (기본 0)")
    parser.add_argument("--list", action="store_true", help="열리는 카메라 인덱스만 확인하고 종료")
    parser.add_argument("--out", help="저장 경로. 생략하면 records/cam<N>_<시각>.avi")
    parser.add_argument("--size", type=parse_size, default=(640, 360), help="해상도. 예: 1280x720")
    parser.add_argument("--fps", type=float, default=30.0, help="카메라와 저장 fps (기본 30)")
    parser.add_argument("--seconds", type=float, help="이 시간만큼 녹화하고 자동 종료")
    parser.add_argument("--codec", default="MJPG", help="저장 코덱 FourCC (기본 MJPG)")
    parser.add_argument("--backend", default="auto", help="카메라 백엔드. auto | CAP_AVFOUNDATION | CAP_DSHOW")
    parser.add_argument("--auto", action="store_true", help="Space 를 안 눌러도 시작하자마자 녹화")
    parser.add_argument("--no-preview", action="store_true", help="미리보기 창 없이 녹화")
    return parser.parse_args()


def default_out(cam_index):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    directory = os.path.join(ROOT_DIR, "records")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"cam{cam_index}_{stamp}.avi")


def draw_preview(frame, recording, elapsed, frames, fps_now, snapshots):
    view = frame.copy()
    height, width = view.shape[:2]
    if recording:
        cv2.rectangle(view, (0, 0), (width - 1, height - 1), (0, 0, 255), 4)
    label = (
        f"{'REC' if recording else 'PAUSED'}  {elapsed:5.1f}s  "
        f"{frames} frames  {fps_now:4.1f} fps"
    )
    cv2.rectangle(view, (0, 0), (width, 26), (0, 0, 0), -1)
    cv2.putText(view, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255) if recording else (200, 200, 200), 1)
    hint = "space: rec/pause   s: snapshot   q: quit"
    if snapshots:
        hint += f"   snaps:{snapshots}"
    cv2.putText(view, hint, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1)
    return view


def main():
    args = parse_args()
    if args.list:
        list_cameras(args.backend)
        return

    width, height = args.size
    camera, backend = open_capture(
        args.cam, width, height, args.backend, args.fps, "MJPG", 1
    )
    if not camera.isOpened():
        print(f"[record] 카메라 {args.cam} 를 열 수 없습니다. --list 로 확인하세요")
        return

    # 주행 코드와 같은 그래버를 쓴다. 드라이버 버퍼에 프레임이 쌓이지 않아
    # 녹화가 실시간을 따라간다.
    grabber = FrameGrabber(camera, f"rec{args.cam}")
    deadline = time.time() + 5.0
    frame = None
    while time.time() < deadline:
        ok, frame, _ = grabber.read()
        if ok:
            break
        time.sleep(0.02)
    if frame is None:
        print("[record] 프레임을 못 받았습니다. 다른 앱이 카메라를 쓰고 있는지 확인하세요")
        grabber.release()
        return

    actual_h, actual_w = frame.shape[:2]
    out_path = args.out or default_out(args.cam)
    out_path = os.path.expanduser(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*str(args.codec)[:4].ljust(4, " ")),
        float(args.fps),
        (actual_w, actual_h),
    )
    if not writer.isOpened():
        print(f"[record] 저장 파일을 열 수 없습니다: {out_path} (codec={args.codec})")
        grabber.release()
        return

    print(f"[record] cam={args.cam} backend={backend} {actual_w}x{actual_h}@{args.fps:g}")
    print(f"[record] -> {out_path}")
    if args.auto:
        print("[record] --auto: 바로 녹화 시작")
    else:
        print("[record] Space 를 눌러 녹화를 시작하세요 (q 로 종료)")

    recording = bool(args.auto)
    frames = 0
    snapshots = 0
    started_at = time.time() if recording else None
    last_written = 0.0
    frame_interval = 1.0 / max(args.fps, 1.0)
    fps_marks = []
    last_seq = -1

    try:
        while True:
            ok, frame, _ = grabber.read()
            if not ok:
                if not grabber.healthy():
                    print("[record] 카메라가 멈췄습니다")
                    break
                time.sleep(0.005)
                continue

            seq = grabber.sequence
            new_frame = seq != last_seq
            last_seq = seq

            now = time.time()
            # 같은 프레임을 두 번 쓰지 않으면서, 저장 fps 간격은 지킨다
            if recording and new_frame and (now - last_written) >= frame_interval * 0.9:
                writer.write(frame)
                frames += 1
                last_written = now
                fps_marks.append(now)
                del fps_marks[:-30]

            elapsed = 0.0 if started_at is None else now - started_at
            if args.seconds and recording and elapsed >= args.seconds:
                print(f"[record] {args.seconds:g}초 도달 - 종료합니다")
                break

            fps_now = (
                (len(fps_marks) - 1) / max(fps_marks[-1] - fps_marks[0], 1e-6)
                if len(fps_marks) >= 2 else 0.0
            )

            if args.no_preview:
                time.sleep(0.002)
                continue

            cv2.imshow(WINDOW, draw_preview(frame, recording, elapsed, frames, fps_now, snapshots))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32:
                recording = not recording
                if recording and started_at is None:
                    started_at = time.time()
                print(f"[record] {'재개' if recording else '일시정지'} ({frames} frames)")
            elif key == ord("s"):
                snap = os.path.splitext(out_path)[0] + f"_snap{snapshots:03d}.jpg"
                cv2.imwrite(snap, frame)
                snapshots += 1
                print(f"[record] 스냅샷 저장: {snap}")
    except KeyboardInterrupt:
        print("\n[record] 중단됨")
    finally:
        writer.release()
        grabber.release()
        cv2.destroyAllWindows()
        size_mb = os.path.getsize(out_path) / 1e6 if os.path.exists(out_path) else 0.0
        duration = frames / max(args.fps, 1.0)
        print(f"[record] 저장 완료: {out_path}")
        print(f"[record] {frames} 프레임 / 약 {duration:.1f}초 / {size_mb:.1f} MB")
        if frames == 0:
            print("[record] ! 저장된 프레임이 없습니다 (Space 를 눌렀는지 확인하세요)")


if __name__ == "__main__":
    main()
