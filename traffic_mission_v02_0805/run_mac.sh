#!/usr/bin/env bash
# macOS 실행 스크립트. run_windows.ps1 과 같은 인자를 받는다.
#
#   ./run_mac.sh                       실제 주행
#   ./run_mac.sh --list-cameras        이 맥에서 열리는 카메라 인덱스 확인
#   ./run_mac.sh --dry-run             시리얼 없이 영상만
#   ./run_mac.sh --cam 0 --traffic-cam 1
#
# 영상만 찍으려면:  python record_camera.py --list  /  --cam 1
#
# 카메라 백엔드와 시리얼 포트는 실행 시 자동으로 맥용으로 잡히므로
# calib.json 을 고칠 필요가 없다. 다만 **카메라 인덱스는 맥과 Windows 가
# 다르게 잡히므로** --list-cameras 로 확인해서 맞춰야 한다.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if command -v conda >/dev/null 2>&1 && conda env list | grep -q '^drive '; then
    exec conda run --no-capture-output -n drive python integrated_traffic_obstacle.py "$@"
  fi
  PYTHON=python3
fi
exec "$PYTHON" integrated_traffic_obstacle.py "$@"
