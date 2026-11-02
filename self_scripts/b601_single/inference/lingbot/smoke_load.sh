#!/usr/bin/env bash
# smoke_load.py 的启动器：先按 env.sh 准备好环境变量，再用 lerobot env 的解释器跑。
# 直接 ./smoke_load.py 会用系统 python3（没有 torch/numpy），所以统一走这个入口。
#
#   ./smoke_load.sh                          # 默认 48 步（首块 16 + 一块 32）
#   ./smoke_load.sh --obs none --steps 8     # 零图，只验形状与延迟
#   ./smoke_load.sh --video-steps 10 --action-steps 10 --guidance 1.0   # 快档对照
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
source "$HERE/env.sh"
exec "$LINGBOT_PY" -u "$HERE/smoke_load.py" "$@"   # -u：不缓冲，重定向到文件时也能实时看到进度
