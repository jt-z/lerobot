#!/usr/bin/env bash
# 路线 A 启动器：本机 lerobot 环境跑控制循环，4090 上的 JAX policy server 出动作。
#
# 用法：
#   bash run_pi05_ws_rollout.sh 330              # 真机（330 集模型，需先起 server）
#   bash run_pi05_ws_rollout.sh 462 --duration 60
#   bash run_pi05_ws_rollout.sh 330 --dry-run --duration 5     # 干跑，不接机器人
#
# 前置（另开一个窗口）：
#   cd /home/kf/dev/lerobot/self_scripts/b601_single/inference
#   source b601_pi05_env.sh && $PY serve_policy.py --model 330 --port 8000
set -uo pipefail
cd "$(dirname "$0")"

MODEL=${1:?用法: bash run_pi05_ws_rollout.sh <330|462> [额外参数...]}
shift || true
PORT=${B601_WS_PORT:-8000}

# lerobot 环境（py3.12 + torch，含 seeed_b601 驱动）
LEROBOT_PY=${LEROBOT_PY:-/home/kf/miniconda3/envs/lerobot/bin/python}
# openpi-client（纯 python，协议定义在 openpi 仓库里）
export PYTHONPATH="/home/kf/b601_pi05/openpi/packages/openpi-client/src:$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

echo "模型: $MODEL    server: 127.0.0.1:$PORT"
command -v $LEROBOT_PY >/dev/null || { echo "❌ 找不到 $LEROBOT_PY"; exit 1; }

# can0 检查（dry-run 不需要）
if [[ ! " $* " =~ " --dry-run " ]]; then
  ip -br link show can0 2>/dev/null | grep -q "can0" || { echo "❌ can0 不存在（sudo ip link set can0 type can bitrate 1000000 up）"; exit 1; }
  ip -br link show can0 | grep -q "UP" || { echo "⚠️ can0 未 UP，尝试拉起需要 sudo"; }
fi

exec $LEROBOT_PY run_pi05_ws_rollout.py --port "$PORT" "$@"
