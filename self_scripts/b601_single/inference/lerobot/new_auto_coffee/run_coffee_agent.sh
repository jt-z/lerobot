#!/bin/bash
# 启动咖啡子任务常驻推理 agent（机械臂 + 三路相机 + ACT/pi0.5 权重常驻显存）。
#
# 起来之后，run_subtask.sh 会自动改走 unix socket，不再为每个子任务重复启动推理进程：
#   bash run_coffee_agent.sh                  # 真机常驻（默认懒加载，用到才占显存）
#   PRELOAD=all bash run_coffee_agent.sh      # 启动即加载 6 份 ACT + pi0.5（约 8.2GB，先看 nvidia-smi）
#   DISPLAY_DATA=0 bash run_coffee_agent.sh   # 不开 Rerun 可视化
#   bash run_coffee_agent.sh --self-test      # 不碰硬件，只验证 socket 协议
#
# 可用环境变量：
#   B601_AGENT_SOCKET   socket 路径（默认 $HOME/dev/lerobot/self_scripts/_logs/coffee_agent.sock）
#   PRELOAD             all 或逗号分隔的 type:path
#   DISPLAY_DATA        1=开 Rerun（默认），0=关
#   LITE_LOAD           1=pi0.5 走 lite 包装器（默认，12GB 卡必需）
# 其它参数原样透传给 coffee_agent.py。

set -u

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"

# 与训练同版本的 lerobot（0.6.2）
_B601_ENV_BIN="$HOME/dev/miniforge3/envs/lerobot_b601/bin"
if [ -x "$_B601_ENV_BIN/python" ]; then
  export PATH="$_B601_ENV_BIN:$PATH"
fi

# 相机设备路径 / CAN 口 / 校准 id 与采集、推理脚本共用一份档案
# shellcheck source=/dev/null
source "$HERE/../_common.sh"

export B601_AGENT_CAM_HAND="$B601_CAM_HAND_DEV"
export B601_AGENT_CAM_FRONT="$B601_CAM_FRONT_DEV"
export B601_AGENT_CAM_TOP="$B601_CAM_TOP_DEV"
export B601_FOLLOWER_PORT="$B601_FOLLOWER_PORT"
export FOLLOWER_ID="$FOLLOWER_ID"

export LITE_LOAD="${LITE_LOAD:-1}"
export LITE_TRACE="${LITE_TRACE:-1}"
export LITE_VISION_BF16="${LITE_VISION_BF16:-1}"

SOCKET="${B601_AGENT_SOCKET:-$LOG_DIR/coffee_agent.sock}"
export B601_AGENT_SOCKET="$SOCKET"
mkdir -p "$LOG_DIR"
AGENT_LOG="$LOG_DIR/coffee_agent_$(date +%Y%m%d_%H%M%S).log"

# PRELOAD=all：把任务表里的 ACT 权重和默认 pi0.5 权重展开成 type:path 列表
PRELOAD="${PRELOAD:-}"
if [ "$PRELOAD" = "all" ]; then
  # shellcheck source=/dev/null
  source "$HERE/coffee_task_table.sh"
  _parts=("pi05:$COFFEE_PI05_MODEL")
  for _t in A1 B1 B2 C2; do
    _p="$(coffee_task_act_ckpt "$_t")"
    [ -f "$_p/model.safetensors" ] && _parts+=("act:$_p")
  done
  for _t in C1 C3; do
    _p="$(coffee_task_act_ckpt "$_t")"
    [ -f "$_p/model.safetensors" ] && _parts+=("act:$_p")
  done
  PRELOAD="$(IFS=,; echo "${_parts[*]}")"
  unset _parts _p _t
fi

echo "=========================================="
echo "coffee agent 常驻推理"
echo "=========================================="
echo "socket ：$SOCKET"
echo "日志   ：$AGENT_LOG"
echo "相机   ：hand=$B601_AGENT_CAM_HAND"
echo "         front=$B601_AGENT_CAM_FRONT"
echo "         top=$B601_AGENT_CAM_TOP"
echo "CAN    ：$B601_FOLLOWER_PORT（id=$FOLLOWER_ID）"
echo "预加载 ：${PRELOAD:-<懒加载>}"
echo "可视化 ：DISPLAY_DATA=${DISPLAY_DATA:-1}"
echo "=========================================="
echo ""

ARGS=(
  --socket "$SOCKET"
  --log-file "$AGENT_LOG"
  --device "${DEVICE:-cuda}"
  --display-data "${DISPLAY_DATA:-1}"
)
if [ -n "$PRELOAD" ]; then
  ARGS+=(--preload "$PRELOAD")
fi

exec python "$HERE/coffee_agent.py" "${ARGS[@]}" "$@"
