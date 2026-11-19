#!/bin/bash
# ACT 真机推理 · coffee_expo 单任务公共入口（由 run_act_<ID>.sh 调用）
#
#   bash run_act_task.sh <A1|B1|B2|C1|C2|C3> [权重目录] [时长秒]
#
# prompt / 权重 / 步数信息统一来自 coffee_task_table.sh。
# 直接调用时和旧的 run_act_A1.sh 等脚本完全等价。

set -e
set -o pipefail

TASK_ID="${1:-}"
if [ -z "$TASK_ID" ]; then
  echo "用法: bash run_act_task.sh <A1|B1|B2|C1|C2|C3> [权重目录] [时长秒]"
  exit 2
fi
shift

# shellcheck source=/dev/null
source "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/coffee_task_table.sh"

if ! coffee_task_valid "$TASK_ID"; then
  echo "❌ 未知任务：$TASK_ID（可用：$COFFEE_TASK_IDS）"
  exit 2
fi

TASK_DESC="$(coffee_task_prompt "$TASK_ID")"
DEFAULT_CKPT="$(coffee_task_act_ckpt "$TASK_ID")"
STEP_INFO="$(coffee_task_step_info "$TASK_ID")"

# shellcheck source=/dev/null
source "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/_act_expo_common.sh"
act_expo_main "$@"
