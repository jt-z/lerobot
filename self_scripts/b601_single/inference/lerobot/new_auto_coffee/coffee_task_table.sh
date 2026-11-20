#!/bin/bash
# coffee_expo 六子任务唯一任务表：prompt / ACT 权重 / pi0.5 权重 / replay episode 解析。
#
# 被 run_subtask.sh、run_coffee.sh、run_act_task.sh 与各 run_act_<ID>.sh source。
# 这里只定义函数和默认值，不检查硬件、不打印、不执行。
#
# ⚠️ prompt 必须与数据集 meta/tasks.parquet 逐字一致（含大小写与句号）。
#    数据集默认：/home/jt/b601_data/b601_pi05_coffee_expo_20261118_181510（250 集 / 6 任务）

COFFEE_TASK_IDS="A1 B1 B2 C1 C2 C3"

# 六任务重训权重的落盘根目录占位。权重到位后可以：
#   1) 修改本文件的占位路径，或
#   2) 用环境变量覆盖单个任务：B601_ACT_CKPT_C1=/path/to/pretrained_model
COFFEE_6TASK_ROOT="${B601_EXPO_6TASK_ROOT:-$HOME/LX/pai0}"

# pi0.5：250 集 run 的 12k 权重一份覆盖 6 个任务（切换只换 prompt）。
# 9k 是上一档（同锚点对比 5/6 任务略差）；102 集旧 run 的 expo012000（无 _250ep 后缀）是 4 任务，别混。
COFFEE_PI05_MODEL="${B601_PI05_MODEL:-$HOME/pi05_bf16/expo012000_250ep}"
COFFEE_PI05_SHA="${B601_PI05_EXPECT_SHA:-f70669263aed4ae1908dddda97362ba47861cff354b630ece37a0e1ef130c541}"

coffee_task_valid() {
  case " $COFFEE_TASK_IDS " in
    *" $1 "*) return 0 ;;
  esac
  return 1
}

coffee_task_prompt() {
  case "$1" in
    A1) echo "Pick up an empty paper cup from the cup dispenser and place it on the silver drip tray under the coffee machine spout. Return to the home pose." ;;
    B1) echo "Use the gripper tip to select Hot Water and press the Start button to dispense hot water into the cup using the current preset settings. Return to the home pose." ;;
    B2) echo "Use the gripper tip to select Americano and press the Start button to brew coffee into the cup using the current preset settings. Return to the home pose." ;;
    C1) echo "Pick up the filled cup from the silver drip tray under the coffee machine spout and place it in cup placement area 1. Return to the home pose." ;;
    C2) echo "Pick up the filled cup from the silver drip tray under the coffee machine spout and place it in cup placement area 2. Return to the home pose." ;;
    C3) echo "Pick up the filled cup from the silver drip tray under the coffee machine spout and place it in cup placement area 3. Return to the home pose." ;;
    *) return 1 ;;
  esac
}

# ACT 权重目录（pretrained_model）。A1/B1/B2/C2 是 2026-11-19 的 4 任务权重；
# C1/C3 是六任务重训的占位路径，权重到位后改这里或用 B601_ACT_CKPT_C1 / C3 覆盖。
coffee_task_act_ckpt() {
  case "$1" in
    A1) echo "${B601_ACT_CKPT_A1:-$HOME/LX/pai0/act_expo_A1_20261119/checkpoints/003100/pretrained_model}" ;;
    B1) echo "${B601_ACT_CKPT_B1:-$HOME/LX/pai0/act_expo_B1_20261119/checkpoints/002600/pretrained_model}" ;;
    B2) echo "${B601_ACT_CKPT_B2:-$HOME/LX/pai0/act_expo_B2_20261119/checkpoints/001800/pretrained_model}" ;;
    C2) echo "${B601_ACT_CKPT_C2:-$HOME/LX/pai0/act_expo_C2_20261119/checkpoints/001800/pretrained_model}" ;;
    C1) echo "${B601_ACT_CKPT_C1:-$COFFEE_6TASK_ROOT/act_expo_C1_6task/checkpoints/last/pretrained_model}" ;;
    C3) echo "${B601_ACT_CKPT_C3:-$COFFEE_6TASK_ROOT/act_expo_C3_6task/checkpoints/last/pretrained_model}" ;;
    *) return 1 ;;
  esac
}

coffee_task_step_info() {
  case "$1" in
    A1) echo "step 003100 / 3100（last）" ;;
    B1) echo "step 002600 / 2600（last）" ;;
    B2) echo "step 001800 / 1800（last）" ;;
    C2) echo "step 001800 / 1800（last）" ;;
    C1|C3) echo "六任务重训权重（占位，未到位）" ;;
    *) echo "unknown" ;;
  esac
}

coffee_task_default_dataset() {
  echo "${B601_COFFEE_DATASET:-/home/jt/b601_data/b601_pi05_coffee_expo_20261118_181510}"
}

# 按 prompt 在数据集里找该任务所有 episode，取中间编号作为默认 replay 目标
# （中间批次通常比最早/最晚的稳定；可用 B601_REPLAY_EPISODE_<ID> 或 --episode 覆盖）。
coffee_task_replay_episode() {
  local task="$1" dataset="${2:-$(coffee_task_default_dataset)}"
  local override_var="B601_REPLAY_EPISODE_${task}"
  local override="${!override_var:-}"
  if [ -n "$override" ]; then
    echo "$override"
    return 0
  fi
  python - "$dataset" "$(coffee_task_prompt "$task")" <<'PY'
import glob
import sys

import pandas as pd

root, prompt = sys.argv[1], sys.argv[2]
matches = []
for path in sorted(glob.glob(root + "/meta/episodes/**/*.parquet", recursive=True)):
    df = pd.read_parquet(path, columns=["episode_index", "tasks"])
    for _, row in df.iterrows():
        if prompt in list(row["tasks"]):
            matches.append(int(row["episode_index"]))
if not matches:
    sys.exit(1)
matches.sort()
print(matches[len(matches) // 2])
PY
}
