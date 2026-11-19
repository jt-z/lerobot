#!/bin/bash
# 单子任务统一入口：run_subtask.sh <A1|B1|B2|C1|C2|C3> [选项]
#
# 外部程序按子任务依次调用本脚本即可；退出码是稳定契约：
#   0 成功 · 2 操作员判失败 · 3 超时 · 4 后端/环境错误 · 130 人为中止
# 最后一次结果同时写入 $HOME/dev/lerobot/self_scripts/_logs/subtask_last.json
#
# 选项：
#   --policy act|replay|pi05   执行方式，默认 act
#   --timeout 40               单个子任务最长执行秒数，默认 40
#   --ckpt PATH               覆盖权重目录（act/pi05 生效）
#   --episode N               覆盖 replay 的 episode（默认按 prompt 自动找）
#   --status-file PATH        自定义结果 JSON 路径
#   --yes                     跳过开跑前的人工 ENTER 确认（调度器内部会传）
#
# 示例：
#   bash run_subtask.sh A1
#   bash run_subtask.sh C2 --policy pi05 --timeout 30
#   bash run_subtask.sh C1 --policy replay --yes
#
# 运行中操作员按键：
#   0 = 判定本子任务成功，提前结束并归零
#   1 = 判定失败，提前结束并归零
#   Ctrl+C = 中止，同样先归零
# 到 --timeout 仍未按键：ACT/pi0.5 视为超时失败（3），replay 正常放完视为成功（0）。

set -u

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/coffee_task_table.sh"

LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
DEFAULT_STATUS_FILE="$LOG_DIR/subtask_last.json"

TASK=""
POLICY="${B601_POLICY:-act}"
TIMEOUT="${B601_SUBTASK_TIMEOUT:-40}"
ASSUME_READY="${B601_ASSUME_READY:-0}"
CKPT_ARG=""
EPISODE_ARG=""
STATUS_FILE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --policy) POLICY="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT="${2:-}"; shift 2 ;;
    --ckpt) CKPT_ARG="${2:-}"; shift 2 ;;
    --episode) EPISODE_ARG="${2:-}"; shift 2 ;;
    --status-file) STATUS_FILE="${2:-}"; shift 2 ;;
    --yes) ASSUME_READY=1; shift ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "❌ 未知选项：$1"; exit 4 ;;
    *)
      if [ -z "$TASK" ]; then
        TASK="$1"
      elif [[ "$1" =~ ^[0-9]+$ ]]; then
        TIMEOUT="$1"   # 兼容旧写法：run_subtask.sh A1 300
      else
        echo "❌ 多余参数：$1"; exit 4
      fi
      shift
      ;;
  esac
done

if [ -z "$TASK" ]; then
  echo "用法: bash run_subtask.sh <A1|B1|B2|C1|C2|C3> [--policy act|replay|pi05] [--timeout 40] [--yes]"
  exit 4
fi
if ! coffee_task_valid "$TASK"; then
  echo "❌ 未知子任务：$TASK（可用：$COFFEE_TASK_IDS）"
  exit 4
fi
case "$POLICY" in
  act|replay|pi05) ;;
  *) echo "❌ 未知策略：$POLICY（可用：act / replay / pi05）"; exit 4 ;;
esac
if ! [[ "$TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "❌ --timeout 必须是数字：$TIMEOUT"; exit 4
fi

# 与训练同版本的 lerobot（0.6.2）：后面的 python / lerobot-rollout 都走这个环境
_B601_ENV_BIN="$HOME/dev/miniforge3/envs/lerobot_b601/bin"
if [ -x "$_B601_ENV_BIN/lerobot-rollout" ]; then
  export PATH="$_B601_ENV_BIN:$PATH"
fi

export B601_ASSUME_READY="$ASSUME_READY"
mkdir -p "$LOG_DIR"
STATUS_FILE="${STATUS_FILE:-$DEFAULT_STATUS_FILE}"
RUN_LOG="$LOG_DIR/subtask_${TASK}_${POLICY}_$(date +%Y%m%d_%H%M%S).log"

PROMPT="$(coffee_task_prompt "$TASK")"
CMD=()
SUP_EXTRA=()

case "$POLICY" in
  act)
    ACT_CKPT="${CKPT_ARG:-$(coffee_task_act_ckpt "$TASK")}"
    if [ ! -f "$ACT_CKPT/config.json" ] || [ ! -f "$ACT_CKPT/model.safetensors" ]; then
      echo "❌ ACT 权重未就绪：$ACT_CKPT"
      echo "   该任务是六任务重训的占位路径。权重到位后二选一："
      echo "     export B601_ACT_CKPT_${TASK}=/path/to/pretrained_model"
      echo "     或修改 coffee_task_table.sh 中的占位路径"
      exit 4
    fi
    CMD=(bash "$HERE/run_act_${TASK}.sh" "$ACT_CKPT" "$TIMEOUT")
    SUP_EXTRA=(--clean-exit-result timeout --hard-timeout "$(python -c "print(${TIMEOUT}+300)")")
    DETAIL="权重：$ACT_CKPT"
    ;;
  pi05)
    PI05_MODEL="${CKPT_ARG:-${B601_PI05_MODEL:-/home/jt/pi05_bf16/expo008000}}"
    if [ ! -f "$PI05_MODEL/model.safetensors" ]; then
      echo "❌ pi0.5 权重未就绪：$PI05_MODEL"
      echo "   可用 B601_PI05_MODEL=/path/to/merged_model 指定"
      exit 4
    fi
    if ! coffee_task_is_trained_act "$TASK"; then
      echo "⚠️  pi0.5 当前权重只训练了 A1/B1/B2/C2；按占位逻辑强行跑 $TASK，"
      echo "    结果不可信，等新权重到位后替换 B601_PI05_MODEL 即可。"
    fi
    CMD=(env \
      "TASK_ID=$TASK" \
      "MODEL_PATH=$PI05_MODEL" \
      "DURATION=$TIMEOUT" \
      "TRAINED_TASKS=${B601_PI05_TRAINED_TASKS:-A1,B1,B2,C1,C2,C3}" \
      "ALLOW_UNTRAINED=${B601_PI05_ALLOW_UNTRAINED:-1}" \
      bash "$HERE/run_pi05_coffee_expo.sh")
    SUP_EXTRA=(--clean-exit-result timeout --hard-timeout "$(python -c "print(${TIMEOUT}+300)")")
    DETAIL="权重：$PI05_MODEL"
    ;;
  replay)
    DATASET="${B601_COFFEE_DATASET:-$(coffee_task_default_dataset)}"
    if [ ! -f "$DATASET/meta/info.json" ]; then
      echo "❌ replay 数据集不存在：$DATASET"
      echo "   可用 B601_COFFEE_DATASET=/path/to/dataset 指定"
      exit 4
    fi
    EPISODE="${EPISODE_ARG:-}"
    if [ -z "$EPISODE" ]; then
      EPISODE="$(coffee_task_replay_episode "$TASK" "$DATASET")" || true
    fi
    if ! [[ "$EPISODE" =~ ^[0-9]+$ ]]; then
      echo "❌ 无法为 $TASK 找到 replay episode（数据集：$DATASET）"
      echo "   可用 --episode N 或 B601_REPLAY_EPISODE_${TASK}=N 指定"
      exit 4
    fi
    CMD=(env \
      "DATASET_ROOT=$DATASET" \
      "EPISODE=$EPISODE" \
      "USE_RERUN=${B601_REPLAY_RERUN:-1}" \
      bash "$HERE/../../03_replay.sh")
    SUP_EXTRA=(--enforce-timeout --clean-exit-result ok --grace 10)
    DETAIL="数据集：$DATASET  episode=$EPISODE"
    ;;
esac

echo "=========================================="
echo "子任务 $TASK（policy=$POLICY，上限 ${TIMEOUT}s）"
echo "=========================================="
echo "prompt：$PROMPT"
echo "$DETAIL"
echo "结果文件：$STATUS_FILE"
echo "运行日志：$RUN_LOG"
if [ "$ASSUME_READY" = "1" ]; then
  echo "人工确认：跳过（--yes / B601_ASSUME_READY=1）"
else
  echo "人工确认：开跑前在本终端按 ENTER"
fi
echo "=========================================="
echo ""

# 后端由监督器以 stdin=DEVNULL 启动（避免抢占按键），所以确认只能在这里做一次。
if [ "$ASSUME_READY" != "1" ]; then
  if [ -t 0 ]; then
    read -r -p "确认从臂在零位（夹爪闭合）、周边安全，按 ENTER 开始 $TASK（Ctrl+C 取消）..." _
  else
    echo "⚠️  非交互终端：跳过确认，请自行确保机械臂在零位"
  fi
  ASSUME_READY=1
  export B601_ASSUME_READY=1
fi

exec python "$HERE/subtask_supervisor.py" \
  --task "$TASK" \
  --policy "$POLICY" \
  --timeout "$TIMEOUT" \
  --status-file "$STATUS_FILE" \
  --log-file "$RUN_LOG" \
  "${SUP_EXTRA[@]}" \
  -- "${CMD[@]}"
