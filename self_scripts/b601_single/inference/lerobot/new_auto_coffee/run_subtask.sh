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
#   --yes                     跳过开跑前的人工空格确认（调度器内部会传）
#   --no-agent                强制起子进程（不连常驻 agent）
#
# 常驻 agent：先 `bash run_coffee_agent.sh` 起服务，本脚本检测到 socket 就会把
# 子任务发给它（省掉 14–16s 启动开销）。socket 不在时自动回退成原来的子进程方式。
# agent 路径可用 B601_AGENT_SOCKET 指定；replay 也走 agent（agent 内直接放数据集，
# 不用再起子进程），agent 不可用时才回退子进程。
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
NO_AGENT="${B601_NO_AGENT:-0}"
CKPT_ARG=""
EPISODE_ARG=""
STATUS_FILE=""
DATASET=""
EPISODE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --policy) POLICY="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT="${2:-}"; shift 2 ;;
    --ckpt) CKPT_ARG="${2:-}"; shift 2 ;;
    --episode) EPISODE_ARG="${2:-}"; shift 2 ;;
    --status-file) STATUS_FILE="${2:-}"; shift 2 ;;
    --yes) ASSUME_READY=1; shift ;;
    --no-agent) NO_AGENT=1; shift ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

# 三套资源都先解析出来：q/w/e 热切换随时可能用到，不只当前策略那一份。
ACT_CKPT="$(coffee_task_act_ckpt "$TASK")"
PI05_MODEL="${B601_PI05_MODEL:-$COFFEE_PI05_MODEL}"
PI05_SHA="${B601_PI05_EXPECT_SHA:-$COFFEE_PI05_SHA}"
case "$POLICY" in
  act)  [ -n "$CKPT_ARG" ] && ACT_CKPT="$CKPT_ARG" ;;
  pi05) [ -n "$CKPT_ARG" ] && PI05_MODEL="$CKPT_ARG" ;;
esac
DATASET="${B601_COFFEE_DATASET:-$(coffee_task_default_dataset)}"
EPISODE="${EPISODE_ARG:-}"
if [ -z "$EPISODE" ] && [ -f "$DATASET/meta/info.json" ]; then
  EPISODE="$(coffee_task_replay_episode "$TASK" "$DATASET" 2>/dev/null || true)"
fi
CONTROL_SOCKET="${B601_TASK_CONTROL_SOCKET:-$LOG_DIR/task_control.sock}"

case "$POLICY" in
  act)
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
    if [ ! -f "$PI05_MODEL/model.safetensors" ]; then
      echo "❌ pi0.5 权重未就绪：$PI05_MODEL"
      echo "   可用 B601_PI05_MODEL=/path/to/merged_model 指定（250 集 12k：expo012000_250ep）"
      exit 4
    fi
    PI05_ENV=(
      "TASK=$TASK"
      "MODEL_PATH=$PI05_MODEL"
      "EXPECT_SHA256=$PI05_SHA"
      "DURATION=$TIMEOUT"
    )
    # 默认由脚本自己声明 12k 的 6 任务；换回 4 任务权重时用 B601_PI05_TRAINED_TASKS 覆盖
    if [ -n "${B601_PI05_TRAINED_TASKS:-}" ]; then
      PI05_ENV+=("TRAINED_TASKS=$B601_PI05_TRAINED_TASKS")
    fi
    CMD=(env "${PI05_ENV[@]}" bash "$HERE/run_pi05_coffee_250ep.sh")
    SUP_EXTRA=(--clean-exit-result timeout --hard-timeout "$(python -c "print(${TIMEOUT}+300)")")
    DETAIL="权重：$PI05_MODEL（sha ${PI05_SHA:0:16}…）"
    ;;
  replay)
    if [ ! -f "$DATASET/meta/info.json" ]; then
      echo "❌ replay 数据集不存在：$DATASET"
      echo "   可用 B601_COFFEE_DATASET=/path/to/dataset 指定"
      exit 4
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
      bash "$HERE/../../../03_replay.sh")
    SUP_EXTRA=(--enforce-timeout --clean-exit-result ok --grace 10)
    DETAIL="数据集：$DATASET  episode=$EPISODE"
    ;;
esac

# ---------------- 常驻 agent 路由 ----------------
# socket 在就发给 coffee_agent.py（机械臂/相机/权重都已在显存里）；否则回退子进程。
AGENT_SOCKET="${B601_AGENT_SOCKET:-$LOG_DIR/coffee_agent.sock}"
AGENT_CKPT=""
case "$POLICY" in
  act)  AGENT_CKPT="$ACT_CKPT" ;;
  pi05) AGENT_CKPT="$PI05_MODEL" ;;
esac
USE_AGENT=0
if [ "$NO_AGENT" = "1" ]; then
  EXEC_MODE="子进程（--no-agent）"
else
  # 不只是看 socket 文件在不在：真发一条 health，确认 agent 还活着且健康。
  # 健康但 busy 也走 agent（由 agent 返回 busy，绝不回退去抢相机）。
  AGENT_STATUS="$(python "$HERE/agent_health.py" --socket "$AGENT_SOCKET" --timeout 1 2>&1)"
  AGENT_RC=$?
  if [ "$AGENT_RC" -eq 0 ]; then
    USE_AGENT=1
    EXEC_MODE="常驻 agent（$AGENT_STATUS）"
  else
    EXEC_MODE="子进程（agent 不可用：$AGENT_STATUS）"
  fi
fi

echo "=========================================="
echo "子任务 $TASK（policy=$POLICY，上限 ${TIMEOUT}s）"
echo "=========================================="
echo "prompt：$PROMPT"
echo "$DETAIL"
echo "执行方式：$EXEC_MODE"
echo "结果文件：$STATUS_FILE"
echo "运行日志：$RUN_LOG"
echo "接管按键：q=pi0.5 | w=ACT | e=归零后 replay（语音程序里按）"
if [ "$ASSUME_READY" = "1" ]; then
  echo "人工确认：跳过（--yes / B601_ASSUME_READY=1）"
else
  echo "人工确认：开跑前在本终端按空格"
fi
echo "=========================================="
echo ""

# 后端由监督器以 stdin=DEVNULL 启动（避免抢占按键），所以确认只能在这里做一次。
if [ "$ASSUME_READY" != "1" ]; then
  if [ -t 0 ]; then
    printf '确认从臂在零位（夹爪闭合）、周边安全，按空格开始 %s（Ctrl+C 取消）...' "$TASK"
    while :; do
      IFS= read -r -n 1 -s key || exit 130
      [ "$key" = " " ] && break
    done
    echo
  else
    echo "⚠️  非交互终端：跳过确认，请自行确保机械臂在零位"
  fi
  ASSUME_READY=1
  export B601_ASSUME_READY=1
fi

if [ "$USE_AGENT" = "1" ]; then
  # agent 模式：后端已在常驻进程里，监督器只发 run/stop；硬超时留作最后兜底
  exec python "$HERE/subtask_supervisor.py" \
    --task "$TASK" \
    --policy "$POLICY" \
    --timeout "$TIMEOUT" \
    --status-file "$STATUS_FILE" \
    --log-file "$RUN_LOG" \
    --agent-socket "$AGENT_SOCKET" \
    --agent-prompt "$PROMPT" \
    --agent-ckpt "$AGENT_CKPT" \
    --agent-dataset "$DATASET" \
    --agent-episode "${EPISODE:-0}" \
    --control-socket "$CONTROL_SOCKET" \
    --agent-act-ckpt "$ACT_CKPT" \
    --agent-pi05-ckpt "$PI05_MODEL" \
    --agent-replay-dataset "$DATASET" \
    --agent-replay-episode "${EPISODE:-0}" \
    --grace 10 \
    --home-wait 30 \
    --hard-timeout "$(python -c "print(${TIMEOUT}+120)")"
else
  exec python "$HERE/subtask_supervisor.py" \
    --task "$TASK" \
    --policy "$POLICY" \
    --timeout "$TIMEOUT" \
    --status-file "$STATUS_FILE" \
    --log-file "$RUN_LOG" \
    "${SUP_EXTRA[@]}" \
    -- "${CMD[@]}"
fi
