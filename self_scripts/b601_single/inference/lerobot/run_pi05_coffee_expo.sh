#!/bin/bash
# 单 B601-RS 用「新训练好的 coffee_expo 合并权重」做 pi05 真机推理（base 策略，不录数据）
#
# 权重：/home/jt/pi05_bf16/<expoXXXXXX>（232 上 pi05 LoRA checkpoint 合并进 pi05_base 后的
#       完整权重：视觉塔/layernorm/投影层留 fp32、其余 bf16，8.71 GiB / 813 张量（含 embed_tokens））
#       现有：expo008000（checkpoints/008000）、expo012000（checkpoints/012000）
#       任务覆盖：这两档是 4 任务权重（A1/B1/B2/C2）；C1/C3 需要 6 任务重训权重。
#       默认用 expo008000；换档直接 MODEL_PATH=/home/jt/pi05_bf16/<另一档> 或改本文件 MODEL_PATH
#       注意：这些是 bf16 全量权重，不是 int4 量化产物（int4 实验在 ~/pi05_int4/）
#   目录内含：model.safetensors / config.json / extras.pt / tokenizer /
#             policy_{pre,post}processor*.json / adapter_model.safetensors（原始 LoRA，仅留档）
#
# 与 run_pi05_lora.sh 的区别（别混用）：
#   - 这里跑的是已经合并好的全量权重，目录里有 model.safetensors；
#   - config.json 里 use_peft 仍是 true（合并时没改），本脚本用 --policy.use_peft=false 覆盖，
#     否则 rollout 会按 adapter_config.json 去找训练机上的 pi05_base（本机不存在）；
#   - 12GB 卡装不下 4.14B 权重，默认走 lx/tools/pi05_lite.py 包装器（构建期跳过 lm_head + 分段计时）。
#
# 用法：
#   ./run_pi05_coffee_expo.sh                     # 默认 A1（端杯上咖啡机托盘）
#   TASK_ID=B1 ./run_pi05_coffee_expo.sh          # 按技能 ID 选任务：A1 / B1 / B2 / C1 / C2 / C3
#   TASK_INDEX=1 ./run_pi05_coffee_expo.sh        # 本地 prompt 下标（0~5），不是数据集 task_index
#   TRAINED_TASKS=A1,B1,B2,C1,C2,C3 ./run_pi05_coffee_expo.sh  # 6 任务重训后声明权重支持的任务
#   ALLOW_UNTRAINED=1 ./run_pi05_coffee_expo.sh    # 仅验证链路，允许跑当前权重没训过的任务
#   DURATION=120 ./run_pi05_coffee_expo.sh        # 跑 120 秒自动退出（默认不限）
#   VERIFY_SHA=1 ./run_pi05_coffee_expo.sh        # 上机前校验 8.3GB 权重 sha256（默认不校验，验一次约 10 秒）
#   MODEL_PATH=/path/to/ckpt ./run_pi05_coffee_expo.sh   # 换权重目录
#   INFERENCE_TYPE=rtc ./run_pi05_coffee_expo.sh # 改回异步 RTC（推荐：换块不停顿，需 queue_threshold 余量）
#   LITE_LOAD=0 ./run_pi05_coffee_expo.sh         # 不用 lite 包装器（仅显存足够的机器）
#
# 参考：lx/pi05_inference.sh（lite 包装器 + RTC 参数）、run_pi05_lora.sh（B601 真机接线）
# 离线自检（不动机器人）：MODEL_PATH=... python lx/tools/pi05_lite.py --bench <MODEL_PATH> --chunks 1

set -e  # 遇到错误立即退出
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# 某些 IDE 终端会把自带的老 libstdc++ 注入 LD_LIBRARY_PATH，导致 pyrealsense2 报 GLIBCXX 缺失
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v 'trae-cn-server' | paste -sd: -)"
  export LD_LIBRARY_PATH
fi

# _common.sh 会把 INFERENCE_TYPE 置 sync、ACTION_TRACE 置 1，用户显式传入的值先存下来
_B601_INFERENCE_TYPE_ENV="${INFERENCE_TYPE:-}"
_B601_ACTION_TRACE_ENV="${ACTION_TRACE:-}"
_B601_TASK_DESC_ENV="${TASK_DESCRIPTION:-}"

. "$REPO_ROOT/self_scripts/b601_common/_env.sh"
. "$SCRIPT_DIR/_common.sh"

# ==================== 模型 ====================
MODEL_PATH="${MODEL_PATH:-/home/jt/pi05_bf16/expo008000}"
EXPECT_SHA256="${EXPECT_SHA256:-e634957060b99f1407ab20cd26ee9de0a318fdceaf5b789c88dc3d95d5a95c58}"
VERIFY_SHA="${VERIFY_SHA:-0}"

POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"   # 与训练 config.json 一致
INFERENCE_TYPE="${_B601_INFERENCE_TYPE_ENV:-sync}"
RTC_QUEUE_THRESHOLD="${RTC_QUEUE_THRESHOLD:-35}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-20}"

ACTION_TRACE="${_B601_ACTION_TRACE_ENV:-0}"        # 1 = 逐 tick 记录动作（仅 sync）
ACTION_TRACE_LOG="$HOME/dev/lerobot/self_scripts/_logs/action_pi05_expo.log"

LITE_LOAD="${LITE_LOAD:-1}"                        # 1 = 构建期跳过 lm_head（12GB 卡必需）
LITE_TRACE="${LITE_TRACE:-1}"
LITE_VISION_BF16="${LITE_VISION_BF16:-1}"
LITE_WRAPPER="$REPO_ROOT/lx/tools/pi05_lite.py"

# ==================== 任务文本（与训练集 meta/tasks.parquet 逐字一致） ====================
# 数据集当前 task_index（250 集版）：0=C2, 1=A1, 2=B1, 3=C1, 4=B2, 5=C3。
# ⚠️ task_index 顺序会随数据集重建变化；本脚本的 TASK_INDEX 只是下面 TASKS 的下标，
#    不是数据集 task_index。选任务请优先用 TASK_ID。
# 旧版 TASK_INDEX 顺序是 A1/B2/B1/C2；本版改为 A1/B1/B2/C1/C2/C3，
# 用数字的老命令请改用 TASK_ID（数字含义已经变了）。
TASK_IDS=(A1 B1 B2 C1 C2 C3)
TASKS=(
"Pick up an empty paper cup from the cup dispenser and place it on the silver drip tray under the coffee machine spout. Return to the home pose."
"Use the gripper tip to select Hot Water and press the Start button to dispense hot water into the cup using the current preset settings. Return to the home pose."
"Use the gripper tip to select Americano and press the Start button to brew coffee into the cup using the current preset settings. Return to the home pose."
"Pick up the filled cup from the silver drip tray under the coffee machine spout and place it in cup placement area 1. Return to the home pose."
"Pick up the filled cup from the silver drip tray under the coffee machine spout and place it in cup placement area 2. Return to the home pose."
"Pick up the filled cup from the silver drip tray under the coffee machine spout and place it in cup placement area 3. Return to the home pose."
)
TASK_INDEX="${TASK_INDEX:-}"
# 采集脚本里的技能 ID → TASKS 下标（注意：不是数据集 task_index）
# 数据量（250 集版）：A1=80 / B1=40 / B2=40 / C1=40 / C2=40 / C3=10
TASK_ID="${TASK_ID:-}"
if [ -n "$TASK_ID" ]; then
  case "$TASK_ID" in
    A1) TASK_INDEX=0 ;;   # 端空杯上咖啡机托盘
    B1) TASK_INDEX=1 ;;   # 选热水并启动
    B2) TASK_INDEX=2 ;;   # 选美式并启动
    C1) TASK_INDEX=3 ;;   # 满杯放到杯位 1
    C2) TASK_INDEX=4 ;;   # 满杯放到杯位 2
    C3) TASK_INDEX=5 ;;   # 满杯放到杯位 3
    *) echo "❌ 未知任务 ID：$TASK_ID（支持 A1 / B1 / B2 / C1 / C2 / C3）"; exit 1 ;;
  esac
fi
TASK_INDEX="${TASK_INDEX:-0}"
if [ -z "${TASKS[$TASK_INDEX]:-}" ]; then
  echo "❌ TASK_INDEX=$TASK_INDEX 越界（可用 0~$((${#TASKS[@]} - 1))）"
  exit 1
fi
TASK_ID="${TASK_IDS[$TASK_INDEX]}"

# 当前权重的任务覆盖范围：expo008000/012000 只训练了 4 个技能。
# 6 任务重训后把它改成 A1,B1,B2,C1,C2,C3（或用环境变量覆盖）。
TRAINED_TASKS="${TRAINED_TASKS:-A1,B1,B2,C2}"
if [[ ",$TRAINED_TASKS," != *",$TASK_ID,"* ]]; then
  if [ "${ALLOW_UNTRAINED:-0}" = "1" ]; then
    echo "⚠️  ALLOW_UNTRAINED=1：$TASK_ID 不在当前权重的训练任务（$TRAINED_TASKS）里，策略输出大概率无效"
  else
    echo "❌ 当前权重未训练 $TASK_ID（TRAINED_TASKS=$TRAINED_TASKS）"
    echo "   expo008000/012000 是 4 任务权重；C1/C3 需要 6 任务重训权重后再跑。"
    echo "   重训后：TRAINED_TASKS=A1,B1,B2,C1,C2,C3 $0"
    echo "   仅验证链路：ALLOW_UNTRAINED=1 $0"
    exit 1
  fi
fi

TASK_DESCRIPTION="${_B601_TASK_DESC_ENV:-${TASKS[$TASK_INDEX]}}"

FPS="${FPS:-30}"            # 训练数据 30Hz，不能改
DURATION="${DURATION:-0}"   # 0 = 一直推理到 Ctrl+C

TITLE="单臂推理（pi05 coffee_expo 合并权重 / B601-RS / base 纯推理）"
RUN_TAG="pi05_coffee_expo"

# ==================== 相机 key 重命名（robot → policy，与训练 preprocessor 一致） ====================
CAMERA_RENAME_MAP='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.hand": "observation.images.left_wrist_0_rgb", "observation.images.top": "observation.images.right_wrist_0_rgb"}'

# ==================== 推理前检查 ====================
b601_require_lerobot
b601_banner "$TITLE"

b601_check_model_files "$MODEL_PATH" \
  model.safetensors config.json extras.pt \
  policy_preprocessor.json policy_postprocessor.json \
  policy_preprocessor_step_2_normalizer_processor.safetensors \
  policy_postprocessor_step_0_unnormalizer_processor.safetensors \
  tokenizer/tokenizer.json

# 权重配置自检：合并权重必须是 pi05、三路相机与训练一致、动作/状态维度对得上
python - "$MODEL_PATH/config.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
feats = cfg.get("input_features", {})
imgs = sorted(k.rsplit(".", 1)[-1] for k in feats if k.startswith("observation.images."))
state_dim = feats.get("observation.state", {}).get("shape", ["?"])[0]
act_dim = cfg.get("output_features", {}).get("action", {}).get("shape", ["?"])[0]
print(f"   类型={cfg.get('type')} 精度={cfg.get('dtype')} chunk={cfg.get('chunk_size')} "
      f"n_action_steps={cfg.get('n_action_steps')} 去噪={cfg.get('num_inference_steps')} "
      f"state={state_dim} action={act_dim} use_peft={cfg.get('use_peft')} 相机={','.join(imgs)}")
assert cfg.get("type") == "pi05", "不是 pi05 权重"
assert imgs == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"], f"相机键不符：{imgs}"
assert cfg.get("dtype") == "bfloat16", f"权重精度不是 bfloat16：{cfg.get('dtype')}"
assert cfg.get("chunk_size") == 50 and cfg.get("n_action_steps") == 50, "chunk/n_action_steps 不是 50"
assert cfg.get("num_inference_steps") == 10, "去噪步数不是 10"
assert state_dim == 32 and act_dim == 7, f"state/action 维度不符：{state_dim}/{act_dim}"
if cfg.get("use_peft"):
    print("   ⚠️  config.json 里 use_peft=true（合并时未改），本次用 --policy.use_peft=false 覆盖")
PY

# 权重完整性（可选，VERIFY_SHA=1 时校验，8.3GB 约 10 秒）：与 232 上导出的 sha256 对齐
if [ "$VERIFY_SHA" = "1" ]; then
  echo "🔐 校验 model.safetensors sha256..."
  echo "$EXPECT_SHA256  $MODEL_PATH/model.safetensors" | sha256sum -c - || {
    echo "❌ 权重 sha256 不匹配，拒绝上机"; exit 1; }
fi

if [ "$LITE_LOAD" = "1" ] && [ ! -f "$LITE_WRAPPER" ]; then
  echo "❌ 缺少 lite 包装器：$LITE_WRAPPER（或设 LITE_LOAD=0）"
  exit 1
fi

b601_check_trace_wrapper
b601_check_can
b601_check_cameras
b601_check_calibration

# ==================== 推理参数总览 ====================
echo ""
echo "=========================================="
echo "推理参数总览（pi05 / coffee_expo 合并权重）"
echo "=========================================="
echo "模型路径：$MODEL_PATH"
echo "权重类型：合并后的完整 bf16 权重（LoRA r=64/α=8 已并入 pi05_base）"
echo "机器人类型：seeed_b601_rs_follower（can0）"
echo "策略：base（纯推理，不录制、不保存数据）"
echo "推理后端：$INFERENCE_TYPE"
if [ "$INFERENCE_TYPE" = "rtc" ]; then
  echo "RTC 参数：queue_threshold=$RTC_QUEUE_THRESHOLD / execution_horizon=$RTC_EXECUTION_HORIZON"
fi
echo "精度 / 去噪步数：$POLICY_DTYPE / $NUM_INFERENCE_STEPS"
echo "显存包装器：$( [ "$LITE_LOAD" = "1" ] && echo "开启（跳过 lm_head + 分段计时）" || echo "关闭" )"
echo "视觉塔精度：$( [ "$LITE_VISION_BF16" = "1" ] && echo "bf16" || echo "fp32" )"
b601_print_run_params
b601_print_robot_notes
b601_print_trace_summary
echo "相机重命名：front→base_0_rgb  hand→left_wrist_0_rgb  top→right_wrist_0_rgb（与训练一致）"
echo "任务：$TASK_ID（本地 prompt 下标 $TASK_INDEX，不是数据集 task_index）"
echo "任务文本：$TASK_DESCRIPTION"
echo "录制/保存数据：❌ 否（base 策略不录制）"
echo "=========================================="
echo ""

b601_confirm_zero
b601_start_run "开始推理（pi05 coffee_expo / base 模式）..." "$RUN_TAG"

if [ "$LITE_LOAD" = "1" ]; then
  ENTRY_WRAPPER="$LITE_WRAPPER"
else
  ENTRY_WRAPPER=""
fi
export ENTRY_WRAPPER LITE_LOAD LITE_TRACE LITE_VISION_BF16

RTC_ARGS=()
if [ "$INFERENCE_TYPE" = "rtc" ]; then
  RTC_ARGS+=(
    --inference.queue_threshold="$RTC_QUEUE_THRESHOLD"
    --inference.rtc.execution_horizon="$RTC_EXECUTION_HORIZON"
  )
fi

b601_run_rollout \
  --strategy.type=base \
  --inference.type="$INFERENCE_TYPE" \
  "${RTC_ARGS[@]}" \
  --policy.path="$MODEL_PATH" \
  --policy.use_peft=false \
  --policy.dtype="$POLICY_DTYPE" \
  --policy.num_inference_steps="$NUM_INFERENCE_STEPS"

# ==================== 结束 ====================
echo ""
echo "=========================================="
echo "✅ 推理结束（pi05 coffee_expo / base 模式）"
echo "=========================================="
if [ "$ACTION_TRACE" = "1" ]; then
  echo "动作追踪日志：$ACTION_TRACE_LOG"
fi
echo "完整日志：$LOG_FILE"
echo "=========================================="
