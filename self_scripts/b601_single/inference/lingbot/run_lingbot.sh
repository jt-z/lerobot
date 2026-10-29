#!/bin/bash
# 单 B601-RS LingBot-VA 推理（base 策略：纯推理，不建数据集、不录帧、不落盘）
# 任务：单臂端起纸杯放上咖啡机托盘、拿方块按按钮等 4 秒、方块放回桌面、再把杯子端回桌面
#       （任务文本/相机/fps/方向修正与 self_scripts/b601_single/02_record.sh 采集完全一致）
#
# 权重形态（别按 ACT/SmolVLA 的习惯套）：
#   MODEL_PATH 指向 LoRA adapter 目录（14 文件 / 168 MB），**没有 model.safetensors**；
#   基座与 VAE/UMT5/tokenizer 由 checkpoint 内的 json 指到 /home/kf/LX/pai0/lingbot/ 下两处，
#   详见同目录 env.sh 顶部注释与 README。
#
# 用法：
#   ./run_lingbot.sh                                  # 默认 050000，跑到 Ctrl+C
#   LINGBOT_STEP=050000 ./run_lingbot.sh              # 指定步数（目前只有 050000 一个 ckpt）
#   DURATION=20 ./run_lingbot.sh                      # 第一次上机：先跑 20 秒
#   DRY=1 DURATION=10 ./run_lingbot.sh                # 空跑：不碰 CAN/相机，验 flag 全链路（会停在连接 can99）
#   VIDEO_STEPS=10 ACTION_STEPS=10 GUIDANCE=1.0 ./run_lingbot.sh   # 快档（延迟约降到 1/4，动作质量待验）
#
# 环境/硬件检查/参数总览/命令拼装等公共逻辑见 ../lerobot/_common.sh

set -e
set -o pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

# _common.sh 用 `DURATION=0` 这种硬赋值设默认值，会把命令行上显式给的同名变量盖掉，
# 所以先存下来、source 完再恢复（否则 `DURATION=20 ./run_lingbot.sh` 静默变成不限时长）。
_USER_DURATION="${DURATION-}"
_USER_FPS="${FPS-}"
_USER_ACTION_TRACE="${ACTION_TRACE-}"

source "$HERE/env.sh"
source "$HERE/../lerobot/_common.sh"

for _v in DURATION FPS ACTION_TRACE; do
  _u="_USER_$_v"
  [ -n "${!_u}" ] && printf -v "$_v" '%s' "${!_u}"
done

# ==================== 模型配置 ====================
MODEL_PATH="$LINGBOT_CKPT"
STEP="$LINGBOT_STEP"

# lingbot_va 没有实现 RTC 语义（predict_action_chunk 不接受 inference_delay/prev_chunk_left_over），
# 传 rtc 会在 context 构建阶段直接报错 → 必须 sync。
# sync 下换块那一拍会阻塞控制环（实测见 README），这也是真机上会"一顿一顿"的唯一原因。
INFERENCE_TYPE="sync"

# 去噪步数（延迟/质量权衡，默认 = config.json 原值）
VIDEO_STEPS="${VIDEO_STEPS:-$(${LINGBOT_PY} -c "import json;print(json.load(open('$MODEL_PATH/config.json'))['num_inference_steps'])")}"
ACTION_STEPS="${ACTION_STEPS:-$(${LINGBOT_PY} -c "import json;print(json.load(open('$MODEL_PATH/config.json'))['action_num_inference_steps'])")}"
GUIDANCE="${GUIDANCE:-$(${LINGBOT_PY} -c "import json;print(json.load(open('$MODEL_PATH/config.json'))['guidance_scale'])")}"

ATTN_MODE="$LINGBOT_ATTN_MODE"
CAMERA_RENAME_MAP="$LINGBOT_CAM_RENAME_MAP"

TITLE="单臂推理（LingBot-VA / $STEP）- B601-RS（make coffee / base 纯推理）"
RUN_TAG="make_coffee_lingbot_${STEP}"
ACTION_TRACE_LOG="$HOME/dev/lerobot/self_scripts/_logs/action_lingbot_test.log"

# DRY=1：不碰硬件，只验 flag 链路与策略加载（rollout 会停在连接 can99，属预期）
if [ "${DRY:-0}" = "1" ]; then
  B601_FOLLOWER_PORT="can99"
  DURATION="${DURATION:-10}"
fi

# ==================== 推理前检查 ====================
b601_require_lerobot
b601_banner "$TITLE"

# 0) 任务文本必须与训练数据集一致（lingbot 是视频世界模型，文本进了 cross-attn，改一个字都是分布外）
if [ "$LINGBOT_TASK" != "$TASK_DESCRIPTION" ]; then
  echo "❌ 任务文本不一致："
  echo "   env.sh   LINGBOT_TASK   : $LINGBOT_TASK"
  echo "   _common  TASK_DESCRIPTION: $TASK_DESCRIPTION"
  echo "   两者必须逐字相同（= 训练数据集 b601_repair_clean_v2/meta/tasks.parquet 里那条）"
  exit 1
fi
TASK_DESCRIPTION="$LINGBOT_TASK"

# 1) 三段权重 + checkpoint 内部路径 + rename_map 一致性 + 依赖（纯读文件，秒级）
"${LINGBOT_PY}" "$HERE/preflight.py" --ckpt "$MODEL_PATH" --base "$LINGBOT_BASE" \
  --wan "$LINGBOT_WAN" --rename-map "$CAMERA_RENAME_MAP"
echo ""

# 2) 动作追踪包装器
b601_check_trace_wrapper

# 3) 显存（π0.5 server 常驻会吃掉 8~9 GB，lingbot 需要约 13 GB，不够会 OOM）
if command -v nvidia-smi >/dev/null 2>&1; then
  FREE=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F", " '{print $1-$2}' | sort -n | head -1)
  echo ""
  echo "显存空闲：${FREE} MiB（lingbot 峰值约 13 GB）"
  if [ "${FREE:-0}" -lt 14000 ]; then
    echo "⚠️  空闲显存不足 14 GB —— 若加载阶段 OOM，先停掉常驻的 π0.5 server："
    echo "   for p in \$(pgrep -f 'openpi_env/bin/python .*serve_polic[y]\\.py'); do kill \$p; done"
  fi
fi

# 4) 硬件（DRY 模式跳过：can99 不存在是预期）
if [ "${DRY:-0}" = "1" ]; then
  echo ""
  echo "🧪 DRY 模式：跳过 CAN/相机检查，robut 端口用 can99（物理上不可能动到机械臂）"
else
  b601_check_can
  b601_check_cameras
  b601_check_calibration
fi

# ==================== 推理参数总览 ====================
LORA_INFO=$("${LINGBOT_PY}" - "$MODEL_PATH" <<'PYEOF'
import json, os, sys
try:
    d = json.load(open(os.path.join(sys.argv[1], "adapter_config.json")))
    print("r=%s / alpha=%s" % (d.get("r"), d.get("lora_alpha")))
except Exception as exc:
    print("(读 adapter_config.json 失败：%s)" % exc)
PYEOF
)
echo ""
echo "=========================================="
echo "推理参数总览（LingBot-VA / LoRA）"
echo "=========================================="
echo "训练步数：$STEP（产物来自 A800-57，数据集 b601_repair_clean_v2 / 497 集）"
echo "模型路径：$MODEL_PATH"
echo "权重类型：LoRA adapter（$LORA_INFO）+ LeRobot 基座 lingbot_va_base（5B transformer）"
echo "冻结模块：$LINGBOT_WAN（Wan2.2 VAE + UMT5-XXL/tokenizer，文本编码器跑 CPU）"
echo "机器人类型：seeed_b601_rs_follower（单臂 B601-RS，$B601_FOLLOWER_PORT）"
echo "策略：base（纯推理，不录制、不保存数据）"
echo "推理方式：sync（lingbot_va 不支持 RTC；换块时阻塞控制环）"
echo "注意力后端：$ATTN_MODE（ckpt 里存的是训练用 flex，推理改 SDPA）"
echo "去噪步数：视频 $VIDEO_STEPS / 动作 $ACTION_STEPS ｜ CFG $GUIDANCE（>1 时每次前向双 batch）"
b601_print_run_params
echo "相机重命名：front→cam_high  hand→cam_left_wrist  top→cam_right_wrist（与训练一致）"
echo "动作块：每块 $(${LINGBOT_PY} -c "import json;d=json.load(open('$MODEL_PATH/config.json'));print(d['frame_chunk_size']*d['action_per_frame'])") 个动作（首块少一帧）"
b601_print_robot_notes
b601_print_trace_summary "model/sent/cmd"
echo "录制/保存数据：❌ 否（base 策略不录制）"
echo "=========================================="
echo ""

if [ "${DRY:-0}" = "1" ]; then
  echo "（DRY 模式跳过 ENTER 确认）"
else
  b601_confirm_zero
fi
b601_start_run "开始单臂模型推理（LingBot-VA $STEP）..." "$RUN_TAG"

# 离线模式：VAE / UMT5 / tokenizer 全走本地目录，禁止再请求 huggingface.co
b601_export_offline_hf "$HF_HUB_CACHE"
export LINGBOT_TASK

b601_run_rollout \
  --strategy.type=base \
  --inference.type="$INFERENCE_TYPE" \
  --policy.path="$MODEL_PATH" \
  --policy.device=cuda \
  --policy.attn_mode="$ATTN_MODE" \
  --policy.num_inference_steps="$VIDEO_STEPS" \
  --policy.action_num_inference_steps="$ACTION_STEPS" \
  --policy.guidance_scale="$GUIDANCE"

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ 单臂推理结束（LingBot-VA $STEP / base 模式）"
echo "=========================================="
echo "本次未保存任何推理数据（base 策略不录制、不落盘）。"
if [ "$ACTION_TRACE" = "1" ]; then
  echo "动作追踪日志：$ACTION_TRACE_LOG"
  echo "  抖动/起步判读：python $HERE/../tools/analyze_rollout.py --log $ACTION_TRACE_LOG"
fi
echo "完整日志：$LOG_FILE"
echo "=========================================="
