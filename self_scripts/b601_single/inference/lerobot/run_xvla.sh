#!/bin/bash
# 单 B601-RS 单臂 X-VLA 推理（lerobot-rollout + sync）
# 任务：单臂端起纸杯放上咖啡机托盘并按下按钮（与采集/微调数据集的任务文本一致）
#
# 前置：先用 self_scripts/b601_single/train/xvla/train_xvla_single_b601.sh
#       在单臂 B601 数据上完成 Phase II 微调，得到 MODEL_PATH。
#       xvla-base 本体无法直接控制 B601（没见过 Seeed/B601 域，且默认输出 20 维双臂末端位姿）。
#
# 与 SmolVLA/ACT 推理脚本的差异：
#   - 语言提示由 florence-2 的语言分支经 facebook/bart-large tokenizer 编码，
#     离线需要该 tokenizer 缓存（Florence-2 主干权重已包含在 checkpoint 内，无需另下）
#   - 动作维度：checkpoint 训练时 action_mode=auto（7 维），推理时自动裁回 7 维，无需手动指定
#   - X-VLA 不支持 RTC，必须 --inference.type=sync
#
# 创建日期：2026-09-10

set -e
set -o pipefail

# ==================== 环境检查 ====================
if ! command -v lerobot-rollout >/dev/null 2>&1; then
  echo "未找到 lerobot-rollout，尝试激活 lerobot conda 环境..."
  for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      source "$base/etc/profile.d/conda.sh"
      conda activate lerobot
      break
    fi
  done
fi
if ! command -v lerobot-rollout >/dev/null 2>&1; then
  echo "❌ 错误：lerobot-rollout 不可用，请手动执行: conda activate lerobot"
  exit 1
fi
echo "✅ lerobot-rollout 可用"

echo "=========================================="
echo "单臂模型推理（X-VLA）- B601-RS 单臂"
echo "=========================================="
echo ""

# ==================== 模型配置 ====================
# ❗ 微调产出的 pretrained_model 目录（lerobot-train 输出 <OUTPUT_DIR>/checkpoints/<step>/pretrained_model）
MODEL_PATH=""

# 训练用数据集（用于自动读取任务文本 + 核对 fps/相机 key）
TRAIN_DATASET_ROOT="$HOME/LX/pai0/b601_data/b601_20260910_164106"

# 任务文本（语言提示）：留空 = 自动从训练数据集 meta/tasks.parquet 读取（推荐，保证与训练一致）
# ⚠️ X-VLA/SmolVLA 都是语言条件模型，此文本必须与训练时完全相同，否则相当于换了指令
TASK_DESCRIPTION=""

# 去噪步数（config.json 默认 10，调小更快但动作质量可能下降；留空用 checkpoint 默认值）
NUM_DENOISING_STEPS=""

# ==================== 硬件配置 ====================
# 单 B601-RS 从臂 = can0（SocketCAN，1Mbps 经典 CAN）
B601_FOLLOWER_PORT="can0"
# 从臂校准 id（须与采集时一致 = self_scripts/b601_single/02_record.sh 的 FOLLOWER_ID）
FOLLOWER_ID="follower"

# 从臂方向修正：必须与采集（self_scripts/b601_single/02_record.sh 的 B601_JOINT_DIRECTIONS）完全一致，全 -1.0
# 原因：seeed_b601_rs_follower.send_action 对每个关节先乘 joint_directions（缺失关节默认 0.0
#       会把目标裁到 0 不动），且推理动作与训练动作必须同坐标系。
B601_JOINT_DIRECTIONS='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'

# 每 tick 相对动作幅度上限（度/步，安全帽）
MAX_RELATIVE_TARGET=30.0

# ==================== 摄像头配置 ====================
# 3 路相机（与 self_scripts/b601_single/02_record.sh 采集一致，相机 key 无 left_/right_ 前缀）：
#   hand = JYU2C 2608076（腕部）、front = JYU2C 2607031、top = JYU2C 2607060（顶部）
CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== 数据集/推理配置 ====================
# lerobot-rollout 强制数据集名以 rollout_ 开头（见 rollout/context.py）
EVAL_DATASET_NAME="b601_data/rollout_single_b601_make_coffee_xvla"
EPISODE_TIME=300   # 单集时长上限（秒）；采集时单集约 104 秒，留足余量
FPS=30             # 必须与训练数据集 fps 一致（30Hz）
PUSH_TO_HUB=false  # 外网不可达时保持 false，数据仅存本地

# ==================== 训练数据集信息（自动读取任务文本） ====================
if [ -z "$TASK_DESCRIPTION" ]; then
  if [ -f "$TRAIN_DATASET_ROOT/meta/tasks.parquet" ]; then
    TASK_DESCRIPTION=$(python -c "
import pandas as pd
t = pd.read_parquet('$TRAIN_DATASET_ROOT/meta/tasks.parquet')
print('|'.join([str(i) for i in t.index.tolist()]))
" 2>/dev/null || true)
  else
    echo "⚠️  找不到训练数据集 $TRAIN_DATASET_ROOT/meta/tasks.parquet，无法自动读取任务文本"
  fi
fi

# ==================== 推理前检查 ====================
echo "1. 检查模型权重..."
if [ -z "$MODEL_PATH" ]; then
  echo "❌ MODEL_PATH 为空：X-VLA 微调权重尚未就位。"
  echo "   请先运行 self_scripts/b601_single/train/xvla/train_xvla_single_b601.sh，"
  echo "   再把 <OUTPUT_DIR>/checkpoints/<step>/pretrained_model 路径填到本脚本顶部 MODEL_PATH。"
  exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 模型路径不存在：$MODEL_PATH"
  exit 1
fi
echo "✅ $MODEL_PATH"
echo "   类型: $(grep -o '"type": *"[a-z0-9_.]*"' "$MODEL_PATH/config.json" 2>/dev/null | head -1)"

echo ""
echo "2. 检查 B601 CAN..."
if ! ip link show $B601_FOLLOWER_PORT >/dev/null 2>&1; then
  echo "❌ $B601_FOLLOWER_PORT 不存在"; exit 1
fi
if ! ip link show $B601_FOLLOWER_PORT 2>/dev/null | grep -q "state UP"; then
  echo "⚠️  配置 $B601_FOLLOWER_PORT（1Mbps 经典 CAN）..."
  sudo ip link set $B601_FOLLOWER_PORT down 2>/dev/null || true
  sudo ip link set $B601_FOLLOWER_PORT type can bitrate 1000000
  sudo ip link set $B601_FOLLOWER_PORT up
fi
echo "✅ B601 CAN 已就绪"

echo ""
echo "3. 检查摄像头（3 路）..."
for cam in \
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0 \
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0 \
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0; do
  if [ ! -e "$cam" ]; then
    echo "❌ 摄像头不存在：$cam"
    exit 1
  else
    echo "✅ $cam"
  fi
done

echo ""
echo "4. 检查从臂校准文件..."
CALIB_FILE="$HOME/.cache/huggingface/lerobot/calibration/robots/seeed_b601_rs_follower/${FOLLOWER_ID}.json"
if [ -f "$CALIB_FILE" ]; then
  echo "✅ $CALIB_FILE"
else
  echo "❌ 缺失校准文件：$CALIB_FILE"
  echo "   请先运行 self_scripts/b601_common/02_follower_calibration.sh"
  exit 1
fi

# X-VLA 语言分支离线依赖 facebook/bart-large tokenizer（Florence-2 主干在 checkpoint 内）
echo ""
echo "5. 检查 X-VLA 资产缓存..."
XVLA_CACHE_DIR="$HOME/lerobot_weights/xvla"
BART_SNAPSHOT=""
for dir in "$XVLA_CACHE_DIR"/models--facebook--bart-large/snapshots/*/; do
  [ -d "$dir" ] && BART_SNAPSHOT="$dir" && break
done
if [ -n "$BART_SNAPSHOT" ] && [ -f "$BART_SNAPSHOT/tokenizer_config.json" ] && [ -f "$BART_SNAPSHOT/vocab.json" ]; then
  echo "✅ bart-large tokenizer 已缓存：$XVLA_CACHE_DIR"
else
  echo "❌ 缺少 facebook/bart-large tokenizer 缓存：$XVLA_CACHE_DIR"
  echo "   联网后执行（会自动落到该缓存目录）："
  echo "     export HF_HUB_CACHE=$XVLA_CACHE_DIR"
  echo "     hf download facebook/bart-large --include 'tokenizer*' --include 'vocab.json' --include 'merges.txt' --include 'special_tokens_map.json' --include 'added_tokens.json' --include 'config.json'"
  exit 1
fi

# ==================== 推理参数总览 ====================
echo ""
echo "=========================================="
echo "推理参数总览"
echo "=========================================="
echo "模型路径：$MODEL_PATH"
echo "机器人类型：seeed_b601_rs_follower（单臂，can0）"
echo "推理方式：sync（X-VLA 不支持 RTC）"
echo "评估数据集：$EVAL_DATASET_NAME"
echo "任务描述（文本提示）：${TASK_DESCRIPTION:-（空！语言分支将失效）}"
echo "推理时长：${EPISODE_TIME}秒 (约 $((EPISODE_TIME / 60)) 分钟)"
echo "推理频率：${FPS} Hz"
echo "摄像头：hand / front / top（3 路）"
echo "B601 方向修正：全 -1.0（与采集一致）"
echo "相对动作上限：$MAX_RELATIVE_TARGET 度/步"
echo "上传到Hub：$PUSH_TO_HUB"
echo "=========================================="
echo ""

if [ -z "$TASK_DESCRIPTION" ]; then
  echo "⚠️  任务文本为空：训练数据集里的 task 为空字符串（采集时 TASK_DESCRIPTION 未填），"
  echo "    推理也传空串与训练一致，但语言条件无效；建议补采带任务文本的数据后重训。"
  echo ""
fi

read -p "确认 B601 从臂处于零位、周边安全，按 ENTER 开始推理，Ctrl+C 取消..." dummy

# ==================== 开始推理 ====================
echo ""
echo "🚀 开始单臂模型推理（X-VLA）..."
echo ""

# Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

# ==== X-VLA 完全离线加载（不访问 Hugging Face）====
# 1) HF_HUB_CACHE 指向训练时使用的缓存目录（含 bart-large tokenizer 与 xvla-base）
# 2) 强制 offline，避免联网 HEAD 请求超时（外网不可达时会卡在 Retry 重试）
export HF_HUB_CACHE="$XVLA_CACHE_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inference_single_b601_xvla_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 动作诊断：每个 tick 记录 模型预期动作(goal) / 实际发送指令(sent) / 电机实际位置(present)
# （B601 侧是否有该 hook 视插件实现而定，无输出则说明插件未接；不影响推理）
rm -rf "$LOG_DIR/action_diag"
export LEROBOT_ACTION_DIAG="$LOG_DIR/action_diag"

# 构建推理命令（lerobot-rollout + episodic + sync）
# 说明：
#   - episodic：录制 num_episodes 个 episode，每个最长 episode_time_s 秒
#   - --dataset.single_task 是语言条件文本提示（X-VLA 必需），与训练任务一致
#   - 相机 key（hand/front/top）与数据集一致，无需 rename_map
#   - 动作 7 维由 checkpoint 的 action_mode=auto 自动裁剪，无需传 --policy.* 动作参数
lerobot-rollout \
  --strategy.type=episodic \
  --inference.type=sync \
  --policy.path="$MODEL_PATH" \
  ${NUM_DENOISING_STEPS:+--policy.num_denoising_steps=$NUM_DENOISING_STEPS} \
  --robot.type=seeed_b601_rs_follower \
  --robot.id=$FOLLOWER_ID \
  --robot.port=$B601_FOLLOWER_PORT \
  --robot.can_adapter=socketcan \
  --robot.gravity_compensation=true \
  --robot.joint_directions="$B601_JOINT_DIRECTIONS" \
  --robot.cameras="$CAMERAS" \
  --robot.max_relative_target=$MAX_RELATIVE_TARGET \
  --dataset.repo_id=$EVAL_DATASET_NAME \
  --dataset.num_episodes=1 \
  --dataset.single_task="$TASK_DESCRIPTION" \
  --dataset.episode_time_s=$EPISODE_TIME \
  --dataset.reset_time_s=10 \
  --dataset.fps=$FPS \
  --fps=$FPS \
  --dataset.video=true \
  --dataset.rgb_encoder.vcodec=h264 \
  --dataset.push_to_hub=$PUSH_TO_HUB \
  --duration=$EPISODE_TIME \
  --display_data=true \
  --display_compressed_images=false 2>&1 | tee "$LOG_FILE"

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ 单臂推理完成（X-VLA）"
echo "=========================================="
echo "评估数据集：$EVAL_DATASET_NAME"
if [ "$PUSH_TO_HUB" = "true" ]; then
  echo "Hugging Face Hub 链接："
  echo "  https://huggingface.co/datasets/$EVAL_DATASET_NAME"
else
  echo "（PUSH_TO_HUB=false，数据仅保存在本地）"
  echo "本地数据目录：$HOME/.cache/huggingface/lerobot/$EVAL_DATASET_NAME*"
fi
echo "完整日志：$LOG_FILE"
echo "=========================================="
