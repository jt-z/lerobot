#!/bin/bash
# smolvla 推理测试脚本：对比测试 4k / 30k / 110k 三个 checkpoint 的效果
# 用法：
#   bash self_scripts/so101_bimanual/inference/old_dataset/run_inference_cap_pen_smolvla.sh 4k     # 测试 4k checkpoint
#   bash self_scripts/so101_bimanual/inference/old_dataset/run_inference_cap_pen_smolvla.sh 30k    # 测试 30k checkpoint
#   bash self_scripts/so101_bimanual/inference/old_dataset/run_inference_cap_pen_smolvla.sh 110k   # 测试 110k checkpoint
# 创建日期：2026-08-24
# 解码步数：smolvla 用 num_steps（flow matching），pi0 用 num_inference_steps（diffusion）
set -e
set -o pipefail

# ==================== 参数：选择 checkpoint ====================
CKPT="${1:-30k}"  # 默认 30k
case "$CKPT" in
  4k|30k|110k) ;;
  *) echo "❌ 用法: bash $0 {4k|30k|110k}"; exit 1 ;;
esac

echo "=========================================="
echo "smolvla 双臂模型推理 - 测试 checkpoint: ${CKPT}"
echo "=========================================="
echo ""

# ==================== 硬件配置 ====================
# 端口映射：ttyACM2=左从臂  ttyACM3=右从臂（与 pi0 推理脚本一致）
LEFT_FOLLOWER_PORT="/dev/ttyACM2"
RIGHT_FOLLOWER_PORT="/dev/ttyACM3"

# ==================== 摄像头配置 ====================
# 注意：smolvla 训练自 cap_pen_and_put_into_holder（ksa 机器采集），与旧映射一致，勿改动
LEFT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

RIGHT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== 模型配置 ====================
# smolvla checkpoint 路径（4k 结构特殊：文件直接在目录下，无 pretrained_model 子目录）
if [ "$CKPT" = "4k" ]; then
  MODEL_PATH="/home/kf/LX/pai0/smol_vla/4k"
else
  MODEL_PATH="/home/kf/LX/pai0/smol_vla/${CKPT}/pretrained_model"
fi

# smolvla 用 num_steps（flow matching 去噪步数）
NUM_STEPS_ARG="--policy.num_steps=6"

# ==================== 数据集配置 ====================
EVAL_DATASET_NAME="hellozjt/rollout_smolvla_cap_pen_${CKPT}"
TASK_DESCRIPTION="Put the cap back on the pen on the table and place it in the pen holder"
EPISODE_TIME=200  # 推理时长（秒）
FPS=20

# ==================== 推理前检查 ====================
echo "1. 检查硬件连接..."
for port in $LEFT_FOLLOWER_PORT $RIGHT_FOLLOWER_PORT; do
  if [ ! -e "$port" ]; then
    echo "❌ 错误：串口不存在 $port"
    exit 1
  else
    echo "✅ $port 已连接"
  fi
done

echo ""
echo "2. 检查摄像头..."
for video in /dev/video0 /dev/video2 /dev/video4 /dev/video6; do
  if [ ! -e "$video" ]; then
    echo "❌ 警告：摄像头不存在 $video"
  else
    echo "✅ $video 已连接"
  fi
done

echo ""
echo "3. 检查模型文件..."
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 错误：模型路径不存在 $MODEL_PATH"
  exit 1
else
  echo "✅ 模型文件存在"
  echo "   路径: $MODEL_PATH"
fi

# ==================== 推理参数总览 ====================
echo ""
echo "=========================================="
echo "推理参数总览（smolvla ${CKPT}）"
echo "=========================================="
echo "模型路径：$MODEL_PATH"
echo "评估数据集：$EVAL_DATASET_NAME"
echo "任务描述：$TASK_DESCRIPTION"
echo "推理时长：${EPISODE_TIME}秒 (约 $((EPISODE_TIME / 60)) 分钟)"
echo "推理频率：${FPS} Hz"
echo "=========================================="
echo ""

# 确认开始
read -p "按 ENTER 开始推理，按 Ctrl+C 取消..." dummy

# ==================== 开始推理 ====================
echo ""
echo "🚀 开始 smolvla ${CKPT} 推理..."
echo ""

# Rerun 环境变量（保留备用）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

# 推理日志文件
LOG_DIR="$HOME/LX/pai0/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inference_smolvla_${CKPT}_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

lerobot-rollout \
  --strategy.type=episodic \
  --inference.type=rtc \
  --policy.path=$MODEL_PATH \
  $NUM_STEPS_ARG \
  --robot.type=bi_so_follower \
  --robot.id=jt_follower_arm \
  --robot.left_arm_config.port=$LEFT_FOLLOWER_PORT \
  --robot.right_arm_config.port=$RIGHT_FOLLOWER_PORT \
  --robot.left_arm_config.cameras="$LEFT_CAMERAS" \
  --robot.right_arm_config.cameras="$RIGHT_CAMERAS" \
  --robot.left_arm_config.max_relative_target='{shoulder_pan: 50.0, shoulder_lift: 50.0, elbow_flex: 50.0, wrist_flex: 50.0, wrist_roll: 50.0, gripper: 50.0}' \
  --robot.right_arm_config.max_relative_target='{shoulder_pan: 50.0, shoulder_lift: 50.0, elbow_flex: 50.0, wrist_flex: 50.0, wrist_roll: 50.0, gripper: 50.0}' \
  --dataset.repo_id=$EVAL_DATASET_NAME \
  --dataset.num_episodes=1 \
  --dataset.single_task="$TASK_DESCRIPTION" \
  --dataset.episode_time_s=$EPISODE_TIME \
  --dataset.reset_time_s=10 \
  --dataset.fps=$FPS \
  --fps=$FPS \
  --dataset.video=true \
  --dataset.rgb_encoder.vcodec=h264 \
  --dataset.push_to_hub=false \
  --duration=$EPISODE_TIME \
  --display_data=false \
  --display_compressed_images=false 2>&1 | tee "$LOG_FILE"

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ smolvla ${CKPT} 推理完成！"
echo "=========================================="
echo "评估数据集：$EVAL_DATASET_NAME"
echo "Hugging Face Hub 链接："
echo "  https://huggingface.co/datasets/$EVAL_DATASET_NAME"
echo "完整日志：$LOG_FILE"
echo "=========================================="
