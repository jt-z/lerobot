#!/bin/bash
# 双臂协同咖啡机推理脚本：左臂夹取杯子，右臂开关倒咖啡，倒满后右臂关闭开关，左臂将咖啡杯放下
# 任务：Pick up the cup with the left arm, turn the switch on with the right arm to pour coffee,
#       turn the switch off when the cup is full, and place the cup down with the left arm
# 创建日期：2026-08-10
#
# 使用训练好的双臂模型进行推理评估

set -e  # 遇到错误立即退出
set -o pipefail  # 管道中任一命令失败则整体失败（配合 tee 使用）

echo "=========================================="
echo "双臂模型推理 - 双臂协同咖啡机（倒咖啡）"
echo "=========================================="
echo ""

# ==================== 硬件配置 ====================
# 当前端口映射（端口号映射.txt，2026-08-20 更新）：
#   ttyacm0=左主臂  ttyacm3=右从臂  ttyacm1=左从臂  ttyacm2=（旧映射，已过时）
# 本脚本使用双臂从动（follower）模式，故取左右从臂
LEFT_FOLLOWER_PORT="/dev/ttyACM2"
RIGHT_FOLLOWER_PORT="/dev/ttyACM3"

# ==================== 摄像头配置 ====================
# 左臂：3个摄像头（手部、顶部、前视）
LEFT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# 右臂：1个摄像头（手部）
# 注意：/dev/video4 硬件只支持 30fps，必须与硬件一致，否则 lerobot 校验失败
RIGHT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== 模型配置 ====================

# ACT :
# MODEL_PATH="/home/jt/dev/lerobot/output_lerobot_train/two_hand/checkpoints_037500/pretrained_model"
# MODEL_PATH="/home/jt/dev/lerobot/output_lerobot_train/two_hand/checkpoints_200000/pretrained_model"

# smolvla :
# MODEL_PATH="/home/jt/dev/lerobot/output_lerobot_train/two_hand/smolvla/300k_checkpoint_pretrained_model"

# pi0（本机训练，最优 checkpoint 005000）:
MODEL_PATH="/home/kf/dev/lerobot/model_weights/pi0_date_lora/005000/pretrained_model"


# ==================== 数据集配置 ====================
# 注意：lerobot-rollout 强制要求数据集名以 rollout_ 开头（见 rollout/context.py）
EVAL_DATASET_NAME="hellozjt/rollout_cap_pen_two_hand"
TASK_DESCRIPTION="Put the cap back on the pen on the table and place it in the pen holder"
EPISODE_TIME=200  # 推理时长（秒）
FPS=20

# ==================== 推理前检查 ====================
echo "1. 检查硬件连接..."

# 检查串口
for port in $LEFT_FOLLOWER_PORT $RIGHT_FOLLOWER_PORT; do
  if [ ! -e "$port" ]; then
    echo "❌ 错误：串口不存在 $port"
    echo "请检查硬件连接和串口映射"
    exit 1
  else
    echo "✅ $port 已连接"
  fi
done

# 检查摄像头
echo ""
echo "2. 检查摄像头..."
for video in /dev/video0 /dev/video2 /dev/video4 /dev/video6; do
  if [ ! -e "$video" ]; then
    echo "❌ 警告：摄像头不存在 $video"
  else
    echo "✅ $video 已连接"
  fi
done

# 检查模型文件
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
echo "推理参数总览"
echo "=========================================="
echo "模型路径：$MODEL_PATH"
echo "评估数据集：$EVAL_DATASET_NAME"
echo "任务描述：$TASK_DESCRIPTION"
echo "推理时长：${EPISODE_TIME}秒 (约 $((EPISODE_TIME / 60)) 分钟)"
echo "推理频率：${FPS} Hz"
echo "录制视频：✅ 是"
echo "上传到Hub：✅ 是"
echo "=========================================="
echo ""

# 确认开始
read -p "按 ENTER 开始推理，按 Ctrl+C 取消..." dummy

# ==================== 开始推理 ====================
echo ""
echo "🚀 开始双臂模型推理..."
echo ""

# 设置 Rerun 缓冲区大小（解决 gRPC transport error）
# 默认 8KB 太小，4 路摄像头每帧约 3.7MB，增大到 10MB
export RERUN_FLUSH_NUM_BYTES=10000000

# 设置 Rerun 内存限制为 30%（解决 1000 帧限制问题）
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

# 推理日志文件（输出同时显示在终端并写入此文件，便于事后查看）
LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inference_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 清除旧的评估数据集缓存（可选）
# rm -rf ~/.cache/huggingface/lerobot/$EVAL_DATASET_NAME


# 构建推理命令（lerobot-rollout + episodic 策略，镜像 lerobot-record 的录制行为）
# 说明：
#   - episodic：录制 num_episodes 个 episode，每个最长 episode_time_s 秒，episode 间有 reset 阶段
#   - --dataset.single_task：推理时的文本条件（pi0 语言条件），必须与训练数据集任务一致
#   - bi_so_follower 的 per-arm cameras 会自动加 left_/right_ 前缀，与数据集 key 匹配
#   - --duration 为总时长上限（保护）；episode 轮转由 episode_time_s 控制
lerobot-rollout \
  --strategy.type=episodic \
  --inference.type=rtc \
  --policy.path=$MODEL_PATH \
  --policy.num_inference_steps=6 \
  --robot.type=bi_so_follower \
  --robot.id=jt_follower_arm \
  --robot.left_arm_config.port=$LEFT_FOLLOWER_PORT \
  --robot.right_arm_config.port=$RIGHT_FOLLOWER_PORT \
  --robot.left_arm_config.cameras="$LEFT_CAMERAS" \
  --robot.right_arm_config.cameras="$RIGHT_CAMERAS" \
  --robot.left_arm_config.max_relative_target=20.0 \
  --robot.right_arm_config.max_relative_target=20.0 \
  --dataset.repo_id=$EVAL_DATASET_NAME \
  --dataset.num_episodes=1 \
  --dataset.single_task="$TASK_DESCRIPTION" \
  --dataset.episode_time_s=$EPISODE_TIME \
  --dataset.reset_time_s=10 \
  --dataset.fps=$FPS \
  --fps=$FPS \
  --dataset.video=true \
  --dataset.rgb_encoder.vcodec=auto \
  --dataset.push_to_hub=true \
  --duration=$EPISODE_TIME \
  --display_data=true \
  --display_compressed_images=false 2>&1 | tee "$LOG_FILE"

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ 双臂推理完成！"
echo "=========================================="
echo "评估数据集：$EVAL_DATASET_NAME"
echo "Hugging Face Hub 链接："
echo "  https://huggingface.co/datasets/$EVAL_DATASET_NAME"
echo "完整日志：$LOG_FILE"
echo ""
echo "💡 提示："
echo "  1. 访问上述链接查看推理视频和轨迹"
echo "  2. 分析模型性能和成功率"
echo "  3. 如需重新推理，直接再次运行此脚本"
echo "=========================================="
