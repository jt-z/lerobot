#!/bin/bash
# 独立测试脚本：ACT 300k 模型双臂推理（笔帽盖回笔并放入笔筒）
# 任务：Put the cap back on the pen on the table and place it in the pen holder
#
# 说明：
#   - 使用 lerobot-rollout + sync 推理（ACT 不支持 RTC，见 rollout/context.py 校验）
#   - 与 self_scripts/so101_bimanual/inference/run_inference_make_coffee_ACT.sh 等其余双臂推理脚本相互独立
# 创建日期：2026-08-24

set -e  # 遇到错误立即退出
set -o pipefail  # 管道中任一命令失败则整体失败（配合 tee 使用）

echo "=========================================="
echo "ACT 300k 独立测试 - 双臂笔帽盖回笔"
echo "=========================================="
echo ""

# ==================== 硬件配置 ====================
# 端口映射（2026-08-26 实测，与 self_scripts/so101_bimanual/02_teleoperate.sh / 03_collect_make_coffee.sh 一致）：
#   左从臂 = USB 序列号 5C82108837
#   右从臂 = USB 序列号 5B61034841
# 本脚本使用双臂从动（follower）模式，故取左右从臂
# 注意：全部使用 /dev/serial/by-id 稳定路径，ttyACM 编号随插拔顺序变化不可靠。
LEFT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82108837-if00"
RIGHT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00"

# ==================== 摄像头配置 ====================
# 摄像头映射（2026-08-26 实测，与 self_scripts/so101_bimanual/02_teleoperate.sh / 03_collect_make_coffee.sh 一致）：
#   左臂手部 = icSpring 无序列号
#   左臂顶部 = icSpring 202404160005
#   右臂手部 = JYU2C-2083 2607060
#   前视     = JYU2C-2083 2607031
# 注意：/dev/videoN 编号随插拔顺序变化，故使用 /dev/v4l/by-id 稳定路径。
# 左臂：3个摄像头（手部、顶部、前视）
LEFT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# 右臂：1个摄像头（手部）
# 注意：JYU2C-2083 硬件只支持 30fps，必须与硬件一致，否则 lerobot 校验失败
RIGHT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== 模型配置 ====================
# ACT 300k（本机训练）
MODEL_PATH="/home/kf/dev/lerobot/model_weights/act_model_300k"

# ACT 单次前向解码，无 num_steps / num_inference_steps 参数，留空即可
NUM_STEPS_ARG=""

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
for video in \
  /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 \
  /dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0 \
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0 \
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0; do
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
echo "推理方式：sync（ACT 不支持 RTC）"
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
echo "🚀 开始 ACT 300k 模型推理..."
echo ""

# 设置 Rerun 缓冲区大小（解决 gRPC transport error）
# 默认 8KB 太小，4 路摄像头每帧约 3.7MB，增大到 10MB
export RERUN_FLUSH_NUM_BYTES=10000000

# 设置 Rerun 内存限制为 30%（解决 1000 帧限制问题）
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

# 推理日志文件（输出同时显示在终端并写入此文件，便于事后查看）
LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/act_300k_inference_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 构建推理命令（lerobot-rollout + sync 推理）
# 说明：
#   - --inference.type=sync：ACT 不支持 RTC，必须用 sync
#   - --strategy.type=episodic：录制 num_episodes 个 episode，每个最长 episode_time_s 秒
#   - --dataset.single_task：数据集文本条件，保持与训练任务一致
#   - bi_so_follower 的 per-arm cameras 会自动加 left_/right_ 前缀，与数据集 key 匹配
lerobot-rollout \
  --strategy.type=episodic \
  --inference.type=sync \
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
  --dataset.push_to_hub=true \
  --duration=$EPISODE_TIME \
  --display_data=true \
  --display_compressed_images=false 2>&1 | tee "$LOG_FILE"

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ ACT 300k 推理完成！"
echo "=========================================="
echo "评估数据集：$EVAL_DATASET_NAME"
echo "Hugging Face Hub 链接："
echo "  https://huggingface.co/datasets/$EVAL_DATASET_NAME"
echo "完整日志：$LOG_FILE"
echo "=========================================="
