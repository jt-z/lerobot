#!/bin/bash
# 数据采集脚本：笔帽盖回笔并放入笔筒
# 任务：Put the cap back on the pen on the table and place it in the pen holder
# 创建日期：2026-08-03
#
# 支持中断继续录制：
#   如果录制过程中断，可以添加 --resume 参数继续录制
#   脚本会自动从上次中断的 episode 继续

set -e  # 遇到错误立即退出

# ==================== 命令行参数 ====================
# 检查是否传入 --resume 参数
RESUME_MODE=false
if [ "$1" == "--resume" ]; then
  RESUME_MODE=true
  echo "🔄 恢复模式：将从上次中断处继续录制"
fi

echo "=========================================="
echo "双臂数据采集 - 笔帽盖回笔并放入笔筒"
echo "=========================================="
echo ""

# ==================== 硬件配置 ====================
LEFT_FOLLOWER_PORT="/dev/ttyLeftFollower"
RIGHT_FOLLOWER_PORT="/dev/ttyRightFollower"
LEFT_LEADER_PORT="/dev/ttyLeftLeader"
RIGHT_LEADER_PORT="/dev/ttyRightLeader"

# ==================== 摄像头配置 ====================
# 左臂：3个摄像头（手部、顶部、前视）
LEFT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# 右臂：1个摄像头（手部）
RIGHT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 20, fourcc: MJPG}
}'

# ==================== 数据集配置 ====================
DATASET_NAME="hellozjt/cap_pen_and_put_into_holder"
TASK_DESCRIPTION="Put the cap back on the pen on the table and place it in the pen holder"
NUM_EPISODES=40
EPISODE_TIME=45  # 每个 episode 录制时长（秒）- 75秒 × 20fps = 1500帧
RESET_TIME=20    # 重置环境时长（秒）
FPS=20

# ==================== 采集前检查 ====================
echo "1. 检查硬件连接..."

# 检查串口
for port in $LEFT_FOLLOWER_PORT $RIGHT_FOLLOWER_PORT $LEFT_LEADER_PORT $RIGHT_LEADER_PORT; do
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

# 检查校准文件
echo ""
echo "3. 检查校准文件..."
CALIB_FOLLOWER_DIR="$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower"
CALIB_LEADER_DIR="$HOME/.cache/huggingface/lerobot/calibration/teleoperators/so_leader"

if [ -d "$CALIB_FOLLOWER_DIR" ] && [ -d "$CALIB_LEADER_DIR" ]; then
  echo "✅ 校准文件目录存在"
  echo "   Follower: $(ls $CALIB_FOLLOWER_DIR | grep jt_follower_arm | wc -l) 个文件"
  echo "   Leader: $(ls $CALIB_LEADER_DIR | grep jt_leader_arm | wc -l) 个文件"
else
  echo "⚠️  校准文件目录不完整，首次运行时会提示校准"
fi

# ==================== 采集参数总览 ====================
echo ""
echo "=========================================="
echo "采集参数总览"
echo "=========================================="
echo "数据集名称：$DATASET_NAME"
echo "任务描述：$TASK_DESCRIPTION"
echo "Episode 数量：$NUM_EPISODES"
echo "每 Episode 时长：${EPISODE_TIME}秒"
echo "重置时长：${RESET_TIME}秒"
echo "采集频率：${FPS} Hz"
echo "预计总时长：约 $((($EPISODE_TIME + $RESET_TIME) * $NUM_EPISODES / 60)) 分钟"
echo "=========================================="
echo ""

# 确认开始
read -p "按 ENTER 开始数据采集，按 Ctrl+C 取消..." dummy

# ==================== 开始采集 ====================
echo ""
echo "🚀 开始数据采集..."
echo ""

# 设置 Rerun 缓冲区大小（解决 gRPC transport error）
# 默认 8KB 太小，4 路摄像头每帧约 3.7MB，增大到 10MB
export RERUN_FLUSH_NUM_BYTES=10000000

# 设置 Rerun 内存限制为 30%（解决 1000 帧限制问题）
# 默认 10% 在约 1000 帧时触达限制，导致录制提前终止
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

# 构建命令
RECORD_CMD="lerobot-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=$LEFT_FOLLOWER_PORT \
  --robot.right_arm_config.port=$RIGHT_FOLLOWER_PORT \
  --robot.id=jt_follower_arm \
  --robot.left_arm_config.cameras=\"$LEFT_CAMERAS\" \
  --robot.right_arm_config.cameras=\"$RIGHT_CAMERAS\" \
  --robot.left_arm_config.max_relative_target=20.0 \
  --robot.right_arm_config.max_relative_target=20.0 \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=$LEFT_LEADER_PORT \
  --teleop.right_arm_config.port=$RIGHT_LEADER_PORT \
  --teleop.id=jt_leader_arm \
  --dataset.repo_id=$DATASET_NAME \
  --dataset.num_episodes=$NUM_EPISODES \
  --dataset.single_task=\"$TASK_DESCRIPTION\" \
  --dataset.fps=$FPS \
  --dataset.episode_time_s=$EPISODE_TIME \
  --dataset.reset_time_s=$RESET_TIME \
  --dataset.video=true \
  --dataset.vcodec=h264 \
  --dataset.push_to_hub=true \
  --display_data=true \
  --display_compressed_images=false"

# 如果是恢复模式，添加 --resume 参数
if [ "$RESUME_MODE" = true ]; then
  RECORD_CMD="$RECORD_CMD --resume=true"
fi

# 执行命令
eval $RECORD_CMD

# ==================== 采集完成 ====================
echo ""
echo "=========================================="
echo "✅ 数据采集完成！"
echo "=========================================="
echo "数据集名称：$DATASET_NAME"
echo "Episode 数量：$NUM_EPISODES"
echo "Hugging Face Hub 链接："
echo "  https://huggingface.co/datasets/$DATASET_NAME"
echo ""
echo "💡 提示："
echo "  如果录制过程中断，可以使用以下命令继续："
echo "  ./collect_cap_pen_data.sh --resume"
echo ""
echo "下一步："
echo "  1. 访问上述链接查看数据集"
echo "  2. 检查数据质量"
echo "  3. 开始训练 ACT 模型"
echo "=========================================="
