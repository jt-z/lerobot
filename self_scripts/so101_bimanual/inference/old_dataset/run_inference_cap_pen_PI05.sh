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
# 端口映射（2026-08-26 校准匹配实测）：
#   左从臂 = USB 序列号 5C82108837（当前枚举为 ttyACM4）
#   右从臂 = USB 序列号 5B61034841（当前枚举为 ttyACM2）
#   ttyACM3 = 残缺线（仅 3 个电机响应），建议拔掉不用
# 注意：ttyACM 编号随插拔顺序变化，故下面用 /dev/serial/by-id 稳定路径，
#       只要适配器与机械臂的物理接线不变就不会变
# 本脚本使用双臂从动（follower）模式，故取左右从臂
LEFT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82108837-if00"
RIGHT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00"

# ==================== 摄像头配置 ====================
# 左臂：2个摄像头（手部、顶部）。PI05 训练数据只用 3 路摄像头（左手/左顶/右手），
#       比 PI0 少一路，故去掉 front（/dev/video6）
# 摄像头编号（2026-08-26 实测，左臂顶部摄像头已换 USB 口/线）：
#   video1=左臂手部（icSpring 202404160005，MJPG 支持，稳定）
#   video2=左臂顶部（icSpring 无序列号，仅 YUYV；原 video0/video8）
#   video4=右臂手部（JYU2C 2607060，MJPG 支持，未变）
# 注意：编号随 USB 插拔顺序变化；video3/video8 是附加节点，不可用于取流
LEFT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/video1, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG}
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
# MODEL_PATH="/home/kf/LX/pai0/lerobot/outputs/train/pi0_date_lora/checkpoints/005000/pretrained_model"

# pi05（本机 LoRA 微调 checkpoint 003000，基座权重在 model_weights/lerobot--pi05_base）:
# MODEL_PATH="/home/kf/dev/lerobot/model_weights/PI05_3000/003000/pretrained_model"
MODEL_PATH="/home/kf/dev/lerobot/model_weights/pi05_20000steps_pretrained_model"

# ==================== 数据集配置 ====================
# 注意：lerobot-rollout 强制要求数据集名以 rollout_ 开头（见 rollout/context.py）
EVAL_DATASET_NAME="hellozjt/rollout_cap_pen_two_hand"
TASK_DESCRIPTION="Put the cap back on the pen on the table and place it in the pen holder"
EPISODE_TIME=50  # 推理时长（秒）
FPS=20

# ==================== 摄像头 Rename 映射 ====================
# PI05 模型输入为 base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb，
# 而 bi_so_follower 机器人输出为 left_top / left_hand / right_hand（与训练数据一致）。
# 必须通过 --rename_map 传入，否则 rollout 会用空映射覆盖 checkpoint 中的 rename_map。
RENAME_MAP='{
  "observation.images.left_top": "observation.images.base_0_rgb",
  "observation.images.left_hand": "observation.images.left_wrist_0_rgb",
  "observation.images.right_hand": "observation.images.right_wrist_0_rgb"
}'

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
for video in /dev/video1 /dev/video2 /dev/video4; do
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
echo "摄像头映射：$(echo $RENAME_MAP | tr -d '\n' | tr -s ' ')"
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
LOG_DIR="./infer_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inference_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 动作诊断：每个 tick 记录 模型预期动作(goal) / 实际发送指令(sent) / 电机实际位置(present)，
# 输出到 $LOG_DIR/action_diag_jt_follower_arm_left.csv 和 _right.csv（左右臂各一文件）
# 用于排查左右臂 12 关节的指令是否符合模型预期与电机实际状态。
# 不需要诊断时注释掉下一行即可关闭（关闭后无额外开销）。
rm -rf "$LOG_DIR/action_diag"
export LEROBOT_ACTION_DIAG="$LOG_DIR/action_diag"

# 清除旧的评估数据集缓存（可选）
# rm -rf ~/.cache/huggingface/lerobot/$EVAL_DATASET_NAME


# 构建推理命令（lerobot-rollout + episodic 策略，镜像 lerobot-record 的录制行为）
# 说明：
#   - episodic：录制 num_episodes 个 episode，每个最长 episode_time_s 秒，episode 间有 reset 阶段
#   - --dataset.single_task：推理时的文本条件（pi05 语言条件），必须与训练数据集任务一致
#   - bi_so_follower 的 per-arm cameras 会自动加 left_/right_ 前缀，与数据集 key 匹配
#   - --rename_map：PI05 模型输入 key 为 base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb，
#     把机器人输出的 left_top/left_hand/right_hand 映射过去（3 路摄像头，无 front）
#   - --duration 为总时长上限（保护）；episode 轮转由 episode_time_s 控制
lerobot-rollout \
  --strategy.type=episodic \
  --inference.type=rtc \
  --policy.path=$MODEL_PATH \
  --policy.num_inference_steps=10 \
  --robot.type=bi_so_follower \
  --robot.id=jt_follower_arm \
  --robot.left_arm_config.port=$LEFT_FOLLOWER_PORT \
  --robot.right_arm_config.port=$RIGHT_FOLLOWER_PORT \
  --robot.left_arm_config.cameras="$LEFT_CAMERAS" \
  --robot.right_arm_config.cameras="$RIGHT_CAMERAS" \
  --robot.left_arm_config.max_relative_target=40.0 \
  --robot.right_arm_config.max_relative_target=40.0 \
  --dataset.repo_id=$EVAL_DATASET_NAME \
  --dataset.num_episodes=1 \
  --dataset.single_task="$TASK_DESCRIPTION" \
  --dataset.episode_time_s=$EPISODE_TIME \
  --dataset.reset_time_s=10 \
  --dataset.fps=$FPS \
  --fps=$FPS \
  --interpolation_multiplier=4 \
  --dataset.video=true \
  --dataset.rgb_encoder.vcodec=h264 \
  --dataset.push_to_hub=true \
  --duration=$EPISODE_TIME \
  --display_data=true \
  --display_compressed_images=false \
  --rename_map="$RENAME_MAP" 2>&1 | tee "$LOG_FILE"

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
