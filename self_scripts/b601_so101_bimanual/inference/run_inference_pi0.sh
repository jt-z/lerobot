#!/bin/bash
# 异构双臂 pi0 推理：左 B601-RS（SocketCAN）+ 右 SO-101（串口）
# 任务：双臂端起纸杯放上咖啡机托盘、右臂按按钮等 4 秒、左臂放回桌面
#       （与 self_scripts/b601_so101_bimanual/02_record.sh 采集的任务描述/相机/方向修正完全一致）
#
# 参照：
#   - run_inference_b601_so101_make_coffee_ACT.sh：同硬件的 ACT 推理脚本（历史文件，未随本次重组入库，本脚本据其改写）
#   - self_scripts/b601_so101_bimanual/02_record.sh：硬件端口 / 相机 / fps / 方向修正 / 任务描述（采集基准）
# 机器人类型：bi_b601_so101_follower（自定义异构类，见 lerobot/src/lerobot/robots/bi_b601_so101_follower/）
# 训练数据集：b601_so101_20260902_165927（30Hz，13 关节=左7+右6，4 路相机 left_hand/left_top/left_front/right_hand）
# 测试模型：020000_b601_pi0（pi0 LoRA，20k checkpoint，num_inference_steps=10）
# 创建日期：2026-09-07

set -e  # 遇到错误立即退出
set -o pipefail  # 管道中任一命令失败则整体失败（配合 tee 使用）

echo "=========================================="
echo "异构双臂推理（pi0）- 左 B601 + 右 SO-101（倒咖啡）"
echo "=========================================="
echo ""

# ==================== 模型配置 ====================
# 本机训练的 pi0 权重（pretrained_model 目录）
MODEL_PATH="/home/kf/dev/lerobot/model_weights/020000_b601_pi0/pretrained_model"

# pi0 训练时用 10 步，RTC 已解决推理卡顿，保持 10 步保证动作质量
NUM_STEPS_ARG="--policy.num_inference_steps=10"

# ==================== 硬件配置 ====================
# 端口映射（与 self_scripts/b601_so101_bimanual/02_record.sh / 01_teleop_test.sh 一致，2026-09 实测）：
#   左 B601 从臂 = can0（SocketCAN，PEAK PCAN-USB，1Mbps 经典 CAN）
#   右 SO-101 从臂 = USB 序列号 5B61034841
# 推理只用从臂，无需主臂（teleop）端口。
B601_FOLLOWER_PORT="can0"
RIGHT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00"

# 设备 ID（校准文件名后缀拼接用，须与采集时一致 = self_scripts/b601_so101_bimanual/02_record.sh 的 FOLLOWER_ID）
FOLLOWER_ID="jt_follower_arm"

# ==================== 摄像头配置 ====================
# 4 路相机（与 self_scripts/b601_so101_bimanual/02_record.sh 采集一致，2026-09 实测）：
#   左 B601：hand=JYU2C 2608076、top=icSpring 202404160005、front=JYU2C 2607031
#   右 SO-101：hand=JYU2C 2607060
# 路径用 /dev/v4l/by-id 稳定路径（/dev/videoN 随插拔顺序变化不可靠）。
LEFT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

RIGHT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== 左 B601 方向修正 ====================
# 必须与采集（04 脚本 B601_JOINT_DIRECTIONS）完全一致，全 -1.0。
# 原因：seeed_b601_rs_follower.send_action 对每个关节先乘 joint_directions
#       （缺失关节默认 0.0，会把目标裁到 0 不动），且推理动作与训练动作必须同坐标系。
B601_JOINT_DIRECTIONS='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'

# 每 tick 相对动作幅度上限（度/步，安全帽）。镜像 b601_so101 make_coffee ACT 脚本用 30.0；
# 若 B601 左臂抖动/动作过大，可参照 self_scripts/b601_so101_bimanual/01_teleop_test.sh 注释里的逐关节更小值单独收紧左臂。
MAX_RELATIVE_TARGET=30.0

# ==================== 数据集配置 ====================
# lerobot-rollout 强制数据集名以 rollout_ 开头（见 rollout/context.py）
EVAL_DATASET_NAME="hellozjt/rollout_b601_so101_pi0_20k"
# 任务描述须与采集时（self_scripts/b601_so101_bimanual/02_record.sh / 训练数据 meta/tasks.parquet）完全一致
TASK_DESCRIPTION="Pick up the paper cup with both arms, place it on the silver tray of the coffee machine, press the button with the right arm (red light on), wait about 4 seconds, release the button (red light off), then place the cup on the table with the left arm"
EPISODE_TIME=400  # 推理时长上限（秒），episode 轮转由策略触达/时长控制
FPS=30  # 推理频率，必须与训练数据集 fps（30Hz）一致，否则 pi0 时序不匹配
# 是否推送到 Hugging Face Hub（外网不可达时改为 false，数据仅保存在本地）
PUSH_TO_HUB=false
# 国内镜像源（配合上传开关；网络可用时生效）
export HF_ENDPOINT=https://hf-mirror.com

# ==================== 推理前检查 ====================
# 1) 模型权重（先检查，路径不对直接退出，避免误触硬件）
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 模型路径不存在：$MODEL_PATH"
  exit 1
fi
echo "✅ 模型文件存在：$MODEL_PATH"

# 2) B601 CAN
echo ""
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

# 3) SO-101 串口
[ -e "$RIGHT_FOLLOWER_PORT" ] || { echo "❌ 串口不存在：$RIGHT_FOLLOWER_PORT"; exit 1; }
echo "✅ $RIGHT_FOLLOWER_PORT 已连接"

# 4) 摄像头（4 路）
echo ""
for cam in \
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0 \
  /dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0 \
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0 \
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0; do
  if [ ! -e "$cam" ]; then
    echo "❌ 警告：摄像头不存在 $cam"
  else
    echo "✅ $cam 已连接"
  fi
done

# 5) 从臂校准文件（推理需与采集同 id，rollout 会自动拆成 *_left/*_right）
echo ""
for f in \
  "$HOME/.cache/huggingface/lerobot/calibration/robots/seeed_b601_rs_follower/${FOLLOWER_ID}_left.json" \
  "$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/${FOLLOWER_ID}_right.json"; do
  if [ -f "$f" ]; then
    echo "✅ $f"
  else
    echo "❌ 缺失校准文件：$f"
    exit 1
  fi
done

# ==================== 推理参数总览 ====================
echo ""
echo "=========================================="
echo "推理参数总览"
echo "=========================================="
echo "模型路径：$MODEL_PATH"
echo "机器人类型：bi_b601_so101_follower（左 B601 can0 + 右 SO-101 串口）"
echo "推理方式：rtc（pi0 支持 RTC，10 步去噪）"
echo "评估数据集：$EVAL_DATASET_NAME"
echo "任务描述：$TASK_DESCRIPTION"
echo "推理时长：${EPISODE_TIME}秒 (约 $((EPISODE_TIME / 60)) 分钟)"
echo "推理频率：${FPS} Hz"
echo "摄像头：left_hand / left_top / left_front / right_hand（4 路）"
echo "B601 方向修正：全 -1.0（与采集一致）"
echo "相对动作上限：$MAX_RELATIVE_TARGET 度/步"
echo "录制视频：✅ 是"
echo "上传到Hub：$PUSH_TO_HUB"
echo "=========================================="
echo ""

read -p "确认左 B601（can0）与右 SO-101 从臂处于零位、周边安全，按 ENTER 开始推理，Ctrl+C 取消..." dummy

# ==================== 开始推理 ====================
echo ""
echo "🚀 开始异构双臂模型推理（pi0）..."
echo ""

# Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inference_b601_so101_pi0_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 右臂动作诊断：记录 模型预期动作(goal) / 实际发送指令(sent) / 电机实际位置(present)，
# 输出到 $LOG_DIR/action_diag_jt_follower_arm_right.csv（B601 无此 hook，仅右臂有效）。
# 不需要时注释掉下面两行即可关闭。
rm -rf "$LOG_DIR/action_diag"
export LEROBOT_ACTION_DIAG="$LOG_DIR/action_diag"

# 构建推理命令（lerobot-rollout + episodic 策略，镜像 self_scripts/b601_so101_bimanual/02_record.sh 的硬件/数据配置）
# 说明：
#   - pi0 支持 RTC，--inference.type=rtc 解决推理卡顿
#   - episodic：录制 num_episodes 个 episode，每个最长 episode_time_s 秒
#   - 左 B601：can0 + socketcan + gravity_compensation + joint_directions(全 -1.0)，与采集一致
#   - 机器人 4 路相机输出 key（left_hand/left_top/left_front/right_hand）与数据集 key 一致，无需 rename_map
#   - --dataset.fps 与 --fps 均须 30（训练数据 30Hz）
#   - --duration 为总时长上限（保护）；episode 轮转由 episode_time_s 控制
lerobot-rollout \
  --strategy.type=episodic \
  --inference.type=rtc \
  --policy.path=$MODEL_PATH \
  $NUM_STEPS_ARG \
  --robot.type=bi_b601_so101_follower \
  --robot.id=$FOLLOWER_ID \
  --robot.left_arm_config.port=$B601_FOLLOWER_PORT \
  --robot.left_arm_config.can_adapter=socketcan \
  --robot.left_arm_config.gravity_compensation=true \
  --robot.left_arm_config.joint_directions="$B601_JOINT_DIRECTIONS" \
  --robot.left_arm_config.cameras="$LEFT_CAMERAS" \
  --robot.left_arm_config.max_relative_target=$MAX_RELATIVE_TARGET \
  --robot.right_arm_config.port=$RIGHT_FOLLOWER_PORT \
  --robot.right_arm_config.cameras="$RIGHT_CAMERAS" \
  --robot.right_arm_config.max_relative_target=$MAX_RELATIVE_TARGET \
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
echo "✅ 异构双臂推理完成（pi0）！"
echo "=========================================="
echo "评估数据集：$EVAL_DATASET_NAME"
if [ "$PUSH_TO_HUB" = true ]; then
  echo "Hugging Face Hub 链接："
  echo "  https://huggingface.co/datasets/$EVAL_DATASET_NAME"
else
  echo "本地缓存路径：~/.cache/huggingface/lerobot/$EVAL_DATASET_NAME*"
fi
echo "完整日志：$LOG_FILE"
echo "=========================================="
