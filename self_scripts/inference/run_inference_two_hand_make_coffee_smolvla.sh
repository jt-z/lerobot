#!/bin/bash
# 双臂 SmolVLA 推理脚本（参照 run_inference_two_hand_make_coffee_ACT.sh 改写）
# 任务：Pick up the paper cup with both arms, place it on the silver tray of the coffee machine,
#       press the button with the right arm (red light on), wait about 4 seconds, release the button
#       (red light off), then place the cup on the table with the left arm
# 创建日期：2026-08-27
#
# ⚠️ 注意：本 SmolVLA 模型（/home/kf/lerobot_weights/smol_vla_new_dataset/10k_pretrained_model）
#   训练自 hellozjt/coffee_cup_button_20260826_232220 数据集（coffee_cup_button 任务，见 train_config.json）。
#   SmolVLA 是语言条件模型：--dataset.single_task 会作为文本提示（prompt）输入模型，
#   必须与训练数据集的任务描述完全一致（已对照 meta/tasks.parquet 确认）。

set -e  # 遇到错误立即退出
set -o pipefail  # 管道中任一命令失败则整体失败（配合 tee 使用）

echo "=========================================="
echo "双臂模型推理（SmolVLA）- 双臂协同咖啡机（倒咖啡）"
echo "=========================================="
echo ""

# ==================== 硬件配置 ====================
# 端口映射（与 collect_make_coffee.sh / teleoperate_dual_so101.sh 一致，2026-08-26 实测）：
#   左从臂 = USB 序列号 5C82108837（当前枚举为 ttyACM2）
#   右从臂 = USB 序列号 5B61034841（当前枚举为 ttyACM3）
# 注意：ttyACM 编号随插拔顺序变化，故下面用 /dev/serial/by-id 稳定路径，
#       只要适配器与机械臂的物理接线不变就不会变
# 本脚本使用双臂从动（follower）模式，故取左右从臂
LEFT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82108837-if00"
RIGHT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00"

# ==================== 摄像头配置 ====================
# SmolVLA 模型输入需要 4 路摄像头：left_hand / left_top / left_front / right_hand
# （与 ACT 一致，见 10k_pretrained_model/config.json 的 input_features）
# 摄像头映射（2026-08-26 实测，与采集训练数据的 collect_make_coffee.sh 一致）：
#   left_hand  = icSpring 无序列号（当前枚举为 /dev/video6）
#   left_top   = icSpring 202404160005（当前枚举为 /dev/video2）
#   left_front = JYU2C-2083 2607031（当前枚举为 /dev/video4）
#   right_hand = JYU2C-2083 2607060（当前枚举为 /dev/video0）
# 注意：/dev/videoN 编号随 USB 插拔顺序变化，故使用 /dev/v4l/by-id 稳定路径；
#       若推理画面中手部/顶部视角反了，交换下面 hand 与 top 两个 by-id 路径即可。
LEFT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# 右臂：1个摄像头（手部）
RIGHT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== 模型配置 ====================
# SmolVLA（本机微调，coffee_cup_button 任务，基于 SmolVLM2-500M-Video-Instruct）
MODEL_PATH="/home/kf/lerobot_weights/smol_vla_new_dataset/10k_pretrained_model"

# SmolVLA 推理所需的基础 VLM（config + processor，用于构建语言 tokenizer 与随机初始化 VLM 后被
# model.safetensors 覆盖）。
# ⚠️ 注意：默认缓存 ~/.cache/huggingface/hub 下同名目录不完整（缺 processor/tokenizer 等文件），
#   本机完整缓存在 /home/kf/lerobot_weights/smol_vla/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct，
#   脚本通过 HF_HUB_CACHE 指向该目录并强制离线（见下方环境变量），完全不会访问 Hugging Face。
VLM_MODEL_NAME="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# ==================== 数据集配置 ====================
# 注意：lerobot-rollout 强制要求数据集名以 rollout_ 开头（见 rollout/context.py）
EVAL_DATASET_NAME="hellozjt/rollout_coffee_cup_button_smolvla"
# SmolVLA 是语言条件模型：此任务描述就是推理时的文本提示，必须与训练任务完全一致
TASK_DESCRIPTION="Pick up the paper cup with both arms, place it on the silver tray of the coffee machine, press the button with the right arm (red light on), wait about 4 seconds, release the button (red light off), then place the cup on the table with the left arm"
EPISODE_TIME=200  # 推理时长（秒）
FPS=20
# 是否推送到 Hugging Face Hub（当前环境外网不可达，保持 false 数据仅保存在本地；
# 联网后可改回 true）
PUSH_TO_HUB=false

# ==================== 摄像头 Rename 映射 ====================
# SmolVLA 模型输入 key 为 observation.images.left_hand / left_top / left_front / right_hand，
# 与 bi_so_follower 机器人的输出 key（自动加 left_/right_ 前缀）完全一致，
# 因此不需要 rename_map（policy_preprocessor.json 中 rename_map 也为空）。

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

# 检查摄像头（4 路）
echo ""
echo "2. 检查摄像头..."
for camera in \
  /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 \
  /dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0 \
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0 \
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0; do
  if [ ! -e "$camera" ]; then
    echo "❌ 警告：摄像头不存在 $camera"
  else
    echo "✅ $camera 已连接"
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
  echo "   类型: $(grep -o '"type": *"[a-z0-9_.]*"' "$MODEL_PATH/config.json" | head -1)"
fi

# 检查基础 VLM 缓存（SmolVLA 需要其 config/processor 构建 tokenizer）
echo ""
echo "4. 检查 SmolVLA 基础 VLM 缓存..."
# 默认缓存 ~/.cache/huggingface/hub 下同名目录不完整，此处指向本机完整缓存
#（含 snapshots 与 processor/tokenizer 文件）
VLM_CACHE_DIR="/home/kf/lerobot_weights/smol_vla/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
# 目录存在不代表缓存完整：校验关键文件（processor_config.json / tokenizer.json）
VLM_SNAPSHOT_DIR=""
for dir in "$VLM_CACHE_DIR"/snapshots/*/; do
  [ -d "$dir" ] && VLM_SNAPSHOT_DIR="$dir" && break
done
if [ -n "$VLM_SNAPSHOT_DIR" ] && [ -f "$VLM_SNAPSHOT_DIR/processor_config.json" ] && [ -f "$VLM_SNAPSHOT_DIR/tokenizer.json" ]; then
  echo "✅ 基础 VLM 已缓存（完整）：$VLM_CACHE_DIR"
else
  echo "❌ 错误：基础 VLM 缓存不完整 $VLM_CACHE_DIR"
  echo "   缺少 processor_config.json / tokenizer.json 等文件，无法离线推理"
  echo "   请从其他机器拷贝完整缓存，或联网后重新下载 $VLM_MODEL_NAME"
  exit 1
fi

# ==================== 推理参数总览 ====================
echo ""
echo "=========================================="
echo "推理参数总览"
echo "=========================================="
echo "模型路径：$MODEL_PATH"
echo "评估数据集：$EVAL_DATASET_NAME"
echo "任务描述（文本提示）：$TASK_DESCRIPTION"
echo "推理时长：${EPISODE_TIME}秒 (约 $((EPISODE_TIME / 60)) 分钟)"
echo "推理频率：${FPS} Hz"
echo "摄像头：left_hand / left_top / left_front / right_hand（4 路）"
echo "录制视频：✅ 是"
echo "上传到Hub：$PUSH_TO_HUB"
echo "=========================================="
echo ""

# 确认开始
read -p "按 ENTER 开始推理，按 Ctrl+C 取消..." dummy

# ==================== 开始推理 ====================
echo ""
echo "🚀 开始双臂模型推理（SmolVLA）..."
echo ""

# 设置 Rerun 缓冲区大小（解决 gRPC transport error）
# 默认 8KB 太小，4 路摄像头每帧约 3.7MB，增大到 10MB
export RERUN_FLUSH_NUM_BYTES=10000000

# 设置 Rerun 内存限制为 30%（解决 1000 帧限制问题）
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

# ==== 基础 VLM 完全离线加载（不访问 Hugging Face）====
# 1) HF_HUB_CACHE 指向本机完整缓存目录（~/.cache/huggingface/hub 下同名缓存不完整）；
# 2) HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE 强制 huggingface_hub 与 transformers 只读本地缓存，
#    避免联网 HEAD 请求超时（当前环境外网不可达，否则会卡在 Retry 重试）。
export HF_HUB_CACHE=/home/kf/lerobot_weights/smol_vla
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 推理日志文件（输出同时显示在终端并写入此文件，便于事后查看）
LOG_DIR="./infer_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inference_smolvla_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 动作诊断：每个 tick 记录 模型预期动作(goal) / 实际发送指令(sent) / 电机实际位置(present)，
# 输出到 $LOG_DIR/action_diag_jt_follower_arm_left.csv 和 _right.csv（左右臂各一文件）
# 用于排查左右臂 12 关节的指令是否符合模型预期与电机实际状态。
# 不需要诊断时注释掉下面两行即可关闭（关闭后无额外开销）。
rm -rf "$LOG_DIR/action_diag"
export LEROBOT_ACTION_DIAG="$LOG_DIR/action_diag"

# 清除旧的评估数据集缓存（可选）
# rm -rf ~/.cache/huggingface/lerobot/$EVAL_DATASET_NAME

# 构建推理命令（lerobot-rollout + episodic 策略，镜像 lerobot-record 的录制行为）
# 说明：
#   - episodic：录制 num_episodes 个 episode，每个最长 episode_time_s 秒，episode 间有 reset 阶段
#   - SmolVLA 使用 sync 推理（默认，与 ACT 相同）：模型内部按 chunk 一次生成 50 步动作、逐步回放；
#     本模型未用 RTC 训练（config.json 中 rtc_config=null），故不能使用 --inference.type=rtc
#   - SmolVLA 无 --policy.num_inference_steps 字段（解码步数由 config.json 的 num_steps=10 决定，
#     若嫌慢可在命令中加 --policy.num_steps=4 覆盖，速度更快但动作质量可能下降）
#   - --dataset.single_task 是语言条件文本提示（SmolVLA 必需），必须与训练任务一致
#   - 无需 rename_map：机器人 4 路摄像头输出 key 与 SmolVLA 模型输入 key 完全一致
#   - --duration 为总时长上限（保护）；episode 轮转由 episode_time_s 控制
#   - --use_torch_compile=true：torch.compile 加速 VLM 推理（max-autotune），缩短每 2.5s 一次
#     chunk 推理停顿（sync 推理卡顿来源），不改变 num_steps=10 故不损失动作精度。
#     ⚠️ 首次运行会触发懒编译（约 2-3 分钟，发生在第一个 tick 的推理里），会消耗掉
#     --duration 预算导致只录 1 帧就结束——属正常现象，编译产物缓存在
#     ~/.cache/torch/inductor，第二次运行即恢复正常。
lerobot-rollout \
  --strategy.type=episodic \
  --policy.path=$MODEL_PATH \
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
  --dataset.video=true \
  --dataset.rgb_encoder.vcodec=h264 \
  --dataset.push_to_hub=$PUSH_TO_HUB \
  --duration=$EPISODE_TIME \
  --display_data=true \
  --display_compressed_images=false \
  --use_torch_compile=true 2>&1 | tee "$LOG_FILE"

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ 双臂推理完成（SmolVLA）！"
echo "=========================================="
echo "评估数据集：$EVAL_DATASET_NAME"
if [ "$PUSH_TO_HUB" = "true" ]; then
  echo "Hugging Face Hub 链接："
  echo "  https://huggingface.co/datasets/$EVAL_DATASET_NAME"
else
  echo "（PUSH_TO_HUB=false，数据仅保存在本地）"
  echo "本地数据目录：$HOME/.cache/huggingface/lerobot/$EVAL_DATASET_NAME"
fi
echo "完整日志：$LOG_FILE"
echo ""
echo "💡 提示："
echo "  1. 查看上述数据目录/链接中的推理视频和轨迹"
echo "  2. 分析模型性能和成功率"
echo "  3. 如需重新推理，直接再次运行此脚本"
echo "=========================================="
