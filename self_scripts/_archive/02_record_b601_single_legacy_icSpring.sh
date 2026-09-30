#!/bin/bash
# ⚠️ 已归档（历史版本，不要再用）：B601 单臂采集的旧版。
#    现用版 = self_scripts/b601_single/02_record.sh，与旧版有三处实质差异：
#      1) 数据根目录：$HOME/LX/pai0/hellozjt/  →  $HOME/LX/pai0/b601_data/（repo_id 去掉 hellozjt 前缀）
#      2) top 相机：icSpring 202404160005（实测只有 20fps、会掉到 10fps）→ JYU2C 2607060
#      3) 任务描述：补充了"拿方块按按钮"环节
#    仅作历史对照保留。
#
# 单臂数据采集：B601-RS 从臂（RobStride，SocketCAN）+ StarArm102 主臂
# 参照 self_scripts/b601_so101_bimanual/02_record.sh（原为 B601+SO-101 异构双臂版，本脚本为 B601 单臂版）
# 使用单臂 lerobot 类型：
#   robot  = seeed_b601_rs_follower（B601-RS，can0）
#   teleop = rebot_102_leader（StarArm102 /dev/ttyUSB0）
# 摄像头：仅使用左臂（B601）3 路 —— hand（JYU2C 2608076）、top（icSpring 202404160005）、front（JYU2C 2607031），
#         已去掉右臂（SO-101）摄像头。
# 录出的是单份 7 关节数据集，可直接用于训练。
#
# 支持中断继续录制（--resume）：
#   如果录制过程中断，可以添加 --resume 参数继续录制。
#   脚本会自动从上次中断的 episode 继续。
#
# 按键结束每个 episode（lerobot-record 内置）：
#   演示完任务后按 n（或右方向键）立即结束当前 episode 并保存。
#   n/右方向键=结束本集 | r/左方向键=重录上一集 | q/Esc=停止整个录制
#
# 安全须知：
#   1. 采集前请确保主/从臂已校准（校准文件见下方检查项）
#   2. 从臂连接后会【使能变硬】
#   3. 移动主臂 -> 从臂跟随（已去除速度限幅，跟手直连；仅保留 B601 关节硬限位 + 温度保护）
#   4. 退出（Ctrl+C）时从臂会回零位（safe_zero）并失能
#   5. 紧急停止 = 切断 48V 电源
#   6. 启动前把【主臂和从臂】都摆到零位
#
# 用法（已弃用）：bash self_scripts/_archive/02_record_b601_single_legacy_icSpring.sh
#   续录：bash self_scripts/_archive/02_record_b601_single_legacy_icSpring.sh --resume <完整repo_id> [追加集数]
#   示例：bash self_scripts/_archive/02_record_b601_single_legacy_icSpring.sh --resume hellozjt/single_b601_make_coffee_20260908_000000 20

set -e
set -o pipefail

# ==================== 环境检查 ====================
if ! command -v lerobot-record >/dev/null 2>&1; then
  echo "未找到 lerobot-record，尝试激活 lerobot conda 环境..."
  for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      source "$base/etc/profile.d/conda.sh"
      conda activate lerobot
      break
    fi
  done
fi

if ! command -v lerobot-record >/dev/null 2>&1; then
  echo "❌ 错误：lerobot-record 不可用"
  echo "   请手动执行: conda activate lerobot"
  exit 1
fi
echo "✅ lerobot-record 可用"

# ==================== 数据保存位置常量 ====================
# 数据集本地根目录与 repo_id 前缀（此处提前定义，供下方 --resume 提示使用）
DATA_ROOT="$HOME/LX/pai0"
# 数据集前缀（本地目录与 repo_id 同名，位于 $DATA_ROOT/hellozjt/ 下）
DATASET_PREFIX="hellozjt/single_b601_make_coffee"

# ==================== 命令行参数 ====================
RESUME_MODE=false
RESUME_REPO_ID=""
NUM_EPISODES_OVERRIDE=""
if [ "$1" == "--resume" ]; then
  RESUME_MODE=true
  if [ -z "$2" ]; then
    echo "❌ 错误：--resume 模式必须指定要续录的数据集完整 repo_id（带时间戳）"
    echo ""
    echo "   可用会话："
    ls -d "$DATA_ROOT/hellozjt/${DATASET_PREFIX#hellozjt/}_"* 2>/dev/null | sed 's|.*/||' | sort
    echo ""
    echo "   示例：bash self_scripts/b601_single/02_record.sh --resume hellozjt/single_b601_make_coffee_20260908_000000"
    exit 1
  fi
  RESUME_REPO_ID="$2"
  NUM_EPISODES_OVERRIDE="${3:-}"
  echo "🔄 恢复模式：续录数据集 $RESUME_REPO_ID"
fi

echo "=========================================="
echo "单臂数据采集：B601-RS（咖啡任务）"
echo "=========================================="
echo ""

# ==================== 硬件配置 ====================
# 校准文件：seeed_b601_rs_follower/follower.json、rebot_102_leader/leader.json
FOLLOWER_ID="follower"
LEADER_ID="leader"

# 从臂 = B601-RS（SocketCAN）
FOLLOWER_PORT="can0"
FOLLOWER_ADAPTER="socketcan"
# 重力补偿（对 6 个 RS 关节做前馈），开启后手感更轻
GRAVITY_COMPENSATION=true
# 从臂关节方向修正（沿用 self_scripts/b601_so101_bimanual/01_teleop_test.sh、02_record.sh 中 B601 的实测值，与 self_scripts/b601_single/01_teleop.sh 保持一致；
# 若某关节方向相反/被限位裁到 0，调整对应符号）
FOLLOWER_JOINT_DIRECTIONS='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'

# 主臂 = StarArm102（CH340 无序列号，只能固定 /dev/ttyUSB0）
# 备选 by-id：/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
LEADER_PORT="/dev/ttyUSB0"

# ==================== 摄像头配置 ====================
# 仅 B601（原左臂）3 路摄像头，路径用 /dev/v4l/by-id 稳定路径：
#   hand（手部，JYU2C 2608076）、top（icSpring 202404160005）、front（JYU2C 2607031）
FOLLOWER_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== 数据集配置 ====================
if [ -n "$RESUME_REPO_ID" ]; then
  DATASET_NAME="$RESUME_REPO_ID"
else
  # 新建：脚本生成带时间戳的 repo_id（与 lerobot 默认 stamp 格式 %Y%m%d_%H%M%S 一致），
  # 后面配合 --dataset.no_stamp=true 使用，保证目录名与 repo_id 完全一致。
  DATASET_NAME="${DATASET_PREFIX}_$(date +%Y%m%d_%H%M%S)"
fi
# 是否推送到 Hugging Face Hub（当前环境外网不可达，建议保持 false，数据只存本地）
PUSH_TO_HUB=false
# 国内镜像源（配合上面开关；网络可用时生效）
export HF_ENDPOINT=https://hf-mirror.com

# 单臂咖啡任务描述（可自行修改；B601 单臂需完成整套动作）
TASK_DESCRIPTION="Pick up the paper cup, place it on the silver tray of the coffee machine, press the button with the arm (red light on), wait about 4 seconds, release the button (red light off), then place the cup on the table"
NUM_EPISODES=100
# --resume 模式下允许用第 3 个命令行参数覆盖本次追加的集数
if [ -n "$NUM_EPISODES_OVERRIDE" ]; then
  NUM_EPISODES="$NUM_EPISODES_OVERRIDE"
fi
# 每个 episode 录制时长的安全上限（秒）；录制由按键结束，该值只是防无限录制的保险
EPISODE_TIME=600
RESET_TIME=5    # 重置环境时长（秒）；按 n / 右方向键可提前跳过等待
# 采集频率（Hz）：与 self_scripts/b601_so101_bimanual/02_record.sh 中 B601 一致，采用 30Hz
FPS=30

# ==================== 采集前检查 ====================
echo "1. 检查硬件..."
# B601 CAN
if ! ip link show $FOLLOWER_PORT >/dev/null 2>&1; then
  echo "❌ $FOLLOWER_PORT 不存在"; exit 1
fi
if ! ip link show $FOLLOWER_PORT 2>/dev/null | grep -q "state UP"; then
  echo "⚠️  配置 $FOLLOWER_PORT（1Mbps 经典 CAN）..."
  sudo ip link set $FOLLOWER_PORT down 2>/dev/null || true
  sudo ip link set $FOLLOWER_PORT type can bitrate 1000000
  sudo ip link set $FOLLOWER_PORT up
fi
# 主臂串口
if [ ! -e "$LEADER_PORT" ]; then
  echo "❌ 串口不存在：$LEADER_PORT"
  echo "   请检查 StarArm102 USB 连接，以及是否被 brltty 抢占:"
  echo "   sudo apt remove -y brltty"
  exit 1
fi
[ -r "$LEADER_PORT" ] || sudo chmod 666 "$LEADER_PORT"
echo "✅ 硬件就绪"

# 摄像头检查
echo ""
echo "2. 检查摄像头..."
for cam in $(echo "$FOLLOWER_CAMERAS" | grep -o 'index_or_path: [^,]*' | awk '{print $2}'); do
  [ -e "$cam" ] && echo "  ✅ $cam" || echo "  ❌ 摄像头不存在: $cam"
done

# 校准文件检查
echo ""
echo "3. 检查校准文件..."
CALIB_OK=true
for f in \
  "$HOME/.cache/huggingface/lerobot/calibration/robots/seeed_b601_rs_follower/${FOLLOWER_ID}.json" \
  "$HOME/.cache/huggingface/lerobot/calibration/teleoperators/rebot_102_leader/${LEADER_ID}.json"; do
  if [ -f "$f" ]; then
    echo "  ✅ $f"
  else
    echo "  ❌ 缺失：$f"
    CALIB_OK=false
  fi
done
if [ "$CALIB_OK" = false ]; then
  echo ""
  echo "❌ 校准文件不完整，请先完成主/从臂校准："
  echo "   从臂（B601-RS）：  lerobot-calibrate --robot.type=seeed_b601_rs_follower --robot.port=$FOLLOWER_PORT --robot.can_adapter=$FOLLOWER_ADAPTER --robot.id=$FOLLOWER_ID"
  echo "   主臂（StarArm102）：lerobot-calibrate --teleop.type=rebot_102_leader --teleop.port=$LEADER_PORT --teleop.id=$LEADER_ID"
  exit 1
fi

# ==================== 数据集保存位置检查 ====================
echo ""
echo "4. 检查数据集保存位置..."
if [ "$RESUME_MODE" = true ]; then
  DATA_DIR="$DATA_ROOT/$DATASET_NAME"
  if [ -d "$DATA_DIR" ]; then
    EXISTING_EPISODES=$(python3 -c "import json; print(json.load(open('$DATA_DIR/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
    echo "✅ 续录目录已找到：$DATA_DIR"
    echo "   已有 $EXISTING_EPISODES 个 episode，本次将追加 $NUM_EPISODES 个"
  else
    echo "❌ 错误：续录目录不存在：$DATA_DIR"
    exit 1
  fi
else
  DATA_DIR="（新建数据集，目录名带时间戳）"
  EXISTING_EPISODES=0
  # 记录本次开始前已存在的 session，用于结束后识别本次新建的目录
  OLD_SESSIONS=$(ls -d "$DATA_ROOT/hellozjt/${DATASET_PREFIX#hellozjt/}_"* 2>/dev/null || true)
  EXISTING_SESSIONS=$(echo "$OLD_SESSIONS" | grep -c . 2>/dev/null || echo 0)
  echo "✅ 数据根目录：$DATA_ROOT/hellozjt/"
  echo "   该目录下已有 $EXISTING_SESSIONS 个历史 session（本次会新建一个带时间戳的目录，不覆盖旧数据）"
fi

# ==================== 采集参数总览 ====================
echo ""
echo "=========================================="
echo "采集参数总览"
echo "=========================================="
echo "录制模式：$( [ "$RESUME_MODE" = true ] && echo "续录（--resume）" || echo "新建数据集" )"
echo "数据集名称：$DATASET_NAME"
echo "本地保存路径：$DATA_DIR"
echo "Episode 进度：已有 $EXISTING_EPISODES 集 + 本次录制 $NUM_EPISODES 集 = 共 $((EXISTING_EPISODES + NUM_EPISODES)) 集"
echo "任务描述：$TASK_DESCRIPTION"
echo "每个 Episode 时长：由按键结束（安全上限 ${EPISODE_TIME}秒）"
echo "重置时长：${RESET_TIME}秒（可按键提前跳过）"
echo "采集频率：${FPS} Hz"
echo "=========================================="
echo ""
echo -e "📝 按键控制（录制过程中随时可用，注意焦点保持在终端窗口）："
echo -e "   n / 右方向键 = 当前 episode 演示完成，结束录制并保存"
echo -e "   r / 左方向键 = 重录上一个 episode"
echo -e "   q / Esc      = 停止整个录制"
echo ""
echo -e "   每个 episode 流程：开始录制 → 演示任务 → 按 n 结束并保存 →"
echo -e "   重置环境（可再按 n 跳过等待）→ 自动进入下一个 episode。"
echo ""
read -p "确认主臂/从臂已在零位、周边安全，按 ENTER 开始采集，Ctrl+C 取消..." dummy

# ==================== 开始采集 ====================
echo ""
echo "🚀 开始数据采集..."
echo ""

# Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/record_single_b601_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志：$LOG_FILE"
echo ""

RECORD_CMD="lerobot-record \
  --robot.type=seeed_b601_rs_follower \
  --robot.id=$FOLLOWER_ID \
  --robot.port=$FOLLOWER_PORT \
  --robot.can_adapter=$FOLLOWER_ADAPTER \
  --robot.gravity_compensation=$GRAVITY_COMPENSATION \
  --robot.joint_directions='$FOLLOWER_JOINT_DIRECTIONS' \
  --robot.cameras=\"$FOLLOWER_CAMERAS\" \
  --teleop.type=rebot_102_leader \
  --teleop.id=$LEADER_ID \
  --teleop.port=$LEADER_PORT \
  --dataset.repo_id=$DATASET_NAME \
  --dataset.num_episodes=$NUM_EPISODES \
  --dataset.single_task=\"$TASK_DESCRIPTION\" \
  --dataset.fps=$FPS \
  --dataset.episode_time_s=$EPISODE_TIME \
  --dataset.reset_time_s=$RESET_TIME \
  --dataset.video=true \
  --dataset.rgb_encoder.vcodec=h264 \
  --dataset.push_to_hub=$PUSH_TO_HUB \
  --display_data=true \
  --display_compressed_images=false"

# 数据保存位置：新建/续录都用显式 --dataset.root。
# 注意：root 会被当作【数据集目录本身】而不是父目录（lerobot 不会自动拼 repo_id），
#       因此新建时必须传入完整带时间戳的路径 $DATA_ROOT/$DATASET_NAME，
#       并用 --dataset.no_stamp=true 阻止 lerobot 重复追加时间戳。
if [ "$RESUME_MODE" = true ]; then
  RECORD_CMD="$RECORD_CMD --resume=true --dataset.root=$DATA_ROOT/$DATASET_NAME"
else
  RECORD_CMD="$RECORD_CMD --dataset.no_stamp=true --dataset.root=$DATA_ROOT/$DATASET_NAME"
fi

# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂、lerobot-record 被
# SIGPIPE(141) 杀死而跳过断开清理（否则电机保持使能、从臂不释放）。
trap '' PIPE

# 执行命令
# 注意：录制完成后断开机械臂时，若夹爪处于过载状态可能报错（Overload error），
#       导致 lerobot-record 非零退出。此时数据已保存，不影响使用，
#       因此这里不因退出码直接中断，而是提示后继续显示结果。
set +e
eval $RECORD_CMD 2>&1 | tee "$LOG_FILE"
RECORD_EXIT=${PIPESTATUS[0]}
set -e
if [ $RECORD_EXIT -ne 0 ]; then
  echo ""
  echo "⚠️  lerobot-record 非零退出（exit=$RECORD_EXIT）"
  echo "    若录制过程已完成（上方日志显示 episode 已保存/视频已编码），"
  echo "    这通常是断开机械臂时的清理错误（如夹爪过载），数据不受影响。"
  echo "    机械臂可能需要断电重启以复位过载保护。"
fi

# ==================== 采集完成 ====================
echo ""
echo "=========================================="
echo "✅ 数据采集结束"
echo "=========================================="

# 识别实际保存目录
if [ "$RESUME_MODE" = true ]; then
  SAVED_DIR="$DATA_DIR"
else
  # 新建模式：找本次开始前不存在、最新的 session 目录
  SAVED_DIR=""
  for d in $(ls -dt "$DATA_ROOT/hellozjt/${DATASET_PREFIX#hellozjt/}_"* 2>/dev/null || true); do
    if ! echo "$OLD_SESSIONS" | grep -qx "$d"; then
      SAVED_DIR="$d"
      break
    fi
  done
  if [ -z "$SAVED_DIR" ]; then
    SAVED_DIR="（未能自动定位，请到 $DATA_ROOT/hellozjt/ 下查看最新目录）"
  fi
fi

# 统计录制完成后的实际集数（读 meta/info.json 的 total_episodes）
TOTAL_EPISODES=0
if [ -d "$SAVED_DIR/meta" ]; then
  TOTAL_EPISODES=$(python3 -c "import json; print(json.load(open('$SAVED_DIR/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
fi
SAVED_EPISODES=$((TOTAL_EPISODES - EXISTING_EPISODES))
if [ "$SAVED_EPISODES" -lt 0 ]; then SAVED_EPISODES=0; fi

echo "数据集名称：$DATASET_NAME"
echo "本地保存路径：$SAVED_DIR"
echo "本次实际保存 Episode：$SAVED_EPISODES / 目标 $NUM_EPISODES"
echo "   （该 session 累计：$TOTAL_EPISODES 集）"
if [ "$PUSH_TO_HUB" = true ]; then
  echo "Hugging Face Hub 链接："
  echo "  https://huggingface.co/datasets/$DATASET_NAME"
else
  echo "上传状态：未推送到 Hub（PUSH_TO_HUB=false，数据仅保存在本地）"
fi
echo ""
echo "💡 提示："
echo "  如果录制过程中断，可以使用以下命令续录："
echo "  ./self_scripts/b601_single/02_record.sh --resume $DATASET_NAME"
echo ""
echo "下一步："
echo "  1. 检查数据质量"
echo "  2. 开始训练模型"
echo "=========================================="
