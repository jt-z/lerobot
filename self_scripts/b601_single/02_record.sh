#!/bin/bash
# 单 B601-RS 数据采集（从臂 B601-RS + 主臂 StarArm102 / reBot Arm 102）
# 使用单臂官方 lerobot 类型：
#   robot  = seeed_b601_rs_follower（B601-RS RobStride 电机，SocketCAN can0）
#   teleop = rebot_102_leader（StarArm102 主臂，/dev/ttyUSB0）
# 录出单份 7 关节数据集（shoulder_pan/lift、elbow_flex、wrist_flex/yaw/roll、gripper），
# 相机 key 无 left_/right_ 前缀（hand=腕部 / front / top=顶部），可直接用于单臂训练。
#
# 参照 self_scripts/b601_so101_bimanual/02_record.sh 改写（去掉 SO-101 右臂，robot/teleop 换单臂类型，
# 校准文件用单臂版本 follower.json / leader.json）。
#
# 支持中断继续录制（--resume）：
#   如果录制过程中断，可以添加 --resume 参数继续录制。
#
# 按键结束每个 episode（lerobot-record 内置）：
#   n/右方向键=结束本集 | r/左方向键=重录上一集 | q/Esc=停止整个录制
#
# 安全须知：
#   1. 采集前请确保主从臂已校准（self_scripts/b601_common/02_follower_calibration.sh / 01_leader_calibration.sh）
#   2. 从臂连接后【使能变硬】，移动主臂 -> 从臂跟随（无速度限幅，跟手直连）
#   3. 退出（Ctrl+C）时从臂回零位坐姿（夹爪按 safe_zero 收至夹稳位，不会自动回收纳 0°，
#      下次启动前请手动把夹爪合到 0°，否则零点基准会偏移）
#   4. 紧急停止 = 切断 48V 电源
#   5. 启动前把主臂和从臂都摆到零位（默认坐姿、夹爪闭合）
#
# 用法：bash self_scripts/b601_single/02_record.sh
#   续录：bash self_scripts/b601_single/02_record.sh --resume <完整repo_id> [追加集数]

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
    ls -d "$HOME/LX/pai0/b601_data"/b601_* 2>/dev/null | sed 's|.*/||' | sort
    echo ""
    echo "   示例：bash self_scripts/b601_single/02_record.sh --resume b601_20260903_120000"
    exit 1
  fi
  RESUME_REPO_ID="$2"
  NUM_EPISODES_OVERRIDE="${3:-}"
  echo "🔄 恢复模式：续录数据集 $RESUME_REPO_ID"
fi

echo "=========================================="
echo "单 B601-RS 数据采集（B601 从臂 + StarArm102 主臂）"
echo "=========================================="
echo ""

# ==================== 硬件配置 ====================
FOLLOWER_ID="follower"   # 从臂校准 id（对应 seeed_b601_rs_follower/follower.json，02 脚本生成）
LEADER_ID="leader"       # 主臂校准 id（对应 rebot_102_leader/leader.json，01 脚本生成）

# B601 从臂 = can0（SocketCAN）
B601_FOLLOWER_PORT="can0"
# StarArm102 主臂串口：**自动探测**（参考 so101_bimanual/inference/old_dataset/run_inference_cap_pen_ACT_300k.sh 的思路）。
#   * 为什么不写死 /dev/ttyUSB0：ttyUSB*/ttyACM* 的编号随插拔顺序和驱动变化 —— 本机实测
#     同一个主臂适配器会在 ttyUSB0 与 ttyACM0 之间变名（09-14 / 09-15 各出现过一次），
#     所以优先用 /dev/serial/by-id 稳定路径。
#   * 但机内还有别的 USB 串口（WCH 1a86:55d4 "USB Single Serial" 等），只按名字挑会挑错，
#     于是用 self_scripts/b601_common/04_leader_port.py 对候选口**真正 ping 主臂舵机（FashionStar id 0~6）**，
#     第一个有应答的才算主臂；都不应答就直接报"没找到"，不会误连到别的设备。
#   * 手动覆盖：B601_LEADER_PORT=/dev/ttyACM0 bash self_scripts/b601_single/02_record.sh
B601_LEADER_PORT="${B601_LEADER_PORT:-}"
LEADER_PORT_PROBE="$(dirname "$(readlink -f "$0")")/../b601_common/04_leader_port.py"

# 从臂方向修正（必须保留，否则 shoulder_lift/gripper 被限位裁到 0 不动；
# 与 03/04 脚本的 B601 左臂一致）
B601_JOINT_DIRECTIONS='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'

# ==================== 摄像头配置 ====================
# 3 路相机（B601 从臂，无 left_/right_ 前缀）：
#   hand = JYU2C 2608076（腕部相机）、front = JYU2C 2607031、
#   top = JYU2C 2607060（顶部相机；原 icSpring 202404160005 实测仅 20fps
#         且会掉到 10fps，JYU2C 稳定 29.7fps，故换为 JYU2C）
# 路径用 /dev/v4l/by-id 稳定路径。
CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== 数据集配置 ====================
# 单 B601 数据根目录（新建），每个 session 存到 $DATA_ROOT/b601_<时间戳>/
DATA_ROOT="$HOME/LX/pai0/b601_data"
mkdir -p "$DATA_ROOT"

if [ -n "$RESUME_REPO_ID" ]; then
  DATASET_NAME="$RESUME_REPO_ID"
else
  # 新建：脚本生成带时间戳的数据集名，配合 --dataset.no_stamp=true 保证目录名一致
  DATASET_NAME="b601_$(date +%Y%m%d_%H%M%S)"
fi
# 是否推送到 Hugging Face Hub（当前环境外网不可达，建议保持 false，数据只存本地）
PUSH_TO_HUB=false
# 国内镜像源（配合上面开关；网络可用时生效）
export HF_ENDPOINT=https://hf-mirror.com

# ⚠️ 任务描述：改成你要采集的实际任务（须与后续训练一致）
# 与已采集数据集 b601_20260910_164106 及其它单臂 B601 脚本保持一致，
# 修改前请确认后续训练/续录都使用同一句描述。
TASK_DESCRIPTION="Pick up the paper cup, place it on the silver tray of the coffee machine, pick up the cube, press the button with the cube (red light on), wait about 4 seconds, release the button (red light off), put the cube on the table first, then move the cup from the coffee machine to the table"
NUM_EPISODES=120
# --resume 模式下允许用第 3 个命令行参数覆盖本次追加的集数
if [ -n "$NUM_EPISODES_OVERRIDE" ]; then
  NUM_EPISODES="$NUM_EPISODES_OVERRIDE"
fi
# 每个 episode 录制时长的安全上限（秒）；录制由按键结束，该值只是防无限录制的保险
EPISODE_TIME=600
RESET_TIME=5    # 重置环境时长（秒）；按 n / 右方向键可提前跳过等待
# 采集频率（Hz）
FPS=30

# ==================== 采集前检查 ====================
echo "1. 检查硬件..."
# B601 CAN
if ! ip link show $B601_FOLLOWER_PORT >/dev/null 2>&1; then
  echo "❌ $B601_FOLLOWER_PORT 不存在"; exit 1
fi
if ! ip link show $B601_FOLLOWER_PORT 2>/dev/null | grep -q "state UP"; then
  echo "⚠️  配置 $B601_FOLLOWER_PORT（1Mbps 经典 CAN）..."
  sudo ip link set $B601_FOLLOWER_PORT down 2>/dev/null || true
  sudo ip link set $B601_FOLLOWER_PORT type can bitrate 1000000
  sudo ip link set $B601_FOLLOWER_PORT up
fi
# 主臂串口（自动探测：优先 by-id 稳定路径，且必须真正 ping 到舵机才算数）
if [ -z "$B601_LEADER_PORT" ]; then
  if [ -f "$LEADER_PORT_PROBE" ]; then
    echo "🔍 探测主臂串口（候选口逐个 ping 舵机 id 0~6）..."
    B601_LEADER_PORT="$(python "$LEADER_PORT_PROBE" || true)"
  else
    echo "⚠️  未找到探测脚本 $LEADER_PORT_PROBE，退回按设备名挑"
    for cand in /dev/serial/by-id/usb-1a86_USB_Serial*-if00-port0 /dev/ttyUSB* /dev/ttyACM*; do
      [ -e "$cand" ] && B601_LEADER_PORT="$cand" && break
    done
  fi
fi
if [ -z "$B601_LEADER_PORT" ] || [ ! -e "$B601_LEADER_PORT" ]; then
  echo "❌ 未找到主臂串口：现有候选串口都没有 StarArm102 舵机应答"
  echo "   当前串口设备：$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | tr '\n' ' ')"
  echo "   排查："
  echo "   1) 主臂 USB 线是否插好：journalctl -k | grep -E 'ch341|cdc_acm' | tail -3"
  echo "   2) 主臂舵机是否**上电**（12V 电源/开关）—— 舵机不上电时任何串口都探测不到"
  echo "   3) 逐个口看探测结果：python $LEADER_PORT_PROBE --list"
  echo "   4) 手动指定：B601_LEADER_PORT=/dev/ttyACM0 bash $0"
  exit 1
fi
[ -r "$B601_LEADER_PORT" ] || sudo chmod 666 "$B601_LEADER_PORT"
echo "✅ 硬件就绪（主臂串口：$B601_LEADER_PORT）"

# 摄像头检查
echo ""
echo "2. 检查摄像头..."
for cam in $(echo "$CAMERAS" | grep -o 'index_or_path: [^,]*' | awk '{print $2}'); do
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
    echo "  ❌ 缺失：$f（请先运行 self_scripts/b601_common/02_follower_calibration.sh / 01_leader_calibration.sh）"
    CALIB_OK=false
  fi
done
if [ "$CALIB_OK" = false ]; then
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
  DATA_DIR="（新建数据集，目录名由框架自动加时间戳生成）"
  EXISTING_EPISODES=0
  # 记录本次开始前已存在的 session，用于结束后识别本次新建的目录
  OLD_SESSIONS=$(ls -d "$DATA_ROOT"/b601_* 2>/dev/null || true)
  EXISTING_SESSIONS=$(echo "$OLD_SESSIONS" | grep -c . 2>/dev/null || echo 0)
  echo "✅ 数据根目录：$DATA_ROOT"
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
echo "任务描述：${TASK_DESCRIPTION:-（未填写！请在脚本中修改 TASK_DESCRIPTION）}"
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
read -p "确认主从臂已在零位（夹爪闭合！）、周边安全，按 ENTER 开始采集，Ctrl+C 取消..." dummy

# ==================== 开始采集 ====================
echo ""
echo "🚀 开始数据采集..."
echo ""

# Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

LOG_DIR="$HOME/LX/pai0/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/record_b601_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志：$LOG_FILE"
echo ""

RECORD_CMD="lerobot-record \
  --robot.type=seeed_b601_rs_follower \
  --robot.id=$FOLLOWER_ID \
  --robot.port=$B601_FOLLOWER_PORT \
  --robot.can_adapter=socketcan \
  --robot.gravity_compensation=true \
  --robot.joint_directions=\"$B601_JOINT_DIRECTIONS\" \
  --robot.cameras=\"$CAMERAS\" \
  --teleop.type=rebot_102_leader \
  --teleop.id=$LEADER_ID \
  --teleop.port=$B601_LEADER_PORT \
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

# 数据保存位置：新建/续录都用显式 --dataset.root（同 04：root 被当作数据集目录本身，
# 新建时必须传完整带时间戳路径并用 --dataset.no_stamp=true 阻止重复追加时间戳）。
if [ "$RESUME_MODE" = true ]; then
  RECORD_CMD="$RECORD_CMD --resume=true --dataset.root=$DATA_ROOT/$DATASET_NAME"
else
  RECORD_CMD="$RECORD_CMD --dataset.no_stamp=true --dataset.root=$DATA_ROOT/$DATASET_NAME"
fi

# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂、lerobot-record 被
# SIGPIPE(141) 杀死而跳过断开清理（否则电机保持使能、从臂不释放）。
trap '' PIPE

# 执行命令（非零退出时提示后继续显示结果，不直接中断）
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
  for d in $(ls -dt "$DATA_ROOT"/b601_* 2>/dev/null || true); do
    if ! echo "$OLD_SESSIONS" | grep -qx "$d"; then
      SAVED_DIR="$d"
      break
    fi
  done
  if [ -z "$SAVED_DIR" ]; then
    SAVED_DIR="（未能自动定位，请到 $DATA_ROOT 下查看最新目录）"
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
echo "=========================================="
