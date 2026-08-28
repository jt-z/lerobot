#!/bin/bash
# 数据采集脚本：双臂夹取纸杯放到咖啡机银色托盘，右臂按按钮，再把杯子放回桌面
# 任务：Pick up the paper cup with both arms, place it on the silver tray of the coffee machine,
#       press the button with the right arm (red light on), wait about 4 seconds, release the button
#       (red light off), then place the cup on the table with the left arm
# 创建日期：2026-08-03
#
# 支持中断继续录制：
#   如果录制过程中断，可以添加 --resume 参数继续录制
#   脚本会自动从上次中断的 episode 继续
#
# 按键结束每个 episode（2026-08-27 新增）：
#   每个 episode 录制由操作者用按键结束，不再按固定时长结束：
#   演示完任务后按 n（或右方向键）立即结束当前 episode 并保存。
#   这是 lerobot-record 内置的键盘控制（n=next / r=re-record / q=quit），无需改框架代码。

set -e  # 遇到错误立即退出

# ==================== 命令行参数 ====================
# --resume 模式：继续录制已存在的数据集
# 用法：bash collect_make_coffee.sh --resume hellozjt/coffee_cup_button_<时间戳> [追加集数]
# 注意：必须传入完整的 repo_id（带时间戳），因为每次新建数据集时框架会自动加时间戳后缀。
#       第 3 个参数可选，指定本次追加的 episode 数（默认 $NUM_EPISODES）。
#       示例：bash collect_make_coffee.sh --resume hellozjt/coffee_cup_button_20260826_232220 100
RESUME_MODE=false
RESUME_REPO_ID=""
NUM_EPISODES_OVERRIDE=""
if [ "$1" == "--resume" ]; then
  RESUME_MODE=true
  if [ -z "$2" ]; then
    echo "❌ 错误：--resume 模式必须指定要续录的数据集完整 repo_id（带时间戳）"
    echo ""
    echo "   可用会话："
    ls -d "$HOME/.cache/huggingface/lerobot/hellozjt/coffee_cup_button_"* 2>/dev/null | sed 's|.*/||' | sort
    echo ""
    echo "   示例：bash collect_make_coffee.sh --resume hellozjt/coffee_cup_button_20260826_232220"
    echo "   示例：bash collect_make_coffee.sh --resume hellozjt/coffee_cup_button_20260826_232220 100  # 追加100集"
    exit 1
  fi
  RESUME_REPO_ID="$2"
  NUM_EPISODES_OVERRIDE="${3:-}"
  echo "🔄 恢复模式：续录数据集 $RESUME_REPO_ID"
fi

echo "=========================================="
echo "双臂数据采集 - 纸杯放到咖啡机银色托盘并按按钮"
echo "=========================================="
echo ""

# ==================== 硬件配置 ====================
# 端口映射（2026-08-26 实测，与 teleoperate_dual_so101.sh 一致）：
#   左从臂 = USB 序列号 5C82108837
#   右从臂 = USB 序列号 5B61034841
#   左主臂 = USB 序列号 5C82106862（已对调修正）
#   右主臂 = USB 序列号 5B61034865（已对调修正）
# 注意：全部使用 /dev/serial/by-id 稳定路径，ttyACM 编号随插拔顺序变化不可靠。
LEFT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82108837-if00"
RIGHT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00"
LEFT_LEADER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82106862-if00"
RIGHT_LEADER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034865-if00"

# ==================== 摄像头配置 ====================
# 摄像头映射（2026-08-26 实测，与 teleoperate_dual_so101.sh 一致）：
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
RIGHT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== 数据集配置 ====================
# 新建模式：使用基础名字（框架会自动加时间戳）
# --resume 模式：使用传入的完整 repo_id（带时间戳）
if [ -n "$RESUME_REPO_ID" ]; then
  DATASET_NAME="$RESUME_REPO_ID"
else
  DATASET_NAME="hellozjt/coffee_cup_button"
fi
# 是否推送到 Hugging Face Hub
# 当前环境外网不可达（huggingface.co 与 hf-mirror.com 均超时），建议保持 false，
# 数据只保存在本地；网络恢复后改为 true 即可自动上传。
PUSH_TO_HUB=false
# 国内镜像源（配合上面开关；网络可用时生效）
export HF_ENDPOINT=https://hf-mirror.com

TASK_DESCRIPTION="Pick up the paper cup with both arms, place it on the silver tray of the coffee machine, press the button with the right arm (red light on), wait about 4 seconds, release the button (red light off), then place the cup on the table with the left arm"
NUM_EPISODES=50
# --resume 模式下允许用第 3 个命令行参数覆盖本次追加的集数
if [ -n "$NUM_EPISODES_OVERRIDE" ]; then
  NUM_EPISODES="$NUM_EPISODES_OVERRIDE"
fi
# 每个 episode 录制时长的安全上限（秒）
# 录制由按键结束：演示完任务后按 n（或右方向键）立即结束当前 episode。
# 该值只是防止忘记按键时无限录制的保险，实际时长取决于操作者何时按键。
EPISODE_TIME=600
RESET_TIME=5    # 重置环境时长（秒）；按 n / 右方向键可提前跳过等待
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

# ==================== 数据集保存位置检查 ====================
echo ""
echo "4. 检查数据集保存位置..."
DATA_ROOT="$HOME/.cache/huggingface/lerobot"
if [ "$RESUME_MODE" = true ]; then
  DATA_DIR="$DATA_ROOT/$DATASET_NAME"
  if [ -d "$DATA_DIR" ]; then
    # 读取框架维护的 total_episodes（meta/info.json），这才是准确集数。
    # 不能用 find videos -name 'episode_*.mp4'：v3.0 实际命名是 chunk-000/file-XXX.mp4，恒为 0。
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
  OLD_SESSIONS=$(ls -d "$DATA_ROOT"/hellozjt/coffee_cup_button_* 2>/dev/null || true)
  EXISTING_SESSIONS=$(echo "$OLD_SESSIONS" | grep -c . 2>/dev/null || echo 0)
  echo "✅ 数据根目录：$DATA_ROOT/hellozjt/"
  echo "   该目录下已有 $EXISTING_SESSIONS 个历史 session（本次会新建一个带时间戳的目录，不覆盖旧数据）"
fi

# ==================== 采集参数总览 ====================
echo ""
echo "=========================================="
echo "采集参数总览"
echo "=========================================="
echo "录制模式：$(if [ "$RESUME_MODE" = true ]; then echo '续录（--resume）'; else echo '新建数据集'; fi)"
echo "数据集名称：$DATASET_NAME"
echo "本地保存路径：$DATA_DIR"
echo "Episode 进度：已有 $EXISTING_EPISODES 集 + 本次录制 $NUM_EPISODES 集 = 共 $((EXISTING_EPISODES + NUM_EPISODES)) 集"
echo "任务描述：$TASK_DESCRIPTION"
echo "每个 Episode 时长：由按键结束（安全上限 ${EPISODE_TIME}秒）"
echo "重置时长：${RESET_TIME}秒（可按键提前跳过）"
echo "采集频率：${FPS} Hz"
echo "预计本次时长：取决于演示节奏，总时长不固定"
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

# 确认开始
read -p "按 ENTER 开始数据采集（每个 episode 演示完成后按 n 结束录制），按 Ctrl+C 取消..." dummy

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
  --dataset.rgb_encoder.vcodec=h264 \
  --dataset.push_to_hub=$PUSH_TO_HUB \
  --display_data=true \
  --display_compressed_images=false"

# 如果是恢复模式，添加 --resume 参数
# 注意：resume 的 --dataset.root 必须指向数据集完整目录（含 repo_id 子路径），
#       不能只给缓存根目录，否则本地元数据加载失败会触发 Hub 下载导致卡住。
if [ "$RESUME_MODE" = true ]; then
  RECORD_CMD="$RECORD_CMD --resume=true --dataset.root=$DATA_ROOT/$DATASET_NAME"
fi

# 执行命令
# 注意：录制完成后断开机械臂时，若夹爪处于过载状态可能报错（Overload error），
#       导致 lerobot-record 非零退出（core dump）。此时数据已保存，不影响使用，
#       因此这里不因退出码直接中断，而是提示后继续显示结果。
set +e
eval $RECORD_CMD
RECORD_EXIT=$?
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
echo "✅ 数据采集完成！"
echo "=========================================="

# 识别实际保存目录
if [ "$RESUME_MODE" = true ]; then
  SAVED_DIR="$DATA_DIR"
else
  # 新建模式：找本次开始前不存在、最新的 session 目录
  SAVED_DIR=""
  for d in $(ls -dt "$DATA_ROOT"/hellozjt/coffee_cup_button_* 2>/dev/null || true); do
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
echo "  ./collect_make_coffee.sh --resume $DATASET_NAME"
echo ""
echo "下一步："
echo "  1. 访问上述链接查看数据集"
echo "  2. 检查数据质量"
echo "  3. 开始训练 ACT 模型"
echo "=========================================="
