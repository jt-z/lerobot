#!/bin/bash
# 单臂 SO-101 数据采集（从臂 so101_follower + 主臂 so101_leader）
# 调用 lerobot 官方 lerobot-record（lerobot/src/lerobot/scripts/lerobot_record.py）
#
# 参照 self_scripts/b601_single/02_record.sh 改写：robot/teleop 换成单臂 SO-101 类型、端口用 SO-101 的 by-id、
# 校准 id 用 so_follower/so_leader 下已有的 jt_*_arm_right.json。
#
# 支持中断继续录制（--resume）。
#
# 按键结束每个 episode（lerobot-record 内置）：
#   n/右方向键=结束本集 | r/左方向键=重录上一集 | q/Esc=停止整个录制
#
# 安全须知：
#   1. 采集前请确保主从臂已校准（校准文件见下方检查项）
#   2. 从臂连接后【使能变硬】，移动主臂 -> 从臂跟随（默认无速度限幅，跟手直连）
#   3. 退出（Ctrl+C）时从臂按自身校准回零并失能
#   4. 启动前把主臂和从臂都摆到零位（默认坐姿、夹爪闭合）
#   5. 紧急停止 = 切断从臂电源
#
# 用法：bash self_scripts/so101_single/record_101.sh
#   续录：bash self_scripts/so101_single/record_101.sh --resume so101_<时间戳> [追加集数]
#
# 可按需覆盖的环境变量：
#   FOLLOWER_PORT / LEADER_PORT  串口（默认用下面写死的 by-id 路径）
#   CAMERAS                      相机配置字符串；设为空字符串 = 不录相机（只录关节）
#   TASK_DESCRIPTION             任务描述
#   NUM_EPISODES / EPISODE_TIME / RESET_TIME / FPS
#   MAX_REL_TARGET               每步每关节最大相对位移（度），默认不限
#   DISPLAY_DATA                 true/false 是否开 rerun 可视化，默认 true

set -e
set -o pipefail

# ==================== 环境检查 ====================
if ! command -v lerobot-record >/dev/null 2>&1; then
  echo "未找到 lerobot-record，尝试激活 lerobot conda 环境..."
  for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
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
if [ "${1:-}" == "--resume" ]; then
  RESUME_MODE=true
  if [ -z "${2:-}" ]; then
    echo "❌ 错误：--resume 模式必须指定要续录的数据集名（带时间戳）"
    echo ""
    echo "   可用会话："
    ls -d "$HOME/LX/pai0/so101_data"/so101_* 2>/dev/null | sed 's|.*/||' | sort
    echo ""
    echo "   示例：bash self_scripts/so101_single/record_101.sh --resume so101_20260921_140000"
    exit 1
  fi
  RESUME_REPO_ID="$2"
  NUM_EPISODES_OVERRIDE="${3:-}"
  echo "🔄 恢复模式：续录数据集 $RESUME_REPO_ID"
fi

echo "=========================================="
echo "单臂 SO-101 数据采集（SO-101 从臂 + SO-101 主臂）"
echo "=========================================="
echo ""

# ==================== 硬件配置 ====================
# 校准文件 id：对应 ~/.cache/huggingface/lerobot/calibration/{robots/so_follower,teleoperators/so_leader}/<id>.json
FOLLOWER_ID="jt_follower_arm_right"
LEADER_ID="jt_leader_arm_right"

# by-id 稳定路径（ttyACM* 编号随插拔顺序变化，不写死）
SO101_FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00}"
SO101_LEADER_PORT="${LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034865-if00}"

# 每步每关节最大相对位移（度）。留空 = 不限幅（跟 lerobot 默认一致）
MAX_REL_TARGET="${MAX_REL_TARGET:-}"

# ==================== 摄像头配置 ====================
# 本套 SO-101 用【新增的两台 JYU2C】（2608006 / 2609007），与 B601 那三路（2608076/2607031/2607060）分开。
# 视角已确认：2609007 = hand（腕部/近景，跟着夹爪动）、2608006 = front（前视第三视角）。
# ⚠️ key 决定数据集 observation 字段名，训练/推理脚本要与之一致；
#    录制中途不要改（改了就得分数据集）。
# 不想录相机就设 CAMERAS=""（只录 6 个关节）。
# 注意：默认值必须单独放一个变量再引用（${CAMERAS-$DEFAULT_CAMERAS}），
# 不要写成 ${CAMERAS-{...}} —— bash 会在第一个 } 处截断参数展开，把 JSON 弄坏。
DEFAULT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2609007-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608006-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'
CAMERAS="${CAMERAS-$DEFAULT_CAMERAS}"

# ==================== 数据集配置 ====================
# 单 SO-101 数据根目录，每个 session 存到 $DATA_ROOT/so101_<时间戳>/
DATA_ROOT="$HOME/LX/pai0/so101_data"
mkdir -p "$DATA_ROOT"

if [ -n "$RESUME_REPO_ID" ]; then
  DATASET_NAME="$RESUME_REPO_ID"
else
  # 新建：脚本生成带时间戳的名字，配合 --dataset.no_stamp=true 保证目录名一致
  DATASET_NAME="so101_$(date +%Y%m%d_%H%M%S)"
fi
# 是否推送到 Hugging Face Hub（当前环境外网不可达，保持 false，数据只存本地）
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
# 国内镜像源（配合上面开关；网络可用时生效）
export HF_ENDPOINT=https://hf-mirror.com

# ⚠️ 任务描述（夹取不同角度的黄色香蕉，放到白色水果盘上）：须与后续训练/推理用的描述完全一致
# （与 04/05、act_inference.sh、smolvla_inference.sh 一样用英文，直接作为数据集 task 字段/推理 prompt）
TASK_DESCRIPTION="${TASK_DESCRIPTION:-Pick up the yellow banana placed at different angles and put it on the white fruit plate}"

NUM_EPISODES="${NUM_EPISODES:-120}"
# --resume 模式下允许用第 3 个命令行参数覆盖本次追加的集数
if [ -n "$NUM_EPISODES_OVERRIDE" ]; then
  NUM_EPISODES="$NUM_EPISODES_OVERRIDE"
fi
# 每个 episode 录制的安全上限（秒）：正常由按键 n / 右方向键提前结束并保存，
# 只有忘记按键时才会跑满这个时长（到点也会自动保存）。
EPISODE_TIME="${EPISODE_TIME:-300}"
RESET_TIME="${RESET_TIME:-5}"   # 重置环境时长（秒）；按 n / 右方向键可提前跳过
FPS="${FPS:-30}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"

if [ -z "$TASK_DESCRIPTION" ]; then
  echo "❌ 错误：TASK_DESCRIPTION 为空"
  echo "   请编辑本脚本填写任务描述（或用环境变量传入），例如："
  echo "   TASK_DESCRIPTION='Pick up the cube and place it in the box' bash $0"
  exit 1
fi

# ==================== 采集前检查 ====================
echo "1. 检查硬件..."
for port in "$SO101_FOLLOWER_PORT" "$SO101_LEADER_PORT"; do
  if [ ! -e "$port" ]; then
    echo "❌ 串口不存在 $port"
    echo "   当前串口设备：$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | tr '\n' ' ')"
    exit 1
  fi
  if [ ! -r "$port" ] || [ ! -w "$port" ]; then
    echo "⚠️  $port 无读写权限，尝试授权..."
    sudo chmod 666 "$port"
  fi
done
echo "✅ 两个串口均已连接"

# 舵机应答检查：ping 6 个 STS3215（id 1~6），确认臂已上电、端口确实对着 SO-101
# 说明：ping 返回 (型号, 通信结果, 错误码)，只有「通信结果 == 0」才算应答（超时返回型号 0）
if python -c "import scservo_sdk" >/dev/null 2>&1; then
  probe_port() {  # $1 = 串口；stdout = 应答的舵机 id；返回 1 = 应答不全
    python - "$1" <<'PY'
import sys
import scservo_sdk as s

port = sys.argv[1]
ph = s.PortHandler(port)
if not ph.openPort():
    print("open failed", file=sys.stderr)
    sys.exit(1)
ph.setBaudRate(1_000_000)
pk = s.PacketHandler(0)
hits = [i for i in range(1, 7) if pk.ping(ph, i)[1] == 0]
ph.closePort()
print(",".join(str(i) for i in hits))
sys.exit(0 if len(hits) == 6 else 1)
PY
  }
  for port in "$SO101_FOLLOWER_PORT" "$SO101_LEADER_PORT"; do
    if hits="$(probe_port "$port")"; then
      echo "✅ $port 舵机应答: $hits"
    else
      echo "❌ $port 舵机应答不全（应答: ${hits:-无}）"
      echo "   1) 确认该臂 12V 电源已开、USB 已插好"
      echo "   2) 确认这条串口对应的确实是 SO-101（不是别的设备）"
      exit 1
    fi
  done
else
  echo "⚠️  跳过舵机应答检查（当前 python 缺 scservo_sdk；请确认已 conda activate lerobot）"
fi

# 摄像头检查（同时打印 key↔设备 映射，key 写错/接错在开始前就能看出来）
echo ""
echo "2. 检查摄像头..."
CAM_KEYS=""
if [ -z "$CAMERAS" ]; then
  echo "  ⚠️  CAMERAS 为空：本次不录相机，只录关节"
else
  # 从 CAMERAS 里抽出 "key=路径"（要求与默认写法一致：一行一路相机）
  CAM_ENTRIES=$(echo "$CAMERAS" | sed -nE 's/^[[:space:]]*([A-Za-z0-9_]+): \{type: [a-z]+, index_or_path: ([^,]+),.*/\1=\2/p')
  if [ -z "$CAM_ENTRIES" ]; then
    echo "  ⚠️  解析不出相机条目（CAMERAS 可能写成了一行），跳过相机检查"
  fi
  CAM_OK=true
  for entry in $CAM_ENTRIES; do
    key="${entry%%=*}"
    cam="${entry#*=}"
    if [ -e "$cam" ]; then
      echo "  ✅ $key = $cam"
    else
      echo "  ❌ $key = $cam（设备不存在）"
      CAM_OK=false
    fi
    CAM_KEYS="${CAM_KEYS:+$CAM_KEYS, }$key"
  done
  if [ "$CAM_OK" = false ]; then
    echo "  请改脚本里的 CAMERAS，或设 CAMERAS=\"\" 只录关节"
    exit 1
  fi
fi

# 校准文件检查
echo ""
echo "3. 检查校准文件..."
CALIB_OK=true
for f in \
  "$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/${FOLLOWER_ID}.json" \
  "$HOME/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/${LEADER_ID}.json"; do
  if [ -f "$f" ]; then
    echo "  ✅ $f"
  else
    echo "  ❌ 缺失：$f（请先用 lerobot-calibrate 校准该臂，或改脚本里的 *_ID）"
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
  DATA_DIR="$DATA_ROOT/$DATASET_NAME"
  EXISTING_EPISODES=0
  echo "✅ 数据根目录：$DATA_ROOT"
  echo "   本次新建：$DATA_DIR"
fi

# ==================== 采集参数总览 ====================
echo ""
echo "=========================================="
echo "采集参数总览"
echo "=========================================="
echo "录制模式：$( [ "$RESUME_MODE" = true ] && echo "续录（--resume）" || echo "新建数据集" )"
echo "从臂：$SO101_FOLLOWER_PORT (id=$FOLLOWER_ID)"
echo "主臂：$SO101_LEADER_PORT (id=$LEADER_ID)"
echo "数据集名称：$DATASET_NAME"
echo "本地保存路径：$DATA_DIR"
echo "Episode 进度：已有 $EXISTING_EPISODES 集 + 本次录制 $NUM_EPISODES 集 = 共 $((EXISTING_EPISODES + NUM_EPISODES)) 集"
echo "任务描述：$TASK_DESCRIPTION"
echo "相机：$( [ -z "$CAMERAS" ] && echo "（不录）" || echo "${CAM_KEYS:-（见上方检查结果）}" )"
echo "每个 Episode 时长：按 n 提前结束并保存（安全上限 ${EPISODE_TIME}秒）"
echo "重置时长：${RESET_TIME}秒（可按键提前跳过）"
echo "采集频率：${FPS} Hz"
echo "=========================================="
echo ""
echo "📝 按键控制（录制过程中随时可用，注意焦点保持在终端窗口）："
echo "   n / 右方向键 = 当前 episode 演示完成，结束录制并保存"
echo "   r / 左方向键 = 重录上一个 episode"
echo "   q / Esc      = 停止整个录制"
echo ""
echo -e "   每个 episode 流程：开始录制 → 演示任务 → 按 n 结束并保存（忘按则 ${EPISODE_TIME}s 到点自动保存）→"
echo -e "   重置环境（${RESET_TIME}秒，可再按 n 跳过等待）→ 自动进入下一个 episode。"
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
LOG_FILE="$LOG_DIR/record_101_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志：$LOG_FILE"
echo ""

RECORD_CMD="lerobot-record \
  --robot.type=so101_follower \
  --robot.id=$FOLLOWER_ID \
  --robot.port=$SO101_FOLLOWER_PORT \
  --teleop.type=so101_leader \
  --teleop.id=$LEADER_ID \
  --teleop.port=$SO101_LEADER_PORT \
  --dataset.repo_id=$DATASET_NAME \
  --dataset.num_episodes=$NUM_EPISODES \
  --dataset.single_task=\"$TASK_DESCRIPTION\" \
  --dataset.fps=$FPS \
  --dataset.episode_time_s=$EPISODE_TIME \
  --dataset.reset_time_s=$RESET_TIME \
  --dataset.video=true \
  --dataset.rgb_encoder.vcodec=h264 \
  --dataset.push_to_hub=$PUSH_TO_HUB \
  --display_data=$DISPLAY_DATA \
  --display_compressed_images=false"

if [ -n "$CAMERAS" ]; then
  RECORD_CMD="$RECORD_CMD --robot.cameras=\"$CAMERAS\""
fi
if [ -n "$MAX_REL_TARGET" ]; then
  RECORD_CMD="$RECORD_CMD --robot.max_relative_target=$MAX_REL_TARGET"
fi

# 数据保存位置：新建/续录都用显式 --dataset.root（root 被当作数据集目录本身，
# 新建时必须传完整带时间戳路径并用 --dataset.no_stamp=true 阻止重复追加时间戳）。
if [ "$RESUME_MODE" = true ]; then
  RECORD_CMD="$RECORD_CMD --resume=true --dataset.root=$DATA_DIR"
else
  RECORD_CMD="$RECORD_CMD --dataset.no_stamp=true --dataset.root=$DATA_DIR"
fi

# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂、lerobot-record 被
# SIGPIPE(141) 杀死而跳过断开清理（否则电机保持使能、从臂不释放）。
trap '' PIPE

# 执行命令（非零退出时提示后继续显示结果，不直接中断）
set +e
eval $RECORD_CMD 2>&1 | tee "$LOG_FILE"
RECORD_EXIT=${PIPESTATUS[0]}
set -e
if [ $RECORD_EXIT -ne 0 ] && [ $RECORD_EXIT -ne 130 ]; then
  echo ""
  echo "⚠️  lerobot-record 非零退出（exit=$RECORD_EXIT）"
  if grep -q "There is no status packet" "$LOG_FILE"; then
    echo "    舵机总线失联（USB 串口还在，但 6 个舵机同时不应答）—— 常见原因："
    echo "    1) 从臂舵机串链/12V 电源接头松动（断点之后的整条链全静默，重新插紧）"
    echo "    2) 主从臂经多级 USB Hub 且与相机共用 -> 把两条臂直接插到主机 USB 口"
    echo "    已保存的 episode 不受影响，可用 --resume 续录。"
  else
    echo "    若录制过程已完成（日志显示 episode 已保存/视频已编码），"
    echo "    这通常是断开机械臂时的清理错误，数据不受影响。"
  fi
fi

# ==================== 采集完成 ====================
echo ""
echo "=========================================="
echo "✅ 数据采集结束"
echo "=========================================="

TOTAL_EPISODES=0
if [ -d "$DATA_DIR/meta" ]; then
  TOTAL_EPISODES=$(python3 -c "import json; print(json.load(open('$DATA_DIR/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
fi
SAVED_EPISODES=$((TOTAL_EPISODES - EXISTING_EPISODES))
if [ "$SAVED_EPISODES" -lt 0 ]; then SAVED_EPISODES=0; fi

echo "数据集名称：$DATASET_NAME"
echo "本地保存路径：$DATA_DIR"
echo "本次实际保存 Episode：$SAVED_EPISODES / 目标 $NUM_EPISODES"
echo "   （该 session 累计：$TOTAL_EPISODES 集）"
if [ "$PUSH_TO_HUB" = true ]; then
  echo "Hugging Face Hub 链接：https://huggingface.co/datasets/$DATASET_NAME"
else
  echo "上传状态：未推送到 Hub（PUSH_TO_HUB=false，数据仅保存在本地）"
fi
echo ""
echo "💡 提示：断开前确认数据已落盘；如录制过程中断，可续录："
echo "   bash self_scripts/so101_single/record_101.sh --resume $DATASET_NAME"
echo "=========================================="
