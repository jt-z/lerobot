#!/bin/bash
# 咖啡杯分区放置数据采集（单 B601-RS 从臂 + StarArm102 主臂）
#
# 任务（接咖啡流程的第二阶段，见 lx/声源.txt）：
#   从咖啡机出液口下方银色托盘上夹取【已经接好咖啡的杯子】，放到指定编号的放杯区域。
#   第一阶段「取杯 + 接咖啡 + 按按钮」不在本脚本范围内（用 02_record.sh 采）。
#
# 区域与 prompt：
#   区域 1/2/3 各对应一条英文 prompt（见下面 REGION_PROMPTS）。
#   ⚠️ lerobot-record 一次会话只能写一条 --dataset.single_task，所以本脚本按区域
#   【依次启动多次 lerobot-record】：第一次 create 新数据集，后面几次用 --resume
#   往同一个数据集追加 episode。每集的 task 由当次会话的 prompt 决定，
#   meta/tasks.parquet 最终会有 3 条 task —— 3 个区域的 prompt 天然落在同一份数据集里，
#   可直接拿去微调同一个 π0.5，不需要 lerobot-edit-dataset merge。
#
#   区域数按当前需求取 3。将来扩到 6 个 30° 分区时：给 REGION_PROMPTS 补 4/5/6 三条，
#   校验和默认区域列表会按数组长度自动跟着变。
#
# 用法：
#   bash self_scripts/b601_single/05_record_place_cup.sh                     # 新建数据集，依次录 1→2→3 区
#   bash self_scripts/b601_single/05_record_place_cup.sh --regions 1,3       # 只录 1、3 区
#   bash self_scripts/b601_single/05_record_place_cup.sh --region 2          # 只录 2 区
#   bash self_scripts/b601_single/05_record_place_cup.sh --episodes 30       # 每区 30 集（默认 40）
#   bash self_scripts/b601_single/05_record_place_cup.sh --resume b601_place_cup_20261109_101500
#   bash self_scripts/b601_single/05_record_place_cup.sh --resume b601_place_cup_20261109_101500 --region 2 --episodes 20
#
# 按键结束每个 episode（lerobot-record 内置）：
#   n/右方向键=结束本集并保存 | r/左方向键=丢弃当前这一集（不保存）重录 | q/Esc=停止整个录制
#
# 安全须知（与 02_record.sh 一致）：
#   1. 采集前请确保主从臂已校准（self_scripts/b601_common/02_follower_calibration.sh / 01_leader_calibration.sh）
#   2. 从臂连接后【使能变硬】，移动主臂 -> 从臂跟随（无速度限幅，跟手直连）
#   3. 退出（Ctrl+C / q）时从臂回零位坐姿（夹爪收至夹稳位，不会自动回收纳 0°，
#      下次启动前请手动把夹爪合到 0°，否则零点基准会偏移）
#   4. 紧急停止 = 切断 48V 电源
#   5. 启动前把主臂和从臂都摆到零位（默认坐姿、夹爪闭合）
#
# CAN：采集前自动复位 can0（down→1Mbps→up）并清空残留帧，避免上次异常退出留下的
#      垃圾帧把电机喂住、导致 connect() 报 "control ack timeout: comm_type=4"

set -e
set -o pipefail

# 设备参数层：按主机名自动选 b601_common/devices/<host>.sh（B601_DEVICE 可覆盖）
_B601_ENV="$(dirname "$(readlink -f "$0")")/../b601_common/_env.sh"
[ -f "$_B601_ENV" ] && . "$_B601_ENV"

# 数据根目录：默认落在代码仓根目录（B601_REPO_ROOT 由 _env.sh 导出），
# 每个 session 存到 $DATA_ROOT/b601_place_cup_<时间戳>/
DATA_ROOT="${B601_REPO_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}"

# ==================== 区域 → prompt ====================
# 中文原意：将咖啡机出液口下银色托盘上的咖啡杯夹取，放置在 N 号放杯区域。
# 润色要点（参考 02_record.sh / 03_collect_make_coffee.sh 的既有 prompt 风格）：
#   * 英文祈使句 "Pick up ... and place it in ..."；不写"then/after/please"这类冗余词
#   * 起点物体、目标位置都用固定短语，三条 prompt 只有区域编号不同
#     → 语言条件只靠编号区分，模型更容易学到"指令 → 轨迹"的对应关系
#   * 控制长度（约 20 词 / 25 token），避免被 π0.5 的文本 tokenizer 截断
#   * 训练、推理两边的 prompt 必须逐字一致（含大小写和空格）
# 数组下标 = 区域编号，[0] 占位不用。
REGION_PROMPTS=(
  ""
  "Pick up the coffee cup from the silver tray under the coffee machine spout and place it in cup placement area 1"
  "Pick up the coffee cup from the silver tray under the coffee machine spout and place it in cup placement area 2"
  "Pick up the coffee cup from the silver tray under the coffee machine spout and place it in cup placement area 3"
)
MAX_REGION=$(( ${#REGION_PROMPTS[@]} - 1 ))

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
usage() {
  cat <<'EOF'
用法：
  bash self_scripts/b601_single/05_record_place_cup.sh                     # 新建数据集，依次录 1→2→3 区
  bash self_scripts/b601_single/05_record_place_cup.sh --regions 1,3       # 只录 1、3 区
  bash self_scripts/b601_single/05_record_place_cup.sh --region 2          # 只录 2 区
  bash self_scripts/b601_single/05_record_place_cup.sh --episodes 30       # 每区 30 集（默认 40）
  bash self_scripts/b601_single/05_record_place_cup.sh --resume b601_place_cup_20261109_101500
  bash self_scripts/b601_single/05_record_place_cup.sh --resume b601_place_cup_20261109_101500 --region 2 --episodes 20

参数：
  --region / --regions 1,2,3   要采集的区域编号（逗号分隔），默认 1,2,3
  --episodes N                 每个区域本次采集的集数，默认 40
  --resume <repo_id>           续录已有数据集（需带完整时间戳名）
  -h, --help                   显示本帮助
EOF
}

REGION_LIST=""
RESUME_REPO_ID=""
NUM_EPISODES_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --region|--regions)
      [ -n "${2:-}" ] || { echo "❌ $1 需要参数，例如 --regions 1,2,3"; exit 1; }
      REGION_LIST="$2"; shift 2 ;;
    --episodes)
      [ -n "${2:-}" ] || { echo "❌ --episodes 需要参数，例如 --episodes 30"; exit 1; }
      NUM_EPISODES_OVERRIDE="$2"; shift 2 ;;
    --resume)
      [ -n "${2:-}" ] || { echo "❌ --resume 需要数据集完整 repo_id（带时间戳）"; exit 1; }
      RESUME_REPO_ID="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "❌ 未知参数：$1"; echo ""; usage; exit 1 ;;
  esac
done

# 默认按 1..MAX_REGION 顺序采集
if [ -z "$REGION_LIST" ]; then
  REGION_LIST=""
  for ((i = 1; i <= MAX_REGION; i++)); do
    REGION_LIST+="${REGION_LIST:+,}$i"
  done
fi

IFS=',' read -r -a REGIONS <<< "$REGION_LIST"
for r in "${REGIONS[@]}"; do
  if ! [[ "$r" =~ ^[0-9]+$ ]] || [ "$r" -lt 1 ] || [ "$r" -gt "$MAX_REGION" ]; then
    echo "❌ 区域编号非法：$r（当前只支持 1~$MAX_REGION）"
    exit 1
  fi
done

if [ -n "$NUM_EPISODES_OVERRIDE" ]; then
  if ! [[ "$NUM_EPISODES_OVERRIDE" =~ ^[0-9]+$ ]] || [ "$NUM_EPISODES_OVERRIDE" -lt 1 ]; then
    echo "❌ --episodes 必须是正整数"
    exit 1
  fi
fi

echo "=========================================="
echo "咖啡杯分区放置数据采集（B601 从臂 + StarArm102 主臂）"
echo "=========================================="
echo ""

# ==================== 硬件配置 ====================
FOLLOWER_ID="${B601_FOLLOWER_ID:-follower}"   # 从臂校准 id（对应 seeed_b601_rs_follower/<id>.json）
LEADER_ID="${B601_LEADER_ID:-leader}"         # 主臂校准 id（对应 rebot_102_leader/<id>.json）

# B601 从臂 = can0（SocketCAN）—— 由设备档案决定，可用 B601_FOLLOWER_PORT 覆盖
B601_FOLLOWER_PORT="${B601_FOLLOWER_PORT:-can0}"
# StarArm102 主臂串口：自动探测（候选口逐个 ping 舵机），可用 B601_LEADER_PORT 覆盖
B601_LEADER_PORT="${B601_LEADER_PORT:-}"
# kfzbox 本机适配：优先 udev 稳定别名（CH340 无序列号，ttyUSB* 会变）
if [ -z "$B601_LEADER_PORT" ] && [ -e /dev/rebot_leader ]; then B601_LEADER_PORT=/dev/rebot_leader; fi
LEADER_PORT_PROBE="$(dirname "$(readlink -f "$0")")/../b601_common/04_leader_port.py"

# 从臂方向修正（必须保留，否则 shoulder_lift/gripper 被限位裁到 0 不动）
# ⚠️ 不能用 "${VAR:-{...}}"：bash 在第一个 '}' 处结束展开，默认值必须走间接变量。
B601_JOINT_DIRECTIONS="${B601_JOINT_DIRECTIONS:-$B601_JOINT_DIRECTIONS_DEFAULT}"

# ==================== 摄像头配置 ====================
# 与 02_record.sh 一致：角色→序列号来自设备档案；缺失时由 b601_cameras_json 兜底并提示
CAMERAS="$(b601_cameras_json)"

# ==================== 数据集配置 ====================
mkdir -p "$DATA_ROOT"

# 每个区域的采集集数（三个区域保持一致，保证数据量均衡）
EPISODES_PER_REGION="${NUM_EPISODES_OVERRIDE:-40}"

if [ -n "$RESUME_REPO_ID" ]; then
  DATASET_NAME="$RESUME_REPO_ID"
  DATA_DIR="$DATA_ROOT/$DATASET_NAME"
else
  # 新建：脚本生成带时间戳的数据集名，配合 --dataset.no_stamp=true 保证目录名一致
  DATASET_NAME="b601_place_cup_$(date +%Y%m%d_%H%M%S)"
  DATA_DIR="$DATA_ROOT/$DATASET_NAME"
fi

# 是否推送到 Hugging Face Hub（当前环境外网不可达，建议保持 false，数据只存本地）
PUSH_TO_HUB=false
# 国内镜像源（配合上面开关；网络可用时生效）
export HF_ENDPOINT=https://hf-mirror.com

# 每个 episode 录制时长的安全上限（秒）；录制由按键结束，该值只是防无限录制的保险
EPISODE_TIME=600
RESET_TIME=5    # 重置环境时长（秒）；按 n / 右方向键可提前跳过等待
FPS=30

# ==================== 采集前检查 ====================
echo "1. 检查硬件..."
# B601 CAN：启动前一律复位接口并清空残留帧（与 01_teleop.sh / 02_record.sh 一致）
can_drain() {
  local port="$1"
  command -v candump >/dev/null 2>&1 || return 0
  timeout 0.3 candump "$port" >/dev/null 2>&1 || true
}

if ! ip link show "$B601_FOLLOWER_PORT" >/dev/null 2>&1; then
  echo "❌ $B601_FOLLOWER_PORT 不存在"; exit 1
fi

echo "♻️  复位 $B601_FOLLOWER_PORT（1Mbps）并清空残留 CAN 帧..."
sudo ip link set "$B601_FOLLOWER_PORT" down 2>/dev/null || true
sudo ip link set "$B601_FOLLOWER_PORT" type can bitrate 1000000 || {
  echo "❌ 配置 $B601_FOLLOWER_PORT 失败（是否被其它进程占用？）"; exit 1
}
sudo ip link set "$B601_FOLLOWER_PORT" up
sleep 0.5   # 给电机侧看门狗留时间丢掉上一条残留指令
can_drain "$B601_FOLLOWER_PORT"

if ! ip link show "$B601_FOLLOWER_PORT" 2>/dev/null | grep -q "state UP"; then
  echo "❌ $B601_FOLLOWER_PORT 未 UP —— 检查 PCAN-USB 是否插好、peak_usb 驱动是否加载"
  ip -details link show "$B601_FOLLOWER_PORT" 2>/dev/null | sed -n '1,6p'
  exit 1
fi
if ! ip -details link show "$B601_FOLLOWER_PORT" 2>/dev/null | grep -q "bitrate 1000000"; then
  echo "⚠️  $B601_FOLLOWER_PORT 的 bitrate 不是 1000000，请确认总线速率"
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
echo "✅ 硬件就绪（从臂 $B601_FOLLOWER_PORT 已复位并清空残留帧；主臂串口：$B601_LEADER_PORT）"

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
if [ -n "$RESUME_REPO_ID" ]; then
  if [ -d "$DATA_DIR/meta" ]; then
    EXISTING_EPISODES=$(python3 -c "import json; print(json.load(open('$DATA_DIR/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
    echo "✅ 续录目录已找到：$DATA_DIR"
    echo "   已有 $EXISTING_EPISODES 个 episode，本次每个区域追加 $EPISODES_PER_REGION 个"
  else
    echo "❌ 错误：续录目录不存在或缺少 meta/：$DATA_DIR"
    echo ""
    echo "   可用会话："
    ls -d "$DATA_ROOT"/b601_place_cup_* 2>/dev/null | sed 's|.*/||' | sort
    exit 1
  fi
else
  echo "✅ 数据根目录：$DATA_ROOT"
  echo "   本次新建数据集：$DATA_DIR"
fi

# ==================== 采集参数总览 ====================
echo ""
echo "=========================================="
echo "采集参数总览"
echo "=========================================="
echo "录制模式：$( [ -n "$RESUME_REPO_ID" ] && echo "续录（--resume）" || echo "新建数据集" )"
echo "数据集名称：$DATASET_NAME"
echo "本地保存路径：$DATA_DIR"
echo "采集区域：${REGIONS[*]}（共 ${#REGIONS[@]} 个）"
for r in "${REGIONS[@]}"; do
  echo "  区域 $r：每个区域 $EPISODES_PER_REGION 集"
  echo "          prompt: ${REGION_PROMPTS[$r]}"
done
echo "本次计划总集数：$(( ${#REGIONS[@]} * EPISODES_PER_REGION ))"
echo "每个 Episode 时长：由按键结束（安全上限 ${EPISODE_TIME}秒）"
echo "重置时长：${RESET_TIME}秒（可按键提前跳过）"
echo "采集频率：${FPS} Hz"
echo "=========================================="
echo ""
echo -e "📝 按键控制（录制过程中随时可用，注意焦点保持在终端窗口）："
echo -e "   n / 右方向键 = 当前 episode 演示完成，结束录制并保存"
echo -e "   r / 左方向键 = 丢弃当前这一集（不保存）并重录"
echo -e "   q / Esc      = 停止整个录制（本区剩余集数不再录，脚本会停下，不会自动跳到下一区）"
echo ""
echo -e "   每个 episode 流程：开始录制 → 从托盘夹杯放到目标区域 → 按 n 结束并保存 →"
echo -e "   重置环境（把杯子放回托盘、机械臂回起点）→ 自动进入下一个 episode。"
echo -e "   ⚠️ 切换区域时会重新启动一次 lerobot-record（数据集用 --resume 续写），"
echo -e "      每次都要重新连接机械臂，属正常现象。"
echo ""

# ==================== 开始采集 ====================
# Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"

# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂、lerobot-record 被
# SIGPIPE(141) 杀死而跳过断开清理（否则电机保持使能、从臂不释放）。
trap '' PIPE

declare -a SAVED_PER_REGION
TOTAL_SAVED=0

for region in "${REGIONS[@]}"; do
  prompt="${REGION_PROMPTS[$region]}"
  echo ""
  echo "=========================================="
  echo "▶ 准备录制区域 $region"
  echo "  prompt: $prompt"
  echo "=========================================="
  read -p "把已接好咖啡的杯子放回出液口下方银色托盘、机械臂回到起点、主从臂在零位，按 ENTER 开始（Ctrl+C 取消）..." dummy

  # 本区开始前的已有集数（用于统计本次实际保存量）
  BEFORE_EPISODES=0
  if [ -f "$DATA_DIR/meta/info.json" ]; then
    BEFORE_EPISODES=$(python3 -c "import json; print(json.load(open('$DATA_DIR/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
  fi

  LOG_FILE="$LOG_DIR/place_cup_r${region}_$(date +%Y%m%d_%H%M%S).log"
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
    --dataset.num_episodes=$EPISODES_PER_REGION \
    --dataset.single_task=\"$prompt\" \
    --dataset.fps=$FPS \
    --dataset.episode_time_s=$EPISODE_TIME \
    --dataset.reset_time_s=$RESET_TIME \
    --dataset.video=true \
    --dataset.rgb_encoder.vcodec=h264 \
    --dataset.push_to_hub=$PUSH_TO_HUB \
    --display_data=true \
    --display_compressed_images=false"

  # 第一次 create 新数据集；之后（以及 --resume）都续写到同一目录。
  # root 被当作数据集目录本身，新建时必须传完整带时间戳路径并用 --no_stamp=true
  # 阻止框架再追加一次时间戳。
  if [ -f "$DATA_DIR/meta/info.json" ]; then
    RECORD_CMD="$RECORD_CMD --resume=true --dataset.root=$DATA_DIR"
  else
    RECORD_CMD="$RECORD_CMD --dataset.no_stamp=true --dataset.root=$DATA_DIR"
  fi

  # 非零退出时提示后继续统计，不直接中断（断开机械臂的清理错误很常见，数据通常没问题）
  set +e
  eval $RECORD_CMD 2>&1 | tee "$LOG_FILE"
  REGION_EXIT=${PIPESTATUS[0]}
  set -e
  if [ $REGION_EXIT -ne 0 ]; then
    echo ""
    echo "⚠️  lerobot-record 非零退出（exit=$REGION_EXIT）"
    echo "    若日志显示 episode 已保存/视频已编码，通常是断开机械臂时的清理错误，数据不受影响。"
    echo "    机械臂可能需要断电重启以复位过载保护。"
  fi

  if [ ! -f "$DATA_DIR/meta/info.json" ]; then
    echo ""
    echo "❌ 数据集没有生成成功（$DATA_DIR/meta/info.json 缺失），中止后续区域，请查看日志："
    echo "   $LOG_FILE"
    exit 1
  fi

  AFTER_EPISODES=$(python3 -c "import json; print(json.load(open('$DATA_DIR/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
  SAVED=$((AFTER_EPISODES - BEFORE_EPISODES))
  [ "$SAVED" -lt 0 ] && SAVED=0
  SAVED_PER_REGION[$region]=$(( ${SAVED_PER_REGION[$region]:-0} + SAVED ))
  TOTAL_SAVED=$((TOTAL_SAVED + SAVED))

  echo ""
  echo "✅ 区域 $region 本次保存 $SAVED 集（该区本次累计 ${SAVED_PER_REGION[$region]} 集；数据集累计 $AFTER_EPISODES 集）"

  if [ "$SAVED" -lt "$EPISODES_PER_REGION" ]; then
    echo "⚠️  本区域只保存了 $SAVED/$EPISODES_PER_REGION 集（提前结束：按了 q / Esc 或录制中断），停止后续区域。"
    echo "    补录命令：bash self_scripts/b601_single/05_record_place_cup.sh --resume $DATASET_NAME --region $region --episodes $((EPISODES_PER_REGION - SAVED))"
    break
  fi
done

# ==================== 采集完成 ====================
echo ""
echo "=========================================="
echo "✅ 分区放置数据采集结束"
echo "=========================================="
echo "数据集名称：$DATASET_NAME"
echo "本地保存路径：$DATA_DIR"
echo "本次保存：$TOTAL_SAVED 集（合计 ${#REGIONS[@]} 个区域 × 目标 ${EPISODES_PER_REGION} 集）"
echo ""
echo "按区域统计（本次运行）："
for r in "${REGIONS[@]}"; do
  printf "  区域 %s：%s 集\n" "$r" "${SAVED_PER_REGION[$r]:-0}"
done

TOTAL_EPISODES=0
if [ -d "$DATA_DIR/meta" ]; then
  TOTAL_EPISODES=$(python3 -c "import json; print(json.load(open('$DATA_DIR/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
fi
echo ""
echo "数据集累计 episode：$TOTAL_EPISODES"
if [ "$PUSH_TO_HUB" = true ]; then
  echo "Hugging Face Hub 链接："
  echo "  https://huggingface.co/datasets/$DATASET_NAME"
else
  echo "上传状态：未推送到 Hub（PUSH_TO_HUB=false，数据仅保存在本地）"
fi
echo ""
echo "💡 提示："
echo "  1. 续录（补某个区域的数据、保持各区均衡）："
echo "     bash self_scripts/b601_single/05_record_place_cup.sh --resume $DATASET_NAME --region <N> --episodes <集数>"
echo "  2. 三个区域的 prompt 已写进同一份数据集的 meta/tasks.parquet；"
echo "     训练/推理时使用的 prompt 必须与上面逐字一致。"
echo "  3. 后续接入环形麦克风声源定位时，只需把声源角度映射成区域编号 1~$MAX_REGION，"
echo "     再取出对应 prompt 送进策略（采集阶段先手动切 prompt 验证）。"
echo "=========================================="
