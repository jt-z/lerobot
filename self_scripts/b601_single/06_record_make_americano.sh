#!/bin/bash
# 全自动咖啡机制作美式咖啡数据采集（单 B601-RS 从臂 + StarArm102 主臂）
#
# 任务（接咖啡流程的第 4 条任务，排在 05_record_place_cup.sh 分区放杯之后）：
#   用夹爪尖端按动全自动咖啡机面板按钮，把出杯量设为【中杯】、咖啡浓度设为【中等】，
#   确认设置生效后启动萃取，制作一杯美式咖啡。
#
# prompt：
#   只有一条（整段任务是一个连续演示），见下面 TASK_DESCRIPTION。
#   ⚠️ 训练/推理用的 prompt 必须与这里逐字一致（含大小写和空格）；采集完成后
#   记得把同一条文本加到 lx/pi05_inference.sh 的 TASKS 数组里。
#
# 用法：
#   bash self_scripts/b601_single/06_record_make_americano.sh                  # 新建数据集，默认 120 集
#   bash self_scripts/b601_single/06_record_make_americano.sh --episodes 60    # 本次录 60 集
#   bash self_scripts/b601_single/06_record_make_americano.sh --resume b601_americano_20261117_201500
#   bash self_scripts/b601_single/06_record_make_americano.sh --resume b601_americano_20261117_201500 --episodes 20
#
# 按键结束每个 episode（lerobot-record 内置）：
#   n/右方向键=结束本集并保存 | r/左方向键=丢弃当前这一集（不保存）重录 | q/Esc=停止整个录制
#
# 安全须知（与 02_record.sh / 05_record_place_cup.sh 一致）：
#   1. 采集前请确保主从臂已校准（self_scripts/b601_common/02_follower_calibration.sh / 01_leader_calibration.sh）
#   2. 从臂连接后【使能变硬】，移动主臂 -> 从臂跟随（无速度限幅，跟手直连）
#   3. 退出（Ctrl+C / q）时从臂回零位坐姿（夹爪收至夹稳位，不会自动回收纳 0°，
#      下次启动前请手动把夹爪合到 0°，否则零点基准会偏移）
#   4. 紧急停止 = 切断 48V 电源
#   5. 启动前把主臂和从臂都摆到零位（默认坐姿、夹爪闭合）
#   6. 按按钮时用夹爪尖端轻按，不要压在面板上持续用力；每集结束后把咖啡机的
#      出杯量/浓度设置复位（长按返回默认或按取消键），否则下一集会从当前设置开始
#
# CAN：采集前自动复位 can0（down→1Mbps→up）并清空残留帧，避免上次异常退出留下的
#      垃圾帧把电机喂住、导致 connect() 报 "control ack timeout: comm_type=4"

set -e
set -o pipefail

# 设备参数层：按主机名自动选 b601_common/devices/<host>.sh（B601_DEVICE 可覆盖）
_B601_ENV="$(dirname "$(readlink -f "$0")")/../b601_common/_env.sh"
[ -f "$_B601_ENV" ] && . "$_B601_ENV"

# 数据根目录：默认落在代码仓根目录（B601_REPO_ROOT 由 _env.sh 导出），
# 每个 session 存到 $DATA_ROOT/b601_americano_<时间戳>/（仓库 .gitignore 已忽略 /b601_*/）
DATA_ROOT="${B601_REPO_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}"

# ==================== 任务文本 ====================
# 中文原意：操作全自动咖啡机制作美式咖啡，按动按钮控制为中杯水量和中等咖啡浓度。
# 润色要点（与 02 / 05 的 prompt 风格一致）：
#   * 英文祈使句，不用 "please/then/after" 这类冗余词
#   * 动作 + 目标设置（中杯 / 中等浓度）+ 启动制作，全部写进一句，避免分阶段歧义
#   * 控制长度（约 30 词 / 60 token 以内），避免被 π0.5 的文本 tokenizer 截断
TASK_DESCRIPTION="Use the gripper tip to press the buttons on the automatic coffee machine to select a medium cup size and medium coffee strength, then start brewing an Americano"

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
  bash self_scripts/b601_single/06_record_make_americano.sh                  # 新建数据集，默认 120 集
  bash self_scripts/b601_single/06_record_make_americano.sh --episodes 60    # 本次录 60 集
  bash self_scripts/b601_single/06_record_make_americano.sh --resume b601_americano_20261117_201500
  bash self_scripts/b601_single/06_record_make_americano.sh --resume b601_americano_20261117_201500 --episodes 20

参数：
  --episodes N                 本次采集集数，默认 120
  --resume <repo_id>           续录已有数据集（需带完整时间戳名）
  -h, --help                   显示本帮助
EOF
}

NUM_EPISODES_OVERRIDE=""
RESUME_REPO_ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --episodes)
      [ -n "${2:-}" ] || { echo "❌ --episodes 需要参数，例如 --episodes 60"; exit 1; }
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

if [ -n "$NUM_EPISODES_OVERRIDE" ]; then
  if ! [[ "$NUM_EPISODES_OVERRIDE" =~ ^[0-9]+$ ]] || [ "$NUM_EPISODES_OVERRIDE" -lt 1 ]; then
    echo "❌ --episodes 必须是正整数"
    exit 1
  fi
fi

echo "=========================================="
echo "全自动咖啡机制作美式数据采集（B601 从臂 + StarArm102 主臂）"
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

NUM_EPISODES="${NUM_EPISODES_OVERRIDE:-120}"

if [ -n "$RESUME_REPO_ID" ]; then
  DATASET_NAME="$RESUME_REPO_ID"
  DATA_DIR="$DATA_ROOT/$DATASET_NAME"
else
  # 新建：脚本生成带时间戳的数据集名，配合 --dataset.no_stamp=true 保证目录名一致
  DATASET_NAME="b601_americano_$(date +%Y%m%d_%H%M%S)"
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
# B601 CAN：启动前一律复位接口并清空残留帧（与 01_teleop.sh / 02 / 05 一致）
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
    echo "   已有 $EXISTING_EPISODES 个 episode，本次将追加 $NUM_EPISODES 个"
  else
    echo "❌ 错误：续录目录不存在或缺少 meta/：$DATA_DIR"
    echo ""
    echo "   可用会话："
    ls -d "$DATA_ROOT"/b601_americano_* 2>/dev/null | sed 's|.*/||' | sort
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
echo "本次计划集数：$NUM_EPISODES"
echo "任务描述：$TASK_DESCRIPTION"
echo "每个 Episode 时长：由按键结束（安全上限 ${EPISODE_TIME}秒）"
echo "重置时长：${RESET_TIME}秒（可按键提前跳过）"
echo "采集频率：${FPS} Hz"
echo "=========================================="
echo ""
echo -e "📝 按键控制（录制过程中随时可用，注意焦点保持在终端窗口）："
echo -e "   n / 右方向键 = 当前 episode 演示完成，结束录制并保存"
echo -e "   r / 左方向键 = 丢弃当前这一集（不保存）并重录"
echo -e "   q / Esc      = 停止整个录制"
echo ""
echo -e "   每个 episode 流程：开始录制 → 按中杯水量、中等浓度、启动萃取 → 按 n 结束并保存 →"
echo -e "   重置环境（把咖啡机设置复位、机械臂回起点）→ 自动进入下一个 episode。"
echo ""
read -p "把咖啡机恢复到待机状态、机械臂回到起点、主从臂在零位，按 ENTER 开始（Ctrl+C 取消）..." dummy

# ==================== 开始采集 ====================
# Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/americano_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志：$LOG_FILE"
echo ""

# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂、lerobot-record 被
# SIGPIPE(141) 杀死而跳过断开清理（否则电机保持使能、从臂不释放）。
trap '' PIPE

# 本集开始前的已有集数（用于统计本次实际保存量）
BEFORE_EPISODES=0
if [ -f "$DATA_DIR/meta/info.json" ]; then
  BEFORE_EPISODES=$(python3 -c "import json; print(json.load(open('$DATA_DIR/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
fi

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

# 新建时用 --dataset.root 指定完整带时间戳路径，并用 --no_stamp=true 阻止框架重复追加时间戳；
# 续录时用 --resume=true 写回同一目录。
if [ -f "$DATA_DIR/meta/info.json" ]; then
  RECORD_CMD="$RECORD_CMD --resume=true --dataset.root=$DATA_DIR"
else
  RECORD_CMD="$RECORD_CMD --dataset.no_stamp=true --dataset.root=$DATA_DIR"
fi

# 非零退出时提示后继续统计，不直接中断（断开机械臂的清理错误很常见，数据通常没问题）
set +e
eval $RECORD_CMD 2>&1 | tee "$LOG_FILE"
RECORD_EXIT=${PIPESTATUS[0]}
set -e
if [ $RECORD_EXIT -ne 0 ]; then
  echo ""
  echo "⚠️  lerobot-record 非零退出（exit=$RECORD_EXIT）"
  echo "    若日志显示 episode 已保存/视频已编码，通常是断开机械臂时的清理错误，数据不受影响。"
  echo "    机械臂可能需要断电重启以复位过载保护。"
fi

# ==================== 采集完成 ====================
TOTAL_EPISODES=0
if [ -d "$DATA_DIR/meta" ]; then
  TOTAL_EPISODES=$(python3 -c "import json; print(json.load(open('$DATA_DIR/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
fi
SAVED_EPISODES=$((TOTAL_EPISODES - BEFORE_EPISODES))
[ "$SAVED_EPISODES" -lt 0 ] && SAVED_EPISODES=0

echo ""
echo "=========================================="
echo "✅ 美式咖啡数据采集结束"
echo "=========================================="
echo "数据集名称：$DATASET_NAME"
echo "本地保存路径：$DATA_DIR"
echo "本次实际保存 Episode：$SAVED_EPISODES / 目标 $NUM_EPISODES"
echo "   （该数据集累计：$TOTAL_EPISODES 集）"
if [ "$PUSH_TO_HUB" = true ]; then
  echo "Hugging Face Hub 链接："
  echo "  https://huggingface.co/datasets/$DATASET_NAME"
else
  echo "上传状态：未推送到 Hub（PUSH_TO_HUB=false，数据仅保存在本地）"
fi
echo ""
echo "💡 提示："
echo "  1. 续录（中断后补齐集数）："
echo "     bash self_scripts/b601_single/06_record_make_americano.sh --resume $DATASET_NAME --episodes <集数>"
echo "  2. 训练/推理使用的 prompt 必须与下面这条逐字一致："
echo "     $TASK_DESCRIPTION"
echo "  3. 若要把本任务并入 b601_mixed 一起训练，记得同步更新 lx/pi05_inference.sh 的 TASKS 数组。"
echo "=========================================="
