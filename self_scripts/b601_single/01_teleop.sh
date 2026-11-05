#!/bin/bash
# 单臂遥操作测试：B601-RS 从臂（RobStride，SocketCAN）+ StarArm102 主臂
# 参照 self_scripts/b601_so101_bimanual/01_teleop_test.sh（原为 B601+SO-101 异构双臂版，本脚本为 B601 单臂版）
# 使用单臂 lerobot 类型：
#   robot  = seeed_b601_rs_follower（B601-RS，can0，RobStride 电机）
#   teleop = rebot_102_leader（StarArm102 /dev/ttyUSB0，FashionStar UART 舵机）
#
# 依赖：lerobot conda 环境（含 lerobot-robot-seeed-b601 包，脚本会自动尝试激活）
# 用法：bash self_scripts/b601_single/01_teleop.sh
# 相机：默认按设备档案开相机并在 rerun 窗口显示画面（B601_DISPLAY_DATA=0 关闭）
# CAN：启动前自动复位 can0（down→1Mbps→up）并清空残留帧，避免上次异常退出留下的
#      垃圾帧把电机喂住、导致 connect() 报 "control ack timeout: comm_type=4"
#
# ⚠️ 安全须知（从臂使能后会变硬，务必先读）：
#   1. 连接后从臂电机【使能变硬】
#   2. 移动主臂 -> 从臂跟随；已去除速度限幅（跟手直连）
#      仅保留 B601 关节硬限位（shoulder_lift/elbow_flex/gripper 等）+ 温度保护
#   3. 退出（Ctrl+C）时从臂会回零位（safe_zero）并失能
#   4. 紧急停止 = 切断 48V 电源
#   5. 启动前务必把【主臂和从臂】都摆到零位（默认坐姿、夹爪闭合），
#      否则第一帧会把从臂快速拉向主臂当前姿态，注意安全！
#   6. 首次测试请小幅、慢速移动主臂，确认方向/比例正常再加大幅度
#
# 若某个关节运动方向相反（或 shoulder_lift/gripper 被限位裁到 0 不动），
# 调整下方 FOLLOWER_JOINT_DIRECTIONS 中对应关节的符号即可。

set -e
set -o pipefail

# 设备参数层 (2026-11-02)：按主机名自动选 b601_common/devices/<host>.sh（B601_DEVICE 可覆盖）
_B601_ENV="$(dirname "$(readlink -f "$0")")/../b601_common/_env.sh"
[ -f "$_B601_ENV" ] && . "$_B601_ENV"

# ==================== 硬件配置 ====================
# 全部参数走设备档案 b601_common/devices/<host>.sh（按主机名自动选），这里只留兜底默认值。
#   kfzbox：follower_kfzbox / leader_kfzbox，主臂 udev 别名 /dev/rebot_leader，三路相机（hand front top；相机别都挂同一个 USB hub）
#   z790  ：follower_z790  / leader_z790，主臂 /dev/ttyUSB0，三路相机
FOLLOWER_ID="${B601_FOLLOWER_ID:-follower}"     # 校准文件 seeed_b601_rs_follower/<id>.json
LEADER_ID="${B601_LEADER_ID:-leader}"           # 校准文件 rebot_102_leader/<id>.json

# 从臂 = B601-RS（SocketCAN）
FOLLOWER_PORT="${B601_FOLLOWER_PORT:-can0}"
FOLLOWER_ADAPTER="${B601_FOLLOWER_ADAPTER:-socketcan}"
CAN_BITRATE="${B601_CAN_BITRATE:-1000000}"
# 重力补偿（对 6 个 RS 关节做前馈），开启后手感更轻，建议保持 true
GRAVITY_COMPENSATION="${B601_GRAVITY_COMPENSATION:-true}"
# 从臂关节方向修正（实测值，必须保留，否则 shoulder_lift/gripper 会被限位裁到 0 不动；
# 某台臂装反或换过电机，就在该机的 devices/<host>.sh 里覆盖 B601_JOINT_DIRECTIONS）
FOLLOWER_JOINT_DIRECTIONS_DEFAULT='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'
FOLLOWER_JOINT_DIRECTIONS="${B601_JOINT_DIRECTIONS:-$FOLLOWER_JOINT_DIRECTIONS_DEFAULT}"

# 主臂 = StarArm102（CH340 无序列号）
# 主臂串口：由设备档案解析（kfzbox=udev 别名优先，z790=固定 ttyUSB0）
LEADER_PORT="${B601_LEADER_PORT:-/dev/ttyUSB0}"

# ==================== 相机 / 可视化 ====================
# 遥操作本身不依赖相机；开相机是让你边遥操边看视角（默认开）。
#   看画面（默认）：bash self_scripts/b601_single/01_teleop.sh
#   不看画面：      B601_DISPLAY_DATA=0 bash self_scripts/b601_single/01_teleop.sh
# 相机角色/序列号来自设备档案：kfzbox 配 hand+front+top 三路（front 需直连主机 USB 口，别挂 hub），z790 有 hand+front+top。
DISPLAY_DATA="${B601_DISPLAY_DATA:-1}"
DISPLAY_MODE="${B601_DISPLAY_MODE:-rerun}"      # rerun | foxglove
CAMERAS=""
if [ "$DISPLAY_DATA" = "1" ]; then
  _cam_warn="$(mktemp)"
  CAMERAS="$(b601_cameras_json 2>"$_cam_warn")" || CAMERAS=""
  if [ -s "$_cam_warn" ]; then cat "$_cam_warn" >&2; fi
  rm -f "$_cam_warn"
  if [ -z "$CAMERAS" ]; then
    echo "⚠️  没有可用相机 —— 遥操作照跑，只是看不到画面（B601_DISPLAY_DATA=0 可显式关掉这条）"
    DISPLAY_DATA=0
  else
    echo "📷 相机（$DISPLAY_MODE 显示）："
    printf '%s\n' "$CAMERAS" | sed '/^[{}$]/d; s/^/   /'
    # ssh 进来跑时补上本地图形会话（rerun viewer 要开窗）
    if [ -z "${DISPLAY:-}" ] && [ -e /tmp/.X11-unix/X0 ]; then export DISPLAY=:0; fi
    if [ -z "${XAUTHORITY:-}" ] && [ -f "$HOME/.Xauthority" ]; then export XAUTHORITY="$HOME/.Xauthority"; fi
    export RERUN_FLUSH_NUM_BYTES="${RERUN_FLUSH_NUM_BYTES:-10000000}"
    export LEROBOT_RERUN_MEMORY_LIMIT="${LEROBOT_RERUN_MEMORY_LIMIT:-30%}"
  fi
fi

echo "=========================================="
echo "单臂遥操作测试：B601-RS + StarArm102"
echo "=========================================="
echo ""

# ==================== 环境检查 ====================
if ! command -v lerobot-teleoperate >/dev/null 2>&1; then
  echo "未找到 lerobot-teleoperate，尝试激活 lerobot conda 环境..."
  for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      source "$base/etc/profile.d/conda.sh"
      conda activate lerobot
      break
    fi
  done
fi
if ! command -v lerobot-teleoperate >/dev/null 2>&1; then
  echo "❌ 错误：lerobot-teleoperate 不可用"
  echo "   请手动执行: conda activate lerobot"
  exit 1
fi
echo "✅ lerobot-teleoperate 可用"

# ==================== 硬件检查 ====================
# B601 CAN：启动前一律复位接口并清空残留帧
#   上次异常退出/急停断电时，PCAN 驱动队列和电机侧可能残留半截指令帧，
#   电机被这些帧喂住后不再响应新的控制帧，表现为 connect() 时报
#   "control ack timeout: comm_type=4"（disable_all/enable_all 超时）。
#   这里 down→重设 bitrate→up 让控制器彻底复位，再把 socket 接收队列读空，
#   等价于"断电重新上电"的清场动作。
can_drain() {
  local port="$1"
  command -v candump >/dev/null 2>&1 || return 0
  timeout 0.3 candump "$port" >/dev/null 2>&1 || true
}

if ! ip link show "$FOLLOWER_PORT" >/dev/null 2>&1; then
  echo "❌ 错误：$FOLLOWER_PORT 不存在，请检查 PCAN-USB 连接"
  exit 1
fi

echo "♻️  复位 $FOLLOWER_PORT（${CAN_BITRATE}bps）并清空残留 CAN 帧..."
sudo ip link set "$FOLLOWER_PORT" down 2>/dev/null || true
sudo ip link set "$FOLLOWER_PORT" type can bitrate "$CAN_BITRATE" || {
  echo "❌ 错误：配置 $FOLLOWER_PORT 失败（是否被其它进程占用？）"
  exit 1
}
sudo ip link set "$FOLLOWER_PORT" up
sleep 0.5   # 给电机侧看门狗留时间丢掉上一条残留指令
can_drain "$FOLLOWER_PORT"

# 启动前确认接口真的 UP，否则直接退出（别等到 lerobot 里才报 ack timeout）
if ! ip link show "$FOLLOWER_PORT" 2>/dev/null | grep -q "state UP"; then
  echo "❌ 错误：$FOLLOWER_PORT 未 UP —— 检查 PCAN-USB 是否插好、peak_usb 驱动是否加载"
  ip -details link show "$FOLLOWER_PORT" 2>/dev/null | sed -n '1,6p'
  exit 1
fi
if ! ip -details link show "$FOLLOWER_PORT" 2>/dev/null | grep -q "bitrate $CAN_BITRATE"; then
  echo "⚠️  $FOLLOWER_PORT 的 bitrate 不是 $CAN_BITRATE，请确认总线速率"
fi
echo "✅ B601 从臂 $FOLLOWER_PORT 已就绪（${CAN_BITRATE}bps，残留帧已清空）"

# 主臂串口
if [ ! -e "$LEADER_PORT" ]; then
  echo "❌ 错误：主臂串口不存在 $LEADER_PORT"
  echo "   请检查 StarArm102 USB 连接，以及是否被 brltty 抢占:"
  echo "   sudo apt remove -y brltty"
  exit 1
fi
[ -r "$LEADER_PORT" ] || sudo chmod 666 "$LEADER_PORT"
echo "✅ 主臂 $LEADER_PORT 已连接"

# ==================== 校准文件检查 ====================
echo ""
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
  echo "❌ 校准文件不完整。请先完成校准："
  echo "   从臂（B601-RS）：  lerobot-calibrate --robot.type=seeed_b601_rs_follower --robot.port=$FOLLOWER_PORT --robot.can_adapter=$FOLLOWER_ADAPTER --robot.id=$FOLLOWER_ID"
  echo "   主臂（StarArm102）：lerobot-calibrate --teleop.type=rebot_102_leader --teleop.port=$LEADER_PORT --teleop.id=$LEADER_ID"
  echo "   或参考 self_scripts/b601_common/01_leader_calibration.sh、02_follower_calibration.sh"
  exit 1
fi

# ==================== 安全确认 ====================
echo ""
echo "⚠️  安全确认（重要）："
echo "  - 从臂连接后会【使能变硬】"
echo "  - 主臂 -> 从臂直接跟随（已去除速度限幅；仅保留 B601 关节硬限位 + 温度保护）"
echo "  - ⚠️ 主臂与从臂姿态不一致时，第一帧/大幅动作从臂会快速移动，务必慢速操作"
echo "  - 退出（Ctrl+C）时从臂会回零位并失能"
echo "  - 紧急停止：切断 B601 从臂 48V 电源"
echo "  - 【关键】启动前把【主臂和从臂】都摆到零位（默认坐姿、夹爪闭合）"
echo "    否则第一帧会把从臂拉向主臂当前姿态（虽受限但仍会动）"
if [ "$DISPLAY_DATA" = "1" ]; then
  echo "  - 启动后会自动弹出 $DISPLAY_MODE 窗口显示相机画面（hand/front/top），关窗不影响遥操作"
fi
echo ""
read -p "确认主臂/从臂已在零位、周边安全，按 ENTER 开始，Ctrl+C 取消..." dummy

echo ""
echo "🚀 开始遥操作（单臂 B601-RS）..."
echo ""

# ==================== 日志 ====================
LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/teleop_single_b601_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂、lerobot-teleoperate 被
# SIGPIPE(141) 杀死而跳过断开清理（否则电机保持使能、从臂不释放）。
trap '' PIPE

# ==================== 启动遥操作 ====================
TELEOP_ARGS=(
  --robot.type=seeed_b601_rs_follower
  --robot.id="$FOLLOWER_ID"
  --robot.port="$FOLLOWER_PORT"
  --robot.can_adapter="$FOLLOWER_ADAPTER"
  --robot.gravity_compensation="$GRAVITY_COMPENSATION"
  --robot.joint_directions="$FOLLOWER_JOINT_DIRECTIONS"
  --teleop.type=rebot_102_leader
  --teleop.id="$LEADER_ID"
  --teleop.port="$LEADER_PORT"
)
if [ -n "$CAMERAS" ]; then TELEOP_ARGS+=(--robot.cameras="$CAMERAS"); fi
if [ "$DISPLAY_DATA" = "1" ]; then
  TELEOP_ARGS+=(--display_data=true --display_mode="$DISPLAY_MODE")
  echo "可视化：$DISPLAY_MODE（DISPLAY=${DISPLAY:-未设置}）"
fi

lerobot-teleoperate "${TELEOP_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "✅ 遥操作结束（从臂已回零位并失能）"
echo "日志：$LOG_FILE"
echo "=========================================="
