#!/bin/bash
# 单臂遥操作测试：B601-RS 从臂（RobStride，SocketCAN）+ StarArm102 主臂
# 参照 self_scripts/b601_so101_bimanual/01_teleop_test.sh（原为 B601+SO-101 异构双臂版，本脚本为 B601 单臂版）
# 使用单臂 lerobot 类型：
#   robot  = seeed_b601_rs_follower（B601-RS，can0，RobStride 电机）
#   teleop = rebot_102_leader（StarArm102 /dev/ttyUSB0，FashionStar UART 舵机）
#
# 依赖：lerobot conda 环境（含 lerobot-robot-seeed-b601 包，脚本会自动尝试激活）
# 用法：bash self_scripts/b601_single/01_teleop.sh
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

# ==================== 硬件配置 ====================
FOLLOWER_ID="follower"          # 对应校准文件 seeed_b601_rs_follower/follower.json
LEADER_ID="leader"              # 对应校准文件 rebot_102_leader/leader.json

# 从臂 = B601-RS（SocketCAN）
FOLLOWER_PORT="can0"
FOLLOWER_ADAPTER="socketcan"
# 重力补偿（对 6 个 RS 关节做前馈），开启后手感更轻，建议保持 true
GRAVITY_COMPENSATION=true
# 从臂关节方向修正（2026-09-08 沿用 self_scripts/b601_so101_bimanual/01_teleop_test.sh、02_record.sh 中 B601 的实测值，必须保留，
# 否则 shoulder_lift/gripper 会被限位裁到 0 不动；wrist_roll 保持 -1.0 手感一致）
FOLLOWER_JOINT_DIRECTIONS='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'

# 主臂 = StarArm102（CH340 无序列号，只能固定 /dev/ttyUSB0）
# 备选 by-id：/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
LEADER_PORT="/dev/ttyUSB0"

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
# B601 CAN
if ! ip link show $FOLLOWER_PORT >/dev/null 2>&1; then
  echo "❌ 错误：$FOLLOWER_PORT 不存在，请检查 PCAN-USB 连接"
  exit 1
fi
if ! ip link show $FOLLOWER_PORT 2>/dev/null | grep -q "state UP"; then
  echo "⚠️  $FOLLOWER_PORT 未 UP，正在配置（1Mbps 经典 CAN）..."
  sudo ip link set $FOLLOWER_PORT down 2>/dev/null || true
  sudo ip link set $FOLLOWER_PORT type can bitrate 1000000
  sudo ip link set $FOLLOWER_PORT up
fi
echo "✅ B601 从臂 $FOLLOWER_PORT 已就绪"

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
echo ""
read -p "确认主臂/从臂已在零位、周边安全，按 ENTER 开始，Ctrl+C 取消..." dummy

echo ""
echo "🚀 开始遥操作（单臂 B601-RS）..."
echo ""

# ==================== 日志 ====================
LOG_DIR="$HOME/LX/pai0/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/teleop_single_b601_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂、lerobot-teleoperate 被
# SIGPIPE(141) 杀死而跳过断开清理（否则电机保持使能、从臂不释放）。
trap '' PIPE

# ==================== 启动遥操作 ====================
lerobot-teleoperate \
  --robot.type=seeed_b601_rs_follower \
  --robot.id=$FOLLOWER_ID \
  --robot.port=$FOLLOWER_PORT \
  --robot.can_adapter=$FOLLOWER_ADAPTER \
  --robot.gravity_compensation=$GRAVITY_COMPENSATION \
  --robot.joint_directions="$FOLLOWER_JOINT_DIRECTIONS" \
  --teleop.type=rebot_102_leader \
  --teleop.id=$LEADER_ID \
  --teleop.port=$LEADER_PORT 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "✅ 遥操作结束（从臂已回零位并失能）"
echo "日志：$LOG_FILE"
echo "=========================================="
