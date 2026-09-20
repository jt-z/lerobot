#!/bin/bash
# 异构双臂遥操作测试：B601（左）+ SO-101 右臂
# 使用统一的 bi 类型（单进程，与数据采集同构）：
#   robot  = bi_b601_so101_follower（左 B601-RS can0 + 右 SO-101 串口）
#   teleop = bi_b601_so101_leader（左 StarArm102 /dev/ttyUSB0 + 右 SO-101 主臂）
#
# 依赖：lerobot conda 环境（含 lerobot-robot-seeed-b601 包）
# 用法：bash self_scripts/b601_so101_bimanual/01_teleop_test.sh
#
# ⚠️ 安全须知（双臂同时使能，务必先读）：
#   1. 连接后两条从臂电机都会【使能变硬】
#   2. 移动对应主臂 -> 对应从臂跟随；动作幅度受 max_relative_target 限制
#   3. 退出（Ctrl+C）时：B601 从臂【回零位坐姿】，SO-101 从臂按自身校准回零
#   4. 紧急停止 = 切断 48V 电源（B601 从臂）；SO-101 从臂也有独立电源
#   5. 启动前务必将【两条主臂和两条从臂】都摆到零位（默认坐姿、夹爪闭合）
#   6. 首次测试请小幅、慢速移动主臂，确认方向/比例正常再加大幅度

set -e
set -o pipefail

# ==================== 硬件配置 ====================
FOLLOWER_ID="jt_follower_arm"
LEADER_ID="jt_leader_arm"

# 左臂 = B601
B601_FOLLOWER_PORT="can0"
B601_LEADER_PORT="/dev/ttyUSB0"
# 左 B601 从臂方向修正（必须保留，否则 shoulder_lift/gripper 被限位裁到 0 不动）
B601_JOINT_DIRECTIONS='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'
# 速度限幅（max_relative_target）：已按需求移除（原为 B601 逐关节 3~10°/步、SO-101 10°/步）。
# ⚠️ 无限幅时：主臂当前姿态会【直接】作用于从臂（无每步减速）。
#    若主臂与从臂姿态不一致，启动首帧或大幅动作时从臂会【快速移动】到主臂姿态，注意安全！
#    仅保留 B601 关节硬限位（joint_limits：shoulder_lift [0,170]、elbow_flex [0,200]、
#    gripper [0,270] 等）+ 温度保护（115°告警/125°中断），见 config_seeed_b601_rs_follower.py
# B601_MAX_RELATIVE_TARGET='{shoulder_pan: 5.0, shoulder_lift: 3.0, elbow_flex: 3.0, wrist_flex: 5.0, wrist_yaw: 5.0, wrist_roll: 5.0, gripper: 10.0}'

# 右臂 = SO-101（by-id 稳定路径）
SO101_RIGHT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00"
SO101_RIGHT_LEADER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034865-if00"

echo "=========================================="
echo "异构双臂遥操作测试：B601（左）+ SO-101 右臂"
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
if ! ip link show $B601_FOLLOWER_PORT >/dev/null 2>&1; then
  echo "❌ 错误：$B601_FOLLOWER_PORT 不存在，请检查 PCAN-USB 连接"
  exit 1
fi
if ! ip link show $B601_FOLLOWER_PORT 2>/dev/null | grep -q "state UP"; then
  echo "⚠️  $B601_FOLLOWER_PORT 未 UP，正在配置（1Mbps 经典 CAN）..."
  sudo ip link set $B601_FOLLOWER_PORT down 2>/dev/null || true
  sudo ip link set $B601_FOLLOWER_PORT type can bitrate 1000000
  sudo ip link set $B601_FOLLOWER_PORT up
fi
echo "✅ B601 从臂 $B601_FOLLOWER_PORT 已就绪"

# 串口
for port in "$B601_LEADER_PORT" "$SO101_RIGHT_FOLLOWER_PORT" "$SO101_RIGHT_LEADER_PORT"; do
  if [ ! -e "$port" ]; then
    echo "❌ 错误：串口不存在 $port"
    echo "   B601 主臂请确认未被 brltty 抢占（sudo apt remove brltty）"
    exit 1
  fi
done
[ -r "$B601_LEADER_PORT" ] || sudo chmod 666 "$B601_LEADER_PORT"
echo "✅ 全部串口已连接"

# ==================== 校准文件检查 ====================
echo ""
CALIB_OK=true
for f in \
  "$HOME/.cache/huggingface/lerobot/calibration/robots/seeed_b601_rs_follower/${FOLLOWER_ID}_left.json" \
  "$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/${FOLLOWER_ID}_right.json" \
  "$HOME/.cache/huggingface/lerobot/calibration/teleoperators/rebot_102_leader/${LEADER_ID}_left.json" \
  "$HOME/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/${LEADER_ID}_right.json"; do
  if [ -f "$f" ]; then
    echo "  ✅ $f"
  else
    echo "  ❌ 缺失：$f"
    CALIB_OK=false
  fi
done
if [ "$CALIB_OK" = false ]; then
  echo ""
  echo "❌ 校准文件不完整，请先完成四臂校准（或确认校准文件命名）"
  exit 1
fi

# ==================== 安全确认 ====================
echo ""
echo "⚠️  安全确认（重要）："
echo "  - 两条从臂连接后会【同时使能变硬】"
echo "  - B601 主臂 -> B601 从臂；SO-101 右主臂 -> SO-101 右从臂"
echo "  - 动作幅度：已去除速度限幅（跟手直连）；仅保留 B601 关节硬限位 + 温度保护"
echo "  - ⚠️ 主臂与从臂姿态不一致时，首帧/大幅动作从臂会快速移动，务必慢速操作"
echo "  - 退出（Ctrl+C）时两条从臂都会回零位"
echo "  - 紧急停止：切断 B601 从臂 48V 电源 / SO-101 从臂电源"
echo "  - 【关键】启动前把【两条主臂和两条从臂】都摆到零位（默认坐姿、夹爪闭合）"
echo "    否则第一帧会把从臂拉向主臂当前姿态（虽受限但仍会动）"
echo ""
read -p "确认四臂已在零位、周边安全，按 ENTER 开始，Ctrl+C 取消..." dummy

echo ""
echo "🚀 开始遥操作（单进程异构双臂）..."
echo ""

# ==================== 日志 ====================
LOG_DIR="$HOME/LX/pai0/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/teleop_bimanual_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂、lerobot-teleoperate 被
# SIGPIPE(141) 杀死而跳过断开清理（否则电机保持使能、从臂不释放）。
trap '' PIPE

# ==================== 启动遥操作 ====================
lerobot-teleoperate \
  --robot.type=bi_b601_so101_follower \
  --robot.id=$FOLLOWER_ID \
  --robot.left_arm_config.port=$B601_FOLLOWER_PORT \
  --robot.left_arm_config.can_adapter=socketcan \
  --robot.left_arm_config.gravity_compensation=true \
  --robot.left_arm_config.joint_directions="$B601_JOINT_DIRECTIONS" \
  --robot.right_arm_config.port=$SO101_RIGHT_FOLLOWER_PORT \
  --teleop.type=bi_b601_so101_leader \
  --teleop.id=$LEADER_ID \
  --teleop.left_arm_config.port=$B601_LEADER_PORT \
  --teleop.right_arm_config.port=$SO101_RIGHT_LEADER_PORT 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "✅ 遥操作结束（从臂已回零位并失能）"
echo "日志：$LOG_FILE"
echo "=========================================="
