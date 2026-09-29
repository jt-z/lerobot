#!/bin/bash
# B601 从臂（B601-RS，RobStride 电机）零位校准脚本
# 功能：失能电机 -> 手动把从臂摆到零位 -> 记录零点（不产生任何运动）
# 说明：B601-RS 是 RobStride 电机，必须用 Seeed 官方 seeed_b601_rs_follower
#       （lerobot 自带 rebot_b601_follower 只支持 Damiao 电机，即 B601-DM，用在本机报 ensure_mode 失败）
# 依赖：lerobot conda 环境（含 lerobot-robot-seeed-b601 包，脚本会自动尝试激活）
# 用法：bash 02_follower_calibration.sh
#
# ⚠️ 安全注意（从臂较大较重）：
#   - 校准过程不会驱动电机，电机保持失能状态，可手动推动
#   - 连接瞬间会有一次"使能->立即失能"切换（电机短暂变硬随即放松），属正常
#   - 请全程与从臂保持安全距离，手动摆位时注意支撑
#   - 把从臂摆到零位（默认坐姿），夹爪闭合，再按回车

set -e
set -o pipefail

ROBOT_PORT="can0"        # SocketCAN 接口（PCAN-USB）
ROBOT_ADAPTER="socketcan"
ROBOT_ID="follower"

echo "=========================================="
echo "B601 从臂零位校准（seeed_b601_rs_follower）"
echo "=========================================="
echo ""

# ==================== 环境检查 ====================
if ! command -v lerobot-calibrate >/dev/null 2>&1; then
  echo "未找到 lerobot-calibrate，尝试激活 lerobot conda 环境..."
  for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      source "$base/etc/profile.d/conda.sh"
      conda activate lerobot
      break
    fi
  done
fi

if ! command -v lerobot-calibrate >/dev/null 2>&1; then
  echo "❌ 错误：lerobot-calibrate 不可用"
  echo "   请手动执行: conda activate lerobot"
  exit 1
fi
echo "✅ lerobot-calibrate 可用"

# ==================== CAN 接口检查 ====================
if ! ip link show $ROBOT_PORT >/dev/null 2>&1; then
  echo "❌ 错误：$ROBOT_PORT 不存在，请检查 PCAN-USB 连接"
  exit 1
fi
if ! ip link show $ROBOT_PORT 2>/dev/null | grep -q "state UP"; then
  echo "⚠️  $ROBOT_PORT 未 UP，正在配置（1Mbps 经典 CAN）..."
  sudo ip link set $ROBOT_PORT down 2>/dev/null || true
  sudo ip link set $ROBOT_PORT type can bitrate 1000000
  sudo ip link set $ROBOT_PORT up
fi
echo "✅ $ROBOT_PORT 已就绪"

# ==================== 开始校准 ====================
echo ""
echo "⚠️  安全提示：从臂较大，请与它保持安全距离，勿站在运动范围内"
echo ""
echo "校准流程（全程不会驱动电机）："
echo "  1. 电机将被失能（可手动推动）"
echo "  2. 请把从臂摆到【零位】= 默认坐姿，夹爪闭合"
echo "  3. 按回车记录零点"
echo ""
read -p "准备好后按 ENTER 开始校准，按 Ctrl+C 取消..." dummy

echo ""
echo "🚀 开始从臂零位校准..."
echo ""

lerobot-calibrate \
  --robot.type=seeed_b601_rs_follower \
  --robot.port=$ROBOT_PORT \
  --robot.can_adapter=$ROBOT_ADAPTER \
  --robot.id=$ROBOT_ID \
  --robot.gravity_compensation=true

echo ""
echo "=========================================="
echo "✅ 从臂校准完成！"
echo "校准文件 id: $ROBOT_ID"
echo "=========================================="
