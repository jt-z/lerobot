#!/bin/bash
# B601 主臂（StarArm102 / reBot Arm 102）关节校准脚本
# 功能：将主臂摆到零位后，逐个舵机记录零位（unlock + set_origin_point）
# 依赖：lerobot conda 环境（脚本会自动尝试激活）
# 用法：bash self_scripts/b601_common/01_leader_calibration.sh
#
# 校准前注意：
#   - 主臂 USB 串口需存在（默认 /dev/ttyUSB0，CH340 7523 无序列号）
#   - 若端口被 brltty 抢占，先执行: sudo apt remove -y brltty
#   - 把主臂摆到零位（默认坐姿、夹爪闭合），再按回车

set -e
set -o pipefail

LEADER_PORT="/dev/ttyUSB0"
LEADER_ID="leader"

echo "=========================================="
echo "B601 主臂关节校准（rebot_102_leader）"
echo "=========================================="
echo ""
# ==================== 端口检查 ====================
if [ ! -e "$LEADER_PORT" ]; then
  echo "❌ 错误：主臂串口不存在 $LEADER_PORT"
  echo "   请检查主臂 USB 连接，以及是否被 brltty 抢占:"
  echo "   sudo apt remove -y brltty"
  exit 1
fi
echo "✅ $LEADER_PORT 已连接"

# ttyUSB0 属主为 root:dialout，无权限则尝试 chmod
if [ ! -r "$LEADER_PORT" ]; then
  echo "⚠️  无读取权限，尝试授权..."
  sudo chmod 666 "$LEADER_PORT"
fi

# ==================== 开始校准 ====================
echo ""
echo "请将主臂摆到零位：默认坐姿（各关节自然下垂）、夹爪闭合"
echo "（参考 Seeed wiki 中 reBot Arm 102 的零位图）"
echo ""
read -p "按 ENTER 开始校准，按 Ctrl+C 取消..." dummy

echo ""
echo "🚀 开始主臂校准（校准过程中会逐个舵机设置零位）..."
echo ""

lerobot-calibrate \
  --teleop.type=rebot_102_leader \
  --teleop.port=$LEADER_PORT \
  --teleop.id=$LEADER_ID

echo ""
echo "=========================================="
echo "✅ 主臂校准完成！"
echo "校准文件 id: $LEADER_ID"
echo "=========================================="
