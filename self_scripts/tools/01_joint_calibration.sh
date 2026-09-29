#!/bin/bash
# B601 主臂（StarArm102 / reBot Arm 102）关节校准脚本
# 功能：将主臂摆到零位后，逐个舵机记录零位（unlock + set_origin_point）
# 依赖：lerobot conda 环境（脚本会自动尝试激活）
# 用法：bash 01_joint_calibration.sh
#
# 校准前注意：
#   - 主臂 USB 串口会自动探测（by-id 稳定路径优先，可用 LEADER_PORT=/dev/ttyUSB0 覆盖）；
#   - 若端口被 brltty 抢占，先执行: sudo apt remove -y brltty
#   - 把主臂摆到零位（默认坐姿、夹爪闭合），再按回车

set -e
set -o pipefail

# 主臂串口：优先 by-id 稳定路径（ttyUSB*/ttyACM* 编号随插拔顺序/驱动变化，
# 机内还有别的 USB 串口 WCH 1a86:55d4，只按名字挑会挑错）。
# 手动覆盖：LEADER_PORT=/dev/ttyUSB0 bash tool/01_joint_calibration.sh
LEADER_PORT="${LEADER_PORT:-}"
if [ -z "$LEADER_PORT" ]; then
  for cand in /dev/serial/by-id/usb-1a86_USB_Serial*-if00-port0 /dev/ttyUSB* /dev/ttyACM*; do
    [ -e "$cand" ] && LEADER_PORT="$cand" && break
  done
fi
LEADER_ID="leader"

echo "=========================================="
echo "B601 主臂关节校准（rebot_102_leader）"
echo "=========================================="
echo ""
# ==================== 端口检查 ====================
if [ -z "$LEADER_PORT" ] || [ ! -e "$LEADER_PORT" ]; then
  echo "❌ 错误：未找到主臂串口"
  echo "   1) 主臂 USB 是否插好、舵机是否上电（12V）"
  echo "   2) 现有串口：$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | tr '\n' ' ')"
  echo "   3) 手动指定：LEADER_PORT=/dev/ttyUSB0 bash $0"
  exit 1
fi
echo "✅ 主臂串口：$LEADER_PORT"

# 串口属主为 root:dialout，无权限则尝试 chmod
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
