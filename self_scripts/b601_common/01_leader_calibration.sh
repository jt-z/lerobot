#!/bin/bash
# B601 主臂（StarArm102 / reBot Arm 102）关节校准脚本
# 功能：将主臂摆到零位后，逐个舵机记录零位（unlock + set_origin_point）
# 依赖：lerobot conda 环境（脚本会自动尝试激活）
# 用法：bash self_scripts/b601_common/01_leader_calibration.sh
#
# 校准前注意：
#   - 主臂 USB 串口会自动探测（优先 by-id 稳定路径，可用 LEADER_PORT=/dev/ttyUSB0 覆盖）
#   - 若端口被 brltty 抢占，先执行: sudo apt remove -y brltty
#   - 把主臂摆到零位（默认坐姿、夹爪闭合），再按回车

set -e
set -o pipefail

# 主臂串口自动探测（与 02_record.sh 口径一致）：
#   ttyUSB*/ttyACM* 编号随插拔顺序/驱动变化（本机实测同一适配器会在 ttyUSB0 与 ttyACM0
#   之间变名），机内还有别的 USB 串口（WCH 1a86:55d4），只按名字挑会挑错，故用
#   04_leader_port.py 对候选口真正 ping 主臂舵机（FashionStar id 0~6），有应答的才算主臂。
LEADER_PORT="${LEADER_PORT:-}"
LEADER_PORT_PROBE="$(dirname "$(readlink -f "$0")")/04_leader_port.py"
LEADER_ID="leader"

echo "=========================================="
echo "B601 主臂关节校准（rebot_102_leader）"
echo "=========================================="
echo ""
# ==================== 端口探测 ====================
if [ -z "$LEADER_PORT" ]; then
  if [ -f "$LEADER_PORT_PROBE" ]; then
    echo "🔍 探测主臂串口（候选口逐个 ping 舵机 id 0~6）..."
    LEADER_PORT="$(python "$LEADER_PORT_PROBE" || true)"
  else
    echo "⚠️  未找到探测脚本 $LEADER_PORT_PROBE，退回按设备名挑"
    for cand in /dev/serial/by-id/usb-1a86_USB_Serial*-if00-port0 /dev/ttyUSB* /dev/ttyACM*; do
      [ -e "$cand" ] && LEADER_PORT="$cand" && break
    done
  fi
fi
if [ -z "$LEADER_PORT" ] || [ ! -e "$LEADER_PORT" ]; then
  echo "❌ 错误：未找到主臂串口（现有候选口都没有 StarArm102 舵机应答）"
  echo "   1) 主臂 USB 是否插好、舵机是否上电（12V）"
  echo "   2) 现有串口：$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | tr '\n' ' ')"
  echo "   3) 逐个口看探测结果：python $LEADER_PORT_PROBE --list"
  echo "   4) 手动指定：LEADER_PORT=/dev/ttyUSB0 bash $0"
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
