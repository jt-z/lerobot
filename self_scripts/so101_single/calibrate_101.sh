#!/bin/bash
# 单臂 SO-101 重新校准（从臂 so101_follower / 主臂 so101_leader）
# 参照 lerobot/src/lerobot/scripts/lerobot_calibrate.py（CLI: lerobot-calibrate）
#   该脚本仅做两件事：make_*_from_config() -> device.connect(calibrate=False) -> device.calibrate()
#   真正的校准交互在 lerobot/robots/so_follower/so_follower.py::calibrate()
#                      与 lerobot/teleoperators/so_leader/so_leader.py::calibrate()
#
# 校准文件位置（id 就是文件名，主从臂各一份）：
#   从臂 ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<FOLLOWER_ID>.json
#   主臂 ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/<LEADER_ID>.json
# 只要 id 不变，重校准后 teleop_101.sh / record_101.sh 会自动用上新文件，无需改脚本。
#
# 用法：
#   bash self_scripts/so101_single/calibrate_101.sh            # 从臂 + 主臂 依次重校准（默认）
#   bash self_scripts/so101_single/calibrate_101.sh follower   # 只校准从臂
#   bash self_scripts/so101_single/calibrate_101.sh leader     # 只校准主臂
#   FOLLOWER_PORT=/dev/ttyACM0 bash self_scripts/so101_single/calibrate_101.sh   # 临时覆盖串口
#
# 可覆盖的环境变量：FOLLOWER_PORT / LEADER_PORT / FOLLOWER_ID / LEADER_ID
#
# ⚠️ 校准流程中的交互（务必看清提示再按键）：
#   1) 若该 id 已有校准文件，lerobot 会先问：
#      "Press ENTER to use provided calibration file ... or type 'c' and press ENTER to run calibration"
#      —— 【重新校准】必须输入 c 再回车；直接回车 = 只是把旧文件写回舵机，不重新校准。
#   2) 提示 "Move ... to the middle of its range of motion and press ENTER"：
#      先把臂摆到【各关节行程中点】（大致= 默认坐姿/中立位），松手后回车。
#   3) 提示 "Move all joints except 'wrist_roll' sequentially through their entire ranges ..."：
#      逐个关节在整个行程内缓慢来回走一遍（夹爪要完全张开→完全闭合），全部走完再回车结束记录。
#      wrist_roll（腕部旋转）不参与扫行程，范围固定 0~4095，不用管它。
#
# ⚠️ 安全注意：
#   - 校准时舵机会被【失能】（变松），可以手动自由摆动，不会主动运动
#   - 手动摆位前先托住臂，SO-101 掉电后关节是无支撑的
#   - 扫行程时【慢速、别硬顶机械限位】，走到顶就回，避免憋住舵机
#   - 校准只是修改舵机内部零点/行程寄存器并写 json，不动数据，可放心重跑

set -e
set -o pipefail

# ==================== 参数解析 ====================
TARGET="${1:-both}"
case "$TARGET" in
  follower|leader|both) ;;
  -h|--help|help)
    echo "用法：bash $0 [follower|leader|both]   （默认 both，从臂 + 主臂依次校准）"
    exit 0
    ;;
  *)
    echo "❌ 错误：未知参数 '$TARGET'（可选：follower | leader | both）"
    exit 1
    ;;
esac

# ==================== 硬件配置 ====================
# 校准文件 id：与 ~/.cache/huggingface/lerobot/calibration/{robots/so_follower,teleoperators/so_leader}/<id>.json 对应
FOLLOWER_ID="${FOLLOWER_ID:-jt_follower_arm_right}"
LEADER_ID="${LEADER_ID:-jt_leader_arm_right}"

# by-id 稳定路径（ttyACM* 编号随插拔顺序变化，不写死）
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00}"
LEADER_PORT="${LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034865-if00}"

CALIB_DIR="$HOME/.cache/huggingface/lerobot/calibration"
FOLLOWER_CALIB="$CALIB_DIR/robots/so_follower/${FOLLOWER_ID}.json"
LEADER_CALIB="$CALIB_DIR/teleoperators/so_leader/${LEADER_ID}.json"

# 本次要处理的臂（--teleop 用主臂 / --robot 用从臂，两者命令形态不同，分开拼）
DO_FOLLOWER=false
DO_LEADER=false
if [ "$TARGET" = "follower" ] || [ "$TARGET" = "both" ]; then DO_FOLLOWER=true; fi
if [ "$TARGET" = "leader" ] || [ "$TARGET" = "both" ]; then DO_LEADER=true; fi

echo "=========================================="
echo "单臂 SO-101 重新校准（target=$TARGET）"
echo "=========================================="
if [ "$DO_FOLLOWER" = true ]; then echo "  从臂 $FOLLOWER_PORT  (id=$FOLLOWER_ID)"; fi
if [ "$DO_LEADER" = true ]; then echo "  主臂 $LEADER_PORT  (id=$LEADER_ID)"; fi
echo ""

# ==================== 环境检查 ====================
if ! command -v lerobot-calibrate >/dev/null 2>&1; then
  echo "未找到 lerobot-calibrate，尝试激活 lerobot conda 环境..."
  for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
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

# ==================== 串口检查 ====================
declare -a ARM_KEYS=() ARM_PORTS=() ARM_LABELS=()
if [ "$DO_FOLLOWER" = true ]; then
  ARM_KEYS+=("follower"); ARM_PORTS+=("$FOLLOWER_PORT"); ARM_LABELS+=("从臂 so101_follower")
fi
if [ "$DO_LEADER" = true ]; then
  ARM_KEYS+=("leader"); ARM_PORTS+=("$LEADER_PORT"); ARM_LABELS+=("主臂 so101_leader")
fi

for i in "${!ARM_PORTS[@]}"; do
  port="${ARM_PORTS[$i]}"
  if [ ! -e "$port" ]; then
    echo "❌ 错误：${ARM_LABELS[$i]} 串口不存在 $port"
    echo "   检查 USB 连接；若被 brltty 抢占: sudo apt remove -y brltty"
    echo "   当前串口设备：$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | tr '\n' ' ')"
    exit 1
  fi
  if [ ! -r "$port" ] || [ ! -w "$port" ]; then
    echo "⚠️  $port 无读写权限，尝试授权..."
    sudo chmod 666 "$port"
  fi
done
echo "✅ 串口均已连接"

# 舵机应答检查：ping 6 个 STS3215（id 1~6），确认臂已上电、且端口确实对着 SO-101
# 说明：ping 返回 (型号, 通信结果, 错误码)，只有「通信结果 == 0」才算应答（超时会返回型号 0）
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
  for i in "${!ARM_PORTS[@]}"; do
    port="${ARM_PORTS[$i]}"
    if hits="$(probe_port "$port")"; then
      echo "✅ ${ARM_LABELS[$i]} 舵机应答: $hits"
    else
      echo "❌ ${ARM_LABELS[$i]} 舵机应答不全（$port，应答: ${hits:-无}）"
      echo "   1) 确认该臂 12V 电源已开、USB 已插好"
      echo "   2) 确认这条串口对应的确实是 SO-101（不是别的设备）"
      echo "   3) 若从臂夹爪处于堵转过载保护态（ping 报错码），可先断电重上电再试"
      exit 1
    fi
  done
else
  echo "⚠️  跳过舵机应答检查（当前 python 缺 scservo_sdk；请确认已 conda activate lerobot）"
fi

# ==================== 备份旧校准文件 ====================
echo ""
echo "备份旧校准文件..."
BACKUP_DIR="$CALIB_DIR/_backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
for f in "$FOLLOWER_CALIB" "$LEADER_CALIB"; do
  if [ -f "$f" ]; then
    cp -p "$f" "$BACKUP_DIR/"
    echo "  ✅ 已备份 $(basename "$f") -> $BACKUP_DIR/"
  else
    echo "  ⚠️  无旧文件：$f（将新建）"
  fi
done

# ==================== 校准前说明 ====================
echo ""
echo "=========================================="
echo "校准操作说明（每条臂都会走一遍下面 3 步）"
echo "=========================================="
echo "  第 1 步：若弹出 'Press ENTER to use provided calibration file ... or type c ...'"
echo "          【重新校准必须输入字母 c 再回车】（直接回车 = 不校准，只把旧文件写回舵机）"
echo "  第 2 步：提示 'Move ... to the middle of its range of motion and press ENTER'"
echo "          把臂摆到【各关节行程中点】（大致= 默认坐姿/中立位），松手后回车"
echo "  第 3 步：提示 'Move all joints except wrist_roll sequentially through their entire ranges'"
echo "          逐个关节在【整个行程】内缓慢来回走一遍（夹爪完全张开→完全闭合），走完回车结束"
echo "          wrist_roll 不用扫，范围固定 0~4095"
echo ""
echo "⚠️  校准时舵机会失能（变松），可以手动自由摆动；请先托住臂再动手，别硬顶机械限位"
echo ""
read -p "准备就绪，按 ENTER 开始校准，Ctrl+C 取消..." dummy

# ==================== 日志 ====================
LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/calibrate_101_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "📝 日志文件：$LOG_FILE"
echo ""
# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂，lerobot-calibrate 被 SIGPIPE(141) 杀死
trap '' PIPE

# ==================== 逐臂校准 ====================
FAILED_ARMS=()
for i in "${!ARM_KEYS[@]}"; do
  key="${ARM_KEYS[$i]}"
  port="${ARM_PORTS[$i]}"
  label="${ARM_LABELS[$i]}"

  echo "=========================================="
  echo "🚀 开始校准：$label（$port）"
  echo "=========================================="
  if [ "$i" -gt 0 ]; then echo "（提醒：这一步是 ${label}，别对着另一条臂操作）"; fi
  echo ""

  if [ "$key" = "follower" ]; then
    CMD="lerobot-calibrate --robot.type=so101_follower --robot.port=\"$port\" --robot.id=\"$FOLLOWER_ID\""
    CALIB_FILE="$FOLLOWER_CALIB"
  else
    CMD="lerobot-calibrate --teleop.type=so101_leader --teleop.port=\"$port\" --teleop.id=\"$LEADER_ID\""
    CALIB_FILE="$LEADER_CALIB"
  fi

  set +e
  eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
  STATUS=${PIPESTATUS[0]}
  set -e

  echo ""
  if [ "$STATUS" -eq 0 ] && [ -f "$CALIB_FILE" ]; then
    echo "✅ $label 校准完成：$CALIB_FILE"
  else
    echo "⚠️  $label 校准未正常结束（退出码 $STATUS）"
    echo "   旧文件备份在：$BACKUP_DIR"
    echo "   恢复方法：cp -p \"$BACKUP_DIR/$(basename "$CALIB_FILE")\" \"$CALIB_FILE\""
    FAILED_ARMS+=("$label")
  fi
  echo ""
done

# ==================== 结果汇总 ====================
echo "=========================================="
echo "校准结果"
echo "=========================================="
for f in \
  "$CALIB_DIR/robots/so_follower/${FOLLOWER_ID}.json" \
  "$CALIB_DIR/teleoperators/so_leader/${LEADER_ID}.json"; do
  if [ -f "$f" ]; then
    echo "  ✅ $f  （修改时间：$(date -r "$f" '+%Y-%m-%d %H:%M:%S')）"
  else
    echo "  ❌ 缺失：$f"
  fi
done
echo ""
echo "备份目录：$BACKUP_DIR"
echo "日志：$LOG_FILE"

if [ "${#FAILED_ARMS[@]}" -gt 0 ]; then
  echo ""
  echo "⚠️  以下臂未正常完成：${FAILED_ARMS[*]}"
  echo "   可单独重跑：bash self_scripts/so101_single/calibrate_101.sh follower|leader"
  exit 1
fi
echo ""
echo "💡 下一步：先跑遥操作核对主从一致性：bash self_scripts/so101_single/teleop_101.sh"
echo "=========================================="
