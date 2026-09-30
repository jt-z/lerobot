#!/bin/bash
# 单臂 SO-101 遥操作：主臂（so101_leader）-> 从臂（so101_follower）
# 参照 lerobot/src/lerobot/scripts/lerobot_teleoperate.py 的单臂用法：
#   robot  = so101_follower（串口）
#   teleop = so101_leader（串口）
#
# 依赖：lerobot conda 环境（含 feetech-servo-sdk）
# 用法：
#   bash self_scripts/so101_single/teleop_101.sh
#   FOLLOWER_PORT=/dev/serial/by-id/... LEADER_PORT=/dev/serial/by-id/... bash self_scripts/so101_single/teleop_101.sh
#
# 端口说明（by-id 稳定路径，随插拔顺序变化的 ttyACM* 编号不写死）：
#   从臂 5B61034841（历史上是 SO-101 右从臂，校准文件 jt_follower_arm_right.json）
#   主臂 5B61034865（历史上是 SO-101 右主臂，校准文件 jt_leader_arm_right.json）
#
# ⚠️ 安全须知：
#   1. 连接后从臂电机【使能变硬】，主臂可自由拖动；验证方法 = 哪条臂变硬哪条就是从臂
#   2. 移动主臂 -> 从臂跟随；默认不做每步限幅（跟手直连），主从姿态不一致时首帧会快速移动
#      （需要限幅就设 MAX_REL_TARGET，例如 MAX_REL_TARGET=10 表示每步每关节最多 10°）
#   3. 退出（Ctrl+C）时从臂按自身校准回零并失能
#   4. 【关键】启动前把主臂和从臂都摆到零位（默认坐姿、夹爪闭合），否则第一帧会把从臂拉向主臂当前姿态
#   5. 若发现主从接反（被认作从臂的那条其实是主臂），立即 Ctrl+C，交换 FOLLOWER_PORT / LEADER_PORT 重来
#   6. 紧急停止：直接切断从臂电源

set -e
set -o pipefail

# ==================== 硬件配置 ====================
# 校准文件 id：与 ~/.cache/huggingface/lerobot/calibration/{robots/so_follower,teleoperators/so_leader}/<id>.json 对应
FOLLOWER_ID="jt_follower_arm_right"
LEADER_ID="jt_leader_arm_right"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00}"
LEADER_PORT="${LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034865-if00}"

FPS="${FPS:-60}"
# 每步每关节最大相对位移（度）。留空 = 不限幅（跟 lerobot 默认 + 前三脚本一致）
MAX_REL_TARGET="${MAX_REL_TARGET:-}"

echo "=========================================="
echo "单臂 SO-101 遥操作（主臂 -> 从臂）"
echo "=========================================="
echo "  从臂 $FOLLOWER_PORT  (id=$FOLLOWER_ID)"
echo "  主臂 $LEADER_PORT  (id=$LEADER_ID)"
echo ""

# ==================== 环境检查 ====================
if ! command -v lerobot-teleoperate >/dev/null 2>&1; then
  echo "未找到 lerobot-teleoperate，尝试激活 lerobot conda 环境..."
  for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
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

# ==================== 串口检查 ====================
for port in "$FOLLOWER_PORT" "$LEADER_PORT"; do
  if [ ! -e "$port" ]; then
    echo "❌ 错误：串口不存在 $port"
    echo "   检查 USB 连接；若被 brltty 抢占: sudo apt remove -y brltty"
    echo "   当前串口设备：$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | tr '\n' ' ')"
    exit 1
  fi
  if [ ! -r "$port" ] || [ ! -w "$port" ]; then
    echo "⚠️  $port 无读写权限，尝试授权..."
    sudo chmod 666 "$port"
  fi
done
echo "✅ 两个串口均已连接"

# ==================== 舵机应答检查 ====================
# ping 6 个 STS3215 舵机（id 1~6）：确认臂已上电、且端口确实对着 SO-101
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
  for port in "$FOLLOWER_PORT" "$LEADER_PORT"; do
    if hits="$(probe_port "$port")"; then
      echo "✅ $port 舵机应答: $hits"
    else
      echo "❌ $port 舵机应答不全（应答: ${hits:-无}）"
      echo "   1) 确认该臂 12V 电源已开、USB 已插好"
      echo "   2) 确认这条串口对应的确实是 SO-101（不是别的设备）"
      exit 1
    fi
  done
else
  echo "⚠️  跳过舵机应答检查（当前 python 缺 scservo_sdk；请确认已 conda activate lerobot）"
fi

# ==================== 校准文件检查 ====================
CALIB_DIR="$HOME/.cache/huggingface/lerobot/calibration"
CALIB_OK=true
for f in \
  "$CALIB_DIR/robots/so_follower/${FOLLOWER_ID}.json" \
  "$CALIB_DIR/teleoperators/so_leader/${LEADER_ID}.json"; do
  if [ -f "$f" ]; then
    echo "  ✅ $f"
  else
    echo "  ❌ 缺失：$f"
    CALIB_OK=false
  fi
done
if [ "$CALIB_OK" = false ]; then
  echo ""
  echo "❌ 校准文件不完整，请先用 lerobot-calibrate 校准这两条臂（或在脚本里改 *_ID）"
  exit 1
fi

# ==================== 安全确认 ====================
echo ""
echo "⚠️  安全确认："
echo "  - 按 ENTER 后从臂立即使能（变硬），随后跟随主臂动作"
echo "  - 不限速时主臂姿态会直接作用到从臂，务必【慢速、小幅】开始"
echo "  - 【关键】现在请确认主臂和从臂都在零位（默认坐姿、夹爪闭合）"
echo ""
read -p "确认已在零位、周边安全，按 ENTER 开始，Ctrl+C 取消..." dummy

echo ""
echo "🚀 开始遥操作（单臂 SO-101）..."
echo ""

# ==================== 日志 ====================
LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/teleop_101_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂、lerobot-teleoperate 被 SIGPIPE(141)
# 杀死而跳过断开清理（否则从臂保持使能、不释放）。
trap '' PIPE

# ==================== 启动遥操作 ====================
# 需要相机时追加（参考 04/05 脚本的 by-id 视频路径）：
#   --robot.cameras="{ hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0, width: 640, height: 480, fps: 30} }"
# 需要可视化时追加：--display_data=true
set +e  # 下面的失败要能捕获（Ctrl+C 退出码 130，异常退出非 0）
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --fps="$FPS" \
  --display_data=false \
  ${MAX_REL_TARGET:+--robot.max_relative_target="$MAX_REL_TARGET"} 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

echo ""
echo "=========================================="
if [ "$STATUS" -eq 0 ] || [ "$STATUS" -eq 130 ]; then
  echo "✅ 遥操作结束（从臂已回零位并失能）"
else
  echo "⚠️  遥操作异常退出（退出码 $STATUS），从臂可能来不及失能（电机仍变硬）"
  if grep -q "There is no status packet" "$LOG_FILE"; then
    echo ""
    echo "诊断：与舵机总线失联（USB 串口还在，但 6 个舵机同时不应答）—— 常见原因："
    echo "  1) 从臂舵机串链/12V 电源接头松动：断点之后的整条链全静默，重新插紧"
    echo "  2) 本机主从臂都挂在【两级串联的 USB Hub】下，且与相机共用（见 lsusb -t）"
    echo "     -> 把两条臂直接插到主机 USB 口，USB 线远离电机动力线"
    echo "  3) 重跑本脚本即可：启动前的「舵机应答检查」会先确认总线是否恢复"
    echo ""
    echo "确认总线是否恢复（应答 1,2,3,4,5,6 即正常）："
    echo "  python -c \"import scservo_sdk as s;p=s.PortHandler('$FOLLOWER_PORT');p.openPort();p.setBaudRate(1000000);k=s.PacketHandler(0);print([i for i in range(1,7) if k.ping(p,i)[1]==0])\""
  fi
fi
echo "日志：$LOG_FILE"
echo "=========================================="
