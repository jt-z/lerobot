#!/bin/bash
# B601 上机前体检（kfzbox 版）：环境 / CAN / 电机 / 主臂串口 / 相机 / 标定文件
# 用法：bash self_scripts/b601_common/07_preflight.sh
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/_env_kfzbox.sh"
HERE="$(dirname "$(readlink -f "$0")")"

ok(){   printf "  \033[32m✅\033[0m %s\n" "$1"; }
bad(){  printf "  \033[31m❌\033[0m %s\n" "$1"; }
warn(){ printf "  \033[33m⚠️\033[0m %s\n" "$1"; }

echo "=========================================="
echo " B601 上机前体检  $(date '+%F %T')  $(hostname)"
echo "=========================================="

echo
echo "1) 环境"
PY="$B601_ENV/bin/python"
"$PY" - <<'PY'
import sys
import torch, lerobot
print(f"  python {sys.version.split()[0]} | torch {torch.__version__} | lerobot {lerobot.__version__}")
print(f"  import -> {lerobot.__file__}")
print(f"  cuda={torch.cuda.is_available()} gpu0={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
PY

echo
echo "2) CAN 接口"
if ip link show can0 >/dev/null 2>&1; then
  info=$(ip -details link show can0 2>/dev/null)
  if echo "$info" | grep -q "state UP"; then ok "can0 UP"
  else
    warn "can0 未 UP，配置 1Mbps..."
    sudo ip link set can0 down 2>/dev/null
    sudo ip link set can0 type can bitrate 1000000
    sudo ip link set can0 up
    ip -details link show can0 | grep -q "state UP" && ok "can0 已配好 (1Mbps)" || bad "can0 配置失败"
  fi
  ip -details link show can0 2>/dev/null | grep -q "bitrate 1000000" && ok "bitrate=1Mbps" || warn "bitrate 不是 1000000"
else
  bad "没有 can0 —— PCAN-USB 没插 / peak_usb 没加载 (lsusb | grep -i peak)"
fi

echo
echo "3) 从臂电机 (RobStride id 1..7)"
if ip link show can0 2>/dev/null | grep -q "state UP"; then
  found=$(timeout 40 "$B601_ENV/bin/motorbridge-cli" scan --vendor robstride --channel can0 --start-id 1 --end-id 7 2>&1 | grep -c "device_id=")
  echo "  扫到 ${found} / 7 个电机"
  if [ "${found:-0}" -eq 7 ]; then
    ok "7 个电机全在线（成品套件，不要重写 ID）"
  else
    rx1=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null || echo 0)
    sleep 3
    rx2=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null || echo 0)
    if [ "$((rx2 - rx1))" -eq 0 ]; then
      bad "总线上 0 报文 —— 从臂大概率【没上电】(48V 开关)，不是代码问题"
    else
      bad "总线有报文但扫描无应答 —— 查 CAN 线 A/B、120R 拨码、电机 ID"
    fi
  fi
else
  bad "can0 未 UP，跳过电机扫描"
fi

echo
echo "4) 主臂串口 (StarArm102)"
mapfile -t ttys < <(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null)
if [ -e /dev/rebot_leader ]; then
  ok "udev 别名 /dev/rebot_leader -> $(readlink -f /dev/rebot_leader)"
elif [ "${#ttys[@]}" -gt 0 ]; then
  warn "有串口但没有 /dev/rebot_leader（udev 规则 /etc/udev/rules.d/99-rebot-leader.rules 在不在？）"
else
  bad "没有任何 USB 串口 —— 主臂 USB 线没插"
fi
if [ "${#ttys[@]}" -gt 0 ]; then
  PORT=$("$PY" "$HERE/04_leader_port.py" 2>/dev/null | tail -1 || true)
  if [ -n "$PORT" ]; then ok "舵机应答正常，主臂串口: $PORT"
  else warn "有串口但 7 个舵机都没应答（主臂 12V 没上电？）"; fi
fi

echo
echo "5) 相机"
mapfile -t cams < <(ls /dev/v4l/by-id/*video-index0 2>/dev/null)
if [ "${#cams[@]}" -eq 0 ]; then bad "没有相机 (/dev/v4l/by-id 为空)"
else
  echo "  发现 ${#cams[@]} 路:"; printf '    %s\n' "${cams[@]}"
  [ "${#cams[@]}" -eq 3 ] && ok "3 路齐（核对 02_record.sh 里 CAMERAS 的序列号）" || warn "不是 3 路"
  echo "  判 hand/front/top：python $HERE/08_cam_probe.py"
fi

echo
echo "6) 标定文件"
for f in "$HOME/.cache/huggingface/lerobot/calibration/robots/seeed_b601_rs_follower/follower.json" \
         "$HOME/.cache/huggingface/lerobot/calibration/teleoperators/rebot_102_leader/leader.json"; do
  if [ -f "$f" ]; then ok "$(basename "$(dirname "$f")")/$(basename "$f")   $(date -r "$f" '+%F %H:%M')"
  else bad "缺 $f"; fi
done
_lf="$HOME/.cache/huggingface/lerobot/calibration/teleoperators/rebot_102_leader/leader.json"
if [ -f "$_lf" ] && [ -n "$(find "$_lf" -mtime +1 2>/dev/null)" ]; then
  warn "主臂 leader.json 超过 1 天没更新（$(date -r "$_lf" '+%F')）—— 若是新主臂没重校过，先跑 01_leader_calibration.sh"
fi

echo
echo "=========================================="
echo " 顺序：01_teleop.sh（试动）→ 02_record.sh（采集）"
echo "=========================================="
