#!/bin/bash
# B601 设备参数层：通用默认值（两台机器共用）
# 由 b601_common/_env.sh 自动 source，不要直接执行。
# 机器专属差异（环境/数据根/主臂串口/相机序列号）写 devices/<host>.sh，不要改这里。
#
# 覆盖方式：任何 B601_* 变量都能在命令行前用环境变量临时覆盖，例如
#   B601_DEVICE=z790 bash self_scripts/b601_single/02_record.sh
#   B601_CAM_TOP=2609007 bash self_scripts/b601_single/02_record.sh

# ---- 从臂 B601-RS（RobStride，SocketCAN）----
export B601_FOLLOWER_PORT="${B601_FOLLOWER_PORT:-can0}"
export B601_FOLLOWER_ADAPTER="${B601_FOLLOWER_ADAPTER:-socketcan}"
export B601_CAN_BITRATE="${B601_CAN_BITRATE:-1000000}"

# ---- 相机通用规格（两台一致，差异只在序列号/角色）----
export B601_CAM_WIDTH="${B601_CAM_WIDTH:-640}"
export B601_CAM_HEIGHT="${B601_CAM_HEIGHT:-480}"
export B601_CAM_FPS="${B601_CAM_FPS:-30}"
export B601_CAM_FOURCC="${B601_CAM_FOURCC:-MJPG}"

# ---- 主臂候选串口（设备档案没写死 B601_LEADER_PORT 时按此顺序取第一个存在的）----
export B601_LEADER_CANDIDATES_DEFAULT="${B601_LEADER_CANDIDATES_DEFAULT:-/dev/rebot_leader /dev/serial/by-id/usb-1a86_USB_Serial*-if00-port0 /dev/ttyUSB* /dev/ttyACM*}"

# ---- 角色 → 相机 by-id 序列号（机器档案里填；留空 = 自动兜底）----
: "${B601_CAM_HAND:=}"
: "${B601_CAM_FRONT:=}"
: "${B601_CAM_TOP:=}"

# 序列号 → by-id 完整路径（找不到返回非零）
b601_cam_path() {
  [ -z "${1:-}" ] && return 1
  local p
  p=$(ls /dev/v4l/by-id/*"$1"*video-index0 2>/dev/null | head -1)
  [ -n "$p" ] && printf '%s' "$p"
}

# 生成 lerobot 的 --robot.cameras JSON
# 三路序列号齐 → 按角色严格绑定；缺 → 按现有相机顺序兜底并告警（顺序未必对，先跑 08_cam_probe.py）
b601_cameras_json() {
  local hand front top
  hand=$(b601_cam_path "$B601_CAM_HAND") || true
  front=$(b601_cam_path "$B601_CAM_FRONT") || true
  top=$(b601_cam_path "$B601_CAM_TOP") || true

  if [ -z "$hand" ] || [ -z "$front" ] || [ -z "$top" ]; then
    local cams=() c
    while IFS= read -r c; do [ -n "$c" ] && cams+=("$c"); done < <(ls /dev/v4l/by-id/*video-index0 2>/dev/null | sort)
    if [ "${#cams[@]}" -eq 0 ]; then
      echo "❌ 没有找到任何 /dev/v4l/by-id/*video-index0 相机（设备档案: ${B601_DEVICE:-?}）" >&2
      return 1
    fi
    hand="${cams[0]}"
    front="${cams[1]:-${cams[0]}}"
    top="${cams[2]:-${cams[0]}}"
    {
      echo "⚠️  相机按现有 ${#cams[@]} 路兜底（hand/front/top 顺序未必对，先跑 08_cam_probe.py 核对画面）："
      printf '     %s\n' "${cams[@]}"
      [ "${#cams[@]}" -lt 3 ] && echo "⚠️  只有 ${#cams[@]} 路相机（应为 3 路），缺的角色会复用同一台相机"
      echo "   要固定角色，把序列号写进 self_scripts/b601_common/devices/${B601_DEVICE:-<host>}.sh 的 B601_CAM_HAND/FRONT/TOP"
    } >&2
  fi

  printf '{\n'
  printf '  hand: {type: opencv, index_or_path: %s, width: %s, height: %s, fps: %s, fourcc: %s},\n' \
    "$hand" "$B601_CAM_WIDTH" "$B601_CAM_HEIGHT" "$B601_CAM_FPS" "$B601_CAM_FOURCC"
  printf '  front: {type: opencv, index_or_path: %s, width: %s, height: %s, fps: %s, fourcc: %s},\n' \
    "$front" "$B601_CAM_WIDTH" "$B601_CAM_HEIGHT" "$B601_CAM_FPS" "$B601_CAM_FOURCC"
  printf '  top: {type: opencv, index_or_path: %s, width: %s, height: %s, fps: %s, fourcc: %s}\n' \
    "$top" "$B601_CAM_WIDTH" "$B601_CAM_HEIGHT" "$B601_CAM_FPS" "$B601_CAM_FOURCC"
  printf '}'
}
