#!/bin/bash
# B601 设备参数层：通用默认值（两台机器共用）
# 由 b601_common/_env.sh 自动 source，不要直接执行。
# 机器专属差异（环境/数据根/主臂串口/相机角色）写 devices/<host>.sh，不要改这里。
#
# 覆盖方式：任何 B601_* 变量都能临时覆盖，例如
#   B601_DEVICE=z790 bash self_scripts/b601_common/07_preflight.sh

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

# ---- 角色 → 相机 by-id 序列号（机器档案里填；留空 = 自动处理）----
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
#   全配 → 严格按角色绑定（序列号不在线会明确告警，不会静默换成别的相机）
#   部分配（例如两路设备只配 hand/top）→ 配了的照绑，没配的角色复用已绑定的那路并告警
#   全没配 → 按现有相机顺序 hand/front/top 兜底（顺序未必对，先跑 08_cam_probe.py）
b601_cameras_json() {
  local hand front top c
  local cams=()
  while IFS= read -r c; do [ -n "$c" ] && cams+=("$c"); done \
    < <(ls /dev/v4l/by-id/*video-index0 2>/dev/null | sort)

  hand=$(b601_cam_path "${B601_CAM_HAND:-}") || true
  front=$(b601_cam_path "${B601_CAM_FRONT:-}") || true
  top=$(b601_cam_path "${B601_CAM_TOP:-}") || true

  if [ "${#cams[@]}" -eq 0 ]; then
    echo "❌ 没有找到任何 /dev/v4l/by-id/*video-index0 相机（设备档案: ${B601_DEVICE:-?}）" >&2
    return 1
  fi

  # 配了序列号但当前不在线 —— 必须说出来，不能悄悄换一台相机
  [ -n "${B601_CAM_HAND:-}" ] && [ -z "$hand" ] \
    && echo "⚠️  配置的 hand 相机 ${B601_CAM_HAND} 当前不在线" >&2
  [ -n "${B601_CAM_FRONT:-}" ] && [ -z "$front" ] \
    && echo "⚠️  配置的 front 相机 ${B601_CAM_FRONT} 当前不在线" >&2
  [ -n "${B601_CAM_TOP:-}" ] && [ -z "$top" ] \
    && echo "⚠️  配置的 top 相机 ${B601_CAM_TOP} 当前不在线" >&2

  if [ -z "$hand" ] && [ -z "$front" ] && [ -z "$top" ]; then
    # 一次都没配 → 老兜底
    hand="${cams[0]}"
    front="${cams[1]:-${cams[0]}}"
    top="${cams[2]:-${cams[0]}}"
    {
      echo "⚠️  相机全按现有 ${#cams[@]} 路兜底（hand/front/top 顺序未必对，先跑 08_cam_probe.py 核对画面）："
      printf '     %s\n' "${cams[@]}"
      [ "${#cams[@]}" -lt 3 ] && echo "⚠️  只有 ${#cams[@]} 路相机（通常应 3 路），缺的角色会复用同一台相机"
      echo "   要固定角色，把序列号写进 self_scripts/b601_common/devices/${B601_DEVICE:-<host>}.sh 的 B601_CAM_HAND/FRONT/TOP"
    } >&2
  else
    # 部分绑定：先拿一个已绑定的路径当填充值
    local fill="${hand:-${top:-${front:-${cams[0]}}}}"
    local dup=""
    if [ -z "$hand" ]; then
      hand="${cams[0]:-$fill}"; dup="$dup hand=${B601_CAM_HAND:-未配}"
    fi
    if [ -z "$top" ]; then
      top="${cams[1]:-${cams[0]:-$fill}}"; dup="$dup top=${B601_CAM_TOP:-未配}"
    fi
    if [ -z "$front" ]; then
      front="$fill"
      [ -n "${B601_CAM_FRONT:-}" ] || dup="$dup front=复用hand"
    fi
    [ -n "$dup" ] && echo "⚠️  相机未全绑:$dup —— 采集时该角色与已绑定相机是同一路画面" >&2
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
