#!/bin/bash
# B601 设备参数层：通用默认值（两台机器共用）
# 由 b601_common/_env.sh 自动 source，不要直接执行。
# 机器专属差异（环境/数据根/主臂串口/相机序列号/机械臂标定 id）写 devices/<host>.sh，不要改这里。
#
# 覆盖方式：任何 B601_* 变量都能临时覆盖，例如
#   B601_DEVICE=z790 bash self_scripts/b601_common/07_preflight.sh

# ---- 从臂 B601-RS（RobStride，SocketCAN）----
export B601_FOLLOWER_PORT="${B601_FOLLOWER_PORT:-can0}"
export B601_FOLLOWER_ADAPTER="${B601_FOLLOWER_ADAPTER:-socketcan}"
export B601_CAN_BITRATE="${B601_CAN_BITRATE:-1000000}"

# ---- 主从臂标识 ----
# 标定文件按 id 命名（…/calibration/<type>/<id>.json）。两台是不同实物臂，id 必须按机区分，
# 否则两台会共用同一份零位 —— 各机在 devices/<host>.sh 里覆盖成 follower_kfzbox / follower_z790。
# 注意：这里只给"兜底值"，真正的 ${B601_FOLLOWER_ID:- DEFAULT} 在 _env.sh 读完档案之后才定，
# 否则本文件（先 source）会把档案里的值压掉。
B601_FOLLOWER_ID_DEFAULT="follower"
B601_LEADER_ID_DEFAULT="leader"

# ---- 从臂关节方向（实测值，两台一致；某台臂装反或换过电机，就在该机档案里覆盖）----
B601_JOINT_DIRECTIONS_DEFAULT='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'

# ---- 重力补偿 / 遥操作可视化 ----
B601_GRAVITY_COMPENSATION_DEFAULT="true"
# 1 = 遥操作时把相机画面送到可视化后端（默认开，方便边遥操边看视角）
B601_DISPLAY_DATA_DEFAULT="1"
B601_DISPLAY_MODE_DEFAULT="rerun"   # rerun | foxglove

# ---- 相机通用规格（两台一致，差异只在序列号/角色）----
export B601_CAM_WIDTH="${B601_CAM_WIDTH:-640}"
export B601_CAM_HEIGHT="${B601_CAM_HEIGHT:-480}"
export B601_CAM_FPS="${B601_CAM_FPS:-30}"
export B601_CAM_FOURCC="${B601_CAM_FOURCC:-MJPG}"

# ---- 相机角色列表（两台可能不同：kfzbox 只有 hand+top，z790 有 hand+front+top）----
B601_CAM_ROLES_DEFAULT="hand front top"

# ---- 主臂候选串口（设备档案没写死 B601_LEADER_PORT 时按此顺序取第一个存在的）----
export B601_LEADER_CANDIDATES_DEFAULT="${B601_LEADER_CANDIDATES_DEFAULT:-/dev/rebot_leader /dev/serial/by-id/usb-1a86_USB_Serial*-if00-port0 /dev/ttyUSB* /dev/ttyACM*}"

# ---- 角色 → 相机 by-id 序列号（机器档案里填；留空 = 该角色本次不采）----
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

# 角色名 → 该角色配置的序列号（未配置输出空）
b601_cam_serial() {
  local var="B601_CAM_$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
  printf '%s' "${!var:-}"
}

# 生成 lerobot 的 --robot.cameras JSON
#   角色顺序 = B601_CAM_ROLES（默认 hand front top）
#   只输出"已配置且当前在线"的角色 —— 不再拿别的相机顶替：kfzbox 实测同一个
#   /dev/video 节点开两次第二路直接连不上/0 帧，所以"复用"这条路是死的。
#   缺的角色会明确告警（这台机就是这么少一路），一个角色都解析不出来时才退回老兜底。
b601_cameras_json() {
  local cams=() c
  while IFS= read -r c; do [ -n "$c" ] && cams+=("$c"); done \
    < <(ls /dev/v4l/by-id/*video-index0 2>/dev/null | sort)

  if [ "${#cams[@]}" -eq 0 ]; then
    echo "❌ 没有找到任何 /dev/v4l/by-id/*video-index0 相机（设备档案: ${B601_DEVICE:-?}）" >&2
    return 1
  fi

  local role serial p role_var
  local roles=() paths=() seen=" "
  for role in ${B601_CAM_ROLES:-hand front top}; do
    role_var="B601_CAM_$(printf '%s' "$role" | tr '[:lower:]' '[:upper:]')"
    serial=$(b601_cam_serial "$role")
    if [ -z "$serial" ]; then
      echo "⚠️  $role 视角未配置（devices/${B601_DEVICE:-?}.sh 的 $role_var 为空）—— 本次不采集该视角" >&2
      continue
    fi
    p=$(b601_cam_path "$serial") || true
    if [ -z "$p" ]; then
      echo "⚠️  $role 配置的相机 $serial 当前不在线 —— 本次不采集该视角（不会拿别的相机顶替）" >&2
      continue
    fi
    case "$seen" in
      *" $p "*)
        echo "❌ $role 与已绑定角色指向同一台相机 $serial —— 同一 /dev/video 开两次必然失败，已跳过该角色" >&2
        continue
        ;;
    esac
    seen="$seen$p "
    roles+=("$role"); paths+=("$p")
  done

  if [ "${#roles[@]}" -eq 0 ]; then
    local i=0
    for role in ${B601_CAM_ROLES:-hand front top}; do
      [ "$i" -ge "${#cams[@]}" ] && break
      roles+=("$role"); paths+=("${cams[$i]}"); i=$((i+1))
    done
    {
      echo "⚠️  没有任何角色配置可用，退回按现有 ${#cams[@]} 路相机的顺序套角色（顺序未必对，先跑 08_cam_probe.py 核对画面）："
      local j=0
      for j in "${!cams[@]}"; do printf '     %s -> %s\n' "${roles[$j]:-?}" "${cams[$j]}"; done
      echo "   要固定角色，把序列号写进 self_scripts/b601_common/devices/${B601_DEVICE:-<host>}.sh 的 B601_CAM_<ROLE>"
    } >&2
  fi

  local n=${#roles[@]} k
  printf '{\n'
  for ((k = 0; k < n; k++)); do
    local body
    body=$(printf 'type: opencv, index_or_path: %s, width: %s, height: %s, fps: %s, fourcc: %s' \
      "${paths[$k]}" "$B601_CAM_WIDTH" "$B601_CAM_HEIGHT" "$B601_CAM_FPS" "$B601_CAM_FOURCC")
    if [ "$k" -lt $((n - 1)) ]; then
      printf '  %s: {%s},\n' "${roles[$k]}" "$body"
    else
      printf '  %s: {%s}\n' "${roles[$k]}" "$body"
    fi
  done
  printf '}'
}
