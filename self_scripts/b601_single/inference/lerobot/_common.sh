#!/usr/bin/env bash
# b601 单臂推理公共库 —— 由 run_act.sh / run_smolvla.sh / run_pi05_lora.sh source。
#
# 本文件只定义默认值与函数；source 时不检查硬件、不打印、不联网。
# 各脚本的固定顺序：source 本文件 → 覆盖模型专属变量 → 调检查函数 → 打印总览 → b601_run_rollout。
#
# 所有硬件/相机/任务文本都须与采集脚本 self_scripts/b601_single/02_record.sh 保持一致。

_B601_INFER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== 公共默认值（脚本可在 source 之后覆盖） ====================
# B601 从臂 = can0（SocketCAN，PEAK PCAN-USB，1Mbps 经典 CAN）；推理只用从臂，不需要主臂
B601_FOLLOWER_PORT="${B601_FOLLOWER_PORT:-can0}"
# 设备 ID = 校准文件名，须与采集时一致
FOLLOWER_ID="${FOLLOWER_ID:-follower}"

# 3 路相机：hand = JYU2C 2608076（腕部）、front = 2607031、top = 2607060（顶部）
# 用 /dev/v4l/by-id 稳定路径（/dev/videoN 随插拔顺序变化，不可靠）
B601_CAMERA_DEVS=(
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0
)
# 相机名 → 设备路径（--robot.cameras 的取值，dracus 解析）
CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# B601 方向修正：必须与采集（02_record.sh 的 B601_JOINT_DIRECTIONS）一致，全 -1.0。
# 原因：seeed_b601_rs_follower.send_action 对每个关节先乘 joint_directions
#       （缺失关节默认 0.0，会把目标裁到 0 不动），且推理动作与训练动作必须同坐标系。
B601_JOINT_DIRECTIONS='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'

# 相对动作幅度上限（robot.max_relative_target）已停用：该裁剪比较的是"本帧 goal 与当前
# 位置之差"，而 goal 先被 joint_limits 裁进量程（最大跨度 ~290 度），凡 ≥300 的值都拦不住
# 任何动作，只会每帧刷警告。不传即默认 None；确实要限速参照 b601_so101_bimanual/01_teleop_test.sh。

# base 策略没有 dataset，任务描述必须走顶层 --task（见 rollout/context.py：
# task_str = cfg.dataset.single_task if cfg.dataset else cfg.task），且须与采集/训练时完全一致
TASK_DESCRIPTION="Pick up the paper cup, place it on the silver tray of the coffee machine, pick up the cube, press the button with the cube (red light on), wait about 4 seconds, release the button (red light off), put the cube on the table first, then move the cup from the coffee machine to the table"

FPS=30          # 必须与训练数据集 fps（30Hz）一致，否则时序错位
DURATION=0      # 总运行时长上限（秒）：0 = 不限，跑到 Ctrl+C；>0 则到点自动退出
INFERENCE_TYPE="sync"   # sync = 换 chunk 时阻塞前向（可记录 model 列）；rtc = 后台线程异步（不阻塞）

# 动作追踪（1 = 用 action_trace.py 包装 lerobot-rollout，运行时打补丁，不改 lerobot 源码）：
#   model = 策略输出（度，数据集 action 坐标系）——仅 INFERENCE_TYPE=sync 有效
#   sent  = 传入 robot.send_action 的 dict（插值器 + robot_action_processor 之后）
#   cmd   = send_action 返回值 = 乘 joint_directions、裁 joint_limits 之后的目标角（度）
ACTION_TRACE=1
ACTION_TRACE_WRAPPER="$_B601_INFER_DIR/../../../tools/action_trace.py"
ACTION_TRACE_LOG="$HOME/dev/lerobot/self_scripts/_logs/action_test.log"

LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"

# ==================== 函数 ====================

# 找 lerobot-rollout（没有就激活 conda 环境），找不到直接退出
b601_require_lerobot() {
  if ! command -v lerobot-rollout >/dev/null 2>&1; then
    echo "未找到 lerobot-rollout，尝试激活 lerobot conda 环境..."
    local base
    for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
      if [ -f "$base/etc/profile.d/conda.sh" ]; then
        source "$base/etc/profile.d/conda.sh"
        conda activate lerobot
        break
      fi
    done
  fi
  if ! command -v lerobot-rollout >/dev/null 2>&1; then
    echo "❌ 错误：lerobot-rollout 不可用"
    echo "   请手动执行: conda activate lerobot"
    exit 1
  fi
  echo "✅ lerobot-rollout 可用"
  echo ""
}

# 顶部标题
b601_banner() {
  echo "=========================================="
  echo "$1"
  echo "=========================================="
  echo ""
}

# 只检查（缺文件直接退出，不打印成功信息）
b601_require_files() {
  local dir="$1"
  shift
  if [ ! -d "$dir" ]; then
    echo "❌ 模型路径不存在：$dir"
    exit 1
  fi
  local f
  for f in "$@"; do
    [ -f "$dir/$f" ] || { echo "❌ 缺少 $dir/$f"; exit 1; }
  done
}

# 检查模型权重：目录存在 + 列出的文件齐全
b601_check_model_files() {
  b601_require_files "$@"
  echo "✅ 模型文件存在：$1"
}

# 动作追踪包装器（开追踪时必须有，避免跑到一半才发现）
b601_check_trace_wrapper() {
  if [ "$ACTION_TRACE" = "1" ] && [ ! -f "$ACTION_TRACE_WRAPPER" ]; then
    echo "❌ 缺少动作追踪包装器：$ACTION_TRACE_WRAPPER（或把 ACTION_TRACE 设为 0）"
    exit 1
  fi
}

# B601 CAN（不存在直接退出；存在但未 UP 则尝试拉起）
b601_check_can() {
  echo ""
  if ! ip link show "$B601_FOLLOWER_PORT" >/dev/null 2>&1; then
    echo "❌ $B601_FOLLOWER_PORT 不存在"
    exit 1
  fi
  if ! ip link show "$B601_FOLLOWER_PORT" 2>/dev/null | grep -q "state UP"; then
    echo "⚠️  配置 $B601_FOLLOWER_PORT（1Mbps 经典 CAN）..."
    sudo ip link set "$B601_FOLLOWER_PORT" down 2>/dev/null || true
    sudo ip link set "$B601_FOLLOWER_PORT" type can bitrate 1000000
    sudo ip link set "$B601_FOLLOWER_PORT" up
  fi
  echo "✅ B601 CAN 已就绪"
}

# 3 路相机
b601_check_cameras() {
  echo ""
  local cam
  for cam in "${B601_CAMERA_DEVS[@]}"; do
    if [ ! -e "$cam" ]; then
      echo "❌ 警告：摄像头不存在 $cam"
    else
      echo "✅ $cam 已连接"
    fi
  done
}

# 从臂校准文件（推理须与采集同 id）
b601_check_calibration() {
  echo ""
  local calib="$HOME/.cache/huggingface/lerobot/calibration/robots/seeed_b601_rs_follower/${FOLLOWER_ID}.json"
  if [ -f "$calib" ]; then
    echo "✅ $calib"
  else
    echo "❌ 缺失校准文件：$calib"
    echo "   请先运行 self_scripts/b601_common/02_follower_calibration.sh 完成从臂校准"
    exit 1
  fi
}

# 参数总览里各策略共用的部分（任务/时长/频率/相机）
b601_print_run_params() {
  echo "任务描述：$TASK_DESCRIPTION"
  if [ "$DURATION" = "0" ]; then
    echo "运行时长：不限（一直推理到 Ctrl+C）"
  else
    echo "运行时长：${DURATION}秒 (约 $((DURATION / 60)) 分钟)"
  fi
  echo "推理频率：${FPS} Hz"
  echo "摄像头：hand / front / top（3 路）"
}

# 参数总览里机器人侧的固定说明（方向修正/相对动作上限），接在摄像头之后
b601_print_robot_notes() {
  echo "B601 方向修正：全 -1.0（与采集一致）"
  echo "相对动作上限：未启用（不传 max_relative_target，默认不做相对裁剪）"
}

# 动作追踪那行总览：$1 = 记录内容（默认 sent/cmd；ACT 传 model/sent/cmd）
b601_print_trace_summary() {
  local what="${1:-sent/cmd}"
  if [ "$ACTION_TRACE" = "1" ]; then
    echo "动作追踪：✅ ${what} 逐 tick 记录到 $ACTION_TRACE_LOG"
    if [ "$INFERENCE_TYPE" = "rtc" ]; then
      echo "           （rtc 后端下 model 列为 n/a，只有 sent/cmd；需要 model 列请把 INFERENCE_TYPE 改成 sync）"
    fi
  else
    echo "动作追踪：❌ 关闭（直接调用 lerobot-rollout）"
  fi
}

# 开跑前的人工确认
b601_confirm_zero() {
  read -p "确认 B601 从臂（can0）处于零位（夹爪闭合）、周边安全，按 ENTER 开始推理，Ctrl+C 取消..." dummy
}

# 打印开跑标题、设好 Rerun 缓冲、开日志文件：$1 = 标题，$2 = 日志文件名中段
b601_start_run() {
  echo ""
  echo "🚀 $1"
  echo ""
  # Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
  export RERUN_FLUSH_NUM_BYTES=10000000
  export LEROBOT_RERUN_MEMORY_LIMIT="30%"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/inference_b601_$2_$(date +%Y%m%d_%H%M%S).log"
  echo "📝 日志文件：$LOG_FILE"
  echo ""
}

# 离线跑 HF 资产（本机无外网）：$1 = HF_HUB_CACHE 目录
b601_export_offline_hf() {
  export HF_HUB_CACHE="$1"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

# 组装并执行 lerobot-rollout：$@ = 策略专属参数（--strategy.type / --inference.type /
# --policy.path / 去噪步数 等），机器人侧公共参数由本函数补齐。
# 可选变量：CAMERA_RENAME_MAP（非空则追加 --rename_map）、ACT_VIZ（1 = 用 ACT 可视化包装器）
b601_run_rollout() {
  local -a cmd=()
  local -a rename_arg=()
  if [ "${ACT_VIZ:-0}" = "1" ]; then
    cmd=(python "$ACT_VIZ_WRAPPER")
    export ACT_VIZ_EVERY_N ACT_VIZ_ANALYSIS ACT_VIZ_HEADS_EVERY ACT_VIZ_HEADS ACT_VIZ_ALPHA ACT_VIZ_BLUEPRINT
    export ACT_VIZ_MP4 ACT_VIZ_MP4_DIR ACT_VIZ_MP4_MODE ACT_VIZ_MP4_CRF
    export ACTION_TRACE ACTION_TRACE_LOG  # viz 包装器据此决定是否顺带装载动作追踪
  elif [ "$ACTION_TRACE" = "1" ]; then
    cmd=(python "$ACTION_TRACE_WRAPPER")
    export ACTION_TRACE_LOG
  else
    cmd=(lerobot-rollout)
  fi
  # 相机 key 重命名（robot → policy）：仅 pi05 基座这类策略需要
  if [ -n "${CAMERA_RENAME_MAP:-}" ]; then
    rename_arg=(--rename_map="$CAMERA_RENAME_MAP")
  fi

  "${cmd[@]}" "$@" \
    --robot.type=seeed_b601_rs_follower \
    --robot.id="$FOLLOWER_ID" \
    --robot.port="$B601_FOLLOWER_PORT" \
    --robot.can_adapter=socketcan \
    --robot.gravity_compensation=true \
    --robot.joint_directions="$B601_JOINT_DIRECTIONS" \
    --robot.cameras="$CAMERAS" \
    "${rename_arg[@]}" \
    --task="$TASK_DESCRIPTION" \
    --fps="$FPS" \
    --duration="$DURATION" \
    --display_data=true \
    --display_compressed_images=false \
    --play_sounds=false 2>&1 | tee "$LOG_FILE"
}
