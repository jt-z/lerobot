#!/bin/bash
# ACT 推理（新权重）：b601 cup_and_button 单任务（取杯→放白托盘→按左侧第一个亮灯按钮→回零位）
# 权重来自 node29 训练：act_cup_and_button_1110_0319 的 25000 步（最终）（4×4090，batch 64，25k 计划）
# 与 run_act_devcams.sh 的差别只有三处：① 权重路径 ② 任务文本（换成 cup_and_button 数据集的那条）
# ③ 环境固定用 lerobot_b601（lerobot 0.6.2，与训练同版本）。相机 / fps / 方向修正 / 动作追踪仍走 _common.sh。
#
# 用法：
#   bash run_act_cup_button.sh                    # 默认 025000 权重，限时 300s
#   bash run_act_cup_button.sh <权重目录>          # 指定 pretrained_model
#   bash run_act_cup_button.sh <权重目录> 600      # 限时 600s
#   bash run_act_cup_button.sh <权重目录> 0        # 0 = 不限时，Ctrl+C 结束
#
# 运行中热键：
#   r               停止 ACT 推理，进入回放菜单（复用同目录 replay_menu.sh）
#   1 / 2 / 3       回放 b601_mixed 的 ep130 / ep160 / ep190（放置区 1/2/3）
#   q               退出回放菜单（随后脚本结束）

set -e
set -o pipefail

# ==================== 环境（先于 _common.sh：避免激活到 lerobot 0.4.4） ====================
_B601_ENV_BIN="$HOME/dev/miniforge3/envs/lerobot_b601/bin"
if [ -x "$_B601_ENV_BIN/lerobot-rollout" ]; then
  export PATH="$_B601_ENV_BIN:$PATH"
fi

source "$(dirname "$(readlink -f "$0")")/_common.sh"

# ==================== 相机（覆盖 _common.sh 的默认值；与 run_act_devcams.sh 相同） ====================
# 当前在机（2026-11-06 实测 by-id）：
#   hand  = 2609010（腕部，夹爪在画面下方）
#   front = 2608006（前视，能看到整条机械臂和咖啡机）
#   top   = 2610057（顶部，俯视桌面）
B601_CAMERA_DEVS=(
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2609010-video-index0  # hand
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608006-video-index0  # front
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2610057-video-index0  # top
)
CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2609010-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608006-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2610057-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== 任务文本（必须与 cup_and_button/meta/tasks.parquet 完全一致） ====================
# 来源：node29 /home/ubuntu/b601/data/cup_and_button/meta/tasks.parquet task_index=0
TASK_DESCRIPTION="Take a cup from the dispenser, put it on the white tray under the coffee spout, press the first lit button on the left with the gripper tip, then return to the home pose when it blinks"

# ==================== 权重 / 时长 ====================
DEFAULT_CKPT="/home/jt/LX/pai0/act_cup_button_20261110/checkpoints/025000/pretrained_model"
MODEL_PATH="${1:-$DEFAULT_CKPT}"
DURATION="${2:-300}"

# ==================== 回放热键（推理中按 r） ====================
REPLAY_MENU="$(dirname "$(readlink -f "$0")")/replay_menu.sh"
REPLAY_DATASET_ROOT="${REPLAY_DATASET_ROOT:-/home/jt/dev/lerobot/b601_mixed}"
B601_ROLLOUT_BACKGROUND=1                     # 后台跑推理，便于热键打断
_HOTKEY_TMP="$(mktemp -d)"
_PID_FILE="$_HOTKEY_TMP/rollout.pid"
_TRIGGER_FILE="$_HOTKEY_TMP/replay.trigger"
export B601_ROLLOUT_PID_FILE="$_PID_FILE"

# ==================== 推理前检查 ====================
b601_require_lerobot
b601_banner "单臂推理（ACT · 新权重 cup_and_button 25k）- B601-RS / hand=2609010 front=2608006 top=2610057"

b601_check_model_files "$MODEL_PATH" config.json model.safetensors
b601_check_trace_wrapper
b601_check_can

# 相机：缺失直接退出（不复用 b601_check_cameras，那里只告警）
echo ""
for cam in "${B601_CAMERA_DEVS[@]}"; do
  if [ ! -e "$cam" ]; then
    echo "❌ 相机不存在：$cam"
    echo "   先跑 lerobot-find-cameras opencv 或 ls /dev/v4l/by-id 确认实际序列号"
    exit 1
  fi
  echo "✅ $cam"
done

b601_check_calibration

# ==================== 推理参数总览 ====================
echo ""
echo "=========================================="
echo "推理参数总览"
echo "=========================================="
echo "模型路径：$MODEL_PATH"
echo "模型来源：act_cup_and_button_1110_0319 / step 25000（单任务：取杯→放白托盘→按左侧第一个亮灯按钮→回零位）"
echo "机器人类型：seeed_b601_rs_follower（单臂 B601-RS，can0）"
echo "策略：base（纯推理，不录制、不保存数据）"
echo "推理方式：sync（ACT 不支持 RTC）"
b601_print_run_params
b601_print_robot_notes
b601_print_trace_summary model/sent/cmd
echo "ACT 注意力可视化：❌ 关闭"
echo "录制/保存数据：❌ 否（base 策略不录制）"
echo "运行中热键：r = 停止推理并进入回放菜单（1/2/3 → ep130/160/190，q 退出）"
echo "=========================================="
echo ""

b601_confirm_zero
b601_start_run "开始单臂模型推理（ACT · 新权重 cup_and_button 25k）..." "cup_button_act_25k"

# ---- 热键监听：按 r 停止推理，进入回放菜单 ----
_HOTKEY_PID=""
if [ -r /dev/tty ]; then
  (
    while IFS= read -rsn1 _k < /dev/tty; do
      case "$_k" in
        r|R)
          : > "$_TRIGGER_FILE"
          # 等 rollout 的 PID 文件落盘（刚启动瞬间按 r 可能还没有）
          _pid=""
          for _i in 1 2 3 4 5 6 7 8 9 10; do
            _pid=$(cat "$_PID_FILE" 2>/dev/null || true)
            if [ -n "$_pid" ]; then break; fi
            sleep 0.2
          done
          if [ -n "$_pid" ]; then kill -INT "$_pid" 2>/dev/null; fi
          sleep 10   # 给 rollout 留出断开/回零时间，仍未退出再兜底 TERM
          if [ -n "$_pid" ] && kill -0 "$_pid" 2>/dev/null; then
            kill -TERM "$_pid" 2>/dev/null
          fi
          exit 0
          ;;
      esac
    done
  ) &
  _HOTKEY_PID=$!
else
  echo "⚠️  没有可读 /dev/tty，热键 r 不可用（Ctrl+C 仍可结束推理）"
fi

if b601_run_rollout \
  --strategy.type=base \
  --inference.type=sync \
  --policy.path="$MODEL_PATH"; then
  _ROLLOUT_RC=0
else
  _ROLLOUT_RC=$?
fi

if [ -n "$_HOTKEY_PID" ]; then kill "$_HOTKEY_PID" 2>/dev/null || true; fi

# ---- 按过 r：不打印推理完成，直接进回放菜单 ----
if [ -f "$_TRIGGER_FILE" ]; then
  rm -rf "$_HOTKEY_TMP"
  echo ""
  echo "=========================================="
  echo "⏹  已按 r：ACT 推理停止，进入数据回放菜单"
  echo "=========================================="
  DATASET_ROOT="$REPLAY_DATASET_ROOT" bash "$REPLAY_MENU" || true
  exit 0
fi

rm -rf "$_HOTKEY_TMP"
if [ "$_ROLLOUT_RC" -ne 0 ]; then
  echo ""
  echo "❌ 推理异常退出（退出码 $_ROLLOUT_RC），完整日志：$LOG_FILE"
  exit "$_ROLLOUT_RC"
fi

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ 单臂推理结束（ACT · 新权重 cup_and_button 25k）"
echo "=========================================="
echo "本次未保存任何推理数据（base 策略不录制、不落盘）。"
echo "完整日志：$LOG_FILE"
echo "=========================================="
