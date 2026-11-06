#!/bin/bash
# 临时验证脚本：用"当前在机的三台 JYU2C 相机"跑一次 ACT 推理。
# 与 run_act.sh 的唯一区别是相机设备路径；任务文本 / fps / 方向修正 / 动作追踪全部沿用 _common.sh。
# 等 _common.sh 改成兼容多套相机后可删除本文件。
#
# 当前在机（2026-11-06 实测 by-id）：
#   hand  = 2609010（腕部，夹爪在画面下方）
#   front = 2608006（前视，能看到整条机械臂和咖啡机）
#   top   = 2610057（顶部，俯视桌面）
#
# 用法：
#   bash run_act_devcams.sh                    # 默认权重 015000，限时 300s
#   bash run_act_devcams.sh <权重目录>          # 指定 pretrained_model，限时 300s
#   bash run_act_devcams.sh <权重目录> 600      # 限时 600s
#   bash run_act_devcams.sh <权重目录> 0        # 0 = 不限时，Ctrl+C 结束

set -e
set -o pipefail

source "$(dirname "$(readlink -f "$0")")/_common.sh"

# ==================== 相机（覆盖 _common.sh 的默认值） ====================
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

# ==================== 权重 / 时长 ====================
DEFAULT_CKPT="/home/jt/LX/pai0/act_b601_20261105_130447/checkpoints/015000/pretrained_model"
MODEL_PATH="${1:-$DEFAULT_CKPT}"
DURATION="${2:-300}"

# ==================== 推理前检查 ====================
b601_require_lerobot
b601_banner "单臂推理（ACT）- B601-RS / 临时相机配置（hand=2609010 front=2608006 top=2610057）"

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
echo "机器人类型：seeed_b601_rs_follower（单臂 B601-RS，can0）"
echo "策略：base（纯推理，不录制、不保存数据）"
echo "推理方式：sync（ACT 不支持 RTC）"
b601_print_run_params
b601_print_robot_notes
b601_print_trace_summary model/sent/cmd
echo "ACT 注意力可视化：❌ 关闭"
echo "录制/保存数据：❌ 否（base 策略不录制）"
echo "=========================================="
echo ""

b601_confirm_zero
b601_start_run "开始单臂模型推理（ACT / 临时相机配置）..." "make_coffee_act_devcams"

b601_run_rollout \
  --strategy.type=base \
  --inference.type=sync \
  --policy.path="$MODEL_PATH"

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ 单臂推理结束（ACT / 临时相机配置）"
echo "=========================================="
echo "本次未保存任何推理数据（base 策略不录制、不落盘）。"
echo "完整日志：$LOG_FILE"
echo "=========================================="
