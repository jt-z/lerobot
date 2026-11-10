#!/bin/bash
# ACT 推理（place_area 三套权重 · 一个入口脚本）
# 三个任务：把咖啡杯从咖啡机出水口下面的银托盘取走，放进 1 / 2 / 3 号取杯区
# 权重来自 node29：act_place_area{1,2,3}_1110_0718 的 6000 步（4×4090，等效 batch 64，各 ~13 epoch）
#
# 与 run_act_devcams.sh 的差别（其余全部沿用 _common.sh 与它一致）：
#   ① 一个入口按 area 选权重 + 任务文本；② 环境固定 lerobot_b601（lerobot 0.6.2，与训练同版本）
#
# 用法：
#   bash run_act_place_area.sh 1                 # area1，默认 006000 权重，限时 300s
#   bash run_act_place_area.sh 2 600             # area2，限时 600s
#   bash run_act_place_area.sh 3 0               # area3，0 = 不限时，Ctrl+C 结束
#   bash run_act_place_area.sh 1 <权重目录> 300   # 指定 pretrained_model（比如换 002000/004000 做对比）
#
# 也可以用三个包装脚本（等价于上面第一种写法）：
#   bash run_act_area1.sh   / bash run_act_area2.sh   / bash run_act_area3.sh

set -e
set -o pipefail

# ==================== 参数：area 必填，其余可省 ====================
AREA="${1:-}"
case "$AREA" in
  1|2|3) ;;
  *)
    echo "用法: bash $(basename "$0") <1|2|3> [权重目录] [时长秒]"
    echo "  1/2/3 = 取杯后放进 1/2/3 号取杯区"
    exit 2
    ;;
esac
CKPT_ARG="${2:-}"
DUR_ARG="${3:-}"

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

# ==================== area → 默认权重 + 任务文本 ====================
# 任务文本必须与对应数据集 meta/tasks.parquet 完全一致（node29 上逐字取的）
case "$AREA" in
  1)
    DEFAULT_CKPT="/home/jt/LX/pai0/act_place_area1_20261110/checkpoints/006000/pretrained_model"
    TASK_DESCRIPTION="Pick up the coffee cup from the silver tray under the coffee machine spout and place it in cup placement area 1"
    RUN_TAG="place_area1_act_6k"
    ;;
  2)
    DEFAULT_CKPT="/home/jt/LX/pai0/act_place_area2_20261110/checkpoints/006000/pretrained_model"
    TASK_DESCRIPTION="Pick up the coffee cup from the silver tray under the coffee machine spout and place it in cup placement area 2"
    RUN_TAG="place_area2_act_6k"
    ;;
  3)
    DEFAULT_CKPT="/home/jt/LX/pai0/act_place_area3_20261110/checkpoints/006000/pretrained_model"
    TASK_DESCRIPTION="Pick up the coffee cup from the silver tray under the coffee machine spout and place it in cup placement area 3"
    RUN_TAG="place_area3_act_6k"
    ;;
esac

MODEL_PATH="${CKPT_ARG:-$DEFAULT_CKPT}"
DURATION="${DUR_ARG:-300}"

# ==================== 推理前检查 ====================
b601_require_lerobot
b601_banner "单臂推理（ACT · place_area${AREA} 6k）- B601-RS / hand=2609010 front=2608006 top=2610057"

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
echo "任务：place_area${AREA}（取杯 → 放进 ${AREA} 号取杯区）"
echo "模型路径：$MODEL_PATH"
echo "模型来源：act_place_area${AREA}_1110_0718 / step 6000（单任务）"
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
b601_start_run "开始单臂模型推理（ACT · place_area${AREA} 6k）..." "$RUN_TAG"

b601_run_rollout \
  --strategy.type=base \
  --inference.type=sync \
  --policy.path="$MODEL_PATH"

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ 单臂推理结束（ACT · place_area${AREA} 6k）"
echo "=========================================="
echo "本次未保存任何推理数据（base 策略不录制、不落盘）。"
echo "完整日志：$LOG_FILE"
echo "=========================================="
