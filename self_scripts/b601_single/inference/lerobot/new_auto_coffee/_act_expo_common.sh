#!/bin/bash
# new_auto_coffee 三个 ACT 任务脚本的公共部分（被 run_act_A1.sh / run_act_B1.sh / run_act_B2.sh source）
# 约定：调用方先设 TASK_ID / TASK_DESC / DEFAULT_CKPT / STEP_INFO / TITLE，再 act_expo_main "$@"
#   $1 = 权重目录（默认 DEFAULT_CKPT）  $2 = 运行秒数（默认 300，0 = 不限）

set -e
set -o pipefail

# ---- 环境：必须先在 PATH 前置 lerobot_b601（0.6.2，与训练同版本），否则会拿到 0.4.4 ----
_B601_ENV_BIN="$HOME/dev/miniforge3/envs/lerobot_b601/bin"
if [ -x "$_B601_ENV_BIN/lerobot-rollout" ]; then
  export PATH="$_B601_ENV_BIN:$PATH"
fi

_HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
# shellcheck source=/dev/null
source "$_HERE/../_common.sh"

act_expo_main() {
  MODEL_PATH="${1:-$DEFAULT_CKPT}"
  DURATION="${2:-300}"
  # ACT 一次前向 ~27ms（30Hz 一帧只有 33ms），块内是开环播放：整块 100 步执行完
  # 才重算，接缝处会「停一帧 + 动作跳一下」。这里保持整块执行（100 步），
  # 跳变由下面的接缝平滑摊开，不再靠缩短执行步数换更频繁的重算。
  N_ACTION_STEPS=100
  # 接缝平滑：换块时若某一帧的动作跳变超过阈值（度），就按「臂关节每帧不超过 N 度」
  # 把它摊开——过渡 tick 数由跳变幅度自动算出（如 17° → 6 tick，44° → 15 tick）。
  # 夹爪不参与限速（接近二值执行器，直通）；阈值默认 12°（正常帧位移 2~6°，接缝平均 17°）。
  SEAM_MAX_STEP_DEG=3
  SEAM_JUMP_THRESHOLD_DEG=12
  TASK_DESCRIPTION="$TASK_DESC"      # _common.sh 里有个默认长任务描述，这里必须覆盖
  TITLE="单臂推理（ACT · coffee_expo ${TASK_ID} ${STEP_INFO}）- B601-RS / base 纯推理"
  RUN_TAG="act_expo_${TASK_ID}"

  b601_require_lerobot
  b601_banner "$TITLE"

  # 权重：ACT 是整模型（不是 LoRA adapter），只需要 config.json + model.safetensors
  b601_check_model_files "$MODEL_PATH" config.json model.safetensors
  b601_check_trace_wrapper
  b601_check_can
  b601_check_cameras
  b601_check_calibration

  echo ""
  echo "=========================================="
  echo "推理参数总览"
  echo "=========================================="
  echo "任务：${TASK_ID}  ${STEP_INFO}"
  echo "模型路径：$MODEL_PATH"
  echo "机器人类型：seeed_b601_rs_follower（单臂 B601-RS，can0）"
  echo "策略：base（纯推理，不录制、不保存数据）"
  echo "推理方式：sync（ACT 不支持 RTC）· 每块执行 ${N_ACTION_STEPS}/100 步 · 接缝平滑 ≤${SEAM_MAX_STEP_DEG}°/帧（>${SEAM_JUMP_THRESHOLD_DEG}° 触发）"
  b601_print_run_params
  b601_print_robot_notes
  b601_print_trace_summary model/sent/cmd
  echo "ACT 注意力可视化：❌ 关闭"
  echo "=========================================="
  echo ""

  b601_confirm_zero
  b601_start_run "开始 ACT 推理（coffee_expo ${TASK_ID} ${STEP_INFO}）..." "pi05_${RUN_TAG}"

  local rc=0
  if b601_run_rollout \
      --strategy.type=base \
      --inference.type=sync \
      --inference.seam_max_step_deg="$SEAM_MAX_STEP_DEG" \
      --inference.seam_jump_threshold_deg="$SEAM_JUMP_THRESHOLD_DEG" \
      --policy.n_action_steps="$N_ACTION_STEPS" \
      --policy.path="$MODEL_PATH"; then
    rc=0
  else
    rc=$?
  fi

  echo ""
  if [ "$rc" -ne 0 ]; then
    echo "❌ 推理异常退出（退出码 $rc），完整日志：$LOG_FILE"
    exit "$rc"
  fi
  echo "=========================================="
  echo "✅ ACT 推理结束（coffee_expo ${TASK_ID} ${STEP_INFO}）"
  echo "=========================================="
  echo "本次未保存任何推理数据（base 策略不录制、不落盘）。"
  echo "完整日志：$LOG_FILE"
}
