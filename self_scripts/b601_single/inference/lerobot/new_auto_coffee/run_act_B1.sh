#!/bin/bash
# ACT 真机推理 · coffee_expo 任务 B1：选 Hot Water → 按 Start 出热水 → 回零位
# 权重：232 /mnt/devdata/b601_expo/runs/act_expo_B1_1119_1745/checkpoints/002600（last）
#       2,600 步 ≈ 13 pass（有效 batch 128），末 loss 0.333，model.safetensors sha256=a0301f1e…032e0f05
# 数据：b601_pi05_coffee_expo_20261118_181510 的 task_index=2（30 集 / 25,810 帧）
#
# 用法：
#   bash run_act_B1.sh              # 默认权重，限时 300 秒
#   bash run_act_B1.sh <权重目录> [秒]
TASK_ID=B1
TASK_DESC="Use the gripper tip to select Hot Water and press the Start button to dispense hot water into the cup using the current preset settings. Return to the home pose."
DEFAULT_CKPT="/home/jt/LX/pai0/act_expo_B1_20261119/checkpoints/002600/pretrained_model"
STEP_INFO="step 002600 / 2600（last）"
source "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/_act_expo_common.sh"
act_expo_main "$@"
