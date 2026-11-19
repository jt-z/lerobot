#!/bin/bash
# ACT 真机推理 · coffee_expo 任务 B2：选 Americano（美式）→ 按 Start 冲煮 → 回零位
# 权重：232 /mnt/devdata/b601_expo/runs/act_expo_B2_1119_1814/checkpoints/001800（last）
#       1,800 步 ≈ 13 pass（有效 batch 128），末 loss 0.632（这个任务动作更多模态、loss 天然更高）
#       model.safetensors sha256=4f93154a…aa0c99ef
# 数据：b601_pi05_coffee_expo_20261118_181510 的 task_index=1（21 集 / 18,078 帧）
#
# 用法：
#   bash run_act_B2.sh              # 默认权重，限时 300 秒
#   bash run_act_B2.sh <权重目录> [秒]
TASK_ID=B2
TASK_DESC="Use the gripper tip to select Americano and press the Start button to brew coffee into the cup using the current preset settings. Return to the home pose."
DEFAULT_CKPT="/home/jt/LX/pai0/act_expo_B2_20261119/checkpoints/001800/pretrained_model"
STEP_INFO="step 001800 / 1800（last）"
source "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/_act_expo_common.sh"
act_expo_main "$@"
