#!/bin/bash
# ACT 真机推理 · coffee_expo 任务 C2：把满杯从咖啡机银色接水托盘 → 放到杯位 2 → 回零位
# 权重：232 /mnt/devdata/b601_expo/runs/act_expo_C2_1119_1834/checkpoints/001800（last）
#       1,800 步 ≈ 13 pass（有效 batch 128），末 loss 0.668（四个任务里最难，中途 loss 到过 1.7）
#       model.safetensors sha256=bc581551…7d0d5515
# 数据：b601_pi05_coffee_expo_20261118_181510 的 task_index=3（21 集 / 17,718 帧）
#
# 用法：
#   bash run_act_C2.sh              # 默认权重，限时 300 秒
#   bash run_act_C2.sh <权重目录> [秒]
TASK_ID=C2
TASK_DESC="Pick up the filled cup from the silver drip tray under the coffee machine spout and place it in cup placement area 2. Return to the home pose."
DEFAULT_CKPT="/home/jt/LX/pai0/act_expo_C2_20261119/checkpoints/001800/pretrained_model"
STEP_INFO="step 001800 / 1800（last）"
source "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/_act_expo_common.sh"
act_expo_main "$@"
