#!/bin/bash
# ACT 真机推理 · coffee_expo 任务 A1：端空杯 → 放到咖啡机下方的银色接水托盘 → 回零位
# 权重：232 /mnt/devdata/b601_expo/runs/act_expo_A1_1119_1712/checkpoints/003100（last）
#       3,100 步 ≈ 13 pass（有效 batch 128），末 loss 0.278，model.safetensors sha256=1aca66f6…dae7919
# 数据：b601_pi05_coffee_expo_20261118_181510 的 task_index=0（30 集 / 30,633 帧）
#
# 用法：
#   bash run_act_A1.sh              # 默认权重，限时 300 秒
#   bash run_act_A1.sh <权重目录>    # 换权重
#   bash run_act_A1.sh <权重目录> 600
TASK_ID=A1
TASK_DESC="Pick up an empty paper cup from the cup dispenser and place it on the silver drip tray under the coffee machine spout. Return to the home pose."
DEFAULT_CKPT="/home/jt/LX/pai0/act_expo_A1_20261119/checkpoints/003100/pretrained_model"
STEP_INFO="step 003100 / 3100（last）"
source "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/_act_expo_common.sh"
act_expo_main "$@"
