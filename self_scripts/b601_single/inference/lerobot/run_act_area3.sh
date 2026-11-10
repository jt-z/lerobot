#!/bin/bash
# 包装脚本：place_area3 的 ACT 推理（等价于 bash run_act_place_area.sh 3 "$@"）
# 用法: bash run_act_area3.sh [权重目录] [时长秒]
#   默认权重 /home/jt/LX/pai0/act_place_area3_20261110/checkpoints/006000/pretrained_model，默认时长 300s
exec bash "$(dirname "$(readlink -f "$0")")/run_act_place_area.sh" 3 "$@"
