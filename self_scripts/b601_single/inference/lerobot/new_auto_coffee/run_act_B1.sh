#!/bin/bash
# ACT 真机推理 · coffee_expo 任务 B1：选 Hot Water → 按 Start → 回零位
# prompt 与默认权重见 coffee_task_table.sh。
#
# 用法：
#   bash run_act_B1.sh              # 默认权重，限时 300 秒
#   bash run_act_B1.sh <权重目录> 600
exec bash "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/run_act_task.sh" B1 "$@"
