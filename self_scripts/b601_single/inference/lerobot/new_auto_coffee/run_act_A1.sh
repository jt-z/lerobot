#!/bin/bash
# ACT 真机推理 · coffee_expo 任务 A1：端空杯 → 咖啡机托盘 → 回零位
# prompt 与默认权重见 coffee_task_table.sh。
#
# 用法：
#   bash run_act_A1.sh              # 默认权重，限时 300 秒
#   bash run_act_A1.sh <权重目录>    # 换权重
#   bash run_act_A1.sh <权重目录> 600
exec bash "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/run_act_task.sh" A1 "$@"
