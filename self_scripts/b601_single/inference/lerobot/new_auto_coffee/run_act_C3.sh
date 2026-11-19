#!/bin/bash
# ACT 真机推理 · coffee_expo 任务 C3：满杯 → 杯位 3 → 回零位
# 六任务重训权重占位；权重到位后改 coffee_task_table.sh 或设 B601_ACT_CKPT_C3。
exec bash "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/run_act_task.sh" C3 "$@"
