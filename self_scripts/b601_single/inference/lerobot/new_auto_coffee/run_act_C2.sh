#!/bin/bash
# ACT 真机推理 · coffee_expo 任务 C2：满杯 → 杯位 2 → 回零位
# prompt 与默认权重见 coffee_task_table.sh。
exec bash "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/run_act_task.sh" C2 "$@"
