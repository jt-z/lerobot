#!/bin/bash
# coffee_expo 任务派发器（兼容旧入口）：A1 / B1 / B2 / C1 / C2 / C3
# 现在统一委托给 run_subtask.sh：默认 ACT，可追加 --policy / --timeout。
#
#   bash run_act_coffee.sh A1            # 端空杯上托盘，ACT，上限 40 秒
#   bash run_act_coffee.sh B2 600        # 兼容旧写法：第二个数字 = 超时秒数
#   bash run_act_coffee.sh C2 --policy replay
#   bash run_act_coffee.sh               # 不带参数 → 打印任务表
set -u
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
T="${1:-}"
case "$T" in
  A1|a1) T=A1; shift ;;
  B1|b1) T=B1; shift ;;
  B2|b2) T=B2; shift ;;
  C1|c1) T=C1; shift ;;
  C2|c2) T=C2; shift ;;
  C3|c3) T=C3; shift ;;
  ""|-h|--help)
    cat <<'EOF'
coffee_expo 单任务入口（统一委托 run_subtask.sh）

  任务   含义                                   ACT 权重              训练帧数
  A1    端空杯 → 咖啡机银色接水托盘 → 回零位      003100                30,633（30 集）
  B1    选 Hot Water → 按 Start → 回零位         002600                25,810（30 集）
  B2    选 Americano → 按 Start → 回零位         001800                18,078（21 集）
  C1    满杯 → 杯位 1 → 回零位                   六任务重训占位（未到位）
  C2    满杯 → 杯位 2 → 回零位                   001800                17,718（21 集）
  C3    满杯 → 杯位 3 → 回零位                   六任务重训占位（未到位）

用法：
  bash run_act_coffee.sh A1 [秒]                 # 秒省略 = 40；0 = 不限时（Ctrl+C 结束）
  bash run_act_coffee.sh C2 --policy replay      # act / replay / pi05
  bash run_act_coffee.sh B1 --timeout 30 --yes   # 跳过开跑确认

运行中：键 0 = 判定成功提前结束 | 键 1 = 判定失败 | Ctrl+C = 中止（均先归零）。
退出码：0 成功 · 2 操作员判失败 · 3 超时 · 4 环境/权重错误 · 130 中止。
整杯流程请用 run_coffee.sh <hot_water|americano> <1|2|3>。
EOF
    exit 0
    ;;
  *) echo "未知任务：$T（可用 A1 / B1 / B2 / C1 / C2 / C3）"; exit 4 ;;
esac

exec bash "$HERE/run_subtask.sh" "$T" "$@"
