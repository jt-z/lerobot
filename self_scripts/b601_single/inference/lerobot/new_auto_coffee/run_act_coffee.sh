#!/bin/bash
# coffee_expo ACT 任务派发器：不用记文件名
#   bash run_act_coffee.sh A1            # 端空杯上托盘，默认 300 秒
#   bash run_act_coffee.sh B2 600        # 美式，600 秒
#   bash run_act_coffee.sh               # 不带参数 → 打印任务表
set -u
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
T="${1:-}"
case "$T" in
  A1|a1) shift; exec bash "$HERE/run_act_A1.sh" "$@" ;;
  B1|b1) shift; exec bash "$HERE/run_act_B1.sh" "$@" ;;
  B2|b2) shift; exec bash "$HERE/run_act_B2.sh" "$@" ;;
  C2|c2) shift; exec bash "$HERE/run_act_C2.sh" "$@" ;;
  ""|-h|--help)
    cat <<'EOF'
coffee_expo ACT 单任务推理（四个技能各一个模型，互不共享）

  任务   含义                                     权重(step)   训练帧数
  A1    端空杯 → 咖啡机银色接水托盘 → 回零位        003100       30,633（30 集）
  B1    选 Hot Water → 按 Start 出热水 → 回零位      002600       25,810（30 集）
  B2    选 Americano → 按 Start 冲煮 → 回零位        001800       18,078（21 集）
  C2    满杯 → 放到杯位 2 → 回零位                   001800       17,718（21 集）

用法：
  bash run_act_coffee.sh A1 [秒]            # 秒省略 = 300；0 = 不限时（Ctrl+C 结束）
  bash run_act_coffee.sh C2 600
  MODEL_PATH=/path/to/pretrained_model bash run_act_coffee.sh A1   # 临时换权重

前置：机械臂在零位（夹爪闭合）、can0 已起、三路相机在线；脚本会逐项自检并在开跑前要求按 ENTER 确认。
EOF
    ;;
  *) echo "未知任务：$T（可用 A1 / B1 / B2 / C2）"; exit 2 ;;
esac
