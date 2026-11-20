#!/bin/bash
# 依次跑 6 个子任务的 pi05 真机推理（同一份 250 集 run 的权重，默认 12k；只换 prompt）
#
# 设计口径：**每段之间必须人工放行**（回车才进下一段），因为每段之间要人把杯/水/按键现场复原。
#   - 任何一段失败（非 0 退出）立即中止队列，避免在错状态下继续跑
#   - 读到 EOF（人走开了 / 输出被重定向）也直接中止，不会自己往下跑
#   - 每段的时长、顺序都可以用环境变量改
#
# 用法：
#   bash run_all_tasks_seq.sh                       # 默认顺序 A1 C2 C1 B2 B1 C3
#   ORDER="A1 B2" bash run_all_tasks_seq.sh          # 只跑指定几段
#   DUR_A1=240 bash run_all_tasks_seq.sh             # 改某段时长（默认见下表）
#
# 单段手跑（等价）：
#   TASK=A1 DURATION=180 ~/dev/lerobot/self_scripts/b601_single/inference/lerobot/new_auto_coffee/run_pi05_coffee_250ep.sh

set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN="${RUN:-$HERE/new_auto_coffee/run_pi05_coffee_250ep.sh}"
[ -f "$RUN" ] || RUN="$HERE/run_pi05_coffee_250ep.sh"   # 兼容脚本被挪到 new_auto_coffee/ 之前的布局
ORDER=${ORDER:-A1 C2 C1 B2 B1 C3}

dur_of() {
  case "$1" in
    A1) echo "${DUR_A1:-180}" ;;   # 空杯→接水盘，动作最长（80 集）
    C2) echo "${DUR_C2:-120}" ;;   # 满杯→放置区2
    C1) echo "${DUR_C1:-120}" ;;   # 满杯→放置区1
    B2) echo "${DUR_B2:-90}"  ;;   # 按 Americano
    B1) echo "${DUR_B1:-90}"  ;;   # 按 Hot Water
    C3) echo "${DUR_C3:-60}"  ;;   # 满杯→放置区3（只有 10 集，最不可靠）
    *)  echo 120 ;;
  esac
}

echo "=================================================="
echo " 即将依次执行：$ORDER"
echo " 每段之间会停下来等你按回车（现场要复原杯子/水量/按键）"
echo " 权重：$( grep -m1 'MODEL_PATH="${MODEL_PATH:-' "$RUN" | sed 's/.*:-//; s/}.*//' )（默认 12k；要 9k 就带 MODEL_PATH+EXPECT_SHA256）"
echo " Ctrl+C 可随时中止整个队列"
echo "=================================================="

for t in $ORDER; do
  d=$(dur_of "$t")
  echo
  echo "--------------------------------------------------"
  read -r -p "▶ 按回车开始 TASK=$t（时长 ${d}s）；Ctrl+C 中止 → " _ || { echo "已中止（未读到输入）"; exit 0; }
  echo "--------------------------------------------------"
  TASK="$t" DURATION="$d" bash "$RUN"
  rc=$?
  echo
  echo "◀ TASK=$t 结束 rc=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "!! $t 非 0 退出，队列中止（已完成段的日志在 ~/dev/lerobot/self_scripts/_logs/）"
    exit "$rc"
  fi
done
echo
echo "=================================================="
echo " 全部段结束。日志目录：~/dev/lerobot/self_scripts/_logs/"
echo "=================================================="
