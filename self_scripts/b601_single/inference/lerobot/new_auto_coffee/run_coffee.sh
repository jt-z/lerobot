#!/bin/bash
# 咖啡展会高层调度：饮品 + 区域 → 自动展开子任务序列并逐个执行。
#
#   热水：A1 → B1 → [人工确认萃取完成] → C<区域>
#   美式：A1 → B2 → [人工确认萃取完成] → C<区域>
#
# 用法：
#   bash run_coffee.sh hot_water 2
#   bash run_coffee.sh americano 3 --policy act
#   bash run_coffee.sh --drink hot_water --area 1 --policy replay --timeout 30
#   bash run_coffee.sh hot_water 2 --dry-run        # 只打印计划
#
# 退出码沿用单子任务契约：0 成功 · 2 操作员判失败 · 3 超时 · 4 环境/权重错误 · 130 中止。
# 任何一步非 0 都会立即停止后续步骤。
#
# 说明：外部程序如果自己控制节拍，建议直接逐个调用 run_subtask.sh，而不是本脚本；
# 本脚本主要给“人工在旁边监控”的整杯流程用。

set -u

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/coffee_task_table.sh"

POLICY="${B601_POLICY:-act}"
TIMEOUT="${B601_SUBTASK_TIMEOUT:-40}"
WAIT_MODE="${B601_WAIT_MODE:-enter}"   # enter | none | 秒数
YES=0
DRY_RUN=0
DRINK=""
AREA=""

while [ $# -gt 0 ]; do
  case "$1" in
    --drink) DRINK="${2:-}"; shift 2 ;;
    --area) AREA="${2:-}"; shift 2 ;;
    --policy) POLICY="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT="${2:-}"; shift 2 ;;
    --wait) WAIT_MODE="${2:-}"; shift 2 ;;
    --yes) YES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "❌ 未知选项：$1"; exit 4 ;;
    *)
      if [ -z "$DRINK" ]; then DRINK="$1"
      elif [ -z "$AREA" ]; then AREA="$1"
      else echo "❌ 多余参数：$1"; exit 4
      fi
      shift
      ;;
  esac
done

case "$DRINK" in
  hot_water|hot|hotwater|热水) DRINK="hot_water"; DRINK_ID="B1" ;;
  americano|美式|america) DRINK="americano"; DRINK_ID="B2" ;;
  *)
    echo "❌ 未指定饮品：hot_water（热水）或 americano（美式）"
    echo "   用法：bash run_coffee.sh <hot_water|americano> <1|2|3>"
    exit 4
    ;;
esac

case "$AREA" in
  1|2|3) ;;
  C1|c1) AREA=1 ;;
  C2|c2) AREA=2 ;;
  C3|c3) AREA=3 ;;
  *)
    echo "❌ 未指定放置区域：1 / 2 / 3（对应 C1 / C2 / C3）"
    exit 4
    ;;
esac

case "$POLICY" in
  act|replay|pi05) ;;
  *) echo "❌ 未知策略：$POLICY（可用：act / replay / pi05）"; exit 4 ;;
esac

PLAN=(A1 "$DRINK_ID" "C${AREA}")

echo "=========================================="
echo "咖啡流程：${DRINK} → 放置区 ${AREA}"
echo "=========================================="
for step in "${PLAN[@]}"; do
  label="$(coffee_task_prompt "$step")"
  if [ "$step" = "$DRINK_ID" ]; then
    echo "  $step：$label"
    if [ "$WAIT_MODE" = "enter" ]; then
      echo "       └─ 之后暂停，人工确认萃取完成再继续"
    elif [ "$WAIT_MODE" != "none" ]; then
      echo "       └─ 之后等待 ${WAIT_MODE}s 再继续"
    fi
  else
    echo "  $step：$label"
  fi
done
echo "策略：$POLICY    单步上限：${TIMEOUT}s"
echo "=========================================="
echo ""

if [ "$DRY_RUN" = "1" ]; then
  echo "（--dry-run：只打印计划，不动机器人）"
  exit 0
fi

if [ "$YES" = "0" ]; then
  if [ -t 0 ]; then
    printf '确认从臂在零位、杯子与咖啡机就绪，按空格开始（Ctrl+C 取消）...'
    while :; do IFS= read -r -n 1 -s _key || exit 130; [ "$_key" = " " ] && break; done
    echo
    YES=1
  else
    echo "⚠️  非交互终端：跳过总确认（等同 --yes），请确保已就绪"
    YES=1
  fi
fi

for step in "${PLAN[@]}"; do
  echo ""
  echo "▶ 执行子任务 $step"
  rc=0
  bash "$HERE/run_subtask.sh" "$step" --policy "$POLICY" --timeout "$TIMEOUT" --yes || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo ""
    echo "❌ 子任务 $step 失败（exit=$rc），流程中止。"
    exit "$rc"
  fi

  if [ "$step" = "$DRINK_ID" ]; then
    case "$WAIT_MODE" in
      enter)
        if [ -t 0 ]; then
          printf '萃取完成后按空格进入放杯阶段（Ctrl+C 中止）...'
          while :; do IFS= read -r -n 1 -s _key || exit 130; [ "$_key" = " " ] && break; done
          echo
        else
          echo "⚠️  非交互终端：跳过 B→C 等待"
        fi
        ;;
      none) ;;
      *)
        if [[ "$WAIT_MODE" =~ ^[0-9]+$ ]]; then
          echo "⏳ 等待 ${WAIT_MODE}s（模拟萃取时间）..."
          sleep "$WAIT_MODE"
        else
          echo "❌ --wait 只支持 enter / none / 秒数，收到：$WAIT_MODE"
          exit 4
        fi
        ;;
    esac
  fi
done

echo ""
echo "=========================================="
echo "✅ 流程完成：${DRINK} → 放置区 ${AREA}"
echo "=========================================="
