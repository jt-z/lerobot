#!/bin/bash
# 交互式回放菜单（B601-RS 单臂）：按 1/2/3 分别回放 b601_mixed 的 130 / 160 / 190，q 退出
# 复用 self_scripts/b601_single/03_replay.sh（CAN 检查 / 标定检查 / rerun 显示 / fps 节拍 / 断开回零）
#
# 用法：
#   bash replay_menu.sh
#   DATASET_ROOT=/path/to/dataset bash replay_menu.sh   # 换数据集（默认 b601_mixed）

set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPLAY_SH="$SCRIPT_DIR/../../03_replay.sh"
DATASET_ROOT="${DATASET_ROOT:-/home/jt/dev/lerobot/b601_mixed}"

if [ ! -f "$REPLAY_SH" ]; then
  echo "❌ 找不到回放脚本：$REPLAY_SH"
  exit 1
fi
if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
  echo "❌ 数据集路径无效：$DATASET_ROOT"
  exit 1
fi

while true; do
  echo ""
  echo "=========================================="
  echo " B601 数据回放菜单  数据集：$(basename "$DATASET_ROOT")"
  echo "=========================================="
  echo "  1) episode 130   → 放置区 1（859 帧 / 28.6s）"
  echo "  2) episode 160   → 放置区 2（644 帧 / 21.5s）"
  echo "  3) episode 190   → 放置区 3（959 帧 / 32.0s）"
  echo "  q) 退出"
  echo "=========================================="
  read -r -p "请选择 [1/2/3/q]: " choice

  case "$choice" in
    1|2|3)
      case "$choice" in
        1) EP=130; DESC="859 帧 / 28.6s" ;;
        2) EP=160; DESC="644 帧 / 21.5s" ;;
        3) EP=190; DESC="959 帧 / 32.0s" ;;
      esac
      echo ""
      echo "🚀 回放 episode $EP（$DESC）"
      # 03_replay.sh 会再要求按 ENTER 确认（安全位姿检查）
      DATASET_ROOT="$DATASET_ROOT" EPISODE="$EP" bash "$REPLAY_SH" || echo "⚠️  回放退出码非 0（episode $EP）"
      ;;
    q|Q|0)
      echo "退出回放菜单。"
      break
      ;;
    *)
      echo "⚠️  无效输入：$choice（请输入 1 / 2 / 3 / q）"
      ;;
  esac
done
