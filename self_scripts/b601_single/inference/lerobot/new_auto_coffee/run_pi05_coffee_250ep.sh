#!/bin/bash
# B601-RS 真机推理：用 **250 集 run 的 pi0.5 权重**（默认 12k：expo012000_250ep）跑 6 个子任务
#
# 结论先说：6 个子任务共用同一份权重，切换子任务**只换 prompt**（--task 文本），
#           不需要重训、也不需要复制机份权重；12k 的 bf16 全量权重已覆盖全部 6 个技能。
#
# 底层仍是 ../run_pi05_coffee_expo.sh（相机 rename_map / lite 包装器 / CAN·相机·校准自检
# / base 策略 / sync 推理全部不变），本脚本只做三件事：
#   1. 把 MODEL_PATH 指到 250 集 run 的 expo012000_250ep，并带上它自己的 sha256；
#   2. 把 TRAINED_TASKS 放成 6 个（旧 102 集权重的默认只声明 A1,B1,B2,C2，C1/C3 会被拦）；
#   3. 把「TASK 短名 / 数据集 task_index」翻译成 base 脚本的 TASK_ID，并核对 prompt 逐字一致。
#
# ⚠️ 提示词的硬约束：--task 必须与数据集 meta/tasks.parquet 里的文本**逐字一致**
#    （pi05 把 prompt 编成 token 与 state token 拼在一起；改一个词就是分布外输入）。
#    本脚本默认会读 kfzbox 上的数据集核对，CHECK_TEXT=0 可跳过。
#    A1..C3 是采集脚本的技能编号，**顺序与数据集 task_index 不同**（见下表），别混。
#
# 用法：
#   ./run_pi05_coffee_250ep.sh                    # 只打印任务表，不动机器人
#   TASK=A1 ./run_pi05_coffee_250ep.sh            # 空杯→接水盘
#   TASK=C3 DURATION=120 ./run_pi05_coffee_250ep.sh
#   TI=5 ./run_pi05_coffee_250ep.sh               # 用数据集 task_index 选（0..5）
#   MODEL_PATH=$HOME/pi05_bf16/expo009000_250ep EXPECT_SHA256=a9803454... ./run_pi05_coffee_250ep.sh  # 退回 9k
#   VERIFY_SHA=1 ./run_pi05_coffee_250ep.sh       # 上机前校验 sha256（8.7GB，约 15 秒）
#
# 权重改名约定的对照（务必别混）：
#   250 集 run：expo009000_250ep / expo012000_250ep   ← 带 _250ep，本脚本用
#   102 集旧 run：expo008000 / expo012000             ← 无后缀，4 任务
#
# 数据集/权重对应关系见 README.md 与 1115新场景咖啡机/07_9k权重合并与3060验证_20261120.md

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
BASE="$(cd "$SCRIPT_DIR/.." && pwd)/run_pi05_coffee_expo.sh"
DS_ROOT="${DS_ROOT:-/home/jt/b601_data/b601_pi05_coffee_expo_20261118_181510}"

# 与底层脚本同版本的环境（这里的 python3 用于 prompt 逐字核对，需要 pandas）
_B601_ENV_BIN="$HOME/dev/miniforge3/envs/lerobot_b601/bin"
if [ -d "$_B601_ENV_BIN" ]; then
  export PATH="$_B601_ENV_BIN:$PATH"
fi

# 默认 12k：6 任务均已训练；9k 与 12k 同锚点对比 5/6 任务更优（均值 2.416→2.319°）
MODEL_PATH="${MODEL_PATH:-$HOME/pi05_bf16/expo012000_250ep}"
EXPECT_SHA256="${EXPECT_SHA256:-f70669263aed4ae1908dddda97362ba47861cff354b630ece37a0e1ef130c541}"
TRAINED_TASKS="${TRAINED_TASKS:-A1,B1,B2,C1,C2,C3}"
CHECK_TEXT="${CHECK_TEXT:-1}"

TASK_IDS=(A1 B1 B2 C1 C2 C3)
TASK_TI=(1 2 4 3 0 5)          # 数据集 task_index
TASK_EPS=(80 40 40 40 40 10)   # 250 集库里的集数
TASK_DESC=(
  "空杯 → 咖啡机下银色接水盘"
  "按 Hot Water 并 Start（出热水）"
  "按 Americano 并 Start（出咖啡）"
  "满杯 → 放置区 1"
  "满杯 → 放置区 2"
  "满杯 → 放置区 3"
)

print_table() {
  echo "可用子任务（同一份 $(basename "$MODEL_PATH") 权重，切换只改 prompt）："
  printf "  %-4s %-8s %-6s %s\n" "TASK" "数据集TI" "集数" "说明"
  local i
  for i in "${!TASK_IDS[@]}"; do
    printf "  %-4s %-8s %-6s %s\n" "${TASK_IDS[$i]}" "${TASK_TI[$i]}" "${TASK_EPS[$i]}" "${TASK_DESC[$i]}"
  done
  echo
  echo "权重：$MODEL_PATH"
  echo "示例：TASK=C1 DURATION=120 $0     |     TI=3 DURATION=120 $0"
}

# ---------- 选任务 ----------
SEL="${TASK:-}"
if [ -z "$SEL" ] && [ -n "${TI:-}" ]; then
  for i in "${!TASK_TI[@]}"; do
    if [ "${TASK_TI[$i]}" = "${TI}" ]; then SEL="${TASK_IDS[$i]}"; fi
  done
  [ -n "$SEL" ] || { echo "❌ TI=$TI 不是数据集 task_index（合法值 0..5）"; print_table; exit 2; }
fi
if [ -z "$SEL" ]; then
  print_table
  exit 0
fi
SEL="$(printf '%s' "$SEL" | tr '[:lower:]' '[:upper:]')"
FOUND=0
for i in "${!TASK_IDS[@]}"; do [ "${TASK_IDS[$i]}" = "$SEL" ] && FOUND=1; done
if [ "$FOUND" != 1 ]; then
  echo "❌ 未知任务 TASK=$SEL（支持 A1 B1 B2 C1 C2 C3，或 TI=0..5）"
  print_table
  exit 2
fi

# ---------- 参数自检 ----------
[ -f "$BASE" ] || { echo "❌ 找不到底层脚本：$BASE"; exit 3; }
[ -f "$MODEL_PATH/model.safetensors" ] || { echo "❌ 权重不存在：$MODEL_PATH/model.safetensors"; print_table; exit 3; }
if [ "$MODEL_PATH" != "$HOME/pi05_bf16/expo009000_250ep" ] && [ "${EXPECT_SHA256}" = "a9803454d9746c94786b47b47dc54da53922fd75d60964e28518935e45ffe9db" ]; then
  echo "⚠️  已换 MODEL_PATH 但 EXPECT_SHA256 还是 expo009000_250ep 的；VERIFY_SHA=1 会校验失败"
fi
if [ "$MODEL_PATH" != "$HOME/pi05_bf16/expo012000_250ep" ] && [ "${EXPECT_SHA256}" = "f70669263aed4ae1908dddda97362ba47861cff354b630ece37a0e1ef130c541" ]; then
  echo "⚠️  已换 MODEL_PATH 但 EXPECT_SHA256 还是 expo012000_250ep 的；VERIFY_SHA=1 会校验失败"
fi

if [ "$CHECK_TEXT" = "1" ] && [ -d "$DS_ROOT/meta" ]; then
  python3 - "$DS_ROOT" "$SEL" "$BASE" <<'PY' || exit 4
import sys, pandas as pd
root, sel, base = sys.argv[1], sys.argv[2], sys.argv[3]
ids = ["A1", "B1", "B2", "C1", "C2", "C3"]
import re
src = open(base, encoding="utf-8").read()
texts = [t.strip().strip('"') for t in re.search(r"TASKS=\((.*?)\n\)", src, re.S).group(1).strip().split("\n") if t.strip()]
want = texts[ids.index(sel)]
t = pd.read_parquet(f"{root}/meta/tasks.parquet")
if "task" in t.columns:
    ds = {int(i): str(x).strip() for i, x in zip(t["task_index"], t["task"])}
else:
    ds = {int(v): str(k).strip() for k, v in t["task_index"].items()}
if want not in ds.values():
    print(f"❌ {sel} 的 prompt 与数据集 tasks.parquet 不逐字一致，拒绝上机")
    sys.exit(4)
print(f"✅ prompt 逐字核对通过：{sel}（数据集 task_index={[k for k,v in ds.items() if v==want]}）")
print(f"   {want}")
PY
elif [ "$CHECK_TEXT" = "1" ]; then
  echo "⚠️  未找到数据集 $DS_ROOT，跳过 prompt 逐字核对"
fi

echo
echo "=== 本机推理选择 ==="
echo "任务      : $SEL  (${TASK_DESC[$(( $(printf '%s\n' "${TASK_IDS[@]}" | grep -n "^$SEL$" | cut -d: -f1) - 1 ))]})"
echo "权重      : $MODEL_PATH"
echo "权重 sha  : ${EXPECT_SHA256:0:16}…（VERIFY_SHA=${VERIFY_SHA:-0}）"
echo "已声明训练任务: $TRAINED_TASKS"
echo "时长      : DURATION=${DURATION:-0}（0=不限）"
echo "======================"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
  echo
  echo "PLAN_ONLY=1：不启动推理。实际会执行："
  echo "  MODEL_PATH=$MODEL_PATH EXPECT_SHA256=${EXPECT_SHA256:0:16}… TRAINED_TASKS=$TRAINED_TASKS TASK_ID=$SEL \\"
  echo "    bash $BASE"
  exit 0
fi

# ---------- 交给底层脚本（它负责 CAN/相机/校准自检 + 参数总览 + 零位确认 + 启动）----------
export MODEL_PATH EXPECT_SHA256 TRAINED_TASKS TASK_ID="$SEL"
exec bash "$BASE" "$@"
