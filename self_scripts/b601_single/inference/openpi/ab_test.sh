#!/usr/bin/env bash
# 真机 A/B 四组一键跑（v3）
#   bash ab_test.sh D            固定噪声 + exec10/lead5（推荐）
#   bash ab_test.sh C            固定噪声 + exec5
#   bash ab_test.sh B            默认噪声 + exec10/lead5
#   bash ab_test.sh A            默认噪声 + exec5
#   CHECK=1 bash ab_test.sh D    只验流程不动机器人（最后一次用 --dry-run，不记账）
# 环境变量：DUR=60  SEED=42  CKPT=trim330  PORT=8000
set -uo pipefail
CASE=${1:-D}
DUR=${DUR:-60}
SEED=${SEED:-42}
CKPT=${CKPT:-trim330}
PORT=${PORT:-8000}
EXECN=${EXECN:-}            # 覆盖执行步数（如 25/50），留空用组默认
LEADN=${LEADN:-}            # 覆盖预取提前量；留空按下式自动：exec10→5
CHECK=${CHECK:-0}
PRIME=${PRIME:-}          # 起点摆位目标：7 个逗号分隔的度（不要带 --prime-state）
SERVE=/home/kf/b601_pi05/start_serve.sh
LOGDIR=/home/kf/b601_pi05/logs
PY=${PY:-/home/kf/b601_pi05/qgf_openpi_env/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
STAMP=$(date +%m%d_%H%M%S)
RUNLOG="$LOGDIR/ab_${CASE}_${STAMP}.log"
mkdir -p "$LOGDIR"

case "$CASE" in
  A) NOISE=fresh; EXEC=5 ;;
  B) NOISE=fresh; EXEC=10 ;;
  C) NOISE=fixed; EXEC=5 ;;
  D) NOISE=fixed; EXEC=10 ;;
  *) echo "用法: bash ab_test.sh A|B|C|D"; exit 2 ;;
esac
NOISE_SEED_USED="-"; [ "$NOISE" = fixed ] && NOISE_SEED_USED=$SEED

exec > >(tee -a "$RUNLOG") 2>&1
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "组=$CASE 噪声=$NOISE exec=$EXEC lead=${LEAD:-auto} dur=${DUR}s check=$CHECK"
[ -n "$PRIME" ] && say "起点摆位: $PRIME"
say "全量日志: $RUNLOG"
say "前置检查: $(ss -ltn 2>/dev/null | grep -c ":$PORT") 个进程占用 $PORT（正常应为 1）"

# ---------- 1. 起 server ----------
say "--- [1/5] 重启 server ($CKPT) ---"
if [ "$NOISE" = fixed ]; then
  B601_NOISE_SEED=$SEED "$SERVE" "$CKPT" 2>&1 | sed 's/^/    /'
else
  "$SERVE" "$CKPT" 2>&1 | sed 's/^/    /'
fi
SLOG=$(ls -t "$LOGDIR"/serve_*.log 2>/dev/null | head -1)
say "server 日志: $SLOG"

# ---------- 2. 等 READY ----------
say "--- [2/5] 等 READY（最多 120s）---"
READY=0
for i in $(seq 1 40); do
  if grep -q "^READY" "$SLOG" 2>/dev/null && ss -ltn 2>/dev/null | grep -q ":$PORT"; then READY=1; break; fi
  sleep 3
done
if [ "$READY" != 1 ]; then
  say "❌ 120s 内没 READY。日志尾部："
  tail -15 "$SLOG" | sed 's/^/    /'
  exit 1
fi
grep -m1 "^READY" "$SLOG" | cut -c1-150 | sed 's/^/    /'
grep -m1 "FIXED flow" "$SLOG" | sed 's/^/    /'
say "✅ server READY (:$PORT)"

# ---------- 3. 预热 ----------
say "--- [3/5] 预热 dry-run 3s（吃掉 JIT ~8s）---"
bash "$HERE/run_rollout.sh" "${CKPT#trim}" --dry-run --duration 3 2>&1 | tail -3 | sed 's/^/    /'
if ! ss -ltn 2>/dev/null | grep -q ":$PORT"; then
  say "❌ 预热后 server 掉了（已知稳定性坑）。重跑本脚本即可。"
  exit 1
fi
say "✅ 预热完成"

# ---------- 4. 正式跑（后台 + 进度心跳）----------
ARGS=(--duration "$DUR")
[ -n "$PRIME" ] && ARGS+=("--prime-state=$PRIME")   # 必须用 = 形式，否则 -5.5 会被当成选项
[ -n "${EXTRA:-}" ] && ARGS+=($EXTRA)           # 其它额外透传参数
[ -n "$EXECN" ] && EXEC=$EXECN
LEAD=${LEADN:-$(( (EXEC + 1) / 2 ))}   # exec10→5, exec25→13, exec50→25
ARGS+=(--execute-steps "$EXEC" --prefetch-lead "$LEAD")
# 关键：正式 run 在后台跑，stdin 不是终端 → 必须 --yes 跳过 ENTER 确认，否则 EOFError 秒退(exit 134)
[ "$CHECK" = 1 ] && ARGS+=(--dry-run) || ARGS+=(--yes)
if [ "$CHECK" = 1 ]; then say "⚠️ CHECK=1：dry-run，机器人不会动"; fi
say "--- [4/5] 正式 rollout ${DUR}s ---"
say "命令: bash run_rollout.sh ${CKPT#trim} ${ARGS[*]}"
if [ "$CHECK" = 0 ]; then
  say "⚠️ 3 秒后机械臂开始动作 —— 手放急停，想取消按 Ctrl+C"
  for k in 3 2 1; do say "   $k"; sleep 1; done
fi
OUT="$LOGDIR/.ab_run_${STAMP}.out"
bash "$HERE/run_rollout.sh" "${CKPT#trim}" "${ARGS[@]}" > "$OUT" 2>&1 &
PID=$!
T0=$(date +%s)
while kill -0 $PID 2>/dev/null; do
  sleep 5
  C=$(ls -t "$LOGDIR"/rollout_pi05_ws_*.csv 2>/dev/null | head -1)
  N=$(( $(wc -l < "$C" 2>/dev/null || echo 1) - 1 ))
  say "  …已跑 $(( $(date +%s) - T0 ))s / ${DUR}s（CSV $N 步）"
done
wait $PID; RC=$?
sed 's/^/    /' "$OUT"
say "run_rollout 退出码=$RC"
rm -f "$OUT"

# ---------- 5. 结果核对 ----------
CSV=$(ls -t "$LOGDIR"/rollout_pi05_ws_*.csv 2>/dev/null | head -1)
say "--- [5/5] 结果核对 ---"
[ -z "$CSV" ] && { say "❌ 没有生成 CSV"; exit 1; }
ROWS=$(( $(wc -l < "$CSV") - 1 ))
EXP=$(( DUR * 30 ))
say "CSV: $CSV"
say "实际步数=$ROWS  预期≈$EXP"
if [ "$CHECK" = 0 ]; then
  TAG=$CASE
  [ "$ROWS" -lt $((EXP/2)) ] && TAG="$CASE-partial"   # 只记有效 run，避免污染对比
  echo "$(date +%H:%M:%S)|$TAG|$NOISE|$EXEC|$NOISE_SEED_USED|$DUR|$CSV" >> "$LOGDIR/ab_cases.log"
  say "已记入 ab_cases.log（$TAG）"
fi
if [ "$ROWS" -lt 5 ]; then
  say "⚠️ 几乎没有数据，跳过分析。原因见上面 run_rollout 输出。日志: $RUNLOG"
  exit 1
fi
echo
"$PY" "$HERE/../tools/analyze_rollout.py" "$CSV" 2>&1 | sed 's/^/    /'
if [ "$ROWS" -lt $((EXP/2)) ]; then
  say "⚠️ 只跑了 ${ROWS} 步（预期 $EXP）。上面 run_rollout 输出里若有 Traceback/ERROR 即为原因。"
  say "   把这条日志发我: $RUNLOG"
else
  say "✅ 跑满。日志: $RUNLOG"
fi
