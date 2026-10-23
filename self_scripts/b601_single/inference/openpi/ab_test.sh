#!/usr/bin/env bash
# 真机 A/B 四组一键跑：不同"噪声模式 × 重规划步数"
#   bash ab_test.sh A   默认噪声 + exec5        （对照，已有 163301.csv）
#   bash ab_test.sh B   默认噪声 + exec10/lead5
#   bash ab_test.sh C   固定噪声 + exec5
#   bash ab_test.sh D   固定噪声 + exec10/lead5  ← 推荐先跑这个
# 可选环境变量：DUR=60（秒）  SEED=42  CKPT=trim330  PORT=8000
set -uo pipefail
CASE=${1:-D}
DUR=${DUR:-60}
SEED=${SEED:-42}
CKPT=${CKPT:-trim330}
PORT=${PORT:-8000}
SERVE=/home/kf/b601_pi05/start_serve.sh
LOGDIR=/home/kf/b601_pi05/logs
PY=${PY:-/home/kf/b601_pi05/qgf_openpi_env/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)

case "$CASE" in
  A) NOISE=fresh; EXEC=5 ;;
  B) NOISE=fresh; EXEC=10 ;;
  C) NOISE=fixed; EXEC=5 ;;
  D) NOISE=fixed; EXEC=10 ;;
  *) echo "用法: bash ab_test.sh A|B|C|D"; exit 2 ;;
esac
NOISE_SEED_USED="-"
[ "$NOISE" = fixed ] && NOISE_SEED_USED=$SEED

echo "==== 组 $CASE：噪声=$NOISE  execute_steps=$EXEC  duration=${DUR}s ===="
if [ "$NOISE" = fixed ]; then
  B601_NOISE_SEED=$SEED "$SERVE" "$CKPT" | grep -E "flow noise|started"
else
  "$SERVE" "$CKPT" | grep -E "flow noise|started"
fi

# 等 READY（看最新的 serve 日志）
SLOG=$(ls -t "$LOGDIR"/serve_*.log 2>/dev/null | head -1)
for i in $(seq 1 40); do
  grep -q "^READY" "$SLOG" 2>/dev/null && break
  sleep 3
done
ss -ltn | grep -q ":$PORT" || { echo "❌ server 没起来，看 $SLOG"; exit 1; }
grep -m1 -E "^READY|FIXED flow" "$SLOG" | cut -c1-140
echo "server OK (:$PORT)"

# 预热：第一次推理要 JIT ~8s，别让它吃掉正式 run 的窗口
echo "[预热] dry-run 3s（吃掉 JIT）"
bash "$HERE/run_rollout.sh" "${CKPT#trim}" --dry-run --duration 3 >/dev/null 2>&1
ss -ltn | grep -q ":$PORT" || { echo "❌ 预热后 server 掉了（已知的稳定性坑），重跑一次本脚本"; exit 1; }

# 正式 run
echo "[正式] ${DUR}s 真机 rollout —— 手放急停，场地清空"
ARGS=(--duration "$DUR")
[ "$EXEC" = 10 ] && ARGS+=(--execute-steps 10 --prefetch-lead 5)
bash "$HERE/run_rollout.sh" "${CKPT#trim}" "${ARGS[@]}" 2>&1 | grep -E "共 |预取|首个|轨迹|ERROR|Traceback"

CSV=$(ls -t "$LOGDIR"/rollout_pi05_ws_*.csv 2>/dev/null | head -1)
# 记组别台账（否则事后只有时间戳，会认错组）
echo "$(date +%H:%M:%S)|$CASE|$NOISE|$EXEC|$NOISE_SEED_USED|$DUR|$CSV" >> "$LOGDIR/ab_cases.log"
echo
echo "==== 轨迹分析：$CSV ===="
"$PY" "$HERE/tools/analyze_rollout.py" "$CSV" 2>/dev/null || \
  python3 "$HERE/tools/analyze_rollout.py" "$CSV" 2>/dev/null || \
  echo "（分析脚本跑不了，把上面 CSV 路径发我，我来算）"
