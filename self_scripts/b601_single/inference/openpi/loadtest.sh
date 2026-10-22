#!/usr/bin/env bash
# π0.5 全参权重加载自检：起 8010 端口 → 等 READY → 零图/零位前向一次 → 自动关掉
# 不碰机器人、不占 8000 端口，用于"权重刚传过来，先确认能加载、能出动作"。
#
# 用法（在本目录下）：
#   bash loadtest.sh                 # 默认 trim330（checkpoints/trim330_16942）
#   bash loadtest.sh 330             # 旧版 330（24,653）
#   bash loadtest.sh 462
#   bash loadtest.sh /path/to/ckpt   # 任意 openpi ckpt（B601_REPO 需自行保证对得上）
set -uo pipefail
cd "$(dirname "$0")" || exit 1
# shellcheck disable=SC1091
source ./env.sh

ALIAS=${1:-trim330}
PORT=${2:-8010}
case "$ALIAS" in
  330)     CKPT="$B601_HOME/checkpoints/330_24653";                     export B601_REPO=b601/v21_clean_eps0_329 ;;
  462)     CKPT="$B601_HOME/checkpoints/462_34750";                     export B601_REPO=b601/v21_clean ;;
  trim330) CKPT="$B601_HOME/checkpoints/trim330_16942";                 export B601_REPO=b601/v21_clean_trim_eps0_329 ;;
  trim8471) CKPT="$B601_HOME/checkpoints/trim330_8471";                 export B601_REPO=b601/v21_clean_trim_eps0_329 ;;
  *)       CKPT="$ALIAS" ;;
esac
LOG=/tmp/loadtest_$(basename "$CKPT").log
: > "$LOG"

{
  echo "ckpt=$CKPT"
  echo "repo=$B601_REPO   port=$PORT   python=$PY"
  echo "files=$(find "$CKPT" -type f | wc -l)  params_bytes=$(du -sb "$CKPT/params" | cut -f1)"
  echo "assets=$(ls "$CKPT/assets/b601/" 2>/dev/null | tr '\n' ' ')"
  [ -f "$CKPT/assets/$B601_REPO/norm_stats.json" ] && echo "norm_stats=OK" || echo "norm_stats=MISSING(检查 B601_REPO)"
} | tee -a "$LOG"

# 清掉可能占着端口的旧 serve_policy（取 PID 再 kill，避免 pkill -f 把当前 shell 带走）
for p in $(pgrep -f 'serve_policy\.py' 2>/dev/null || true); do
  [ "$p" != "$$" ] && kill "$p" 2>/dev/null && echo "killed old server pid=$p"
done
sleep 1

"$PY" serve_policy.py --ckpt "$CKPT" --port "$PORT" >> "$LOG" 2>&1 &
SPID=$!
echo "server_pid=$SPID" >> "$LOG"

for _ in $(seq 1 80); do
  grep -q '^READY' "$LOG" && break
  kill -0 "$SPID" 2>/dev/null || { echo "SERVER_DIED（看 $LOG）" | tee -a "$LOG"; break; }
  sleep 3
done
grep -E '^READY' "$LOG" | head -2

if grep -q '^READY' "$LOG"; then
  "$PY" - "$PORT" 2>&1 <<'PY' | tee -a "$LOG"
import sys, time, json
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
p = WebsocketClientPolicy('127.0.0.1', int(sys.argv[1]))
try:
    print('metadata', json.dumps(p.get_server_metadata(), ensure_ascii=False)[:160])
except Exception as e:
    print('metadata_unavailable', type(e).__name__)
obs = dict(state=np.zeros(7, dtype=np.float32),
           images={k: np.zeros((480, 640, 3), dtype=np.uint8) for k in ('front', 'hand', 'top')},
           prompt='pour coffee into the cup')
t0 = time.time(); out = p.infer(obs); dt = (time.time() - t0) * 1000
a = np.asarray(out['actions'], dtype=np.float32)[:, :7]
print('first_call_ms', round(dt, 1), 'actions', a.shape, 'step0', np.round(a[0], 2).tolist())
t0 = time.time(); p.infer(obs)
print('steady_ms', round((time.time() - t0) * 1000, 1))
PY
  kill "$SPID" 2>/dev/null
  sleep 2
  echo "LOADTEST_OK（可以把 server 起到 8000 跑真机了）" | tee -a "$LOG"
else
  kill "$SPID" 2>/dev/null
  echo "LOADTEST_FAIL —— 完整日志：$LOG"; exit 1
fi
