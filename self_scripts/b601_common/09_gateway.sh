#!/bin/bash
# MotorBridge Gateway（Studio 网页版的后端，路由模式，只绑回环口）
# 用法：bash self_scripts/b601_common/09_gateway.sh [start|stop|status]
#   网页端：https://motorbridge.github.io/motorbridge-studio/  （必须在本机浏览器打开，它连 ws://127.0.0.1:9002）
. "$(dirname "$(readlink -f "$0")")/_env_kfzbox.sh"
PORT="${MB_PORT:-9002}"
LOG="$B601_REPO_ROOT/self_scripts/_logs/motorbridge-gateway.log"
PIDF="/tmp/mbgw.pid"
mkdir -p "$(dirname "$LOG")"

running(){ [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

case "${1:-start}" in
  start)
    if running; then echo "已在运行 pid=$(cat $PIDF)"; exit 0; fi
    setsid nohup "$B601_ENV/bin/motorbridge-gateway" --bind 127.0.0.1:$PORT > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"; sleep 3
    ss -ltn | grep -q ":$PORT" && echo "OK 监听 ws://127.0.0.1:$PORT (pid=$(cat $PIDF))，日志 $LOG" \
      || { echo "启动失败，日志："; tail -20 "$LOG"; exit 1; }
    ;;
  stop)
    if running; then kill "$(cat $PIDF)" && rm -f "$PIDF" && echo "已停"; else echo "没在跑"; fi
    ;;
  status)
    if running; then echo "running pid=$(cat $PIDF)"; ss -ltn | grep ":$PORT" || true; else echo "not running"; fi
    ;;
  *) echo "用法: $0 [start|stop|status]"; exit 1 ;;
esac
