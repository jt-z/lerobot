#!/bin/bash
# 离线回归：不启动机械臂、不读相机、不弹终端。
# 用法：bash run_offline_checks.sh
set -u

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
VOICE_DIR="${B601_VOICE_DIR:-/home/jt/dev/m260c-voice-assistant}"
PY="${B601_PY:-/home/jt/dev/miniforge3/envs/lerobot_b601/bin/python}"
# 任务表的 episode 解析调用的是 python；确保用它而不是系统 python
export PATH="$(dirname "$PY"):$PATH"

# 让默认值检查不受当前 shell 覆盖影响
unset B601_POLICY B601_SUBTASK_TIMEOUT B601_WAIT_MODE B601_ASSUME_READY
unset B601_REPLAY_RERUN
unset B601_REPLAY_EPISODE_A1 B601_REPLAY_EPISODE_B1 B601_REPLAY_EPISODE_B2
unset B601_REPLAY_EPISODE_C1 B601_REPLAY_EPISODE_C2 B601_REPLAY_EPISODE_C3

PASS=0
FAIL=0
WARN=0

ok()   { echo "[PASS] $*"; PASS=$((PASS + 1)); }
bad()  { echo "[FAIL] $*"; FAIL=$((FAIL + 1)); }
warn() { echo "[WARN] $*"; WARN=$((WARN + 1)); }

echo "============================================================"
echo "coffee_expo 离线回归"
echo "  CLI   : $ROOT"
echo "  voice : $VOICE_DIR"
echo "  python: $PY"
echo "============================================================"

# ---------------------------------------------------------------- 1. 语法
echo ""
echo "--- 1. shell 语法 ---"
for f in run_subtask.sh run_coffee.sh run_act_coffee.sh coffee_task_table.sh run_pi05_coffee_250ep.sh; do
  if bash -n "$ROOT/$f" 2>/dev/null; then
    ok "bash -n $f"
  else
    bad "bash -n $f"
  fi
done

# ---------------------------------------------------------------- 2. 任务表
echo ""
echo "--- 2. 任务表 ---"
# shellcheck source=/dev/null
if source "$ROOT/coffee_task_table.sh"; then
  ok "source coffee_task_table.sh"
else
  bad "source coffee_task_table.sh"
fi

if [ "${COFFEE_TASK_IDS:-}" = "A1 B1 B2 C1 C2 C3" ]; then
  ok "任务 ID: $COFFEE_TASK_IDS"
else
  bad "任务 ID 异常: ${COFFEE_TASK_IDS:-<空>}"
fi

EXPECTED="A1:159 B1:47 B2:77 C1:136 C2:98 C3:245"
for pair in $EXPECTED; do
  task="${pair%%:*}"
  want="${pair##*:}"
  got="$(coffee_task_replay_episode "$task" 2>/dev/null || true)"
  if [ "$got" = "$want" ]; then
    ok "replay 中间 episode $task=$got"
  else
    warn "replay episode $task 期望 $want，实际 ${got:-<空>}（数据集可能已更换）"
  fi
done

# ---------------------------------------------------------------- 3. CLI 契约
echo ""
echo "--- 3. CLI 契约（不会动机器人）---"
out="$(bash "$ROOT/run_subtask.sh" X9 2>&1)"
rc=$?
if [ "$rc" -eq 4 ] && printf '%s' "$out" | grep -q "未知子任务"; then
  ok "未知任务 X9 -> rc=4"
else
  bad "未知任务 X9 -> rc=$rc（期望 4）"
fi

bash "$ROOT/run_subtask.sh" A1 --policy nope >/dev/null 2>&1
rc=$?
[ "$rc" -eq 4 ] && ok "未知策略 -> rc=4" || bad "未知策略 -> rc=$rc（期望 4）"

bash "$ROOT/run_subtask.sh" >/dev/null 2>&1
rc=$?
[ "$rc" -eq 4 ] && ok "缺少任务 -> rc=4" || bad "缺少任务 -> rc=$rc（期望 4）"

bash "$ROOT/run_coffee.sh" >/dev/null 2>&1
rc=$?
[ "$rc" -eq 4 ] && ok "run_coffee 缺少参数 -> rc=4" || bad "run_coffee 缺参 -> rc=$rc（期望 4）"

bash "$ROOT/run_coffee.sh" hot_water 9 >/dev/null 2>&1
rc=$?
[ "$rc" -eq 4 ] && ok "非法区域 9 -> rc=4" || bad "非法区域 9 -> rc=$rc（期望 4）"

# ---------------------------------------------------------------- 4. dry-run
echo ""
echo "--- 4. 整杯 dry-run ---"
out="$(bash "$ROOT/run_coffee.sh" hot_water 2 --dry-run 2>&1)"
rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q "A1" \
   && printf '%s' "$out" | grep -q "B1" \
   && printf '%s' "$out" | grep -q "C2"; then
  ok "hot_water + 2 -> A1/B1/C2"
else
  bad "hot_water + 2 dry-run 异常（rc=$rc）"
fi

out="$(bash "$ROOT/run_coffee.sh" americano 3 --dry-run 2>&1)"
rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q "B2" \
   && printf '%s' "$out" | grep -q "C3"; then
  ok "americano + 3 -> B2/C3"
else
  bad "americano + 3 dry-run 异常（rc=$rc）"
fi

# ---------------------------------------------------------------- 5. replay 默认
echo ""
echo "--- 5. replay 默认参数 ---"
if grep -q 'B601_REPLAY_RERUN:-1' "$ROOT/run_subtask.sh"; then
  ok "replay 默认 B601_REPLAY_RERUN=1"
else
  bad "replay 默认 rerun 不是 1"
fi

if grep -q -- '--grace 10' "$ROOT/run_subtask.sh"; then
  ok "replay 启动宽限 10s"
else
  bad "replay 启动宽限不是 10s"
fi

# ---------------------------------------------------------------- 6. 语音静态
echo ""
echo "--- 6. 语音调度层离线检查 ---"
VOICE_FILE="$VOICE_DIR/demo/coffee_voice_trigger.py"
if [ ! -f "$VOICE_FILE" ]; then
  if [ "${B601_SKIP_VOICE:-0}" = "1" ]; then
    warn "跳过语音检查（B601_SKIP_VOICE=1）"
  else
    bad "找不到 $VOICE_FILE"
  fi
else
  if "$PY" -m py_compile "$VOICE_FILE" 2>/dev/null; then
    ok "py_compile coffee_voice_trigger.py"
  else
    bad "py_compile coffee_voice_trigger.py"
  fi

  for pat in 'new_auto_coffee' 'run_subtask.sh' 'subtask_supervisor.py' 'def ask_drink'; do
    if grep -q "$pat" "$VOICE_FILE"; then
      ok "语音文件包含 $pat"
    else
      bad "语音文件缺少 $pat"
    fi
  done

  if B601_VOICE_DIR="$VOICE_DIR" "$PY" "$HERE/check_voice_offline.py"; then
    ok "check_voice_offline.py"
  else
    bad "check_voice_offline.py"
  fi
fi

# ---------------------------------------------------------------- 汇总
echo ""
echo "============================================================"
echo "PASS=$PASS  FAIL=$FAIL  WARN=$WARN"
if [ "$FAIL" -eq 0 ]; then
  echo "ALL OFFLINE CHECKS PASSED"
  exit 0
fi
echo "OFFLINE CHECKS FAILED"
exit 1
