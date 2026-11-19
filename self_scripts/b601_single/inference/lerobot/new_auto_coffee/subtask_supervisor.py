#!/usr/bin/env python
"""单子任务监督器：超时 / 按键 0、1 / 中止，统一归零等待与退出码。

它本身不做推理，只负责“看住”一个后端进程（ACT / pi0.5 / replay）：

  * 键 0：操作员判定本子任务成功 → 发 SIGINT 提前结束，等机械臂归零后以 0 退出；
  * 键 1：操作员判定失败       → 同样先归零，以 2 退出；
  * Ctrl+C / SIGTERM：中止     → 同样先归零，以 130 退出；
  * 超时：ACT/pi0.5 由 lerobot-rollout 的 --duration 到点自行退出（退出后
    return_to_initial_position + disconnect 里的 safe_zero 负责归零）；replay 由本
    监督器在 --timeout + --grace 后发 SIGINT。归零都发生在后端子进程内部，
    本进程只等待它退出，不会自己去动电机。

退出码是给外部程序看的稳定契约：
  0 = 成功（键 0，或 replay 正常放完）
  2 = 操作员判定失败（键 1）
  3 = 超时（ACT/pi0.5 跑满时长仍无人确认，或 replay 超过上限）
  4 = 后端异常（进程非零退出、权重/环境错误等）
  130 = 人工中止（Ctrl+C / SIGTERM）

用法（一般由 run_subtask.sh 调用，不直接手敲）：
  python subtask_supervisor.py --task A1 --policy act --timeout 40 \
      --clean-exit-result timeout --status-file ... --log-file ... -- \
      bash run_act_A1.sh <ckpt> 40
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

EXIT_CODES = {
    "ok": 0,
    "operator_fail": 2,
    "timeout": 3,
    "error": 4,
    "aborted": 130,
}

# 这些退出码说明子进程是被我们主动打断后正常收尾的，不是崩溃
_BENIGN_CHILD_RCS = (0, 130, -2, -15)


def _start_key_listener(events: queue.Queue):
    """用 lerobot 自带的键盘后端监听 0/1；没有 TTY/显示器时返回 None。"""
    try:
        from lerobot.utils.keyboard_input import create_key_listener
    except Exception as exc:  # noqa: BLE001 - 缺依赖时降级为无键盘
        print(f"[supervisor] 键盘监听不可用：{exc}", file=sys.stderr)
        return None

    def dispatch(name: str) -> None:
        if name in ("0", "1"):
            events.put(name)

    try:
        return create_key_listener(dispatch, controls_help="0=成功 1=失败")
    except Exception as exc:  # noqa: BLE001 - 任何后端异常都降级
        print(f"[supervisor] 键盘监听启动失败：{exc}", file=sys.stderr)
        return None


def _pump(stream, log_fh, lock: threading.Lock) -> None:
    for line in stream:
        with lock:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_fh is not None:
                log_fh.write(line)
                log_fh.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="coffee 单子任务监督器")
    parser.add_argument("--task", required=True, help="子任务 ID：A1/B1/B2/C1/C2/C3")
    parser.add_argument("--policy", required=True, help="执行方式：act/replay/pi05")
    parser.add_argument("--timeout", type=float, default=40.0, help="单个子任务最长执行秒数")
    parser.add_argument(
        "--enforce-timeout",
        action="store_true",
        help="由本进程计时并在超时后发 SIGINT（replay 用；ACT/pi0.5 由 rollout --duration 负责）",
    )
    parser.add_argument("--hard-timeout", type=float, default=0.0, help="兜底硬超时（秒），0=关闭")
    parser.add_argument("--grace", type=float, default=5.0, help="计时起点宽限（replay 启动/读元数据）")
    parser.add_argument("--home-wait", type=float, default=30.0, help="发停止信号后等待归零的最长秒数")
    parser.add_argument(
        "--clean-exit-result",
        choices=("ok", "timeout"),
        default="timeout",
        help="没有按键、子进程正常退出时的判定：ACT/pi0.5 用 timeout，replay 用 ok",
    )
    parser.add_argument("--status-file", default="", help="最后一次结果的 JSON 落盘路径，空=不写")
    parser.add_argument("--log-file", default="", help="子进程输出与服务行一起追加到这个日志")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="-- 后面的后端命令")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("[supervisor] 缺少要执行的后端命令", file=sys.stderr)
        return EXIT_CODES["error"]

    log_fh = open(args.log_file, "a", encoding="utf-8") if args.log_file else None
    log_lock = threading.Lock()

    def log(msg: str) -> None:
        with log_lock:
            print(msg, flush=True)
            if log_fh is not None:
                log_fh.write(msg + "\n")
                log_fh.flush()

    log("=" * 58)
    log(f"[SUBTASK] {args.task} 开始（policy={args.policy}, timeout={args.timeout:.0f}s）")
    log("           键 0 = 判定成功提前结束 | 键 1 = 判定失败 | Ctrl+C = 中止")
    log("           以上三种都会先慢速回到零位再退出")
    log("=" * 58)

    keys: queue.Queue = queue.Queue()
    listener = _start_key_listener(keys)
    if listener is None:
        log("[supervisor] ⚠️ 键盘不可用：只能等超时或 Ctrl+C")

    signal_state = {"name": None}

    def on_signal(signum, _frame):
        if signal_state["name"] is None:
            signal_state["name"] = signal.Signals(signum).name

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001 - 起不来就是环境错误
        log(f"[supervisor] ❌ 无法启动后端：{exc}")
        if log_fh is not None:
            log_fh.close()
        return EXIT_CODES["error"]

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = proc.pid

    reader = threading.Thread(target=_pump, args=(proc.stdout, log_fh, log_lock), daemon=True)
    reader.start()

    trigger: str | None = None
    stop_deadline: float | None = None
    escalated = False
    soft_deadline = started + args.timeout + args.grace if args.enforce_timeout else None
    hard_deadline = started + args.hard_timeout if args.hard_timeout > 0 else None

    def request_stop(reason: str, label: str) -> None:
        nonlocal trigger, stop_deadline
        if trigger is not None:
            return
        trigger = reason
        log(f"[supervisor] {label} → 发送 SIGINT，等待后端慢速归零...")
        try:
            os.killpg(pgid, signal.SIGINT)
        except ProcessLookupError:
            pass
        stop_deadline = time.monotonic() + args.home_wait

    while proc.poll() is None:
        now = time.monotonic()

        try:
            key = keys.get_nowait()
        except queue.Empty:
            key = None
        if key == "0":
            request_stop("key_0", "按键 0：判定成功")
        elif key == "1":
            request_stop("key_1", "按键 1：判定失败")

        if trigger is None and signal_state["name"]:
            request_stop("signal", f"收到 {signal_state['name']}：中止")

        if trigger is None and soft_deadline is not None and now >= soft_deadline:
            request_stop("timeout", f"达到 {args.timeout:.0f}s 执行上限")

        if trigger is None and hard_deadline is not None and now >= hard_deadline:
            request_stop("timeout", f"达到 {args.hard_timeout:.0f}s 硬超时")

        if trigger is not None and stop_deadline is not None and now >= stop_deadline:
            if not escalated:
                log("[supervisor] ⚠️ 归零等待超时 → SIGTERM")
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                escalated = True
                stop_deadline = now + 3.0
            else:
                log("[supervisor] ⚠️ 仍未退出 → SIGKILL")
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stop_deadline = None

        time.sleep(0.1)

    child_rc = proc.returncode
    reader.join(timeout=2.0)
    elapsed = time.monotonic() - started

    if escalated:
        # 归零等待都超时了，说明机械臂不一定回到零位，不能对外报成功
        result = "error"
    elif trigger is not None and child_rc not in _BENIGN_CHILD_RCS:
        result = "error"
    elif trigger == "key_0":
        result = "ok"
    elif trigger == "key_1":
        result = "operator_fail"
    elif trigger == "timeout":
        result = "timeout"
    elif trigger == "signal":
        result = "aborted"
    elif child_rc != 0:
        result = "error"
    else:
        result = args.clean_exit_result

    exit_code = EXIT_CODES[result]
    finished_at = datetime.now().isoformat(timespec="seconds")
    log(
        f"[SUBTASK] {args.task} RESULT={result} RC={exit_code} "
        f"CHILD_RC={child_rc} POLICY={args.policy} "
        f"ELAPSED={elapsed:.1f}s TRIGGER={trigger or 'natural'}"
    )

    if args.status_file:
        payload = {
            "task": args.task,
            "policy": args.policy,
            "result": result,
            "exit_code": exit_code,
            "child_rc": child_rc,
            "trigger": trigger or "natural",
            "elapsed_s": round(elapsed, 2),
            "timeout_s": args.timeout,
            "finished_at": finished_at,
        }
        try:
            parent = os.path.dirname(args.status_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(args.status_file, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
        except OSError as exc:
            log(f"[supervisor] ⚠️ 状态文件写入失败：{exc}")

    if listener is not None:
        try:
            listener.stop()
        except Exception:  # noqa: BLE001 - 退出阶段不再报错
            pass
    if log_fh is not None:
        log_fh.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
