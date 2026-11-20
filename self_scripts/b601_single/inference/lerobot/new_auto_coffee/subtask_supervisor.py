#!/usr/bin/env python
"""单子任务监督器：超时 / 按键 0、1 / 中止，统一归零等待与退出码。

它本身不做推理，只负责“看住”一个后端：

  A. 常驻 agent 模式（--agent-socket）：把 run 指令发给 coffee_agent.py，由 agent 在
     已升温的机械臂/相机/权重上执行；本进程只负责按键/信号/超时 → stop 指令。
  B. 子进程模式（-- 后面的命令）：拉起 ACT / pi0.5 / replay 后端进程，按原有逻辑看住它。

两种模式对外的契约完全一致：

  * 键 0：操作员判定本子任务成功 → 提前结束，等机械臂归零后以 0 退出；
  * 键 1：操作员判定失败       → 同样先归零，以 2 退出；
  * Ctrl+C / SIGTERM：中止     → 同样先归零，以 130 退出；
  * 超时：由后端（rollout --duration / agent 的 duration）到点退出，归零后以 3 退出；
    replay 由本监督器在 --timeout + --grace 后发 SIGINT/stop。
  * 后端异常：4。

用法：
  # 常驻 agent（run_subtask.sh 自动选择）
  python subtask_supervisor.py --task A1 --policy act --timeout 40 \
      --agent-socket /path/agent.sock --agent-prompt "..." --agent-ckpt /path/model

  # 子进程（原行为）
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
import socket
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

# 触发 → 结果（与 coffee_agent.py 的 TRIGGER_OF 对应）
_RESULT_OF_TRIGGER = {
    "key_0": "ok",
    "key_1": "operator_fail",
    "timeout": "timeout",
    "signal": "aborted",
}
_STOP_REASON_OF_TRIGGER = {
    "key_0": "ok",
    "key_1": "operator_fail",
    "timeout": "timeout",
    "signal": "aborted",
}


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


def _write_status(status_file: str, payload: dict, log) -> None:
    if not status_file:
        return
    try:
        parent = os.path.dirname(status_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(status_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    except OSError as exc:
        log(f"[supervisor] ⚠️ 状态文件写入失败：{exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="coffee 单子任务监督器")
    parser.add_argument("--task", required=True, help="子任务 ID：A1/B1/B2/C1/C2/C3")
    parser.add_argument("--policy", required=True, help="执行方式：act/replay/pi05")
    parser.add_argument("--timeout", type=float, default=40.0, help="单个子任务最长执行秒数")
    parser.add_argument(
        "--enforce-timeout",
        action="store_true",
        help="由本进程计时并在超时后发停止（replay 用；agent 模式下由 agent 的 duration 负责）",
    )
    parser.add_argument("--hard-timeout", type=float, default=0.0, help="兜底硬超时（秒），0=关闭")
    parser.add_argument("--grace", type=float, default=5.0, help="计时起点宽限（replay 启动/读元数据）")
    parser.add_argument("--home-wait", type=float, default=30.0, help="发停止信号后等待归零的最长秒数")
    parser.add_argument(
        "--clean-exit-result",
        choices=("ok", "timeout"),
        default="timeout",
        help="没有按键、后端正常退出时的判定：ACT/pi0.5 用 timeout，replay 用 ok",
    )
    parser.add_argument("--status-file", default="", help="最后一次结果的 JSON 落盘路径，空=不写")
    parser.add_argument("--log-file", default="", help="子进程输出与服务行一起追加到这个日志")
    # ---- 常驻 agent 模式 ----
    parser.add_argument("--agent-socket", default="", help="coffee_agent.py 的 unix socket；非空则走 agent")
    parser.add_argument("--agent-prompt", default="", help="交给 agent 的任务文本（与训练逐字一致）")
    parser.add_argument("--agent-ckpt", default="", help="交给 agent 的初始权重目录（act/pi05）")
    parser.add_argument("--agent-dataset", default="", help="replay 用：数据集根目录")
    parser.add_argument("--agent-episode", type=int, default=0, help="replay 用：episode 编号")
    parser.add_argument(
        "--control-socket",
        default=os.environ.get("B601_TASK_CONTROL_SOCKET", ""),
        help="q/w/e 接管用的控制 unix socket（空=不开放）",
    )
    parser.add_argument("--agent-act-ckpt", default="", help="热切到 ACT 时使用的权重")
    parser.add_argument("--agent-pi05-ckpt", default="", help="热切到 pi0.5 时使用的权重")
    parser.add_argument("--agent-replay-dataset", default="", help="热切到 replay 时使用的数据集")
    parser.add_argument("--agent-replay-episode", type=int, default=0, help="热切到 replay 时使用的 episode")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="-- 后面的后端命令（agent 模式可省略）")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 子进程模式（原逻辑，行为不变）
# ---------------------------------------------------------------------------


def _run_spawn(args, cmd, log, log_fh, log_lock, keys: queue.Queue, signal_state: dict) -> dict | None:
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
        return None

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
    elif trigger is not None:
        result = _RESULT_OF_TRIGGER[trigger]
    elif child_rc != 0:
        result = "error"
    else:
        result = args.clean_exit_result

    return {
        "mode": "spawn",
        "result": result,
        "trigger": trigger or "natural",
        "elapsed": elapsed,
        "child_rc": child_rc,
    }


# ---------------------------------------------------------------------------
# 常驻 agent 模式
# ---------------------------------------------------------------------------


def _agent_send(sock: socket.socket, payload: dict) -> None:
    sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))


class _ControlServer:
    """本地控制通道：语音程序按 q/w/e 时发 switch/zero，supervisor 再转发给 agent。"""

    def __init__(self, path: str, q: queue.Queue, log) -> None:
        self.path = path
        self.q = q
        self.log = log
        self._stop = threading.Event()
        self._srv: socket.socket | None = None

    def start(self) -> bool:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(self.path):
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(self.path)
            except OSError:
                try:
                    os.unlink(self.path)
                except OSError:
                    pass
            else:
                probe.close()
                self.log(f"[supervisor] ⚠️ 控制 socket 已被占用：{self.path}")
                return False
            finally:
                probe.close()
        try:
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(self.path)
            os.chmod(self.path, 0o600)
            srv.listen(4)
            srv.settimeout(0.5)
        except OSError as exc:
            self.log(f"[supervisor] ⚠️ 控制通道启动失败：{exc}")
            return False
        self._srv = srv
        threading.Thread(target=self._accept_loop, name="ctl", daemon=True).start()
        self.log(f"[supervisor] 控制通道：{self.path}")
        return True

    def _accept_loop(self) -> None:
        assert self._srv is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn: socket.socket) -> None:
        wlock = threading.Lock()
        conn.settimeout(0.5)
        buf = b""
        try:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    try:
                        req = json.loads(line)
                        if not isinstance(req, dict):
                            raise ValueError("请求必须是 JSON 对象")
                        op = str(req.get("op", "")).strip()
                    except Exception as exc:  # noqa: BLE001
                        self.send(conn, wlock, {"event": "error", "error": "bad_request", "message": str(exc)})
                        continue
                    if op not in ("switch", "zero"):
                        self.send(
                            conn,
                            wlock,
                            {"event": "error", "error": "bad_request", "message": f"未知 op：{op!r}"},
                        )
                        continue
                    self.q.put((op, req, conn, wlock))
        finally:
            try:
                conn.close()
            except OSError:
                pass

    @staticmethod
    def send(conn: socket.socket, wlock: threading.Lock, payload: dict) -> None:
        try:
            data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            with wlock:
                conn.sendall(data)
        except OSError:
            pass

    def stop(self) -> None:
        self._stop.set()
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
        try:
            if os.path.exists(self.path):
                os.unlink(self.path)
        except OSError:
            pass


def _ckpt_ready(path: str) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "config.json")) and os.path.isfile(
        os.path.join(path, "model.safetensors")
    )


def _build_switch_payload(args, policy: str) -> tuple[dict | None, str | None]:
    """按目标策略构造完整 switch 请求；权重/数据集不可用时不打断当前任务。"""
    base = {"op": "switch", "task": args.task, "policy": policy, "timeout": args.timeout}
    if policy == "act":
        ckpt = args.agent_act_ckpt or args.agent_ckpt
        if not _ckpt_ready(ckpt):
            return None, f"ACT 权重未就绪：{ckpt or '<空>'}"
        return {**base, "prompt": args.agent_prompt, "ckpt": ckpt}, None
    if policy == "pi05":
        ckpt = args.agent_pi05_ckpt or args.agent_ckpt
        if not _ckpt_ready(ckpt):
            return None, f"pi0.5 权重未就绪：{ckpt or '<空>'}"
        return {**base, "prompt": args.agent_prompt, "ckpt": ckpt}, None
    if policy == "replay":
        dataset = args.agent_replay_dataset
        if not dataset or not os.path.isfile(os.path.join(dataset, "meta/info.json")):
            return None, f"replay 数据集不可用：{dataset or '<空>'}"
        return {**base, "dataset": dataset, "episode": args.agent_replay_episode}, None
    return None, f"未知策略：{policy!r}（可用 act / pi05 / replay）"


def _run_agent(
    args, log, log_fh, log_lock, keys: queue.Queue, signal_state: dict,
    ctrl_q: queue.Queue | None = None, ctrl: "_ControlServer | None" = None,
) -> dict | None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        sock.connect(args.agent_socket)
    except OSError as exc:
        log(f"[supervisor] ❌ 连不上 agent：{args.agent_socket}（{exc}）")
        log("           先启动常驻服务：bash run_coffee_agent.sh（或用 --no-agent 回退子进程）")
        sock.close()
        return None
    sock.settimeout(0.1)

    started = time.monotonic()
    try:
        _agent_send(
            sock,
            {
                "op": "run",
                "task": args.task,
                "policy": args.policy,
                "timeout": args.timeout,
                "prompt": args.agent_prompt,
                "ckpt": args.agent_ckpt,
                "dataset": args.agent_dataset,
                "episode": args.agent_episode,
            },
        )
    except OSError as exc:
        log(f"[supervisor] ❌ 发送 run 失败：{exc}")
        sock.close()
        return None

    trigger: str | None = None
    stop_deadline: float | None = None
    escalated = False
    done: dict | None = None
    soft_deadline = started + args.timeout + args.grace
    hard_deadline = started + args.hard_timeout if args.hard_timeout > 0 else None
    buf = b""
    expected_run_id: int | None = None
    handover = False
    pending_ctrl = None
    pending_zero = None
    prev_run_id: int | None = None
    prev_policy = args.policy

    def ctrl_send(pair, payload: dict) -> None:
        if pair is not None and ctrl is not None:
            ctrl.send(pair[0], pair[1], payload)

    def reset_deadlines(now: float) -> None:
        nonlocal started, soft_deadline, hard_deadline
        started = now
        soft_deadline = now + args.timeout + args.grace
        hard_deadline = now + args.hard_timeout if args.hard_timeout > 0 else None

    def request_stop(reason: str, label: str) -> None:
        nonlocal trigger, stop_deadline
        if trigger is not None:
            return
        trigger = reason
        stop_reason = _STOP_REASON_OF_TRIGGER[reason]
        log(f"[supervisor] {label} → 通知 agent stop({stop_reason})，等待慢速归零...")
        try:
            _agent_send(sock, {"op": "stop", "reason": stop_reason})
        except OSError:
            pass
        stop_deadline = time.monotonic() + args.home_wait

    try:
        while True:
            now = time.monotonic()

            # ---- q/w/e 控制请求：转发给 agent，并重置本段的计时 ----
            if ctrl_q is not None:
                while True:
                    try:
                        op, req, cconn, cwlock = ctrl_q.get_nowait()
                    except queue.Empty:
                        break
                    if handover:
                        ctrl.send(
                            cconn,
                            cwlock,
                            {"event": "error", "error": "busy", "message": "上一次切换还在进行"},
                        )
                        continue
                    if op == "zero":
                        try:
                            _agent_send(sock, {"op": "zero"})
                        except OSError as exc:
                            ctrl.send(
                                cconn, cwlock,
                                {"event": "error", "error": "send_failed", "message": str(exc)},
                            )
                            continue
                        pending_zero = (cconn, cwlock)
                        ctrl.send(cconn, cwlock, {"event": "zeroing"})
                        continue

                    policy = str(req.get("policy", "")).strip().lower()
                    payload, err = _build_switch_payload(args, policy)
                    if err:
                        log(f"[supervisor] ⚠️ 接管请求被拒绝：{err}")
                        ctrl.send(
                            cconn, cwlock,
                            {"event": "error", "error": "not_ready", "message": err},
                        )
                        continue
                    try:
                        _agent_send(sock, payload)
                    except OSError as exc:
                        ctrl.send(
                            cconn, cwlock,
                            {"event": "error", "error": "send_failed", "message": str(exc)},
                        )
                        continue

                    log(f"[supervisor] 收到接管请求 → {policy}（停止当前段并归零后启动）")
                    ctrl.send(cconn, cwlock, {"event": "switching", "policy": policy})
                    pending_ctrl = (cconn, cwlock)
                    prev_run_id = expected_run_id
                    prev_policy = args.policy
                    expected_run_id = None
                    handover = True
                    args.policy = policy
                    trigger = None
                    stop_deadline = None
                    escalated = False
                    done = None
                    reset_deadlines(now)

            if done is None:
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    chunk = None
                except OSError as exc:
                    log(f"[supervisor] ⚠️ agent 连接中断：{exc}")
                    break
                if chunk == b"":
                    log("[supervisor] ⚠️ agent 关闭了连接")
                    break
                if chunk:
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        line = raw.decode("utf-8", "replace").strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            log(f"[supervisor] ⚠️ 无法解析 agent 消息：{line[:200]}")
                            continue
                        event = msg.get("event")
                        if event == "log":
                            log(msg.get("line", ""))
                        elif event == "accepted":
                            expected_run_id = msg.get("run_id")
                            log(
                                f"[supervisor] agent 已接受：{msg.get('task')}"
                                f"（{msg.get('policy')}, run_id={expected_run_id}）"
                            )
                            reset_deadlines(time.monotonic())
                            if handover:
                                handover = False
                                ctrl_send(
                                    pending_ctrl,
                                    {
                                        "event": "accepted",
                                        "policy": msg.get("policy"),
                                        "run_id": expected_run_id,
                                    },
                                )
                                pending_ctrl = None
                                if trigger is not None:
                                    # 切换期间操作员已按 0/1/Ctrl+C：把停止补发给新会话
                                    try:
                                        _agent_send(
                                            sock,
                                            {
                                                "op": "stop",
                                                "reason": _STOP_REASON_OF_TRIGGER[trigger],
                                            },
                                        )
                                    except OSError:
                                        pass
                        elif event == "stop_ack":
                            log(
                                f"[supervisor] agent {'已受理' if msg.get('accepted') else '未受理'}"
                                f" stop({msg.get('reason')})"
                            )
                        elif event == "zeroed":
                            log("[supervisor] agent 已归零")
                            ctrl_send(
                                pending_zero,
                                {"event": "zeroed", "already": msg.get("already")},
                            )
                            pending_zero = None
                            done = {
                                "result": "ok",
                                "exit_code": 0,
                                "trigger": "zero",
                                "task": args.task,
                                "policy": args.policy,
                            }
                        elif event == "error":
                            log(f"[supervisor] ❌ agent 错误：{msg.get('error')} {msg.get('message', '')}")
                            if handover:
                                # switch 校验失败：agent 没动当前会话，保留任务继续跑
                                handover = False
                                expected_run_id = prev_run_id
                                args.policy = prev_policy
                                ctrl_send(
                                    pending_ctrl,
                                    {
                                        "event": "error",
                                        "error": msg.get("error", "error"),
                                        "message": msg.get("message", ""),
                                    },
                                )
                                pending_ctrl = None
                                reset_deadlines(time.monotonic())
                            elif msg.get("error") == "busy":
                                done = {"result": "error", "exit_code": EXIT_CODES["error"]}
                        elif event == "done":
                            if handover:
                                log("[supervisor] 忽略切换前的旧 done")
                            elif expected_run_id is not None and msg.get("run_id") != expected_run_id:
                                log(
                                    f"[supervisor] 忽略旧会话 done（run_id={msg.get('run_id')}，"
                                    f"当前={expected_run_id}）"
                                )
                            else:
                                done = msg
                                ctrl_send(
                                    pending_ctrl,
                                    {"event": "done", "result": msg.get("result"),
                                     "run_id": msg.get("run_id")},
                                )
                                pending_ctrl = None
                        elif event == "bye":
                            log("[supervisor] agent 已退出")

            if done is not None:
                break

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

            if trigger is None and now >= soft_deadline:
                request_stop("timeout", f"达到 {args.timeout:.0f}s 执行上限")

            if trigger is None and hard_deadline is not None and now >= hard_deadline:
                request_stop("timeout", f"达到 {args.hard_timeout:.0f}s 硬超时")

            if trigger is not None and stop_deadline is not None and now >= stop_deadline:
                if not escalated:
                    log("[supervisor] ⚠️ agent 归零等待超时 → 停止等待（agent 会继续收尾）")
                    escalated = True
                    break

            time.sleep(0.1)
    finally:
        sock.close()

    elapsed = time.monotonic() - started
    # 结果判定：agent 明确报 error 优先；否则按键/信号/超时优先于 agent 的自然结束
    if escalated:
        result = "error"
    elif done is not None and done.get("result") == "error":
        result = "error"
    elif trigger is not None:
        result = _RESULT_OF_TRIGGER[trigger]
    elif done is not None and done.get("result") in EXIT_CODES:
        result = done["result"]
    else:
        result = "error"

    return {
        "mode": "agent",
        "result": result,
        "trigger": trigger or (done or {}).get("trigger", "natural"),
        "elapsed": elapsed,
        "child_rc": (done or {}).get("exit_code"),
    }


# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not args.agent_socket and not cmd:
        print("[supervisor] 缺少要执行的后端命令（或 --agent-socket）", file=sys.stderr)
        return EXIT_CODES["error"]

    log_fh = open(args.log_file, "a", encoding="utf-8") if args.log_file else None
    log_lock = threading.Lock()

    def log(msg: str) -> None:
        with log_lock:
            print(msg, flush=True)
            if log_fh is not None:
                log_fh.write(msg + "\n")
                log_fh.flush()

    mode = "agent" if args.agent_socket else "spawn"
    log("=" * 58)
    log(f"[SUBTASK] {args.task} 开始（policy={args.policy}, timeout={args.timeout:g}s, mode={mode}）")
    log("           键 0 = 判定成功提前结束 | 键 1 = 判定失败 | Ctrl+C = 中止")
    log("           以上三种都会先慢速回到零位再退出")
    if args.agent_socket:
        log("           语音侧 q=pi0.5 接管 | w=ACT 接管 | e=归零后 replay")
    log("=" * 58)

    keys: queue.Queue = queue.Queue()
    ctrl_q: queue.Queue = queue.Queue()
    ctrl = None
    if args.agent_socket and args.control_socket:
        ctrl = _ControlServer(args.control_socket, ctrl_q, log)
        ctrl.start()
    elif args.control_socket:
        log("[supervisor] ⚠️ 子进程模式不支持 q/w/e 接管")
    listener = _start_key_listener(keys)
    if listener is None:
        log("[supervisor] ⚠️ 键盘不可用：只能等超时或 Ctrl+C")

    signal_state = {"name": None}

    def on_signal(signum, _frame):
        if signal_state["name"] is None:
            signal_state["name"] = signal.Signals(signum).name

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        if args.agent_socket:
            outcome = _run_agent(
                args, log, log_fh, log_lock, keys, signal_state, ctrl_q, ctrl
            )
        else:
            outcome = _run_spawn(args, cmd, log, log_fh, log_lock, keys, signal_state)
    finally:
        if ctrl is not None:
            ctrl.stop()

    if outcome is None:
        if listener is not None:
            try:
                listener.stop()
            except Exception:  # noqa: BLE001
                pass
        if log_fh is not None:
            log_fh.close()
        return EXIT_CODES["error"]

    result = outcome["result"]
    exit_code = EXIT_CODES[result]
    log(
        f"[SUBTASK] {args.task} RESULT={result} RC={exit_code} "
        f"CHILD_RC={outcome['child_rc']} POLICY={args.policy} "
        f"ELAPSED={outcome['elapsed']:.1f}s TRIGGER={outcome['trigger']}"
    )

    payload = {
        "task": args.task,
        "policy": args.policy,
        "result": result,
        "exit_code": exit_code,
        "child_rc": outcome["child_rc"],
        "trigger": outcome["trigger"],
        "elapsed_s": round(outcome["elapsed"], 2),
        "timeout_s": args.timeout,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "mode": outcome["mode"],
    }
    _write_status(args.status_file, payload, log)

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
