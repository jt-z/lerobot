#!/usr/bin/env python
"""离线验证常驻 agent：协议 + supervisor 结果映射。不碰机械臂/相机/权重。

由 test/run_offline_checks.sh 调用，也可以单独跑：
    python test/check_agent_offline.py
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AGENT = ROOT / "coffee_agent.py"
SUPERVISOR = ROOT / "subtask_supervisor.py"
SOCK = "/tmp/coffee_agent_offline.sock"
AGENT_LOG = "/tmp/coffee_agent_offline.log"
STATUS = "/tmp/coffee_agent_offline_status.json"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


class Client:
    def __init__(self, path: str) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect(path)
        self.buf = b""

    def send(self, obj: dict) -> None:
        self.sock.sendall((json.dumps(obj) + "\n").encode())

    def recv_event(self, want: str, timeout: float = 8.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if b"\n" in self.buf:
                raw, self.buf = self.buf.split(b"\n", 1)
                msg = json.loads(raw)
                if msg.get("event") == want:
                    return msg
            self.sock.settimeout(max(0.05, deadline - time.time()))
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            self.buf += chunk
        raise AssertionError(f"等不到事件 {want}")

    def close(self) -> None:
        self.sock.close()


def wait_socket(path: str, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.05)
    return False


def check_protocol(cli: Client) -> None:
    cli.send({"op": "health"})
    health = cli.recv_event("health")
    check("health ok", health.get("ok") is True, str(health))
    check("health self_test", health.get("self_test") is True)

    cli.send({"op": "run", "task": "A1", "policy": "act", "timeout": 3, "prompt": "p", "ckpt": "/tmp/fake"})
    check("accepted", cli.recv_event("accepted").get("task") == "A1")
    time.sleep(0.15)
    cli.send({"op": "stop", "reason": "ok"})
    check("stop_ack", cli.recv_event("stop_ack").get("accepted") is True)
    done = cli.recv_event("done")
    check("done ok/0", done.get("result") == "ok" and done.get("exit_code") == 0, str(done))

    cli.send({"op": "run", "task": "B1", "policy": "pi05", "timeout": 3, "prompt": "p", "ckpt": "/tmp/fake"})
    cli.recv_event("accepted")
    time.sleep(0.1)
    cli.send({"op": "stop", "reason": "operator_fail"})
    cli.recv_event("stop_ack")
    done = cli.recv_event("done")
    check(
        "done operator_fail/2",
        done.get("result") == "operator_fail" and done.get("exit_code") == 2,
        str(done),
    )

    cli.send({"op": "run", "task": "C2", "policy": "act", "timeout": 0.4, "prompt": "p", "ckpt": "/tmp/fake"})
    cli.recv_event("accepted")
    done = cli.recv_event("done")
    check("done timeout/3", done.get("result") == "timeout" and done.get("exit_code") == 3, str(done))

    # replay：自然放完 = ok（与子进程 replay 的 clean-exit-result=ok 一致）
    cli.send({"op": "run", "task": "C1", "policy": "replay", "timeout": 0.3,
              "dataset": "/tmp/fake_ds", "episode": 3})
    cli.recv_event("accepted")
    done = cli.recv_event("done")
    check(
        "replay 自然结束 -> ok/0",
        done.get("result") == "ok" and done.get("exit_code") == 0,
        str(done),
    )

    cli.send({"op": "run", "task": "A1", "policy": "act", "timeout": 2, "prompt": "p", "ckpt": "/tmp/fake"})
    cli.recv_event("accepted")
    cli.send({"op": "run", "task": "B1", "policy": "act", "timeout": 2, "prompt": "p", "ckpt": "/tmp/fake"})
    check("busy", cli.recv_event("error").get("error") == "busy")
    cli.send({"op": "stop", "reason": "ok"})
    cli.recv_event("stop_ack")
    check("busy 后收尾", cli.recv_event("done").get("result") == "ok")

    cli.send({"op": "run", "task": "A1", "policy": "act", "timeout": 0, "prompt": "p", "ckpt": "/tmp/fake"})
    check("bad_request timeout=0", cli.recv_event("error").get("error") == "bad_request")

    cli.send({"op": "stop", "reason": "nope"})
    check("bad stop reason", cli.recv_event("error").get("error") == "bad_request")


def check_health_helper() -> None:
    helper = ROOT / "agent_health.py"
    r = subprocess.run(
        [sys.executable, str(helper), "--socket", SOCK], capture_output=True, text=True, timeout=15
    )
    check("agent_health 健康 -> 0", r.returncode == 0, r.stdout.strip() or r.stderr.strip())
    check("agent_health 输出 ok=true", "ok=true" in r.stdout, r.stdout.strip())

    r = subprocess.run(
        [sys.executable, str(helper), "--socket", "/tmp/coffee_no_such_agent.sock", "--timeout", "0.3"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    check("agent_health 无 socket -> 1", r.returncode == 1, r.stdout.strip() or r.stderr.strip())


def run_supervisor(extra: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SUPERVISOR), "--status-file", STATUS, *extra],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def check_supervisor(cli_path: str) -> None:
    base = ["--agent-prompt", "p", "--agent-ckpt", "/tmp/fake"]

    r = run_supervisor(["--task", "A1", "--policy", "act", "--timeout", "0.4",
                        "--agent-socket", cli_path, *base])
    check("supervisor 超时 -> 3", r.returncode == 3, f"rc={r.returncode}")
    if os.path.exists(STATUS):
        status = json.load(open(STATUS))
        check("状态文件 mode=agent", status.get("mode") == "agent", str(status))
        check("状态文件 result=timeout", status.get("result") == "timeout", str(status))

    r = run_supervisor(["--task", "A1", "--policy", "act", "--timeout", "1",
                        "--agent-socket", "/tmp/coffee_no_such_agent.sock", *base])
    check("连不上 agent -> 4", r.returncode == 4, f"rc={r.returncode}")

    # SIGTERM → 130（等 supervisor 真正连上 agent 再发）
    proc = subprocess.Popen(
        [sys.executable, str(SUPERVISOR), "--task", "B1", "--policy", "pi05", "--timeout", "5",
         "--agent-socket", cli_path, *base],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 40
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line or "已接受" in line:
            break
    proc.send_signal(signal.SIGTERM)
    check("SIGTERM -> 130", proc.wait(timeout=30) == 130)

    # 键 0/1：直接调用 _run_agent，预置按键队列（不依赖真实键盘）
    sys.path.insert(0, str(ROOT))
    import subtask_supervisor as sup  # noqa: PLC0415

    def direct(keys: list[str]) -> dict | None:
        args = argparse.Namespace(
            task="C2", policy="act", timeout=5.0, grace=10.0, hard_timeout=0.0, home_wait=30.0,
            agent_socket=cli_path, agent_prompt="p", agent_ckpt="/tmp/fake",
            agent_dataset="", agent_episode=0,
        )
        kq: queue.Queue = queue.Queue()
        for key in keys:
            kq.put(key)
        return sup._run_agent(args, lambda m: None, None, threading.Lock(), kq, {"name": None})

    out = direct(["0"])
    check("键 0 -> ok", bool(out) and out["result"] == "ok", str(out))
    out = direct(["1"])
    check("键 1 -> operator_fail", bool(out) and out["result"] == "operator_fail", str(out))


def main() -> int:
    for path in (SOCK, AGENT_LOG, STATUS):
        if os.path.exists(path):
            os.unlink(path)

    agent = subprocess.Popen(
        [sys.executable, str(AGENT), "--socket", SOCK, "--self-test",
         "--display-data", "0", "--log-file", AGENT_LOG],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        check("agent socket 启动", wait_socket(SOCK))
        cli = Client(SOCK)
        check_health_helper()
        check_protocol(cli)
        cli.send({"op": "shutdown"})
        cli.recv_event("bye")
        cli.close()
        check("agent 正常退出", agent.wait(timeout=15) == 0)
        check("socket 已清理", not os.path.exists(SOCK))

        # shutdown 后重新起一个给 supervisor 用
        agent2 = subprocess.Popen(
            [sys.executable, str(AGENT), "--socket", SOCK, "--self-test",
             "--display-data", "0", "--log-file", AGENT_LOG],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            check("agent socket 重启", wait_socket(SOCK))
            check_supervisor(SOCK)
        finally:
            agent2.terminate()
            try:
                agent2.wait(timeout=15)
            except subprocess.TimeoutExpired:
                agent2.kill()
    finally:
        if agent.poll() is None:
            agent.terminate()
            try:
                agent.wait(timeout=15)
            except subprocess.TimeoutExpired:
                agent.kill()

    if fails:
        print("\nagent 离线检查失败：" + ", ".join(fails))
        return 1
    print("\nAGENT OFFLINE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
