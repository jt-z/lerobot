#!/usr/bin/env python
"""探测 coffee_agent 的健康状态（unix socket 的 health 指令）。

只依赖标准库，几十毫秒返回，可被 shell / 语音程序直接调用：

    python agent_health.py                     # 默认 socket，打印一行状态
    python agent_health.py --socket /path.sock
    python agent_health.py --quiet             # 只给退出码
    python agent_health.py --json              # 原始 JSON
    python agent_health.py --wait 15           # 最多等 15 秒直到健康（systemd/开机自检）

退出码：
    0 = agent 在线且健康（busy 也算健康，只表示正有子任务在跑）
    1 = socket 不存在 / 连不上 / health 返回不健康
    2 = 参数错误

输出示例：
    ok busy=false loaded=0 pid=12345 uptime=1.2s socket=/home/jt/.../coffee_agent.sock
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

DEFAULT_SOCKET = str(Path.home() / "dev/lerobot/self_scripts/_logs/coffee_agent.sock")


def probe(path: str, timeout: float = 1.0) -> dict:
    """发一条 health，返回响应；任何失败都抛 OSError/ValueError。"""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(path)
        sock.sendall(b'{"op":"health"}\n')
        buf = b""
        deadline = time.monotonic() + timeout
        while b"\n" not in buf:
            if time.monotonic() >= deadline:
                raise OSError("health 响应超时")
            chunk = sock.recv(4096)
            if not chunk:
                raise OSError("agent 提前关闭连接")
            buf += chunk
    finally:
        sock.close()
    msg = json.loads(buf.split(b"\n", 1)[0])
    if not isinstance(msg, dict) or msg.get("event") != "health":
        raise ValueError(f"非 health 响应：{msg!r}")
    return msg


def format_line(msg: dict, path: str) -> str:
    return (
        f"ok={str(bool(msg.get('ok'))).lower()} "
        f"busy={str(bool(msg.get('busy'))).lower()} "
        f"loaded={len(msg.get('loaded') or [])} "
        f"pid={msg.get('pid', '?')} "
        f"uptime={msg.get('uptime_s', '?')}s "
        f"socket={path}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="coffee_agent 健康探测")
    parser.add_argument(
        "--socket",
        default=os.environ.get("B601_AGENT_SOCKET", DEFAULT_SOCKET),
        help="agent 的 unix socket 路径",
    )
    parser.add_argument("--timeout", type=float, default=1.0, help="单次探测超时（秒）")
    parser.add_argument("--wait", type=float, default=0.0, help="最多等待多少秒直到健康（0=只探一次）")
    parser.add_argument("--interval", type=float, default=0.3, help="--wait 时的重试间隔（秒）")
    parser.add_argument("--quiet", action="store_true", help="不打印，只用退出码")
    parser.add_argument("--json", action="store_true", help="打印原始 JSON")
    args = parser.parse_args(argv)

    path = os.path.expanduser(args.socket)
    deadline = time.monotonic() + max(args.wait, 0.0)
    last_error = ""

    while True:
        try:
            msg = probe(path, args.timeout)
            if msg.get("ok") is True:
                if not args.quiet:
                    if args.json:
                        print(json.dumps(msg, ensure_ascii=False))
                    else:
                        print(format_line(msg, path))
                return 0
            last_error = f"health 不健康：{msg}"
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if time.monotonic() >= deadline:
            if not args.quiet:
                print(f"unavailable {last_error}")
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
