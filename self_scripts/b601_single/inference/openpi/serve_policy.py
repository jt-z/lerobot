#!/usr/bin/env python3
"""在 z790 的 4090 上把 b601 π0.5 起成 openpi websocket policy server。

机器人侧（或采集脚本）用 openpi_client.websocket_client_policy.WebsocketClientPolicy 连，
不需要把 JAX 权重转成 lerobot 格式。

用法：
  source env.sh
  $PY serve_policy.py --model 330 --port 8000
  # 客户端：
  #   python -c "from openpi_client.websocket_client_policy import WebsocketClientPolicy as P; \
  #              p=P('127.0.0.1',8000); print(p.infer({'state':..., 'images':{...}, 'prompt':'...'}))"

注意：
  - 4090 24G，默认 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6（约 14G），够 3 相机单样本推理
  - 协议同 A800 训练时的 policy 输入：state(7) + images{front,hand,top} + prompt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["330", "462"], default="330")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from b601_config import register

    base = Path(os.environ["B601_HOME"])
    ckpt = a.ckpt or str(base / "checkpoints" / ("330_24653" if a.model == "330" else "462_34750"))
    cfg = register()
    logging.info("loading checkpoint %s", ckpt)
    policy = create_trained_policy(cfg, Path(ckpt), pytorch_device="cuda")
    metadata = dict(
        model=f"b601_pi05_full_{a.model}",
        checkpoint=ckpt,
        horizon=50,
        execute_steps=5,
        state_dim=7,
        action_dim=7,
        cameras=["front", "hand", "top"],
        action_note="action 使用数据集原有负号约定（action ≈ -state），单位：度",
        fps=30,
    )
    server = WebsocketPolicyServer(policy, host=a.host, port=a.port, metadata=metadata)
    logging.info("listening on %s:%s", a.host, a.port)
    print("READY", json.dumps(metadata, ensure_ascii=False), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
