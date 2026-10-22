#!/usr/bin/env python3
"""b601 π0.5 websocket 客户端：本地观测 → 4090 上的 policy server → 50 步动作块。

跑在 lerobot 环境（py3.12，torch）里，policy 在 JAX 环境（4090）里，两边只通过 websocket 通信。

依赖：openpi-client（纯 python，已随 openpi 代码在 /home/kf/b601_pi05/openpi/packages/openpi-client/src）

自检（不需要机器人）：
  python ws_client.py --self-test --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import argparse
import time

import numpy as np

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper')
CAMS = ('front', 'hand', 'top')


def to_uint8_hwc(img: np.ndarray) -> np.ndarray:
    """统一成 uint8 HWC —— 训练侧的 _parse_image 就是这么吃的（float 会 ×255，CHW 会转置）。"""
    a = np.asarray(img)
    if np.issubdtype(a.dtype, np.floating):
        a = np.clip(a, 0.0, 1.0) * 255.0
        a = a.astype(np.uint8)
    else:
        a = a.astype(np.uint8, copy=False)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = np.transpose(a, (1, 2, 0))
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f'图像形状不对: {a.shape}')
    return np.ascontiguousarray(a)


class Pi05WSClient:
    def __init__(self, host: str = '127.0.0.1', port: int = 8000, timeout: float = 10.0):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self._c = WebsocketClientPolicy(host=host, port=port)
        self.metadata = self._c.get_server_metadata()
        self._timeout = timeout
        self.calls = 0
        self.total_latency = 0.0

    def infer(self, state_deg, images: dict, prompt: str) -> np.ndarray:
        """state_deg: (7,) 度；images: {front,hand,top: uint8 HWC}；返回 (50,7) 度。"""
        st = np.asarray(state_deg, dtype=np.float32).reshape(-1)
        if st.shape != (7,):
            raise ValueError(f'state 需要 7 维，收到 {st.shape}')
        obs = dict(
            state=st,
            images={c: to_uint8_hwc(images[c]) for c in CAMS},
            prompt=prompt,
        )
        t0 = time.perf_counter()
        out = self._c.infer(obs)
        dt = time.perf_counter() - t0
        self.calls += 1
        self.total_latency += dt
        a = np.asarray(out['actions'], dtype=np.float32)
        return a[:, :7]

    @property
    def avg_latency_ms(self) -> float:
        return 1000.0 * self.total_latency / max(1, self.calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--self-test', action='store_true', help='用假观测验证连通性，不接机器人')
    a = ap.parse_args()
    if not a.self_test:
        ap.error('目前只支持 --self-test；真机请用 run_rollout.py')

    cli = Pi05WSClient(a.host, a.port)
    print('server metadata:', cli.metadata)
    rng = np.random.default_rng(0)
    for i in range(3):
        acts = cli.infer(rng.normal(0, 30, 7),
                         {c: (rng.random((480, 640, 3)) * 255).astype(np.uint8) for c in CAMS},
                         'Pick up the paper cup')
        print(f'call{i}: chunk={acts.shape} first={np.round(acts[0], 2).tolist()}')
    print(f'平均延迟 {cli.avg_latency_ms:.1f} ms')


if __name__ == '__main__':
    main()
