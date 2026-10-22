#!/usr/bin/env python3
"""把一份观测（npz 或数据集锚点）发给 policy server，打印 50 步动作块的结构。

用法：
  # 当前场景
  python query_chunk.py --obs /home/kf/b601_pi05/logs/live_obs.npz
  # 数据集里的某集（对照，证明 server 本身正常）
  python query_chunk.py --anchors /home/kf/b601_pi05/anchors_6.npz --index 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "openpi"))
from ws_client import CAMS, Pi05WSClient  # noqa: E402

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper']
TASK = ("Pick up the paper cup, place it on the silver tray of the coffee machine, pick up the cube, "
        "press the button with the cube (red light on), wait about 4 seconds, release the button "
        "(red light off), put the cube on the table first, then move the cup from the coffee machine "
        "to the table")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obs', type=Path, default=None, help='capture_obs.py 存的 npz')
    ap.add_argument('--anchors', type=Path, default=None, help='数据集锚点 npz')
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--repeats', type=int, default=3, help='同一帧问多次，看采样方差')
    a = ap.parse_args()

    if a.obs:
        z = np.load(a.obs)
        state = z['state'].astype(np.float32)
        images = {c: z[f'img_{c}'] for c in CAMS}
        tag = f'live({a.obs.name})'
    elif a.anchors:
        z = np.load(a.anchors)
        i = a.index
        state = z['states'][i].astype(np.float32)
        images = {c: np.transpose(z[f'img_{c}'][i], (1, 2, 0)) for c in CAMS}  # CHW float -> HWC float
        tag = f'dataset ep{int(z["episodes"][i])}'
    else:
        ap.error('需要 --obs 或 --anchors')

    cli = Pi05WSClient(a.host, a.port)
    print(f'[观测来源] {tag}')
    print('state:', np.round(state, 1).tolist())
    for r in range(a.repeats):
        chunk = cli.infer(state, images, TASK)
        print(f'\n--- 第 {r+1} 次采样，块 {chunk.shape} ---')
        print('  关节            首步    末步    块内 min   max   行程   |Δ|均值')
        for i, j in enumerate(JOINTS):
            col = chunk[:, i]
            print(f'  {j:14s} {col[0]:7.1f} {col[-1]:7.1f} {col.min():9.1f} {col.max():6.1f} '
                  f'{col.max()-col.min():6.1f}  {np.abs(np.diff(col)).mean():6.2f}')
        print(f'  平均延迟 {cli.avg_latency_ms:.0f} ms')
    print('\n判读：若"行程"普遍 <2° 且首步≈-state，说明模型在输出"保持不动"；'
          '若行程几十度，说明模型有意图，问题在下发/执行侧。')


if __name__ == '__main__':
    main()
