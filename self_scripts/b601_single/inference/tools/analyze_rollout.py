#!/usr/bin/env python3
"""分析 rollout CSV：跟随质量 / 块边界抖动 / 原地磨判定。

用法：
  python tools/analyze_rollout.py /home/kf/b601_pi05/logs/rollout_pi05_ws_XXXX.csv

注意：CSV 里 cmd_* 是数据集原有约定（action ≈ −state），比较时必须取负号。
"""
from __future__ import annotations

import csv
import sys

import numpy as np

J = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper']


def load(p):
    rows = list(csv.DictReader(open(p, newline='')))
    cmd = np.array([[float(r['cmd_' + j]) for j in J] for r in rows], dtype=np.float64)
    st = np.array([[float(r['state_' + j]) for j in J] for r in rows], dtype=np.float64)
    si = np.array([int(r['step_in_chunk']) for r in rows])
    ch = np.array([int(r['chunk_id']) for r in rows])
    late = np.array([int(r['late']) for r in rows])
    t = np.array([float(r['t']) for r in rows])
    return cmd, st, si, ch, late, t


def main():
    p = sys.argv[1]
    cmd, st, si, ch, late, t = load(p)
    n_chunk = int(ch.max()) + 1
    print(f'文件: {p}')
    print(f'步数={len(cmd)} 时长={t[-1]:.1f}s 块数={n_chunk} late={late.sum()} ({100*late.mean():.1f}%)')

    err = np.abs(-cmd[:, :6] - st[:, :6])
    print(f'\n[跟随] |−cmd−state| 中位={np.median(err):.2f}°  p90={np.percentile(err,90):.2f}° （>5° 说明真的跟不上）')

    print('\n[原地磨] 净位移 vs 绝对行程')
    for i, j in enumerate(J):
        net = abs(st[-1, i] - st[0, i])
        trav = np.abs(np.diff(st[:, i])).sum()
        print(f'  {j:14s} 净={net:7.1f}°  行程={trav:8.1f}°  摆动比={trav/max(net,0.1):6.1f}')

    d = np.abs(np.diff(cmd, axis=0)).mean(1)
    b = si[1:] == 0
    w = ~b
    print(f'\n[块边界] 块内 |Δcmd|={d[w].mean():.3f}°  边界={d[b].mean():.3f}°  '
          f'倍数={d[b].mean()/max(d[w].mean(),1e-9):.1f}x')
    step_dir = np.diff(cmd, axis=0)
    opp = same = 0
    for i in np.where(b)[0]:
        if i < 1:
            continue
        if float(np.dot(step_dir[i - 1], step_dir[i])) < 0:
            opp += 1
        else:
            same += 1
    print(f'  边界方向反转率={100*opp/max(opp+same,1):.0f}%  （>50% = 每次重规划都在改主意）')
    print(f'\n判定：'
          f'{"跟随正常" if np.median(err) < 5 else "跟随异常"} · '
          f'{"边界抖动显著" if d[b].mean() > 2*d[w].mean() else "边界抖动不明显"} · '
          f'{"多为原地磨" if np.abs(st[-1]-st[0]).mean() < 5 else "有真实推进"}')


if __name__ == '__main__':
    main()
