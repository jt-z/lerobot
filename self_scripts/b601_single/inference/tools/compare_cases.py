#!/usr/bin/env python3
"""把 ab_cases.log 里各组的 rollout CSV 汇总成一张 2×2 表。

用法：
  python tools/compare_cases.py                 # 默认 /home/kf/b601_pi05/logs
  python tools/compare_cases.py <logs目录>

输出列：
  组 · 噪声 · exec · 步/块 · late% · 跟随误差中位 · 块内Δcmd · 边界Δcmd · 倍数 · 边界反向率
  · 各关节净位移(平均) · 摆动比(平均) · 结论
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

J = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper']


def stats(path):
    rows = list(csv.DictReader(open(path, newline='')))
    if not rows:
        return None
    cmd = np.array([[float(r['cmd_' + j]) for j in J] for r in rows])
    st = np.array([[float(r['state_' + j]) for j in J] for r in rows])
    si = np.array([int(r['step_in_chunk']) for r in rows])
    late = np.array([int(r.get('late', 0)) for r in rows])
    err = np.abs(-cmd[:, :6] - st[:, :6])
    d = np.abs(np.diff(cmd, axis=0)).mean(1)
    b = si[1:] == 0
    w = ~b
    sd = np.diff(cmd, axis=0)
    opp = same = 0
    for i in np.where(b)[0]:
        if i < 1:
            continue
        if float(np.dot(sd[i - 1], sd[i])) < 0:
            opp += 1
        else:
            same += 1
    net = np.abs(st[-1] - st[0])
    trav = np.abs(np.diff(st, axis=0)).sum(0)
    return dict(
        steps=len(rows), chunks=len(np.unique([r['chunk_id'] for r in rows])),
        late_pct=100 * late.mean(),
        err=float(np.median(err)),
        d_in=float(d[w].mean()) if w.any() else float('nan'),
        d_bd=float(d[b].mean()) if b.any() else float('nan'),
        rev=100 * opp / max(opp + same, 1),
        net=float(net.mean()), swing=float(np.mean(trav / np.maximum(net, 0.1))),
    )


def main():
    logs = sys.argv[1] if len(sys.argv) > 1 else '/home/kf/b601_pi05/logs'
    ledger = os.path.join(logs, 'ab_cases.log')
    if not os.path.exists(ledger):
        print(f'没有台账 {ledger}（用 ab_test.sh 跑的会自动写）')
        return
    hdr = (f'{"组":>2s} {"噪声":>5s} {"exec":>4s} {"步/块":>6s} {"late%":>6s} '
           f'{"跟随":>6s} {"块内Δ":>7s} {"边界Δ":>7s} {"倍数":>5s} {"反转%":>6s} '
           f'{"净位移均":>8s} {"摆动比":>7s}')
    print(hdr)
    print('-' * len(hdr))
    for line in open(ledger):
        parts = line.strip().split('|')
        if len(parts) < 7:
            continue
        _, case, noise, ex, seed, dur, path = parts[:7]
        if not os.path.exists(path):
            print(f'{case:>2s}  缺失 {path}')
            continue
        s = stats(path)
        if not s:
            continue
        print(f'{case:>2s} {noise:>5s} {ex:>4s} {s["steps"]/s["chunks"]:6.2f} {s["late_pct"]:6.1f} '
              f'{s["err"]:6.2f} {s["d_in"]:7.3f} {s["d_bd"]:7.3f} '
              f'{s["d_bd"]/max(s["d_in"],1e-9):5.1f} {s["rev"]:6.0f} '
              f'{s["net"]:8.2f} {s["swing"]:7.1f}')
    print('\n判读：反转% <40 且 净位移明显大于 A → 噪声是主因；'
          '若固定噪声组净位移仍 ~10° → 闭环漂移(层2)，需 DAgger')


if __name__ == '__main__':
    main()
