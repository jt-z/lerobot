#!/usr/bin/env python3
"""四组 rollout 的分组明细：净位移(各关节) + 分段推进"""
import csv
import numpy as np

J = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper']
LOG = '/home/kf/b601_pi05/logs/'
cases = {
    'A 默认噪声 exec5 ': LOG + 'rollout_pi05_ws_20261022_163301.csv',
    'B 默认噪声 exec10': LOG + 'rollout_pi05_ws_20261023_143725.csv',
    'C 固定噪声 exec5 ': LOG + 'rollout_pi05_ws_20261023_143906.csv',
    'D 固定噪声 exec10': LOG + 'rollout_pi05_ws_20261023_143540.csv',
}

for k, p in cases.items():
    rows = list(csv.DictReader(open(p, newline='')))
    st = np.array([[float(r['state_' + j]) for j in J] for r in rows])
    cmd = np.array([[float(r['cmd_' + j]) for j in J] for r in rows])
    t = np.array([float(r['t']) for r in rows])
    net = st[-1] - st[0]
    print(f'\n{k}  时长={t[-1]:.1f}s 步={len(rows)}')
    print('  关节         起点      终点    净位移(方向)')
    for j, i in zip(J, range(7)):
        print(f'  {j:14s} {st[0,i]:7.2f} {st[-1,i]:8.2f} {net[i]:+8.2f}')
    print('  分段:')
    for a, b in [(0, 10), (10, 30), (30, 60)]:
        m = (t >= a) & (t < b)
        if m.sum() > 2:
            s0, s1 = st[m][0], st[m][-1]
            trav = np.abs(np.diff(st[m], axis=0)).sum(0).mean()
            print(f'    {a:2d}-{b:2d}s  净={np.abs(s1-s0).mean():5.2f}°  行程={trav:7.1f}°  '
                  f'摆比={trav/max(np.abs(s1-s0).mean(),0.1):6.1f}')
    # 最大单关节净位移，判断有没有"往一个方向走"
    print(f'  最大关节净位移 = {np.abs(net).max():.2f}°  ({J[int(np.abs(net).argmax())]})')
    print(f'  gripper: 起 {st[0,6]:.1f} → 终 {st[-1,6]:.1f}  (行程 {np.abs(np.diff(st[:,6])).sum():.1f}°)')
