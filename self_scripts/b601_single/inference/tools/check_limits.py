#!/usr/bin/env python3
"""检查是不是被限位钳住了：cmd / state 的极值分布"""
import csv
import numpy as np

J = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper']
LOG = '/home/kf/b601_pi05/logs/'
cases = {
    'A': LOG + 'rollout_pi05_ws_20261022_163301.csv',
    'B': LOG + 'rollout_pi05_ws_20261023_143725.csv',
    'C': LOG + 'rollout_pi05_ws_20261023_143906.csv',
    'D': LOG + 'rollout_pi05_ws_20261023_143540.csv',
}

for k, p in cases.items():
    rows = list(csv.DictReader(open(p, newline='')))
    st = np.array([[float(r['state_' + j]) for j in J] for r in rows])
    cmd = np.array([[float(r['cmd_' + j]) for j in J] for r in rows])
    print(f'\n=== {k} ===')
    print(f'{"关节":14s} {"state min":>10s} {"state max":>10s} {"cmd min":>10s} {"cmd max":>10s} {"撞顶步数%":>9s}')
    for i, j in enumerate(J):
        hi = np.abs(st[:, i]) >= 12.0
        print(f'{j:14s} {st[:,i].min():10.2f} {st[:,i].max():10.2f} '
              f'{cmd[:,i].min():10.2f} {cmd[:,i].max():10.2f} {100*hi.mean():8.1f}%')
    # 有多少比例的步数处于 |state| >= 12
    print(f'整机 |state|>=12 的步数占比: {100*(np.abs(st[:, :6])>=12).any(1).mean():.1f}%')
    print(f'全局 |cmd| max = {np.abs(cmd).max():.2f}°   |state| max = {np.abs(st).max():.2f}°')
