#!/usr/bin/env python3
"""b601 π0.5 离线动作误差评测 —— z790 4090 版（与 A800 上 b601_eval_mae.py 完全同口径）。

指标：
  1. 每关节 MAE（度）与整体 MAE，比较对象是 50 步 GT 动作块
  2. 平凡解基线：hold+ = 复制当前 state 50 步；hold- = 复制 -state 50 步（本数据的正确平凡解）
  3. 视觉消融：三路图像换成另一锚点的图像，看 50 步动作平均变化（度）

两种数据来源：
  A) --anchors x.npz      由 A800 上 export_anchors.py 导出（推荐，不用搬 22G 数据集）
  B) --dataset-root DIR   本地 v2.1 数据集（episode_XXXXXX.parquet + 每集 mp4）

用法：
  source b601_pi05_env.sh
  $PY infer_offline.py --ckpt "${B601_CKPT[330]}" --anchors anchors_6.npz
  $PY infer_offline.py --ckpt "${B601_CKPT[462]}" --anchors anchors_6.npz --out offline_462.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('B601_SFT_ROOT', '/home/kf/b601_pi05'))
sys.path.insert(0, str(ROOT / 'scripts'))
from b601_config import register  # noqa: E402

HORIZON = 50
CAMS = ('front', 'hand', 'top')


def anchors_from_npz(path: Path):
    z = np.load(path)
    out = []
    for i, eid in enumerate(z['episodes']):
        out.append(dict(episode=int(eid), frame=int(z['frames'][i]),
                        state=z['states'][i].astype(np.float32),
                        gt=z['gts'][i].astype(np.float32),
                        images={c: z[f'img_{c}'][i].astype(np.float32) for c in CAMS},
                        prompt=str(z['prompts'][i])))
    return out


def anchors_from_dataset(dataset_root: Path, episodes, stride: int = 0):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset('b601/v21_clean', root=str(dataset_root),
                        delta_timestamps={'action': [i / 30 for i in range(HORIZON)]},
                        video_backend='pyav')
    out = []
    for eid in episodes:
        ep = ds.meta.episodes[eid]
        start = int(ds.episode_data_index['from'][eid])
        length = ep['length']
        frame = start + (length // 2 if stride == 0 else min(stride, length - 1))
        row = ds[frame]
        out.append(dict(episode=eid, frame=frame,
                        state=np.asarray(row['observation.state'].numpy(), dtype=np.float32),
                        gt=np.asarray(row['action'].numpy(), dtype=np.float32),
                        images={c: np.asarray(row[f'observation.images.{c}'].numpy(), dtype=np.float32) for c in CAMS},
                        prompt=str(row['task'])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--anchors', type=Path, default=None)
    ap.add_argument('--dataset-root', type=Path, default=None)
    ap.add_argument('--episodes', type=int, nargs='+', default=[0, 40, 120])
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    if not a.anchors and not a.dataset_root:
        ap.error('需要 --anchors 或 --dataset-root')

    from openpi.policies.policy_config import create_trained_policy
    cfg = register()
    policy = create_trained_policy(cfg, a.ckpt, pytorch_device='cuda')
    anchors = anchors_from_npz(a.anchors) if a.anchors else anchors_from_dataset(a.dataset_root, a.episodes)
    noise = np.random.default_rng(42).normal(size=(HORIZON, cfg.model.action_dim)).astype(np.float32)

    report = dict(checkpoint=str(a.ckpt), source=str(a.anchors or a.dataset_root),
                  horizon=HORIZON, rows=[])
    for anc in anchors:
        pred = np.asarray(policy.infer(dict(state=anc['state'], images=anc['images'],
                                            prompt=anc['prompt']), noise=noise)['actions'], dtype=np.float32)[:, :7]
        gt = anc['gt'][:, :7]
        hold_plus = np.repeat(anc['state'][None], HORIZON, axis=0)
        hold_minus = -hold_plus
        other = next(x for x in anchors if x['episode'] != anc['episode'])
        pred_abl = np.asarray(policy.infer(dict(state=anc['state'], images=other['images'],
                                                prompt=anc['prompt']), noise=noise)['actions'],
                              dtype=np.float32)[:, :7]
        row = dict(episode=anc['episode'], frame=int(anc['frame']),
                   mae_model=float(np.abs(pred - gt).mean()),
                   mae_hold_plus=float(np.abs(hold_plus - gt).mean()),
                   mae_hold_minus=float(np.abs(hold_minus - gt).mean()),
                   per_joint_model=np.abs(pred - gt).mean(0).round(3).tolist(),
                   per_joint_hold_minus=np.abs(hold_minus - gt).mean(0).round(3).tolist(),
                   vision_ablation_delta=float(np.abs(pred_abl - pred).mean()))
        row['ratio_vs_hold_minus'] = round(row['mae_model'] / row['mae_hold_minus'], 3) if row['mae_hold_minus'] else None
        row['ratio_vs_hold_plus'] = round(row['mae_model'] / row['mae_hold_plus'], 3) if row['mae_hold_plus'] else None
        report['rows'].append(row)
        print('EVAL', json.dumps(row), flush=True)

    model = np.mean([r['mae_model'] for r in report['rows']])
    hm = np.mean([r['mae_hold_minus'] for r in report['rows']])
    hp = np.mean([r['mae_hold_plus'] for r in report['rows']])
    report['summary'] = dict(
        n_anchors=len(report['rows']),
        mae_model=round(float(model), 3),
        mae_model_std=round(float(np.std([r['mae_model'] for r in report['rows']])), 3),
        mae_hold_minus=round(float(hm), 3),
        mae_hold_minus_std=round(float(np.std([r['mae_hold_minus'] for r in report['rows']])), 3),
        mae_hold_plus=round(float(hp), 3),
        ratio_vs_hold_minus=round(float(model / hm), 3) if hm else None,
        ratio_vs_hold_plus=round(float(model / hp), 3) if hp else None,
        vision_ablation=round(float(np.mean([r['vision_ablation_delta'] for r in report['rows']])), 3))
    out = a.out or Path(f"offline_mae_{a.ckpt.name}.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print('SUMMARY', json.dumps(report['summary']), flush=True)
    print('WROTE', out.resolve(), flush=True)


if __name__ == '__main__':
    main()
