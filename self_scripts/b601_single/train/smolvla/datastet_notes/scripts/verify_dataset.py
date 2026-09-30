#!/usr/bin/env python3
"""b601 数据集一致性校验。

结构层（默认，秒级，不需要 GPU / lerobot 环境）:
    1. 物理总行数 == meta total_frames          <- 一行捕获 file-009 残留块
    2. 每个文件 物理行号 == index 值
    3. 全局 index 单调唯一，恰为 0..N-1
    4. episode 边界语义：dataset_from_index 处确为该集且 frame_index==0
    5. 每集独占一段连续区间（episode_index 切换次数 == 集数-1）

解码层（--decode，需 lerobot conda 环境）:
    LeRobotDataset 实拉 每集首末帧 + N 个随机帧，校验 episode/frame 定位与图像解码

用法:
    python verify_dataset.py --root /data/share/b601_20260910_164106
    python verify_dataset.py --root /data/share/b601_20260910_164106 --decode
    python verify_dataset.py --root ... --decode --random 400 --seed 0

分工（重要）
------------
本仓库有**两份** verify_dataset.py，刻意分工，不要互相覆盖：

    datastet_notes/scripts/verify_dataset.py   ← 本文件
        「人读报告」：5 项编号检查，输出便于人工判读；**只校验，不改数据**。
    bench/verify_dataset.py
        「CI / 同步钩子」：额外提供 `--fix`（自动修复，落盘前验证）、
        `--video`（ffprobe 容器帧数 vs meta）、`--json`。

⚠️ 两份的**检测判据不同**，改判据时必须同步两边：

    * 本文件判据 2 是「每个文件 物理起始行号 == 该文件首行 index」
      —— 这是**症状级**检测。注意那个残留块会把 file-010 之后的**所有**文件
      整体推移，所以「错位数 = 受影响文件数」会远多于 1，不要据此决定删除范围。
    * bench/ 那份用「**`index` 值重复**」作为判据 —— 这是**根因级**检测，
      能精确定位到 1 个 row group；`--fix` 也只删这一处。

    真正需要修复的实体只有一处（残留 row group）；本文件判据 1（总行数不符）
    与判据 2 都是它的症状。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_ROOT = "/data/share/b601_20260910_164106"


def structural(root: str) -> tuple[bool, pd.DataFrame, pd.DataFrame]:
    info = json.load(open(os.path.join(root, "meta/info.json")))
    want = info["total_frames"]

    eps = pd.concat(
        [
            pd.read_parquet(f)
            for f in sorted(glob.glob(os.path.join(root, "meta/episodes/chunk-*/file-*.parquet")))
        ]
    ).sort_values("episode_index").reset_index(drop=True)

    files = sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet")))
    parts, phys, bad = [], 0, []
    for f in files:
        d = pd.read_parquet(f, columns=["index", "episode_index", "frame_index"])
        lo, hi = int(d["index"].min()), int(d["index"].max())
        if lo != phys or hi != phys + len(d) - 1:
            bad.append((os.path.basename(f), phys, lo))
        phys += len(d)
        parts.append(d)
    data = pd.concat(parts, ignore_index=True)

    ok = True
    print("=" * 62)
    print("结构层校验")
    print("=" * 62)
    print(f"数据文件        : {len(files)} 个 / 行数 {phys}")

    c1 = phys == want
    print(f"[{'OK ' if c1 else 'FAIL'}] 1. 总行数 == meta total_frames : {phys} vs {want}"
          f"{'' if c1 else f'  (差 {phys - want})'}")
    ok &= c1

    c2 = not bad
    print(f"[{'OK ' if c2 else 'FAIL'}] 2. 文件边界 物理行号==index   : 错位 {len(bad)} 个")
    for b in bad:
        print(f"        {b[0]}: 物理起始 {b[1]} 但 index 起始 {b[2]}（差 {b[2] - b[1]}）")
    ok &= c2

    a = data["index"].values
    c3 = bool((a == np.arange(len(a))).all())
    print(f"[{'OK ' if c3 else 'FAIL'}] 3. index 单调唯一 0..{len(a) - 1}")
    ok &= c3

    errs = []
    for _, e in eps.iterrows():
        ep, s, t, ln = int(e["episode_index"]), int(e["dataset_from_index"]), int(e["dataset_to_index"]), int(e["length"])
        if s >= len(data) or t > len(data):
            errs.append((ep, "索引越界"))
            continue
        first, last = data.iloc[s], data.iloc[t - 1]
        if not (first["episode_index"] == ep and first["frame_index"] == 0):
            errs.append((ep, "head", int(first["episode_index"]), int(first["frame_index"])))
        if not (last["episode_index"] == ep and last["frame_index"] == ln - 1):
            errs.append((ep, "tail", int(last["episode_index"]), int(last["frame_index"])))
    c4 = not errs
    print(f"[{'OK ' if c4 else 'FAIL'}] 4. episode 边界语义          : {len(eps)} 集, 错误 {len(errs)}")
    for e in errs[:5]:
        print(f"        {e}")
    ok &= c4

    switch = np.flatnonzero(np.diff(data["episode_index"].values) != 0) + 1
    c5 = len(switch) == len(eps) - 1
    print(f"[{'OK ' if c5 else 'FAIL'}] 5. 每集独占一段连续区间      : 切换 {len(switch)} 次, 期望 {len(eps) - 1}")
    ok &= c5

    print(f"\n结构层: {'全部通过' if ok else '存在问题'}")
    return ok, eps, data


def decode(root: str, eps: pd.DataFrame, n_random: int, seed: int,
           repo_id: str, samples_per_ep: int) -> bool:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    print("=" * 62)
    print("解码层校验")
    print("=" * 62)
    print(f"ds: episodes={ds.num_episodes} frames={ds.num_frames} | "
          f"meta: episodes={len(eps)} frames={int(eps['length'].sum())}")

    starts = eps["dataset_from_index"].values.astype(np.int64)
    lengths = eps["length"].values.astype(np.int64)

    def check(idx: int) -> str | None:
        k = int(np.searchsorted(starts, idx, side="right") - 1)
        ep, ln = int(eps["episode_index"].values[k]), int(lengths[k])
        f = ds[idx]
        if int(f["episode_index"]) != ep:
            return f"idx={idx} episode 不符: got {int(f['episode_index'])} want {ep}"
        if int(f["frame_index"]) != idx - starts[k]:
            return f"idx={idx} frame 不符: got {int(f['frame_index'])} want {idx - starts[k]}"
        for kk, v in f.items():
            if kk.startswith("observation.images"):
                if tuple(v.shape)[0] != 3:
                    return f"idx={idx} {kk} shape 异常: {tuple(v.shape)}"
                if float(v.abs().mean()) == 0:
                    return f"idx={idx} {kk} 全零（疑似解码失败）"
        return None

    fails, n = [], 0
    for k, ep in enumerate(eps["episode_index"].values):
        s, ln = int(starts[k]), int(lengths[k])
        picks = sorted({s, s + ln // 2, s + ln - 1} if samples_per_ep >= 3
                       else set(np.linspace(s, s + ln - 1, samples_per_ep, dtype=int)))
        for idx in picks:
            e = check(int(idx)); n += 1
            if e:
                fails.append(e)
    print(f"每集采样 {samples_per_ep} 帧 x {len(eps)} 集 = {n} 帧, 失败 {len(fails)}")

    rng = np.random.default_rng(seed)
    tot = int(eps["length"].sum())
    for idx in rng.integers(0, tot, n_random):
        e = check(int(idx)); n += 1
        if e:
            fails.append(e)
    print(f"随机 {n_random} 帧（seed={seed}）, 累计检查 {n} 帧, 失败 {len(fails)}")

    for f in fails[:10]:
        print("   ", f)
    ok = not fails
    print(f"\n解码层: {'全部通过' if ok else '存在问题'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--decode", action="store_true", help="追加解码层校验（需 lerobot 环境）")
    ap.add_argument("--repo-id", default="hellozjt/b601_20260910_164106")
    ap.add_argument("--random", type=int, default=400, help="随机帧数，默认 400")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples-per-ep", type=int, default=3, help="每集采样帧数（首/中/末）")
    a = ap.parse_args()

    ok, eps, _ = structural(a.root)
    if a.decode:
        ok &= decode(a.root, eps, a.random, a.seed, a.repo_id, a.samples_per_ep)

    print("=" * 62)
    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
