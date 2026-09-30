#!/usr/bin/env python3
"""
LeRobot 数据集完整性检查 / 修复 / 校验。

背景
----
本数据集的源端在录制 ep161 时崩溃过一次（残留 `tmp*/observation.images.*_161.mp4`），
崩溃瞬间 writer 缓冲里那 11,005 行被刷进了 `data/chunk-000/file-009.parquet` 的
**最后一个 row group**，而这个 row group 仍然打着 `episode_index=161` 的标签。

后果：`LeRobotDataset` 依赖「物理行号 == `index` 值」来定位 episode。
这个多出来的块把 file-010 之后所有 episode 的物理位置整体后移 11,005，
于是 state/action 会取到错误 episode 的行，而视频仍按 meta 的 episode/timestamp 查
→ **图文错配，且 loss 曲线看着正常**。9-16、9-18 已各修一次，只要源端再增量重传就会复现。

判据（重要）
------------
**不要用「物理行号 != index」当作要删除的目标。** 那个多出来的块会把下游**所有**
row group 都推移，于是 60+ 个组全部"错位" —— 照此删除会删掉大半个数据集。

真正的判据是 **`index` 值出现重复**：那个块里的 418360..429364 与后面 file-010/011
里合法行的 `index` 值**一一重复**。去掉重复的那一份（保留物理位置靠后的那份，
因为合法的数据是后写入的），剩下的正好是 0..N-1。

本脚本据此定位要丢弃的行，并在**落盘前验证**「丢弃后 index 恰好是 0..N-1」；
验证不过就拒绝写入。

用法
----
    # 1) 只检查（只读，安全；有问题时退出码 1）—— 建议每次同步后先跑这个
    python bench/verify_dataset.py --root /data/share/b601_20260910_164106

    # 2) 检查并修复（先备份为 <file>.parquet.bak，写临时文件校验后原子替换）
    python bench/verify_dataset.py --root /data/share/b601_20260910_164106 --fix

    # 3) 额外做视频解码抽查（需要装了 lerobot 的环境）
    python bench/verify_dataset.py --root ... --decode 400

    # 4) 机器可读输出（给 CI / 同步钩子用）
    python bench/verify_dataset.py --root ... --json --no-video

退出码：0 = 通过；1 = 检出问题（或修复失败）；2 = 脚本自身出错。

分工（重要）
------------
本仓库有**两份** verify_dataset.py，刻意分工，不要互相覆盖：

    datastet_notes/scripts/verify_dataset.py
        「人读报告」：5 项编号检查，输出便于人工判读；**只校验，不改数据**。
    bench/verify_dataset.py   ← 本文件
        「CI / 同步钩子」：额外提供 `--fix`（自动修复，落盘前验证）、
        `--video`（ffprobe 容器帧数 vs meta）、`--json`（stdout 纯 JSON）。

⚠️ 两份的**检测判据不同**，改判据时必须同步两边：

    * datastet_notes/ 那份用「文件边界 物理起始行号 == index」—— **症状级**检测。
    * 本文件用「**`index` 值重复**」—— **根因级**检测，能精确定位到 1 个 row group。

    为什么必须用后者：残留块会把下游**所有** row group 整体推移 11,005，
    按「位置错位」去删会命中 60+ 个组（实测 62 个），删掉大半个数据集。
    真正多余的行只有一处，其 `index` 值与后面合法行的 `index` **一一重复**。

设计原则
--------
* **默认只读**。修复必须显式 `--fix`。
* **落盘前先干跑验证**：只有「丢弃后全链路自洽」才写。
* **不硬编码** file-009 / 11,005 —— 靠不变量检测。
* 基础检查不需要 lerobot / torch；只有 `--decode` 才需要。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n) -> str:
    return f"{int(n):,}"


def data_files(root: Path) -> list[Path]:
    """按 LeRobot 的加载顺序返回 data parquet（chunk 升序 + file 升序）。"""
    return sorted((root / "data").glob("chunk-*/*.parquet"))


def read_meta(root: Path):
    info = json.loads((root / "meta" / "info.json").read_text())
    ep_files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    eps = pa.concat_tables([pq.read_table(f) for f in ep_files]) if ep_files else None
    return info, eps


# --------------------------------------------------------------------------- #
# 核心：定位多出来的行
# --------------------------------------------------------------------------- #


def scan(root: Path) -> dict:
    """读取 data parquet，找出 `index` 值重复的行（= 多出来的那部分）。只读。"""
    files = data_files(root)
    if not files:
        raise RuntimeError(f"在 {root}/data 下没找到 parquet")

    idx_parts: list[np.ndarray] = []
    epi_parts: list[np.ndarray] = []
    rg_map: list[dict] = []          # 每个 row group 的 (file, rg, 起始物理行, 行数)
    per_file: list[dict] = []

    for f in files:
        pf = pq.ParquetFile(f)
        tbl = pf.read(columns=["index", "episode_index"])
        idx = tbl["index"].to_numpy()
        epi = tbl["episode_index"].to_numpy()
        start = sum(len(x) for x in idx_parts)
        local = 0
        for r in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(r).num_rows
            rg_map.append({"file": f, "name": f.name, "row_group": r,
                           "start": start + local, "rows": int(n)})
            local += n
        per_file.append({"name": f.name, "rows": int(len(idx)),
                         "row_groups": pf.metadata.num_row_groups})
        idx_parts.append(idx)
        epi_parts.append(epi)

    idx = np.concatenate(idx_parts)
    epi = np.concatenate(epi_parts)
    total = len(idx)

    # ---- 判据：index 值重复 ----
    order = np.arange(total)
    vals, first_pos = np.unique(idx, return_index=True)
    _, last_pos_rev = np.unique(idx[::-1], return_index=True)
    last_pos = total - 1 - last_pos_rev
    pos_of_val = np.searchsorted(vals, idx)              # 每行 → 其 index 值在 vals 里的下标
    counts = np.bincount(pos_of_val, minlength=len(vals))
    is_dup = counts[pos_of_val] > 1                      # 逐行：该行的 index 值是否重复

    bogus = np.zeros(total, dtype=bool)
    strategy = None
    if is_dup.any():
        keep_last = last_pos[pos_of_val]                 # 丢掉较早的那份
        keep_first = first_pos[pos_of_val]               # 丢掉较晚的那份
        cand_late = is_dup & (order != keep_last)
        cand_early = is_dup & (order != keep_first)
        cands = [("保留物理位置靠后的一份（丢掉较早的）", cand_late),
                 ("保留物理位置靠前的一份（丢掉较晚的）", cand_early)]
        for label, cand in cands:
            rest = idx[~cand]
            if len(rest) and np.array_equal(rest, np.arange(len(rest), dtype=idx.dtype)):
                bogus, strategy = cand, label
                break
        if strategy is None:      # 两种都不自洽 → 仍然报出来，但标为不可自动修复
            bogus, strategy = cand_late, None

    # ---- 症状（不是删除目标）：物理行号 != index 的下游错位 ----
    surviving = np.arange(total, dtype=np.int64)[~bogus]
    misaligned = int(np.sum(idx[~bogus] != surviving)) if total else 0

    return {
        "files": per_file,
        "rg_map": rg_map,
        "total_rows": int(total),
        "index": idx,
        "episode_index": epi,
        "bogus_mask": bogus,
        "n_bogus": int(bogus.sum()),
        "strategy": strategy,
        "n_misaligned_rows": misaligned,
    }


def bogus_row_groups(sc: dict) -> tuple[list[dict], bool]:
    """把 bogus 行归到 row group。返回 (完全被污染的组, 是否有组只被部分污染)。"""
    bogus = sc["bogus_mask"]
    full, partial = [], False
    for m in sc["rg_map"]:
        seg = bogus[m["start"]: m["start"] + m["rows"]]
        if seg.all():
            full.append(m)
        elif seg.any():
            partial = True
    return full, partial


def check_meta_consistency(root: Path, sc: dict, info: dict) -> list[str]:
    errs: list[str] = []
    tf = info.get("total_frames")
    if tf is not None:
        eff = sc["total_rows"] - sc["n_bogus"]
        if sc["n_bogus"] == 0 and sc["total_rows"] != tf:
            errs.append(
                f"总行数对不上：parquet {human(sc['total_rows'])} vs meta total_frames "
                f"{human(tf)}（差 {human(sc['total_rows'] - tf)} 行）"
            )
        elif sc["n_bogus"] and eff != tf:
            errs.append(
                f"去掉 {human(sc['n_bogus'])} 个重复行后剩 {human(eff)}，"
                f"仍不等于 meta total_frames {human(tf)} —— 损坏形态与已知模式不同"
            )
    ne = info.get("total_episodes")
    if ne is not None:
        seen = len(np.unique(sc["episode_index"]))
        if seen > ne:
            errs.append(f"episode 数 {seen} 多于 meta 的 {ne}（重复行的副作用）")
    return errs


def check_episodes(root: Path, sc: dict, eps) -> list[str]:
    """在排除重复行之后，每个 episode 的行是否连续、frame_index 是否 0..len-1。"""
    if eps is None:
        return ["（找不到 meta/episodes，跳过 episode 边界检查）"]
    keep = ~sc["bogus_mask"]
    epi = sc["episode_index"][keep]
    errs: list[str] = []
    for i in range(eps.num_rows):
        eidx = eps["episode_index"][i].as_py()
        L = eps["length"][i].as_py()
        a = eps["dataset_from_index"][i].as_py()
        b = eps["dataset_to_index"][i].as_py()
        if b - a != L:
            errs.append(f"ep{eidx}: meta 自相矛盾 to-from={b - a} != length={L}")
            continue
        if b > len(epi):
            errs.append(f"ep{eidx}: dataset_to_index={human(b)} 超出数据行数 {human(len(epi))}")
            continue
        seg = epi[a:b]
        if not np.all(seg == eidx):
            errs.append(f"ep{eidx}: 区间 [{human(a)},{human(b)}) 里有 "
                        f"{human(int((seg != eidx).sum()))} 行属于别的 episode")
    return errs


def check_videos(root: Path, info: dict, eps, fps: int) -> tuple[list[str], int]:
    """视频文件是否存在、容器声明的帧数是否覆盖 meta 要求的时间戳。"""
    if shutil.which("ffprobe") is None:
        return ["（跳过视频检查：未找到 ffprobe）"], 0
    if eps is None:
        return ["（跳过视频检查：无 meta/episodes）"], 0

    cams = [k for k in info.get("features", {}) if k.startswith("observation.images.")]
    errs, n_checked = [], 0
    for cam in cams:
        key = f"videos/{cam}"
        need: dict[tuple, float] = {}
        for i in range(eps.num_rows):
            c = eps[key + "/chunk_index"][i].as_py()
            fi = eps[key + "/file_index"][i].as_py()
            to_ts = float(eps[key + "/to_timestamp"][i].as_py())
            need[(c, fi)] = max(need.get((c, fi), 0.0), to_ts)

        for (c, fi), to_ts in sorted(need.items()):
            p = root / "videos" / cam / f"chunk-{c:03d}" / f"file-{fi:03d}.mp4"
            if not p.exists():
                errs.append(f"缺视频文件 {p.relative_to(root)}")
                continue
            n_checked += 1
            try:
                out = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=nb_frames,duration", "-of", "json", str(p)],
                    capture_output=True, text=True, timeout=120,
                )
                st = json.loads(out.stdout)["streams"][0]
                nbf = st.get("nb_frames")
                actual = int(nbf) if nbf not in (None, "N/A") \
                    else int(round(float(st.get("duration", 0)) * fps))
            except Exception as ex:  # noqa: BLE001
                errs.append(f"ffprobe 读不了 {p.name}: {ex}")
                continue
            want = int(round(to_ts * fps))
            if actual + 2 < want:
                errs.append(f"{p.relative_to(root)} 帧数不足：容器 {human(actual)} "
                            f"< meta 要求 {human(want)}")
    return errs, n_checked


# --------------------------------------------------------------------------- #
# 修复
# --------------------------------------------------------------------------- #


def _rewrite(f: Path, drop_rows_local: np.ndarray, drop_rgs: list[int]) -> None:
    """丢掉指定内容并原子替换；先备份。"""
    pf = pq.ParquetFile(f)
    schema = pf.schema_arrow

    bak = f.with_suffix(f.suffix + ".bak")
    if bak.exists():
        print(f"  [备份] {bak.name} 已存在，保留不覆盖 (sha256 {sha256(bak)[:12]}…)")
    else:
        shutil.copy2(f, bak)
        print(f"  [备份] {f.name} → {bak.name} (sha256 {sha256(bak)[:12]}…)")
    print(f"  [原始] {f.name} sha256 {sha256(f)[:12]}…")

    if drop_rgs:
        keep_rgs = [i for i in range(pf.metadata.num_row_groups) if i not in set(drop_rgs)]
        tbl = pf.read_row_groups(keep_rgs)
        how = f"丢 row group {sorted(drop_rgs)}"
    else:
        full = pf.read()
        mask = np.ones(full.num_rows, dtype=bool)
        mask[drop_rows_local] = False
        tbl = full.filter(pa.array(mask))
        how = f"按行过滤（丢 {human(len(drop_rows_local))} 行）"

    tmp = f.with_suffix(f.suffix + ".tmp")
    try:
        pq.write_table(tbl, tmp, compression="snappy")
        chk = pq.ParquetFile(tmp)
        if chk.metadata.num_rows != tbl.num_rows:
            raise RuntimeError("重写后行数不符")
        if not chk.schema_arrow.equals(schema):
            raise RuntimeError("重写后 arrow schema 不一致")
        if chk.schema_arrow.metadata != schema.metadata:
            raise RuntimeError("重写后 schema 元数据（huggingface blob）丢失")
        os.replace(tmp, f)
        print(f"  [完成] {f.name}: {how} → 现 {human(chk.metadata.num_rows)} 行")
    finally:
        if tmp.exists():
            tmp.unlink()


def repair(root: Path, sc: dict, info: dict, assume_yes: bool) -> bool:
    if sc["n_bogus"] == 0:
        print("无需修复。")
        return True
    if sc["strategy"] is None:
        print("✗ 无法自动判定该丢哪一份重复行（两种取法都不自洽），请人工检查。",
              file=sys.stderr)
        return False

    full_rgs, partial = bogus_row_groups(sc)
    print(f"\n检出 {human(sc['n_bogus'])} 行 `index` 值重复（= 多出来的部分）：")
    by_file: dict[str, list[dict]] = {}
    for m in full_rgs:
        by_file.setdefault(m["name"], []).append(m)
    for name, ms in sorted(by_file.items()):
        tot = sum(m["rows"] for m in ms)
        print(f"  {name}: row group {[m['row_group'] for m in ms]}，共 {human(tot)} 行")
    if partial:
        print("  （有 row group 只被部分污染，将按行过滤）")

    print(f"\n判定策略：{sc['strategy']}")
    tf = info.get("total_frames")
    print(f"修复后总行数：{human(sc['total_rows'] - sc['n_bogus'])}"
          + (f"　meta total_frames = {human(tf)}" if tf is not None else ""))

    if not assume_yes:
        try:
            ans = input("\n确认执行修复？(yes/no) ").strip().lower()
        except EOFError:
            ans = "no"
        if ans not in ("y", "yes"):
            print("已取消。")
            return False

    bogus = sc["bogus_mask"]

    # 按物理顺序收集「含 bogus 行的文件」→ (路径, 本文件内局部行号, 完全污染的 row group)
    order: list[str] = []
    for m in sc["rg_map"]:
        if m["name"] not in order:
            order.append(m["name"])
    affected = []
    for name in order:
        ms = [m for m in sc["rg_map"] if m["name"] == name]
        fstart = ms[0]["start"]
        file_rows = sum(m["rows"] for m in ms)
        seg = bogus[fstart: fstart + file_rows]
        if not seg.any():
            continue
        full = [m["row_group"] for m in ms if bogus[m["start"]: m["start"] + m["rows"]].all()]
        n_touched = sum(1 for m in ms if bogus[m["start"]: m["start"] + m["rows"]].any())
        affected.append((ms[0]["file"], np.nonzero(seg)[0], full, len(full) != n_touched))

    for f, rows_local, full_rgs_of_file, is_partial in affected:
        if is_partial:
            _rewrite(f, rows_local, [])          # 按行过滤
        else:
            _rewrite(f, np.array([], dtype=int), full_rgs_of_file)  # 整块丢 row group

    print("\n重新扫描…")
    sc2 = scan(root)
    info2, eps2 = read_meta(root)
    errs = check_meta_consistency(root, sc2, info2) + check_episodes(root, sc2, eps2)
    if sc2["n_bogus"] or errs:
        print("✗ 修复后仍存在问题：", file=sys.stderr)
        for e in errs:
            print("   -", e, file=sys.stderr)
        print("\n可从 <file>.parquet.bak 还原。", file=sys.stderr)
        return False
    print("✓ 修复后全部不变量成立。")
    return True


# --------------------------------------------------------------------------- #
# 可选：用 LeRobotDataset 真拉帧
# --------------------------------------------------------------------------- #


def decode_probe(root: Path, repo_id: str | None, n_random: int) -> tuple[list[str], str]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as ex:  # noqa: BLE001
        return [], f"（跳过解码抽查：import lerobot 失败 — {ex}）"

    import random
    ds = LeRobotDataset(repo_id or root.name, root=str(root))
    probes: list[int] = []
    n_ep = len(ds.meta.episodes)
    for i in range(n_ep):
        a = int(ds.meta.episodes[i]["dataset_from_index"])
        b = int(ds.meta.episodes[i]["dataset_to_index"])
        probes += [a, max(a, b - 1)]
    random.seed(0)
    probes += random.sample(range(len(ds)), min(n_random, len(ds)))

    errs, n_fail = [], 0
    for i in probes:
        try:
            ds[i]
        except Exception as ex:  # noqa: BLE001
            n_fail += 1
            if n_fail <= 5:
                errs.append(f"idx {i} 解码失败：{str(ex)[:100]}")
    return errs, f"解码抽查 {human(len(probes))} 帧（{n_ep}×2 边界 + {min(n_random, len(ds))} 随机），失败 {n_fail}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LeRobot 数据集完整性检查 / 修复 / 校验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", required=True, type=Path, help="数据集根目录")
    ap.add_argument("--fix", action="store_true", help="检出问题后执行修复（默认只检查）")
    ap.add_argument("--yes", "-y", action="store_true", help="修复时不交互确认（给钩子用）")
    ap.add_argument("--decode", type=int, default=0, metavar="N", help="额外随机抽查 N 帧解码（需 lerobot）")
    ap.add_argument("--repo-id", default=None, help="传给 LeRobotDataset 的 repo_id（默认用 root 名）")
    ap.add_argument("--video", dest="video", action="store_true", default=None, help="强制做视频检查")
    ap.add_argument("--no-video", dest="video", action="store_false", help="跳过视频检查")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"✗ 不是目录：{root}", file=sys.stderr)
        return 2

    try:
        info, eps = read_meta(root)
        sc = scan(root)
    except Exception as ex:  # noqa: BLE001
        print(f"✗ 扫描失败：{ex}", file=sys.stderr)
        return 2

    errs = check_meta_consistency(root, sc, info) + check_episodes(root, sc, eps)
    video_msgs, n_vid = [], 0
    if args.video is not False:
        try:
            verrs, n_vid = check_videos(root, info, eps, int(info.get("fps", 30)))
            errs += verrs
            video_msgs = verrs or [f"（视频检查通过：{human(n_vid)} 个文件）"]
        except Exception as ex:  # noqa: BLE001
            video_msgs = [f"（视频检查出错：{ex}）"]

    decode_msgs = []
    if args.decode:
        derrs, dmsg = decode_probe(root, args.repo_id, args.decode)
        errs += derrs
        decode_msgs = [dmsg]

    ok = (sc["n_bogus"] == 0) and not errs

    if not args.json:
        print(f"数据集：{root}")
        print(f"  数据文件        : {len(sc['files'])} 个")
        print(f"  总行数          : {human(sc['total_rows'])}")
        print(f"  meta 应有行数   : {human(info.get('total_frames', -1))}")
        if sc["n_bogus"]:
            full_rgs, partial = bogus_row_groups(sc)
            print(f"  重复行（多出来）: {human(sc['n_bogus'])} 行  ← 检出损坏")
            print(f"  下游错位行      : {human(sc['n_misaligned_rows'])} 行（是这个块的副作用，不是删除目标）")
            print(f"  涉及 row group  : {len(full_rgs)} 个完整"
                  + ("，另有部分污染" if partial else ""))
            for m in full_rgs[:6]:
                print(f"      {m['name']} rg#{m['row_group']}  {human(m['rows'])} 行")
            if len(full_rgs) > 6:
                print(f"      … 另有 {len(full_rgs) - 6} 个")
            print(f"  自动修复策略    : {sc['strategy'] or '✗ 无法自动判定，需人工检查'}")
        else:
            print("  重复行          : 0")
        for m in video_msgs:
            print(f"  {m}")
        for m in decode_msgs:
            print(f"  {m}")
        if errs:
            print("\n✗ 检出问题：")
            for e in errs:
                print("   -", e)
        if sc["n_bogus"]:
            print(f"\n✗ 数据集损坏：{human(sc['n_bogus'])} 行 `index` 值重复（上面已列出位置）。")
            if sc["strategy"]:
                print("   加 --fix 可自动修复（先备份为 <file>.parquet.bak，落盘前会再验证一次）。")
            else:
                print("   ✗ 无法自动判定该丢哪一份，需人工检查。")
        elif not errs:
            print("\n✓ 全部不变量成立（index 无重复、物理行号 == index、episode 边界自洽）")

    # ---- 修复（若要求），之后以「最终状态」出报告 ----
    fixed = False
    if not ok and args.fix:
        if args.json:
            # --json 模式下让修复过程的进度输出走 stderr，保证 stdout 只有一份 JSON
            _real, sys.stdout = sys.stdout, sys.stderr
            try:
                fixed = repair(root, sc, info, args.yes)
            finally:
                sys.stdout = _real
        else:
            fixed = repair(root, sc, info, args.yes)
        if fixed:
            try:
                info, eps = read_meta(root)
                sc2 = scan(root)
                errs2 = check_meta_consistency(root, sc2, info) + check_episodes(root, sc2, eps)
                sc, errs = sc2, errs2
                ok = (sc["n_bogus"] == 0) and not errs
            except Exception as ex:  # noqa: BLE001
                print(f"✗ 修复后复检失败：{ex}", file=sys.stderr)
                ok = False

    if args.json:
        print(json.dumps({
            "root": str(root),
            "total_rows": sc["total_rows"],
            "meta_total_frames": info.get("total_frames"),
            "n_files": len(sc["files"]),
            "n_bogus_rows": sc["n_bogus"],
            "n_misaligned_rows": sc["n_misaligned_rows"],
            "strategy": sc["strategy"],
            "fixed": fixed,
            "errors": errs,
            "ok": ok,
        }, ensure_ascii=False, indent=2))
    elif fixed:
        print(f"\n修复完成：现在 {human(sc['total_rows'])} 行，"
              f"meta {human(info.get('total_frames', -1))} 行。")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
