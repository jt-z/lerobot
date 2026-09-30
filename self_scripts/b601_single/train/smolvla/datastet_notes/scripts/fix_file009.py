#!/usr/bin/env python3
"""修复 b601 数据集 file-009.parquet 末尾残留的 11,005 行废弃 row group。

机理见 ../02-损坏机理-物理行号等于index.md，操作细节见 ../03-file009-修复操作.md。

源端每次重传都会把该文件覆盖回损坏版，因此本脚本设计为**幂等**：
    - 无损坏       -> 直接退出，不动任何文件
    - 已知损坏模式 -> 备份（若无）后，按 row group 边界无损丢弃末尾块，原子替换
    - 其他不一致   -> 拒绝执行，要求人工排查

用法:
    python fix_file009.py [--root /data/share/b601_20260910_164106] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_ROOT = "/data/share/b601_20260910_164106"
CORRUPT_SHA256 = "81518caf0f01893a98ac4cac2b49b054e3f7dc7d9526f1fe90c122211a3f0068"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--dry-run", action="store_true", help="只检查，不写盘")
    a = ap.parse_args()
    root = a.root

    target = os.path.join(root, "data/chunk-000/file-009.parquet")
    info_path = os.path.join(root, "meta/info.json")
    if not os.path.exists(target) or not os.path.exists(info_path):
        print(f"[错误] 找不到数据集: {root}", file=sys.stderr)
        return 2

    want = json.load(open(info_path))["total_frames"]
    files = sorted(glob.glob(os.path.join(root, "data/chunk-000/*.parquet")))
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in files)
    diff = total - want
    print(f"数据文件      : {len(files)} 个")
    print(f"物理总行数    : {total}")
    print(f"meta 期望     : {want}")
    print(f"差值          : {diff}")

    if diff == 0:
        print("\n[OK] 无需修复。")
        return 0
    if diff < 0:
        print(
            "\n[拒绝] 物理行数少于 meta，这不是文档描述的损坏模式（应为多出 11,005 行）。"
            "请人工排查。",
            file=sys.stderr,
        )
        return 3

    src = pq.ParquetFile(target)
    md = src.metadata
    rg = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
    print(f"file-009 行数 : {md.num_rows}")
    print(f"file-009 分组 : {rg}")

    if len(rg) < 2 or rg[-1] != diff:
        print(
            f"\n[拒绝] 末尾 row group 为 {rg[-1] if rg else 0}，与差值 {diff} 不符，"
            "模式与已知损坏不同（已知模式为末尾单独一块 11005 行）。请人工确认后再修。",
            file=sys.stderr,
        )
        return 4

    keep = md.num_rows - diff
    print(f"\n判定：丢弃末尾 row group（{diff} 行），保留前 {len(rg) - 1} 组共 {keep} 行")

    if a.dry_run:
        print("[dry-run] 未写盘。")
        return 0

    # 只读需要保留的 row group —— 废弃块不进内存
    tbl = pa.concat_tables([src.read_row_group(i) for i in range(md.num_row_groups - 1)])
    if md.metadata:  # 保住 ARROW:schema / huggingface 等文件级 key-value
        tbl = tbl.replace_schema_metadata(md.metadata)

    bak = target + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(target, bak)
        hint = ""
        if os.path.abspath(root) == os.path.abspath(DEFAULT_ROOT):
            hint = f"  （原始损坏版 sha256 应为 {CORRUPT_SHA256[:8]}...）"
        print(f"备份 -> {bak}{hint}")
    else:
        print(f"备份已存在，保留原样: {bak}")

    tmp = target + ".repair_tmp"
    pq.write_table(tbl, tmp, compression="snappy")  # 原文件各列均为 SNAPPY
    n = pq.ParquetFile(tmp).metadata.num_rows
    if n != keep:
        print(f"[错误] 临时文件行数 {n} != {keep}，未替换。", file=sys.stderr)
        return 5
    os.replace(tmp, target)
    print(f"已原子替换 -> {target}  ({n} 行)")

    # 复验：全局边界对齐
    phys, bad = 0, []
    for p in files:
        d = pq.ParquetFile(p).metadata.num_rows
        lo = pq.read_table(p, columns=["index"]).column("index")[0].as_py()
        if lo != phys:
            bad.append((os.path.basename(p), phys, lo))
        phys += d
    print(f"复验：物理总行数 {phys}（期望 {want}），边界错位文件 {len(bad)} 个")
    for b in bad:
        print(f"  {b}")
    if bad or phys != want:
        print("[警告] 复验未通过，请人工检查。", file=sys.stderr)
        return 6

    print("\n[OK] 修复完成且复验通过。建议再跑 verify_dataset.py（可加 --decode）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
