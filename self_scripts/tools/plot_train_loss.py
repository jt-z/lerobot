"""从 lerobot-train 的日志里解析 loss / lr / grad-norm，画成训练曲线图。

用法：
    python self_scripts/tools/plot_train_loss.py [日志路径] [--save 输出.png] [--span 500]

不传日志路径时，自动取 self_scripts/output_lerobot_train/logs/ 下最新的 *.log。

说明：
    - lerobot 的日志被 tqdm 的 \r 刷新和 logging 的 \n 混着写，所以先把 \r 也当换行切开。
    - 逐步的 loss 来自 tqdm 行（每步一条，步数精确）；grad-norm / data_s 等只有
      log_freq 倍数的 INFO 行才有，按「INFO 行之前最近的一条 tqdm 步数」对齐，步数因此也是精确的。
    - loss 原始曲线噪声大，另叠一条 EMA 平滑线看趋势。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SELF_SCRIPTS = Path(__file__).resolve().parent.parent
LOG_DIR = SELF_SCRIPTS / "output_lerobot_train" / "logs"

# tqdm 进度行：形如 "Training:  42%|███ | 6330/150000 [01:02<03:04, 3.58s/it, loss=0.123, lr=1.2e-05]"
RE_TQDM = re.compile(r"(\d+)/(\d+)\s*\[[^\]]*?loss=([0-9.eE+-]+)")
# INFO 指标行：形如 "INFO 2026-10-07 01:12:13 ot_train.py:444 step:150K ... loss:0.024 grdn:0.174 lr:2.5e-06 ..."
# 注意：tqdm 不换行就刷新，logging 的 INFO 常常直接黏在进度行尾巴上，所以这里不能锚定行首。
RE_INFO = re.compile(
    r"INFO (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \S+ "
    r"step:\S+ .*?loss:(?P<loss>[0-9.eE+-]+) "
    r"grdn:(?P<grdn>[0-9.eE+-]+) lr:(?P<lr>[0-9.eE+-]+) "
    r"updt_s:(?P<updt>[0-9.eE+-]+) data_s:(?P<data>[0-9.eE+-]+)"
)


def parse(log_path: Path, span: int = 500) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """返回 (逐步 loss 表, 每 log_freq 步的明细表, 计划总步数)。"""
    text = log_path.read_bytes().decode("utf-8", "replace")
    lines = text.replace("\r", "\n").split("\n")

    step_rows: list[tuple[int, float]] = []  # step, loss
    info_rows: list[dict] = []
    cur_step, total = 0, 0

    for line in lines:
        m = RE_TQDM.search(line)
        if m:
            cur_step, total = int(m.group(1)), int(m.group(2))
            # tqdm 每步会刷新多次，同一 step 只保留最后一条（loss 已按窗口平均）
            if step_rows and step_rows[-1][0] == cur_step:
                step_rows[-1] = (cur_step, float(m.group(3)))
            else:
                step_rows.append((cur_step, float(m.group(3))))

        m = RE_INFO.search(line)
        if m:
            info_rows.append(
                {
                    "timestamp": pd.to_datetime(m.group("ts")),
                    "step": cur_step,
                    "loss": float(m.group("loss")),
                    "grdn": float(m.group("grdn")),
                    "lr": float(m.group("lr")),
                    "updt_s": float(m.group("updt")),
                    "data_s": float(m.group("data")),
                }
            )

    steps = pd.DataFrame(step_rows, columns=["step", "loss"])
    if not steps.empty:
        steps["loss_ema"] = steps["loss"].ewm(span=span, adjust=False).mean()
    info = pd.DataFrame(info_rows)
    return steps, info, total


def summarize(info: pd.DataFrame, total: int) -> None:
    """按百分比刻度打印 loss 采样，方便直接看收敛情况。"""
    if info.empty:
        return
    marks = [int(total * p) for p in (0.01, 0.05, 0.1, 0.2, 0.33, 0.5, 0.67, 0.8, 0.9, 1.0)]
    print(f"\n=== loss 采样（总 {total:,} 步）===")
    print(f"{'step':>8}{'loss':>10}{'grdn':>10}{'lr':>12}")
    for k in marks:
        near = info.iloc[(info["step"] - k).abs().argmin()]
        print(f"{int(near['step']):>8}{near['loss']:>10.4f}{near['grdn']:>10.4f}{near['lr']:>12.3e}")


def plot(steps: pd.DataFrame, info: pd.DataFrame, total: int, out: Path, job: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    ax = axes[0]
    ax.plot(steps["step"], steps["loss"], lw=0.4, color="#9ecae1", alpha=0.7, label="loss (per step)")
    ax.plot(steps["step"], steps["loss_ema"], lw=1.8, color="#08519c", label="loss (EMA span=500)")
    ax.set_yscale("log")
    ax.set_ylabel("loss (log)")
    ax.set_title(f"{job} — {total:,} steps")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    if not info.empty:
        ax.plot(info["step"], info["lr"], lw=1.2, color="#d94801")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    ax = axes[2]
    if not info.empty:
        ax.plot(info["step"], info["grdn"], lw=0.6, color="#31a354", alpha=0.8)
    ax.set_ylabel("grad norm")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    for a in axes:
        a.set_xlim(0, total)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="从 lerobot-train 日志画训练曲线")
    ap.add_argument("log", nargs="?", type=Path, help="日志路径；缺省取 logs/ 下最新的 .log")
    ap.add_argument("--save", type=Path, help="输出 PNG 路径；缺省与日志同名加 .png")
    ap.add_argument("--span", type=int, default=500, help="loss EMA 平滑窗口（默认 500 步）")
    args = ap.parse_args()

    log_path = args.log or max(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not log_path.exists():
        raise SystemExit(f"日志不存在: {log_path}")
    out = args.save or log_path.with_suffix(".png")

    steps, info, total = parse(log_path, args.span)
    total = total or int(steps["step"].max())
    print(f"日志: {log_path}")
    print(f"解析到 {len(steps):,} 条逐步记录、{len(info):,} 条 log_freq 明细（计划 {total:,} 步）")
    summarize(info, total)
    plot(steps, info, total, out, log_path.stem)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()
