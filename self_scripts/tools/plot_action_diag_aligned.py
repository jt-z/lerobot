"""对齐左右臂 action_diag CSV，按时间绘制各关节轨迹（goal / sent / present）。

用法：
    uv run python self_scripts/tools/plot_action_diag_aligned.py \
        [left_csv] [right_csv] [--save path.png]

说明：
    - 左右 CSV 的采样时间戳略有差异，这里按 joint 用 merge_asof 就近对齐。
    - 每个 joint 一张图：3 行 (goal / sent / present) × 左臂(蓝实线) vs 右臂(橙虚线)。
    - 行数不足时差值列 (diff) 显示在对齐图上，便于观察左右差异。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_LEFT = Path(__file__).parent.parent / "_logs" / "inference_logs" / "action_diag" / "action_diag_jt_follower_arm_left.csv"
DEFAULT_RIGHT = Path(__file__).parent.parent / "_logs" / "inference_logs" / "action_diag" / "action_diag_jt_follower_arm_right.csv"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FIELDS = ["goal_pos", "sent_pos", "present_pos"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.astype({"timestamp": float})
    # 每个 tick 对 joint 排序，保证 merge_asof 前按 (joint, timestamp) 有序
    df = df.sort_values(["joint", "timestamp"]).reset_index(drop=True)
    return df


def align(left: pd.DataFrame, right: pd.DataFrame, tolerance: float = 0.1) -> pd.DataFrame:
    """按 joint 分别用 merge_asof 就近对齐左右臂，返回宽表。"""
    cols = ["timestamp", "joint"] + FIELDS + ["clamped"]
    out = pd.DataFrame()
    for joint in JOINTS:
        l = left[left["joint"] == joint][cols].rename(
            columns={c: f"left_{c}" for c in FIELDS} | {"timestamp": "ts_l", "clamped": "left_clamped"}
        )
        r = right[right["joint"] == joint][cols].rename(
            columns={c: f"right_{c}" for c in FIELDS} | {"timestamp": "ts_r", "clamped": "right_clamped"}
        )
        m = pd.merge_asof(l.sort_values("ts_l"), r.sort_values("ts_r"), left_on="ts_l", right_on="ts_r",
                          direction="nearest", tolerance=tolerance)
        m = m.drop(columns=["joint_x", "joint_y"])
        m.insert(0, "joint", joint)
        out = pd.concat([out, m], ignore_index=True)
    return out


def error_report(m: pd.DataFrame):
    """打印各关节控制指令(sent)与实际位置(present)的误差统计，判断是否跟踪一致。"""
    for side, pre in (("left_", "left_"), ("right_", "right_")):
        m[side + "sent_diff"] = m[pre + "sent_pos"] - m[pre + "present_pos"]
    print("\n=== 各关节 |控制指令-实际位置| 误差统计 (度) ===")
    print(f"{'joint':<14}{'side':<5}{'MAE_sent':>9}{'Max|sent|':>10}{'|sent|<=1':>10}{'|sent|<=2':>10}{'|sent|<=5':>10}")
    for joint, g in m.groupby("joint"):
        for side, pre in (("L", "left_"), ("R", "right_")):
            sd = g[pre + "sent_diff"].abs()
            print(f"{joint:<14}{side:<5}{sd.mean():>9.2f}{sd.max():>10.2f}"
                  f"{(sd <= 1).mean() * 100:>9.1f}{(sd <= 2).mean() * 100:>9.1f}{(sd <= 5).mean() * 100:>9.1f}")
    print("\n=== 总体(全部样本) ===")
    for side in ("left_", "right_"):
        sd = m[side + "sent_diff"].abs()
        print(f"{side:6}: MAE={sd.mean():.2f} 中位={sd.median():.2f} Max={sd.max():.2f} "
              f"|d|<=1°:{(sd <= 1).mean() * 100:.1f}%  <=2°:{(sd <= 2).mean() * 100:.1f}%  <=5°:{(sd <= 5).mean() * 100:.1f}%")


def plot(m: pd.DataFrame, save: Path | None):
    n_joints = len(JOINTS)
    fig, axes = plt.subplots(nrows=len(FIELDS), ncols=n_joints, figsize=(3.2 * n_joints, 3.4 * len(FIELDS)),
                             sharex=False, squeeze=False)
    colors = {"left": "tab:blue", "right": "tab:orange"}
    for i, joint in enumerate(JOINTS):
        d = m[m["joint"] == joint].sort_values("ts_l")
        t = (d["ts_l"] + d["ts_r"].fillna(d["ts_l"])) / 2 - d["ts_l"].min()  # 相对时间(秒)，忽略缺失
        for j, field in enumerate(FIELDS):
            ax = axes[j][i]
            for arm, col in (("left", "left_" + field), ("right", "right_" + field)):
                ax.plot(t, d[col], color=colors[arm], linestyle="-", linewidth=1.2,
                        label=arm if j == 0 else None)
            if j == 0:
                ax.set_title(joint)
            ax.set_ylabel(field.split("_")[0])
            ax.grid(alpha=0.3)
            if j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Aligned joint trajectories: left (blue) vs right (orange)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save:
        fig.savefig(save, dpi=150)
        print(f"saved: {save}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("left", nargs="?", default=DEFAULT_LEFT, type=Path)
    ap.add_argument("right", nargs="?", default=DEFAULT_RIGHT, type=Path)
    ap.add_argument("--save", type=Path, default=None, help="保存图片路径，不传则弹窗显示")
    ap.add_argument("--tolerance", type=float, default=0.1, help="时间对齐容差(秒)")
    args = ap.parse_args()

    m = align(load(args.left), load(args.right), tolerance=args.tolerance)
    print(f"aligned rows: {len(m)}  (left={len(load(args.left))}, right={len(load(args.right))})")
    print(f"joints: {m['joint'].unique().tolist()}")
    print(m.head(12).to_string(index=False))
    error_report(m)
    plot(m, args.save)


if __name__ == "__main__":
    main()
