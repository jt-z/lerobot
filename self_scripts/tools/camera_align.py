#!/usr/bin/env python
"""相机对齐工具：把当前实拍画面实时叠加到"数据集零位参考图"上，方便边调相机边看。

模型是对着训练数据当时拍到的画面学的；相机被碰过/挪过就会抓空。本工具实时显示
四联画面（参考 | 当前 | 50%混合 | 红绿边缘叠加）并给出量化对齐指标，改一下相机
就能立刻看到效果。

用法（在 lerobot conda 环境里执行）：
    python tool/camera_align.py                    # 打开实时窗口（默认三路相机）
    python tool/camera_align.py --cams hand top     # 只开这两路
    python tool/camera_align.py --rebuild-ref      # 重建基准图（末段视频文件的零位帧中位数）
    python tool/camera_align.py --ref-episode 120   # 改用某一集的零位帧做参考
    python tool/camera_align.py --no-gui           # 无窗口：抓一帧、存图、打印指标，回车重拍
    python tool/camera_align.py --export-ref       # 只导出/重建基准图，不碰相机

基准图（默认，自动生成到 logs/camera_align/ref/ref_auto_<cam>.png）：
    只取数据集**最后 TAIL_VIDEO_FILES 个视频文件**（≈最后 20 多集）里的"零位帧"（每集第 0 帧
    state≈0），再按 front 相机的互相一致性锚定 + 剔除离群，最后逐像素取中位数合成。
    原因：采集期间相机/布局动过，早期帧（如 ep000）和后期帧对不上（NCC 只有 0.3~0.4，
    后期彼此 0.7~0.8），用后期这一致的一组做基准才代表"模型训练时最后的样子"。

实时窗口按键：1/2/3 切换相机 | s 保存当前四联图 | q 或 Esc 退出

指标：边缘图 FFT 互相关得到的最佳平移 (dx,dy) px 与对齐后 NCC。
      相机位置正确时应满足 |dx|,|dy| <= 5 px 且 NCC >= 0.6；偏了就按红绿边缘图里
      红/绿分离的方向微调相机（重合处是黄色）。

输出：logs/camera_align/ref/ref_ep<NNN>_<cam>.png   参考图（首次自动从数据集导出）
      logs/camera_align/lowgui_<cam>_r<NN>.png      无窗口模式每次抓帧的四联图
      logs/camera_align/live_<cam>_<时间>.png        实时窗口里按 s 保存的四联图

注意：拍摄时机械臂必须在零位（与参考帧同姿态），否则腕部相机那路没意义。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

PAI0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(PAI0, "b601_data", "b601_20260910_164106")
OUT_DIR = os.path.join(PAI0, "logs", "camera_align")
REF_DIR = os.path.join(OUT_DIR, "ref")
CAMS = ["hand", "front", "top"]
# 与 run_inference_b601_make_coffee_ACT_50k.sh 的 CAMERAS 保持一致
BY_ID = {
    "hand": "/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0",
    "front": "/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0",
    "top": "/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0",
}
VIDEO_NODES = ["/dev/video0", "/dev/video2", "/dev/video4"]
SCALE = 2          # 下采样倍数（只影响相关计算速度）
MAX_SHIFT_PX = 80  # 平移搜索范围（原始像素）
OK_SHIFT_PX, OK_NCC = 5, 0.60
PANEL_NAMES = ["参考(数据集零位)", "当前实拍", "50% 混合", "红绿边缘(重合=黄)"]


# ---------------------------------------------------------------- 参考图
TAIL_VIDEO_FILES = 3   # 用数据集最后 N 个视频文件里的零位帧建基准（视频文件越靠后越接近当前状态）
REF_N_MAX = 6          # 参与中位数合成的零位帧上限
REF_NCC_MIN = 0.55     # 与锚定帧的 NCC 低于此值视为"当时布局又变了"，剔除


def ref_path(episode: int, cam: str) -> str:
    return os.path.join(REF_DIR, f"ref_ep{episode:03d}_{cam}.png")


def ref_auto_path(cam: str) -> str:
    return os.path.join(REF_DIR, f"ref_auto_{cam}.png")


def export_ref(dataset: str, episode: int, cams: list[str]) -> None:
    """从数据集里取某集第 0 帧（零位）的三路画面作为参考图。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    os.makedirs(REF_DIR, exist_ok=True)
    ds = LeRobotDataset(repo_id=f"local/{os.path.basename(dataset)}", root=dataset, episodes=[episode])
    row = ds[0]
    state = [float(v) for v in row["observation.state"]]
    print(f"[ref] ep{episode} 第 0 帧 state = " + " ".join(f"{v:.1f}" for v in state))
    if max(abs(v) for v in state[:6]) > 5:
        print("      ⚠️  该帧不在零位，参考图可能不可用，请换 --ref-episode")
    for cam in cams:
        img = (row[f"observation.images.{cam}"].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(img).save(ref_path(episode, cam))
    print(f"      参考图已写入 {REF_DIR}")


def _episodes_table(dataset: str):
    import pandas as pd

    files = sorted(__import__("glob").glob(f"{dataset}/meta/episodes/**/*.parquet", recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _zero_pose_starts(dataset: str, tol: float = 5.0):
    """所有 episode 的第 0 帧里状态≈0（零位）的那些。"""
    import glob

    import pandas as pd

    rows = []
    cols = ["episode_index", "frame_index", "timestamp", "observation.state"]
    for f in sorted(glob.glob(f"{dataset}/data/**/*.parquet", recursive=True)):
        df = pd.read_parquet(f, columns=cols)
        s = df[df.frame_index == 0]
        if not len(s):
            continue
        st = np.stack(s["observation.state"].to_numpy())
        ok = np.all(np.abs(st[:, :6]) <= tol, axis=1)
        gr = st[:, 6]
        ok &= (gr <= tol * 2.4) | (gr >= 360 - tol * 2.4)
        rows.append(s[ok])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def build_auto_ref(dataset: str, cams: list[str]) -> list[int]:
    """用"靠后的那几个视频文件"里的零位帧，取互相一致的一组做中位数基准图。

    数据集在采集期间相机/布局动过，早期帧和后期帧对不上；越靠后的文件越接近当前状态，
    所以只在最后 TAIL_VIDEO_FILES 个视频文件里挑，并要求彼此 NCC 一致（剔除离群）。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    os.makedirs(REF_DIR, exist_ok=True)
    starts = _zero_pose_starts(dataset)
    etab = _episodes_table(dataset)
    fkey = "videos/observation.images.front/file_index"
    fmax = int(etab[fkey].max())
    cand = etab.merge(starts[["episode_index"]], on="episode_index")[["episode_index", fkey]]
    cand = cand[cand[fkey] >= fmax - TAIL_VIDEO_FILES + 1].sort_values(fkey)
    if not len(cand):
        print(f"[ref] ⚠️  最后 {TAIL_VIDEO_FILES} 个视频文件里没有零位帧，改用全部零位帧的最后一集")
        cand = etab.merge(starts[["episode_index"]], on="episode_index")[["episode_index", fkey]].sort_values(fkey).tail(1)
    eps = [int(e) for e in cand["episode_index"]]
    files = [int(v) for v in cand[fkey]]
    print(f"[ref] 候选零位帧（末 {TAIL_VIDEO_FILES} 个视频文件，file_index >= {fmax - TAIL_VIDEO_FILES + 1}）: "
          f"ep{eps}")

    frames: dict[int, dict[str, Image.Image]] = {}
    for ep in eps:
        ds = LeRobotDataset(repo_id=f"local/{os.path.basename(dataset)}", root=dataset, episodes=[ep])
        row = ds[0]
        frames[ep] = {
            cam: Image.fromarray((row[f"observation.images.{cam}"].numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            for cam in cams
        }

    # 以 front 相机互相最一致的那一帧为锚，剔除离群（当时的相机/布局又动过）
    ref_cam = "front" if "front" in cams else cams[0]
    eds = {ep: edges(gray(frames[ep][ref_cam])) for ep in eps}
    keep = eps
    if len(eps) > 1:
        sims = {ep: [ncc(eds[ep], eds[o]) for o in eps if o != ep] for ep in eps}
        anchor = max(eps, key=lambda e: float(np.median(sims[e])))
        keep = [e for e in eps if e == anchor or ncc(eds[anchor], eds[e]) >= REF_NCC_MIN]
        if len(keep) > REF_N_MAX:
            keep = sorted(keep, key=lambda e: -ncc(eds[anchor], eds[e]))[:REF_N_MAX]
        keep.sort()
        print(f"[ref] 锚定 ep{anchor}；保留与它一致的零位帧: ep{keep}")
        for e in eps:
            mark = "*" if e == anchor else ("o" if e in keep else "x")
            print(f"       {mark} ep{e:03d} (file-{files[eps.index(e)]:03d}) NCC(锚)={ncc(eds[anchor], eds[e]):.3f}")
    else:
        print(f"[ref] 只有一个候选帧 ep{eps[0]}")

    for cam in cams:
        stack = np.stack([np.asarray(frames[e][cam].convert("RGB")).astype(np.float32) for e in keep])
        Image.fromarray(np.median(stack, axis=0).astype(np.uint8)).save(ref_auto_path(cam))
        for e in keep:  # 顺便把各帧也存成单集参考，方便 --ref-episode 对比
            frames[e][cam].save(ref_path(e, cam))
    print(f"[ref] 基准图（{len(keep)} 帧中位数）已写入 {REF_DIR}/ref_auto_*.png")
    return keep


def ensure_ref(dataset: str, cams: list[str], episode: int | None, rebuild: bool) -> dict[str, str]:
    """决定用哪套参考图；返回 {cam: 路径}。"""
    if episode is not None:
        missing = [c for c in cams if not os.path.exists(ref_path(episode, c))]
        if missing:
            print(f"[ref] 缺少 ep{episode} 的 {missing} 参考图，从数据集导出 ...")
            export_ref(dataset, episode, cams)
        return {c: ref_path(episode, c) for c in cams}

    if rebuild or any(not os.path.exists(ref_auto_path(c)) for c in cams):
        build_auto_ref(dataset, cams)
    return {c: ref_auto_path(c) for c in cams}


# ---------------------------------------------------------------- 图像指标 / 叠加
def gray(a: Image.Image) -> np.ndarray:
    return np.asarray(a.convert("L"), dtype=np.float32)


def edges(g: np.ndarray) -> np.ndarray:
    img = g[::SCALE, ::SCALE]
    gx, gy = np.gradient(img)
    return np.hypot(gx, gy)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def align_metrics(ref_edges: np.ndarray, now_edges: np.ndarray) -> tuple[int, int, float, float]:
    """返回 (dx, dy, NCC@最佳平移, NCC@0)：把当前图平移 (dx,dy) px 后与参考对齐。"""
    cc = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(ref_edges) * np.conj(np.fft.rfft2(now_edges)),
                                      s=ref_edges.shape))
    cy, cx = np.array(ref_edges.shape) // 2
    m = max(1, MAX_SHIFT_PX // SCALE)
    win = cc[cy - m:cy + m + 1, cx - m:cx + m + 1]
    dy, dx = np.unravel_index(np.argmax(win), win.shape)
    dx, dy = dx - m, dy - m
    h, w = ref_edges.shape
    y0, y1, x0, x1 = max(0, dy), min(h, h + dy), max(0, dx), min(w, w + dx)
    if y1 - y0 < 16 or x1 - x0 < 16:
        return 2 * dx, 2 * dy, 0.0, 0.0
    aligned = ncc(ref_edges[y0:y1, x0:x1], now_edges[y0 - dy:y1 - dy, x0 - dx:x1 - dx])
    return 2 * dx, 2 * dy, aligned, ncc(ref_edges, now_edges)


def verdict(dx: int, dy: int, ns: float) -> str:
    ok = abs(dx) <= OK_SHIFT_PX and abs(dy) <= OK_SHIFT_PX and ns >= OK_NCC
    txt = "对齐良好" if ok else "有偏移"
    if abs(dx) >= MAX_SHIFT_PX or abs(dy) >= MAX_SHIFT_PX:
        txt += f"(平移已达搜索上限 ±{MAX_SHIFT_PX}px，差异很大)"
    return txt


def edge_overlay(ref_img: Image.Image, now_img: Image.Image) -> np.ndarray:
    """红=参考边缘、绿=当前边缘、黄=重合。"""
    re, ne = edges(gray(ref_img)), edges(gray(now_img))
    norm = max(re.max(), ne.max(), 1e-6)
    rgb = np.zeros((*re.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(re / norm * 255 * 2.0, 0, 255)
    rgb[..., 1] = np.clip(ne / norm * 255 * 2.0, 0, 255)
    return rgb


def panels(ref_img: Image.Image, now_img: Image.Image) -> list[np.ndarray]:
    """四联面板的 numpy 数组：参考 | 当前 | 混合 | 红绿边缘（后两张尺寸减半）。"""
    ref, now = ref_img.convert("RGB"), now_img.convert("RGB")
    return [np.asarray(ref), np.asarray(now),
            np.asarray(Image.blend(ref, now, 0.5)), edge_overlay(ref, now)]


def compose(plist: list[np.ndarray], size: tuple[int, int]) -> Image.Image:
    """把四联面板拼成一张图（边缘面板放大到同一尺寸）。"""
    w, h = size
    canvas = Image.new("RGB", (w * 4 + 30, h), (25, 25, 25))
    for i, arr in enumerate(plist):
        canvas.paste(Image.fromarray(arr).resize((w, h), Image.NEAREST), (i * (w + 10), 0))
    return canvas


# ---------------------------------------------------------------- 相机
def check_cameras_free() -> None:
    busy = subprocess.run(["fuser", *VIDEO_NODES], capture_output=True, text=True).stdout.strip()
    if busy:
        print(f"❌ 摄像头被占用（PID {busy}）—— 有 run 正在跑吗？停掉再调相机。")
        sys.exit(1)


def open_cameras(cams: list[str]) -> dict:
    """打开（并保持）相机句柄，前若干帧用于等自动曝光稳定。"""
    import cv2

    check_cameras_free()
    caps = {}
    for cam in cams:
        cap = cv2.VideoCapture(BY_ID[cam], cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"⚠️  {cam} 打开失败：{BY_ID[cam]}")
            continue
        for _ in range(20):
            cap.read()
        caps[cam] = cap
    return caps


def grab(cap) -> Image.Image | None:
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return Image.fromarray(frame[:, :, ::-1])


def load_refs(paths: dict[str, str]) -> tuple[dict[str, Image.Image], tuple[int, int] | None]:
    out: dict[str, Image.Image] = {}
    size = None
    for cam, p in paths.items():
        if os.path.exists(p):
            img = Image.open(p).convert("RGB")
            out[cam] = img
            size = img.size
    return out, size


# ---------------------------------------------------------------- 实时窗口
def run_gui(cams: list[str], paths: dict[str, str]) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "template"):
        raise RuntimeError(f"matplotlib 后端 {matplotlib.get_backend()} 不支持交互显示")
    # 中文字体：DejaVu Sans 没有中文字形，会显示成方框
    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK SC", "Droid Sans Fallback", "WenQuanYi Zen Hei", "Source Han Sans SC"):
        if cand in available:
            matplotlib.rcParams["font.family"] = [cand, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

    refs, size = load_refs(paths)
    cams = [c for c in cams if c in refs]
    if not cams or size is None:
        print("❌ 没有可用的参考图。")
        return
    caps = open_cameras(cams)
    cams = [c for c in cams if c in caps]
    if not cams:
        print("❌ 没有可用相机。")
        return

    state = {"cam": cams[0], "quit": False, "save": False}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    try:
        fig.canvas.manager.set_window_title("camera_align — 相机对齐（1/2/3 切相机，s 保存，q 退出）")
    except Exception:
        pass
    artists = []
    for ax, name in zip(axes.ravel(), PANEL_NAMES):
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        artists.append(ax.imshow(np.zeros((*size[::-1], 3), dtype=np.uint8), interpolation="nearest",
                                 aspect="auto"))
    suptitle = fig.suptitle("初始化 ...", fontsize=12)

    def on_key(ev):
        k = ev.key
        if k in ("q", "escape"):
            state["quit"] = True
        elif k in ("1", "2", "3"):
            idx = int(k) - 1
            if idx < len(cams):
                state["cam"] = cams[idx]
        elif k == "s":
            state["save"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    print(f"实时窗口已打开（相机：{'/'.join(cams)}）\n"
          f"  1/2/3 切换相机 | s 保存四联图 | q 或 Esc 退出\n"
          f"  判定标准：|平移| <= {OK_SHIFT_PX}px 且 NCC >= {OK_NCC}")
    t_last = 0.0
    try:
        while not state["quit"]:
            cam = state["cam"]
            now = grab(caps[cam])
            if now is None:
                plt.pause(0.05)
                continue

            ref = refs[cam]
            dx, dy, ns, n0 = align_metrics(edges(gray(ref)), edges(gray(now)))
            for art, arr in zip(artists, panels(ref, now)):
                art.set_data(arr)
            v = verdict(dx, dy, ns)
            suptitle.set_text(f"{cam}   平移=({dx:+d},{dy:+d})px   NCC={ns:.3f}   判定: {v}"
                              + ("" if v == "对齐良好" else f"  (需 |平移|<={OK_SHIFT_PX}px 且 NCC>={OK_NCC})"))
            suptitle.set_color("green" if v == "对齐良好" else "crimson")
            if state["save"]:
                dst = os.path.join(OUT_DIR, f"live_{cam}_{time.strftime('%H%M%S')}.png")
                compose(panels(ref, now), size).save(dst)
                print(f"  已保存 {dst}")
                state["save"] = False
            fig.canvas.draw_idle()
            plt.pause(0.02)
            if time.time() - t_last > 10:  # 心跳，避免看起来像卡死
                print(f"  [{cam}] 平移=({dx:+d},{dy:+d})px NCC={ns:.3f} {v}")
                t_last = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        for cap in caps.values():
            cap.release()
        plt.close(fig)
    print("已退出。")


# ---------------------------------------------------------------- 无窗口模式
def run_headless(cams: list[str], paths: dict[str, str], once: bool) -> None:
    caps = open_cameras(cams)
    refs, size = load_refs(paths)
    round_idx = 1
    try:
        while True:
            print(f"\n--- 第 {round_idx} 次抓帧 ---")
            print(f"  {'cam':6s} {'平移(dx,dy)px':>14s} {'NCC@平移':>9s} {'NCC@0':>8s}  判定")
            for cam in cams:
                if cam not in caps or cam not in refs:
                    continue
                now = grab(caps[cam])
                if now is None:
                    print(f"  {cam:6s} 抓帧失败")
                    continue
                ref = refs[cam]
                dx, dy, ns, n0 = align_metrics(edges(gray(ref)), edges(gray(now)))
                print(f"  {cam:6s} {f'({dx:+4d},{dy:+4d})':>14s} {ns:9.3f} {n0:8.3f}  {verdict(dx, dy, ns)}")
                dst = os.path.join(OUT_DIR, f"lowgui_{cam}_r{round_idx:02d}.png")
                compose(panels(ref, now), size).save(dst)
            print(f"  四联图已存到 {OUT_DIR}/lowgui_*_r{round_idx:02d}.png（第 4 格看红/绿分离方向）")
            if once:
                break
            if input("调好相机后回车重拍 / 输入 q 退出 > ").strip().lower() == "q":
                break
            round_idx += 1
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        for cap in caps.values():
            cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description="把当前相机画面实时叠加到数据集零位参考图，辅助调相机位置")
    ap.add_argument("--cams", nargs="+", default=CAMS, choices=CAMS, help="要处理的相机（默认全部）")
    ap.add_argument("--ref-episode", type=int, default=None,
                    help="用指定 episode 的第 0 帧做参考（默认用末段视频文件的零位帧中位数基准 ref_auto_*）")
    ap.add_argument("--ref-dataset", default=DEFAULT_DATASET, help="数据集根目录")
    ap.add_argument("--rebuild-ref", action="store_true", help="强制重建 ref_auto_* 基准图")
    ap.add_argument("--export-ref", action="store_true", help="只导出/重建参考图，不抓相机")
    ap.add_argument("--no-gui", action="store_true", help="不开实时窗口（存图 + 终端指标）")
    ap.add_argument("--once", action="store_true", help="无窗口模式下只抓一次就退出")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = ensure_ref(args.ref_dataset, args.cams, args.ref_episode, args.rebuild_ref)
    print("[ref] 参考图：" + "  ".join(f"{c}={os.path.relpath(p, PAI0)}" for c, p in paths.items()))
    if args.export_ref:
        return

    print("\n提示：请让机械臂处于零位（与参考帧同姿态）后再调整相机。")
    if args.no_gui:
        run_headless(args.cams, paths, args.once)
        return
    try:
        run_gui(args.cams, paths)
    except Exception as exc:
        print(f"⚠️  实时窗口不可用（{type(exc).__name__}: {exc}），改用无窗口模式。")
        run_headless(args.cams, paths, args.once)


if __name__ == "__main__":
    main()
