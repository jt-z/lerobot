#!/usr/bin/env python3
"""同一场景下，用 hand / top 两路相机各拍一组不同分辨率的照片，方便肉眼对比。

独立小工具，不依赖也不修改仓库里任何其它脚本。

用法（在仓库根目录）:
  python self_scripts/b601_single/00b_res_compare.py
      # 默认拍 640x480 / 800x600 / 1024x576 / 1280x720 / 1920x1080，
      # 每路每档存一张无损 PNG + 一张 1:1 中心裁切（看真实细节）

  python self_scripts/b601_single/00b_res_compare.py --res 640x480,1280x720
  python self_scripts/b601_single/00b_res_compare.py --crop 800x450      # 改中心裁切大小
  python self_scripts/b601_single/00b_res_compare.py --crop 0x0          # 不裁切
  python self_scripts/b601_single/00b_res_compare.py --hand 2610057 --top 2609010

相机默认取设备档案里的 B601_CAM_HAND / B601_CAM_TOP（本机档案 kfzbox.sh 已定义），
没设就按 /dev/v4l/by-id 的顺序取前两路。

输出（默认 self_scripts/_logs/cam_res_compare/<时间戳>/）:
  <role>_<实际分辨率>.png          原始帧，无损 PNG（不要用 JPG，压缩会污染分辨率对比）
  <role>_<实际分辨率>_crop1x1.png  中心 1:1 裁切（默认 640x360），像素级看细节
  compare_<role>_full.jpg          该路各档位缩放到同高后的拼图（看取景/视场）
  compare_<role>_crop1x1.png       该路各档位 1:1 裁切拼图（看清晰度差异）
  summary.txt                      本次的参数、各档位实际分辨率/文件大小/亮度

注意：相机是一路一路顺序抓的，抓完这段时间里别动场景/别挪相机，否则不同档位拍到的
不是同一个画面。每档只抓一帧（取最后一张非空帧），不追求帧率。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import cv2
import numpy as np

DEFAULT_OUT = os.path.expanduser("~/dev/lerobot/self_scripts/_logs/cam_res_compare")
DEFAULT_RES = "640x480,800x600,1024x576,1280x720,1920x1080"


def list_cams() -> list[str]:
    return sorted(glob.glob("/dev/v4l/by-id/*video-index0"))


def short(path: str) -> str:
    """by-id 文件名里取出序列号片段，例如 2610057"""
    base = os.path.basename(path).replace("-video-index0", "")
    return base.split("-")[-1] if "-" in base else base


def by_serial(serial: str) -> str | None:
    hits = sorted(glob.glob(f"/dev/v4l/by-id/*{serial}*video-index0"))
    return hits[0] if hits else None


def profile_serials() -> tuple[str, str]:
    """从 b601_common/devices/<host>.sh 里读 hand/top 序列号，避免默认按顺序取导致角色弄反"""
    host = os.environ.get("B601_DEVICE") or os.uname().nodename
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "b601_common", "devices", f"{host}.sh")
    try:
        with open(path, encoding="utf-8") as fh:
            txt = fh.read()
    except OSError:
        return "", ""
    vals = {}
    for role in ("hand", "top"):
        m = re.search(rf"^export\s+B601_CAM_{role.upper()}=.*$", txt, re.M)
        if m:
            nums = re.findall(r"\d{4,}", m.group(0))
            vals[role] = nums[0] if nums else ""
    return vals.get("hand", ""), vals.get("top", "")


def open_cam(path: str, w: int, h: int, fps: int, fourcc: str):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)   # 缓冲=1 会丢一半帧，抓帧也建议别设 1
    return cap


def grab(cap, tries: int = 20):
    """前几帧常是空/花屏或旧分辨率残留，多读几帧取最后一张非空的"""
    frame = None
    for _ in range(tries):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
        time.sleep(0.02)
    return frame


def center_crop(img, cw: int, ch: int):
    h, w = img.shape[:2]
    cw, ch = min(cw, w), min(ch, h)
    x, y = (w - cw) // 2, (h - ch) // 2
    return img[y:y + ch, x:x + cw]


def bar(width: int, text: str, height: int = 34):
    """白底黑字的标题条；贴在拼图上方，不污染对比像素本身"""
    b = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(b, text, (8, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return b


def label(img, text: str):
    """在缩略图上直接画字（缩略图本来就被缩放过，画字不影响分辨率对比）"""
    vis = img.copy()
    s = max(0.6, vis.shape[1] / 900)
    cv2.putText(vis, text, (10, int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                s, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def parse_res(spec: str) -> list[tuple[int, int]]:
    out = []
    for item in spec.replace(" ", "").split(","):
        if not item:
            continue
        try:
            w, h = item.lower().split("x")
            out.append((int(w), int(h)))
        except ValueError:
            raise SystemExit(f"❌ 分辨率写法不对: {item}（应形如 1280x720）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="", help="hand 相机序列号片段（默认读 B601_CAM_HAND）")
    ap.add_argument("--top", default="", help="top 相机序列号片段（默认读 B601_CAM_TOP）")
    ap.add_argument("--res", default=DEFAULT_RES, help=f"要拍的分辨率列表，逗号分隔（默认 {DEFAULT_RES}）")
    ap.add_argument("--crop", default="640x360", help="中心 1:1 裁切尺寸，0x0=不裁")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--out", default="", help="输出目录（默认 _logs/cam_res_compare/<时间戳>）")
    a = ap.parse_args()

    devs = list_cams()
    if not devs:
        print("❌ 没有 /dev/v4l/by-id/*video-index0")
        return 1

    prof_hand, prof_top = profile_serials()
    hand_serial = a.hand or os.environ.get("B601_CAM_HAND", "") or prof_hand
    top_serial = a.top or os.environ.get("B601_CAM_TOP", "") or prof_top
    hand_path = by_serial(hand_serial) if hand_serial else (devs[0] if devs else None)
    top_path = by_serial(top_serial) if top_serial else (devs[1] if len(devs) > 1 else None)
    if hand_serial and not hand_path:
        print(f"❌ 找不到 hand 相机 {hand_serial}")
        return 1
    if top_serial and not top_path:
        print(f"❌ 找不到 top 相机 {top_serial}")
        return 1
    if not hand_serial or not top_serial:
        print("⚠️  没读到设备档案里的相机序列号，按 /dev/v4l/by-id 顺序取前两路（角色可能弄反，"
              "可用 --hand/--top 指定）")

    targets = [(r, p) for r, p in (("hand", hand_path), ("top", top_path)) if p]
    modes = parse_res(a.res)
    cw, ch = parse_res(a.crop)[0] if a.crop and "x" in a.crop else (0, 0)
    outdir = a.out or os.path.join(DEFAULT_OUT, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)

    print(f"输出目录：{outdir}")
    print(f"要拍的档位：{ ' / '.join(f'{w}x{h}' for w, h in modes) }   编码 {a.fourcc} @{a.fps}fps")
    for role, path in targets:
        print(f"   {role:5s} -> {short(path)}   {path}")
    print("\n⚠️  抓图期间不要动场景/挪相机，否则各档位不是同一个画面\n")

    lines = []
    crops: dict[str, list[tuple[str, np.ndarray]]] = {}
    fulls: dict[str, list[tuple[str, np.ndarray]]] = {}

    for role, path in targets:
        for w, h in modes:
            cap = open_cam(path, w, h, a.fps, a.fourcc)
            if cap is None:
                print(f"❌ {role:5s} 打不开: {path}")
                continue
            f = grab(cap)
            cap.release()
            if f is None:
                print(f"❌ {role:5s} {w}x{h} 读不到帧")
                continue

            ah, aw = f.shape[:2]
            tag = f"{role}_{aw}x{ah}"
            full_p = os.path.join(outdir, f"{tag}.png")
            cv2.imwrite(full_p, f)
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            mean, peak = float(g.mean()), int(g.max())
            kb = os.path.getsize(full_p) / 1024
            note = "" if (aw, ah) == (w, h) else f"  ⚠️ 相机不支持 {w}x{h}，回落到 {aw}x{ah}"
            print(f"✅ {role:5s} 申请 {w}x{h:<5} -> 实际 {aw}x{ah}   {kb:7.0f} KB  均值={mean:5.1f} 峰值={peak}{note}")
            lines.append(f"{role}\t申请 {w}x{h}\t实际 {aw}x{ah}\t{os.path.basename(full_p)}\t{kb:.0f} KB\t均值 {mean:.1f}\t峰值 {peak}")

            fulls.setdefault(role, []).append((f"{aw}x{ah}", f))
            if cw and ch:
                c = center_crop(f, cw, ch)
                crop_p = os.path.join(outdir, f"{tag}_crop1x1.png")
                cv2.imwrite(crop_p, c)
                crops.setdefault(role, []).append((f"{aw}x{ah}", c))

    if not fulls:
        print("\n❌ 一张都没抓到，先跑 00_dual_cam.py 确认相机能出帧")
        return 1

    # ---- 拼图：同一路相机的各档位放在一起 ----
    for role, items in fulls.items():
        H = 480
        thumbs = [label(cv2.resize(f, (round(f.shape[1] * H / f.shape[0]), H), interpolation=cv2.INTER_AREA),
                        f"{r}  {f.shape[1]}x{f.shape[0]}")
                  for r, f in items]
        p = os.path.join(outdir, f"compare_{role}_full.jpg")
        cv2.imwrite(p, np.hstack(thumbs), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"   取景对比 -> {p}（{' / '.join(r for r, _ in items)}，已缩放到同高）")

    for role, items in crops.items():
        # cv2.putText 不支持中文，拼图上的文字一律用 ASCII
        panels = [np.vstack([bar(c.shape[1], f"{r}  1:1 center crop {c.shape[1]}x{c.shape[0]}"), c])
                  for r, c in items]
        p = os.path.join(outdir, f"compare_{role}_crop1x1.png")
        cv2.imwrite(p, np.hstack(panels))
        print(f"   细节对比 -> {p}（像素 1:1，看清晰度差异）")

    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"编码: {a.fourcc} @{a.fps}fps   裁切: {cw}x{ch}\n")
        fh.write("\n".join(lines) + "\n")

    print(f"\n完成，全部文件在：{outdir}")
    print("对比建议：compare_<role>_full.jpg 看取景是否一致（各档已缩放到同高）；")
    print("         compare_<role>_crop1x1.png 是各档中心 1:1 裁切，同一裁切尺寸下高分辨率")
    print("         覆盖的实际范围更小、相当于更贴近中心，重点看文字/边缘的清晰度。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
