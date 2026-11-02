#!/usr/bin/env python3
"""两路相机同时打开（kfzbox 当前插了 hand + top 两路 JYU2C）

独立小工具，不依赖也不修改仓库里任何其它脚本。

用法（在仓库根目录）:
  python self_scripts/b601_single/00_dual_cam.py                # 同时开两路，各存一帧
  python self_scripts/b601_single/00_dual_cam.py --seconds 5    # 连续读 5 秒，报实际帧率
  python self_scripts/b601_single/00_dual_cam.py --preview      # 实时预览（需带 GUI 的 OpenCV）
  python self_scripts/b601_single/00_dual_cam.py --hand 2609010 --top 2610057
  python self_scripts/b601_single/00_dual_cam.py --sequential   # 强制一路一路抓（总线带宽不够时用）

同时打开失败时，脚本会自动退化成"顺序抓帧"：先抓 A 再抓 B，保证两路都验到，
并把两帧拼成一张对比图，方便判断哪路是 hand、哪路是 top。
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

import cv2

DEFAULT_OUT = os.path.expanduser("~/dev/lerobot/self_scripts/_logs/cam_dual")


def list_cams() -> list[str]:
    return sorted(glob.glob("/dev/v4l/by-id/*video-index0"))


def short(path: str) -> str:
    """by-id 文件名里取出序列号片段，例如 2609010"""
    base = os.path.basename(path).replace("-video-index0", "")
    return base.split("-")[-1] if "-" in base else base


def by_serial(serial: str) -> str | None:
    hits = sorted(glob.glob(f"/dev/v4l/by-id/*{serial}*video-index0"))
    return hits[0] if hits else None


def open_cam(path, w, h, fps, fourcc):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def grab(cap, tries=15):
    """前几帧常是空/花屏，取最后一帧非空的"""
    frame = None
    for _ in range(tries):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
        time.sleep(0.03)
    return frame


def usb_path(dev: str) -> str:
    """相机在 USB 树上的位置，用于解释带宽冲突（换口时看这个）"""
    try:
        v4l = os.path.realpath(dev)
        node = os.path.basename(v4l)
        sysfs = os.path.realpath(f"/sys/class/video4linux/{node}/device")
        return sysfs.replace("/sys/devices/", "").split(":1.")[0]
    except Exception:
        return "?"


def recent_bandwidth_errors(n=4) -> list[str]:
    try:
        out = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return []
    hits = [l for l in out.splitlines() if "Not enough bandwidth" in l or "No space left" in l]
    return hits[-n:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="", help="hand 相机序列号片段（留空=取第 1 路）")
    ap.add_argument("--top", default="", help="top 相机序列号片段（留空=取第 2 路）")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--seconds", type=float, default=0.0, help="连续读多少秒（0=只抓一帧）")
    ap.add_argument("--preview", action="store_true", help="开窗口实时预览（q/ESC 退出）")
    ap.add_argument("--sequential", action="store_true", help="一路一路抓，不强求同时")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    devs = list_cams()
    if not devs:
        print("❌ 没有 /dev/v4l/by-id/*video-index0")
        return 1
    print(f"发现 {len(devs)} 路相机：")
    for d in devs:
        print(f"    {d}   (序列号 {short(d)})")

    hand_path = by_serial(a.hand) if a.hand else (devs[0] if devs else None)
    top_path = by_serial(a.top) if a.top else (devs[1] if len(devs) > 1 else None)
    if a.hand and not hand_path:
        print(f"❌ 找不到 hand 序列号 {a.hand}")
        return 1
    if a.top and not top_path:
        print(f"❌ 找不到 top 序列号 {a.top}")
        return 1

    targets = [("hand", hand_path), ("top", top_path)]
    targets = [(r, p) for r, p in targets if p]
    os.makedirs(a.out, exist_ok=True)

    # ---------- 方式一：同时打开 ----------
    frames, degraded = {}, False
    if not a.sequential:
        caps = {}
        for role, path in targets:
            cap = open_cam(path, a.width, a.height, a.fps, a.fourcc)
            if cap is None:
                print(f"❌ {role:5s} VideoCapture 打不开: {path}")
                degraded = True
                continue
            caps[role] = (cap, path)
        time.sleep(0.4)
        for role, (cap, path) in caps.items():
            f = grab(cap)
            if f is None:
                print(f"❌ {role:5s} 打开成功但读不到帧（多半是 USB 带宽不够）: {short(path)}")
                degraded = True
            else:
                h, w = f.shape[:2]
                print(f"✅ {role:5s} 同时打开可用  {short(path)}  实际 {w}x{h}")
                frames[role] = (f, path)
        for cap, _ in caps.values():
            cap.release()

    # ---------- 方式二：顺序抓帧（同时不可行时自动退到这条） ----------
    if degraded or a.sequential:
        if degraded:
            print("\n→ 退化为顺序抓帧（同时开两路失败，原因见文末诊断）")
        for role, path in targets:
            if role in frames and not a.sequential:
                continue
            cap = open_cam(path, a.width, a.height, a.fps, a.fourcc)
            if cap is None:
                print(f"❌ {role:5s} 打不开: {path}")
                continue
            f = grab(cap)
            cap.release()
            if f is None:
                print(f"❌ {role:5s} 顺序模式也读不到帧: {path}")
                continue
            h, w = f.shape[:2]
            print(f"✅ {role:5s} 顺序抓帧成功  {short(path)}  实际 {w}x{h}")
            frames[role] = (f, path)

    if not frames:
        print("\n❌ 两路都不出帧，先查线/供电：lsusb | grep -i joyand, ls /dev/v4l/by-id/")
        return 1

    # ---------- 存帧 + 拼对比图 ----------
    for role, (f, path) in frames.items():
        vis = f.copy()
        cv2.putText(vis, f"{role}  {short(path)}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        out = os.path.join(a.out, f"{role}_{short(path)}.jpg")
        cv2.imwrite(out, vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print(f"   存帧 {role:5s} -> {out}")

    if len(frames) == 2:
        import numpy as np
        order = [r for r in ("hand", "top") if r in frames]
        imgs = []
        for r in order:
            f = frames[r][0]
            f = cv2.resize(f, (a.width, a.height))
            cv2.putText(f, f"{r} {short(frames[r][1])}", (8, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            imgs.append(f)
        side = np.hstack(imgs)
        p = os.path.join(a.out, "compare_hand_top.jpg")
        cv2.imwrite(p, side, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print(f"   对比图 -> {p}（左={order[0]} 右={order[1]}）")

    # ---------- 可选：实时预览 ----------
    if a.preview:
        try:
            caps = {}
            for role, path in targets:
                cap = open_cam(path, a.width, a.height, a.fps, a.fourcc)
                if cap is not None and grab(cap) is not None:
                    caps[role] = cap
            if not caps:
                print("⚠️  预览：没有可用相机")
            else:
                for role in caps:
                    cv2.namedWindow(role, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(role, a.width, a.height)
                t0 = time.perf_counter()
                n = {r: 0 for r in caps}
                while True:
                    for role, cap in caps.items():
                        ok, f = cap.read()
                        if ok and f is not None:
                            n[role] += 1
                            cv2.putText(f, f"{role} #{n[role]}", (8, 26),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.imshow(role, f)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
                dt = time.perf_counter() - t0
                for r in caps:
                    print(f"   预览 {r:5s} {n[r]} 帧 / {dt:.1f}s = {n[r]/dt:.1f} fps")
        except cv2.error as e:
            print(f"⚠️  这个 OpenCV 没编 GUI 支持，--preview 用不了（不影响抓帧）：{str(e).splitlines()[-1][:120]}")
        finally:
            cv2.destroyAllWindows()

    # ---------- 带宽/诊断 ----------
    errs = recent_bandwidth_errors()
    if degraded or errs:
        print("\n诊断：")
        print(f"   hand 在 USB 树上的位置: {usb_path(hand_path) if hand_path else '-'}")
        print(f"   top  在 USB 树上的位置: {usb_path(top_path) if top_path else '-'}")
        for e in errs:
            print(f"   dmesg: {e}")
        print("   若两路挂在同一条 USB2 集线器链上（1-10.x.x），同时开就抢同一份等时带宽。")
        print("   解法：把其中一路换到机箱另一个 USB 口（别共用同一个扩展坞/hub），")
        print("        然后重跑本脚本；仍不行就降分辨率/帧率（--width 352 --height 288 --fps 15）。")

    print(f"\n完成：{len(frames)}/2 路出帧。核对画面归属后把序列号填进 "
          f"self_scripts/b601_common/devices/kfzbox.sh 的 B601_CAM_HAND / B601_CAM_TOP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
