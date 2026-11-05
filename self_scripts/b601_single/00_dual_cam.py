#!/usr/bin/env python3
"""两路相机同时打开（kfzbox 当前插了 hand + top 两路 JYU2C）

独立小工具，不依赖也不修改仓库里任何其它脚本。

用法（在仓库根目录）:
  python self_scripts/b601_single/00_dual_cam.py                # 同时开两路，各存一帧
  python self_scripts/b601_single/00_dual_cam.py --seconds 5    # 连续读 5 秒，报实际帧率
  python self_scripts/b601_single/00_dual_cam.py --preview      # 实时预览（需带 GUI 的 OpenCV）
  python self_scripts/b601_single/00_dual_cam.py --hand 2609010 --top 2610057
  python self_scripts/b601_single/00_dual_cam.py --sequential   # 强制一路一路抓（总线带宽不够时用）
  python self_scripts/b601_single/00_dual_cam.py --width 1280 --height 720 --preview   # 720p 预览
  python self_scripts/b601_single/00_dual_cam.py --width 1920 --height 1080            # 1080p 抓帧

默认就是 1280x720（MJPG 30fps）。注意 V4L2 不是任意分辨率都能出：申请不到时驱动会
回落到最接近的合法模式，脚本会把"实际"分辨率打出来；如果实际值和你要的不一致，
用 --list-modes 看这台相机到底支持哪些模式（相机不支持的分辨率强行设置是没用的）。

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
    # 关键：BUFFERSIZE=1 会让每路帧率直接砍半（UVC 只有 1 个缓冲，读一帧丢一帧），
    # 实测 720p 单路 15.2 fps vs 缓冲 4 的 29.7 fps。别为了"低延迟"设成 1。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
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


def list_modes(path: str) -> None:
    """打印相机支持的分辨率/帧率（优先 v4l2-ctl，没有就用 OpenCV 探测常见档位）"""
    print(f"\n--- {short(path)} 支持的格式 ---")
    try:
        out = subprocess.run(["v4l2-ctl", "--list-formats-ext", "-d", path],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            print(out.stdout.strip())
            return
    except Exception:
        pass

    probes = [(3840, 2160), (2560, 1440), (1920, 1080), (1600, 1200), (1280, 960),
              (1280, 720), (1024, 768), (800, 600), (640, 480), (352, 288)]
    for w, h in probes:
        cap = open_cam(path, w, h, 30, "MJPG")
        if cap is None:
            print("   ❌ 打不开，跳过")
            return
        f = grab(cap, tries=3)
        aw, ah = (f.shape[1], f.shape[0]) if f is not None else (0, 0)
        cap.release()
        mark = "✅" if (aw, ah) == (w, h) else ("~" if f is not None else "❌")
        print(f"   {mark} 申请 {w}x{h} -> 实际 {aw}x{ah}")
    print("   （~ 表示相机不支持该分辨率，驱动回落到了最接近的模式）")


def check_res(role: str, path: str, want: tuple[int, int], got: tuple[int, int]) -> None:
    if want == got:
        return
    print(f"   ⚠️  {role} 申请 {want[0]}x{want[1]}，相机实际给的是 {got[0]}x{got[1]}"
          f"（这台不支持该模式，看 --list-modes）")


def usb_path(dev: str) -> str:
    """相机在 USB 树上的位置，用于解释带宽冲突（换口时看这个）"""
    try:
        v4l = os.path.realpath(dev)
        node = os.path.basename(v4l)
        sysfs = os.path.realpath(f"/sys/class/video4linux/{node}/device")
        return sysfs.replace("/sys/devices/", "").split(":1.")[0]
    except Exception:
        return "?"


def bandwidth_errors() -> list[str]:
    """dmesg 里的带宽类报错（历史 + 本次）；用于前后对比，避免把旧报错误报成当前故障"""
    try:
        out = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return []
    return [l for l in out.splitlines()
            if "Not enough bandwidth" in l or "No space left" in l]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="", help="hand 相机序列号片段（留空=取第 1 路）")
    ap.add_argument("--top", default="", help="top 相机序列号片段（留空=取第 2 路）")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--seconds", type=float, default=0.0, help="连续读多少秒（0=只抓一帧）")
    ap.add_argument("--preview", action="store_true", help="开窗口实时预览（q/ESC 退出）")
    ap.add_argument("--sequential", action="store_true", help="一路一路抓，不强求同时")
    ap.add_argument("--list-modes", action="store_true", help="列出相机支持的分辨率/帧率后退出")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    devs = list_cams()
    if not devs:
        print("❌ 没有 /dev/v4l/by-id/*video-index0")
        return 1
    dmesg_base = len(bandwidth_errors())   # 本次运行前的历史报错条数，用来只报新增
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

    if a.list_modes:
        for _, path in [("hand", hand_path), ("top", top_path)]:
            if path:
                list_modes(path)
        return 0

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
                check_res(role, path, (a.width, a.height), (w, h))
                frames[role] = (f, path)

        # 持续读帧：真实采集看的不是"能打开"，而是"同时开还能一直出帧"
        # 每路一个线程，避免串行 read 互相拖累（串行会把帧率算成最慢那路的一半）
        if a.seconds > 0 and not degraded and len(caps) == len(targets):
            import threading
            print(f"\n连续读 {a.seconds:g} 秒（每路一个读线程，模拟真实采集）...")
            n = {r: 0 for r in caps}
            dead = set()
            stop = threading.Event()
            lock = threading.Lock()

            def reader(role, cap, path):
                while not stop.is_set():
                    ok, f = cap.read()
                    if ok and f is not None:
                        with lock:
                            n[role] += 1
                            frames[role] = (f, path)      # 留最后一帧存盘
                    else:
                        dead.add(role)
                        time.sleep(0.005)

            t0 = time.perf_counter()
            threads = [threading.Thread(target=reader, args=(r, c, p), daemon=True)
                       for r, (c, p) in caps.items()]
            for t in threads:
                t.start()
            time.sleep(a.seconds)
            stop.set()
            for t in threads:
                t.join(timeout=2)
            dt = time.perf_counter() - t0
            for role in caps:
                obs = n[role] / dt if dt else 0
                mark = "✅" if obs >= a.fps * 0.8 else ("❌" if n[role] == 0 else "⚠️")
                print(f"   {mark} {role:5s} {n[role]:4d} 帧 / {dt:.1f}s = {obs:5.1f} fps（目标 {a.fps}）")
            if dead:
                degraded = True
                print(f"   ⚠️ 中途读帧失败: {sorted(dead)}")
            low = [r for r in caps if n[r] / dt < a.fps * 0.8]
            if low:
                print("   注：已经排除过的原因（本机实测 720p 两路同进程 30.5/29.7 fps）：")
                print("       ① 不是 USB 带宽：两路 720p 同开、拆两个进程也都是 ~30 fps；")
                print("       ② 不是同进程 cv2 互斥：真正的杀手是 CAP_PROP_BUFFERSIZE=1（丢一半帧）。")
                print("       若仍不达标，先确认这行没被改回 1，再单开一路测该分辨率的上限。")

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
            check_res(role, path, (a.width, a.height), (w, h))
            frames[role] = (f, path)

    if not frames:
        print("\n❌ 两路都不出帧，先查线/供电：lsusb | grep -i joyand, ls /dev/v4l/by-id/")
        return 1

    # ---------- 存帧 + 拼对比图 ----------
    for role, (f, path) in frames.items():
        vis = f.copy()
        s = max(0.6, f.shape[1] / 1000)          # 高分辨率下图注按宽度放大
        cv2.putText(vis, f"{role}  {short(path)}  {f.shape[1]}x{f.shape[0]}", (10, int(30 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, s, (0, 255, 0), 2, cv2.LINE_AA)
        out = os.path.join(a.out, f"{role}_{short(path)}.jpg")
        cv2.imwrite(out, vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)   # 用原始帧统计，别被我画上去的绿字污染
        mean, peak = float(g.mean()), int(g.max())
        print(f"   存帧 {role:5s} -> {out}   画面均值={mean:5.1f} 峰值={peak}")
        if peak <= 2:
            print(f"   ⚠️  {role} 出来的几乎是纯黑帧（峰值 {peak}）——镜头盖没摘/被挡/传感器或排线问题，不是亮度不够")
        elif mean < 20:
            print(f"   ⚠️  {role} 画面很暗（均值 {mean:.1f}）——补光或调 exposure")

    if len(frames) == 2:
        import numpy as np
        order = [r for r in ("hand", "top") if r in frames]
        s = max(0.6, a.width / 1000)
        imgs = []
        for r in order:
            f = frames[r][0]
            f = cv2.resize(f, (a.width, a.height))
            cv2.putText(f, f"{r} {short(frames[r][1])}", (10, int(32 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, s, (0, 255, 0), 2, cv2.LINE_AA)
            imgs.append(f)
        side = np.hstack(imgs)
        p = os.path.join(a.out, "compare_hand_top.jpg")
        cv2.imwrite(p, side, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print(f"   对比图 -> {p}（左={order[0]} 右={order[1]}）")

    # ---------- 可选：实时预览 ----------
    if a.preview:
        try:
            caps, sizes = {}, {}
            for role, path in targets:
                cap = open_cam(path, a.width, a.height, a.fps, a.fourcc)
                if cap is None:
                    continue
                f = grab(cap)
                if f is not None:
                    caps[role] = cap
                    sizes[role] = f.shape[:2]
            if not caps:
                print("⚠️  预览：没有可用相机")
            else:
                for role, (h, w) in sizes.items():
                    cv2.namedWindow(role, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(role, w, h)
                t0 = time.perf_counter()
                n = {r: 0 for r in caps}
                while True:
                    for role, cap in caps.items():
                        ok, f = cap.read()
                        if ok and f is not None:
                            n[role] += 1
                            scale = max(0.6, f.shape[1] / 1000)   # 720p 时字放大，别看不清
                            cv2.putText(f, f"{role} #{n[role]}  {f.shape[1]}x{f.shape[0]}",
                                        (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                                        scale, (0, 255, 0), 2, cv2.LINE_AA)
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

    # ---------- 带宽/诊断（只在本次真的失败时才报，且只列本次新增的 dmesg） ----------
    errs = bandwidth_errors()[dmesg_base:]
    if degraded:
        print("\n诊断（本次运行新出现的带宽报错）：")
        print(f"   hand 在 USB 树上的位置: {usb_path(hand_path) if hand_path else '-'}")
        print(f"   top  在 USB 树上的位置: {usb_path(top_path) if top_path else '-'}")
        for e in errs:
            print(f"   dmesg: {e}")
        if not errs:
            print("   dmesg 无新增带宽报错，可能是相机/线材本身问题（换线、换口各试一次）")
        print("   若两路挂在同一条 USB2 集线器链上（1-10.x.x 这种），同时开就抢同一份等时带宽。")
        print("   解法：把其中一路换到机箱另一个 USB 口（别再共用同一个扩展坞/hub），")
        print("        然后重跑本脚本；仍不行就降分辨率/帧率（--width 352 --height 288 --fps 15）。")
        print(f"   注意：分辨率越高越吃带宽，{a.width}x{a.height}@30 两路同开对 USB2 往往顶不住。")
    else:
        print("   同时开两路：正常，没有新增带宽报错（这台不用再动插口）")

    print(f"\n完成：{len(frames)}/2 路出帧。核对画面归属后把序列号填进 "
          f"self_scripts/b601_common/devices/kfzbox.sh 的 B601_CAM_HAND / B601_CAM_TOP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
