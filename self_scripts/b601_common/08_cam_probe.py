#!/usr/bin/env python3
"""抓每个 /dev/v4l/by-id 相机一帧存 jpg, 便于人工/模型判断哪路是 hand(腕部)/front/top。

用法:  python self_scripts/b601_common/08_cam_probe.py [输出目录]
判据:  文件名带序号+序列号; 每张图上叠加了设备序列号字样, 便于与 05 脚本的 CAMERAS 对齐。
不写任何设备配置, 只做一次只读抓帧。
"""
from __future__ import annotations

import glob
import os
import sys
import time

import cv2

DEFAULT_OUT = os.path.expanduser("~/dev/lerobot/self_scripts/_logs/cam_probe")


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    devs = sorted(glob.glob("/dev/v4l/by-id/*video-index0"))
    if not devs:
        print("没有找到任何 /dev/v4l/by-id/*video-index0 —— 相机没插好")
        return 1
    print(f"发现 {len(devs)} 路相机, 输出到 {out_dir}")
    ok = 0
    for idx, dev in enumerate(devs):
        serial = os.path.basename(dev).replace("-video-index0", "")
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
        frame = None
        for _ in range(15):          # 前几帧常常是空/花屏
            ret, f = cap.read()
            if ret and f is not None:
                frame = f
            time.sleep(0.05)
        cap.release()
        if frame is None:
            print(f"[{idx}] FAIL 读不到帧: {dev}")
            continue
        tag = f"{idx}_{serial[:42]}"
        path = os.path.join(out_dir, f"{tag}.jpg")
        cv2.putText(frame, f"#{idx} {serial[:46]}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        print(f"[{idx}] OK {dev} -> {path} ({frame.shape[1]}x{frame.shape[0]})")
        ok += 1
    print(f"完成: {ok}/{len(devs)} 路抓帧成功")
    print("\n把下面这些路径对应到 hand / front / top, 然后写进 05_b601_record.sh 的 CAMERAS:")
    for d in devs:
        print("   ", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
