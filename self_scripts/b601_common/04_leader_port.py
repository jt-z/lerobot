#!/usr/bin/env python
"""探测 StarArm102 主臂（rebot_102_leader）的串口。

背景（参考 so101_bimanual/inference/old_dataset/run_inference_cap_pen_ACT_300k.sh 的原则）：
  * 优先用 /dev/serial/by-id 稳定路径 —— ttyUSB*/ttyACM* 的编号随插拔顺序/驱动变化，
    同一个适配器今天可能叫 ttyUSB0，明天就变成 ttyACM0（本机 09-14 到 09-15 实测发生过）。
  * 但本机还有别的 USB 串口设备（例如 WCH 1a86:55d4 "USB Single Serial"），
    只按名字挑会挑错，结果是在 lerobot 里报 "Servo not found (id=0)"。
  * 所以这里对每个候选串口**真正 ping 一遍主臂舵机（FashionStar，id 0~6）**，
    第一个有应答的才算主臂；都不应答就明确报"没找到"。

用法:
  python self_scripts/b601_common/04_leader_port.py                  # 找主臂串口；找到则打印路径(stdout)，诊断走 stderr
  python self_scripts/b601_common/04_leader_port.py --baud 1000000   # 指定波特率（默认 1000000，与 02_record.sh 一致）
  python self_scripts/b601_common/04_leader_port.py --list           # 只列出候选串口与探测结果，不挑一个
  python self_scripts/b601_common/04_leader_port.py --port /dev/ttyACM0   # 只验证指定串口

退出码: 0 = 找到（或 --list 执行完成）；1 = 没有任何候选口应答
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

# 主臂 7 个关节的舵机 id（含夹爪），见 rebot_102_leader 的 joint_ids
MOTOR_IDS = {0: "shoulder_pan", 1: "shoulder_lift", 2: "elbow_flex", 3: "wrist_flex",
             4: "wrist_yaw", 5: "wrist_roll", 6: "gripper"}


def candidates() -> list[str]:
    """候选串口，按优先级：by-id 稳定路径 > ttyUSB* > ttyACM*（去重：by-id 与 tty 指向同一设备时只留一个）。"""
    pats = [
        "/dev/serial/by-id/usb-1a86_USB_Serial*",          # CH340（无序列号）：主臂最常用的那根
        "/dev/serial/by-id/usb-WCH.CN_USB_Single_Serial*",  # WCH CH9102 等
        "/dev/serial/by-id/usb-1a86_USB_Single_Serial*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
    ]
    out, seen = [], set()
    for p in pats:
        for f in sorted(glob.glob(p)):
            key = os.path.realpath(f)  # by-id 软链与其指向的 tty 视为同一个设备
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
    return out


def describe(path: str) -> str:
    """给出串口的身份信息，便于排查（VID:PID / product / 物理链路）。"""
    real = os.path.realpath(path)
    tty = os.path.basename(real)
    sysdir = f"/sys/class/tty/{tty}/device"
    info = []
    for field in ("idVendor", "idProduct", "product", "manufacturer"):
        for up in (1, 2, 3):  # 回退几层，兼容 tty 挂在接口下
            f = os.path.join(sysdir, *([".."] * up), field)
            if os.path.exists(f):
                try:
                    v = open(f).read().strip()
                    if v:
                        info.append(f"{field}={v}")
                    break
                except OSError:
                    pass
    chain = os.path.realpath(sysdir)
    if "usb" in chain:
        chain = chain.split("/usb")[-1].join(("usb", ""))
    return f"{' '.join(info) or '未知设备'}  链路={chain}"


def probe(path: str, baud: int, full: bool = False) -> list[int]:
    """ping 舵机；先试 id 0（shoulder_pan），命中后再（可选）把所有 id 试完。返回应答的 id 列表。"""
    from motorbridge_smart_servo import FashionStarServo

    try:
        bus = FashionStarServo(path, baudrate=baud)
    except Exception as exc:  # 端口被占用 / 权限 / 设备消失
        print(f"    {path}: 打开失败（{type(exc).__name__}: {exc}）", file=sys.stderr)
        return []
    try:
        hit = []
        for sid in MOTOR_IDS:
            try:
                if bus.ping(sid):
                    hit.append(sid)
            except Exception:
                pass
            if hit and not full:
                break  # 只验 id 0 时，命中即返回
        return hit
    finally:
        try:
            bus.close()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="探测 StarArm102 主臂串口（FashionStar 舵机应答）")
    ap.add_argument("--baud", type=int, default=int(os.environ.get("B601_LEADER_BAUD", 1000000)))
    ap.add_argument("--port", help="只验证这个串口（不做候选遍历）")
    ap.add_argument("--list", action="store_true", help="列出所有候选口及探测结果，不输出最终选择")
    args = ap.parse_args()

    ports = [args.port] if args.port else candidates()
    if not ports:
        print("没有找到任何候选串口（/dev/ttyUSB*、/dev/ttyACM*、/dev/serial/by-id/* 都为空）", file=sys.stderr)
        return 1

    print(f"候选串口 {len(ports)} 个，波特率 {args.baud}：", file=sys.stderr)
    chosen, results = None, []
    for p in ports:
        t0 = time.time()
        hit = probe(p, args.baud, full=not args.list)
        dt = time.time() - t0
        results.append((p, hit))
        mark = "✅" if hit else "❌"
        names = ",".join(MOTOR_IDS[i] for i in hit) if hit else "-"
        print(f"  {mark} {p}  ({describe(p)})  {dt:.1f}s  应答={names}", file=sys.stderr)
        if hit and chosen is None:
            chosen = p
            if not args.list:
                break

    if args.list:
        return 0
    if chosen is None:
        print("❌ 所有候选串口都没有主臂舵机应答：请检查 1) 主臂 USB 线是否插好 "
              "2) 主臂舵机是否上电（12V 电源/开关）", file=sys.stderr)
        return 1
    print(chosen)  # 只把路径写 stdout，供 shell 捕获
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
