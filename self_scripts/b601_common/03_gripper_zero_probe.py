#!/usr/bin/env python3
"""
夹爪零点判别实验：验证 RobStride 电机"使能时是否把当前位置写为 home（0 点）并持久化"。

背景：B601 从臂夹爪在"非 0 位使能"后出现 0 点漂移，且重新摆回 0 位再启动仍漂，
只能靠 MotorBridge Studio 重新校零写默认参数恢复。本脚本用两阶段实验判别驱动行为。

只操作夹爪电机（CAN id 0x07 / 0xFD, rs-00），不会使能其它关节，也不会驱动夹爪移动
（仅 enable/disable + 读反馈）。全程需要你手动摆夹爪位置。

用法：
  阶段1（夹爪当前在机械 0°附近，且系统已校零正常时执行）：
      python self_scripts/b601_common/03_gripper_zero_probe.py --phase 1
  阶段2（按提示断开 48V 电源重启后，夹爪仍停留在 150° 位置时执行）：
      python self_scripts/b601_common/03_gripper_zero_probe.py --phase 2

每个读数都会追加到 CSV（默认 ~/LX/pai0/logs/gripper_zero_probe.csv），用于跨断电对比；
脚本的完整交互输出会同时 tee 到一份带时间戳的运行日志
（默认 ~/LX/pai0/logs/gripper_zero_probe_run_<时间戳>.log），方便事后复盘。

结果判读：
  - 阶段2 的 baseline_raw（上电后、未使能时读到）若 ≈0 → 上电即清零/零点持久在 0，
    说明漂移不是"使能写 home"造成，需另查；
  - baseline_raw 若 ≈150（夹爪停的位置）→ 驱动保留上电位置为参考；
  - enable 后读数跳变到 ≈0 → 使能瞬间把当前位置置为 home（RAM）；
    断电重启后仍 ≈0 而夹爪物理在 150 → home 被持久化写入 flash（即漂移根因）。
"""

import argparse
import csv
import math
import os
import sys
import time

import motorbridge
from motorbridge import Controller, Mode

GRIPPER_SEND_ID = 0x07
GRIPPER_RECV_ID = 0xFD
GRIPPER_MODEL = "rs-00"

LOG_PATH = os.path.expanduser("~/LX/pai0/logs/gripper_zero_probe.csv")


class _Tee:
    """Duplicate all writes to a log file while still printing to the console."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, s):
        self._stream.write(s)
        self._fh.write(s)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def fileno(self):
        return self._stream.fileno()


def log_row(tag: str, raw_deg, note: str = "") -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["timestamp", "phase", "tag", "gripper_raw_deg", "note"])
        writer.writerow(
            [time.strftime("%Y-%m-%d %H:%M:%S"), phase, tag,
             f"{raw_deg:.4f}" if raw_deg is not None else "NA", note]
        )


def read_raw_deg(gripper, bus) -> float | None:
    """Request feedback once and return the gripper raw angle in degrees (no scaling)."""
    for _ in range(5):
        try:
            gripper.request_feedback()
            bus.poll_feedback_once()
            state = gripper.get_state()
            if state is not None:
                return math.degrees(state.pos)
        except Exception as exc:
            print(f"  (read retry: {exc})")
        time.sleep(0.5)
    return None


def wait_enter(msg: str) -> None:
    print(f"\n>>> {msg}")
    input("    准备好后按 ENTER 继续，Ctrl+C 中止...")


def main() -> None:
    global phase
    global LOG_PATH
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True,
                        help="1=夹爪在 0° 使能基线+手动到150°再使能；2=断电重启后复测")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--csv", default=LOG_PATH)
    parser.add_argument("--log", default=None,
                        help="详细运行日志路径（默认 ~/LX/pai0/logs/gripper_zero_probe_run_<时间戳>.log）")
    args = parser.parse_args()
    phase = args.phase
    LOG_PATH = os.path.expanduser(args.csv)

    run_log = args.log or os.path.expanduser(
        f"~/LX/pai0/logs/gripper_zero_probe_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    os.makedirs(os.path.dirname(run_log), exist_ok=True)
    _log_fh = open(run_log, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_fh)  # 终端 + 日志双写
    sys.stderr = _Tee(sys.__stderr__, _log_fh)

    print("=" * 70)
    print(f"夹爪零点判别实验 phase{phase}  (can={args.channel})")
    print("=" * 70)
    print(f"📄 CSV 记录：{LOG_PATH}")
    print(f"📄 详细日志：{run_log}")

    bus = Controller(channel=args.channel)
    gripper = bus.add_robstride_motor(GRIPPER_SEND_ID, GRIPPER_RECV_ID, GRIPPER_MODEL)
    print(f"✅ 已连接 can0，夹爪电机 id=0x{GRIPPER_SEND_ID:02X}/0x{GRIPPER_RECV_ID:02X} ({GRIPPER_MODEL})")
    print("⚠️  实验只使能夹爪电机；如夹爪此前因顶出处于 error，enable 可能失败，请先断电重启。")

    if phase == 1:
        wait_enter("请确认夹爪当前在【机械 0°（完全闭合收纳位）】附近后开始。")
        raw = read_raw_deg(gripper, bus)
        log_row("baseline_manual_0", raw, "phase1 手动摆0后首次读")
        print(f"    夹爪原始角度（手动0°位，未使能）: {raw:.4f}°" if raw is not None else "    读取失败")

        wait_enter("现在将【使能】夹爪电机（复现 lerobot configure 的 enable 动作）。")
        gripper.ensure_mode(Mode.MIT)
        gripper.enable()
        time.sleep(1.0)
        raw = read_raw_deg(gripper, bus)
        log_row("enable_at_0", raw, "phase1 在0°使能后读")
        print(f"    夹爪原始角度（使能后）: {raw:.4f}°" if raw is not None else "    读取失败")
        gripper.disable()
        print("    已失能夹爪。")

        wait_enter("请【手动】把夹爪推到 ~150°（张开到工作区中部），保持住。")
        raw = read_raw_deg(gripper, bus)
        log_row("manual_150_before_enable", raw, "phase1 手动150°失能读数")
        print(f"    夹爪原始角度（手动150°位，未使能）: {raw:.4f}°" if raw is not None else "    读取失败")

        wait_enter("现在再次【使能】夹爪电机——观察使能瞬间读数是否被重置为 0。")
        gripper.ensure_mode(Mode.MIT)
        gripper.enable()
        time.sleep(1.0)
        raw = read_raw_deg(gripper, bus)
        log_row("enable_at_150", raw, "phase1 在150°使能后读")
        print(f"    夹爪原始角度（150°位使能后）: {raw:.4f}°" if raw is not None else "    读取失败")
        gripper.disable()

        print("\n" + "=" * 70)
        print("阶段1 完成。夹爪请保持停在 ~150° 位置（不要动），现在：")
        print("  1. 断开 B601 从臂 48V 电源（彻底断电）")
        print("  2. 重新上电（夹爪仍停在 150° 位置，不要掰动）")
        print("  3. 运行：python self_scripts/b601_common/03_gripper_zero_probe.py --phase 2")
        print("=" * 70)

    else:  # phase 2
        wait_enter("已断电重启、夹爪仍停在同一位置？确认后开始读上电后的原始角度。")
        raw = read_raw_deg(gripper, bus)
        log_row("baseline_after_boot", raw, "phase2 断电重启后未使能读")
        print(f"    夹爪原始角度（上电后、未使能）: {raw:.4f}°" if raw is not None else "    读取失败")
        if raw is not None:
            if abs(raw) < 5.0:
                print("    → 读数≈0：上电后零点回到 0（说明零点在 flash，非‘使能写 home’）")
            elif abs(abs(raw) - 150.0) < 15.0:
                print("    → 读数≈150：上电后保留物理位置（零点即上电位置或被持久化为上次使能位）")

        wait_enter("现在【使能】夹爪（复现 configure）看读数是否跳变。")
        gripper.ensure_mode(Mode.MIT)
        gripper.enable()
        time.sleep(1.0)
        raw = read_raw_deg(gripper, bus)
        log_row("enable_after_boot", raw, "phase2 使能后读")
        print(f"    夹爪原始角度（使能后）: {raw:.4f}°" if raw is not None else "    读取失败")
        gripper.disable()
        print("\n对比 phase2 的 baseline_after_boot 与 enable_after_boot、以及 phase1 末尾记录，即可判断零点行为。")

    bus.close_bus()

    print("\n✅ 本阶段结束。")
    print(f"    CSV 记录：{LOG_PATH}")
    print(f"    详细日志：{run_log}（含本阶段全部读数与交互，可连同 CSV 一起反馈分析）")
    sys.stdout.flush()
    sys.stderr.flush()
    _log_fh.close()


if __name__ == "__main__":
    main()
