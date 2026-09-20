#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时监控 bi_b601_so101_follower 双臂电机角度与温度（纯只读，不使能电机）。

左臂 = Seeed B601-RS（RobStride 电机，SocketCAN）
    参考 lerobot_robot_seeed_b601/seeed_b601_rs_follower.py：
    - 不走 SeeedB601RSFollower.connect()（会 enable_all 使能电机变硬），
      直接用 motorbridge.Controller 只读反馈：
      motor.request_feedback() -> bus.poll_feedback_once() -> motor.get_state()
    - 角度 = degrees(state.pos)，温度 = state.t_mos / state.t_rotor (°C)

右臂 = SO-101（Feetech STS3215，USB 串口）
    参考 lerobot/robots/so_follower/so_follower.py：
    - 不调用 SOFollower.connect()（其 configure() 末尾会 enable_torque），
      只做 bus.connect()（开串口 + 握手，不发使能指令），然后 sync_read：
      Present_Position（依校准文件输出角度）与 Present_Temperature（°C）

用法：
    conda activate lerobot
    python self_scripts/b601_so101_bimanual/03_monitor_motors.py                            # 默认 can0 + 右臂 by-id 串口
    python self_scripts/b601_so101_bimanual/03_monitor_motors.py --so101-port /dev/ttyACM1  # 指定右臂串口
    python self_scripts/b601_so101_bimanual/03_monitor_motors.py --no-right                 # 只看左臂 B601
    python self_scripts/b601_so101_bimanual/03_monitor_motors.py --no-left                  # 只看右臂 SO-101

注意：
    - 监控进程独占对应串口/CAN，勿与遥操作/录制进程同时打开同一硬件。
    - 纯监控不发送任何使能/目标指令，退出后电机保持原状态。
    - 左臂若连不上，检查 CAN：sudo ip link set can0 up type can bitrate 1000000
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation
from matplotlib.container import BarContainer

# ==================== 硬件定义（与参考实现保持一致） ====================

# 左臂 B601：CAN ID 与电机型号（config_seeed_b601_rs_follower.py / seeed_b601_rs_follower.py）
B601_MOTOR_CAN_IDS: dict[str, tuple[int, int]] = {
    "shoulder_pan":  (0x01, 0xFD),
    "shoulder_lift": (0x02, 0xFD),
    "elbow_flex":    (0x03, 0xFD),
    "wrist_flex":    (0x04, 0xFD),
    "wrist_yaw":     (0x05, 0xFD),
    "wrist_roll":    (0x06, 0xFD),
    "gripper":       (0x07, 0xFD),
}
B601_MOTOR_MODELS: dict[str, str] = {
    "shoulder_pan":  "rs-06",
    "shoulder_lift": "rs-06",
    "elbow_flex":    "rs-06",
    "wrist_flex":    "rs-00",
    "wrist_yaw":     "rs-00",
    "wrist_roll":    "rs-00",
    "gripper":       "rs-00",
}
# RobStride 温度保护阈值（config_seeed_b601_rs_follower.py）
B601_TEMP_WARN_C = 80.0
B601_TEMP_ALARM_C = 115.0

# 右臂 SO-101：电机名与 ID（so_follower.py）
SO101_MOTORS: list[str] = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
# STS3215 出厂默认过温保护
SO101_TEMP_WARN_C = 70.0


# ==================== 共享状态 ====================

class SharedState:
    """读线程 -> GUI 线程 的共享快照（加锁读写）。"""

    def __init__(self, left_motors: list[str], right_motors: list[str]):
        self.lock = threading.Lock()
        self.left = {
            "pos": {m: None for m in left_motors},
            "t_mos": {m: None for m in left_motors},
            "t_rotor": {m: None for m in left_motors},
            "ts": 0.0, "err": None,
        }
        self.right = {
            "pos": {m: None for m in right_motors},
            "temp": {m: None for m in right_motors},
            "ts": 0.0, "err": None, "calibrated": True,
        }


def temp_color(t: float | None) -> str:
    if t is None:
        return "#b0b0b0"
    if t >= 90.0:
        return "#d62728"  # 红
    if t >= 70.0:
        return "#ff7f0e"  # 橙
    if t >= 55.0:
        return "#f0c419"  # 黄
    return "#2e8b57"      # 绿


# ==================== 左臂读线程：B601-RS（motorbridge 只读） ====================

class B601Reader(threading.Thread):
    def __init__(self, channel: str, state: SharedState, stop_event: threading.Event, hz: float):
        super().__init__(daemon=True, name="b601-reader")
        self.channel = channel
        self.state = state
        self.stop_event = stop_event
        self.period = 1.0 / hz

    def run(self) -> None:
        from motorbridge import Controller

        bus = None
        motors: dict = {}
        fail_count = 0
        while not self.stop_event.is_set():
            if bus is None:
                try:
                    bus = Controller(channel=self.channel)
                    motors = {
                        name: bus.add_robstride_motor(mid, fid, B601_MOTOR_MODELS[name])
                        for name, (mid, fid) in B601_MOTOR_CAN_IDS.items()
                    }
                    fail_count = 0
                    with self.state.lock:
                        self.state.left["err"] = None
                except Exception as e:  # noqa: BLE001
                    with self.state.lock:
                        self.state.left["err"] = f"连接失败: {e}"
                    time.sleep(3.0)
                    continue

            try:
                # 与 get_observation / _read_motor_temperatures 相同的只读流程
                for m in motors.values():
                    m.request_feedback()
                bus.poll_feedback_once()

                pos, t_mos, t_rotor = {}, {}, {}
                for name, m in motors.items():
                    st = m.get_state()
                    if st is None:
                        pos[name] = t_mos[name] = t_rotor[name] = None
                    else:
                        pos[name] = math.degrees(st.pos)
                        t_mos[name] = float(st.t_mos)
                        t_rotor[name] = float(st.t_rotor)

                with self.state.lock:
                    self.state.left.update(pos=pos, t_mos=t_mos, t_rotor=t_rotor,
                                           ts=time.time(), err=None)
                fail_count = 0
            except Exception as e:  # noqa: BLE001
                fail_count += 1
                with self.state.lock:
                    self.state.left["err"] = f"读取失败({fail_count}): {e}"
                if fail_count >= 5:
                    # 连续失败 -> 重建 CAN 控制器
                    for m in motors.values():
                        try:
                            m.close()
                        except Exception:  # noqa: BLE001
                            pass
                    try:
                        bus.shutdown()
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        bus.close()
                    except Exception:  # noqa: BLE001
                        pass
                    bus, motors = None, {}
                    time.sleep(2.0)
                continue
            time.sleep(self.period)


# ==================== 右臂读线程：SO-101（Feetech 只读） ====================

def _build_so101_bus(port: str, calib_id: str):
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    # 仅复用 SOFollower 的校准文件加载与 bus 构建，绝不调用其 connect()。
    follower = SOFollower(SOFollowerRobotConfig(port=port, id=calib_id))
    return follower.bus


class SO101Reader(threading.Thread):
    def __init__(self, port: str, calib_id: str, state: SharedState,
                 stop_event: threading.Event, hz: float):
        super().__init__(daemon=True, name="so101-reader")
        self.port = port
        self.calib_id = calib_id
        self.state = state
        self.stop_event = stop_event
        self.period = 1.0 / hz

    def run(self) -> None:
        bus = None
        calibrated = True
        while not self.stop_event.is_set():
            if bus is None:
                try:
                    bus = _build_so101_bus(self.port, self.calib_id)
                    bus.connect()  # 仅开串口 + 握手，不使能力矩
                    calibrated = bool(bus.calibration)
                    with self.state.lock:
                        self.state.right["err"] = None
                        self.state.right["calibrated"] = calibrated
                except Exception as e:  # noqa: BLE001
                    with self.state.lock:
                        self.state.right["err"] = f"连接失败: {e}"
                    bus = None
                    time.sleep(3.0)
                    continue

            try:
                pos = bus.sync_read("Present_Position", normalize=calibrated, num_retry=2)
                temp = bus.sync_read("Present_Temperature", num_retry=2)
                with self.state.lock:
                    self.state.right.update(pos=dict(pos), temp=dict(temp),
                                            ts=time.time(), err=None)
            except Exception as e:  # noqa: BLE001
                with self.state.lock:
                    self.state.right["err"] = f"读取失败: {e}"
                try:
                    bus.disconnect()
                except Exception:  # noqa: BLE001
                    pass
                bus = None
                time.sleep(2.0)
                continue
            time.sleep(self.period)


# ==================== 中文字体（缺失时优雅回退） ====================

def setup_cjk_font() -> bool:
    candidates = [
        "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei", "PingFang SC",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return any(name in installed for name in candidates)


# ==================== GUI ====================

class MonitorGUI:
    def __init__(self, args, state: SharedState):
        self.args = args
        self.state = state
        self.left_motors = list(B601_MOTOR_CAN_IDS) if not args.no_left else []
        self.right_motors = list(SO101_MOTORS) if not args.no_right else []
        self.cjk = setup_cjk_font()

        def t(zh: str, en: str) -> str:
            return zh if self.cjk else en

        self._t = t

        self.fig, ((ax_la, ax_ra), (ax_lt, ax_rt)) = plt.subplots(2, 2, figsize=(13.5, 7.5))
        self.fig.suptitle(t("bi_b601_so101_follower 双臂电机监控（只读）",
                            "bi_b601_so101_follower motor monitor (read-only)"),
                          fontsize=13, fontweight="bold")
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.09,
                                 wspace=0.22, hspace=0.35)
        self.fps_text = self.fig.text(0.01, 0.01, "", fontsize=8, color="#555555")

        # --- 上排：角度 ---
        self.ax_langle = ax_la
        self.x_l = np.arange(len(self.left_motors))
        self.bars_langle = ax_la.bar(self.x_l, np.zeros(len(self.x_l)),
                                     color="#4c72b0", width=0.62)
        self.texts_langle = [ax_la.text(x, 0, "--", ha="center", va="bottom", fontsize=8)
                             for x in self.x_l]
        ax_la.set_xticks(self.x_l, self.left_motors, rotation=20, fontsize=8)
        ax_la.set_ylabel(t("角度 (°)", "angle (deg)"), fontsize=9)
        ax_la.grid(axis="y", alpha=0.3)

        self.ax_rangle = ax_ra
        self.x_r = np.arange(len(self.right_motors))
        self.bars_rangle = ax_ra.bar(self.x_r, np.zeros(len(self.x_r)),
                                     color="#557a3e", width=0.62)
        self.texts_rangle = [ax_ra.text(x, 0, "--", ha="center", va="bottom", fontsize=8)
                             for x in self.x_r]
        ax_ra.set_xticks(self.x_r, self.right_motors, rotation=20, fontsize=8)
        ax_ra.set_ylabel(t("角度 (°)", "angle (deg)"), fontsize=9)
        ax_ra.grid(axis="y", alpha=0.3)

        # --- 下排：温度 ---
        self.ax_ltemp = ax_lt
        w = 0.38
        self.bars_lmos = ax_lt.bar(self.x_l - w / 2, np.zeros(len(self.x_l)), w,
                                   label=t("MOS 温度", "MOS temp"))
        self.bars_lrotor = ax_lt.bar(self.x_l + w / 2, np.zeros(len(self.x_l)), w,
                                     label=t("转子温度", "rotor temp"))
        self.texts_ltemp = [ax_lt.text(x - w / 2, 0, "--", ha="center", va="bottom", fontsize=7)
                            for x in self.x_l]
        self.texts_ltemp += [ax_lt.text(x + w / 2, 0, "--", ha="center", va="bottom", fontsize=7)
                             for x in self.x_l]
        ax_lt.axhline(B601_TEMP_WARN_C, ls="--", lw=0.9, color="#ff7f0e", alpha=0.8)
        ax_lt.axhline(B601_TEMP_ALARM_C, ls="--", lw=0.9, color="#d62728", alpha=0.8)
        ax_lt.text(len(self.left_motors) - 0.45, B601_TEMP_WARN_C + 1.5,
                   f"{B601_TEMP_WARN_C:.0f}°C", fontsize=7, color="#ff7f0e", clip_on=True)
        ax_lt.text(len(self.left_motors) - 0.45, B601_TEMP_ALARM_C + 1.5,
                   f"{B601_TEMP_ALARM_C:.0f}°C", fontsize=7, color="#d62728", clip_on=True)
        ax_lt.set_xticks(self.x_l, self.left_motors, rotation=20, fontsize=8)
        ax_lt.set_ylabel(t("温度 (°C)", "temp (°C)"), fontsize=9)
        ax_lt.set_ylim(0, 100)
        ax_lt.legend(fontsize=8, loc="upper left")
        ax_lt.grid(axis="y", alpha=0.3)

        self.ax_rtemp = ax_rt
        self.bars_rtemp = ax_rt.bar(self.x_r, np.zeros(len(self.x_r)),
                                    color="#2e8b57", width=0.62)
        self.texts_rtemp = [ax_rt.text(x, 0, "--", ha="center", va="bottom", fontsize=8)
                            for x in self.x_r]
        ax_rt.axhline(SO101_TEMP_WARN_C, ls="--", lw=0.9, color="#ff7f0e", alpha=0.8)
        ax_rt.text(len(self.right_motors) - 0.55, SO101_TEMP_WARN_C + 1.5,
                   f"{SO101_TEMP_WARN_C:.0f}°C", fontsize=7, color="#ff7f0e", clip_on=True)
        ax_rt.set_xticks(self.x_r, self.right_motors, rotation=20, fontsize=8)
        ax_rt.set_ylabel(t("温度 (°C)", "temp (°C)"), fontsize=9)
        ax_rt.set_ylim(0, 100)
        ax_rt.grid(axis="y", alpha=0.3)

        self.last_print = 0.0
        self._last_frame_ts = time.perf_counter()
        self._frame_dt = 0.0

        self.anim = FuncAnimation(self.fig, self._update, interval=int(args.interval * 1000),
                                  cache_frame_data=False)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_close(self, _event) -> None:
        self._stop_event.set()

    def _set_stale_title(self, ax, base: str, ts: float, err: str | None, now: float) -> bool:
        """设置面板标题，返回数据是否新鲜。"""
        if err is not None:
            ax.set_title(f"{base} — {err}", fontsize=10, color="#d62728")
            return False
        age = now - ts
        if age > 3.0:
            ax.set_title(f"{base} — {self._t('数据超时', 'data stale')}"
                         f" ({age:.0f}s)", fontsize=10, color="#b8860b")
            return False
        ax.set_title(f"{base} — {age:.1f}s", fontsize=10)
        return True

    @staticmethod
    def _update_angle_panel(ax, bars: BarContainer, texts, values: dict, fresh: bool,
                            calibrated: bool, base_color: str) -> None:
        vals = []
        for rect, text, name in zip(bars, texts, values):
            v = values.get(name)
            if v is None or not fresh:
                rect.set_height(0.0)
                rect.set_color("#b0b0b0")
                text.set_text("--")
                text.set_position((rect.get_x() + rect.get_width() / 2, 0.2))
            else:
                rect.set_height(v)
                rect.set_color(base_color)
                text.set_text(f"{v:.1f}°" if calibrated else f"{v:.0f}")
                text.set_position((rect.get_x() + rect.get_width() / 2, v))
                vals.append(v)
        lo = min(vals + [-15.0]) - 10.0
        hi = max(vals + [15.0]) + 10.0
        ax.set_ylim(lo, hi)

    @staticmethod
    def _update_temp_panel(ax, series: list[tuple[BarContainer, list, dict]], fresh: bool) -> None:
        all_vals = []
        for bars, texts, values in series:
            for rect, text, name in zip(bars, texts, values):
                v = values.get(name)
                if v is None or not fresh:
                    rect.set_height(0.0)
                    rect.set_color("#b0b0b0")
                    text.set_text("--")
                    text.set_position((rect.get_x() + rect.get_width() / 2, 0.2))
                else:
                    rect.set_height(v)
                    rect.set_color(temp_color(v))
                    text.set_text(f"{v:.0f}")
                    text.set_position((rect.get_x() + rect.get_width() / 2, v))
                    all_vals.append(v)
        top = max(85.0, (max(all_vals) + 15.0) if all_vals else 85.0)
        ax.set_ylim(0, top)

    def _update(self, _frame):
        now = time.time()
        perf_now = time.perf_counter()
        self._frame_dt = 0.8 * self._frame_dt + 0.2 * (perf_now - self._last_frame_ts)
        self._last_frame_ts = perf_now
        self.fps_text.set_text(f"{1.0 / self._frame_dt:.1f} Hz"
                               if self._frame_dt > 0 else "")

        with self.state.lock:
            left = {k: ({m: v for m, v in d.items()} if isinstance(d, dict) else d)
                    for k, d in self.state.left.items()}
            right = {k: ({m: v for m, v in d.items()} if isinstance(d, dict) else d)
                     for k, d in self.state.right.items()}

        # ---- 左臂 ----
        if self.left_motors:
            fresh_l = self._set_stale_title(
                self.ax_langle, self._t(f"左臂 B601-RS @ {self.args.can}（角度）",
                                        f"Left B601-RS @ {self.args.can} (angle)"),
                left["ts"], left["err"], now)
            self._update_angle_panel(self.ax_langle, self.bars_langle, self.texts_langle,
                                     left["pos"], fresh_l, True, "#4c72b0")
            fresh_lt = self._set_stale_title(
                self.ax_ltemp, self._t(f"左臂 B601-RS（温度）", "Left B601-RS (temp)"),
                left["ts"], left["err"], now)
            self._update_temp_panel(
                self.ax_ltemp,
                [(self.bars_lmos, self.texts_ltemp[:len(self.left_motors)], left["t_mos"]),
                 (self.bars_lrotor, self.texts_ltemp[len(self.left_motors):], left["t_rotor"])],
                fresh_lt)

        # ---- 右臂 ----
        if self.right_motors:
            unit_note = "" if right["calibrated"] else self._t("（未校准:原始ticks）", " (raw ticks)")
            fresh_r = self._set_stale_title(
                self.ax_rangle, self._t(f"右臂 SO-101（角度）{unit_note}",
                                        f"Right SO-101 (angle){unit_note}"),
                right["ts"], right["err"], now)
            self._update_angle_panel(self.ax_rangle, self.bars_rangle, self.texts_rangle,
                                     right["pos"], fresh_r, right["calibrated"], "#557a3e")
            fresh_rt = self._set_stale_title(
                self.ax_rtemp, self._t("右臂 SO-101（温度）", "Right SO-101 (temp)"),
                right["ts"], right["err"], now)
            self._update_temp_panel(
                self.ax_rtemp,
                [(self.bars_rtemp, self.texts_rtemp, right["temp"])],
                fresh_rt)

        # ---- 控制台摘要 ----
        if self.args.print_interval > 0 and now - self.last_print >= self.args.print_interval:
            self.last_print = now
            stamp = datetime.now().strftime("%H:%M:%S")
            if self.left_motors and left["ts"] > 0:
                pos_s = " ".join(
                    f"{m[:6]}={'--' if left['pos'][m] is None else format(left['pos'][m], '.1f')}"
                    for m in self.left_motors)
                mos_s = " ".join(
                    f"{m[:6]}={'--' if left['t_mos'][m] is None else format(left['t_mos'][m], '.0f')}"
                    for m in self.left_motors)
                print(f"[{stamp}] [左B601] 角度: {pos_s} | MOS温: {mos_s} °C", flush=True)
            if self.right_motors and right["ts"] > 0:
                pos_s = " ".join(
                    f"{m[:6]}={'--' if right['pos'][m] is None else format(right['pos'][m], '.1f')}"
                    for m in self.right_motors)
                temp_s = " ".join(
                    f"{m[:6]}={'--' if right['temp'][m] is None else format(right['temp'][m], '.0f')}"
                    for m in self.right_motors)
                print(f"[{stamp}] [右SO101] 角度: {pos_s} | 温度: {temp_s} °C", flush=True)


# ==================== 入口 ====================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="bi_b601_so101_follower 双臂电机温度/角度实时监控（只读）")
    parser.add_argument("--can", default="can0", help="左臂 B601 SocketCAN 接口（默认 can0）")
    parser.add_argument("--so101-port",
                        default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00",
                        help="右臂 SO-101 串口")
    parser.add_argument("--id", default="jt_follower_arm",
                        help="校准文件 ID（右臂加载 {id}_right.json，默认 jt_follower_arm）")
    parser.add_argument("--hz", type=float, default=30.0, help="B601 读取频率 Hz（默认 30）")
    parser.add_argument("--so101-hz", type=float, default=10.0, help="SO-101 读取频率 Hz（默认 10）")
    parser.add_argument("--interval", type=float, default=0.1, help="界面刷新间隔秒（默认 0.1）")
    parser.add_argument("--print-interval", type=float, default=5.0,
                        help="控制台摘要打印间隔秒（默认 5，0 关闭）")
    parser.add_argument("--no-left", action="store_true", help="不监控左臂 B601")
    parser.add_argument("--no-right", action="store_true", help="不监控右臂 SO-101")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_left and args.no_right:
        raise SystemExit("左右臂都被禁用了，没有可监控的硬件。")

    print("=" * 60)
    print("bi_b601_so101_follower 只读监控")
    if not args.no_left:
        print(f"  左臂 B601-RS : CAN={args.can}（{len(B601_MOTOR_CAN_IDS)} 电机）")
    if not args.no_right:
        print(f"  右臂 SO-101  : {args.so101_port}（校准 id: {args.id}_right）")
    print("  只读模式：不会发送任何使能/目标指令")
    print("=" * 60)

    stop_event = threading.Event()
    state = SharedState(list(B601_MOTOR_CAN_IDS), list(SO101_MOTORS))

    readers: list[threading.Thread] = []
    if not args.no_left:
        readers.append(B601Reader(args.can, state, stop_event, args.hz))
    if not args.no_right:
        readers.append(SO101Reader(args.so101_port, f"{args.id}_right",
                                   state, stop_event, args.so101_hz))
    for r in readers:
        r.start()

    try:
        gui = MonitorGUI(args, state)
        gui._stop_event = stop_event  # 供窗口关闭时通知读线程
        plt.show()
    finally:
        stop_event.set()
        for r in readers:
            r.join(timeout=3.0)
        print("监控已退出（电机未被本程序改动）。")


if __name__ == "__main__":
    main()
