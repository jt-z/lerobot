#!/usr/bin/env python
"""lerobot-rollout 的动作追踪包装器（运行时打补丁，不改 lerobot / 机器人驱动源码）。

用法与 lerobot-rollout 完全一致，只是把命令换成：
    python action_trace.py --strategy.type=base --policy.path=... ...

逐个控制 tick 记录三组数值到日志（默认 self_scripts/_logs/action_test.log，
可用环境变量 ACTION_TRACE_LOG 覆盖）：

    model : 策略输出 —— SyncInferenceEngine.get_action 返回的 tensor（已过 policy 的
            postprocessor，单位=度，与训练数据集 action 同坐标系）。
            ACT 的 chunk 由 select_action 逐步弹出，所以这里就是"模型预测的 action 序列"。
    sent  : 真正传进 robot.send_action 的 dict（已经过 interpolator + robot_action_processor）。
    cmd   : robot.send_action 的返回值 = 乘过 joint_directions、裁过 joint_limits 之后的目标角
            （单位=度）；再乘 pi/180 就是真正下发到电机的弧度值。

由此可以判断中间环节是否改动了模型输出：
    sent != model  → 插值器 / robot_action_processor 改了它
    |cmd| != |sent| → 机器人层改了幅值（joint_limits 裁剪；纯方向翻转不改变幅值）

只覆盖 `--inference.type=sync`（本仓库脚本用的后端）。补丁装在 rollout 用的
`ThreadSafeRobot.send_action` 上，因此每次经它下发动作都会记一行，行首 src 列标明来源
（loop = 控制循环 send_next_action，init-ret = 收尾回初始位，other = 其它）；机器人内部
直接调用自身 send_action 的路径（如断连时的 safe_zero）不经过包装器，不会出现在日志里。
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Mapping

_TRACE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
# 日志统一落在仓库内 self_scripts/_logs/（本文件所在 tools/ 的同级），便于集中留存
_LOG_DIR = os.path.join(os.path.dirname(_TRACE_DIR), "_logs")
LOG_PATH = os.environ.get("ACTION_TRACE_LOG") or os.path.join(_LOG_DIR, "action_test.log")

# 关节顺序（用于对齐三组数值；顺序里没有的 key 追加在后面）
PREFERRED_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]
# 浮点比较容差（度）：低于此差异视为"没被改动"
TOL = 1e-3

_lock = threading.Lock()
_fh = None
_header_written = False
_tick = 0
_key_order: list[str] | None = None
_last_model: dict[str, float] | None = None
_warned = False
# 统计：被中间环节改动的 tick 数 / 关节集合
_stat_proc_modified = 0
_stat_clipped = 0
_stat_clipped_joints: set[str] = set()


def _warn(msg: str) -> None:
    """只提示一次，且绝不影响推理。"""
    global _warned
    if not _warned:
        _warned = True
        print(f"[action_trace] 追踪降级（推理不受影响）：{msg}", file=sys.stderr, flush=True)


def _joint(key: str) -> str:
    """'shoulder_pan.pos' -> 'shoulder_pan'（机器人和数据集的 key 命名统一）。"""
    key = str(key)
    return key[:-4] if key.endswith(".pos") else key


def _float_map(obj) -> dict[str, float] | None:
    """dict / tensor -> {joint: 度}；解析不了返回 None。"""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        out: dict[str, float] = {}
        for key, val in obj.items():
            try:
                out[_joint(key)] = float(val)
            except (TypeError, ValueError):
                continue
        return out or None
    try:
        flat = obj.reshape(-1).tolist()
    except Exception:
        return None
    return {f"a{i}": float(v) for i, v in enumerate(flat)} if flat else None


def _file():
    global _fh
    if _fh is None:
        directory = os.path.dirname(LOG_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        _fh = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
        print(f"[action_trace] 动作追踪已开启，记录 model/sent/cmd 到 {LOG_PATH}", flush=True)
    return _fh


def _write_header() -> None:
    global _header_written
    if _header_written:
        return
    _header_written = True
    _file().write(
        "\n"
        + "=" * 110
        + "\n"
        + f"[action_trace] 运行开始 {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        + f"  argv : {' '.join(sys.argv[1:])}\n"
        + "  model: 策略输出(度, 与数据集 action 同坐标系; ACT 每 tick 弹出 chunk 中的一步)\n"
        + "  sent : 传入 robot.send_action 的 dict(已经过 interpolator + robot_action_processor)\n"
        + "  cmd  : send_action 返回值 = 乘 joint_directions、裁 joint_limits 之后的目标角(度)\n"
        + "         -> 乘 pi/180 即真正下发到电机的弧度值\n"
        + "  标记 : sent!=model[...] 中间环节改动了模型输出;"
        + " |cmd|!=|sent|[...] 机器人层改动了幅值(限位裁剪)\n"
        + f"  列序 : {' '.join(_key_order or PREFERRED_ORDER)}\n"
        + "=" * 110
        + "\n"
    )


def _group(values: dict[str, float] | None, order: list[str]) -> str:
    if not values:
        return "         -"
    return " ".join(f"{values[k]:9.3f}" if k in values else "        -" for k in order)


def _write_row(src: str, sent: dict[str, float] | None, cmd: dict[str, float] | None) -> None:
    """写一行：tick + 时间 + 来源 + model/sent/cmd 三组数值 + 改动标记。"""
    global _tick, _key_order, _last_model, _stat_proc_modified, _stat_clipped

    names: list[str] = []
    for values in (sent, cmd, _last_model):
        if values:
            names.extend(k for k in values if k not in names)
    if _key_order is None:
        head = [k for k in PREFERRED_ORDER if k in names]
        _key_order = head + [k for k in sorted(names) if k not in head]
    order = _key_order

    model, _last_model = _last_model, None  # 每个模型动作只配一行
    tags: list[str] = []
    if model and sent:
        changed = [k for k in order if k in model and k in sent and abs(sent[k] - model[k]) > TOL]
        if changed:
            _stat_proc_modified += 1
            tags.append("sent!=model[" + ",".join(changed) + "]")
    if sent and cmd:
        clipped = [k for k in order if k in sent and k in cmd and abs(abs(cmd[k]) - abs(sent[k])) > TOL]
        if clipped:
            _stat_clipped += 1
            _stat_clipped_joints.update(clipped)
            tags.append("|cmd|!=|sent|[" + ",".join(clipped) + "]")

    _tick += 1
    stamp = f"{time.strftime('%H:%M:%S')}.{int(time.time() % 1 * 1000):03d}"
    _write_header()  # 首行前写表头（此时列序已确定）
    _file().write(
        f"{_tick:06d} {stamp} {src:<9}"
        f" | model {_group(model, order)}"
        f" | sent {_group(sent, order)}"
        f" | cmd {_group(cmd, order)}"
        + ("  " + " ".join(tags) if tags else "")
        + "\n"
    )


def _src_tag() -> str:
    """按调用栈判断这次 send_action 来自哪条路径。"""
    try:
        frame = sys._getframe(1)
        for _ in range(10):
            if frame is None:
                break
            name = frame.f_code.co_name
            if name == "send_next_action":
                return "loop"
            if name == "return_to_initial_position":
                return "init-ret"
            frame = frame.f_back
    except Exception:
        pass
    return "other"


def _patch_sync_engine() -> None:
    """记录策略（模型）输出。"""
    from lerobot.rollout.inference.sync import SyncInferenceEngine

    original = SyncInferenceEngine.get_action

    def get_action(self, obs_frame):
        action = original(self, obs_frame)
        if action is not None:
            global _last_model
            try:
                keys = [_joint(k) for k in getattr(self, "_ordered_action_keys", ())]
                values = [float(v) for v in action.reshape(-1).tolist()]
                if len(keys) != len(values):
                    keys = [f"a{i}" for i in range(len(values))]
                _last_model = dict(zip(keys, values))
            except Exception as exc:  # 追踪失败不能影响推理
                _warn(f"记录 model 输出失败：{exc}")
        return action

    SyncInferenceEngine.get_action = get_action


def _patch_robot_send() -> None:
    """记录真正下发给机器人的 action（入参 + 返回值）。"""
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot

    original = ThreadSafeRobot.send_action

    def send_action(self, action):
        with _lock:
            sent = _float_map(action)
        result = original(self, action)  # 机器人内部在这里做 ×joint_directions / joint_limits
        try:
            with _lock:
                _write_row(_src_tag(), sent, _float_map(result))
        except Exception as exc:  # 追踪失败不能影响推理
            _warn(f"记录 send_action 失败：{exc}")
        return result

    ThreadSafeRobot.send_action = send_action


def _write_summary() -> None:
    if _fh is None:
        return
    try:
        _fh.write(
            f"[action_trace] 运行结束 {time.strftime('%Y-%m-%d %H:%M:%S')} | 共 {_tick} tick\n"
            f"  sent!=model（插值器/processor 改动）: {_stat_proc_modified} tick\n"
            f"  |cmd|!=|sent|（机器人层限位裁剪）: {_stat_clipped} tick"
            + (f"，关节: {','.join(sorted(_stat_clipped_joints))}" if _stat_clipped_joints else "")
            + "\n"
        )
    except Exception as exc:
        _warn(f"写统计失败：{exc}")


def main() -> None:
    _patch_sync_engine()
    _patch_robot_send()
    from lerobot.scripts.lerobot_rollout import main as rollout_main

    try:
        rollout_main()
    finally:
        _write_summary()


if __name__ == "__main__":
    main()
