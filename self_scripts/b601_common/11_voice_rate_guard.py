#!/usr/bin/env python
"""lerobot-record 进程内包装器：超速语音提示 + 一键回零结束本集（z 键）。

功能一：超速语音提示
--------------------
采集脚本给从臂设了 `--robot.max_relative_target`（默认 6°/帧）。主臂操作速度超出
这个上限时，`ensure_safe_goal_position()` 会在 `lerobot.robots.utils` 上打一条
WARNING（"Relative goal position magnitude had to be clamped to be safe."），
而且只要还在超速就**每帧都打**，日志会刷屏。

这个包装器在进程内给该 logger 挂一个过滤器：
  * 一段连续超速里只在第一帧播报一次 "too fast"，并且只保留这一条日志；
  * 停顿超过 --reset-after 秒后再次超速，算新的一段，会再提示一次；
  * 两次提示之间至少隔 --cooldown 秒，避免快速抖动时连珠炮。

功能二：一键回零快捷键（--home-key，默认关闭）
--------------------------------------------
指定按键（采集脚本传 z）后，录制过程中按下该键：
  1. 立刻停止读取主臂，从臂不再跟随（"取消跟随"）；
  2. 从当前动作位置平滑插值到零位 home pose（默认 2.5 秒，smoothstep 起步/收尾）；
  3. 到达零位后保持 0.5 秒，然后自动置位 exit_early，等同按一次右方向键结束并保存本集。
插值帧照常写进数据集，因此轨迹末尾就是标准的"回 home pose"段，与 prompt 一致。

本集结束后进入 reset 阶段（终端提示 "Reset the environment"）时，从臂保持脱开并停在
零位，不跟随主臂；等 `dataset.save_episode()` 真正写完（含视频编码）之后，再用
--reengage-seconds（默认 3 秒）从零位缓慢对齐到主臂当前位姿，然后恢复正常跟随。
这样保存期间从臂不会动，也不会在本集刚结束时猛甩向主臂。reset / 保存期间不响应快捷键。

用法（由采集脚本调用，自身参数写在 `--` 前面，后面原样透传给 lerobot-record）：
  python self_scripts/b601_common/11_voice_rate_guard.py \
      --home-key z --home-seconds 2.5 --home-hold 0.5 --reengage-seconds 3.0 -- \
      --robot.type=seeed_b601_rs_follower --teleop.type=rebot_102_leader ...

不写 `--` 时，所有参数都当作 lerobot-record 的参数（即只开超速提示）。
"""

import argparse
import logging
import sys
import time

CLAMP_MARK = "Relative goal position magnitude had to be clamped"
LOGGER_NAME = "lerobot.robots.utils"


def speak(phrase: str) -> None:
    """调用 lerobot 自带的 TTS（Linux 下是 spd-say，非阻塞）。"""
    try:
        from lerobot.utils.utils import say

        say(phrase)
    except Exception:  # 没装 spd-say / 没起 speech-dispatcher 时静默降级
        pass


class FastMotionGuard(logging.Filter):
    """识别限幅 warning，按"段"播报并丢弃同段内重复的日志。"""

    def __init__(self, phrase: str, cooldown: float, reset_after: float):
        super().__init__()
        self.phrase = phrase
        self.cooldown = cooldown
        self.reset_after = reset_after
        self._last_clamp = 0.0
        self._last_spoken = 0.0

    def filter(self, record: logging.LogRecord) -> bool:
        if CLAMP_MARK not in record.getMessage():
            return True

        now = time.monotonic()
        new_burst = (now - self._last_clamp) > self.reset_after
        self._last_clamp = now

        # 同一段连续超速：只保留第一条，后续整条丢掉
        if not new_burst:
            return False
        # 新的一段，但离上次播报太近：这一次也丢掉
        if now - self._last_spoken < self.cooldown:
            return False

        self._last_spoken = now
        speak(self.phrase)
        return True


def install_home_shortcut(key: str, seconds: float, hold: float, reengage_seconds: float) -> None:
    """接管录制循环：按 `key` 后断跟随 -> 插值回零位 -> 保持 `hold` 秒 -> 结束本集。

    本集结束后从臂保持"脱开 + 零位"，reset 阶段（"Reset the environment"）只停在零位
    不跟随主臂；等 `dataset.save_episode()` 真正写完（含视频编码）之后，再用
    `reengage_seconds` 秒从零位缓慢对齐到主臂当前位姿，然后恢复普通跟随。
    reset 阶段和保存期间都不响应本快捷键。
    """
    import lerobot.scripts.lerobot_record as record_mod
    import lerobot.utils.keyboard_input as kb

    state: dict = {
        "robot": None,
        "teleop": None,
        "teleop_action_processor": None,
        "robot_action_processor": None,
        "fps": 30,
        "recording": False,
        "last_action": None,
        "start": None,
        "active": False,
        "done": False,
        "frame": 0,
        "released": False,
    }

    def ramp_to_leader() -> None:
        """保存完成后：按 fps 节奏把从臂从零位平滑推到主臂当前位姿（不写入数据集）。"""
        robot = state["robot"]
        teleop = state["teleop"]
        if robot is None or teleop is None:
            return
        teleop_action_processor = state["teleop_action_processor"]
        robot_action_processor = state["robot_action_processor"]
        fps = int(state["fps"] or 30)
        frames = max(1, int(round(reengage_seconds * fps)))
        period = 1.0 / fps
        next_tick = time.perf_counter()
        for i in range(1, frames + 1):
            ratio = i / frames
            ease = ratio * ratio * (3.0 - 2.0 * ratio)
            obs = robot.get_observation()
            leader_action = teleop.get_action()
            action = {name: value * ease for name, value in leader_action.items()}
            if teleop_action_processor is not None:
                action = teleop_action_processor((action, obs))
            if robot_action_processor is not None:
                action = robot_action_processor((action, obs))
            robot.send_action(action)
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def wrap_save_episode(dataset) -> None:
        """在本集数据真正写盘之后再做"缓慢对齐"，保证保存期间从臂仍是脱开的。"""
        if getattr(dataset, "_b601_home_shortcut", False):
            return
        orig_save_episode = dataset.save_episode

        def save_episode(*args, **kwargs):
            result = orig_save_episode(*args, **kwargs)
            if state["released"]:
                state["released"] = False
                print("[home] 本集已写入完成，开始缓慢对齐回主臂...", flush=True)
                ramp_to_leader()
                print("[home] 已重新对齐主臂，恢复跟随。", flush=True)
            return result

        dataset.save_episode = save_episode
        dataset._b601_home_shortcut = True

    def request_home() -> None:
        if state["released"]:
            print(f"[home] '{key}' 忽略：从臂正在重新对齐主臂。", flush=True)
            return
        if not state["recording"] or state["last_action"] is None:
            print(f"[home] '{key}' 忽略：当前不在录制阶段。", flush=True)
            return
        if state["active"]:
            print("[home] 正在回零位，忽略重复按键。", flush=True)
            return
        state["active"] = True
        state["done"] = False
        state["start"] = dict(state["last_action"])
        state["frame"] = 0
        print(f"[home] '{key}' 已触发：取消跟随主臂，{seconds:.1f}s 平滑回零位。", flush=True)

    # 在共享的键盘分发上多挂一个按键，避免再开一个 TTY 读取线程抢同一个 stdin。
    orig_create_key_listener = kb.create_key_listener

    def create_key_listener(dispatch, *, controls_help: str = ""):
        def wrapped(name: str) -> None:
            if name.lower() == key:
                request_home()
            else:
                dispatch(name)

        return orig_create_key_listener(wrapped, controls_help=f"{controls_help}, {key}=home+end")

    kb.create_key_listener = create_key_listener

    # 包一层 record_loop：录制阶段和紧随其后的 reset 阶段都接管 get_action。
    orig_record_loop = record_mod.record_loop

    def record_loop(*args, **kwargs):
        events = kwargs.get("events")
        teleop = kwargs.get("teleop")
        dataset = kwargs.get("dataset")
        fps = int(kwargs.get("fps") or 30)

        if teleop is None or events is None or not hasattr(teleop, "get_action"):
            return orig_record_loop(*args, **kwargs)

        ramp_frames = max(1, int(round(seconds * fps)))
        hold_frames = max(0, int(round(hold * fps)))

        state.update(
            {
                "robot": kwargs.get("robot"),
                "teleop": teleop,
                "teleop_action_processor": kwargs.get("teleop_action_processor"),
                "robot_action_processor": kwargs.get("robot_action_processor"),
                "fps": fps,
                "recording": dataset is not None,
            }
        )
        if not state["active"]:
            # 新的一段录制/重置开始，旧的跟随基准作废；回零过程本身不受影响
            state["last_action"] = None

        if dataset is not None:
            wrap_save_episode(dataset)
            if state["released"]:
                # 没走到 save（例如"重录本集"分支）时，开录前先把从臂对齐，避免录进过渡帧
                state["released"] = False
                print("[home] 开录前先缓慢对齐回主臂...", flush=True)
                ramp_to_leader()
                print("[home] 已重新对齐主臂，恢复跟随。", flush=True)

        orig_get_action = teleop.get_action

        def get_action():
            if state["active"]:
                # 回零插值：期间完全不读主臂
                state["frame"] += 1
                elapsed = state["frame"]
                if elapsed <= ramp_frames:
                    # smoothstep：起步和收尾都无速度突变
                    ratio = elapsed / ramp_frames
                    blend = ratio * ratio * (3.0 - 2.0 * ratio)
                    action = {name: value * (1.0 - blend) for name, value in state["start"].items()}
                else:
                    action = {name: 0.0 for name in state["start"]}

                if elapsed >= ramp_frames + hold_frames and not state["done"]:
                    # 结束后保持 active，剩余帧继续发零位，直到本集被 exit_early 收尾
                    state["done"] = True
                    events["exit_early"] = True
                    print(f"[home] 已到零位并保持 {hold:.1f}s，结束本集并保存。", flush=True)

                return action

            leader_action = orig_get_action()

            if state["released"]:
                # 本集收尾后的 reset 阶段：停在零位，等 save_episode 写完再缓慢对齐
                return {name: 0.0 for name in leader_action}

            state["last_action"] = dict(leader_action)
            return leader_action

        teleop.get_action = get_action
        try:
            return orig_record_loop(*args, **kwargs)
        finally:
            # 删掉实例属性，恢复成类上的原始 get_action
            try:
                del teleop.get_action
            except AttributeError:
                pass
            finished_homing = state["done"]
            state.update(
                {
                    "recording": False,
                    "last_action": None,
                    "start": None,
                    "active": False,
                    "done": False,
                    "frame": 0,
                }
            )
            if finished_homing:
                # 本集已回零结束：save_episode 写完后再缓慢对齐回主臂
                state["released"] = True

    record_mod.record_loop = record_loop


def main() -> None:
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        own_args, record_args = argv[:split], argv[split + 1 :]
    else:
        own_args, record_args = [], argv

    parser = argparse.ArgumentParser(description="lerobot-record 包装器（超速提示 + 一键回零）")
    parser.add_argument("--phrase", default="too fast", help="播报的短语")
    parser.add_argument("--cooldown", type=float, default=2.0, help="两次播报最短间隔（秒）")
    parser.add_argument("--reset-after", type=float, default=1.0,
                        help="停顿超过多少秒算一段新的超速（秒）")
    parser.add_argument("--home-key", default="",
                        help="一键回零快捷键（如 z）；留空则关闭该功能")
    parser.add_argument("--home-seconds", type=float, default=2.5,
                        help="从当前位姿插值回零位所需时间（秒）")
    parser.add_argument("--home-hold", type=float, default=0.5,
                        help="到达零位后保持多久再结束本集（秒）")
    parser.add_argument("--reengage-seconds", type=float, default=3.0,
                        help="本集结束进入 reset 后，从零位缓慢对齐回主臂位姿的时间（秒）")
    args = parser.parse_args(own_args)

    logging.getLogger(LOGGER_NAME).addFilter(
        FastMotionGuard(args.phrase, args.cooldown, args.reset_after)
    )

    if args.home_key:
        if args.home_seconds <= 0 or args.home_hold < 0 or args.reengage_seconds <= 0:
            print(
                "❌ --home-seconds / --reengage-seconds 必须 > 0，--home-hold 必须 >= 0",
                file=sys.stderr,
            )
            raise SystemExit(1)
        install_home_shortcut(
            args.home_key.lower(), args.home_seconds, args.home_hold, args.reengage_seconds
        )

    # 原样交给 lerobot-record（draccus 从 sys.argv 解析）
    sys.argv = ["lerobot-record", *record_args]
    from lerobot.scripts.lerobot_record import main as record_main

    record_main()


if __name__ == "__main__":
    main()
