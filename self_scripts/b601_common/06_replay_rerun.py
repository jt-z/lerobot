#!/usr/bin/env python
"""重放数据集 episode 的 action，并用 rerun 实时显示 observation.state + 三路相机画面。

为什么需要这个脚本：
  lerobot 自带的 lerobot-replay（src/lerobot/scripts/lerobot_replay.py）**没有任何显示选项** ——
  它的配置只有 robot / dataset / play_sounds，`grep -i display` 一无所获；
  只有 lerobot-rollout 才有 `--display_data / --display_mode / --display_ip / --display_port`。
  所以"重放时看相机画面"不是某个开关没打开，而是 replay 这条路径本身没做可视化。

  这里复用 lerobot 的公共 API 把两者接起来（**不修改 lerobot 源码**）：
    * LeRobotDataset / DatasetReplayConfig  —— 读数据集、取 action 列
    * make_robot_from_config / make_default_robot_action_processor —— 与 lerobot-replay 完全一致的下发路径
    * CycleTimer(dataset.fps) —— 与 lerobot-replay 相同的按 fps 节拍
    * lerobot.utils.visualization_utils —— 与 lerobot-rollout 同一个 rerun 后端（含蓝图/布局）

用法（机器人、数据集参数与 lerobot-replay 一致，另加显示参数）:
  python self_scripts/b601_common/06_replay_rerun.py \
    --robot.type=seeed_b601_rs_follower --robot.id=follower --robot.port=can0 \
    --robot.can_adapter=socketcan --robot.gravity_compensation=true \
    --robot.joint_directions="{...}" --robot.cameras="{...}" \
    --dataset.repo_id=local/b601_20260915_141534 --dataset.root=... --dataset.episode=0 \
    --display_data=true --display_mode=rerun

  # 远端 rerun viewer（在另一台机器上跑 `rerun --serve`）:
    --display_ip=192.168.1.20 --display_port=9876

  # 关掉显示则等价于 lerobot-replay（此时也不需要相机）:
    --display_data=false

键盘控制（保持终端焦点）:
    p        暂停 / 保持当前帧姿态
    r        继续播放
    ← / →    暂停并按回放速度慢速倒退 / 前进 2 秒，到位后保持暂停
    v        开 / 关 rerun 画面刷新（不打断回放）
"""

# 注意：这里**不能**加 `from __future__ import annotations` ——
# lerobot 的 parser.wrap() 依赖 dataclass 字段的注解是真实类型（draccus 会调用
# typing.get_type_hints），加了之后注解变成字符串，--help / 参数解析会直接报 TypeError。

import logging
import sys
import threading
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import RobotConfig, make_robot_from_config  # noqa: F401
from lerobot.scripts.lerobot_replay import DatasetReplayConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.keyboard_input import TerminalKeyListener
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)


@dataclass
class ReplayDisplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # 语音播报（与 lerobot-replay 同名参数，便于脚本参数共用）
    play_sounds: bool = False
    # ---- 显示：参数名与 lerobot-rollout 的 --display_* 保持一致 ----
    display_data: bool = True
    display_mode: str = "rerun"  # rerun | foxglove
    display_ip: str | None = None  # rerun: 远端 viewer 的 IP；foxglove: 监听网卡
    display_port: int | None = None
    display_compressed_images: bool = False


class ReplayKeyboardController:
    """重放期间的键盘状态机。

    p    暂停（每个 tick 继续下发当前帧目标角，机械臂保持姿态）
    r    继续播放
    ←/→  暂停并按回放速度逐帧倒退/前进 2s，到位后保持暂停
    v    开/关 rerun 画面（只停/恢复刷新，不打断回放）
    """

    def __init__(self, num_frames: int, fps: int):
        self.num_frames = num_frames
        self.step_frames = max(1, round(2 * fps))  # 左右键一次 2 秒
        self._lock = threading.Lock()
        self.idx = 0
        self.paused = False
        self.target: int | None = None
        self.display_enabled = True

    def set_display_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.display_enabled = enabled

    def display_is_enabled(self) -> bool:
        with self._lock:
            return self.display_enabled

    def on_key(self, name: str) -> None:
        key = name.lower()
        with self._lock:
            if key == "p":
                if not self.paused:
                    self.paused = True
                    self.target = None
                    print(f"\n[replay] 暂停 @ frame {self.idx}（r 继续，←/→ 调整）", flush=True)
            elif key == "r":
                self.paused = False
                self.target = None
                print(f"\n[replay] 继续播放 @ frame {self.idx}", flush=True)
            elif key in ("left", "right"):
                base = self.target if self.target is not None else self.idx
                delta = -self.step_frames if key == "left" else self.step_frames
                self.target = max(0, min(self.num_frames - 1, base + delta))
                self.paused = True
                direction = "倒退" if delta < 0 else "前进"
                print(f"\n[replay] 暂停，按回放速度{direction} 2s → frame {self.target}", flush=True)
            elif key == "v":
                self.display_enabled = not self.display_enabled
                state = "开" if self.display_enabled else "关"
                print(f"\n[replay] 可视画面：{state}", flush=True)

    def next_action_index(self) -> int | None:
        """返回本 tick 要下发的帧号；None 表示整段播放结束。"""
        with self._lock:
            if self.target is not None:
                # 慢速拖动：每 tick 只走一帧，走完 2s 的量后停在目标帧。
                if self.idx < self.target:
                    self.idx += 1
                elif self.idx > self.target:
                    self.idx -= 1
                if self.idx == self.target:
                    self.target = None
                return self.idx
            if self.paused:
                return self.idx
            if self.idx >= self.num_frames:
                return None
            idx = self.idx
            self.idx += 1
            return idx


@parser.wrap()
def replay(cfg: ReplayDisplayConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])
    actions = dataset.select_columns(ACTION)
    robot_action_processor = make_default_robot_action_processor()

    controller = ReplayKeyboardController(dataset.num_frames, dataset.fps)
    controller.set_display_enabled(cfg.display_data)
    # 只读当前终端：避免 X11 全局钩子把其它窗口里的方向键也当成回放控制。
    keyboard = TerminalKeyListener(controller.on_key)
    keyboard.start()
    if sys.stdin.isatty():
        print("回放控制：p 暂停 | r 继续 | ←/→ 慢速倒退/前进 2s | v 开/关可视画面", flush=True)
    else:
        logging.warning("stdin 不是终端，键盘控制不可用（p/r/←/→/v）")

    display_initialized = False
    if cfg.display_data:
        logging.info(
            f"Initializing {cfg.display_mode} visualization (ip={cfg.display_ip}, port={cfg.display_port})"
        )
        init_visualization(
            cfg.display_mode, session_name="replay", ip=cfg.display_ip, port=cfg.display_port
        )
        display_initialized = True

    robot.connect()
    # 必须按数据集自身 fps 重放，否则轨迹速度不对；只下发、不写数据。
    timer = CycleTimer(dataset.fps, records_data=False)
    try:
        log_say("Replaying episode", cfg.play_sounds, blocking=True)
        while True:
            timer.tick()

            with timer.section("read_frame"):
                idx = controller.next_action_index()
                if idx is None:
                    break
                action_array = actions[idx][ACTION]
                action = {
                    name: action_array[i] for i, name in enumerate(dataset.features[ACTION]["names"])
                }

            with timer.section("observe"):
                robot_obs = robot.get_observation()

            with timer.section("send"):
                processed_action = robot_action_processor((action, robot_obs))
                robot.send_action(processed_action)

            if controller.display_is_enabled():
                if not display_initialized:
                    # 启动时 --display_data=false，用户按 v 后再开。
                    init_visualization(
                        cfg.display_mode,
                        session_name="replay",
                        ip=cfg.display_ip,
                        port=cfg.display_port,
                    )
                    display_initialized = True
                # 与 lerobot-rollout 的 telemetry 一样：observation（含三路相机图）+ action 一起送 rerun
                with timer.section("display"):
                    log_visualization_data(
                        cfg.display_mode,
                        observation=robot_obs,
                        action=action,
                        compress_images=cfg.display_compressed_images,
                    )

            timer.wait()
    finally:
        keyboard.stop()
        timer.log_run_summary()
        robot.disconnect()
        if display_initialized:
            shutdown_visualization(cfg.display_mode)


def main() -> None:
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
