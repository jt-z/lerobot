#!/usr/bin/env python
"""咖啡子任务常驻推理 agent：机械臂 + 三路相机 + ACT/pi0.5 权重常驻。

为什么要有它：每个子任务单独起 `lerobot-rollout` 要 14–16s（导入 6–7s、相机 5s、
CAN 1–2s、权重 0.6–2s）。本进程把这些一次性做好，`run_subtask.sh` 只发一条 unix socket
指令，省掉重复启动开销。

设计要点：

* **不是每帧 RPC**。控制环仍在 agent 内用 lerobot 的 ``BaseStrategy`` 跑，时序与现有
  脚本完全一致；socket 只负责「开始/停止/查询」。
* 机械臂与相机由本进程独占。交给 rollout 的是 :class:`_PersistentRobotProxy`，
  ``connect/disconnect`` 不真正执行，其余属性透传，避免每个子任务结束时关掉相机和 CAN。
* 策略按 ``(type, path)`` 缓存：ACT 每个任务一份、pi0.5 一份，首次使用时加载，之后常驻显存。
* 每个子任务仍新建 context / processor / inference engine，并在退出时平滑归零。

协议（NDJSON，一行一个 JSON）：

请求::

    {"op":"health"}
    {"op":"status"}
    {"op":"run","task":"A1","policy":"act","timeout":40,"prompt":"...","ckpt":"/path/to/pretrained_model"}
    {"op":"switch","task":"A1","policy":"pi05","timeout":40,"prompt":"...","ckpt":"/path/to/model"}
    {"op":"switch","task":"A1","policy":"replay","timeout":40,"dataset":"/path/to/dataset","episode":12}
    {"op":"zero"}
    {"op":"stop","reason":"ok|operator_fail|timeout|aborted"}
    {"op":"shutdown"}

响应::

    {"event":"health","ok":true,"busy":false,"loaded":[...],"pid":123,"uptime_s":1.2}
    {"event":"status","busy":true,"task":"A1","policy":"act","run_id":3}
    {"event":"accepted","task":"A1","policy":"act","run_id":1}
    {"event":"switching","from":"act","to":"pi05","task":"A1"}
    {"event":"zeroing","task":"A1"}
    {"event":"zeroed","task":"A1","already":false}
    {"event":"log","line":"..."}
    {"event":"done","task":"A1","result":"ok","exit_code":0,"elapsed_s":40.0,"trigger":"natural","run_id":1}
    {"event":"error","error":"busy|bad_request|internal","message":"..."}

``switch`` 会先以 ``stop(switch)`` 结束当前会话并等待归零，再在同一机械臂/相机上
启动新策略；旧会话的 ``done`` 被抑制。``zero`` 只停并归零，不启动新策略。

结果映射：ok→0、operator_fail→2、timeout→3、error→4、aborted→130。

用法::

    bash run_coffee_agent.sh                     # 真机常驻
    bash run_coffee_agent.sh --self-test         # 不碰硬件，验证协议
    python coffee_agent.py --socket /tmp/a.sock --self-test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

EXIT_CODES = {
    "ok": 0,
    "operator_fail": 2,
    "timeout": 3,
    "error": 4,
    "aborted": 130,
    "switch": 0,
    "zero": 0,
}
STOP_REASONS = ("ok", "operator_fail", "timeout", "aborted")
# 停止原因 → 状态文件里的 trigger（与 subtask_supervisor.py 保持一致）
TRIGGER_OF = {
    "ok": "key_0",
    "operator_fail": "key_1",
    "timeout": "timeout",
    "aborted": "signal",
    "switch": "handover",
    "zero": "zero",
}

# 与 self_scripts/b601_single/inference/lerobot/_common.sh 完全一致（全 -1.0）
JOINT_DIRECTIONS = {
    "shoulder_pan": -1.0,
    "shoulder_lift": -1.0,
    "elbow_flex": -1.0,
    "wrist_flex": -1.0,
    "wrist_yaw": -1.0,
    "wrist_roll": -1.0,
    "gripper": -1.0,
}

# pi0.5 相机名映射（与 run_pi05_coffee_expo.sh 一致）
PI05_RENAME_MAP = {
    "observation.images.front": "observation.images.base_0_rgb",
    "observation.images.hand": "observation.images.left_wrist_0_rgb",
    "observation.images.top": "observation.images.right_wrist_0_rgb",
}

log = logging.getLogger("coffee_agent")


# ---------------------------------------------------------------------------
# 机械臂代理
# ---------------------------------------------------------------------------


class _PersistentRobotProxy:
    """把常驻机械臂交给 rollout，但屏蔽 connect/disconnect。

    rollout 的每个子任务结束都会 ``robot.disconnect()``；如果透传，相机会被关掉、
    CAN 会被释放，下一个子任务又得重连。这里只拦这两个方法，其余全部透传。
    """

    def __init__(self, robot: Any) -> None:
        self._robot = robot
        self.connect_calls = 0
        self.disconnect_calls = 0

    # --- 被拦截的两个生命周期方法 ---
    def connect(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.connect_calls += 1
        return None

    def disconnect(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.disconnect_calls += 1
        return None

    # --- 其余全部透传 ---
    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot, name)

    @property
    def is_connected(self) -> bool:
        return bool(self._robot.is_connected)


# ---------------------------------------------------------------------------
# 单次 run 的会话状态
# ---------------------------------------------------------------------------


@dataclass
class _Session:
    task: str
    policy: str
    run_id: int = 0
    dataset: str = ""
    episode: int = 0
    stop_event: threading.Event = field(default_factory=threading.Event)
    reason: str | None = None
    conn: Any = None
    wlock: threading.Lock = field(default_factory=threading.Lock)
    suppress_done: bool = False

    @property
    def trigger(self) -> str:
        return TRIGGER_OF.get(self.reason or "", "natural")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CoffeeAgent:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.socket_path = Path(args.socket).expanduser()
        self.log_file = Path(args.log_file).expanduser() if args.log_file else None
        self.device = args.device
        self.display_data = bool(args.display_data)
        self.self_test = bool(args.self_test)

        self.started_at = time.monotonic()
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._session: _Session | None = None
        self._run_thread: threading.Thread | None = None
        self._handover = False
        self._handover_thread: threading.Thread | None = None
        self._run_seq = 0
        self._home_position: dict | None = None   # A1 起点（操作员确认的零位），全体子任务共用的归零目标

        self._policy_lock = threading.Lock()
        self._policy_load_lock = threading.Lock()
        self._policies: dict[tuple[str, str], Any] = {}
        self._orig_load_policy = None
        self._preload_specs = _parse_preload(args.preload)

        self.robot = None
        self.robot_cfg = None
        self._robot_proxy: _PersistentRobotProxy | None = None
        self._visualization_ready = False
        self._server: socket.socket | None = None

    # ------------------------------------------------------------------ 日志

    def _setup_logging(self) -> None:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(self.log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            log.addHandler(fh)
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        log.addHandler(sh)
        log.propagate = False

    def _log(self, msg: str) -> None:
        log.info(msg)

    # ------------------------------------------------------------ 硬件/策略初始化

    def _init_hardware(self) -> None:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots import make_robot_from_config
        from lerobot.utils.import_utils import register_third_party_plugins
        from lerobot_robot_seeed_b601 import SeeedB601RSFollowerConfig

        register_third_party_plugins()

        cameras = {}
        for name in ("hand", "front", "top"):
            dev = os.environ.get(f"B601_AGENT_CAM_{name.upper()}", "")
            if not dev:
                raise SystemExit(
                    f"❌ 缺少相机设备 B601_AGENT_CAM_{name.upper()}；"
                    "请用 run_coffee_agent.sh 启动（它会从 _common.sh 取设备路径）"
                )
            if not Path(dev).exists():
                raise SystemExit(f"❌ 相机不存在：{dev}（B601_AGENT_CAM_{name.upper()}）")
            cameras[name] = OpenCVCameraConfig(
                index_or_path=Path(dev), fps=30, width=640, height=480, fourcc="MJPG"
            )

        port = os.environ.get("B601_FOLLOWER_PORT", "can0")
        robot_id = os.environ.get("FOLLOWER_ID", "follower")
        self.robot_cfg = SeeedB601RSFollowerConfig(
            port=port,
            id=robot_id,
            can_adapter="socketcan",
            cameras=cameras,
            joint_directions=dict(JOINT_DIRECTIONS),
            gravity_compensation=True,
        )
        log.info("连接机械臂 %s（id=%s，相机 hand/front/top）...", port, robot_id)
        self.robot = make_robot_from_config(self.robot_cfg)
        self.robot.connect()
        self._robot_proxy = _PersistentRobotProxy(self.robot)
        log.info("机械臂与相机已常驻：%s", self.robot)

    def _init_pi05_lite(self) -> None:
        """pi0.5 需要 lite 包装器（跳过 lm_head，12GB 卡才能装下）。"""
        if os.environ.get("LITE_LOAD", "1") != "1":
            log.info("LITE_LOAD=0：pi0.5 使用原生 from_pretrained")
            return
        lite_dir = Path(self.args.repo_root) / "lx" / "tools"
        if str(lite_dir) not in sys.path:
            sys.path.insert(0, str(lite_dir))
        try:
            import pi05_lite  # type: ignore

            pi05_lite._install_lite_load()
            pi05_lite._install_timing()
            log.info("已装载 pi05 lite 包装器（%s）", lite_dir / "pi05_lite.py")
        except Exception as exc:  # noqa: BLE001 - 缺 pi05 依赖时只影响 pi05
            log.warning("pi05 lite 包装器装载失败：%s", exc)

    def _install_rollout_patches(self) -> None:
        from lerobot.rollout import context as rollout_context

        self._orig_load_policy = rollout_context._load_pretrained_policy
        proxy = self._robot_proxy

        def _make_robot(_cfg):
            return proxy

        agent = self

        def _cached_load(policy_config):
            return agent._get_or_load_policy(policy_config)

        rollout_context.make_robot_from_config = _make_robot
        rollout_context._load_pretrained_policy = _cached_load
        log.info("已接管 rollout 的机器人与策略加载（机械臂/权重常驻）")

    # ------------------------------------------------------------------ 策略

    @staticmethod
    def _policy_key(policy_config: Any) -> tuple[str, str]:
        return (str(getattr(policy_config, "type", "?")), str(policy_config.pretrained_path))

    @staticmethod
    def _sync_policy_config(policy: Any, policy_config: Any) -> None:
        """缓存命中时把本次 run 的关键字段同步到常驻 policy 上。"""
        for attr in ("n_action_steps",):
            if hasattr(policy_config, attr) and hasattr(getattr(policy, "config", None), attr):
                setattr(policy.config, attr, getattr(policy_config, attr))

    def _get_or_load_policy(self, policy_config: Any) -> Any:
        key = self._policy_key(policy_config)
        with self._policy_lock:
            cached = self._policies.get(key)
        if cached is not None:
            self._sync_policy_config(cached, policy_config)
            return cached
        with self._policy_load_lock:
            with self._policy_lock:
                cached = self._policies.get(key)
            if cached is not None:
                self._sync_policy_config(cached, policy_config)
                return cached
            started = time.monotonic()
            policy = self._orig_load_policy(policy_config)
            with self._policy_lock:
                self._policies[key] = policy
            log.info("策略已常驻显存：%s %s（加载 %.1fs）", key[0], key[1], time.monotonic() - started)
            return policy

    @staticmethod
    def _build_policy_config(policy_type: str, path: str) -> Any:
        from lerobot.configs import PreTrainedConfig

        if policy_type == "act":
            overrides = ["--n_action_steps=100"]
        elif policy_type == "pi05":
            overrides = [
                "--use_peft=false",
                "--dtype=bfloat16",
                "--num_inference_steps=10",
            ]
        else:
            raise ValueError(f"未知策略类型：{policy_type}")
        cfg = PreTrainedConfig.from_pretrained(path, cli_overrides=overrides)
        cfg.pretrained_path = path
        return cfg

    def _preload(self) -> None:
        for policy_type, path in self._preload_specs:
            try:
                if not (Path(path) / "config.json").is_file():
                    log.warning("预加载跳过（缺少 config.json）：%s", path)
                    continue
                self._get_or_load_policy(self._build_policy_config(policy_type, path))
            except Exception:  # noqa: BLE001 - 预加载失败不影响后续按需加载
                log.exception("预加载失败：%s %s", policy_type, path)

    def _loaded_models(self) -> list[str]:
        with self._policy_lock:
            return [f"{t}:{p}" for t, p in self._policies]

    # ------------------------------------------------------------------ 协议发送

    @staticmethod
    def _send(conn: socket.socket, wlock: threading.Lock, payload: dict) -> bool:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            with wlock:
                conn.sendall(line.encode("utf-8"))
            return True
        except OSError:
            return False

    def _session_send(self, sess: _Session, payload: dict) -> None:
        if sess.conn is None:
            return
        if not self._send(sess.conn, sess.wlock, payload):
            log.debug("会话已断开，丢弃事件：%s", payload.get("event"))

    # ------------------------------------------------------------------ 状态查询

    def _health_payload(self) -> dict:
        with self._lock:
            busy = self._session is not None
            task = self._session.task if self._session else None
        return {
            "event": "health",
            "ok": True,
            "busy": busy,
            "task": task,
            "loaded": self._loaded_models(),
            "self_test": self.self_test,
            "pid": os.getpid(),
            "uptime_s": round(time.monotonic() - self.started_at, 2),
            "socket": str(self.socket_path),
        }

    # ------------------------------------------------------------------ 停止

    def _request_stop(self, reason: str) -> bool:
        with self._lock:
            sess = self._session
            if sess is None:
                return False
            if sess.reason is not None:
                return False
            sess.reason = reason
            sess.stop_event.set()
        log.info("收到停止请求：%s", reason)
        return True

    # ------------------------------------------------------------------ run

    def _validate_run(self, req: dict) -> tuple[str, str, float, str, str, str, int]:
        task = str(req.get("task", "")).strip()
        policy = str(req.get("policy", "")).strip().lower()
        prompt = str(req.get("prompt", "")).strip()
        ckpt = str(req.get("ckpt", "")).strip()
        dataset = str(req.get("dataset", "")).strip()
        try:
            timeout = float(req.get("timeout", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"timeout 非法：{req.get('timeout')!r}") from exc
        try:
            episode = int(req.get("episode", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"episode 非法：{req.get('episode')!r}") from exc

        if not task:
            raise ValueError("缺少 task")
        if policy not in ("act", "pi05", "replay"):
            raise ValueError(f"policy 必须是 act / pi05 / replay，收到 {policy!r}")
        if timeout <= 0:
            raise ValueError(f"timeout 必须为正数，收到 {timeout}")

        if policy == "replay":
            if not self.self_test:
                if not dataset:
                    raise ValueError("replay 缺少 dataset（数据集根目录）")
                if not (Path(dataset) / "meta/info.json").is_file():
                    raise ValueError(f"数据集缺少 meta/info.json：{dataset}")
                if episode < 0:
                    raise ValueError(f"episode 必须 >= 0，收到 {episode}")
            return task, policy, timeout, "", "", dataset, episode

        # act / pi05：走策略推理
        if not ckpt:
            raise ValueError("缺少 ckpt")
        if not self.self_test:
            # 自检只验协议，不要求权重落盘
            if not (Path(ckpt) / "config.json").is_file():
                raise ValueError(f"权重目录缺少 config.json：{ckpt}")
            if not (Path(ckpt) / "model.safetensors").is_file():
                raise ValueError(f"权重目录缺少 model.safetensors：{ckpt}")
        if not prompt:
            raise ValueError("缺少 prompt（必须与训练数据集任务文本逐字一致）")
        return task, policy, timeout, prompt, ckpt, "", 0

    def _start_run(self, req: dict, conn: socket.socket, wlock: threading.Lock) -> None:
        try:
            task, policy, timeout, prompt, ckpt, dataset, episode = self._validate_run(req)
        except ValueError as exc:
            self._send(conn, wlock, {"event": "error", "error": "bad_request", "message": str(exc)})
            return

        with self._lock:
            handover = self._handover
            busy = handover or self._session is not None
            if not busy:
                self._run_seq += 1
                sess = _Session(
                    task=task,
                    policy=policy,
                    run_id=self._run_seq,
                    dataset=dataset,
                    episode=episode,
                    conn=conn,
                    wlock=wlock,
                )
                self._session = sess

        if busy:
            self._send(
                conn,
                wlock,
                {
                    "event": "error",
                    "error": "busy",
                    "message": "正在切换策略，请稍后" if handover else "已有子任务在跑，先发 stop 或等它结束",
                },
            )
            return

        # 先回 accepted，再起 worker：避免极短自检里 done 抢在 accepted 前面
        self._session_send(
            sess, {"event": "accepted", "task": task, "policy": policy, "run_id": sess.run_id}
        )
        with self._lock:
            self._run_thread = threading.Thread(
                target=self._run_worker,
                args=(sess, timeout, prompt, ckpt, dataset, episode),
                name=f"run-{task}",
                daemon=True,
            )
            self._run_thread.start()

    def _run_worker(
        self, sess: _Session, timeout: float, prompt: str, ckpt: str, dataset: str, episode: int
    ) -> None:
        started = time.monotonic()
        result = "error"
        detail = ""
        try:
            if self.self_test:
                self._execute_selftest(sess, timeout)
            elif sess.policy == "replay":
                self._execute_replay(sess, timeout, dataset, episode)
            else:
                self._execute_run(sess, timeout, prompt, ckpt)
            if sess.reason is not None:
                result = sess.reason
            else:
                # 没有 stop 请求：ACT/pi0.5 跑满 duration = 超时；replay 放完 = 成功
                result = "ok" if sess.policy == "replay" else "timeout"
        except Exception as exc:  # noqa: BLE001 - 任何异常都映射成后端错误
            result = "error"
            detail = f"{type(exc).__name__}: {exc}"
            log.error("子任务异常：%s\n%s", detail, traceback.format_exc())
        finally:
            elapsed = time.monotonic() - started
            exit_code = EXIT_CODES[result]
            payload = {
                "event": "done",
                "task": sess.task,
                "policy": sess.policy,
                "run_id": sess.run_id,
                "result": result,
                "exit_code": exit_code,
                "elapsed_s": round(elapsed, 2),
                "timeout_s": timeout,
                "trigger": sess.trigger,
            }
            if detail:
                payload["detail"] = detail
            log.info(
                "子任务结束：%s result=%s rc=%d elapsed=%.1fs trigger=%s",
                sess.task,
                result,
                exit_code,
                elapsed,
                sess.trigger,
            )
            with self._lock:
                suppress = sess.suppress_done
                self._session = None
                self._run_thread = None
            if not suppress:
                self._session_send(sess, payload)

    def _execute_selftest(self, sess: _Session, timeout: float) -> None:
        """不碰硬件/权重：模拟一段可被 stop 打断的执行。"""
        limit = min(max(timeout, 0.05), 2.0)
        log.info("[self-test] 模拟执行 %s（上限 %.2fs）", sess.task, limit)
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if sess.stop_event.wait(0.05):
                break

    def _bind_home_position(self, task: str, ctx: Any) -> None:
        """以 A1 起点为固定零位：首次 A1 时记录，之后每段 teardown 都回到这里，避免链式漂移。"""
        if self._home_position is None:
            initial = getattr(ctx.hardware, "initial_position", None)
            if task == "A1" and initial:
                self._home_position = dict(initial)
                log.info("已记录固定零位（A1 起点，%d 个关节）", len(self._home_position))
            return
        ctx.hardware.initial_position = dict(self._home_position)

    def _execute_run(self, sess: _Session, timeout: float, prompt: str, ckpt: str) -> None:
        from lerobot.rollout import RolloutConfig, build_rollout_context, create_strategy
        from lerobot.rollout.configs import BaseStrategyConfig
        from lerobot.rollout.inference import SyncInferenceConfig

        policy_config = self._build_policy_config(sess.policy, ckpt)
        if sess.policy == "act":
            inference = SyncInferenceConfig(seam_max_step_deg=3.0, seam_jump_threshold_deg=12.0)
            rename_map: dict[str, str] = {}
        else:
            inference = SyncInferenceConfig()
            rename_map = dict(PI05_RENAME_MAP)

        cfg = RolloutConfig(
            robot=self.robot_cfg,
            policy=policy_config,
            strategy=BaseStrategyConfig(),
            inference=inference,
            fps=30.0,
            duration=timeout,
            task=prompt,
            display_data=self.display_data,
            display_compressed_images=False,
            play_sounds=False,
            device=self.device,
            rename_map=rename_map,
        )

        log.info(
            "开始 %s（policy=%s, ckpt=%s, timeout=%.0fs）",
            sess.task,
            sess.policy,
            ckpt,
            timeout,
        )
        ctx = build_rollout_context(cfg, sess.stop_event)
        self._bind_home_position(sess.task, ctx)
        strategy = create_strategy(cfg.strategy)
        try:
            strategy.setup(ctx)
            strategy.run(ctx)
        finally:
            # setup 失败也要 teardown：停掉 inference engine，并尽量把机械臂带回零位
            strategy.teardown(ctx)
            log.info(
                "子任务收尾完成（归零目标：%s）",
                "A1 记录的固定零位" if self._home_position else "本段起点",
            )

    def _execute_replay(
        self, sess: _Session, timeout: float, dataset_root: str, episode: int
    ) -> None:
        """重放数据集 episode（等价 06_replay_rerun.py，但复用常驻机械臂/相机）。

        与策略推理无关：按数据集 fps 逐帧下发录制的 action，不加载权重。
        """
        from lerobot.datasets import LeRobotDataset
        from lerobot.processor import make_default_robot_action_processor
        from lerobot.utils.constants import ACTION
        from lerobot.utils.cycle_timer import CycleTimer
        from lerobot.utils.visualization_utils import log_visualization_data

        repo_id = f"local/{Path(dataset_root).name}"
        dataset = LeRobotDataset(repo_id, root=dataset_root, episodes=[episode])
        actions = dataset.select_columns(ACTION)
        action_names = list(dataset.features[ACTION]["names"])
        processor = make_default_robot_action_processor()
        fps = float(dataset.fps)
        total = len(actions)
        log.info(
            "replay 开始：%s episode=%d（%d 帧 @ %.0fHz，上限 %.0fs）",
            dataset_root,
            episode,
            total,
            fps,
            timeout,
        )

        initial = {k: v for k, v in self.robot.get_observation().items() if k.endswith(".pos")}
        if self._home_position is None and sess.task == "A1" and initial:
            self._home_position = dict(initial)
            log.info("已记录固定零位（A1 起点，%d 个关节）", len(self._home_position))
        home = dict(self._home_position) if self._home_position else initial
        log.info("replay 归零目标：%s", "A1 记录的固定零位" if self._home_position else "replay 起点")
        timer = CycleTimer(fps, records_data=False)
        deadline = time.monotonic() + timeout
        try:
            idx = 0
            while idx < total:
                if sess.stop_event.is_set():
                    log.info("replay 收到停止请求 @ frame %d", idx)
                    break
                if time.monotonic() >= deadline:
                    sess.reason = "timeout"
                    log.info("replay 达到 %.0fs 上限 @ frame %d", timeout, idx)
                    break
                timer.tick()
                row = actions[idx][ACTION]
                action = {name: row[i] for i, name in enumerate(action_names)}
                obs = self.robot.get_observation()
                self.robot.send_action(processor((action, obs)))
                if self.display_data:
                    log_visualization_data(
                        "rerun", observation=obs, action=action, compress_images=False
                    )
                idx += 1
                timer.wait()
        finally:
            timer.log_run_summary()
            self._smooth_home(home)

    def _smooth_home(self, target: dict, duration_s: float = 3.0, fps: int = 50) -> None:
        """replay 结束后慢速归零（优先固定零位，否则回到 replay 起点）。"""
        if not target:
            return
        from lerobot.utils.robot_utils import precise_sleep

        try:
            current = {k: v for k, v in self.robot.get_observation().items() if k in target}
            steps = max(int(duration_s * fps), 1)
            for step in range(1, steps + 1):
                t = step / steps
                self.robot.send_action({k: current[k] * (1 - t) + target[k] * t for k in current})
                precise_sleep(1 / fps)
        except Exception:  # noqa: BLE001 - 归零失败也不能让 agent 退出
            log.exception("replay 归零失败")
        else:
            log.info("replay 归零完成")

    # ------------------------------------------------------------------ 热切换

    def _status_payload(self) -> dict:
        with self._lock:
            sess = self._session
            return {
                "event": "status",
                "ok": True,
                "busy": sess is not None,
                "handover": self._handover,
                "task": sess.task if sess else None,
                "policy": sess.policy if sess else None,
                "run_id": sess.run_id if sess else 0,
                "loaded": self._loaded_models(),
                "pid": os.getpid(),
            }

    def _start_switch(self, req: dict, conn: socket.socket, wlock: threading.Lock) -> None:
        """停当前会话（含 3s 归零），再在同一机械臂/相机上起新策略。"""
        try:
            task, policy, timeout, prompt, ckpt, dataset, episode = self._validate_run(req)
        except ValueError as exc:
            self._send(
                conn, wlock, {"event": "error", "error": "bad_request", "message": str(exc)}
            )
            return

        with self._lock:
            if self._handover:
                self._send(
                    conn,
                    wlock,
                    {"event": "error", "error": "busy", "message": "已有一次切换在进行"},
                )
                return
            old = self._session
            old_thread = self._run_thread
            from_policy = old.policy if old else None
            self._handover = True

        log.info("热切换请求：%s → %s（task=%s）", from_policy, policy, task)
        self._send(
            conn,
            wlock,
            {"event": "switching", "from": from_policy, "to": policy, "task": task},
        )

        def worker() -> None:
            try:
                if old is not None:
                    with self._lock:
                        old.suppress_done = True
                        if old.reason is None:
                            old.reason = "switch"
                            old.stop_event.set()
                    if old_thread is not None:
                        old_thread.join(timeout=60.0)
                    if old_thread is not None and old_thread.is_alive():
                        self._send(
                            conn,
                            wlock,
                            {
                                "event": "error",
                                "error": "internal",
                                "message": "上一段 60s 内未完成归零，取消切换",
                            },
                        )
                        return

                with self._lock:
                    if self._session is not None:
                        self._send(
                            conn,
                            wlock,
                            {"event": "error", "error": "busy", "message": "切换窗口被占用"},
                        )
                        return
                    self._run_seq += 1
                    sess = _Session(
                        task=task,
                        policy=policy,
                        run_id=self._run_seq,
                        dataset=dataset,
                        episode=episode,
                        conn=conn,
                        wlock=wlock,
                    )
                    self._session = sess

                log.info("热切换生效：%s（run_id=%d）", policy, sess.run_id)
                self._session_send(
                    sess,
                    {
                        "event": "accepted",
                        "task": task,
                        "policy": policy,
                        "run_id": sess.run_id,
                        "switched": True,
                    },
                )
                with self._lock:
                    self._run_thread = threading.Thread(
                        target=self._run_worker,
                        args=(sess, timeout, prompt, ckpt, dataset, episode),
                        name=f"run-{task}-{sess.run_id}",
                        daemon=True,
                    )
                    self._run_thread.start()
            finally:
                with self._lock:
                    self._handover = False

        self._handover_thread = threading.Thread(
            target=worker, name="handover", daemon=True
        )
        self._handover_thread.start()

    def _start_zero(self, req: dict, conn: socket.socket, wlock: threading.Lock) -> None:
        """只停当前会话并等待归零，不启动新策略。"""
        with self._lock:
            if self._handover:
                self._send(
                    conn,
                    wlock,
                    {"event": "error", "error": "busy", "message": "已有一次切换在进行"},
                )
                return
            old = self._session
            old_thread = self._run_thread
            if old is None:
                self._send(conn, wlock, {"event": "zeroed", "already": True})
                return
            self._handover = True

        log.info("归零请求：停止 %s 并等待归零", old.policy)
        self._send(conn, wlock, {"event": "zeroing", "task": old.task})

        def worker() -> None:
            try:
                with self._lock:
                    old.suppress_done = True
                    if old.reason is None:
                        old.reason = "zero"
                        old.stop_event.set()
                if old_thread is not None:
                    old_thread.join(timeout=60.0)
                if old_thread is not None and old_thread.is_alive():
                    self._send(
                        conn,
                        wlock,
                        {"event": "error", "error": "internal", "message": "60s 内未完成归零"},
                    )
                    return
                log.info("归零完成：%s", old.task)
                self._send(conn, wlock, {"event": "zeroed", "task": old.task, "already": False})
            finally:
                with self._lock:
                    self._handover = False

        self._handover_thread = threading.Thread(target=worker, name="zero", daemon=True)
        self._handover_thread.start()

    # ------------------------------------------------------------------ 连接处理

    def _handle_conn(self, conn: socket.socket) -> None:
        wlock = threading.Lock()
        log.info("客户端已连接")
        try:
            buf = b""
            while not self._shutdown.is_set():
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.decode("utf-8", "replace").strip()
                    if line:
                        self._handle_line(line, conn, wlock)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass
            log.info("客户端已断开")

    def _handle_line(self, line: str, conn: socket.socket, wlock: threading.Lock) -> None:
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("请求必须是 JSON 对象")
        except Exception as exc:  # noqa: BLE001
            self._send(conn, wlock, {"event": "error", "error": "bad_request", "message": str(exc)})
            return

        op = str(req.get("op", "")).strip()
        if op == "health":
            self._send(conn, wlock, self._health_payload())
        elif op == "status":
            self._send(conn, wlock, self._status_payload())
        elif op == "run":
            self._start_run(req, conn, wlock)
        elif op == "switch":
            self._start_switch(req, conn, wlock)
        elif op == "zero":
            self._start_zero(req, conn, wlock)
        elif op == "stop":
            reason = str(req.get("reason", "")).strip()
            if reason not in STOP_REASONS:
                self._send(
                    conn,
                    wlock,
                    {
                        "event": "error",
                        "error": "bad_request",
                        "message": f"stop.reason 必须是 {'|'.join(STOP_REASONS)}，收到 {reason!r}",
                    },
                )
                return
            accepted = self._request_stop(reason)
            self._send(
                conn,
                wlock,
                {"event": "stop_ack", "reason": reason, "accepted": accepted},
            )
        elif op == "shutdown":
            self._request_stop("aborted")
            self._send(conn, wlock, {"event": "bye"})
            self._shutdown.set()
        else:
            self._send(
                conn,
                wlock,
                {"event": "error", "error": "bad_request", "message": f"未知 op：{op!r}"},
            )

    # ------------------------------------------------------------------ 服务主循环

    def _bind_socket(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            # 只在确实没人监听时清理，避免把另一个 agent 的 socket 删掉
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(self.socket_path))
            except OSError:
                self.socket_path.unlink()
            else:
                probe.close()
                raise SystemExit(f"❌ socket 已被占用：{self.socket_path}（已有 agent 在跑？）")
            finally:
                probe.close()

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        srv.listen(8)
        srv.settimeout(0.5)
        self._server = srv
        log.info("监听 unix socket：%s", self.socket_path)

    def _on_signal(self, signum, _frame) -> None:
        log.info("收到信号 %s：结束当前子任务并退出", signal.Signals(signum).name)
        self._request_stop("aborted")
        self._shutdown.set()

    def serve(self) -> None:
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        if self.display_data and not self.self_test:
            try:
                from lerobot.utils.visualization_utils import init_visualization

                init_visualization("rerun", session_name="coffee_agent")
                self._visualization_ready = True
            except Exception as exc:  # noqa: BLE001 - 可视化失败不影响推理
                log.warning("Rerun 初始化失败（继续无可视化）：%s", exc)

        if not self.self_test:
            self._init_pi05_lite()
            self._init_hardware()
            self._install_rollout_patches()
            if self._preload_specs:
                threading.Thread(target=self._preload, name="preload", daemon=True).start()

        self._bind_socket()
        assert self._server is not None
        log.info("coffee_agent 就绪（pid=%d, self_test=%s）", os.getpid(), self.self_test)

        try:
            while not self._shutdown.is_set():
                try:
                    conn, _ = self._server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        log.info("agent 退出中...")
        self._request_stop("aborted")
        thread = self._run_thread
        if thread is not None:
            thread.join(timeout=30.0)
        handover = self._handover_thread
        if handover is not None:
            handover.join(timeout=30.0)
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except OSError:
            pass
        if self._visualization_ready:
            try:
                from lerobot.utils.visualization_utils import shutdown_visualization

                shutdown_visualization("rerun")
            except Exception:  # noqa: BLE001
                pass
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception:  # noqa: BLE001 - 退出阶段不抛
                log.exception("断开机械臂失败")
        log.info("agent 已退出")


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------


def _parse_preload(spec: str | None) -> list[tuple[str, str]]:
    """解析 ``--preload``：``type:path`` / ``type=path``，逗号分隔。"""
    if not spec:
        return []
    out: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        sep = ":" if ":" in item else ("=" if "=" in item else None)
        if sep is None:
            raise SystemExit(f"❌ --preload 格式应为 type:path，收到 {item!r}")
        ptype, path = item.split(sep, 1)
        ptype = ptype.strip().lower()
        if ptype not in ("act", "pi05"):
            raise SystemExit(f"❌ --preload 的策略类型只能是 act/pi05，收到 {ptype!r}")
        out.append((ptype, path.strip()))
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[4]  # .../dev/lerobot
    parser = argparse.ArgumentParser(description="咖啡子任务常驻推理 agent")
    parser.add_argument(
        "--socket",
        default=os.environ.get(
            "B601_AGENT_SOCKET",
            str(Path.home() / "dev/lerobot/self_scripts/_logs/coffee_agent.sock"),
        ),
        help="unix socket 路径",
    )
    parser.add_argument("--log-file", default="", help="agent 日志文件（空=只输出 stderr）")
    parser.add_argument(
        "--preload",
        default=os.environ.get("PRELOAD", ""),
        help="启动后异步预加载，逗号分隔的 type:path（如 pi05:/path,act:/path2）",
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument(
        "--display-data",
        type=int,
        default=int(os.environ.get("DISPLAY_DATA", "1")),
        help="1=启用 Rerun 可视化（默认），0=关闭",
    )
    parser.add_argument("--self-test", action="store_true", help="不连硬件、不加载权重，仅验证协议")
    parser.add_argument("--repo-root", default=str(repo_root), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    agent = CoffeeAgent(args)
    agent._setup_logging()
    log.info(
        "启动参数：socket=%s log=%s device=%s self_test=%s preload=%s",
        agent.socket_path,
        agent.log_file or "<stderr>",
        agent.device,
        agent.self_test,
        args.preload or "<空>",
    )
    log.info("启动时间：%s", datetime.now().isoformat(timespec="seconds"))
    try:
        agent.serve()
    except SystemExit as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
