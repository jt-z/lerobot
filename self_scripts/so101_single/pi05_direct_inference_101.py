#!/usr/bin/env python
"""pi05-so100_101 单臂 SO-101 直接控制环推理（带 shoulder_lift 口径映射）

与同目录 pi05_inference_101.sh 的 lerobot-rollout 路径的区别：
  lerobot-rollout 走官方配方，策略输出原样下发；本脚本自己控制"下发什么"，
  可以对 shoulder_lift 做 本机 = a*模型 + b 的映射补偿后再发给从臂。
  其余照旧复用 lerobot 的 SOFollower（相机/校准/限幅/退出失能）+ 权重自带的 pre/post 处理器。

两种模式：
  check（默认）：读状态+相机、跑前向、打印【原始动作】与【映射后动作】；不发送动作，从臂不动。
  run          ：闭环下发（每 tick 前向一次，pi0.5 内部按 chunk=50 排队；带限幅与时长上限）。

⚠️ shoulder_lift 口径问题（实测数据，见下面映射预设）：
  本权重与 MolmoAct2-SO100_101 是同一批 1209 个社区 SO-100/101 数据集训出来的，
  只有 shoulder_lift 这一路和本机口径对不上：
    - 离线实测：本机肩 -97.5 时，模型输出 **-231.7**（其余 5 个关节只差 0~3°，等于"保持姿态"）
    - 该权重自带归一化统计里 shoulder_lift 的 q01/q99 = 24.4/196.3、min/max = -270/218.5
      （-270 这种值在多关节都出现，疑似缺失数据哨兵值）；本机数据集是 q01/q99 = -98.0/78.6、
      校准量程约 -76~+121 度。也就是说模型的输出落在它自己分布的下极端，
      而按分位对齐的映射会把它换算成 -361 度 —— 本机根本到不了。
  所以本脚本提供了几组补偿预设（见下），并把默认设为依据最硬的"镜像映射"那一组。

映射预设（本机 = a*模型 + b；默认用第 3 组"镜像映射"）：
  1) 单点对齐 / 姿态保持： --lift-scale 1.0  --lift-offset 134.2
     把模型 -231.7 映射回本机 -97.5（与当前姿态一致），行程两端是否贴合未验证。
  2) 分布两端对齐（q01/q99）： --lift-scale 1.0274 --lift-offset -123.08
     把模型 24.4/196.3 映射到本机 -98.0/78.6；注意它会把模型输出 -231.7 换算成 -361（超量程）。
  3) 镜像映射（默认）： --lift-scale -1.0 --lift-offset 90 --lift-wrap
     即"本机 = 90 - 模型"，并把结果折回 [-180, 180)（模型输出常超出 ±180，必须折）。

为什么第 3 组更硬（2026-09-22 拉了 repo_list 里 1220 个数据集的 meta 统计）：
  社区数据里 shoulder_lift 是**单峰**的：每份数据一票，min 的中位 ~54（q05=7.2）、max 的中位 ~187（q95=198），
  即"手臂收起(home)"读的是**最大值 ~188**、"抬起/伸出去"读的是**小值**；而本机正好相反：home = -97.9（本机最小值）、
  抬起 = 最大 +85。同一条物理轴、**方向相反**，且 C = 本机home + 社区max ≈ -98 + 188 ≈ 90，跨度也吻合
  （本机 183°，社区里用满行程的几份如 imsyed00 是 182°）。所以 shoulder_lift 不是"差一个常数"，而是
  **镜像 + 偏移**：本机 = 90 - 模型。其余关节（elbow/wf/wr/gripper）方向与本机一致，不需要镜像。
  验证：pi05 模型输出 -201.4 → 90+201.4=291.4 → 折回 -68.6；单点对齐给 -67.2（两者几乎重合，可交叉印证）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PAI0 = Path(__file__).resolve().parents[2]

MODEL_PATH = str(PAI0 / "pi0.5" / "pi05-so100_101")
DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00"
DEFAULT_ROBOT_ID = "jt_follower_arm_right"
DEFAULT_TASK = "Pick up the yellow banana placed at different angles and put it on the white fruit plate"

# 关节顺序（权重 README 明确：shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper）
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LIFT_IDX = JOINT_NAMES.index("shoulder_lift")
ELBOW_IDX = JOINT_NAMES.index("elbow_flex")

# 相机 key -> by-id 路径（与 record_101.sh / act_inference_101.sh 一致）
# 槽位映射：hand -> camera_0，front -> camera_1（权重只有 camera_0..3 四个槽位，缺的自动填空图）
CAMERAS = {
    "hand": "/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2609007-video-index0",
    "front": "/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608006-video-index0",
}
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30

# 默认的 shoulder_lift 映射：镜像映射（社区数据统计推出的方案，本机 = 90 - 模型）
DEFAULT_LIFT_SCALE = -1.0
DEFAULT_LIFT_OFFSET = 90.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wrap180(v: float) -> float:
    """把角度折回 [-180, 180)：单圈舵机的读数天然是 mod 360 的，镜像映射后常超出 ±180。"""
    return (float(v) + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------- 硬件
def build_robot(args):
    """构造并连接 SO-101 从臂（同时打开 2 路相机）。"""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    cam_cfgs = {
        key: OpenCVCameraConfig(
            index_or_path=path, fps=CAM_FPS, width=CAM_WIDTH, height=CAM_HEIGHT, fourcc="MJPG"
        )
        for key, path in CAMERAS.items()
    }
    cfg = SOFollowerRobotConfig(
        port=args.port,
        id=args.robot_id,
        cameras=cam_cfgs,
        max_relative_target=(args.max_relative_target if args.max_relative_target > 0 else None),
    )
    robot = SOFollower(cfg)
    log(f"连接从臂 {args.port} (id={args.robot_id})，同时打开相机 {list(CAMERAS)} ...")
    robot.connect()
    log("从臂 + 相机已连接（从臂已使能变硬）")
    return robot


def read_state(obs: dict) -> np.ndarray:
    return np.asarray([obs[f"{m}.pos"] for m in JOINT_NAMES], dtype=np.float32)


def read_images(obs: dict) -> list[np.ndarray]:
    return [np.asarray(obs[key]) for key in CAMERAS]


# ----------------------------------------------------------------- 模型
def load_policy(args):
    """加载 pi0.5 权重与权重自带的 pre/post 处理器。"""
    import torch

    import lerobot.policies.pi05.processor_pi05  # noqa: F401  注册 pi05_sentencepiece_tokenizer
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    cfg = PI05Config.from_pretrained(args.model_path)
    cfg.device = "cuda"
    cfg.dtype = args.dtype
    cfg.compile_model = False
    # 权重 config.json 里的分词器路径指向训练机（/lustre/...），必须换成权重自带的
    cfg.text_tokenizer_name = str(Path(args.model_path) / "tokenizer" / "tokenizer.model")

    log(f"加载模型（dtype={args.dtype}，fp32 权重约 16.6GB，请耐心等待）...")
    t0 = time.perf_counter()
    policy = PI05Policy.from_pretrained(args.model_path, config=cfg).eval().to("cuda")
    log(f"模型加载完成：{time.perf_counter() - t0:.1f}s，显存 {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=args.model_path,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    policy.reset()
    return policy, pre, post


def lift_mapping_enabled(args) -> bool:
    return args.lift_scale != 1.0 or args.lift_offset != 0.0


def elbow_mapping_enabled(args) -> bool:
    return args.elbow_scale != 1.0 or args.elbow_offset != 0.0


def apply_map(v: float, scale: float, offset: float, do_wrap: bool) -> float:
    out = scale * v + offset
    return wrap180(out) if do_wrap else out


def predict(policy, pre, post, obs_raw: dict, state: np.ndarray, args):
    """一次前向：返回 (本机口径动作(6,), 模型口径原始动作(6,), 耗时)。

    映射作用在两个方向：喂给模型的状态先换算到模型口径（反变换），
    模型出来的动作再换算回本机口径（正变换），保证两头一致。
    """
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference

    obs = {
        "observation.state": np.array(state, dtype=np.float32, copy=True),
        "observation.images.camera_0": obs_raw["hand"],  # hand -> camera_0
        "observation.images.camera_1": obs_raw["front"],  # front -> camera_1
    }
    if lift_mapping_enabled(args):
        model_lift = (state[LIFT_IDX] - args.lift_offset) / args.lift_scale
        obs["observation.state"][LIFT_IDX] = wrap180(model_lift) if args.lift_wrap else model_lift
    if elbow_mapping_enabled(args):
        model_elbow = (state[ELBOW_IDX] - args.elbow_offset) / args.elbow_scale
        obs["observation.state"][ELBOW_IDX] = wrap180(model_elbow) if args.lift_wrap else model_elbow

    obs = prepare_observation_for_inference(obs, torch.device("cuda"), args.task, "so_follower")
    t0 = time.perf_counter()
    with torch.inference_mode():
        prepared = pre(obs)
        action = policy.select_action(prepared)
        action = post(action)
    dt = time.perf_counter() - t0

    raw = action[0].float().cpu().numpy().astype(np.float32)
    mapped = raw.copy()
    if lift_mapping_enabled(args):
        mapped[LIFT_IDX] = apply_map(raw[LIFT_IDX], args.lift_scale, args.lift_offset, args.lift_wrap)
    if elbow_mapping_enabled(args):
        mapped[ELBOW_IDX] = apply_map(raw[ELBOW_IDX], args.elbow_scale, args.elbow_offset, args.lift_wrap)
    return mapped, raw, dt


# ----------------------------------------------------------------- 输出
def describe_images(images: list[np.ndarray]) -> None:
    for key, img in zip(CAMERAS, images):
        log(f"  相机 {key}: shape={img.shape} dtype={img.dtype} 均值={img.mean():.1f} 路径={CAMERAS[key]}")


def save_cameras(images: list[np.ndarray], state: np.ndarray, out_dir: Path) -> None:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for key, img in zip(CAMERAS, images):
        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(out_dir / f"cam_{key}.png")
    (out_dir / "state.json").write_text(
        json.dumps(dict(zip(JOINT_NAMES, [float(v) for v in state])), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"已保存首帧图像与关节状态：{out_dir}")


def log_mapping(args) -> None:
    if lift_mapping_enabled(args):
        log(
            f"shoulder_lift 映射已启用：本机 = {args.lift_scale:g}*模型 + ({args.lift_offset:g})"
            f"{'，结果折回 [-180,180)' if args.lift_wrap else ''}"
            f"（喂给模型的状态用反变换 (our-({args.lift_offset:g}))/{args.lift_scale:g}）"
        )
    else:
        log("⚠️ shoulder_lift 映射未启用：模型输出原样下发（肩会被限幅顶向量程边界）")
    if elbow_mapping_enabled(args):
        log(
            f"elbow_flex 映射已启用：本机 = {args.elbow_scale:g}*模型 + ({args.elbow_offset:g})"
            "（社区数据里 elbow 的零点比本机高 ~88°，按需开启；默认关闭）"
        )


# ----------------------------------------------------------------- 模式
def run_check(args, robot, policy, pre, post) -> None:
    log("=== check 模式：只验证相机/模型/映射，不发送动作（从臂不动） ===")
    obs = robot.get_observation()
    state = read_state(obs)
    images = read_images(obs)
    describe_images(images)
    log("当前关节状态：" + ", ".join(f"{m}={v:.1f}" for m, v in zip(JOINT_NAMES, state)))
    log_mapping(args)
    save_cameras(images, state, Path(args.save_dir))

    obs_raw = dict(zip(CAMERAS, images))
    mapped, raw, dt = predict(policy, pre, post, obs_raw, state, args)
    log(f"前向完成：{dt * 1000:.0f}ms")
    log("模型原始动作：" + ", ".join(f"{m}={v:.1f}" for m, v in zip(JOINT_NAMES, raw)))
    log("映射后动作  ：" + ", ".join(f"{m}={v:.1f}" for m, v in zip(JOINT_NAMES, mapped)))
    log("与当前状态差：" + ", ".join(f"{m}={v:+.1f}" for m, v in zip(JOINT_NAMES, mapped - state)))
    log("check 通过。加 --mode run 才会真正下发动作。")


def run_loop(args, robot, policy, pre, post) -> None:
    log("=== run 模式：闭环推理（每 tick 读状态+相机 -> 前向 -> 下发） ===")
    if args.max_relative_target > 0:
        log(f"安全限幅：每步每关节最多 {args.max_relative_target} 度（=0 表示不限）")
    else:
        log("⚠️ 未启用相对位移限幅（max_relative_target<=0）")
    log_mapping(args)
    log(f"pi0.5 chunk=50：每 50 tick 重新前向一次，其余 tick 从队列取动作（日志里首次/重规划 ≈ 数百 ms）")

    t_start = time.monotonic()
    t_end = t_start + args.duration if args.duration > 0 else None
    period = 1.0 / args.fps
    tick = 0
    saved_first = False

    try:
        while True:
            obs = robot.get_observation()
            state = read_state(obs)
            images = read_images(obs)
            if not saved_first:
                describe_images(images)
                save_cameras(images, state, Path(args.save_dir))
                saved_first = True

            mapped, raw, dt = predict(policy, pre, post, dict(zip(CAMERAS, images)), state, args)

            goal = {f"{m}.pos": float(v) for m, v in zip(JOINT_NAMES, mapped)}
            step_t0 = time.monotonic()
            sent = robot.send_action(goal)
            clamped = max(abs(sent[f"{m}.pos"] - goal[f"{m}.pos"]) for m in JOINT_NAMES)

            tick += 1
            if tick % args.log_every == 0:
                log(
                    f"[{tick:5d}] t={time.monotonic() - t_start:6.1f}s 前向 {dt * 1000:5.0f}ms  "
                    f"下发: " + ", ".join(f"{m}={v:.0f}" for m, v in zip(JOINT_NAMES, mapped))
                )
            if clamped > 1e-3 and tick % args.log_every == 0:
                log(f"        ⚠️ 被限幅 {clamped:.1f} 度（max_relative_target 生效）")

            if t_end is not None and time.monotonic() >= t_end:
                log(f"到达设定时长（{args.duration}s），停止。")
                return
            sleep_s = period - (time.monotonic() - step_t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        log("收到 Ctrl+C，退出推理。")


# ----------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="pi05-so100_101 单臂 SO-101 直接控制环（check 只验证，run 闭环下发）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["check", "run"], default="check", help="check=不动臂；run=闭环推理")
    p.add_argument("--model-path", default=MODEL_PATH, help="本机 pi0.5 权重目录")
    p.add_argument("--port", default=DEFAULT_PORT, help="SO-101 从臂串口（by-id）")
    p.add_argument("--robot-id", default=DEFAULT_ROBOT_ID, help="校准文件 id")
    p.add_argument("--task", default=DEFAULT_TASK, help="语言指令（与采集/训练一致）")
    p.add_argument("--fps", type=float, default=30.0, help="下发频率（训练数据 30Hz）")
    p.add_argument("--duration", type=float, default=0.0, help="run 模式总时长（秒），0=跑到 Ctrl+C")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="推理精度")
    p.add_argument(
        "--max-relative-target", type=float, default=10.0, help="每步每关节最大相对位移（度），<=0=不限"
    )
    p.add_argument("--lift-scale", type=float, default=DEFAULT_LIFT_SCALE, help="shoulder_lift 映射系数 a")
    p.add_argument("--lift-offset", type=float, default=DEFAULT_LIFT_OFFSET, help="shoulder_lift 映射偏置 b")
    p.add_argument(
        "--lift-wrap",
        action="store_true",
        default=True,
        help="把映射结果折回 [-180,180)（镜像映射必须开；用 --no-lift-wrap 关闭）",
    )
    p.add_argument("--no-lift-wrap", dest="lift_wrap", action="store_false", help="关闭角度折回")
    p.add_argument(
        "--elbow-scale", type=float, default=1.0, help="elbow_flex 映射系数 a（社区数据估计 ≈ +88 的零点差）"
    )
    p.add_argument(
        "--elbow-offset", type=float, default=0.0, help="elbow_flex 映射偏置 b；推荐预设 -88（本机 = 模型 - 88）"
    )
    p.add_argument("--log-every", type=int, default=30, help="每 N tick 打印一次状态")
    p.add_argument(
        "--save-dir",
        default=str(PAI0 / "logs" / f"pi05_direct_{time.strftime('%Y%m%d_%H%M%S')}"),
        help="首帧图像/状态输出目录",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.model_path).is_dir():
        log(f"❌ 模型目录不存在：{args.model_path}")
        return 1

    robot = None
    try:
        policy, pre, post = load_policy(args)
        robot = build_robot(args)
        # 连上后先把当前姿态写成目标（避免电机寄存器里的旧目标在上电时甩臂）
        hold = {f"{m}.pos": float(v) for m, v in zip(JOINT_NAMES, read_state(robot.get_observation()))}
        robot.send_action(hold)
        log("已就地保持当前姿态：" + ", ".join(f"{m}={v:.1f}" for m, v in zip(JOINT_NAMES, hold.values())))
        if args.mode == "check":
            run_check(args, robot, policy, pre, post)
        else:
            run_loop(args, robot, policy, pre, post)
        return 0
    finally:
        if robot is not None:
            log("断开从臂（按校准回零并失能）...")
            robot.disconnect()
            log("已断开。")


if __name__ == "__main__":
    sys.exit(main())
