#!/usr/bin/env python3
"""LingBot-VA 权重加载 + 延迟自检（不连机械臂、不发动作、不碰 CAN/相机）。

验的是四件事：
  1. 三段权重（LoRA adapter + LeRobot 基座 + diffusers 冻结模块）能在本机拼起来；
  2. checkpoint 的 pre/post processor（含 rename_map）能按 rollout 的方式装载；
  3. 一次 select_action 的动作形状/单位/坐标系对不对（action ≈ -state）；
  4. **每块重建要多久** —— lingbot_va 是自回归视频世界模型，开销不在"每步"而在"换块"，
     这条决定了真机上会慢放到什么程度（见 README「延迟」一节）。

用法：
  ./smoke_load.py                              # 用 capture_obs.py 存的最新实拍帧
  ./smoke_load.py --obs /path/live_obs.npz     # 指定观测
  ./smoke_load.py --obs none                   # 零图（只验形状与延迟，别拿来判行为）
  ./smoke_load.py --steps 64 --video-steps 10 --action-steps 10   # 试快档配置
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import _env

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
CAMS = ["front", "hand", "top"]


def build_frame(obs_path: str | None, height: int, width: int) -> tuple[dict, str]:
    """robot 侧观测（dataset frame 形状）——与控制环里 build_dataset_frame 的产物同口径。"""
    if obs_path and obs_path != "none" and os.path.isfile(obs_path):
        z = np.load(obs_path)
        state = z["state"].astype(np.float32)
        frame = {"observation.state": state}
        for c in CAMS:
            img = z[f"img_{c}"]
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) if img.dtype.kind == "f" else img.astype(np.uint8)
            frame[f"observation.images.{c}"] = img
        return frame, f"实拍帧 {obs_path}（state={np.round(state, 1).tolist()}）"

    rng = np.random.default_rng(0)
    frame = {"observation.state": np.zeros(7, np.float32)}
    for c in CAMS:
        frame[f"observation.images.{c}"] = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    return frame, "随机噪声图（只验形状/延迟，行为无意义）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=_env.ckpt())
    ap.add_argument("--wan", default=_env.wan())
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default=_env.attn_mode())
    ap.add_argument("--obs", default="/home/kf/b601_pi05/logs/live_obs.npz")
    ap.add_argument("--task", default=_env.task())
    ap.add_argument("--steps", type=int, default=48, help="要跑多少个 select_action（默认 48 → 覆盖首块 16 + 一块 32）")
    ap.add_argument("--video-steps", type=int, default=None, help="覆盖 num_inference_steps")
    ap.add_argument("--action-steps", type=int, default=None, help="覆盖 action_num_inference_steps")
    ap.add_argument("--guidance", type=float, default=None, help="覆盖 guidance_scale（1.0 = 关 CFG，前向次数减半）")
    ap.add_argument("--save-npz", default=None, help="把产出的动作序列存成 npz，便于事后分析")
    args = ap.parse_args()

    rename = _env.rename_map()
    dev = torch.device(args.device)

    print(f"checkpoint : {args.ckpt}")
    print(f"WAN 模块   : {args.wan}")
    print(f"attn_mode  : {args.attn_mode}   device: {dev}")
    print("-" * 78)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.rollout.context import _load_pretrained_policy

    # ---------- 1. config ----------
    t0 = time.perf_counter()
    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.pretrained_path = args.ckpt
    cfg.wan_pretrained_path = args.wan
    cfg.attn_mode = args.attn_mode
    cfg.device = str(dev)
    if args.video_steps is not None:
        cfg.num_inference_steps = args.video_steps
    if args.action_steps is not None:
        cfg.action_num_inference_steps = args.action_steps
    if args.guidance is not None:
        cfg.guidance_scale = args.guidance
    chunk = int(cfg.frame_chunk_size) * int(cfg.action_per_frame)
    print(f"[1/5] config 读完（{time.perf_counter() - t0:.1f}s）：type={cfg.type} use_peft={cfg.use_peft} "
          f"chunk={chunk} 步  视频去噪 {cfg.num_inference_steps} / 动作去噪 {cfg.action_num_inference_steps} "
          f"/ CFG {cfg.guidance_scale}")

    # ---------- 2. 策略（与 rollout 完全同一条加载路径）----------
    t0 = time.perf_counter()
    policy = _load_pretrained_policy(cfg)
    load_s = time.perf_counter() - t0
    lora = sum(p.numel() for n, p in policy.named_parameters() if "lora" in n.lower())
    print(f"[2/5] LOAD_OK {type(policy).__name__}（{load_s:.1f}s）  lora_params={lora:,}  "
          f"device={next(policy.parameters()).device}")

    # ---------- 3. pre/post processor ----------
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.ckpt,
        preprocessor_overrides={
            "device_processor": {"device": str(dev)},
            "rename_observations_processor": {"rename_map": rename},
        },
    )
    print(f"[3/5] processor 装载完成；rename_map = {rename}")

    frame, desc = build_frame(args.obs, int(cfg.height), int(cfg.width))
    print(f"[4/5] 观测：{desc}")

    # ---------- 4. 跑控制环 ----------
    policy.reset()
    torch.cuda.reset_peak_memory_stats(dev) if dev.type == "cuda" else None
    lat: list[float] = []
    acts: list[np.ndarray] = []
    for i in range(args.steps):
        obs = prepare_observation_for_inference(dict(frame), dev, args.task, "seeed_b601_rs_follower")
        t = time.perf_counter()
        with torch.inference_mode():
            a = policy.select_action(pre(obs))
            a = post(a)
        lat.append((time.perf_counter() - t) * 1000.0)
        acts.append(a.squeeze(0).to(torch.float32).cpu().numpy())
        if i == 0 or lat[-1] > 500:
            print(f"      第 {i + 1:3d} 步 {lat[-1]:8.0f} ms" + ("   ← 换块重建" if i else "   ← 首块（含 UMT5 编码 + VAE + 全流程）"))

    acts_arr = np.asarray(acts, np.float32)
    lat_arr = np.asarray(lat)
    steady = lat_arr[lat_arr < np.percentile(lat_arr, 80)]
    refills = np.where(lat_arr > max(500.0, 5 * np.median(steady)))[0]

    print("-" * 78)
    print(f"[5/5] 延迟：共 {len(lat_arr)} 步 / {lat_arr.sum() / 1000:.1f}s "
          f"（p50 {np.median(lat_arr):.0f} ms，max {lat_arr.max():.0f} ms）")
    if len(refills):
        print(f"      换块次数 {len(refills)}，每次重建 {lat_arr[refills].mean():.0f} ms "
              f"（{lat_arr[refills].min():.0f}–{lat_arr[refills].max():.0f} ms）→ 每块 {chunk} 个动作")
        per_act = lat_arr[refills].mean() / chunk
        print(f"      折合 {per_act:.0f} ms/动作 → 30Hz（33.3 ms/动作）下慢放约 {per_act / 33.3:.1f}×")

    state = frame["observation.state"].astype(np.float32)
    print(f"\n  动作块（{len(acts_arr)} 步）：")
    print("  关节              首步     末步   min     max   行程   |Δ|均值   |a+state|均值")
    for j, name in enumerate(JOINTS):
        col = acts_arr[:, j]
        print(f"  {name:14s} {col[0]:8.1f} {col[-1]:7.1f} {col.min():7.1f} {col.max():7.1f} "
              f"{col.max() - col.min():6.1f} {np.abs(np.diff(col)).mean():8.2f} "
              f"{np.abs(col + state[j]).mean():12.1f}")
    print("  判读：本数据集 action ≈ -state（joint_directions 全 -1），所以"
          "「|a+state| 小 + 行程小」= 模型在输出保持不动，不是链路问题。")

    if dev.type == "cuda":
        print(f"\n  显存：峰值 {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB / "
              f"保留 {torch.cuda.max_memory_reserved(dev) / 2**30:.1f} GiB")

    if args.save_npz:
        np.savez_compressed(args.save_npz, actions=acts_arr, latency_ms=lat_arr, state=state)
        print(f"  动作已存：{args.save_npz}")

    print("\nLOADTEST_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
