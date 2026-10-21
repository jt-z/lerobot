#!/usr/bin/env python3
"""b601 π0.5 单次推理：给一组观测（state + 三路图像），输出 50 步动作块。

用途：真机接入前的“喂一帧看看”验证；也可在采集回放时逐帧对比。

用法：
  source b601_pi05_env.sh
  $PY infer_once.py --model 330 \
      --state -3.7 99.3 60.2 20.0 -13.5 -11.2 142.0 \
      --front /path/front.png --hand /path/hand.png --top /path/top.png \
      --prompt "Pick up the paper cup, ..." --out once_330.npz

说明：
  - state 单位=度、7 维（B601 单臂 7 关节），顺序与数据集 observation.state 一致
  - 输出的 50×7 动作块同样是“角度、且是数据集原有的负号约定”（action ≈ −state），直接喂给 B601 的控制器即可
  - 每个动作块执行前 5 步（execute_steps=5），然后重新观测——这与训练时的口径一致
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

HORIZON = 50


def load_image(path: Path) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["330", "462"], default="330")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--state", nargs=7, type=float, required=True)
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--hand", type=Path, required=True)
    ap.add_argument("--top", type=Path, required=True)
    ap.add_argument("--prompt", default="Pick up the paper cup, place it on the silver tray of the coffee machine, "
                                       "pick up the cube, press the button with the cube (red light on), wait about "
                                       "4 seconds, release the button (red light off), put the cube on the table "
                                       "first, then move the cup from the coffee machine to the table")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("once.npz"))
    a = ap.parse_args()

    from openpi.policies.policy_config import create_trained_policy
    from b601_config import register

    base = Path(os.environ["B601_HOME"])
    ckpt = a.ckpt or str(base / "checkpoints" / ("330_24653" if a.model == "330" else "462_34750"))
    cfg = register()
    policy = create_trained_policy(cfg, Path(ckpt), pytorch_device="cuda")

    obs = dict(
        state=np.asarray(a.state, dtype=np.float32),
        images=dict(front=load_image(a.front), hand=load_image(a.hand), top=load_image(a.top)),
        prompt=a.prompt,
    )
    noise = np.random.default_rng(a.seed).normal(size=(HORIZON, 32)).astype(np.float32)
    out = policy.infer(obs, noise=noise)
    actions = np.asarray(out["actions"], dtype=np.float32)
    print("checkpoint:", ckpt)
    print("action chunk:", actions.shape, "dtype", actions.dtype)
    np.set_printoptions(precision=2, suppress=True)
    print("前 5 步（将直接执行）:\n", actions[:5, :7])
    np.savez(a.out, actions=actions, state=obs["state"], noise=noise)
    print("saved:", a.out)
    print(json.dumps(dict(model=a.model, ckpt=ckpt, horizon=HORIZON, first5=actions[:5, :7].round(2).tolist()),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
