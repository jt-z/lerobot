"""pi05 离线动作误差验证：判断「模型没学到」还是「推理接口错位」。

用法：
    python self_scripts/b601_single/train/pi05/bench/offline_action_error.py \
        [--ckpt <checkpoints/150000/pretrained_model>] [--episodes 0,40,120] [--samples 4]

做什么：
    1. 按 lerobot-rollout 的同一条路径加载：checkpoint 自带 config + 适配器 + 自带 pre/post 处理器
       （rename_map、归一化统计、分词器都从 checkpoint 里来，不自己拼）；
    2. 从数据集指定 episode 里按比例抽若干帧，喂模型得到 50 步动作 chunk；
    3. 与数据集里同一起点的真实 action chunk 逐步比对，输出「误差 vs 预测步长」曲线；
    4. 三条对照：
       - hold  : 直接输出「起始帧动作不变」（本数据集 action ≈ -state，这是最强的平凡解）
       - 逐关节误差表：看是某几个关节全错，还是整体都错
       - 视觉消融：保持 state 不变、把图像换成另一帧的，看输出变不变（测视觉依赖）
       - 适配器对照：disable_adapter 前后输出差异（验证适配器确实生效）

怎么读结果：
    - model 明显优于 hold，且误差随步长增长的斜率与真实轨迹量级相当 → 模型学到了，问题在推理侧；
    - model ≈ hold（误差曲线基本重合）→ 模型退化成了「复制当前状态」，没学到轨迹；
    - 视觉消融的输出差异 ≈ 0 → 模型根本没用图像，等于闭眼拟合。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import lerobot.policies.pi05.processor_pi05  # noqa: F401  注册 pi05 的处理器步骤
import lerobot.policies.smolvla.processor_smolvla  # noqa: F401  注册 smolvla 的处理器步骤（对照用）
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_STATE

PI05_DIR = Path(__file__).resolve().parent.parent
SELF_SCRIPTS = PI05_DIR.parents[2]  # .../self_scripts
DEFAULT_CKPT = SELF_SCRIPTS / "output_lerobot_train" / "pi05_b601_20260910_164106_20260930" / "checkpoints" / "150000" / "pretrained_model"
DEFAULT_ROOT = "/data/share/b601_20260910_164106_20260930"
DEFAULT_REPO = "hellozjt/b601_20260910_164106_20260930"
JOINTS = ["j0", "j1", "j2", "j3", "j4", "j5", "j6"]


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- 加载
def load_policy(ckpt: Path, device: str, dtype: str):
    """照 lerobot-rollout 的方式加载：适配器(有则加) + 底座 + checkpoint 自带处理器。

    同时支持 pi05（LoRA 适配器）与 smolvla（全参微调，无适配器），便于做对照。
    """
    from lerobot.policies.factory import get_policy_class

    # 必须走基类：checkpoint 的 config.json 由 draccus.encode(self, PreTrainedConfig) 写出、带 "type" 键，
    # 用具体子类.from_pretrained() 会因 "type" 非法字段报 DecodingError。
    # 基类内部走 ChoiceRegistry，先按 type 解析出子类，再 pop("type") 后重新解析。
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = device
    if getattr(cfg, "dtype", None):  # pi05 有 dtype（bfloat16）；smolvla 无
        cfg.dtype = dtype
    if hasattr(cfg, "compile_model"):
        cfg.compile_model = False
    policy_cls = get_policy_class(cfg.type)
    log(f"策略类型: {cfg.type}   chunk={cfg.chunk_size}  n_action_steps={cfg.n_action_steps}")

    if (ckpt / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel

        peft_cfg = PeftConfig.from_pretrained(ckpt)
        log(f"适配器: {ckpt}")
        log(f"底座  : {peft_cfg.base_model_name_or_path}")
        base = policy_cls.from_pretrained(
            pretrained_name_or_path=peft_cfg.base_model_name_or_path, config=cfg
        )
        policy = PeftModel.from_pretrained(base, str(ckpt), config=peft_cfg).eval().to(device)
        n_lora = sum(p.numel() for n, p in policy.named_parameters() if "lora_" in n)
        log(f"LoRA 参数: {n_lora:,} 个")
        assert n_lora > 0, "没加载到 LoRA 参数，检查 checkpoint 目录"
    else:
        log(f"全参权重: {ckpt}")
        policy = policy_cls.from_pretrained(pretrained_name_or_path=str(ckpt), config=cfg).eval().to(device)

    pre, post = make_pre_post_processors(
        cfg, pretrained_path=str(ckpt), preprocessor_overrides={"device_processor": {"device": device}}
    )
    policy.reset()
    return policy, pre, post, cfg


def build_obs(item: dict, image_keys: list[str], task: str, robot_type: str, device: str) -> dict:
    """把数据集取出的单帧拼成 pre 处理器要的观测（图像转成 HWC uint8，与真实控制环一致）。"""
    obs = {OBS_STATE: item[OBS_STATE].numpy().astype(np.float32)}
    for k in image_keys:
        img = item[k]
        if img.ndim == 3:  # (C,H,W) float -> (H,W,C) uint8
            img = (img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        obs[k] = img
    return prepare_observation_for_inference(obs, torch.device(device), task, robot_type)


def predict_chunk(policy, pre, post, obs: dict, seed: int = 0) -> np.ndarray:
    """一次前向拿完整 chunk，返回未归一化的 (T, D) 动作（度）。"""
    torch.manual_seed(seed)  # flow 采样有随机性，固定住才可比
    with torch.inference_mode():
        prepared = pre(obs)
        chunk = policy.predict_action_chunk(prepared)
        chunk = post(chunk)
    chunk = chunk.detach().float().cpu().numpy()
    if chunk.ndim == 3:
        chunk = chunk[0]
    return chunk


# --------------------------------------------------------------------------- 抽样
def pick_frames(ds: LeRobotDataset, episodes: list[int], samples: int) -> list[tuple[int, int]]:
    """每个 episode 按比例抽 samples 帧，返回 (episode_index, 全局帧号)。"""
    eps = {e["episode_index"]: e for e in ds.meta.episodes}
    out: list[tuple[int, int]] = []
    for ep in episodes:
        if ep not in eps:
            log(f"  ⚠️ episode {ep} 不存在，跳过")
            continue
        lo, hi = int(eps[ep]["dataset_from_index"]), int(eps[ep]["dataset_to_index"])
        for i in range(samples):
            frac = (i + 0.5) / samples  # 避开首尾边界
            out.append((ep, int(lo + frac * (hi - lo - 1))))
    return out


# --------------------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="pi05 离线动作误差验证", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--dataset-root", default=DEFAULT_ROOT)
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--episodes", default="0,40,120", help="逗号分隔的 episode 序号")
    ap.add_argument("--samples", type=int, default=4, help="每个 episode 抽几帧")
    ap.add_argument("--task", default=None, help="语言指令；缺省取数据集里的任务文本")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--save-dir", type=Path, default=PI05_DIR / "bench" / "out")
    ap.add_argument("--no-adapter-compare", action="store_true", help="跳过 disable_adapter 对照")
    args = ap.parse_args()

    policy, pre, post, cfg = load_policy(args.ckpt, args.device, args.dtype)

    chunk_size = cfg.chunk_size
    ds_meta = LeRobotDataset(args.repo_id, root=args.dataset_root)  # 只为拿任务文本/机器人类型
    tasks = ds_meta.meta.tasks
    # meta.tasks 在不同版本里是 dict 或 pandas.DataFrame（索引即任务文本）
    if hasattr(tasks, "index"):
        task = args.task or str(tasks.index[0])
    else:
        task = args.task or next(iter(tasks.values()))
    robot_type = ds_meta.meta.robot_type or ""
    log(f"任务文本: {task!r}   robot_type={robot_type!r}   fps={ds_meta.meta.fps}")

    # delta_timestamps 让 action 一次取出整段 chunk（与训练时构造目标的机制一致）
    ds = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        delta_timestamps={ACTION: [i / ds_meta.meta.fps for i in range(chunk_size)]},
    )
    image_keys = [k for k in ds.meta.features if k.startswith("observation.images.")]
    log(f"数据集图像键: {image_keys}")

    episodes = [int(x) for x in args.episodes.split(",") if x.strip()]
    frames = pick_frames(ds, episodes, args.samples)
    log(f"抽样帧数: {len(frames)} → {[(e, i) for e, i in frames[:6]]}...\n")

    # ---- 逐帧前向，收集 预测 / 真实 / 平凡解 ----
    err_model, err_hold, all_pred, all_gt = [], [], [], []
    per_joint = np.zeros((len(JOINTS), 2))  # [sum|model-gt|, sum|hold-gt|]
    vision_gap = []
    adapter_gap = None

    for n, (ep, idx) in enumerate(frames):
        item = ds[idx]
        gt = item[ACTION].numpy().astype(np.float32)  # (T, 7) 真实动作，度
        state = item[OBS_STATE].numpy().astype(np.float32)
        obs = build_obs(item, image_keys, task, robot_type, args.device)
        pred = predict_chunk(policy, pre, post, obs)
        T = min(len(pred), len(gt))
        pred, gt = pred[:T], gt[:T]
        hold = np.repeat(item[ACTION].numpy()[0][None], T, axis=0)  # 起始动作不变

        dm, dh = np.abs(pred - gt).mean(1), np.abs(hold - gt).mean(1)
        err_model.append(dm), err_hold.append(dh)
        all_pred.append(pred), all_gt.append(gt)
        per_joint[:, 0] += np.abs(pred - gt).sum(0)
        per_joint[:, 1] += np.abs(hold - gt).sum(0)

        # 视觉消融：换另一帧的图像，state 不动
        other = ds[frames[(n + 1) % len(frames)][1]]
        obs_shuf = build_obs(other, image_keys, task, robot_type, args.device)
        obs_shuf[OBS_STATE] = obs[OBS_STATE].clone()
        gap = np.abs(predict_chunk(policy, pre, post, obs_shuf) - pred).mean()
        vision_gap.append(gap)

        if n == 0 and not args.no_adapter_compare and hasattr(policy, "disable_adapter"):
            with policy.disable_adapter():
                base_chunk = predict_chunk(policy, pre, post, obs)
            adapter_gap = float(np.abs(base_chunk - pred).mean())
            log(f"[适配器对照] 开/关适配器输出平均差 = {adapter_gap:.2f} 度")

        if n < 8 or n % 6 == 0:
            log(f"[{n:3d}] ep{ep:3d} frame{idx:6d}  model MAE={dm.mean():6.2f}°  hold MAE={dh.mean():6.2f}°  视觉消融差={gap:5.2f}°")

    err_model = np.mean(err_model, axis=0)
    err_hold = np.mean(err_hold, axis=0)
    per_joint /= len(frames) * chunk_size
    vision_gap = float(np.mean(vision_gap))

    # ---- 汇总 ----
    log("\n=== 每步 MAE（7 关节平均，度）===")
    steps = [0, 1, 2, 4, 9, 19, 29, 39, 49]
    steps = [s for s in steps if s < chunk_size]
    log("step    " + "".join(f"{s:>8}" for s in steps))
    log("model   " + "".join(f"{err_model[s]:>8.2f}" for s in steps))
    log("hold    " + "".join(f"{err_hold[s]:>8.2f}" for s in steps))

    log("\n=== 逐关节 MAE（度）===")
    log(f"{'joint':<8}{'model':>9}{'hold':>9}{'model/hold':>12}")
    for j, name in enumerate(JOINTS):
        ratio = per_joint[j, 0] / max(per_joint[j, 1], 1e-6)
        log(f"{name:<8}{per_joint[j, 0]:>9.2f}{per_joint[j, 1]:>9.2f}{ratio:>12.2f}")

    log(f"\n整体 MAE：model={err_model.mean():.2f}°   hold={err_hold.mean():.2f}°   比值={err_model.mean() / max(err_hold.mean(), 1e-6):.2f}")
    log(f"视觉消融：只换图像，输出平均变化 {vision_gap:.2f}°")
    if adapter_gap is not None:
        log(f"适配器贡献：开关适配器输出差 {adapter_gap:.2f}°")

    log("\n=== 判读 ===")
    if vision_gap < 0.5:
        log("✗ 视觉消融几乎无变化 → 模型基本没用图像，属于「闭眼拟合」，重训前先解决这个")
    elif err_model.mean() < 0.5 * err_hold.mean():
        log("✓ 模型明显优于平凡解且依赖图像 → 拟合是有效的，问题更可能在推理/下发侧")
    else:
        log("✗ 模型与平凡解误差相当 → 基本退化成「复制当前状态」，确实没学到轨迹")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.save_dir / "offline_action_error.npz", err_model=err_model, err_hold=err_hold, per_joint=per_joint)
    (args.save_dir / "offline_action_error.json").write_text(
        json.dumps(
            {
                "ckpt": str(args.ckpt),
                "mae_model": err_model.tolist(),
                "mae_hold": err_hold.tolist(),
                "per_joint_model": per_joint[:, 0].tolist(),
                "per_joint_hold": per_joint[:, 1].tolist(),
                "vision_gap": vision_gap,
                "adapter_gap": adapter_gap,
                "frames": frames,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"\n结果已保存: {args.save_dir}/offline_action_error.{{json,npz}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
