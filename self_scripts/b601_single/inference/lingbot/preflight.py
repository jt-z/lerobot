#!/usr/bin/env python3
"""LingBot-VA 推理前预检：不加载权重、不占显存、不碰硬件，纯读文件 + 解析 json（秒级）。

查的是"换机/换 checkpoint 之后最容易静默错"的几处：
  * 三段权重是否齐全（adapter / LeRobot 基座 / diffusers 冻结模块）
  * adapter_config 与 config.json 里写死的路径是否还是训练机的 /home/ubuntu/...
  * checkpoint 的 rename_map 是否与脚本传的一致（不一致 → 相机与策略输入错配，不会报错只会跑歪）
  * 关键推理超参（chunk 长度、去噪步数、CFG、dtype、attn_mode）是否符合预期
  * 运行时依赖（diffusers/transformers/peft）是否齐

用法：
  ./preflight.py                      # 用 env.sh 里的默认 checkpoint
  LINGBOT_STEP=050000 ./preflight.py  # 指定别的步数
退出码：0 = 全部通过；1 = 有 FAIL（会打印可直接粘贴的修复命令）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import _env

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_rows: list[tuple[str, str]] = []


def say(level: str, msg: str) -> None:
    _rows.append((level, msg))
    icon = {OK: "✅", WARN: "⚠️ ", FAIL: "❌"}[level]
    print(f"{icon} {msg}")


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n:.1f}GB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=_env.ckpt())
    ap.add_argument("--base", default=_env.base())
    ap.add_argument("--wan", default=_env.wan())
    ap.add_argument("--rename-map", default=json.dumps(_env.rename_map()))
    args = ap.parse_args()

    ckpt, base, wan = args.ckpt, args.base, args.wan
    print(f"checkpoint : {ckpt}")
    print(f"LeRobot 基座: {base}")
    print(f"WAN 冻结模块: {wan}")
    print("-" * 78)

    # ---------- 1. adapter 目录文件清单 ----------
    need = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    if not os.path.isdir(ckpt):
        say(FAIL, f"adapter 目录不存在：{ckpt}")
        print_fix(ckpt, base, wan)
        return 1
    missing = [f for f in need if not os.path.isfile(os.path.join(ckpt, f))]
    if missing:
        say(FAIL, f"adapter 目录缺文件：{', '.join(missing)}")
    else:
        tot = sum(os.path.getsize(os.path.join(ckpt, f)) for f in need)
        say(OK, f"adapter 文件齐全（{len(need)} 个 / {human(tot)}）")
    if os.path.isfile(os.path.join(ckpt, "model.safetensors")):
        say(WARN, "目录里有 model.safetensors —— 这像是全量权重，不是 LoRA adapter；确认 use_peft 标志")

    # ---------- 2. adapter_config ----------
    ac = load_json(os.path.join(ckpt, "adapter_config.json"))
    base_in_ckpt = ac.get("base_model_name_or_path", "")
    if not os.path.isdir(base_in_ckpt):
        say(FAIL, f"adapter_config.base_model_name_or_path 不可达：{base_in_ckpt}")
    elif "/home/ubuntu" in base_in_ckpt:
        say(FAIL, f"adapter_config 里仍是训练机路径：{base_in_ckpt}")
    else:
        say(OK, f"LoRA adapter r={ac.get('r')} / alpha={ac.get('lora_alpha')} / "
                f"target_modules={len(ac.get('target_modules') or [])} 个；基座 = {base_in_ckpt}")

    # ---------- 3. config.json ----------
    cf = load_json(os.path.join(ckpt, "config.json"))
    if cf.get("type") != "lingbot_va":
        say(FAIL, f"config.type={cf.get('type')!r}，不是 lingbot_va")
    else:
        say(OK, f"policy.type=lingbot_va  use_peft={cf.get('use_peft')}  dtype={cf.get('dtype')}")
    wan_in_cfg = cf.get("wan_pretrained_path", "")
    if not os.path.isdir(wan_in_cfg) or "/home/ubuntu" in str(wan_in_cfg):
        say(FAIL, f"config.wan_pretrained_path 不可用：{wan_in_cfg}")
    else:
        say(OK, f"config.wan_pretrained_path = {wan_in_cfg}")

    apf, fps_ = int(cf.get("action_per_frame", 0)), int(cf.get("frame_chunk_size", 0))
    chunk = apf * fps_
    say(OK, f"action_per_frame={apf} × frame_chunk_size={fps_} → 每块 {chunk} 个动作"
            f"（首块只给 {(fps_ - 1) * apf} 个：第 0 帧是条件帧）")
    say(OK, f"去噪步数：视频 {cf.get('num_inference_steps')} / 动作 {cf.get('action_num_inference_steps')}"
            f"；CFG 视频 {cf.get('guidance_scale')} / 动作 {cf.get('action_guidance_scale')}"
            f"（视频 CFG>1 时每次前向都是双 batch）")
    cam = cf.get("obs_cam_keys") or []
    say(OK, f"策略侧相机顺序（宽度拼接，顺序敏感）：{cam}")
    say(OK, f"输入特征：{list((cf.get('input_features') or {}).keys())}  ← 没有 STATE，state 不参与推理")
    if str(cf.get("attn_mode")) == "flex":
        say(WARN, "config.attn_mode=flex（训练用）；推理请用 --policy.attn_mode=torch（run_lingbot.sh 默认已覆盖）")
    if cf.get("text_encoder_device") == "cpu":
        say(OK, "text_encoder_device=cpu（UMT5-XXL 不吃显存，但每集首次编码要几秒）")

    # ---------- 4. 三方路径一致 ----------
    if base_in_ckpt != base:
        say(WARN, f"env 的 LINGBOT_BASE({base}) 与 adapter_config 里的基座({base_in_ckpt}) 不一致；以 adapter_config 为准")
    if wan_in_cfg != wan:
        say(WARN, f"env 的 LINGBOT_WAN({wan}) 与 config 里的 WAN 路径({wan_in_cfg}) 不一致；以 config 为准")

    # ---------- 5. 冻结模块 / 基座 ----------
    for sub, files in {
        "vae": ["config.json", "diffusion_pytorch_model.safetensors"],
        "text_encoder": ["config.json", "model.safetensors.index.json"],
        "tokenizer": ["spiece.model", "tokenizer.json", "tokenizer_config.json"],
    }.items():
        d = os.path.join(wan_in_cfg, sub)
        miss = [f for f in files if not os.path.isfile(os.path.join(d, f))]
        if miss:
            say(FAIL, f"{d} 缺 {', '.join(miss)}")
        else:
            say(OK, f"{sub}/ 齐全")
    bs = os.path.join(base, "model.safetensors")
    if os.path.isfile(bs):
        say(OK, f"基座 transformer 权重 {human(os.path.getsize(bs))}：{bs}")
    else:
        say(FAIL, f"基座权重缺失：{bs}")

    # ---------- 6. rename_map 一致性（最贵的一处静默错） ----------
    pp = load_json(os.path.join(ckpt, "policy_preprocessor.json"))
    ckpt_map = {}
    for st in pp.get("steps", []):
        if st.get("registry_name") == "rename_observations_processor":
            ckpt_map = st.get("config", {}).get("rename_map") or {}
    try:
        env_map = json.loads(args.rename_map) if args.rename_map else {}
    except json.JSONDecodeError:
        env_map = {}
        say(FAIL, "env 的 LINGBOT_CAM_RENAME_MAP 不是合法 JSON")
    if ckpt_map == env_map:
        say(OK, f"rename_map 与 checkpoint 一致：{ckpt_map}")
    else:
        say(FAIL, "rename_map 与 checkpoint 不一致（相机与策略输入会错配）\n"
                 f"     checkpoint: {ckpt_map}\n"
                 f"     脚本/env  : {env_map}")
    missing_cam = set(cf.get("obs_cam_keys") or []) - set(ckpt_map.values())
    if missing_cam:
        say(FAIL, f"策略要的相机在 rename_map 的目标侧没有对应的：{missing_cam}")

    # ---------- 7. 运行时依赖（查的是真正跑推理的 LINGBOT_PY，不是当前解释器） ----------
    import subprocess

    py = _env.py()
    probe = ("import importlib,json;"
             "print(json.dumps({m: getattr(importlib.import_module(m), '__version__', '?') "
             "for m in ('torch', 'diffusers', 'transformers', 'peft')}))")
    try:
        out = subprocess.run([py, "-c", probe], capture_output=True, text=True, timeout=180)
        vers = json.loads(out.stdout) if out.returncode == 0 else {}
    except Exception:  # noqa: BLE001
        vers = {}
    if not vers:
        say(FAIL, f"{py} 里 import torch/diffusers/transformers/peft 失败：{(out.stderr or '').strip().splitlines()[-1:]}")
    else:
        say(OK, f"{py} 依赖版本：" + "  ".join(f"{k}={v}" for k, v in vers.items()))
        if not os.path.isdir(base):
            say(FAIL, f"基座目录不存在：{base}")

    n_fail = sum(1 for lv, _ in _rows if lv == FAIL)
    n_warn = sum(1 for lv, _ in _rows if lv == WARN)
    print("-" * 78)
    if n_fail:
        print(f"PREFLIGHT_FAIL  失败 {n_fail} / 警告 {n_warn}")
        print_fix(ckpt, base, wan)
        return 1
    print(f"PREFLIGHT_OK    {len(_rows)} 项通过（警告 {n_warn}）")
    return 0


def print_fix(ckpt: str, base: str, wan: str) -> None:
    print("""
若提示仍是训练机路径（/home/ubuntu/lxx/pai0/models/...），一次性改完（已做 .bak 备份）：
  cd {ckpt} && for f in adapter_config.json config.json train_config.json; do
      cp -a "$f" "$f.bak-$(date +%Y%m%d)"; done
  sed -i "s#/home/ubuntu/lxx/pai0/models/#{root}/#g" \\
      adapter_config.json config.json train_config.json
（root = {base} 和 {wan} 的公共父目录）
""".format(ckpt=ckpt, base=base, wan=wan,
           root=os.path.commonpath([base, wan])))


if __name__ == "__main__":
    sys.exit(main())
