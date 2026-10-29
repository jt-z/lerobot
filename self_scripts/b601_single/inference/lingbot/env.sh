#!/usr/bin/env bash
# LingBot-VA（lerobot 原生策略 type=lingbot_va）推理公共环境。
# 只导出变量，不做检查、不打印、不联网；由 preflight.py / smoke_load.py / run_lingbot.sh 使用。
#
# 权重三段（缺一不可，2026-10-23 核实）：
#   ① LoRA adapter  /home/kf/LX/pai0/lingbot/050000/pretrained_model   （A800-57 训练产物，r=16/alpha=32）
#   ② LeRobot 基座  /home/kf/LX/pai0/lingbot/lingbot_va_base           （只有 5B transformer 的 model.safetensors，10.2 GB）
#   ③ diffusers 冻结模块 /home/kf/LX/pai0/lingbot/lingbot-va-base      （vae/ + text_encoder/ + tokenizer/，约 20 GB）

export B601_LINGBOT_ROOT="${B601_LINGBOT_ROOT:-/home/kf/LX/pai0/lingbot}"

export LINGBOT_STEP="${LINGBOT_STEP:-050000}"
export LINGBOT_CKPT="${LINGBOT_CKPT:-$B601_LINGBOT_ROOT/$LINGBOT_STEP/pretrained_model}"
export LINGBOT_BASE="${LINGBOT_BASE:-$B601_LINGBOT_ROOT/lingbot_va_base}"
export LINGBOT_WAN="${LINGBOT_WAN:-$B601_LINGBOT_ROOT/lingbot-va-base}"

# ---- 运行时 ----
export LINGBOT_PY="${LINGBOT_PY:-/home/kf/miniconda3/envs/lerobot/bin/python}"
# 注意力后端：checkpoint 里存的是 flex（训练用，需要 torch flex-attention + compile）；
# 推理一律改 torch（PyTorch SDPA，永远可用）。改成 flex 前先看 README「已知坑」第 2 条。
export LINGBOT_ATTN_MODE="${LINGBOT_ATTN_MODE:-torch}"

# UMT5-XXL 文本编码器跑在 CPU 上（config.text_encoder_device=cpu，省 11 GB 显存）。
# 线程数被压成 1 时一次前向要几分钟（57 实测：1 线程 >240 s / 16 线程 21.9 s / GPU 0.11 s）。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

# 本机无外网：VAE / UMT5 / tokenizer 全部走 LINGBOT_WAN 本地目录，禁止再去 huggingface.co
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/kf/dev/lerobot/model_weights}"

# 任务文本，必须与训练数据集 b601_repair_clean_v2/meta/tasks.parquet 逐字一致（训练集只有 1 条 task）。
# run_lingbot.sh 会拿它和 _common.sh 的 TASK_DESCRIPTION（= 采集脚本用的那句）逐字比对，不一致直接退出。
export LINGBOT_TASK="Pick up the paper cup, place it on the silver tray of the coffee machine, pick up the cube, press the button with the cube (red light on), wait about 4 seconds, release the button (red light off), put the cube on the table first, then move the cup from the coffee machine to the table"

# 相机 key 重命名（robot → policy）。必须与 checkpoint 内
# policy_preprocessor.json 的 rename_observations_processor.rename_map 逐字一致。
# 注意：cam_left_wrist 接的是 hand（腕部），cam_right_wrist 接的是 top（顶部），训练端就是这么映射的，别"修正"。
export LINGBOT_CAM_RENAME_MAP='{"observation.images.front": "observation.images.cam_high", "observation.images.hand": "observation.images.cam_left_wrist", "observation.images.top": "observation.images.cam_right_wrist"}'
