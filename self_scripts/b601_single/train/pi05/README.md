# Pi0.5 单臂 B601 训练（make coffee 任务）

## 目录结构

```
pi05/
├── README.md                  ← 本文件
├── retrain_r64/               ← ✅ 当前使用的配方（全量重训，修正版）
│   ├── pi05_train_config_r64.json
│   └── start_train_pi05_r64.sh
├── old_150k/                  ← ⚠️ 已废弃：原配方 + 150k 步那次的产物
│   ├── pi05_train_config.json
│   └── start_train_pi05.sh
└── bench/                     ← 离线诊断工具（不属于训练流程）
    ├── offline_action_error.py
    └── overfit_test.sh
```

## 为什么要重训

原配方（`old_150k/`）用 LoRA `r=16 / lr=2.5e-5` 跑了 **150,000 步**，训练 loss 从 0.35 降到 0.021，但**离线动作误差输给「输出常数」的平凡解**：

| 权重 | 整体动作 MAE | MAE / 平凡解 |
|---|---|---|
| 原配方 150k（`old_150k`） | 10.92° | **1.09（比平凡解还差）** |
| SmolVLA 90k（参考） | 5.31° | 0.53 |
| 修正配方 3000 步（过拟合探针） | **2.55°** | **0.25** |

实机表现也印证了：30k / 50k / 100k 权重推理时机械臂只是上下摆动，完全没学到轨迹。

`bench/overfit_test.sh` 做过对照实验（同一架构、同一份数据、同样冻结 VLM 只训 expert，只改 LoRA rank 与 lr），结论是：

> **不是「单臂 + 冻结 VLM + 只微调 MoE」学不动，而是原配方 lr 低了约 40 倍、rank 偏小。**

## 新配方改了什么（只有 3 处，其余逐字相同）

| 项 | old_150k | retrain_r64 |
|---|---|---|
| `peft.r` | 16 | **64** |
| `policy.optimizer_lr` | 2.5e-5 | **1e-3** |
| `policy.scheduler_decay_lr` | 2.5e-6 | **1e-4** |

依据：`docs/source/peft_training.mdx` —— LoRA 的 lr 应比全参微调放大 ~10 倍（1e-4 → 1e-3）。

注意：本仓库的 `PeftConfig` **不暴露 `lora_alpha`**，α 恒为 PEFT 默认 8（r=64 时有效缩放 α/r=0.125），与官方示例的 `r64/α64` 不完全等价。

其余关键设置两版一致：底座 `lerobot/pi05_base`、`freeze_vision_encoder=true`、`train_expert_only=false`、QUANTILES 归一化、相机映射 `front/hand/top → base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb`。

## 怎么跑

```bash
# 建议先 dry-run，确认数据校验、选卡、训练量推导都正常
bash self_scripts/b601_single/train/pi05/retrain_r64/start_train_pi05_r64.sh --dry-run

# 正式训练（共用机器上务必显式指定卡，否则 auto 会占用全部可用卡）
GPU_LIST=0,1,2,3 bash self_scripts/b601_single/train/pi05/retrain_r64/start_train_pi05_r64.sh
```

输出：`self_scripts/output_lerobot_train/pi05_b601_20260910_164106_20260930_r64_lr1e3/`
日志：`self_scripts/output_lerobot_train/logs/pi05_b601_20260910_164106_20260930_r64_lr1e3.log`

常用环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `GPU_LIST` | `auto` | **共用机器上务必显式指定**，如 `0,1,2,3` |
| `NUM_PASSES` | `2` | steps = NUM_PASSES × 总帧数 ÷ (batch_size × 卡数) |
| `STEPS` | 空 | 直接指定绝对步数，优先于 `NUM_PASSES` |
| `QUIET_MINS` | `5` | 数据集传输静默期 |

`--resume` 续训、`--skip-data-check` 跳过校验（不推荐）、`--dry-run` 只检查不启动。

## 训练完怎么判断好不好

不要只看 loss（本数据 `action[t] ≈ -state[t]`，且 state 被写进 prompt，**低 loss 可以靠「抄当前状态」拿到**）。用离线误差脚本，和 `hold`（`-state[0]` 常数）平凡解比：

```bash
PYTHONPATH=/home/ksa/lerobot/src CUDA_VISIBLE_DEVICES=<空闲卡> \
  /home/ksa/.conda/envs/lerobot/bin/python \
  self_scripts/b601_single/train/pi05/bench/offline_action_error.py \
  --ckpt <checkpoint 目录>/pretrained_model --episodes 0,40,120 --samples 4
```

判读：MAE **明显小于** `hold` 才说明学到了（参考 SmolVLA 比值 0.53）；接近或大于 `hold` 就是没学到。

## 归档说明

`old_150k/` 里的脚本只是留档，路径已按归档后的层级修正过，但它指向的 `output_dir` 已存在，直接跑会被「输出目录已存在」拦下 —— 这符合预期。要复现旧配方请改 `output_dir`。

`bench/` 是诊断工具，不参与训练；`overfit_test.sh` 默认读的仍是 `old_150k/pi05_train_config.json`（它本来就是为验证原配方而写）。
