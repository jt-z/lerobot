#!/bin/bash
# 启动 Pi0.5 训练（使用预训练模型）
# so101_bimanual 双臂 cap_pen 任务的 PI05 LoRA 微调（自旧仓库 stage3_train/dataset1_makecoffee_train 迁移）
SELF_SCRIPTS_DIR=$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)
cd "$SELF_SCRIPTS_DIR"

# 训练输出目录（2026-08-25 起从头训练用新目录，与旧的 3000 步产物 task2_make_coffee/ 分开）
OUTPUT_DIR="output_lerobot_train/pi05_cap_pen_lora"
CHECKPOINT_CONFIG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
# export HF_ENDPOINT=https://hf-mirror.com

# 续训总步数：从 checkpoint 续训到 20000 步（checkpoint 里的 train_config.json 写死了旧 steps，
# 必须在 CLI 覆盖，否则 range(3000, 3000) 会一步不训立即结束）
RESUME_STEPS=20000

if [ -f "$CHECKPOINT_CONFIG" ]; then
    echo "检测到checkpoint，将从上次训练继续（续训至 ${RESUME_STEPS} 步）"
    accelerate launch --multi_gpu --num_processes=8 $(which lerobot-train) \
        --config_path="$CHECKPOINT_CONFIG" \
        --resume=true \
        --steps=$RESUME_STEPS \
        --output_dir=$OUTPUT_DIR
else
    echo "未检测到checkpoint，开始新训练（从头训练至 20000 步，使用预训练权重）"
    accelerate launch --multi_gpu --num_processes=8 $(which lerobot-train) \
        --policy.pretrained_path=/home/ksa/.cache/modelscope/models/lerobot--pi05_base/snapshots/master \
        --config_path="$SELF_SCRIPTS_DIR/so101_bimanual/train/configs/pi05_train_config_20000steps.json"
fi