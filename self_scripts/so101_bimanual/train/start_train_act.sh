#!/bin/bash
# 启动训练（支持从checkpoint继续训练）
# so101_bimanual 双臂 make_coffee 任务的 ACT 训练（自旧仓库 stage3_train/dataset_new_makecoffee_train 迁移）
SELF_SCRIPTS_DIR=$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)
cd "$SELF_SCRIPTS_DIR"

# 激活 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "lerobot" ]; then
    echo "激活 lerobot conda 环境..."
    eval "$(conda shell.bash hook)"
    conda activate lerobot
fi

# 使用HF国内镜像，避免 huggingface.co 网络不可达
export HF_ENDPOINT=https://hf-mirror.com

# 从头训练：将下方任一 CHECKPOINT_CONFIG 取消注释即可改为续训
# CHECKPOINT_CONFIG="output_lerobot_train/put_ball2cup/checkpoints/last/pretrained_model/train_config.json"
# CHECKPOINT_CONFIG="output_lerobot_train/demo_data_dual/checkpoints/last/pretrained_model/train_config.json"
# CHECKPOINT_CONFIG="output_lerobot_train/cap_pen_and_put_coffee_cup_button_20260826_232220/checkpoints/last/pretrained_model/train_config.json"


if [ -f "$CHECKPOINT_CONFIG" ]; then
    echo "检测到checkpoint，将从上次训练继续（step=2000）"
    accelerate launch --multi_gpu --num_processes=8 $(which lerobot-train) --config_path="$CHECKPOINT_CONFIG" --resume=true
else
    echo "未检测到checkpoint，开始新训练"
    # accelerate launch --multi_gpu --num_processes=8 $(which lerobot-train) --config_path=train_config_act.json
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch  --num_processes=8 $(which lerobot-train) --config_path=so101_bimanual/train/act_train_config.json
fi