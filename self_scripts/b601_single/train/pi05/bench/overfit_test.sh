#!/bin/bash
# pi05 过拟合小实验：判断「原 LoRA 配方太弱」还是「这条路根本学不动」
#
# 背景
# ----
# pi05 LoRA(150k, lr 2.5e-5, r16/α8) 在整数据集上 loss 降到 0.021 就躺平，离线动作误差
# 还输给「输出常数」的平凡解（同脚本下 SmolVLA 能赢，见 bench/out/compare_pi05_vs_smolvla.png）。
# 本实验把数据集缩到 1 个 episode、把步数拉长，看**同一配方能不能把训练子集过拟合**：
#   - 能过拟合（loss→很低 且 离线误差→很小）→ 配方本身可行，整数据上的失败是容量/步数/调度问题
#   - 过拟合不了（loss 卡住、离线误差仍≈平凡解）→ 结构性学不动，与步数/lr 无关
#
# 两臂（唯一变量 = LoRA 配方，其余逐字相同）
# -------------------------------------------
#   A 原配方     : r=16, lr=2.5e-5, decay=2.5e-6   （= 150k 那次 run 的配置，α 取 PEFT 默认 8）
#   B 官方 LoRA  : r=64, lr=1e-3,  decay=1e-4      （peft_training.mdx 的 lr 放大 10 倍规则）
#     ⚠ 本仓库的 PeftConfig 不暴露 lora_alpha（α 恒为 PEFT 默认 8），所以 B 臂是
#       「更大 rank + 40× lr」的更强 LoRA，而不是逐字复刻 r64/α64。若 B 能过拟合，
#       「原配方太弱」已成立；若 B 也不行，才需要再补 α 变量。
#
# 数据子集：episodes=[0]（约 3k 帧）→ 3000 步 × 有效 batch 16 ≈ 16 pass，足够过拟合
# 用法：bash self_scripts/b601_single/train/pi05/bench/overfit_test.sh [--dry-run]
# 日志：bench/overfit/logs/{A,B}.log   权重：output_lerobot_train/pi05_overfit_{A,B}/

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
SELF_SCRIPTS_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)   # .../self_scripts
TRAIN_CFG="$SCRIPT_DIR/../old_150k/pi05_train_config.json"  # 复用原配方配置（已归档到 old_150k/），只加 CLI 覆盖

PI05_BASE=${PI05_BASE:-/home/ksa/.cache/modelscope/models/lerobot--pi05_base/snapshots/master}
EPISODES=${EPISODES:-"[0]"}
STEPS=${STEPS:-3000}
SAVE_FREQ=${SAVE_FREQ:-500}
WARMUP=${WARMUP:-100}
GPU_A=${GPU_A:-0,1}
GPU_B=${GPU_B:-2,3}
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

LOG_DIR="$SCRIPT_DIR/overfit/logs"
mkdir -p "$LOG_DIR"

die() { echo "✗ $*" >&2; exit 3; }

# ---------------- 环境 ----------------
[ -f "$TRAIN_CFG" ] || die "找不到 $TRAIN_CFG"
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "lerobot" ]; then
    eval "$(conda shell.bash hook)" || die "conda 不可用"
    conda activate lerobot || die "conda activate lerobot 失败"
fi
PY=$(which python) || die "找不到 python"
[ -d "$PI05_BASE" ] || die "找不到 pi05_base：$PI05_BASE"
cd "$SELF_SCRIPTS_DIR" || die "cd 失败"

echo "python      : $PY"
echo "训练配置    : $TRAIN_CFG"
echo "数据子集    : episodes=$EPISODES"
echo "步数/保存   : steps=$STEPS  save_freq=$SAVE_FREQ  warmup=$WARMUP"
echo "A 臂卡      : $GPU_A    B 臂卡: $GPU_B"
echo

launch() {   # launch <name> <gpu_list> <port> <extra args...>
    local name=$1 gpus=$2 port=$3; shift 3
    local out="output_lerobot_train/pi05_overfit_$name"
    local cfg_abs
    cfg_abs=$(python -c "import json;print(json.load(open('$TRAIN_CFG'))['output_dir'])")
    [ "$cfg_abs" = "$out" ] && die "$name 的 output_dir 与原始 run 冲突"

    local cmd=(accelerate launch --multi_gpu --num_processes=2 --main_process_port="$port" "$(which lerobot-train)"
        --policy.pretrained_path="$PI05_BASE"
        --config_path="$TRAIN_CFG"
        --policy.push_to_hub=false
        --dataset.episodes="$EPISODES"
        --output_dir="$out"
        --steps="$STEPS"
        --save_freq="$SAVE_FREQ"
        --policy.scheduler_decay_steps="$STEPS"
        --policy.scheduler_warmup_steps="$WARMUP"
        "$@")

    echo "── $name ──"
    echo "  CUDA_VISIBLE_DEVICES=$gpus ${cmd[*]}"
    if [ "$DRY_RUN" != 1 ]; then
        CUDA_VISIBLE_DEVICES="$gpus" HF_ENDPOINT=https://hf-mirror.com \
            setsid nohup "${cmd[@]}" >"$LOG_DIR/$name.log" 2>&1 &
        echo "  → PID $!  日志 $LOG_DIR/$name.log"
    fi
}

# A：原配方（r/lr 全部沿用配置里的值，不加任何覆盖）
launch A "$GPU_A" 29500
# B：更强 LoRA（只改 rank 与 lr）；端口必须与 A 不同，否则 accelerate 会 EADDRINUSE
launch B "$GPU_B" 29501 --peft.r=64 --policy.optimizer_lr=1e-3 --policy.scheduler_decay_lr=1e-4

echo
echo "tail -f $LOG_DIR/A.log   /   $LOG_DIR/B.log"
