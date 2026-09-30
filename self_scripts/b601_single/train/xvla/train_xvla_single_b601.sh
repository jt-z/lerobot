#!/bin/bash
# 单 B601-RS 单臂 X-VLA 微调（Phase II 域适配）
# 任务数据：$HOME/LX/pai0/b601_data/<session>（7 关节 = 6 关节 + 夹爪，30Hz，hand/front/top 三路相机）
# 底座：lerobot/xvla-base（Florence-2 + soft prompt + flow matching，0.9B）
#
# 为什么必须微调：
#   xvla-base 的 domain_id 表里没有 Seeed/B601，默认输出 20 维 ee6d（双臂末端位姿），
#   直接推理 ≈ 随机动作。Phase II 域适配 = 新增 soft prompt 并在目标数据上训练，
#   官方建议不冻结 VLM 编码器、使用 bfloat16 防 OOM。
#
# 单臂关键点：
#   X-VLA 默认 action_mode=ee6d（20 维双臂末端）确实"看起来像双臂"，
#   但框架支持任意维度：--policy.action_mode=auto 会按数据集真实动作维度自动
#   7 → pad 到 20 → 推理时再裁回 7，无需任何代码改动。
#
# 微调完成后，推理脚本见：
#   self_scripts/b601_single/inference/pretrained_models/run_inference_single_b601_make_coffee_xvla.sh
#
# 用法（脚本内部全用绝对路径，任意 cwd 均可）：
#   bash self_scripts/b601_single/train/xvla/train_xvla_single_b601.sh
#   换数据集：bash self_scripts/b601_single/train/xvla/train_xvla_single_b601.sh <session 名>
#
# 创建日期：2026-09-10

set -e
set -o pipefail

# ==================== 环境检查 ====================
if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "未找到 lerobot-train，尝试激活 lerobot conda 环境..."
  for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      source "$base/etc/profile.d/conda.sh"
      conda activate lerobot
      break
    fi
  done
fi
if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "❌ 错误：lerobot-train 不可用"
  echo "   请手动执行: conda activate lerobot"
  exit 1
fi
echo "✅ lerobot-train 可用"

# ==================== 配置 ====================
# 数据集：默认取最近一次采集 session，也可用命令行第一个参数指定（不带路径）
DATA_ROOT="$HOME/LX/pai0/b601_data"
DATASET_NAME="${1:-b601_20260910_164106}"
DATASET_ROOT="$DATA_ROOT/$DATASET_NAME"
# 与采集时 --dataset.repo_id 保持一致（本机 root 已给定，不会联网拉取）
DATASET_REPO_ID="$DATASET_NAME"

# 底座权重（HF 仓库 id 或本地目录）
XVLA_BASE="lerobot/xvla-base"

# HF 缓存目录（xvla-base + facebook/bart-large tokenizer 都缓存在这里）
HF_CACHE="$HOME/lerobot_weights/xvla"
# 国内镜像（训练需要联网下载底座；改成本地目录后可置 false）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_CACHE="$HF_CACHE"

# 是否先下载底座与 tokenizer（已缓存过可改为 false 跳过）
DOWNLOAD_ASSETS=true

# 训练输出目录（lerobot-train 要求 output_dir 不存在，否则报 FileExistsError）
OUTPUT_DIR="$HOME/dev/lerobot/model_weights/xvla_model/${DATASET_NAME}_xvla_auto"
JOB_NAME="b601_single_xvla"

# ==================== 超参 ====================
STEPS=10000           # 当前 10 集 / 约 3.1 万帧，1 万步约 2.5 个 epoch；数据加集后可同步加大
BATCH_SIZE=8
NUM_WORKERS=4
SAVE_FREQ=1000        # 每 1k 步存一个 checkpoint（便于挑中间权重）
LOG_FREQ=50
LR=1e-4
# 留出 10% episode 做离线 eval loss（eval_steps>0 要求 dataset.eval_split>0）；不需要时置空
EVAL_ARGS="--dataset.eval_split=0.1 --eval_steps=500"

# ==================== 数据集检查 ====================
echo ""
echo "1. 检查数据集..."
if [ ! -d "$DATASET_ROOT" ]; then
  echo "❌ 数据集不存在：$DATASET_ROOT"
  echo "   可用 session："
  ls -d "$DATA_ROOT"/b601_* 2>/dev/null | sed 's|.*/||'
  exit 1
fi
DATASET_INFO=$(python -c "
import json
i = json.load(open('$DATASET_ROOT/meta/info.json'))
print(i['total_episodes'], i['total_frames'], i['fps'], i['robot_type'])
" 2>/dev/null) || { echo "❌ 无法读取 $DATASET_ROOT/meta/info.json"; exit 1; }
read -r EPISODES FRAMES FPS ROBOT_TYPE <<< "$DATASET_INFO"
echo "✅ $DATASET_ROOT"
echo "   episodes=$EPISODES  frames=$FRAMES  fps=$FPS  robot_type=$ROBOT_TYPE"

if [ "$ROBOT_TYPE" != "seeed_b601_rs_follower" ]; then
  echo "⚠️  机器人类型不是 seeed_b601_rs_follower（当前 $ROBOT_TYPE），请确认选对了数据集"
fi
if [ "$EPISODES" -lt 20 ]; then
  echo "⚠️  只有 $EPISODES 集，微调能跑通但泛化会很差，建议继续采集到 30~50 集再训"
fi

# X-VLA 是语言条件模型：文本提示来自数据集 meta/tasks.parquet
TASK_STR=$(python -c "
import pandas as pd
t = pd.read_parquet('$DATASET_ROOT/meta/tasks.parquet')
print('|'.join([str(i) for i in t.index.tolist()]))
" 2>/dev/null || true)
echo "   任务文本：${TASK_STR:-（空）}"
if [ -z "$TASK_STR" ]; then
  echo ""
  echo "⚠️  警告：数据集里的任务文本为空字符串（采集时 self_scripts/b601_single/02_record.sh 的 TASK_DESCRIPTION 未填）。"
  echo "    X-VLA 是语言条件模型，空 prompt 会让语言分支失效，只能靠图像+本体状态学任务。"
  echo "    建议：先在 self_scripts/b601_single/02_record.sh 填好 TASK_DESCRIPTION 再补采，或改用带任务文本的数据集。"
  echo "    推理脚本的 --dataset.single_task 必须与此处文本完全一致。"
fi

# ==================== 下载底座与 tokenizer ====================
echo ""
echo "2. 准备底座权重与 tokenizer..."
mkdir -p "$HF_CACHE"
if [ "$DOWNLOAD_ASSETS" = true ]; then
  # XVLA 的 Florence-2 主干权重在 checkpoint 内，无需单独下载；
  # 但语言分支用 AutoTokenizer 加载 facebook/bart-large，必须缓存到本地。
  echo "   ↓ $XVLA_BASE"
  hf download "$XVLA_BASE"
  echo "   ↓ facebook/bart-large（仅 tokenizer 文件）"
  hf download facebook/bart-large \
    --include "tokenizer*" --include "vocab.json" --include "merges.txt" \
    --include "special_tokens_map.json" --include "added_tokens.json" --include "config.json"
else
  echo "   （DOWNLOAD_ASSETS=false，跳过下载）"
fi
if [ ! -d "$HF_CACHE/models--facebook--bart-large" ] && [ ! -d "$HF_CACHE/models--lerobot--xvla-base" ]; then
  echo "⚠️  缓存目录里没找到 xvla-base / bart-large，若训练时报离线错误请把 DOWNLOAD_ASSETS 改为 true"
fi

if [ -e "$OUTPUT_DIR" ]; then
  echo ""
  echo "❌ 输出目录已存在：$OUTPUT_DIR"
  echo "   lerobot-train 不会覆盖已有目录。请改名 OUTPUT_DIR，或用 --resume 续训。"
  exit 1
fi

# ==================== 参数总览 ====================
echo ""
echo "=========================================="
echo "X-VLA 单臂微调参数总览"
echo "=========================================="
echo "数据集：$DATASET_ROOT（$EPISODES 集 / $FRAMES 帧 / ${FPS}Hz）"
echo "底座：$XVLA_BASE"
echo "输出：$OUTPUT_DIR"
echo "动作空间：action_mode=auto（7 关节 → pad 20 → 推理裁回 7）"
echo "训练：steps=$STEPS  batch_size=$BATCH_SIZE  lr=$LR  dtype=bfloat16"
echo "Eval：${EVAL_ARGS:-（关闭）}"
echo "HF 缓存：$HF_CACHE"
echo "=========================================="
echo ""
read -p "确认后按 ENTER 开始训练，Ctrl+C 取消..." dummy

# ==================== 开始训练 ====================
echo ""
echo "🚀 开始 X-VLA 微调..."
echo ""

LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_xvla_single_b601_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志：$LOG_FILE"
echo ""

# 参数说明：
#   --policy.path            : 从 xvla-base 初始化（Phase II 域适配，不是从头训）
#   --policy.action_mode=auto: 按数据集真实动作维度自动 pad/裁剪，单臂 7 维可直接用
#   --policy.max_action_dim  : 模型侧动作维度上限（默认即 20，显式写出便于核对）
#   --policy.dtype=bfloat16  : 官方建议，0.9B 底座全精度易 OOM
#   --policy.push_to_hub=false: 本机不推 Hub（默认 true，不关会因缺 repo_id 报错）
#   冻结开关保持默认（freeze_vision/language=false, train_policy_transformer/soft_prompts=true）
#   = Phase II 的正确配置，故无需显式传参
#   domain_id 由预处理器固定为 0（Bridge），训练与推理一致；B601 属于新域，
#   若后续要独立分配域号需改 XVLAAddDomainIdProcessorStep 的默认值（本脚本未改动）
lerobot-train \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.path="$XVLA_BASE" \
  --policy.action_mode=auto \
  --policy.max_action_dim=20 \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --steps=$STEPS \
  --batch_size=$BATCH_SIZE \
  --num_workers=$NUM_WORKERS \
  --save_checkpoint=true \
  --save_freq=$SAVE_FREQ \
  --log_freq=$LOG_FREQ \
  --policy.optimizer_lr=$LR \
  --wandb.enable=false \
  $EVAL_ARGS 2>&1 | tee "$LOG_FILE"

# ==================== 训练完成 ====================
echo ""
echo "=========================================="
echo "✅ X-VLA 微调结束"
echo "=========================================="
echo "权重目录：$OUTPUT_DIR/checkpoints/<step>/pretrained_model"
echo "  （last 软链接指向最新 checkpoint；把该 pretrained_model 目录填进推理脚本的 MODEL_PATH）"
echo "完整日志：$LOG_FILE"
echo "=========================================="
