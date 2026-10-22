#!/bin/bash
# 单 B601-RS SmolVLA 推理（base 策略：纯推理，不建数据集、不录帧、不落盘）
# 任务：单臂端起纸杯放上咖啡机托盘、拿方块按按钮等 4 秒、方块放回桌面、再把杯子端回桌面
#       （任务文本/相机/fps/方向修正与 self_scripts/b601_single/02_record.sh 采集完全一致）
#
# 用法：
#   bash run_inference_smolvla.sh
#
# 模型权重：/home/kf/LX/pai0/smolvla_b601_20260910_164106_20260924/checkpoints/090000/pretrained_model
#   （90000 步 SmolVLA，7 关节 = 单臂；2026-09-30 从 ubuntu-232 同步，12 文件 / 1.5G，双端 sha256 一致）
#   训练配置要点：type=smolvla，chunk_size=50，n_action_steps=50，num_steps=10（flow matching 去噪步数）；
#     输入 state(7) + hand/front/top(3x480x640)，MEAN_STD 归一化；训练数据 b601_20260910_164106（30Hz）
#   回退旧权重：/home/kf/dev/lerobot/model_weights/220_new_datasets_smolvla_model/80k_pretrained_model
#
# 需要把推理过程存成数据集时，改用带 --strategy.type=episodic + --dataset.* 的录制型脚本
# 环境/硬件检查/参数总览/命令拼装等公共逻辑见 _common.sh

set -e  # 遇到错误立即退出
set -o pipefail  # 管道中任一命令失败则整体失败（配合 tee 使用）

source "$(dirname "$(readlink -f "$0")")/_common.sh"

# ==================== 模型配置 ====================
MODEL_PATH="/home/kf/LX/pai0/smolvla_b601_20260910_164106_20260924/checkpoints/090000/pretrained_model"
TITLE="单臂推理（SmolVLA 90k）- B601-RS（make coffee / base 纯推理）"
RUN_TAG="make_coffee_smolvla_90k"
ACTION_TRACE_LOG="$HOME/dev/lerobot/self_scripts/_logs/action_smolvla_test.log"

# VLM 主干（SmolVLM2-500M-Video-Instruct）本地快照
# 本机无外网：模型 config.json 里是 Hub id "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"，
# SmolVLA 会拿它去做 AutoConfig/AutoProcessor/AutoTokenizer，离线时反复重试 huggingface.co。
#   VLM_CACHE：HF 标准缓存布局的父目录（内含 models--HuggingFaceTB--.../{blobs,refs,snapshots}），
#     推理时设为 HF_HUB_CACHE，Hub id 就能离线解析到本地。关键：模型自带的
#     policy_preprocessor.json 把 tokenizer_name 写死成 Hub id，而 processor pipeline 不吃
#     --policy.vlm_model_name 覆盖，只有靠缓存目录才能命中本地分词器。
#   VLM_PATH：上面的快照目录，同时用 --policy.vlm_model_name 显式指向它。
VLM_CACHE="/home/kf/dev/lerobot/model_weights"
VLM_PATH="$VLM_CACHE/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"

# SmolVLA 用 num_steps（flow matching 去噪步数），须与训练 config.json 的 num_steps=10 一致
# 想更快可降到 6（动作略糙），想更稳可升到 10 以上（更慢）
NUM_STEPS_ARG="--policy.num_steps=10"

# sync = 每个 chunk 用尽时在控制环里直接前向（会短暂卡顿，但能记录 model 列）；
# rtc  = 后台线程异步生成下一段动作，控制环不阻塞。SmolVLA 支持 RTC，且 10 步去噪 + 3 路相机
#        单次前向较贵，sync 容易在换 chunk 时抖动，故默认 rtc。
INFERENCE_TYPE="rtc"

# ==================== 推理前检查 ====================
b601_require_lerobot
b601_banner "$TITLE"

# 1) 模型权重
b601_check_model_files "$MODEL_PATH" \
  config.json model.safetensors policy_preprocessor.json policy_postprocessor.json

# 1b) VLM 主干本地权重（缺失则 AutoConfig/AutoProcessor/AutoTokenizer 会去联网，提前报错退出）
b601_require_files "$VLM_PATH" \
  config.json preprocessor_config.json processor_config.json tokenizer.json
[ -f "$VLM_CACHE/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/refs/main" ] || {
  echo "❌ 缺少 HF 缓存布局（tokenizer_name 无法离线解析）：$VLM_CACHE/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/refs/main"
  exit 1
}
echo "✅ VLM 主干（SmolVLM2-500M）本地路径：$VLM_PATH"

# 2) 动作追踪包装器
b601_check_trace_wrapper

# 3) 硬件
b601_check_can
b601_check_cameras
b601_check_calibration

# ==================== 推理参数总览 ====================
echo ""
echo "=========================================="
echo "推理参数总览（SmolVLA 90k）"
echo "=========================================="
echo "模型路径：$MODEL_PATH"
echo "VLM 主干：$VLM_PATH（本地快照，不走 huggingface.co）"
echo "机器人类型：seeed_b601_rs_follower（单臂 B601-RS，can0）"
echo "策略：base（纯推理，不录制、不保存数据）"
echo "推理方式：${INFERENCE_TYPE}（smolvla 支持 RTC；sync 会阻塞控制环、但可记录 model 动作列）"
echo "去噪步数：${NUM_STEPS_ARG:-（用 config.json 默认值）}"
b601_print_run_params
b601_print_robot_notes
b601_print_trace_summary
echo "录制/保存数据：❌ 否（base 策略不录制）"
echo "=========================================="
echo ""

b601_confirm_zero
b601_start_run "开始单臂模型推理（SmolVLA 90k）..." "$RUN_TAG"

# 离线模式：模型与 VLM 主干/分词器全部走本地（HF_HUB_CACHE 让写死的 Hub id tokenizer_name
# 命中本地缓存，否则报 "Couldn't instantiate the backend tokenizer"）
b601_export_offline_hf "$VLM_CACHE"

b601_run_rollout \
  --strategy.type=base \
  --inference.type="$INFERENCE_TYPE" \
  --policy.path="$MODEL_PATH" \
  --policy.vlm_model_name="$VLM_PATH" \
  $NUM_STEPS_ARG

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ 单臂推理结束（SmolVLA 90k / base 模式）"
echo "=========================================="
echo "本次未保存任何推理数据（base 策略不录制、不落盘）。"
echo "完整日志：$LOG_FILE"
echo "=========================================="
