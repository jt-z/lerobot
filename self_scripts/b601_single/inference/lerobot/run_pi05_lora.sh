#!/bin/bash
# 单 B601-RS π0.5 LoRA 推理（base 策略：纯推理，不建数据集、不录帧、不落盘）
# 任务：单臂端起纸杯放上咖啡机托盘、拿方块按按钮等 4 秒、方块放回桌面、再把杯子端回桌面
#       （任务文本/相机/fps/方向修正与 self_scripts/b601_single/02_record.sh 采集完全一致）
#
# 用法：
#   ./run_pi05_lora.sh                               # 默认 run（r64 重训）的 040000
#   ./run_pi05_lora.sh 038000                        # 同一个 run 别的步数（每 2000 步一个）
#   CKPT_ROOT=/home/kf/LX/pai0/pi05_b601_20260910_164106_20260930/checkpoints \
#     ./run_pi05_lora.sh 030000                      # 旧 r16 run（对比基线）
#   CKPT_ROOT=/home/kf/LX/pai0/pi05_overfit_B/checkpoints \
#     ./run_pi05_lora.sh 003000                      # 过拟合探针 B（r=64 / lr=1e-3）
#
# 权重形态（重要，别按 SmolVLA 的习惯套）：MODEL_PATH 指向 LoRA adapter，不是全量权重，
#   目录里【没有 model.safetensors】，只有 adapter_config.json + adapter_model.safetensors
#   + config.json + train_config.json + policy_{pre,post}processor*.json（14 文件，随 rank 16~62 MB）。
#   加载语义（rollout/context.py::_load_pretrained_policy）：use_peft=True 时先按
#   adapter_config.json 的 base_model_name_or_path 起 pi05 基座，再用 PeftModel 挂 adapter，
#   所以本脚本不需要 --policy.vlm_model_name（pi05 无此字段）。
#   本机依赖：pi05 基座 lerobot--pi05_base/snapshots/master（14.4 GB）、
#             PaliGemma 分词器 google--paligemma-3b-pt-224/snapshots/master（均在 model_weights/）
# checkpoint 从训练机拷来后，4 个 json 里写死的 /home/ksa/... 路径要先改成本机路径，
#   否则下面的预检会拦下并打印该跑的 sed 命令。
#
# 需要把推理过程存成数据集时，改用带 --strategy.type=episodic + --dataset.* 的录制型脚本
# 环境/硬件检查/参数总览/命令拼装等公共逻辑见 _common.sh

set -e  # 遇到错误立即退出
set -o pipefail  # 管道中任一命令失败则整体失败（配合 tee 使用）

source "$(dirname "$(readlink -f "$0")")/_common.sh"

# ==================== 模型配置 ====================
# 要跑的权重 = CKPT_ROOT（哪个 run 的 checkpoints 目录）+ STEP（第一个参数）
STEP="${1:-040000}"
CKPT_ROOT="${CKPT_ROOT:-/home/kf/LX/pai0/pi05_b601_20260910_164106_20260930_r64_lr1e3/checkpoints}"
MODEL_PATH="$CKPT_ROOT/$STEP/pretrained_model"
RUN_NAME="$(basename "$(dirname "$CKPT_ROOT")")"

# 加载该 checkpoint 需要的两个本机依赖（只用于预检；真正取路径是从 checkpoint 的 json 里读，
# 见下面第 1b 步，避免脚本与 checkpoint 内容说法不一致）
PI05_BASE_EXPECT="/home/kf/dev/lerobot/model_weights/lerobot--pi05_base/snapshots/master"
TOKENIZER_EXPECT="/home/kf/dev/lerobot/model_weights/google--paligemma-3b-pt-224/snapshots/master"

# 离线解析用的 HF 缓存根（本机无外网）。pi05 的 tokenizer_name 在 policy_preprocessor.json 里
# 写的是绝对路径，正常不吃这个变量；保留它可让任何残留的 Hub id 在本地命中，避免联网重试。
HF_CACHE="/home/kf/dev/lerobot/model_weights"

# 去噪步数：须与训练 config.json 的 num_inference_steps=10 一致（降到 6 更快、动作略糙）
NUM_INFERENCE_STEPS_ARG="--policy.num_inference_steps=10"

# sync = 换 chunk 时在控制环里直接前向（卡顿，但 action_trace 能记 model 列）；
# rtc  = 后台线程异步生成下一段动作，控制环不阻塞。pi05 声明 supports_rtc=True，10 步去噪 +
#        3 路相机单次前向较贵，故默认 rtc。⚠️ 本 checkpoint 训练时 rtc_config=null，此处是推理期
#        guided RTC（rtc.mode 默认 guided），不需要 RTC 训练过的权重；首次上机若抖动先改 sync。
INFERENCE_TYPE="rtc"

TITLE="单臂推理（pi05 LoRA）- B601-RS（make coffee / base 纯推理）"
RUN_TAG="make_coffee_pi05_${STEP}"
ACTION_TRACE_LOG="$HOME/dev/lerobot/self_scripts/_logs/action_pi05_test.log"

# ==================== 相机 key 重命名（robot → policy） ====================
# pi05 基座把三路相机命名为 base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb，训练即用这套名字
# （见 checkpoint 的 policy_preprocessor.json 的 rename_observations_processor.rename_map）。
# 不传 rename_map 时，rollout 预检会报 "Visual feature mismatch between policy and robot hardware"。
# ⚠️ 映射必须与训练 checkpoint 里的 rename_map 完全一致，否则相机与策略输入错配。
CAMERA_RENAME_MAP='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.hand": "observation.images.left_wrist_0_rgb", "observation.images.top": "observation.images.right_wrist_0_rgb"}'

# ==================== 推理前检查 ====================
b601_require_lerobot
b601_banner "$TITLE"

# 1) 模型权重（LoRA checkpoint 没有 model.safetensors，别照 SmolVLA 的清单查）
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 模型路径不存在：$MODEL_PATH"
  exit 1
fi
if [ -f "$MODEL_PATH/model.safetensors" ]; then
  echo "⚠️  检测到 model.safetensors —— 这看起来是全量权重而非 LoRA adapter。"
  echo "    若确实要跑全量权重（旧的 020000_b601_pi0 那种），本脚本的检查项需放宽；确认后继续。"
else
  b601_require_files "$MODEL_PATH" \
    adapter_config.json adapter_model.safetensors config.json \
    policy_preprocessor.json policy_postprocessor.json
  echo "✅ LoRA checkpoint 文件齐全：$MODEL_PATH"
fi

# 1b) 解析 checkpoint 内部引用，确认本机路径存在
#     （两道坎：①从训练机拷来没改路径 ②本机缺基座/分词器 —— 都提前报错，不要跑到加载一半才炸）
echo ""
CKPT_INTERNAL_CHECK=""
CKPT_INTERNAL_CHECK=$(python3 - "$MODEL_PATH" "$TOKENIZER_EXPECT" <<'PYEOF'
import json, os, sys

mp = sys.argv[1]
tok_hint = sys.argv[2]

def rd(p):
    with open(p) as f:
        return json.load(f)

try:
    ad = rd(os.path.join(mp, "adapter_config.json"))
    cf = rd(os.path.join(mp, "config.json"))
    pp = rd(os.path.join(mp, "policy_preprocessor.json"))
except Exception as exc:
    print("FAIL|无法解析 checkpoint 内的 json：%s" % exc)
    sys.exit(0)

tok = None
for st in pp.get("steps", []):
    if st.get("registry_name") == "tokenizer_processor":
        tok = st.get("config", {}).get("tokenizer_name")

items = [
    ("pi05 基座 (adapter_config.base_model_name_or_path)", ad.get("base_model_name_or_path")),
    ("config.pretrained_path", cf.get("pretrained_path")),
    ("PaliGemma 分词器 (tokenizer_name)", tok),
]
print("INFO|pi05 基座         : %s" % ad.get("base_model_name_or_path"))
print("INFO|config.pretrained : %s" % cf.get("pretrained_path"))
print("INFO|tokenizer_name    : %s" % tok)
print("INFO|use_peft=%s  type=%s  chunk_size=%s  n_action_steps=%s  num_inference_steps=%s"
      % (cf.get("use_peft"), cf.get("type"), cf.get("chunk_size"),
         cf.get("n_action_steps"), cf.get("num_inference_steps")))

bad = []
for name, path in items:
    if not path:
        bad.append("%s 为空" % name)
    elif not os.path.isdir(path):
        bad.append("%s 指向的目录不存在：%s" % (name, path))
    elif "/home/ksa" in path:
        bad.append("%s 仍是训练机的路径：%s" % (name, path))

if not os.path.isdir(tok_hint):
    bad.append("备选分词器目录也不存在：%s" % tok_hint)

if bad:
    print("FAIL|" + " ;; ".join(bad))
else:
    print("OK|全部本机可达")
PYEOF
)

echo "$CKPT_INTERNAL_CHECK" | sed -n 's/^INFO|/   /p'
if echo "$CKPT_INTERNAL_CHECK" | grep -q '^FAIL|'; then
  echo "❌ checkpoint 内部路径不可用："
  echo "$CKPT_INTERNAL_CHECK" | sed -n 's/^FAIL|/   /p' | tr ';;' '\n'
  echo ""
  echo "   若提示仍是 /home/ksa/.cache/modelscope/... 路径，先跑（16~62 MB，秒级）："
  echo "     sed -i \"s#/home/ksa/.cache/modelscope/models/#/home/kf/dev/lerobot/model_weights/#g\" \\"
  echo "       $MODEL_PATH/{adapter_config.json,config.json,policy_preprocessor.json,train_config.json}"
  exit 1
fi
echo "✅ checkpoint 内部引用的基座/分词器在本机可达"

# 2) 动作追踪包装器
b601_check_trace_wrapper

# 3) 硬件
b601_check_can
b601_check_cameras
b601_check_calibration

# ==================== 推理参数总览 ====================
echo ""
echo "=========================================="
echo "推理参数总览（pi05 / LoRA）"
echo "=========================================="
echo "训练步数：$STEP（run: $RUN_NAME）"
echo "模型路径：$MODEL_PATH"
# LoRA 的 r / alpha 直接从 checkpoint 的 adapter_config.json 读（r16 与 r64 混跑时不至于看错）
LORA_INFO=$(python3 - "$MODEL_PATH" <<'PYEOF'
import json, os, sys
try:
    d = json.load(open(os.path.join(sys.argv[1], "adapter_config.json")))
    print("r=%s / alpha=%s" % (d.get("r"), d.get("lora_alpha")))
except Exception as exc:
    print("(读 adapter_config.json 失败：%s)" % exc)
PYEOF
)
echo "权重类型：LoRA adapter（$LORA_INFO），基座按 adapter_config.json 自动加载"
echo "机器人类型：seeed_b601_rs_follower（单臂 B601-RS，can0）"
echo "策略：base（纯推理，不录制、不保存数据）"
echo "推理方式：${INFERENCE_TYPE}（pi05 支持 RTC；sync 会阻塞控制环、但可记录 model 动作列）"
echo "去噪步数：${NUM_INFERENCE_STEPS_ARG:-（用 config.json 默认值）}"
b601_print_run_params
echo "相机重命名：front→base_0_rgb  hand→left_wrist_0_rgb  top→right_wrist_0_rgb（与训练一致）"
b601_print_robot_notes
b601_print_trace_summary
echo "录制/保存数据：❌ 否（base 策略不录制）"
echo "=========================================="
echo ""

b601_confirm_zero
b601_start_run "开始单臂模型推理（pi05 $STEP）..." "$RUN_TAG"

# 离线模式：基座与分词器全部走本地，禁止再请求 huggingface.co
b601_export_offline_hf "$HF_CACHE"

b601_run_rollout \
  --strategy.type=base \
  --inference.type="$INFERENCE_TYPE" \
  --policy.path="$MODEL_PATH" \
  $NUM_INFERENCE_STEPS_ARG

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ 单臂推理结束（pi05 $STEP / base 模式）"
echo "=========================================="
echo "本次未保存任何推理数据（base 策略不录制、不落盘）。"
if [ "$ACTION_TRACE" = "1" ]; then
  echo "动作追踪日志：$ACTION_TRACE_LOG"
fi
echo "完整日志：$LOG_FILE"
echo "=========================================="
