#!/bin/bash
# 单 B601-RS pi05 推理：从臂 B601-RS（SocketCAN can0）
# 策略：base —— 纯推理，不建数据集、不录帧、不编码、不 finalize（推理数据不保存）
# 任务：单臂端起纸杯放上咖啡机托盘、拿方块按按钮等 4 秒、方块放回桌面、再把杯子端回桌面
#       （与 self_scripts/b601_single/02_record.sh 采集的任务描述/相机/fps/方向修正完全一致）
#
# 参照：
#   - run_inference_smolvla.sh：单臂纯推理脚本（本脚本的模板：环境/模型/硬件/校准检查、参数总览、action_trace）
#   - run_inference_ACT.sh：单臂 ACT 纯推理脚本（检查项与日志风格来源）
#   - so101_bimanual 旧的 run_inference_pi0.sh：pi05/pi0 系族的 num_inference_steps 与 RTC 用法来源
# 机器人类型：seeed_b601_rs_follower（单臂官方类型，见 lerobot_robot_seeed_b601 包）
#
# ================== 权重说明（重要，别按 SmolVLA 的习惯套） ==================
# MODEL_PATH 指向的是 **LoRA adapter**，不是全量权重，所以该目录里【没有 model.safetensors】：
#   pretrained_model/  : adapter_config.json + adapter_model.safetensors(5.4 MB) + config.json
#                        + train_config.json + README.md + policy_{pre,post}processor*.json/safetensors
#   training_state/    : optimizer/rng/scheduler 等（推理用不到）
#   合计 14 个文件 / 16 MB。训练配置要点（config.json）：
#     - type=pi05，chunk_size=50，n_action_steps=50，num_inference_steps=10（flow matching 去噪步数）
#     - LoRA：r=16 / alpha=8 / 1,352,704 参数，target_modules 含 gemma_expert 的 q/v_proj 与
#             model.{state_proj,action_in_proj,action_out_proj,time_mlp_in,time_mlp_out}
#     - 输入：observation.state(7) + observation.images.hand/front/top(3x480x640)，语言指令 = 上面的 task
#     - 输出：action(7)，STATE/ACTION 均 QUANTILES 归一化（VISUAL=IDENTITY）
#     - 训练数据：b601_20260910_164106（30Hz，robot_type=seeed_b601_rs_follower）
#   加载语义（rollout/context.py::_load_pretrained_policy，--policy.path 走的就是它）：
#     use_peft=True 时先按 adapter_config.json 的 base_model_name_or_path 起 pi05 基座，
#     再用 PeftModel 把 adapter 挂上去。**所以本脚本不需要 --policy.vlm_model_name**（pi05 无此字段）。
#
# 权重来源与搬运（2026-10-08）：
#   ubuntu-232（8x3090 训练机）：/home/ksa/devdata/Projects/lerobot/self_scripts/output_lerobot_train/
#                                pi05_b601_20260910_164106_20260930/checkpoints/030000
#   本机落点：/home/kf/LX/pai0/pi05_b601_20260910_164106_20260930/checkpoints/030000
#   14 个文件 / 16 MB 经 LAN 同步，双端 sha256 全一致；随后把 checkpoint 内 4 个文件里写死的
#   232 路径改成本机路径（下面 MODEL_PATH 的检查会拦住没改的情况）：
#     sed -i "s#/home/ksa/.cache/modelscope/models/#/home/kf/dev/lerobot/model_weights/#g" \
#       <ckpt>/pretrained_model/{adapter_config.json,config.json,policy_preprocessor.json,train_config.json}
#   本机两个依赖（都在 /home/kf/dev/lerobot/model_weights/，与 232 那份同源）：
#     pi05 基座  lerobot--pi05_base/snapshots/master            （14.4 GB，config.json sha256 与 232 相同）
#     PaliGemma 分词器 google--paligemma-3b-pt-224/snapshots/master
#
# 换 checkpoint 有两条路：
#   ① 同步数换步数 —— 把步数当第一个参数传进去（同 run 的 checkpoints/<步数>/pretrained_model）；
#   ② 换到别的 run  —— 用环境变量 CKPT_ROOT 指到那个 run 的 checkpoints 目录（默认是旧的全量 run）。
#   本机旧 run 现有 030000 / 050000 / 100000 / 150000；另有 pi05_overfit_B/003000（r=64+lr1e-3 的过拟合探针）。
#   注意：新拷过来的 checkpoint（含换了 run 的）都要先跑上面那条 sed，否则本脚本的预检会拦下并打印该跑的命令。
# 需要把推理过程存成数据集时，改用带 --strategy.type=episodic + --dataset.* 的录制型脚本
#   （照 run_inference_pi0.sh 的写法，但 base 策略不能带任何 --dataset.* 参数）。
# 创建日期：2026-10-08

set -e  # 遇到错误立即退出
set -o pipefail  # 管道中任一命令失败则整体失败（配合 tee 使用）

# ==================== 环境检查 ====================
if ! command -v lerobot-rollout >/dev/null 2>&1; then
  echo "未找到 lerobot-rollout，尝试激活 lerobot conda 环境..."
  for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      source "$base/etc/profile.d/conda.sh"
      conda activate lerobot
      break
    fi
  done
fi
if ! command -v lerobot-rollout >/dev/null 2>&1; then
  echo "❌ 错误：lerobot-rollout 不可用"
  echo "   请手动执行: conda activate lerobot"
  exit 1
fi
echo "✅ lerobot-rollout 可用"
echo ""

echo "=========================================="
echo "单臂推理（pi05 LoRA）- B601-RS（make coffee / base 纯推理）"
echo "=========================================="
echo ""

# ==================== 模型配置 ====================
# pi05 LoRA adapter（与 SmolVLA/ACT 同一数据集 b601_20260910_164106 训练）
# 要跑的权重 = CKPT_ROOT（哪个 run 的 checkpoints 目录）+ STEP（第一个参数，默认 030000）：
#   ./run_inference_pi05.sh                              # 默认 run 的 030000
#   ./run_inference_pi05.sh 050000                       # 默认 run 的 050000
#   CKPT_ROOT=/home/kf/LX/pai0/pi05_overfit_B/checkpoints \
#     ./run_inference_pi05.sh 003000                     # 过拟合探针 B 的 003000（r=64 / lr=1e-3）
#   换重训产出时同理：CKPT_ROOT=/home/kf/LX/pai0/<新 run>/checkpoints ./run_inference_pi05.sh <步数>
# 取法：<CKPT_ROOT>/<STEP>/pretrained_model
STEP="${1:-030000}"
CKPT_ROOT="${CKPT_ROOT:-/home/kf/LX/pai0/pi05_b601_20260910_164106_20260930/checkpoints}"
MODEL_PATH="$CKPT_ROOT/$STEP/pretrained_model"

# 加载该 checkpoint 需要的两个本机依赖（只用于预检；真正取路径是从 checkpoint 的 json 里读，
# 见下面第 1b 步，避免脚本与 checkpoint 内容说法不一致）
PI05_BASE_EXPECT="/home/kf/dev/lerobot/model_weights/lerobot--pi05_base/snapshots/master"
TOKENIZER_EXPECT="/home/kf/dev/lerobot/model_weights/google--paligemma-3b-pt-224/snapshots/master"

# 离线解析用的 HF 缓存根（本机无外网）。pi05 的 tokenizer_name 在 policy_preprocessor.json 里
# 写的是绝对路径，正常不吃这个变量；但保留它可让任何残留的 Hub id 在本地命中，避免联网重试。
HF_CACHE="/home/kf/dev/lerobot/model_weights"

# 去噪步数：须与训练 config.json 的 num_inference_steps=10 一致。
# 想更快可降到 6（动作略糙），想更稳可升到 10 以上（更慢）。留空则用 config.json 默认值。
NUM_INFERENCE_STEPS_ARG="--policy.num_inference_steps=10"

# ==================== 推理后端 ====================
# sync = 同步：每个 chunk 用尽时在控制环里直接前向（换 chunk 那一下会卡顿）；
#        只有它能让 action_trace 记到 model 列（SyncInferenceEngine 路径）。
# rtc  = 实时分块：后台线程异步生成下一段动作，控制环不阻塞。pi05 声明 supports_rtc=True，
#        故默认用 rtc —— 10 步去噪 + 3 路相机的 pi05 单次前向较贵，sync 在换 chunk 时容易抖动。
# ⚠️ 本 checkpoint 训练时 rtc_config=null（没做 RTC 训练），此处是推理期 guided RTC
#    （rtc.mode 默认 guided，见 policies/rtc/configuration_rtc.py），不需要 RTC 训练过的权重。
#    首次上机若出现抖动/异常动作，先改成 sync 排除 RTC 因素。
INFERENCE_TYPE="rtc"

# ==================== 动作追踪配置 ====================
# 1 = 用 action_trace.py 包装 lerobot-rollout（运行时打补丁，不改 lerobot 源码），
#     逐 tick 把三组数值追加到 ACTION_TRACE_LOG：
#       model = 策略输出（度，数据集 action 坐标系）——⚠️ 仅 INFERENCE_TYPE=sync 时有效，
#               rtc 后端不经过 SyncInferenceEngine，该列会是 n/a
#       sent  = 传入 robot.send_action 的 dict（插值器 + robot_action_processor 之后）
#       cmd   = send_action 返回值 = 乘 joint_directions、裁 joint_limits 之后的目标角（度）
#     用于核对"模型输出"到"下发电机"之间到底哪一步改了动作（日志自带 sent!=model /
#     |cmd|!=|sent| 标记与结尾统计）。0 = 直接用 lerobot-rollout，不记录。
#     ⚠️ ACT 专用的注意力可视化（act_feature_viz.py）对 pi05 不适用，本脚本不启用。
ACTION_TRACE=1
ACTION_TRACE_WRAPPER="$(dirname "$(readlink -f "$0")")/../../tools/action_trace.py"
ACTION_TRACE_LOG="$HOME/dev/lerobot/self_scripts/_logs/action_pi05_test.log"

# ==================== 硬件配置 ====================
# B601 从臂 = can0（SocketCAN，PEAK PCAN-USB，1Mbps 经典 CAN）
# 推理只用从臂，无需主臂（StarArm102 / /dev/ttyUSB0）
B601_FOLLOWER_PORT="can0"

# 设备 ID（校准文件名，须与采集时一致 = b601_single/02_record.sh 的 FOLLOWER_ID）
FOLLOWER_ID="follower"

# ==================== 摄像头配置 ====================
# 3 路相机（与 self_scripts/b601_single/02_record.sh / 训练数据一致）：
#   hand = JYU2C 2608076（腕部）、front = JYU2C 2607031、top = JYU2C 2607060（顶部）
# 路径用 /dev/v4l/by-id 稳定路径（/dev/videoN 随插拔顺序变化不可靠）。
CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== 相机 key 重命名（robot → policy） ====================
# pi05 基座把三路相机命名为 base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb，
# 训练时即用这套名字（见 checkpoint 的 policy_preprocessor.json 里的
# rename_observations_processor.rename_map），robot 侧 hand/front/top 必须先改名再喂给策略。
# 不传 rename_map 时，rollout 预检（rollout/context.py 的视觉特征校验）会直接报
# "Visual feature mismatch between policy and robot hardware" 并退出。
# ⚠️ 映射关系必须与训练 checkpoint 里的 rename_map 完全一致，否则相机与策略输入错配。
CAMERA_RENAME_MAP='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.hand": "observation.images.left_wrist_0_rgb", "observation.images.top": "observation.images.right_wrist_0_rgb"}'

# ==================== B601 方向修正 ====================
# 必须与采集（self_scripts/b601_single/02_record.sh 的 B601_JOINT_DIRECTIONS）完全一致，全 -1.0。
# 原因：seeed_b601_rs_follower.send_action 对每个关节先乘 joint_directions
#       （缺失关节默认 0.0，会把目标裁到 0 不动），且推理动作与训练动作必须同坐标系。
B601_JOINT_DIRECTIONS='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'

# 相对动作幅度上限（robot.max_relative_target，单位：度/步 @30Hz）未启用（同 ACT/SmolVLA 脚本）：
#   send_action 里该裁剪比较的是"本帧 goal 与当前位置之差"，而 goal 先被 joint_limits
#   裁进关节量程（最大跨度 ~290 度），因此凡 ≥300 的值都拦不住任何动作，只会每帧刷警告。
#   故不传该参数（默认 None = 不做相对裁剪）。若确实需要限速，参照 b601_so101_bimanual/01_teleop_test.sh。

# ==================== 任务与运行时长 ====================
# base 策略没有 dataset，任务描述必须走顶层 --task 传入
# （见 rollout/context.py：task_str = cfg.dataset.single_task if cfg.dataset else cfg.task）
# pi05 是语言条件策略，该描述会作为 PaliGemma tokenizer 输入，须与采集/训练时完全一致
TASK_DESCRIPTION="Pick up the paper cup, place it on the silver tray of the coffee machine, pick up the cube, press the button with the cube (red light on), wait about 4 seconds, release the button (red light off), put the cube on the table first, then move the cup from the coffee machine to the table"
DURATION=0  # 总运行时长上限（秒）：0 = 不限，一直推理到 Ctrl+C；>0 则到点自动退出
FPS=30  # 推理频率，必须与训练数据集 fps（30Hz）一致，否则 flow matching 时序不匹配

# ==================== 推理前检查 ====================
# 1) 模型权重（先检查，路径不对直接退出，避免误触硬件）
#    注意：LoRA checkpoint 没有 model.safetensors，别照 SmolVLA 的清单查。
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 模型路径不存在：$MODEL_PATH"
  exit 1
fi
if [ -f "$MODEL_PATH/model.safetensors" ]; then
  echo "⚠️  检测到 model.safetensors —— 这看起来是全量权重而非 LoRA adapter。"
  echo "    若确实要跑全量权重（旧的 020000_b601_pi0 那种），本脚本的检查项需放宽；确认后继续。"
else
  for f in adapter_config.json adapter_model.safetensors config.json \
           policy_preprocessor.json policy_postprocessor.json; do
    [ -f "$MODEL_PATH/$f" ] || { echo "❌ 缺少 $MODEL_PATH/$f"; exit 1; }
  done
  echo "✅ LoRA checkpoint 文件齐全：$MODEL_PATH"
fi

# 1b) 解析 checkpoint 内部引用，确认本机路径存在
#     （两道坎：①从 232 拷来没改路径 ②本机缺基座/分词器 —— 都提前报错，不要跑到加载一半才炸）
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
        bad.append("%s 仍是 232 的路径：%s" % (name, path))

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
  echo "   若提示仍是 232 的 /home/ksa/.cache/modelscope/... 路径，先跑（16 MB，秒级）："
  echo "     sed -i \"s#/home/ksa/.cache/modelscope/models/#/home/kf/dev/lerobot/model_weights/#g\" \\"
  echo "       $MODEL_PATH/{adapter_config.json,config.json,policy_preprocessor.json,train_config.json}"
  exit 1
fi
echo "✅ checkpoint 内部引用的基座/分词器在本机可达"

# 2) 动作追踪包装器（开追踪时必须有，避免跑到一半才发现）
if [ "$ACTION_TRACE" = "1" ] && [ ! -f "$ACTION_TRACE_WRAPPER" ]; then
  echo "❌ 缺少动作追踪包装器：$ACTION_TRACE_WRAPPER（或把 ACTION_TRACE 设为 0）"
  exit 1
fi

# 3) B601 CAN
echo ""
if ! ip link show $B601_FOLLOWER_PORT >/dev/null 2>&1; then
  echo "❌ $B601_FOLLOWER_PORT 不存在"; exit 1
fi
if ! ip link show $B601_FOLLOWER_PORT 2>/dev/null | grep -q "state UP"; then
  echo "⚠️  配置 $B601_FOLLOWER_PORT（1Mbps 经典 CAN）..."
  sudo ip link set $B601_FOLLOWER_PORT down 2>/dev/null || true
  sudo ip link set $B601_FOLLOWER_PORT type can bitrate 1000000
  sudo ip link set $B601_FOLLOWER_PORT up
fi
echo "✅ B601 CAN 已就绪"

# 4) 摄像头（3 路）
echo ""
for cam in \
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0 \
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0 \
  /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0; do
  if [ ! -e "$cam" ]; then
    echo "❌ 警告：摄像头不存在 $cam"
  else
    echo "✅ $cam 已连接"
  fi
done

# 5) 从臂校准文件（推理需与采集同 id）
echo ""
CALIB_FILE="$HOME/.cache/huggingface/lerobot/calibration/robots/seeed_b601_rs_follower/${FOLLOWER_ID}.json"
if [ -f "$CALIB_FILE" ]; then
  echo "✅ $CALIB_FILE"
else
  echo "❌ 缺失校准文件：$CALIB_FILE"
  echo "   请先运行 self_scripts/b601_common/02_follower_calibration.sh 完成从臂校准"
  exit 1
fi

# ==================== 推理参数总览 ====================
echo ""
echo "=========================================="
echo "推理参数总览（pi05 / LoRA）"
echo "=========================================="
echo "训练步数：$STEP（run: $(basename "$(dirname "$CKPT_ROOT")")）"
echo "模型路径：$MODEL_PATH"
echo "权重类型：LoRA adapter（r=16 / alpha=8），基座按 adapter_config.json 自动加载"
echo "机器人类型：seeed_b601_rs_follower（单臂 B601-RS，can0）"
echo "策略：base（纯推理，不录制、不保存数据）"
echo "推理方式：${INFERENCE_TYPE}（pi05 支持 RTC；sync 会阻塞控制环、但可记录 model 动作列）"
echo "去噪步数：${NUM_INFERENCE_STEPS_ARG:-（用 config.json 默认值）}"
echo "任务描述：$TASK_DESCRIPTION"
if [ "$DURATION" = "0" ]; then
  echo "运行时长：不限（一直推理到 Ctrl+C）"
else
  echo "运行时长：${DURATION}秒 (约 $((DURATION / 60)) 分钟)"
fi
echo "推理频率：${FPS} Hz"
echo "摄像头：hand / front / top（3 路）"
echo "相机重命名：front→base_0_rgb  hand→left_wrist_0_rgb  top→right_wrist_0_rgb（与训练一致）"
echo "B601 方向修正：全 -1.0（与采集一致）"
echo "相对动作上限：未启用（不传 max_relative_target，默认不做相对裁剪）"
if [ "$ACTION_TRACE" = "1" ]; then
  echo "动作追踪：✅ sent/cmd 逐 tick 记录到 $ACTION_TRACE_LOG"
  if [ "$INFERENCE_TYPE" = "rtc" ]; then
    echo "           （rtc 后端下 model 列为 n/a，只有 sent/cmd；需要 model 列请把 INFERENCE_TYPE 改成 sync）"
  fi
else
  echo "动作追踪：❌ 关闭（直接调用 lerobot-rollout）"
fi
echo "录制/保存数据：❌ 否（base 策略不录制）"
echo "=========================================="
echo ""

read -p "确认 B601 从臂（can0）处于零位（夹爪闭合）、周边安全，按 ENTER 开始推理，Ctrl+C 取消..." dummy

# ==================== 开始推理 ====================
echo ""
echo "🚀 开始单臂模型推理（pi05 $STEP）..."
echo ""

# Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

# 离线模式：基座与分词器全部走本地，禁止再请求 huggingface.co
#   pi05 的 tokenizer_name 在 policy_preprocessor.json 里已是绝对路径，正常不依赖缓存；
#   HF_HUB_CACHE 仍指向本机 model_weights，兜住任何残留 Hub id 解析。
export HF_HUB_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inference_b601_make_coffee_pi05_${STEP}_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 构建推理命令（lerobot-rollout base 策略：纯推理，无录制、无落盘）
# 说明：
#   - base 策略不创建数据集，因此不能传任何 --dataset.* 参数（传了会直接报错），
#     任务描述改用顶层 --task 传入
#   - pi05：--policy.num_inference_steps 控制 flow matching 去噪步数（训练用 10）
#   - 不传 --policy.vlm_model_name：pi05 没有这个字段，基座由 adapter_config.json 决定
#   - B601：can0 + socketcan + gravity_compensation + joint_directions(全 -1.0)，与采集一致
#   - 机器人 3 路相机输出 key 为 hand/front/top，须用 --rename_map 改成策略侧的
#     base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb（与训练一致，见 CAMERA_RENAME_MAP）
#   - --fps 须 30（训练数据 30Hz）；--duration=0 表示不限时长（跑到 Ctrl+C）
#   - ACTION_TRACE=1 时用 action_trace.py 包装（参数完全一致，只多一路动作记录）
#   - 需要覆盖 RTC 参数时直接加 CLI，例如 --inference.rtc.execution_horizon=10、
#     --inference.queue_threshold=30（默认值见 policies/rtc/configuration_rtc.py）
if [ "$ACTION_TRACE" = "1" ]; then
  ROLLOUT_CMD=(python "$ACTION_TRACE_WRAPPER")
  export ACTION_TRACE_LOG
else
  ROLLOUT_CMD=(lerobot-rollout)
fi
"${ROLLOUT_CMD[@]}" \
  --strategy.type=base \
  --inference.type=$INFERENCE_TYPE \
  --policy.path=$MODEL_PATH \
  $NUM_INFERENCE_STEPS_ARG \
  --robot.type=seeed_b601_rs_follower \
  --robot.id=$FOLLOWER_ID \
  --robot.port=$B601_FOLLOWER_PORT \
  --robot.can_adapter=socketcan \
  --robot.gravity_compensation=true \
  --robot.joint_directions="$B601_JOINT_DIRECTIONS" \
  --robot.cameras="$CAMERAS" \
  --rename_map="$CAMERA_RENAME_MAP" \
  --task="$TASK_DESCRIPTION" \
  --fps=$FPS \
  --duration=$DURATION \
  --display_data=true \
  --display_compressed_images=false \
  --play_sounds=false 2>&1 | tee "$LOG_FILE"

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
