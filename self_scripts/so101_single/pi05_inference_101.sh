#!/bin/bash
# 单臂 SO-101 pi0.5 直接推理（hqfang/pi05-so100_101，step 150k）
# 默认策略：base —— 纯推理，不建数据集、不录帧、不编码、不 finalize（推理数据不保存）
# 保存模式（SAVE_DATA=true 或第一个参数传 save）：strategy=episodic —— 推理边跑边录成数据集，
#   右键 = 结束当前 episode 并保存 | 左键 = 丢弃当前 episode（不保存）| ESC = 结束整个会话；
#   每次运行新建一个文件夹（$DATA_ROOT/<repo_id>，标 rollout_ 前缀），存下的就是训练数据格式。
#
# 参照：
#   - self_scripts/so101_single/act_inference_101.sh：单臂 SO-101 ACT 推理（硬件端口/相机/校准/FPS 基准）
#   - self_scripts/so101_single/record_101.sh：单臂 SO-101 采集（端口 / 相机 / fps / 任务描述 / 校准 id）
#   - pi0.5/pi05-so100_101/README.md：该权重的官方加载说明（相机槽位 / 归一化 / tokenizer）
#
# 模型：pi0.5（lerobot 策略，type=pi05），本地权重 /home/kf/dev/lerobot/model_weights/pi0.5/pi05-so100_101
#   - 训练数据：allenai/MolmoAct2-SO100_101-Dataset 那批 SO-100/101 混合数据（1209 个数据集、
#     36877 集、1922 万帧）——和 MolmoAct2-SO100_101 是同一批异构数据，各家校准零点不同，
#     README 明确写"actions use the original dataset units"，所以本机臂的零点口径不一定对得上
#   - 输入：observation.state(6，顺序 shoulder_pan/lift/elbow/wrist_flex/wrist_roll/gripper)
#     + observation.images.camera_0..camera_3（1~4 路都行，letterbox 到 224x224）
#   - 输出：action(6)，绝对关节角，chunk=50；归一化用权重自带的 q01/q99 处理器（不要自己再归一化）
#
# 相机槽位：camera_N 只是"输入槽位"，训练时每个 episode 随机分配视角，不是固定物理相机
#   → 本机 2 路映射到 camera_0/camera_1，剩下 2 个槽位由模型自动填 masked 空图（README 支持 1~4 路）
#
# ⚠️ 前置条件：该权重依赖官方补丁（pi0.5/pi05-so100_101/code/lerobot.patch），
#    它给 lerobot 加了两样东西：
#      1) pi05_sentencepiece_tokenizer 处理器（权重自带 policy_preprocessor.json 里就要用它，
#         没有它连处理器都装不起来）
#      2) pi05 config 的 checkpoint_joint_stack / training_attention_backend 字段 + 相机 padding 修正
#    没打补丁时本脚本会在检查阶段直接报错并给出命令，不会带着半吊子配置去动硬件。
#
# 用法（第一个参数选后端，其余参数原样传给后端）：
#   bash self_scripts/so101_single/pi05_inference_101.sh                 # 默认：lerobot-rollout 官方配方（动作原样下发）
#   bash self_scripts/so101_single/pi05_inference_101.sh direct          # 直接控制环 + shoulder_lift 映射，闭环跑 60s
#   bash self_scripts/so101_single/pi05_inference_101.sh direct-check    # 直接控制环 check：只打印原始/映射后动作，不动臂
#   DURATION=0 bash self_scripts/so101_single/pi05_inference_101.sh direct          # 不限时长
#   LIFT_SCALE=1.0 LIFT_OFFSET=134.2 LIFT_WRAP=0 bash ... direct   # 换成"单点对齐"那组映射
#   （默认 = 镜像映射 LIFT_SCALE=-1 LIFT_OFFSET=90 LIFT_WRAP=1，见下方说明）
#   bash self_scripts/so101_single/pi05_inference_101.sh save       # 推理 + 录数据（右键保存本段 / 左键丢弃）
#   NUM_EPISODES=20 bash self_scripts/so101_single/pi05_inference_101.sh save   # 录 20 段后自动结束
#
# 保存模式说明：
#   - 只有 rollout 后端能落盘（direct 是自写控制环，无数据集支持；传了会直接报错）
#   - 每个 episode 到 EPISODE_TIME 秒也会自动保存一次（防止忘记按键丢数据），
#     想在读秒前结束就按右键；这一整段不想要就按左键（丢弃并重录）
#
# 两种后端的区别（为什么要有 direct）：
#   rollout：走 lerobot-rollout 的官方加载/预处理/下发管线，最贴近权重说明；但**没法插自定义后处理**，
#            shoulder_lift 只能原样下发（实测模型给 -231.7，本机量程约 -76~+121，会被限幅顶到边界）。
#   direct ：自己写控制环（用同一套权重 pre/post 处理器 + SOFollower），下发前对 shoulder_lift 做
#            本机 = a*模型 + b 的映射补偿，并逐 tick 打日志便于判断行为。
#
# 可按需覆盖的环境变量：
#   MODEL_PATH / FOLLOWER_PORT / FOLLOWER_ID / CAMERAS / TASK_DESCRIPTION / FPS / DURATION
#   POLICY_DTYPE（默认 bfloat16；rollout 后端用）
#   INFERENCE_TYPE（默认 sync；rollout 后端用，可试 rtc）
#   MAX_REL_TARGET  每步每关节最大相对位移（度），默认 10
#   LIFT_SCALE / LIFT_OFFSET  direct 后端的 shoulder_lift 映射（默认 1.0 / 134.2）
#   SAVE_DATA（默认 false）保存模式开关，等价于第一个参数传 save
#   DATA_ROOT / DATASET_NAME 保存模式的存放根目录（默认 $HOME/LX/pai0/rollout_data）
#                            与数据集名（默认 rollout_so101_pi05_<时间戳>；必须 rollout_ 前缀）
#   NUM_EPISODES / EPISODE_TIME / RESET_TIME / PUSH_TO_HUB  保存模式的录制参数

set -e  # 遇到错误立即退出
set -o pipefail  # 管道中任一命令失败则整体失败（配合 tee 使用）

# ==================== 环境检查 ====================
if ! command -v lerobot-rollout >/dev/null 2>&1; then
  echo "未找到 lerobot-rollout，尝试激活 lerobot conda 环境..."
  for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
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

# ==================== 后端/模式 ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_PY="$SCRIPT_DIR/pi05_direct_inference_101.py"
BACKEND="rollout"
DIRECT_MODE="run"
SAVE_DATA="${SAVE_DATA:-false}"   # true = 推理过程中把数据录成数据集（仅 rollout 后端支持）
case "${1:-}" in
  direct)       BACKEND="direct";  DIRECT_MODE="run";   shift ;;
  direct-check) BACKEND="direct";  DIRECT_MODE="check"; shift ;;
  save)         BACKEND="rollout"; SAVE_DATA=true;      shift ;;
  rollout|"")   BACKEND="rollout"; shift 2>/dev/null || true ;;
  *) echo "❌ 未知参数：$1（可用：direct / direct-check / save；不带参数 = lerobot-rollout）"; exit 1 ;;
esac

echo "=========================================="
echo "单臂推理（pi0.5 / SO-101 混合权重）- SO-101（$( [ "$SAVE_DATA" = true ] && echo "episodic 推理+录数据" || echo "base 纯推理" )）"
echo "=========================================="
echo ""

# ==================== 模型配置 ====================
MODEL_PATH="${MODEL_PATH:-/home/kf/dev/lerobot/model_weights/pi0.5/pi05-so100_101}"
# 权重 config.json 里 text_tokenizer_name 指向训练机路径（/lustre/...），本机必须覆盖成权重自带的
TOKENIZER_PATH="$MODEL_PATH/tokenizer/tokenizer.model"
PATCH_PATH="$MODEL_PATH/code/lerobot.patch"
# 权重是 fp32 存的（16.6GB）；24GB 卡用 bfloat16 更稳（视觉通路内部仍保持 fp32，见 modeling_pi05.py）
POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"

# pi0.5 单次前向用 10 步去噪（与 config.json 的 num_inference_steps 一致），显式写出来便于调
NUM_STEPS_ARG="--policy.num_inference_steps=10"
# compile_model 在本机保持关闭：max-autotune 编译耗时长、且首次推理容易卡在编译上
POLICY_EXTRA_ARGS="--policy.text_tokenizer_name=$TOKENIZER_PATH --policy.compile_model=false --policy.dtype=$POLICY_DTYPE"

# ==================== 相机槽位映射 ====================
# 权重只有 camera_0..camera_3 这四个槽位，把本机两路按固定顺序固定映射（rollout 会把它盖进
# policy 预处理器的 rename_observations_processor，见 rollout/context.py）。
# ⚠️ 一次会话内不要换映射顺序；camera_N 是槽位不是物理相机，但顺序换了输入就换了。
RENAME_MAP="{\"observation.images.hand\": \"observation.images.camera_0\", \"observation.images.front\": \"observation.images.camera_1\"}"

# ==================== 硬件配置 ====================
# 从臂 = SO-101（与 act_inference_101.sh / record_101.sh 同一套端口与校准 id）
# 推理只用从臂，无需主臂（so101_leader / 5B61034865）
SO101_FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00}"
FOLLOWER_ID="${FOLLOWER_ID:-jt_follower_arm_right}"

# 每步每关节最大相对位移（度）。留空/0 = 不限幅（跟采集一致，但首次跑混合权重建议限幅）
MAX_REL_TARGET="${MAX_REL_TARGET:-10}"

# direct 后端的 shoulder_lift 口径映射（本机 = a*模型 + b，结果折回 [-180,180)）。
# 默认是"镜像映射"（1220 份社区数据统计推出：社区数据里肩"收起"读 ~188、本机读 -97.9，
# 同一条物理轴方向相反，C ≈ 本机home + 社区max ≈ 90）：本机 = 90 - 模型。
# 另两组备选（改环境变量即可）：
#   单点对齐/姿态保持： LIFT_SCALE=1.0  LIFT_OFFSET=134.2 LIFT_WRAP=0
#   分布两端对齐(q01/q99)： LIFT_SCALE=1.0274 LIFT_OFFSET=-123.08 LIFT_WRAP=0（会把输出换算到超量程，慎用）
LIFT_SCALE="${LIFT_SCALE:--1.0}"
LIFT_OFFSET="${LIFT_OFFSET:-90}"
LIFT_WRAP="${LIFT_WRAP:-1}"   # 1=折回 [-180,180)（镜像映射必须开）

# ==================== 摄像头配置 ====================
# 2 路相机，与 record_101.sh 采集一致：
#   hand  = JYU2C 2609007（腕部）→ camera_0
#   front = JYU2C 2608006（前视）→ camera_1
DEFAULT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2609007-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608006-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'
# 注意：默认值必须单独放一个变量再引用，不要写成 ${CAMERAS-{...}}（bash 会在第一个 } 处截断）
CAMERAS="${CAMERAS-$DEFAULT_CAMERAS}"

# ==================== 任务与运行时长 ====================
# base 策略没有 dataset，任务描述必须走顶层 --task 传入
# 任务描述与 self_scripts/so101_single/record_101.sh / 数据集 meta/tasks.parquet 完全一致
TASK_DESCRIPTION="${TASK_DESCRIPTION:-Pick up the yellow banana placed at different angles and put it on the white fruit plate}"
# 总运行时长上限（秒）：0 = 不限，一直推理到 Ctrl+C；>0 则到点自动退出。
# direct 后端首次上硬件默认只跑 60s；rollout 后端不限制（跟 act_inference_101.sh 一致）。
if [ "$BACKEND" = "direct" ] && [ "$DIRECT_MODE" = "run" ]; then
  DURATION="${DURATION:-60}"
else
  DURATION="${DURATION:-0}"
fi
FPS="${FPS:-30}"  # 推理频率，须与训练数据 30Hz 一致
INFERENCE_TYPE="${INFERENCE_TYPE:-sync}"  # pi0.5 支持 rtc；先用 sync（ACT 那套同款）

# ==================== 保存模式（SAVE_DATA）数据集配置 ====================
# 策略：episodic（lerobot-rollout 的录制型策略，键位与 record_101.sh 一致）
#   - rollout 的数据集名必须以 rollout_ 开头（见 rollout/context.py 的校验）
#   - 每次运行新建一个带时间戳的文件夹；--dataset.no_stamp=true 保证目录名与 repo_id 一致
#   - push_to_hub=false：本机外网不可达，数据只存本地
if [ "$SAVE_DATA" = true ] && [ "$BACKEND" != "rollout" ]; then
  echo "❌ 保存模式只支持 rollout 后端（direct 是自写控制环，没有数据集支持）"
  echo "   用法：bash $0 save   或   SAVE_DATA=true bash $0"
  exit 1
fi
DATA_ROOT="${DATA_ROOT:-$HOME/LX/pai0/rollout_data}"
if [ "$SAVE_DATA" = true ]; then
  DATASET_NAME="${DATASET_NAME:-rollout_so101_pi05_$(date +%Y%m%d_%H%M%S)}"
  DATA_DIR="$DATA_ROOT/$DATASET_NAME"
  NUM_EPISODES="${NUM_EPISODES:-10}"        # 录满这么多段就自动结束（ESC 可提前收工）
  EPISODE_TIME="${EPISODE_TIME:-60}"        # 单段最长时间（秒）：到点也会自动保存
  RESET_TIME="${RESET_TIME:-10}"            # 段间重置时长（秒）：可选，按右键可提前跳过
  PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
  mkdir -p "$DATA_ROOT"
fi

# ==================== 推理前检查 ====================
# 1) 模型权重（先检查，路径不对直接退出，避免误触硬件）
echo "1. 检查模型文件..."
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 模型路径不存在：$MODEL_PATH"
  exit 1
fi
for f in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json \
         policy_preprocessor_step_3_normalizer_processor.safetensors \
         policy_postprocessor_step_0_unnormalizer_processor.safetensors \
         tokenizer/tokenizer.model; do
  [ -f "$MODEL_PATH/$f" ] || { echo "❌ 缺少 $MODEL_PATH/$f"; exit 1; }
done
echo "✅ 模型文件存在：$MODEL_PATH（含 q01/q99 归一化处理器与自带 tokenizer）"

# 2) lerobot 补丁自检：权重自带的 policy_preprocessor.json 用 pi05_sentencepiece_tokenizer，
#    本地 lerobot 没打补丁时该处理器不存在 -> 加载必失败，这里提前拦住。
echo ""
echo "2. 检查 lerobot 是否已打该权重的补丁..."
if python - <<'PY' >/dev/null 2>&1
import importlib

import lerobot.policies.pi05.processor_pi05  # noqa: F401  触发处理器注册
from lerobot.processor import ProcessorStepRegistry

ProcessorStepRegistry.get("pi05_sentencepiece_tokenizer")
PY
then
  echo "✅ pi05_sentencepiece_tokenizer 已注册（补丁已生效）"
else
  echo "❌ 本地 lerobot 缺少 pi05_sentencepiece_tokenizer 处理器，无法加载该权重"
  echo "   该权重附带官方补丁，需要先应用到 lerobot 仓库（uv.lock 与本地版本不同，可跳过）："
  echo "     cd $HOME/LX/pai0/lerobot"
  echo "     git apply --exclude=uv.lock $PATCH_PATH"
  echo "   （补丁只给 pi05 加 sentencepiece 分词器/可选字段，并修正相机 padding，默认行为与打补丁前一致）"
  echo "   同时确认 sentencepiece 已安装：python -c 'import sentencepiece'"
  exit 1
fi

# 3) 从臂串口 + 舵机应答
echo ""
echo "3. 检查从臂..."
if [ ! -e "$SO101_FOLLOWER_PORT" ]; then
  echo "❌ 串口不存在 $SO101_FOLLOWER_PORT"
  echo "   当前串口设备：$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | tr '\n' ' ')"
  exit 1
fi
if [ ! -r "$SO101_FOLLOWER_PORT" ] || [ ! -w "$SO101_FOLLOWER_PORT" ]; then
  echo "⚠️  $SO101_FOLLOWER_PORT 无读写权限，尝试授权..."
  sudo chmod 666 "$SO101_FOLLOWER_PORT"
fi
if python -c "import scservo_sdk" >/dev/null 2>&1; then
  if hits="$(python - "$SO101_FOLLOWER_PORT" <<'PY'
import sys
import scservo_sdk as s

port = sys.argv[1]
ph = s.PortHandler(port)
if not ph.openPort():
    print("open failed", file=sys.stderr)
    sys.exit(1)
ph.setBaudRate(1_000_000)
pk = s.PacketHandler(0)
hits = [i for i in range(1, 7) if pk.ping(ph, i)[1] == 0]
ph.closePort()
print(",".join(str(i) for i in hits))
sys.exit(0 if len(hits) == 6 else 1)
PY
)"; then
    echo "✅ 从臂 $SO101_FOLLOWER_PORT 舵机应答: $hits"
  else
    echo "❌ 从臂舵机应答不全（应答: ${hits:-无}）"
    echo "   1) 确认该臂 12V 电源已开、USB 已插好"
    echo "   2) 确认这条串口对应的确实是 SO-101"
    exit 1
  fi
else
  echo "⚠️  跳过舵机应答检查（python 缺 scservo_sdk）"
fi

# 4) 摄像头（key ↔ 设备映射，key 写错/接错在开始前就能看出来）
echo ""
echo "4. 检查摄像头..."
CAM_ENTRIES=$(echo "$CAMERAS" | sed -nE 's/^[[:space:]]*([A-Za-z0-9_]+): \{type: [a-z]+, index_or_path: ([^,]+),.*/\1=\2/p')
if [ -z "$CAM_ENTRIES" ]; then
  echo "  ⚠️  解析不出相机条目（CAMERAS 可能写成了一行），跳过相机检查"
fi
CAM_OK=true
for entry in $CAM_ENTRIES; do
  key="${entry%%=*}"
  cam="${entry#*=}"
  if [ -e "$cam" ]; then
    echo "  ✅ $key = $cam"
  else
    echo "  ❌ $key = $cam（设备不存在，可用 ls /dev/v4l/by-id/ 确认）"
    CAM_OK=false
  fi
done
if [ "$CAM_OK" = false ]; then
  exit 1
fi

# 5) 从臂校准文件（推理需与采集同 id；SO-101 的 robot.name = so_follower）
echo ""
CALIB_FILE="$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/${FOLLOWER_ID}.json"
if [ -f "$CALIB_FILE" ]; then
  echo "✅ 校准文件：$CALIB_FILE"
else
  echo "❌ 缺失校准文件：$CALIB_FILE"
  echo "   请先运行 self_scripts/so101_single/calibrate_101.sh 完成从臂校准"
  exit 1
fi

# ==================== 推理参数总览 ====================
echo ""
echo "=========================================="
echo "推理参数总览"
echo "=========================================="
echo "模型路径：$MODEL_PATH"
echo "机器人类型：so101_follower（单臂 SO-101，$SO101_FOLLOWER_PORT）"
if [ "$BACKEND" = "direct" ]; then
  echo "后端：direct 直接控制环（模式：$DIRECT_MODE；脚本：$DIRECT_PY）"
  echo "shoulder_lift 映射：本机 = ${LIFT_SCALE}*模型 + (${LIFT_OFFSET})$( [ "$LIFT_WRAP" = "1" ] && echo "，折回 [-180,180)" )"
  echo "策略：纯推理，不录制、不保存数据"
  echo "推理方式：每 tick 前向（pi0.5 内部 chunk=50 排队）"
  echo "模型精度：$POLICY_DTYPE"
else
  echo "后端：lerobot-rollout 官方配方（动作原样下发，无 shoulder_lift 映射）"
  if [ "$SAVE_DATA" = true ]; then
    echo "策略：episodic（推理 + 录数据）"
  else
    echo "策略：base（纯推理，不录制、不保存数据）"
  fi
  echo "推理方式：$INFERENCE_TYPE"
  echo "模型精度：$POLICY_DTYPE，去噪步数 10"
fi
echo "任务描述：$TASK_DESCRIPTION"
if [ "$DURATION" = "0" ]; then
  echo "运行时长：不限（一直推理到 Ctrl+C）"
else
  echo "运行时长：${DURATION}秒 (约 $((DURATION / 60)) 分钟)"
fi
echo "推理频率：${FPS} Hz（训练数据 30Hz，chunk=50 步）"
echo "相机：$(echo "$CAM_ENTRIES" | sed 's/=[^ ]*//g' | tr '\n' ' ')→ camera_0 / camera_1（camera_2/3 自动填空图）"
if [ -n "$MAX_REL_TARGET" ] && [ "$MAX_REL_TARGET" != "0" ]; then
  echo "相对动作上限：${MAX_REL_TARGET} 度/步"
else
  echo "相对动作上限：未启用（不传 max_relative_target）"
fi
if [ "$SAVE_DATA" = true ]; then
  echo "录制/保存数据：✅ 是（episodic 策略）"
  echo "数据集名称：$DATASET_NAME"
  echo "本地保存路径：$DATA_DIR（本次新建）"
  echo "目标段数：$NUM_EPISODES 段；单段最长 ${EPISODE_TIME}秒（到点自动保存）；段间重置 ${RESET_TIME}秒"
  echo "上传状态：$([ "$PUSH_TO_HUB" = true ] && echo "推送到 Hub" || echo "不上传（仅本地，PUSH_TO_HUB=false）")"
else
  echo "录制/保存数据：❌ 否（base 策略不录制）"
fi
echo "=========================================="
echo ""

if [ "$SAVE_DATA" = true ]; then
  echo "📝 按键控制（焦点保持在终端窗口）："
  echo "   右键 / n = 保存刚推理的这一段（结束本段并落盘）"
  echo "   左键 / r = 不要这一段（丢弃并重录）"
  echo "   ESC      = 结束整个会话"
  echo ""
fi

read -p "确认从臂（$SO101_FOLLOWER_PORT）处于零位（夹爪闭合）、周边安全，按 ENTER 开始推理，Ctrl+C 取消..." dummy

# ==================== 开始推理 ====================
echo ""
echo "🚀 开始单臂 SO-101 模型推理（pi0.5）..."
echo ""

# Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inference_so101_pi05_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 推理命令说明（lerobot-rollout）：
#   - base 策略（默认）：不创建数据集，不能传任何 --dataset.* 参数；任务描述走顶层 --task
#   - episodic 策略（SAVE_DATA=true）：必须带 --dataset.*，逐段录制落盘（右键保存 / 左键丢弃）
#   - --rename_map 把本机 hand/front 映射到权重的 camera_0/camera_1（rollout 会覆盖 policy 的
#     重命名处理器；设了它之后 rollout 会跳过"相机名必须与策略一致"的校验）
#     （rename_map 只作用于喂给策略的输入；数据集里存的仍是 hand/front，与 record_101.sh 一致）
#   - camera_2/camera_3 缺数据时，pi0.5 内部自动补 masked 空图（见 modeling_pi05.py 的
#     missing_img_keys 分支），所以 2 路相机可以直接跑，不需要 empty_cameras
#   - --policy.text_tokenizer_name 必须指向权重自带的 tokenizer（config.json 里是训练机路径）
#   - 动作是绝对关节角（chunk=50），已由权重自带后处理器还原成原始单位

# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂、lerobot-rollout 被 SIGPIPE(141)
# 杀死而跳过断开清理（否则从臂保持使能、不释放）。
trap '' PIPE

set +e  # Ctrl+C 退出码 130，异常退出非 0，都要能捕获后继续打印结果
if [ "$BACKEND" = "direct" ]; then
  # direct：自己写控制环（复用权重的 pre/post 处理器 + SOFollower），下发前做 shoulder_lift 映射。
  # --duration=0（check 模式）由 py 自己忽略；run 模式用上面算好的 DURATION。
  python "$DIRECT_PY" \
    --mode "$DIRECT_MODE" \
    --model-path "$MODEL_PATH" \
    --port "$SO101_FOLLOWER_PORT" \
    --robot-id "$FOLLOWER_ID" \
    --task "$TASK_DESCRIPTION" \
    --fps "$FPS" \
    --duration "$DURATION" \
    --max-relative-target "$MAX_REL_TARGET" \
    --dtype "$POLICY_DTYPE" \
    --lift-scale "$LIFT_SCALE" \
    --lift-offset "$LIFT_OFFSET" \
    $( [ "$LIFT_WRAP" = "1" ] && echo "--lift-wrap" || echo "--no-lift-wrap" ) \
    "$@" 2>&1 | tee "$LOG_FILE"
else
  # 用数组拼参数：数据集里带 JSON / 空格 / 花括号（CAMERAS / RENAME_MAP / 任务描述），
  # 数组能整块原样传给 lerobot-rollout，不用 eval 也不会被二次分词。
  ROLLOUT_ARGS=(
    --inference.type="$INFERENCE_TYPE"
    --policy.path="$MODEL_PATH"
    # 注意：这两个变量是"多个 flag 拼成的字符串"，必须不加引号让它按空格拆成多个参数
    # shellcheck disable=SC2206
    $NUM_STEPS_ARG
    # shellcheck disable=SC2206
    $POLICY_EXTRA_ARGS
    --rename_map="$RENAME_MAP"
    --robot.type=so101_follower
    --robot.id="$FOLLOWER_ID"
    --robot.port="$SO101_FOLLOWER_PORT"
    --robot.cameras="$CAMERAS"
    --task="$TASK_DESCRIPTION"
    --fps="$FPS"
    --duration="$DURATION"
    --display_data=true
    --display_compressed_images=false
    --play_sounds=false
  )
  if [ "$SAVE_DATA" = true ]; then
    ROLLOUT_ARGS+=(
      --strategy.type=episodic
      --dataset.repo_id="$DATASET_NAME"
      --dataset.root="$DATA_DIR"
      --dataset.no_stamp=true
      --dataset.single_task="$TASK_DESCRIPTION"
      --dataset.fps="$FPS"
      --dataset.num_episodes="$NUM_EPISODES"
      --dataset.episode_time_s="$EPISODE_TIME"
      --dataset.reset_time_s="$RESET_TIME"
      --dataset.video=true
      --dataset.rgb_encoder.vcodec=h264
      --dataset.push_to_hub="$PUSH_TO_HUB"
    )
  else
    ROLLOUT_ARGS+=(--strategy.type=base)
  fi
  if [ -n "$MAX_REL_TARGET" ] && [ "$MAX_REL_TARGET" != "0" ]; then
    ROLLOUT_ARGS+=(--robot.max_relative_target="$MAX_REL_TARGET")
  fi
  lerobot-rollout "${ROLLOUT_ARGS[@]}" "$@" 2>&1 | tee "$LOG_FILE"
fi
STATUS=${PIPESTATUS[0]}
set -e

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
if [ "$STATUS" -eq 0 ] || [ "$STATUS" -eq 130 ]; then
  echo "✅ 单臂推理结束（pi0.5，后端：$BACKEND，从臂已回初始位并失能）"
else
  echo "⚠️  推理异常退出（退出码 $STATUS），从臂可能来不及失能（电机仍变硬）"
  if grep -q "There is no status packet" "$LOG_FILE"; then
    echo ""
    echo "诊断：与舵机总线失联（串口还在，但 6 个舵机同时不应答）—— 常见原因："
    echo "  1) 从臂舵机串链/12V 电源接头松动：断点之后的整条链全静默，重新插紧"
    echo "  2) 从臂与相机共用 USB Hub -> 直接把臂插到主机 USB 口"
  fi
fi
if [ "$SAVE_DATA" = true ]; then
  SAVED_EPISODES=$(python3 -c "import json; print(json.load(open('$DATA_DIR/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
  echo "本次已保存推理数据：$SAVED_EPISODES 段"
  echo "数据集名称：$DATASET_NAME"
  echo "本地保存路径：$DATA_DIR"
else
  echo "本次未保存任何推理数据（base 策略不录制、不落盘）。"
fi
echo "完整日志：$LOG_FILE"
echo "=========================================="
