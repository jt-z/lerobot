#!/bin/bash
# 单 B601-RS SmolVLA 推理：从臂 B601-RS（SocketCAN can0）
# 策略：base —— 纯推理，不建数据集、不录帧、不编码、不 finalize（推理数据不保存）
# 任务：单臂端起纸杯放上咖啡机托盘、拿方块按按钮等 4 秒、方块放回桌面、再把杯子端回桌面
#       （与 self_scripts/b601_single/02_record.sh 采集的任务描述/相机/fps/方向修正完全一致）
#
# 参照：
#   - run_inference_ACT.sh：单臂 B601 ACT 纯推理脚本（检查项/日志/参数风格，本脚本的模板）
#   - so101_bimanual/inference/old_dataset/run_inference_cap_pen_smolvla.sh：SmolVLA 推理脚本（num_steps 用法、rtc 后端来源）
# 机器人类型：seeed_b601_rs_follower（单臂官方类型，见 lerobot_robot_seeed_b601 包）
# 模型权重：/home/kf/LX/pai0/220_new_datasets_smolvla_model/80k_pretrained_model（80000 步 SmolVLA，7 关节 = 单臂）
#   训练配置要点（80k_pretrained_model/{config,train_config}.json）：
#     - type=smolvla，chunk_size=50，n_action_steps=50，num_steps=10（flow matching 去噪步数）
#     - 输入：observation.state(7) + observation.images.hand/front/top（3x480x640），语言指令 = 上面的 task
#     - 输出：action(7)，STATE/ACTION 均 MEAN_STD 归一化（VISUAL=IDENTITY）
#     - VLM 主干：config.json 里是 Hub id，本脚本用 --policy.vlm_model_name 覆盖成
#       /home/kf/LX/pai0/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct 的本地快照（离线可跑）
#     - 训练数据：b601_20260910_164106（30Hz，robot_type=seeed_b601_rs_follower）
#       —— 与 180_new_datasets_act_model 用的数据集完全相同，故任务描述/相机/方向修正沿用 ACT 推理脚本
# 需要把推理过程存成数据集时，改用带 --strategy.type=episodic + --dataset.* 的录制型脚本
# 创建日期：2026-09-20

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
echo "单臂推理（SmolVLA 80k）- B601-RS（make coffee / base 纯推理）"
echo "=========================================="
echo ""

# ==================== 模型配置 ====================
# SmolVLA 80k 权重（pretrained_model 目录，与 ACT 同一数据集 b601_20260910_164106 训练）
MODEL_PATH="/home/kf/LX/pai0/220_new_datasets_smolvla_model/80k_pretrained_model"

# VLM 主干（SmolVLM2-500M-Video-Instruct）本地目录
# 本机无外网：模型 config.json 里存的是 Hub id "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"，
# SmolVLA 会拿它去 AutoConfig/AutoProcessor/AutoTokenizer 取配置与分词器，离线时反复重试
# huggingface.co 并报 Network is unreachable。本地这份权重已经在下面两个路径里，无需重新下载。
#
# VLM_CACHE：HF 标准缓存布局的父目录（内含 models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/{blobs,refs,snapshots}）
#   —— 推理时把它设为 HF_HUB_CACHE，Hub id 就能在离线状态下被解析到本地；
#      这条很关键：模型自带的 policy_preprocessor.json 把 tokenizer_name 写死成 Hub id，
#      而 processor pipeline 不吃 --policy.vlm_model_name 覆盖，只有靠缓存目录才能命中本地分词器。
VLM_CACHE="/home/kf/LX/pai0"
# VLM_PATH：上面的快照目录，同时用 --policy.vlm_model_name 显式指向它
VLM_PATH="$VLM_CACHE/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"

# SmolVLA 用 num_steps（flow matching 去噪步数），须与训练 config.json 的 num_steps=10 一致；
# 想更快可降到 6（动作略糙），想更稳可升到 10 以上（更慢）
NUM_STEPS_ARG="--policy.num_steps=10"

# ==================== 推理后端 ====================
# sync = 同步：每个 chunk 用尽时在控制环里直接前向（会短暂卡顿），也是 action_trace 记录 model 列的前提
# rtc  = 实时分块：后台线程异步生成下一段动作，控制环不阻塞（SmolVLA 支持 RTC，见 modeling_smolvla.supports_rtc）；
#        默认用 rtc，5 步去噪 + 3 路相机的 SmolVLA 单次前向较贵，sync 容易在换 chunk 时抖动
INFERENCE_TYPE="rtc"

# ==================== 动作追踪配置 ====================
# 1 = 用 action_trace.py 包装 lerobot-rollout（运行时打补丁，不改 lerobot 源码），
#     逐 tick 把三组数值追加到 ACTION_TRACE_LOG：
#       model = 策略输出（度，数据集 action 坐标系）——⚠️ 仅在 INFERENCE_TYPE=sync 时有效，
#               rtc 后端不经过 SyncInferenceEngine，该列会是 n/a
#       sent  = 传入 robot.send_action 的 dict（插值器 + robot_action_processor 之后）
#       cmd   = send_action 返回值 = 乘 joint_directions、裁 joint_limits 之后的目标角（度）
#     用于核对"模型输出"到"下发电机"之间到底哪一步改了动作（日志自带 sent!=model /
#     |cmd|!=|sent| 标记与结尾统计）。0 = 直接用 lerobot-rollout，不记录。
#     ⚠️ ACT 专用的注意力可视化（act_feature_viz.py）对 SmolVLA 不适用，本脚本不启用。
ACTION_TRACE=1
ACTION_TRACE_WRAPPER="$HOME/LX/pai0/action_trace.py"
ACTION_TRACE_LOG="$HOME/LX/pai0/action_smolvla_test.log"

# ==================== 硬件配置 ====================
# B601 从臂 = can0（SocketCAN，PEAK PCAN-USB，1Mbps 经典 CAN）
# 推理只用从臂，无需主臂（StarArm102 / /dev/ttyUSB0）
B601_FOLLOWER_PORT="can0"

# 设备 ID（校准文件名，须与采集时一致 = 05 脚本 FOLLOWER_ID）
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

# ==================== B601 方向修正 ====================
# 必须与采集（self_scripts/b601_single/02_record.sh 的 B601_JOINT_DIRECTIONS）完全一致，全 -1.0。
# 原因：seeed_b601_rs_follower.send_action 对每个关节先乘 joint_directions
#       （缺失关节默认 0.0，会把目标裁到 0 不动），且推理动作与训练动作必须同坐标系。
B601_JOINT_DIRECTIONS='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'

# 相对动作幅度上限（robot.max_relative_target，单位：度/步 @30Hz）已停用（同 ACT 脚本）：
#   send_action 里该裁剪比较的是"本帧 goal 与当前位置之差"，而 goal 先被 joint_limits
#   裁进关节量程（最大跨度 ~290 度），因此凡 ≥300 的值都拦不住任何动作，只会每帧刷警告。
#   故不传该参数（默认 None = 不做相对裁剪）；若确实需要限速，参照 b601_so101_bimanual/01_teleop_test.sh。
# SmolVLA 特有：训练用 MEAN_STD 归一化，动作幅度由归一化统计决定，不需要额外限幅。

# ==================== 任务与运行时长 ====================
# base 策略没有 dataset，任务描述必须走顶层 --task 传入
# （见 rollout/context.py：task_str = cfg.dataset.single_task if cfg.dataset else cfg.task）
# SmolVLA 是语言条件策略，该描述会作为 tokenizer 输入，须与采集/训练时完全一致
TASK_DESCRIPTION="Pick up the paper cup, place it on the silver tray of the coffee machine, pick up the cube, press the button with the cube (red light on), wait about 4 seconds, release the button (red light off), put the cube on the table first, then move the cup from the coffee machine to the table"
DURATION=0  # 总运行时长上限（秒）：0 = 不限，一直推理到 Ctrl+C；>0 则到点自动退出
FPS=30  # 推理频率，必须与训练数据集 fps（30Hz）一致，否则 SmolVLA/flow matching 时序不匹配

# ==================== 推理前检查 ====================
# 1) 模型权重（先检查，路径不对直接退出，避免误触硬件）
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 模型路径不存在：$MODEL_PATH"
  exit 1
fi
for f in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
  [ -f "$MODEL_PATH/$f" ] || { echo "❌ 缺少 $MODEL_PATH/$f"; exit 1; }
done
echo "✅ 模型文件存在：$MODEL_PATH"

# 1b) VLM 主干本地权重（缺失则 AutoConfig/AutoProcessor/AutoTokenizer 会去联网，直接提前报错退出）
for f in config.json preprocessor_config.json processor_config.json tokenizer.json; do
  [ -f "$VLM_PATH/$f" ] || { echo "❌ 缺少 VLM 主干文件：$VLM_PATH/$f"; exit 1; }
done
[ -f "$VLM_CACHE/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/refs/main" ] || {
  echo "❌ 缺少 HF 缓存布局（tokenizer_name 无法离线解析）：$VLM_CACHE/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/refs/main"
  exit 1
}
echo "✅ VLM 主干（SmolVLM2-500M）本地路径：$VLM_PATH"

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
echo "推理参数总览（SmolVLA 80k）"
echo "=========================================="
echo "模型路径：$MODEL_PATH"
echo "VLM 主干：$VLM_PATH（本地快照，不走 huggingface.co）"
echo "机器人类型：seeed_b601_rs_follower（单臂 B601-RS，can0）"
echo "策略：base（纯推理，不录制、不保存数据）"
echo "推理方式：${INFERENCE_TYPE}（smolvla 支持 RTC；sync 会阻塞控制环、但可记录 model 动作列）"
echo "去噪步数：${NUM_STEPS_ARG:-（用 config.json 默认值）}"
echo "任务描述：$TASK_DESCRIPTION"
if [ "$DURATION" = "0" ]; then
  echo "运行时长：不限（一直推理到 Ctrl+C）"
else
  echo "运行时长：${DURATION}秒 (约 $((DURATION / 60)) 分钟)"
fi
echo "推理频率：${FPS} Hz"
echo "摄像头：hand / front / top（3 路）"
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
echo "🚀 开始单臂模型推理（SmolVLA 80k）..."
echo ""

# Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

# 离线模式：模型与 VLM 主干/分词器全部走本地，禁止再请求 huggingface.co
#   HF_HUB_CACHE 指向 VLM_CACHE 后，"HuggingFaceTB/SmolVLM2-500M-Video-Instruct" 会在本地缓存里
#   命中（policy_preprocessor.json 里写死的 tokenizer_name 就靠这条解析，否则报
#   "Couldn't instantiate the backend tokenizer"）；如仍需联网拉取则注释掉这三行。
export HF_HUB_CACHE="$VLM_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOG_DIR="$HOME/LX/pai0/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inference_b601_make_coffee_smolvla_80k_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 构建推理命令（lerobot-rollout base 策略：纯推理，无录制、无落盘）
# 说明：
#   - base 策略不创建数据集，因此不能传任何 --dataset.* 参数（传了会直接报错），
#     任务描述改用顶层 --task 传入
#   - SmolVLA：--policy.num_steps 控制 flow matching 去噪步数（训练用 10）
#   - B601：can0 + socketcan + gravity_compensation + joint_directions(全 -1.0)，与采集一致
#   - 机器人 3 路相机输出 key（hand/front/top）与训练时一致，无需 rename_map
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
  --policy.vlm_model_name=$VLM_PATH \
  $NUM_STEPS_ARG \
  --robot.type=seeed_b601_rs_follower \
  --robot.id=$FOLLOWER_ID \
  --robot.port=$B601_FOLLOWER_PORT \
  --robot.can_adapter=socketcan \
  --robot.gravity_compensation=true \
  --robot.joint_directions="$B601_JOINT_DIRECTIONS" \
  --robot.cameras="$CAMERAS" \
  --task="$TASK_DESCRIPTION" \
  --fps=$FPS \
  --duration=$DURATION \
  --display_data=true \
  --display_compressed_images=false \
  --play_sounds=false 2>&1 | tee "$LOG_FILE"

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ 单臂推理结束（SmolVLA 80k / base 模式）"
echo "=========================================="
echo "本次未保存任何推理数据（base 策略不录制、不落盘）。"
echo "完整日志：$LOG_FILE"
echo "=========================================="
