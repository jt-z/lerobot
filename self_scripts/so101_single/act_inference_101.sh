#!/bin/bash
# 单臂 SO-101 ACT 推理：从臂 so101_follower（USB 串口，6 关节）
# 默认：推理 + 录数据（strategy=episodic）—— 推理边跑边把数据录成数据集：
#   右键 = 结束当前 episode 并保存（保存刚推理的这一段）| 左键 = 丢弃当前 episode（不保存）
#   | ESC = 结束整个会话；每次运行新建一个文件夹（$DATA_ROOT/<repo_id>，rollout_ 前缀）。
# 纯推理（不建数据集、不录帧、不编码、不 finalize）：第一个参数传 base。
# 任务：夹取不同角度的黄色香蕉放到白色水果盘上
#       （与 self_scripts/so101_single/record_101.sh 采集的任务描述 / 相机 / fps 完全一致）
#
# 参照：
#   - act_inference.sh：单臂 B601-RS ACT 推理脚本（检查项/日志/参数风格）
#   - self_scripts/so101_single/record_101.sh：单臂 SO-101 采集脚本（端口 / 相机 / fps / 任务描述 / 校准 id）
# 机器人类型：so101_follower（单臂 SO-101 从臂）
# 模型权重：/home/kf/LX/pai0/020000/pretrained_model（20000 步 ACT）
#   训练配置要点（020000/pretrained_model/{config,train_config}.json）：
#     - type=act，chunk_size=100，n_action_steps=100，无 temporal ensemble（单次前向解码）
#     - 输入：observation.state(6) + observation.images.hand/front（2x480x640）
#     - 输出：action(6)，MEAN_STD 归一化
#     - 训练数据：so101_20260921_172236（92 集，30Hz，robot_type=so_follower）
#   注意：SO-101 与 B601 不同 —— so101_follower 没有 gravity_compensation / joint_directions
#         参数，也不需要 CAN 配置，故本脚本不含这些项。
#
# 用法：
#   bash self_scripts/so101_single/act_inference_101.sh           # 默认：推理 + 录数据（右键保存 / 左键丢弃）
#   bash self_scripts/so101_single/act_inference_101.sh base      # 纯推理（不录、不存）
#   NUM_EPISODES=20 bash self_scripts/so101_single/act_inference_101.sh        # 录 20 段后自动结束
#
# 录制说明：
#   - 每个 episode 到 EPISODE_TIME 秒也会自动保存一次（防止忘记按键丢数据），
#     想在读秒前结束就按右键；这一整段不想要就按左键（丢弃并重录）
#   - 键位与 self_scripts/so101_single/record_101.sh 一致，存下来的数据可直接与采集数据一起训练
#
# 可按需覆盖的环境变量：
#   FOLLOWER_PORT / FOLLOWER_ID / MODEL_PATH / CAMERAS / TASK_DESCRIPTION / FPS / DURATION
#   MAX_REL_TARGET   每步每关节最大相对位移（度），默认不限
#   SAVE_DATA（默认 true）录制开关；SAVE_DATA=false 等价于第一个参数传 base
#   DATA_ROOT / DATASET_NAME 存放根目录（默认 $HOME/LX/pai0/rollout_data）
#                            与数据集名（默认 rollout_so101_act_<时间戳>；必须 rollout_ 前缀）
#   NUM_EPISODES / EPISODE_TIME / RESET_TIME / PUSH_TO_HUB  录制参数

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

# ==================== 模式 ====================
# 默认就是"推理 + 录数据"；只有显式传 base（或 SAVE_DATA=false）才是纯推理
SAVE_DATA="${SAVE_DATA:-true}"
case "${1:-}" in
  save) SAVE_DATA=true;  shift ;;
  base) SAVE_DATA=false; shift ;;
  "")   ;;
  *) echo "❌ 未知参数：$1（可用：base = 纯推理；不带参数 = 推理+录数据）"; exit 1 ;;
esac

echo "=========================================="
echo "单臂推理（ACT）- SO-101（$( [ "$SAVE_DATA" = true ] && echo "episodic 推理+录数据" || echo "base 纯推理" )）"
echo "=========================================="
echo ""

# ==================== 模型配置 ====================
# 单臂 SO-101 的 ACT 权重（pretrained_model 目录，内含 config.json + model.safetensors）
MODEL_PATH="${MODEL_PATH:-/home/kf/LX/pai0/045000/pretrained_model}"

# ACT 单次前向解码，无 num_steps / num_inference_steps 参数，留空即可
NUM_STEPS_ARG=""

# ==================== 硬件配置 ====================
# 从臂 = SO-101（USB 单串口，by-id 稳定路径；ttyACM* 编号随插拔顺序变化不写死）
# 推理只用从臂，无需主臂（so101_leader / 5B61034865）
SO101_FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00}"

# 设备 ID（校准文件名，须与采集时一致 = record_101.sh 的 FOLLOWER_ID）
FOLLOWER_ID="${FOLLOWER_ID:-jt_follower_arm_right}"

# 每步每关节最大相对位移（度）。留空 = 不限幅（与采集一致）
MAX_REL_TARGET="${MAX_REL_TARGET:-}"

# ==================== 摄像头配置 ====================
# 2 路相机，与 record_101.sh 采集一致：
#   hand  = JYU2C 2609007（腕部/近景，跟着夹爪动）
#   front = JYU2C 2608006（前视第三视角）
# ⚠️ key 决定 observation 字段名（observation.images.hand / .front），必须与训练时一致；
#    路径用 /dev/v4l/by-id 稳定路径（/dev/videoN 随插拔顺序变化不可靠）。
DEFAULT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2609007-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608006-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'
# 注意：默认值必须单独放一个变量再引用，不要写成 ${CAMERAS-{...}}（bash 会在第一个 } 处截断）
CAMERAS="${CAMERAS-$DEFAULT_CAMERAS}"

# ==================== 任务与运行时长 ====================
# base 策略没有 dataset，任务描述必须走顶层 --task 传入
# （见 rollout/context.py：task_str = cfg.dataset.single_task if cfg.dataset else cfg.task）
# 任务描述须与采集时（record_101.sh）及训练数据 meta/tasks.parquet 完全一致
TASK_DESCRIPTION="${TASK_DESCRIPTION:-Pick up the yellow banana placed at different angles and put it on the white fruit plate}"
DURATION="${DURATION:-0}"  # 总运行时长上限（秒）：0 = 不限，一直推理到 Ctrl+C；>0 则到点自动退出
FPS="${FPS:-30}"  # 推理频率，必须与训练数据集 fps（30Hz）一致，否则 ACT 时序不匹配

# ==================== 录制数据集配置（默认开启） ====================
# 策略：episodic（lerobot-rollout 的录制型策略，键位与 record_101.sh 一致）
#   - rollout 的数据集名必须以 rollout_ 开头（见 rollout/context.py 的校验）
#   - 每次运行新建一个带时间戳的文件夹；--dataset.no_stamp=true 保证目录名与 repo_id 一致
#   - push_to_hub=false：本机外网不可达，数据只存本地
DATA_ROOT="${DATA_ROOT:-$HOME/LX/pai0/rollout_data}"
if [ "$SAVE_DATA" = true ]; then
  DATASET_NAME="${DATASET_NAME:-rollout_so101_act_$(date +%Y%m%d_%H%M%S)}"
  DATA_DIR="$DATA_ROOT/$DATASET_NAME"
  NUM_EPISODES="${NUM_EPISODES:-10}"        # 录满这么多段就自动结束（ESC 可提前收工）
  EPISODE_TIME="${EPISODE_TIME:-60}"        # 单段最长时间（秒）：到点也会自动保存
  RESET_TIME="${RESET_TIME:-10}"            # 段间重置时长（秒）：可选，按右键可提前跳过
  PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
  mkdir -p "$DATA_ROOT"
fi

if [ -z "$TASK_DESCRIPTION" ]; then
  echo "❌ 错误：TASK_DESCRIPTION 为空"
  exit 1
fi

# ==================== 推理前检查 ====================
# 1) 模型权重（先检查，路径不对直接退出，避免误触硬件）
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 模型路径不存在：$MODEL_PATH"
  exit 1
fi
[ -f "$MODEL_PATH/config.json" ] || { echo "❌ 缺少 $MODEL_PATH/config.json"; exit 1; }
[ -f "$MODEL_PATH/model.safetensors" ] || { echo "❌ 缺少 $MODEL_PATH/model.safetensors"; exit 1; }
echo "✅ 模型文件存在：$MODEL_PATH"

# 2) 从臂串口
echo ""
if [ ! -e "$SO101_FOLLOWER_PORT" ]; then
  echo "❌ 串口不存在 $SO101_FOLLOWER_PORT"
  echo "   当前串口设备：$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | tr '\n' ' ')"
  exit 1
fi
if [ ! -r "$SO101_FOLLOWER_PORT" ] || [ ! -w "$SO101_FOLLOWER_PORT" ]; then
  echo "⚠️  $SO101_FOLLOWER_PORT 无读写权限，尝试授权..."
  sudo chmod 666 "$SO101_FOLLOWER_PORT"
fi
echo "✅ 从臂串口已连接：$SO101_FOLLOWER_PORT"

# 舵机应答检查：ping 6 个 STS3215（id 1~6），确认臂已上电、端口确实对着 SO-101
# 说明：ping 返回 (型号, 通信结果, 错误码)，只有「通信结果 == 0」才算应答（超时返回型号 0）
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
    echo "✅ 从臂舵机应答: $hits"
  else
    echo "❌ 从臂舵机应答不全（应答: ${hits:-无}）"
    echo "   1) 确认该臂 12V 电源已开、USB 已插好"
    echo "   2) 确认这条串口对应的确实是 SO-101（不是别的设备）"
    exit 1
  fi
else
  echo "⚠️  跳过舵机应答检查（当前 python 缺 scservo_sdk；请确认已 conda activate lerobot）"
fi

# 3) 摄像头（key ↔ 设备映射，key 写错/接错在开始前就能看出来）
echo ""
if [ -z "$CAMERAS" ]; then
  echo "❌ CAMERAS 为空：ACT 模型需要 hand/front 两路图像输入，不能只喂关节"
  exit 1
fi
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
    echo "  ❌ $key = $cam（设备不存在）"
    CAM_OK=false
  fi
done
if [ "$CAM_OK" = false ]; then
  echo "  请改脚本里的 CAMERAS（by-id 路径可用 ls /dev/v4l/by-id/ 确认）"
  exit 1
fi

# 4) 从臂校准文件（推理需与采集同 id；SO-101 的 robot.name = so_follower）
echo ""
CALIB_FILE="$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/${FOLLOWER_ID}.json"
if [ -f "$CALIB_FILE" ]; then
  echo "✅ $CALIB_FILE"
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
if [ "$SAVE_DATA" = true ]; then
  echo "策略：episodic（推理 + 录数据）"
else
  echo "策略：base（纯推理，不录制、不保存数据）"
fi
echo "推理方式：sync（ACT 不支持 RTC）"
echo "任务描述：$TASK_DESCRIPTION"
if [ "$DURATION" = "0" ]; then
  echo "运行时长：不限（一直推理到 Ctrl+C）"
else
  echo "运行时长：${DURATION}秒 (约 $((DURATION / 60)) 分钟)"
fi
echo "推理频率：${FPS} Hz"
echo "摄像头：$(echo "$CAM_ENTRIES" | sed 's/=[^ ]*//g' | tr '\n' ' ')"
if [ -n "$MAX_REL_TARGET" ]; then
  echo "相对动作上限：${MAX_REL_TARGET} 度/步"
else
  echo "相对动作上限：未启用（不传 max_relative_target，默认不做相对裁剪）"
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
echo "🚀 开始单臂 SO-101 模型推理（ACT 20k）..."
echo ""

# Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

LOG_DIR="$HOME/LX/pai0/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inference_so101_act_20k_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 推理命令说明（lerobot-rollout）：
#   - base 策略（默认）：不创建数据集，因此不能传任何 --dataset.* 参数（传了会直接报错），
#     任务描述改用顶层 --task 传入
#   - episodic 策略（SAVE_DATA=true）：必须带 --dataset.*，逐段录制落盘（右键保存 / 左键丢弃）
#   - ACT 不支持 RTC，必须 --inference.type=sync
#   - 相机 key（hand/front）与训练时一致，无需 rename_map；数据集里存的也是 hand/front
#   - --fps 须 30（训练数据 30Hz）；--duration=0 表示不限时长（跑到 Ctrl+C）

# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂、lerobot-rollout 被 SIGPIPE(141)
# 杀死而跳过断开清理（否则从臂保持使能、不释放）。
trap '' PIPE

# 用数组拼参数：CAMERAS / 任务描述里带 JSON、空格、花括号，数组能整块原样传给
# lerobot-rollout，不用 eval 也不会被二次分词。
ROLLOUT_ARGS=(
  --inference.type=sync
  --policy.path="$MODEL_PATH"
  # NUM_STEPS_ARG 是"多个 flag 拼成的字符串"（ACT 下为空），不加引号让它按空格拆开
  # shellcheck disable=SC2206
  $NUM_STEPS_ARG
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
if [ -n "$MAX_REL_TARGET" ]; then
  ROLLOUT_ARGS+=(--robot.max_relative_target="$MAX_REL_TARGET")
fi

set +e  # Ctrl+C 退出码 130，异常退出非 0，都要能捕获后继续打印结果
lerobot-rollout "${ROLLOUT_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
if [ "$STATUS" -eq 0 ] || [ "$STATUS" -eq 130 ]; then
  echo "✅ 单臂推理结束（ACT / $([ "$SAVE_DATA" = true ] && echo "episodic 录数据" || echo "base" ) 模式，从臂已回初始位并失能）"
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
  echo "本次未保存任何推理数据（base 纯推理，不录制、不落盘）。"
  echo "需要边推理边存数据：直接运行 bash self_scripts/so101_single/act_inference_101.sh（默认就是录数据）"
fi
echo "完整日志：$LOG_FILE"
echo "=========================================="
