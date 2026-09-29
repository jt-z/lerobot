#!/bin/bash
# DAgger 数据扩充采集（连续录制模式）- 单 B601-RS
# 任务：与 inference/run_inference_ACT.sh 完全一致（倒咖啡）
#
# 目的：
#   用已训练的 ACT 50k 模型自主执行，在出现失败时人工接管纠错，
#   把【自主帧 + 纠错帧】连续录制成新数据集，用于后续微调 / 训练世界模型、VLA。
#
# 参照：
#   - run_inference_b601_make_coffee_ACT_50k.sh：硬件/相机/方向修正/任务描述（推理基准）
#   - self_scripts/b601_single/02_record.sh：主臂 StarArm102 端口 / 校准文件路径
#   - 采集策略实现：lerobot/src/lerobot/rollout/strategies/dagger.py
#
# 硬件：
#   robot  = seeed_b601_rs_follower（B601-RS 从臂，SocketCAN can0）
#   teleop = rebot_102_leader（StarArm102 主臂，/dev/ttyUSB0）  ← DAgger 必需
#
# 采集模式说明（--strategy.record_autonomous=true）：
#   - 策略自主帧（intervention=False）与人工纠错帧（intervention=True）全部录制
#   - episode 按【时间】轮转（不是按任务），时长由 target_video_file_size_mb 反推
#     本配置 3×640x480@30fps：约 9.1 秒/MB → 200MB 约 30 分钟一刀
#   - 后台自动上传按 episode 计数触发
#
# ⚠️ 与纠错模式的关键差异（已知行为，非 Bug）：
#   1. PAUSED 状态下只保持当前位置，【不响应遥操作】，因此无法在 PAUSED 下移动机械臂
#   2. 若要把机械臂手动复位到任务起点，只能在 CORRECTING 下驱动 → 【复位动作会被录进数据集】
#      这些帧同样带任务描述标签，训练世界模型/VLA 时属于标签噪声，请尽量用最少的帧完成复位
#   3. rebot_102_leader 的 feedback_features 为空（非驱动型主臂）：
#      - 暂停时不会自动把主臂拖到从臂位姿
#      - 开始纠错瞬间，从臂会先“平滑滑向主臂当前位姿”（约 1 秒）
#      → 按 Tab 开始纠错前，请先手动把主臂摆到从臂附近，避免从臂突然移动
#   4. 连续模式下 Enter（上传键）不生效：上传只由 episode 计数自动触发
#
# 创建日期：2026-09-14

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
echo "DAgger 数据扩充采集（连续录制）- B601-RS（倒咖啡）"
echo "=========================================="
echo ""

# ==================== 模型配置 ====================
# 用于自主执行的 ACT 权重（哪一版策略出错，就针对它做纠错扩充）
# 100k step（8 卡 × batch 8 训练），与 inference/run_inference_ACT.sh 使用的权重一致
MODEL_PATH="/home/kf/LX/pai0/100000/pretrained_model"

# ACT 单次前向解码，无 num_steps / num_inference_steps 参数，留空即可
NUM_STEPS_ARG=""

# ==================== 硬件配置 ====================
# B601 从臂 = can0（SocketCAN，1Mbps 经典 CAN）
B601_FOLLOWER_PORT="can0"
FOLLOWER_ID="follower"

# StarArm102 主臂 = /dev/ttyUSB0（CH340 无序列号，无 by-id 稳定路径）
B601_LEADER_PORT="/dev/ttyUSB0"
LEADER_ID="leader"

# ==================== 摄像头配置 ====================
# 3 路相机（与采集/推理一致，路径用 /dev/v4l/by-id 稳定路径）：
#   hand = JYU2C 2608076（腕部）、front = JYU2C 2607031、top = JYU2C 2607060（顶部）
CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== B601 方向修正 ====================
# 必须与采集/推理完全一致，全 -1.0（缺失关节默认 0.0，会把目标裁到 0 不动）
B601_JOINT_DIRECTIONS='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'

# 每 tick 相对动作幅度上限（度/步，安全帽）。与推理脚本一致用 100
MAX_RELATIVE_TARGET=100

# ==================== 数据集配置 ====================
# 注意：lerobot-rollout 强制数据集名以 rollout_ 开头（见 rollout/context.py）
DAGGER_REPO_ID="hellozjt/rollout_b601_make_coffee_dagger"
# 本地保存目录（root 即数据集目录本身，配合 no_stamp=true 保证目录名可控）
# 首次运行要求该目录不存在或为空（LeRobotDataset.create）；
# 想继续往里追加 episode：同一 repo_id/root，在命令上加 --resume=true
DAGGER_ROOT="$HOME/LX/pai0/b601_data/rollout_b601_make_coffee_dagger"

# 任务描述须与采集/训练时（self_scripts/b601_single/02_record.sh / 训练数据 meta）完全一致
TASK_DESCRIPTION="Pick up the paper cup, place it on the silver tray of the coffee machine, pick up the cube, press the button with the cube (red light on), wait about 4 seconds, release the button (red light off), put the cube on the table first, then move the cup from the coffee machine to the table"

FPS=30  # 必须与训练数据集 fps（30Hz）一致

# 本次目标：先只录 1 个 episode 的完整任务轨迹（自主帧 + 纠错帧）
#   - 正常结束：完成一个任务周期后按 ESC（ESC 会保存当前 episode 并 finalize）
#   - 时长上限 900s 仅作兜底：小于 200MB 触发的自动轮转（约 30 分钟），
#     即使忘了按 ESC，也只会保存 1 个 episode（900s 时中断并保存）
SESSION_DURATION=900

# episode 轮转阈值（视频体积，MB）。估算：约 9.1 秒/MB（3×640x480@30fps）
#   200MB ≈ 30 分钟一刀（库默认）；50MB ≈ 7.6 分钟；20MB ≈ 3 分钟
EPISODE_VIDEO_MB=200

# 后台自动上传间隔（episode 数）。离线环境务必设大：
# 连续模式下后台 push 会占用 episode 保存锁，网络不可达时反复失败/超时会拖慢保存。
# （注销点：dagger.py::_background_push 未检查 --dataset.push_to_hub）
UPLOAD_EVERY_N=999999

# 是否推送到 Hugging Face Hub（外网不可达时保持 false）
PUSH_TO_HUB=false
export HF_ENDPOINT=https://hf-mirror.com

# ==================== 采集前检查 ====================
# 1) 模型权重
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 模型路径不存在：$MODEL_PATH"
  exit 1
fi
[ -f "$MODEL_PATH/config.json" ] || { echo "❌ 缺少 $MODEL_PATH/config.json"; exit 1; }
[ -f "$MODEL_PATH/model.safetensors" ] || { echo "❌ 缺少 $MODEL_PATH/model.safetensors"; exit 1; }
echo "✅ 模型文件存在：$MODEL_PATH"

# 2) B601 CAN
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

# 3) 主臂串口（DAgger 必需）
echo ""
if [ ! -e "$B601_LEADER_PORT" ]; then
  echo "❌ 主臂串口不存在：$B601_LEADER_PORT"
  echo "   请确认未被 brltty 抢占（sudo apt remove brltty）"
  exit 1
fi
[ -r "$B601_LEADER_PORT" ] || sudo chmod 666 "$B601_LEADER_PORT"
echo "✅ 主臂串口已就绪：$B601_LEADER_PORT"

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

# 5) 校准文件（从臂 + 主臂）
echo ""
CALIB_OK=true
for f in \
  "$HOME/.cache/huggingface/lerobot/calibration/robots/seeed_b601_rs_follower/${FOLLOWER_ID}.json" \
  "$HOME/.cache/huggingface/lerobot/calibration/teleoperators/rebot_102_leader/${LEADER_ID}.json"; do
  if [ -f "$f" ]; then
    echo "✅ $f"
  else
    echo "❌ 缺失：$f（请先运行 self_scripts/b601_common/02_follower_calibration.sh / 01_leader_calibration.sh）"
    CALIB_OK=false
  fi
done
if [ "$CALIB_OK" = false ]; then
  exit 1
fi

# 6) 输出目录（DAgger 建库要求目录不存在或为空，避免覆盖已有数据）
echo ""
echo "✅ 数据集保存目录：$DAGGER_ROOT"

# ==================== 采集参数总览 ====================
echo ""
echo "=========================================="
echo "DAgger 采集参数总览"
echo "=========================================="
echo "策略：dagger（连续录制模式 record_autonomous=true）"
echo "模型路径：$MODEL_PATH"
echo "机器人：seeed_b601_rs_follower（单臂 B601-RS，can0）"
echo "主臂：rebot_102_leader（StarArm102，$B601_LEADER_PORT）"
echo "数据集 repo_id：$DAGGER_REPO_ID"
echo "本地保存路径：$DAGGER_ROOT"
echo "任务描述：$TASK_DESCRIPTION"
echo "会话时长上限：${SESSION_DURATION}秒 (约 $((SESSION_DURATION / 60)) 分钟，0=不限)"
echo "episode 轮转：每约 $((EPISODE_VIDEO_MB * 9)) 秒（${EPISODE_VIDEO_MB}MB 视频）"
echo "采集频率：${FPS} Hz"
echo "摄像头：hand / front / top（3 路）"
echo "B601 方向修正：全 -1.0（与采集/推理一致）"
echo "上传到 Hub：$PUSH_TO_HUB（后台自动上传已关闭）"
echo "=========================================="
echo ""
echo -e "📝 按键控制（焦点保持在终端窗口）："
echo -e "   空格 Space = 暂停 / 恢复【策略自主执行】（AUTONOMOUS <-> PAUSED）"
echo -e "   Tab        = 开始 / 结束【人工纠错录制】（PAUSED -> CORRECTING -> PAUSED）"
echo -e "   r 或 ←     = 丢弃当前 episode 已录的帧，不落盘（纠错失败时用它重来）"
echo -e "   ESC        = 结束整个采集（保存当前 episode 并 finalize 数据集）"
echo -e "   Enter      = 无效（连续模式下上传键不生效，上传由 episode 计数触发）"
echo ""
echo -e "🧭 典型操作流程（一次任务周期）："
echo -e "   1. 策略自主执行，观察是否将要失败"
echo -e "   2. 按 空格 暂停 -> 从臂保持不动，先把主臂手动摆到从臂附近"
echo -e "   3. 按 Tab 开始纠错 -> 从臂平滑滑到主臂位姿，随后由你接管"
echo -e "   4. 纠错 / 复位完成后，按 Tab 结束纠错，再按 空格 恢复策略自主执行"
echo -e "   5. 纠错搞砸了：还没按 Tab 之前直接按 r（或 ←）= 丢掉已录的帧，退回 PAUSED 重来"
echo -e "      （连续模式下丢掉的是【本集自上次保存以来】的全部帧，等于这一遍从头再来）"
echo -e "   6. 重复 1-5；结束采集用 ESC（会保存当前 episode）"
echo ""

read -p "确认从臂已使能、周边安全、主臂已握在手中（防跌落），按 ENTER 开始采集，Ctrl+C 取消..." dummy

# ==================== 开始采集 ====================
echo ""
echo "🚀 开始 DAgger 数据扩充采集..."
echo ""

# Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

LOG_DIR="$HOME/LX/pai0/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/dagger_b601_make_coffee_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 构建采集命令（lerobot-rollout + dagger 策略，连续录制模式）
# 说明：
#   - ACT 不支持 RTC，必须 --inference.type=sync
#   - record_autonomous=true：自主帧与纠错帧都录
#   - streaming_encoding 强制开启（边采边编码 h264，避免磁盘 I/O 阻塞控制循环）
#   - no_stamp=true + 显式 root：保证保存目录名可控（否则会自动追加时间戳）
#   - strategy.num_episodes 在连续模式下不被使用，故不传（仅从 dataset.num_episodes 解析）
# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂、进程被 SIGPIPE 杀死而
# 跳过 teardown（否则数据集不会 finalize、电机不释放）。
trap '' PIPE

set +e
lerobot-rollout \
  --strategy.type=dagger \
  --strategy.record_autonomous=true \
  --strategy.upload_every_n_episodes=$UPLOAD_EVERY_N \
  --strategy.target_video_file_size_mb=$EPISODE_VIDEO_MB \
  --inference.type=sync \
  --policy.path=$MODEL_PATH \
  $NUM_STEPS_ARG \
  --robot.type=seeed_b601_rs_follower \
  --robot.id=$FOLLOWER_ID \
  --robot.port=$B601_FOLLOWER_PORT \
  --robot.can_adapter=socketcan \
  --robot.gravity_compensation=true \
  --robot.joint_directions="$B601_JOINT_DIRECTIONS" \
  --robot.cameras="$CAMERAS" \
  --robot.max_relative_target=$MAX_RELATIVE_TARGET \
  --teleop.type=rebot_102_leader \
  --teleop.id=$LEADER_ID \
  --teleop.port=$B601_LEADER_PORT \
  --dataset.repo_id=$DAGGER_REPO_ID \
  --dataset.root=$DAGGER_ROOT \
  --dataset.no_stamp=true \
  --dataset.single_task="$TASK_DESCRIPTION" \
  --dataset.fps=$FPS \
  --fps=$FPS \
  --dataset.video=true \
  --dataset.streaming_encoding=true \
  --dataset.rgb_encoder.vcodec=h264 \
  --dataset.push_to_hub=$PUSH_TO_HUB \
  --duration=$SESSION_DURATION \
  --display_data=true \
  --display_compressed_images=false 2>&1 | tee "$LOG_FILE"
DAGGER_EXIT=${PIPESTATUS[0]}
set -e

if [ $DAGGER_EXIT -ne 0 ]; then
  echo ""
  echo "⚠️  lerobot-rollout 非零退出（exit=$DAGGER_EXIT）"
  echo "    若日志显示 episode 已保存/视频已编码，通常是断开机械臂时的清理错误，数据不受影响。"
fi

# ==================== 采集完成 ====================
echo ""
echo "=========================================="
echo "✅ DAgger 数据扩充采集结束"
echo "=========================================="
echo "数据集 repo_id：$DAGGER_REPO_ID"
echo "本地保存路径：$DAGGER_ROOT"

if [ -f "$DAGGER_ROOT/meta/info.json" ]; then
  EPISODES=$(python3 -c "import json; print(json.load(open('$DAGGER_ROOT/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")
  FRAMES=$(python3 -c "import json; print(json.load(open('$DAGGER_ROOT/meta/info.json'))['total_frames'])" 2>/dev/null || echo "?")
  echo "本次采集：$EPISODES 个 episode，$FRAMES 帧"
else
  echo "⚠️  未找到 $DAGGER_ROOT/meta/info.json，请检查日志"
fi
echo "完整日志：$LOG_FILE"
echo ""
echo "💡 后续步骤（把 DAgger 数据并入原数据集后微调）："
echo "   1) 去掉 intervention 特征（原数据集没有该列，否则 merge 会因特征不一致报错）"
echo "   2) merge 到原 b601 数据集"
echo "   3) 用合并后的数据集微调策略"
echo "   详细命令见脚本末尾注释。"
echo "=========================================="

# =============================================================================
# 后续步骤详解（手动执行，不在本脚本内自动跑）
# -----------------------------------------------------------------------------
# 本脚本产出的 DAgger 数据集（多一列 intervention）：
#   repo_id = hellozjt/rollout_b601_make_coffee_dagger
#   root    = /home/kf/LX/pai0/b601_data/rollout_b601_make_coffee_dagger
# 原训练数据集：
#   repo_id = b601_20260910_164106
#   root    = /home/kf/LX/pai0/b601_data/b601_20260910_164106
#
# 步骤 1：去掉 intervention 特征（原数据集没有该列，否则 merge 会因特征不一致报错）
#   lerobot-edit-dataset \
#     --repo_id hellozjt/rollout_b601_make_coffee_dagger \
#     --root /home/kf/LX/pai0/b601_data/rollout_b601_make_coffee_dagger \
#     --new_repo_id hellozjt/rollout_b601_make_coffee_dagger_nointerv \
#     --new_root /home/kf/LX/pai0/b601_data/rollout_b601_make_coffee_dagger_nointerv \
#     --operation.type remove_feature \
#     --operation.feature_names "['intervention']"
#
# 步骤 2：合并【原数据集 + DAgger 数据集】
#   （merge 会忽略 --repo_id/--root，只认 --new_repo_id/--new_root 与 operation.repo_ids/roots）
#   lerobot-edit-dataset \
#     --new_repo_id b601_make_coffee_merged \
#     --new_root /home/kf/LX/pai0/b601_data/b601_make_coffee_merged \
#     --operation.type merge \
#     --operation.repo_ids "['b601_20260910_164106', 'hellozjt/rollout_b601_make_coffee_dagger_nointerv']" \
#     --operation.roots "['/home/kf/LX/pai0/b601_data/b601_20260910_164106', '/home/kf/LX/pai0/b601_data/rollout_b601_make_coffee_dagger_nointerv']"
#
#   合并前置条件（aggregate 会强校验，见 datasets/aggregate.py）：
#     - fps 一致（都 30）
#     - robot_type 一致（都 seeed_b601_rs_follower）
#     - features 完全一致（去掉 intervention 后应一致；视频编码参数差异会被忽略）
#
# 步骤 3：用合并后的数据集微调
#   python lerobot/src/lerobot/scripts/lerobot_train.py \
#     --dataset.repo_id=b601_make_coffee_merged \
#     --dataset.root=/home/kf/LX/pai0/b601_data/b601_make_coffee_merged \
#     --policy.path=/home/kf/LX/pai0/50kact_b601/pretrained_model \
#     --output_dir=... --batch_size=... --steps=...
# =============================================================================
