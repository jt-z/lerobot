#!/bin/bash
# 单 B601-RS ACT 推理：从臂 B601-RS（SocketCAN can0）
# 策略：base —— 纯推理，不建数据集、不录帧、不编码、不 finalize（推理数据不保存）
# 任务：单臂端起纸杯放上咖啡机托盘、拿方块按按钮等 4 秒、方块放回桌面、再把杯子端回桌面
#       （与 self_scripts/b601_single/02_record.sh 采集的任务描述/相机/fps/方向修正完全一致）
#
# 参照：
#   - run_inference_b601_so101_make_coffee_ACT.sh：异构双臂 ACT 推理脚本（历史文件，未随本次重组入库，本脚本据其改写）
#   - self_scripts/b601_single/02_record.sh：硬件端口 / 相机 / fps / 方向修正 / 任务描述（单臂采集基准）
# 机器人类型：seeed_b601_rs_follower（单臂官方类型，见 lerobot_robot_seeed_b601 包）
# 模型权重：/home/kf/dev/lerobot/model_weights/50kact_b601（50000 步 ACT，7 关节 = 单臂）
#   训练配置要点（50kact_b601/pretrained_model/{config,train_config}.json）：
#     - type=act，chunk_size=100，n_action_steps=100，无 temporal ensemble（单次前向解码）
#     - 输入：observation.state(7) + observation.images.hand/front/top（3x480x640）
#     - 输出：action(7)，MEAN_STD 归一化
#     - 训练数据：30Hz，robot_type=seeed_b601_rs_follower
# 需要把推理过程存成数据集时，改用 self_scripts/b601_single/04_dagger_collect.sh
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
echo "单臂推理（ACT 50k）- B601-RS（倒咖啡 / base 纯推理）"
echo "=========================================="
echo ""

# ==================== 模型配置 ====================
# 新数据集微调后的 ACT 权重（pretrained_model 目录）
# MODEL_PATH="/home/kf/dev/lerobot/model_weights/100000/pretrained_model"
MODEL_PATH="/home/kf/dev/lerobot/model_weights/180_new_datasets_act_model/100k_pretrained_model"


# ACT 单次前向解码，无 num_steps / num_inference_steps 参数，留空即可
NUM_STEPS_ARG=""

# ==================== 动作追踪配置 ====================
# 1 = 用 action_trace.py 包装 lerobot-rollout（运行时打补丁，不改 lerobot 源码），
#     逐 tick 把三组数值追加到 ACTION_TRACE_LOG：
#       model = 策略输出（度，数据集 action 坐标系）
#       sent  = 传入 robot.send_action 的 dict（插值器 + robot_action_processor 之后）
#       cmd   = send_action 返回值 = 乘 joint_directions、裁 joint_limits 之后的目标角（度）
#     用于核对"模型输出"到"下发电机"之间到底哪一步改了动作（日志自带 sent!=model /
#     |cmd|!=|sent| 标记与结尾统计）。0 = 直接用 lerobot-rollout，不记录。
ACTION_TRACE=1
ACTION_TRACE_WRAPPER="$(dirname "$(readlink -f "$0")")/../../tools/action_trace.py"
ACTION_TRACE_LOG="$HOME/dev/lerobot/self_scripts/_logs/action_test.log"

# ==================== ACT 内部注意力可视化 ====================
# 1 = 用 act_feature_viz.py 包装 lerobot-rollout（运行时打补丁，不改 lerobot / ACT 源码）：
#     抓取最后一层 decoder 的交叉注意力（100 个动作 query → 902 个 token，其中 3×300 是
#     hand/front/top 三路相机的 15×20 feature map），按 head 聚合后还原成像素网格、
#     上采样并叠加在相机原图上，写进 Rerun 现有窗口，实体 act_viz/attn/<cam>。
#     ACT 每 100 tick 才前向一次（chunk_size=100 @30Hz ≈ 3.3s），所以缓存整段注意力，
#     每 ACT_VIZ_EVERY_N 次取动作换一帧显示"当前执行到 chunk 内第 k 步"的注意力分布
#     （默认 1 = 30Hz，最流畅；调大省 CPU 但画面会跳）。
#     分析视图（ACT_VIZ_ANALYSIS=1，默认开；全部复用已算出的张量，按低帧率更新，参见各 MP4）：
#       attn_heads/<cam>  8 个 head 各自的 15×20 热图（2×4，左上→右下 = head 0..7）→ head 分工
#       timeline/<cam>    横轴=chunk 内第 k 步、纵轴=token 的"步骤×空间"图 → 注意力随动作推进的漂移
#       mass/<cam>        每 tick 一个点：各路相机的注意力质量 ΣA（+ latent/state 质量、当前 k）
#       selfattn          100×100 decoder 自注意力（行=query 步、列=key 步）→ chunk 内动作是否分段
#     默认同时把每帧落成 MP4（三路横拼一份，方便事后统一回看），见下面的 ACT_VIZ_MP4*。
#     注意：叠图底图是"该 chunk 生成时"的那一帧相机图（注意力与该帧严格对应），
#     因此 step k 越大，底图相对当前时刻越旧，这是 chunk 机制本身决定的。
# 0 = 直接用 lerobot-rollout（无可视化）。
# 与 ACTION_TRACE 可同时为 1：此时统一用 act_feature_viz.py 作入口，
# 它会在装载自身补丁前顺带装上 action_trace.py 的动作追踪补丁。
ACT_VIZ=1
ACT_VIZ_WRAPPER="$(dirname "$(readlink -f "$0")")/../../tools/act_feature_viz.py"
ACT_VIZ_EVERY_N=1     # 每 N 次取动作刷一帧（1 = 30Hz；调大则降帧率、省 CPU）
ACT_VIZ_ANALYSIS=1    # 1 = 额外输出上面四个分析视图与对应的 MP4；0 = 只留主叠图
ACT_VIZ_HEADS_EVERY=10  # per-head 网格刷新间隔（单位 tick，10 ≈ 3Hz）
ACT_VIZ_HEADS=mean    # head 聚合方式：mean | max（只影响主叠图与质量曲线）
ACT_VIZ_ALPHA=0.55    # 热力图最大叠加强度 0~1（按注意力强弱渐变，弱处保留原图）
ACT_VIZ_BLUEPRINT=1   # 1 = 补发含 act_viz 视图的 Rerun blueprint（会覆盖 lerobot 默认布局）

# ---- 离线 MP4（默认开启，跑完看一个文件即可，和 Rerun 是否可用无关）----
ACT_VIZ_MP4=1         # 1 = 每帧同时写 MP4；0 = 只在线看
ACT_VIZ_MP4_DIR=""    # 输出目录；留空 = self_scripts/tools/logs/act_viz_<启动时间戳>/
ACT_VIZ_MP4_MODE=tiled  # tiled（默认，hand/front/top 横拼成 1920x480 一个文件）| per_cam | both
ACT_VIZ_MP4_CRF=23    # x264 质量（越小越清晰、文件越大）
# 帧率自动取 --fps/(EVERY_N×interpolation_multiplier)，保证视频时长=真实时长。
# Ctrl+C 正常退出会写 moov 收尾；kill -9 会导致 MP4 不完整。

# ==================== 硬件配置 ====================
# B601 从臂 = can0（SocketCAN，PEAK PCAN-USB，1Mbps 经典 CAN）
# 推理只用从臂，无需主臂（StarArm102 / /dev/ttyUSB0）
B601_FOLLOWER_PORT="can0"

# 设备 ID（校准文件名，须与采集时一致 = self_scripts/b601_single/02_record.sh 的 FOLLOWER_ID）
FOLLOWER_ID="follower"

# ==================== 摄像头配置 ====================
# 3 路相机（与 self_scripts/b601_single/02_record.sh 采集一致，2026-09 实测）：
#   hand = JYU2C 2608076（腕部）、front = JYU2C 2607031、top = JYU2C 2607060（顶部）
# 路径用 /dev/v4l/by-id 稳定路径（/dev/videoN 随插拔顺序变化不可靠）。
CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== B601 方向修正 ====================
# 必须与采集（self_scripts/b601_single/02_record.sh B601_JOINT_DIRECTIONS）完全一致，全 -1.0。
# 原因：seeed_b601_rs_follower.send_action 对每个关节先乘 joint_directions
#       （缺失关节默认 0.0，会把目标裁到 0 不动），且推理动作与训练动作必须同坐标系。
B601_JOINT_DIRECTIONS='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'

# 相对动作幅度上限（robot.max_relative_target，单位：度/步 @30Hz）已停用：
#   send_action 里该裁剪比较的是"本帧 goal 与当前位置之差"，而 goal 先被 joint_limits
#   裁进关节量程（最大跨度 ~290 度），因此凡 ≥300 的值都拦不住任何动作，只会每帧刷
#   "Relative goal position magnitude had to be clamped" 警告（30/100/1000 均如此）。
#   故不传该参数（默认 None = 不做相对裁剪）；若确实需要限速，参照
#   self_scripts/b601_so101_bimanual/01_teleop_test.sh 里的逐关节小值（如 5/3/3/5/5/5/10 度/步），代价是动作变慢。

# ==================== 任务与运行时长 ====================
# base 策略没有 dataset，任务描述必须走顶层 --task 传入
# （见 rollout/context.py：task_str = cfg.dataset.single_task if cfg.dataset else cfg.task）
# 任务描述须与采集时（self_scripts/b601_single/02_record.sh / 训练数据 meta/tasks.parquet）完全一致
TASK_DESCRIPTION="Pick up the paper cup, place it on the silver tray of the coffee machine, pick up the cube, press the button with the cube (red light on), wait about 4 seconds, release the button (red light off), put the cube on the table first, then move the cup from the coffee machine to the table"
DURATION=0  # 总运行时长上限（秒）：0 = 不限，一直推理到 Ctrl+C；>0 则到点自动退出
FPS=30  # 推理频率，必须与训练数据集 fps（30Hz）一致，否则 ACT 时序不匹配

# ==================== 推理前检查 ====================
# 1) 模型权重（先检查，路径不对直接退出，避免误触硬件）
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 模型路径不存在：$MODEL_PATH"
  exit 1
fi
[ -f "$MODEL_PATH/config.json" ] || { echo "❌ 缺少 $MODEL_PATH/config.json"; exit 1; }
[ -f "$MODEL_PATH/model.safetensors" ] || { echo "❌ 缺少 $MODEL_PATH/model.safetensors"; exit 1; }
echo "✅ 模型文件存在：$MODEL_PATH"

# 2) 动作追踪包装器（开追踪时必须有，避免跑到一半才发现）
if [ "$ACTION_TRACE" = "1" ] && [ ! -f "$ACTION_TRACE_WRAPPER" ]; then
  echo "❌ 缺少动作追踪包装器：$ACTION_TRACE_WRAPPER（或把 ACTION_TRACE 设为 0）"
  exit 1
fi

# 2b) ACT 内部注意力可视化包装器
if [ "$ACT_VIZ" = "1" ] && [ ! -f "$ACT_VIZ_WRAPPER" ]; then
  echo "❌ 缺少 ACT 可视化包装器：$ACT_VIZ_WRAPPER（或把 ACT_VIZ 设为 0）"
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
echo "推理参数总览"
echo "=========================================="
echo "模型路径：$MODEL_PATH"
echo "机器人类型：seeed_b601_rs_follower（单臂 B601-RS，can0）"
echo "策略：base（纯推理，不录制、不保存数据）"
echo "推理方式：sync（ACT 不支持 RTC）"
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
  echo "动作追踪：✅ model/sent/cmd 逐 tick 记录到 $ACTION_TRACE_LOG"
else
  echo "动作追踪：❌ 关闭（直接调用 lerobot-rollout）"
fi
if [ "$ACT_VIZ" = "1" ]; then
  echo "ACT 注意力可视化：✅ decoder 交叉注意力 → Rerun act_viz/attn/{hand,front,top}"
  echo "                    每 ${ACT_VIZ_EVERY_N} 次取动作换一帧（滚动 chunk 内第 k 步），head 聚合=${ACT_VIZ_HEADS}，alpha=${ACT_VIZ_ALPHA}"
  if [ "$ACT_VIZ_ANALYSIS" = "1" ]; then
    echo "ACT 分析视图：✅ attn_heads/<cam>（${ACT_VIZ_HEADS_EVERY} tick 刷一次）、timeline/<cam>（每 chunk）、mass/*（每 tick）、selfattn（每 chunk）"
  else
    echo "ACT 分析视图：❌ 关闭（ACT_VIZ_ANALYSIS=0）"
  fi
  if [ "$ACT_VIZ_MP4" = "1" ]; then
    echo "ACT 注意力 MP4：✅ 模式=${ACT_VIZ_MP4_MODE}，crf=${ACT_VIZ_MP4_CRF}，输出=${ACT_VIZ_MP4_DIR:-self_scripts/tools/logs/act_viz_<时间戳>/}"
    if [ "$ACT_VIZ_ANALYSIS" = "1" ]; then
      echo "                   外加 act_analysis_heads / _timeline / _selfattn 三个分析视频（低帧率）"
    fi
  else
    echo "ACT 注意力 MP4：❌ 关闭（ACT_VIZ_MP4=0）"
  fi
else
  echo "ACT 注意力可视化：❌ 关闭"
fi
echo "录制/保存数据：❌ 否（base 策略不录制）"
echo "=========================================="
echo ""

read -p "确认 B601 从臂（can0）处于零位（夹爪闭合）、周边安全，按 ENTER 开始推理，Ctrl+C 取消..." dummy

# ==================== 开始推理 ====================
echo ""
echo "🚀 开始单臂模型推理（ACT 50k）..."
echo ""

# Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

LOG_DIR="$HOME/dev/lerobot/self_scripts/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inference_b601_make_coffee_act_50k_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

# 构建推理命令（lerobot-rollout base 策略：纯推理，无录制、无落盘）
# 说明：
#   - base 策略不创建数据集，因此不能传任何 --dataset.* 参数（传了会直接报错），
#     任务描述改用顶层 --task 传入
#   - ACT 不支持 RTC，必须 --inference.type=sync
#   - B601：can0 + socketcan + gravity_compensation + joint_directions(全 -1.0)，与采集一致
#   - 机器人 3 路相机输出 key（hand/front/top）与训练时一致，无需 rename_map
#   - --fps 须 30（训练数据 30Hz）；--duration=0 表示不限时长（跑到 Ctrl+C）
#   - ACTION_TRACE=1 时用 action_trace.py 包装（参数完全一致，只多一路动作记录）
#   - ACT_VIZ=1 时用 act_feature_viz.py 包装（参数完全一致，只多一路 ACT 交叉注意力可视化）；
#     它与 ACTION_TRACE 可同时开启：viz 包装器会顺带装上 action_trace 的补丁，
#     所以两者都开时入口用 viz 包装器（优先级高于 action_trace.py）
if [ "$ACT_VIZ" = "1" ]; then
  ROLLOUT_CMD=(python "$ACT_VIZ_WRAPPER")
  export ACT_VIZ_EVERY_N ACT_VIZ_ANALYSIS ACT_VIZ_HEADS_EVERY ACT_VIZ_HEADS ACT_VIZ_ALPHA ACT_VIZ_BLUEPRINT
  export ACT_VIZ_MP4 ACT_VIZ_MP4_DIR ACT_VIZ_MP4_MODE ACT_VIZ_MP4_CRF
  export ACTION_TRACE ACTION_TRACE_LOG  # viz 包装器据此决定是否顺带装载动作追踪
elif [ "$ACTION_TRACE" = "1" ]; then
  ROLLOUT_CMD=(python "$ACTION_TRACE_WRAPPER")
  export ACTION_TRACE_LOG
else
  ROLLOUT_CMD=(lerobot-rollout)
fi
"${ROLLOUT_CMD[@]}" \
  --strategy.type=base \
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
  --task="$TASK_DESCRIPTION" \
  --fps=$FPS \
  --duration=$DURATION \
  --display_data=true \
  --display_compressed_images=false \
  --play_sounds=false 2>&1 | tee "$LOG_FILE"

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ 单臂推理结束（ACT 50k / base 模式）"
echo "=========================================="
echo "本次未保存任何推理数据（base 策略不录制、不落盘）。"
echo "如需保存推理数据，请改用 self_scripts/b601_single/04_dagger_collect.sh"
echo "完整日志：$LOG_FILE"
echo "=========================================="
