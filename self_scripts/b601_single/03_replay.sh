#!/bin/bash
# B601-RS 单臂**数据kfinfer**：把已录制 episode 的 action 序列原样下发给从臂（不推理、不录制）
# 调用：lerobot-replay（lerobot/src/lerobot/scripts/lerobot_replay.py）
#
# 用途：验证某次采集的轨迹在真机上是否物理可复现（数据质量/标定/方向修正的自检手段）。
#   kfinfer会按数据集 fps 逐帧把 action 下发给机器人；使用该数据集时的同一套
#   joint_directions / 端口 / 机器人类型，否则关节方向会反。
#
# 默认kfinfer的数据集：/home/kf/LX/pai0/b601_data/b601_20260915_002947（1 集 / 2909 帧 / 30 Hz）
# 参照：self_scripts/b601_single/02_record.sh（硬件端口/相机/方向修正）、inference/run_inference_ACT.sh
# 创建日期：2026-09-15
#
# ⚠️ 安全：kfinfer下发的是**绝对目标角**（数据集里是主臂/录制坐标系的位置），
#    如果当前位姿与 episode 首帧差得多，机械臂会猛地甩到首帧位置。
#    开始前请把从臂摆到接近首帧姿态（本脚本会打印首帧 state 供对照），并留出运动空间。

set -e  # 遇到错误立即退出
set -o pipefail  # 管道中任一命令失败则整体失败（配合 tee 使用）

# ==================== 环境检查 ====================
REPLAY_CMD="${REPLAY_CMD:-lerobot-replay}"  # 也可用 python -m lerobot.scripts.lerobot_replay
if ! command -v "${REPLAY_CMD%% *}" >/dev/null 2>&1; then
  echo "未找到 $REPLAY_CMD，尝试激活 lerobot conda 环境..."
  for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      source "$base/etc/profile.d/conda.sh"
      conda activate lerobot
      break
    fi
  done
fi
if ! command -v "${REPLAY_CMD%% *}" >/dev/null 2>&1; then
  echo "❌ 错误：$REPLAY_CMD 不可用"
  echo "   请手动执行: conda activate lerobot"
  exit 1
fi
echo "✅ $REPLAY_CMD 可用"
echo ""

echo "=========================================="
echo "B601-RS 数据kfinfer（replay 已录 episode）"
echo "=========================================="
echo ""

# ==================== 数据集配置 ====================
DATASET_ROOT="/home/kf/LX/pai0/b601_data/b601_20260915_141534"
# 本地数据集的 repo_id 只是个标签（已有 root，不会去 Hub 下载）
REPO_ID="local/$(basename "$DATASET_ROOT")"
EPISODE=0   # 要kfinfer的 episode 序号
FPS=30      # 与采集一致（实际以数据集 meta 里的 fps 为准）

# ==================== 硬件配置 ====================
# B601 从臂 = can0（SocketCAN，PEAK PCAN-USB，1Mbps 经典 CAN）
B601_FOLLOWER_PORT="can0"
FOLLOWER_ID="follower"   # 与采集时一致（校准文件 seeed_b601_rs_follower/follower.json）

# 方向修正：必须与采集（self_scripts/b601_single/02_record.sh）完全一致，全 -1.0。
# 原因：seeed_b601_rs_follower.send_action 先对每个关节乘 joint_directions
#       （缺失关节默认 0.0，会把目标裁到 0 不动），kfinfer动作与录制动作必须同坐标系。
B601_JOINT_DIRECTIONS='{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}'

# 重放时是否把 observation.state + 三路相机画面实时送到 rerun 显示。
#   注意：lerobot 自带的 lerobot-replay **没有任何显示选项**（配置只有 robot/dataset/play_sounds），
#   只有 lerobot-rollout 才有 --display_data/--display_mode —— 所以这不是"开关没打开"，
#   而是 replay 这条路径没做可视化。这里用 self_scripts/b601_common/06_replay_rerun.py（复用 lerobot 公共 API：
#   LeRobotDataset + make_robot_from_config + visualization_utils，不改 lerobot 源码）实现。
#   开启后必须同时开相机（没相机就没有画面可显示）。
USE_RERUN="${USE_RERUN:-1}"   # 0 则等价于原生 lerobot-replay（不显示、不读相机）
REPLAY_RERUN="$(dirname "$(readlink -f "$0")")/../b601_common/06_replay_rerun.py"
DISPLAY_MODE="rerun"          # rerun | foxglove
DISPLAY_IP=""                 # 留空=本机 viewer；填 IP 则连远端 rerun --serve
DISPLAY_PORT=""               # 远端端口（配合 DISPLAY_IP）
# rerun 缓冲/内存（与 inference/run_inference_ACT.sh 一致，避免 gRPC transport error 与帧数上限）
export RERUN_FLUSH_NUM_BYTES="${RERUN_FLUSH_NUM_BYTES:-10000000}"
export LEROBOT_RERUN_MEMORY_LIMIT="${LEROBOT_RERUN_MEMORY_LIMIT:-30%}"

# 相机：重放本身不需要图像（processor 是 Identity、目标是绝对角），
# 但开了 rerun 显示就必须开相机，否则没有画面可显示。
USE_CAMERAS=0
if [ "$USE_RERUN" = "1" ]; then USE_CAMERAS=1; fi
CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== kfinfer前检查 ====================
# 1) 数据集
if [ ! -d "$DATASET_ROOT" ]; then
  echo "❌ 数据集路径不存在：$DATASET_ROOT"
  exit 1
fi
[ -f "$DATASET_ROOT/meta/info.json" ] || { echo "❌ 缺少 $DATASET_ROOT/meta/info.json"; exit 1; }
echo "✅ 数据集存在：$DATASET_ROOT"

# 2) 数据集信息 + episode 首帧 state（kfinfer的绝对目标基准，用于对照当前位姿）
python - "$DATASET_ROOT" "$EPISODE" <<'PY' || echo "⚠️  数据集信息读取失败（继续）"
import glob, json, sys
import pandas as pd
root, ep = sys.argv[1], int(sys.argv[2])
info = json.load(open(f"{root}/meta/info.json"))
print(f"   robot_type={info.get('robot_type')}  fps={info.get('fps')}  "
      f"episodes={info.get('total_episodes')}  frames={info.get('total_frames')}")
if ep >= int(info.get("total_episodes") or 0):
    print(f"   ❌ episode {ep} 超出范围（共 {info.get('total_episodes')} 集）")
    sys.exit(1)
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True)):
    df = pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
    d = df[df.episode_index == ep]
    if not len(d):
        continue
    st = d.iloc[0]["observation.state"]
    ac = d.iloc[0]["action"]
    print(f"   episode {ep}: {len(d)} 帧 ≈ {len(d) / float(info.get('fps') or 30):.1f} s")
    print("   首帧 state : " + "  ".join(f"{n}={v:7.1f}" for n, v in zip(names, st)))
    print("   首帧 action: " + "  ".join(f"{n}={v:7.1f}" for n, v in zip(names, ac)))
    break
PY

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

# 4) 从臂校准文件（须与采集同 id）
echo ""
CALIB_FILE="$HOME/.cache/huggingface/lerobot/calibration/robots/seeed_b601_rs_follower/${FOLLOWER_ID}.json"
if [ -f "$CALIB_FILE" ]; then
  echo "✅ $CALIB_FILE"
else
  echo "❌ 缺失校准文件：$CALIB_FILE"
  echo "   请先运行 self_scripts/b601_common/02_follower_calibration.sh 完成从臂校准"
  exit 1
fi

# 5) rerun 显示（重放路径没有自带可视化，用 self_scripts/b601_common/06_replay_rerun.py 实现）
echo ""
if [ "$USE_RERUN" = "1" ]; then
  if [ ! -f "$REPLAY_RERUN" ]; then
    echo "❌ 缺少 $REPLAY_RERUN（重放显示脚本）"; exit 1
  fi
  if ! python -c "import rerun" >/dev/null 2>&1; then
    echo "❌ 未安装 rerun-sdk（conda activate lerobot && pip install 'lerobot[viz]'）"; exit 1
  fi
  echo "✅ rerun 显示已启用（$REPLAY_RERUN，模式 $DISPLAY_MODE${DISPLAY_IP:+ @ $DISPLAY_IP:$DISPLAY_PORT}）"
else
  echo "ℹ️  未启用 rerun 显示（USE_RERUN=0）→ 等价于原生 lerobot-replay，不读相机"
fi

# ==================== 参数总览 ====================
echo ""
echo "=========================================="
echo "kfinfer参数总览"
echo "=========================================="
echo "数据集：$DATASET_ROOT"
echo "repo_id：$REPO_ID（本地 root，不上传/不下载）"
echo "episode：$EPISODE"
echo "机器人：seeed_b601_rs_follower（can0，id=$FOLLOWER_ID）"
echo "相机：$([ "$USE_CAMERAS" = "1" ] && echo "开启 hand/front/top（供 rerun 显示）" || echo "关闭（重放不需要图像）")"
echo "rerun 显示：$([ "$USE_RERUN" = "1" ] && echo "开启（observation.state + hand/front/top 画面）" || echo "关闭")"
echo "方向修正：全 -1.0（与采集一致）"
echo "下发方式：按数据集 fps（30）逐帧下发绝对目标角；不录制、不保存"
echo "结束动作：断开时 safe_zero 回零位"
echo "=========================================="
echo ""
echo "⚠️  kfinfer会把从臂甩到 episode 首帧姿态，请确认："
echo "   1) 从臂周边无人/无障碍，手臂可自由运动；"
echo "   2) 当前姿态与上面打印的首帧 state 接近（差太多会猛地甩过去）；"
echo "   3) 手放在急停/电源开关附近。"
read -p "确认无误，按 ENTER 开始kfinfer，Ctrl+C 取消..." dummy

# ==================== 开始kfinfer ====================
echo ""
echo "🚀 开始kfinfer episode $EPISODE ..."
echo ""

LOG_DIR="$HOME/LX/pai0/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/replay_b601_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件：$LOG_FILE"
echo ""

EXTRA_ARGS=()
[ "$USE_CAMERAS" = "1" ] && EXTRA_ARGS+=(--robot.cameras="$CAMERAS")

# 启动命令：开了 rerun 就走 self_scripts/b601_common/06_replay_rerun.py（原生 lerobot-replay 没有显示选项）
if [ "$USE_RERUN" = "1" ]; then
  RUN_CMD=(python "$REPLAY_RERUN" --display_data=true --display_mode="$DISPLAY_MODE" --display_compressed_images=false)
  [ -n "$DISPLAY_IP" ] && RUN_CMD+=(--display_ip="$DISPLAY_IP" --display_port="${DISPLAY_PORT:-9876}")
else
  RUN_CMD=($REPLAY_CMD)
fi

"${RUN_CMD[@]}" \
  --robot.type=seeed_b601_rs_follower \
  --robot.id=$FOLLOWER_ID \
  --robot.port=$B601_FOLLOWER_PORT \
  --robot.can_adapter=socketcan \
  --robot.gravity_compensation=true \
  --robot.joint_directions="$B601_JOINT_DIRECTIONS" \
  "${EXTRA_ARGS[@]}" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.episode=$EPISODE \
  --dataset.fps=$FPS \
  --play_sounds=false 2>&1 | tee "$LOG_FILE"

# ==================== kfinfer完成 ====================
echo ""
echo "=========================================="
echo "✅ kfinfer结束（episode $EPISODE）"
echo "=========================================="
echo "本次未保存任何数据（replay 只下发 action）。"
echo "完整日志：$LOG_FILE"
echo "=========================================="
