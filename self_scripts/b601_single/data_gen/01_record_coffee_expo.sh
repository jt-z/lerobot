#!/bin/bash
# 模块化咖啡展会技能数据采集（单 B601-RS 从臂 + StarArm102 主臂）
#
# 设计原则：模块独立采集，展会运行时组合。
#   完整任务 = A1 + B_i + Wait + C_j（i∈{1,2} 饮品，j∈{1,2,3} 分区），共 2×3=6 种组合，
#   但只训练 6 个基础技能，用一个共享 π0.5 模型按 prompt 调用。
#
# 六个技能（每集只包含一个模块的完整轨迹，不混搭）：
#   A1 CUP_TO_MACHINE  取空杯放到咖啡机出液托盘
#   B1 MAKE_HOT_WATER  选择热水并启动（只负责启动，不等待出水完成）
#   B2 MAKE_AMERICANO  选择美式并启动（只负责启动，不等待萃取完成）
#   C1 PLACE_AREA_1    把托盘上已装好的杯子放到 1 区
#   C2 PLACE_AREA_2    放到 2 区
#   C3 PLACE_AREA_3    放到 3 区
#
# 数据组织：六个 prompt 全部写进同一份数据集
#   $B601_DATA_ROOT/b601_pi05_coffee_expo_<时间戳>/
#     ├── data/  videos/  meta/tasks.parquet（6 条 task）
#   第一次模块用 create 新建，之后所有模块都用 --resume 往同一目录追加。
#
# ⚠️ prompt 三处必须逐字一致（含大小写和空格）：本脚本、训练数据 meta、推理脚本。
# ⚠️ 所有模块统一以 home pose 开始和结束；每个模块开始前跑零位自检。
#
# 默认采集计划（每个技能 30 集，共 180 集；C1~C3 各 15 集热水来源 + 15 集美式来源）：
#   A1×30 → B1×30 → C1×15 → C2×15 → C3×15（热水来源）
#         → B2×30 → C1×15 → C2×15 → C3×15（美式来源）
#
# 用法：
#   bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh
#       # 完整计划（上面 180 集）；每个模块开录前会停下来等 ENTER
#   bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh --episodes 20
#       # 每个技能 20 集（C 区分两批各 10 集），共 120 集
#   bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh --plan hot_water
#       # 只采热水链路：A1、B1、C1~C3（C 各半批）
#   bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh --plan americano
#       # 只采美式链路：A1、B2、C1~C3（C 各半批）
#   bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh --tasks A1,B1
#       # 只采指定技能（各 EPISODES 集；C 区不拆分来源）
#   bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh --task C1 --episodes 15 --source hot_water
#       # 单技能补录；--source 只用于日志标记，方便回填 15 热水 / 15 美式的配额
#   bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh --resume b601_pi05_coffee_expo_20261118_101500 --task C2 --episodes 15 --source americano
#       # 续录已有数据集（repo_id 必须带完整时间戳）
#
# 按键结束每个 episode（lerobot-record 内置）：
#   n/右方向键=结束本集并保存 | r/左方向键=丢弃当前这一集（不保存）重录 | q/Esc=停止整个录制
# 快捷键（本脚本额外启用，由 b601_common/11_voice_rate_guard.py 注入）：
#   z = 取消跟随主臂 → 从当前位姿平滑插值回零位 home pose（默认 2.5s）
#       → 保持 0.5s → 自动结束并保存本集（等同按 n）
#       本集结束后 reset 阶段从臂停在零位不跟随；等数据写完（含视频编码）后
#       再用 3s（可调）缓慢对齐回主臂当前位姿，然后恢复跟随
#       可用 B601_HOME_KEY / B601_HOME_SECONDS / B601_HOME_HOLD / B601_HOME_REENGAGE 覆盖
#
# 安全须知（与 02_record.sh / 05 / 06 一致）：
#   1. 采集前请确保主从臂已校准（self_scripts/b601_common/02_follower_calibration.sh / 01_leader_calibration.sh）
#   2. 从臂连接后【使能变硬】，移动主臂 -> 从臂跟随（有速度限幅：默认 40°/帧 @30fps ≈ 1200°/s，
#      可用 B601_MAX_RELATIVE_TARGET 覆盖）
#   3. 退出（Ctrl+C / q）时从臂回零位坐姿（夹爪收至夹稳位，不会自动回收纳 0°，
#      下次启动前请手动把夹爪合到 0°，否则零点基准会偏移）
#   4. 紧急停止 = 切断 48V 电源
#   5. 启动前把主臂和从臂都摆到零位（默认坐姿、夹爪闭合）
#   6. 按按钮时用夹爪尖端轻按，不要压在面板上持续用力
#   7. B1/B2 只负责"启动制作"，启动后立即回 home pose 并按 n 结束，不要等出水/萃取完成；
#      等待与取杯由外部调度程序衔接
#
# CAN：采集前自动复位 can0（down→1Mbps→up）并清空残留帧，避免上次异常退出留下的
#      垃圾帧把电机喂住、导致 connect() 报 "control ack timeout: comm_type=4"

set -e
set -o pipefail

# 设备参数层：按主机名自动选 b601_common/devices/<host>.sh（B601_DEVICE 可覆盖）
_B601_ENV="$(dirname "$(readlink -f "$0")")/../../b601_common/_env.sh"
[ -f "$_B601_ENV" ] && . "$_B601_ENV"

# ==================== 任务定义（prompt 为唯一事实源） ====================
# 中文原意仅供参考，不要写进数据集（英文 tokenizer 对中文不友好）。
declare -A TASK_NAME=(
  [A1]="CUP_TO_MACHINE"
  [B1]="MAKE_HOT_WATER"
  [B2]="MAKE_AMERICANO"
  [C1]="PLACE_AREA_1"
  [C2]="PLACE_AREA_2"
  [C3]="PLACE_AREA_3"
)

declare -A TASK_PROMPT=(
  [A1]="Pick up an empty paper cup from the cup dispenser and place it on the silver drip tray under the coffee machine spout. Return to the home pose."
  [B1]="Use the gripper tip to select Hot Water and press the Start button to dispense hot water into the cup using the current preset settings. Return to the home pose."
  [B2]="Use the gripper tip to select Americano and press the Start button to brew coffee into the cup using the current preset settings. Return to the home pose."
  [C1]="Pick up the filled cup from the silver drip tray under the coffee machine spout and place it in cup placement area 1. Return to the home pose."
  [C2]="Pick up the filled cup from the silver drip tray under the coffee machine spout and place it in cup placement area 2. Return to the home pose."
  [C3]="Pick up the filled cup from the silver drip tray under the coffee machine spout and place it in cup placement area 3. Return to the home pose."
)

# 每个模块开录前的环境准备提示（数据采集阶段由人工摆好初态）
task_prep() {
  case "$1" in
    A1)
      echo "  1) 取杯器内放好空杯（一次只放一个可取，重置时补一个）"
      echo "  2) 咖啡机保持待机，出液托盘为空"
      ;;
    B1)
      echo "  1) 把【空杯】放在出液口下方银色托盘中心"
      echo "  2) 咖啡机参数预置为热水，屏幕停在可选择的待机界面"
      ;;
    B2)
      echo "  1) 把【空杯】放在出液口下方银色托盘中心"
      echo "  2) 咖啡机参数预置为美式，屏幕停在可选择的待机界面"
      ;;
    C1|C2|C3)
      echo "  1) 把【已装好的杯子】（热水或美式，按本批来源）放在出液口下方银色托盘中心"
      echo "  2) 区域 ${1#C} 清空，中心位置留出，边界标记可见"
      ;;
  esac
  echo "  3) 主臂、从臂都摆回 home pose（坐姿、夹爪闭合）"
}

# 每个模块的 episode 演示流程提示
task_flow() {
  case "$1" in
    A1) echo "开始录制 → 取空杯 → 放到出液托盘 → 回 home pose → 按 n 保存 → 重置（补空杯、机械臂回起点）" ;;
    B1) echo "开始录制 → 用夹爪尖端选热水 → 按 Start → 立即回 home pose → 按 n 保存 → 重置（换空杯、参数复位、机械臂回起点）" ;;
    B2) echo "开始录制 → 用夹爪尖端选美式 → 按 Start → 立即回 home pose → 按 n 保存 → 重置（换空杯、参数复位、机械臂回起点）" ;;
    C1|C2|C3) echo "开始录制 → 从托盘夹起已装好的杯子 → 放到区域 ${1#C} → 回 home pose → 按 n 保存 → 重置（换一杯到托盘、机械臂回起点）" ;;
  esac
}

# ==================== 命令行参数 ====================
usage() {
  cat <<'EOF'
用法：
  bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh [参数]

参数：
  --plan full|hot_water|americano
                              采集计划，默认 full：
                                full      = A1 + B1 + C(热水半批) + B2 + C(美式半批)
                                hot_water = A1 + B1 + C(热水半批)
                                americano = A1 + B2 + C(美式半批)
                              与 --task/--tasks 同时出现时，以 --task/--tasks 为准
  --task <ID>                 只采一个技能：A1 / B1 / B2 / C1 / C2 / C3
  --tasks A1,B1,...           只采多个指定技能（C 区不拆分来源）
  --episodes N                每个技能的总集数，默认 30；计划模式里 C 区按 N/2 分两批
  --source <标记>             本次采集的杯子来源（hot_water / americano），仅用于日志标记
  --resume <repo_id>          续录已有数据集（需带完整时间戳名）
  --dry-run                   只打印采集计划并退出（不连硬件、不录制）
  -h, --help                  显示本帮助

示例：
  bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh
  bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh --episodes 20
  bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh --plan hot_water
  bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh --tasks A1,B1,B2 --episodes 15
  bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh --resume b601_pi05_coffee_expo_20261118_101500 --task C1 --episodes 15 --source hot_water
EOF
}

PLAN="full"
SINGLE_TASKS=()
NUM_EPISODES_OVERRIDE=""
SOURCE_TAG=""
RESUME_REPO_ID=""
DRY_RUN=false

while [ $# -gt 0 ]; do
  case "$1" in
    --plan)
      [ -n "${2:-}" ] || { echo "❌ --plan 需要参数：full / hot_water / americano"; exit 1; }
      PLAN="$2"; shift 2 ;;
    --task)
      [ -n "${2:-}" ] || { echo "❌ --task 需要技能 ID，例如 --task C1"; exit 1; }
      SINGLE_TASKS=("$2"); shift 2 ;;
    --tasks)
      [ -n "${2:-}" ] || { echo "❌ --tasks 需要技能 ID 列表，例如 --tasks A1,B1"; exit 1; }
      IFS=',' read -r -a SINGLE_TASKS <<< "$2"; shift 2 ;;
    --episodes)
      [ -n "${2:-}" ] || { echo "❌ --episodes 需要正整数"; exit 1; }
      NUM_EPISODES_OVERRIDE="$2"; shift 2 ;;
    --source)
      [ -n "${2:-}" ] || { echo "❌ --source 需要标记，例如 hot_water / americano"; exit 1; }
      SOURCE_TAG="$2"; shift 2 ;;
    --resume)
      [ -n "${2:-}" ] || { echo "❌ --resume 需要数据集完整 repo_id（带时间戳）"; exit 1; }
      RESUME_REPO_ID="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "❌ 未知参数：$1"; echo ""; usage; exit 1 ;;
  esac
done

case "$PLAN" in
  full|hot_water|americano) ;;
  *) echo "❌ --plan 只能是 full / hot_water / americano（收到：$PLAN）"; exit 1 ;;
esac

if [ -n "$NUM_EPISODES_OVERRIDE" ]; then
  if ! [[ "$NUM_EPISODES_OVERRIDE" =~ ^[0-9]+$ ]] || [ "$NUM_EPISODES_OVERRIDE" -lt 1 ]; then
    echo "❌ --episodes 必须是正整数"; exit 1
  fi
fi

for t in "${SINGLE_TASKS[@]}"; do
  [ -n "${TASK_PROMPT[$t]:-}" ] || {
    echo "❌ 未知技能 ID：$t（支持 A1 / B1 / B2 / C1 / C2 / C3）"; exit 1
  }
done

# ==================== 环境检查 ====================
if [ "$DRY_RUN" = false ]; then
  if ! command -v lerobot-record >/dev/null 2>&1; then
    echo "未找到 lerobot-record，尝试激活 lerobot conda 环境..."
    for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/dev/miniforge3" /opt/conda; do
      if [ -f "$base/etc/profile.d/conda.sh" ]; then
        source "$base/etc/profile.d/conda.sh"
        conda activate lerobot
        break
      fi
    done
  fi
  if ! command -v lerobot-record >/dev/null 2>&1; then
    echo "❌ 错误：lerobot-record 不可用"
    echo "   请手动执行: conda activate lerobot"
    exit 1
  fi
  echo "✅ lerobot-record 可用"
fi

echo "=========================================="
echo "模块化咖啡展会技能数据采集（B601 从臂 + StarArm102 主臂）"
echo "=========================================="
echo ""

# ==================== 硬件配置 ====================
FOLLOWER_ID="${B601_FOLLOWER_ID:-follower}"   # 从臂校准 id（对应 seeed_b601_rs_follower/<id>.json）
LEADER_ID="${B601_LEADER_ID:-leader}"         # 主臂校准 id（对应 rebot_102_leader/<id>.json）

# B601 从臂 = can0（SocketCAN）—— 由设备档案决定，可用 B601_FOLLOWER_PORT 覆盖
B601_FOLLOWER_PORT="${B601_FOLLOWER_PORT:-can0}"
# StarArm102 主臂串口：自动探测（候选口逐个 ping 舵机），可用 B601_LEADER_PORT 覆盖
B601_LEADER_PORT="${B601_LEADER_PORT:-}"
# kfzbox 本机适配：优先 udev 稳定别名（CH340 无序列号，ttyUSB* 会变）
if [ -z "$B601_LEADER_PORT" ] && [ -e /dev/rebot_leader ]; then B601_LEADER_PORT=/dev/rebot_leader; fi
LEADER_PORT_PROBE="$(dirname "$(readlink -f "$0")")/../../b601_common/04_leader_port.py"

# 从臂方向修正（必须保留，否则 shoulder_lift/gripper 被限位裁到 0 不动）
# ⚠️ 不能用 "${VAR:-{...}}"：bash 在第一个 '}' 处结束展开，默认值必须走间接变量。
B601_JOINT_DIRECTIONS="${B601_JOINT_DIRECTIONS:-$B601_JOINT_DIRECTIONS_DEFAULT}"

# 启动零位自检脚本；MAX_RELATIVE_TARGET 是从臂单步最大位移（度/帧），
# 30fps 下 40°/帧 ≈ 1200°/s，只兜住"碰歪猛甩"，不干扰正常遥操作。
START_POSE_CHECK="$(dirname "$(readlink -f "$0")")/../../b601_common/10_start_pose_check.py"
MAX_RELATIVE_TARGET="${B601_MAX_RELATIVE_TARGET:-40.0}"
# lerobot-record 包装器：超速时语音提示 "too fast"（每段只响一次）+ z 键一键回零
VOICE_GUARD="$(dirname "$(readlink -f "$0")")/../../b601_common/11_voice_rate_guard.py"
# z 快捷键：取消跟随主臂 -> 平滑插值回零位 home pose -> 保持 0.5s -> 自动结束本集（等同按 n）
# 本集保存后进入 reset 阶段，再用 REENGAGE 秒从零位缓慢对齐回主臂当前位姿
HOME_KEY="${B601_HOME_KEY:-z}"
HOME_SECONDS="${B601_HOME_SECONDS:-2.5}"
HOME_HOLD="${B601_HOME_HOLD:-0.5}"
HOME_REENGAGE="${B601_HOME_REENGAGE:-3.0}"

# 设备档案里 B601_DISPLAY_DATA 是 1/0，draccus 的 bool 参数用 true/false 更稳
DISPLAY_DATA=true
case "${B601_DISPLAY_DATA:-1}" in
  0|false|False) DISPLAY_DATA=false ;;
esac

# ==================== 摄像头配置 ====================
# 与 02_record.sh 一致：角色→序列号来自设备档案；缺失时由 b601_cameras_json 兜底并提示
if [ "$DRY_RUN" = true ]; then
  CAMERAS="{}"
else
  CAMERAS="$(b601_cameras_json)"
fi

# ==================== 数据集配置 ====================
# 数据根目录：与代码仓分开，落在设备档案的 B601_DATA_ROOT（kfzbox=~/b601_data，z790=~/LX/pai0/b601_data）
DATA_ROOT="${B601_DATA_ROOT:-$HOME/b601_data}"
[ "$DRY_RUN" = true ] || mkdir -p "$DATA_ROOT"

# 每个技能的总集数；计划模式下 C1~C3 各拆成 ceil(N/2) + floor(N/2) 两批
EPISODES_PER_TASK="${NUM_EPISODES_OVERRIDE:-30}"

if [ -n "$RESUME_REPO_ID" ]; then
  DATASET_NAME="$RESUME_REPO_ID"
  DATA_DIR="$DATA_ROOT/$DATASET_NAME"
else
  # 新建：脚本生成带时间戳的数据集名，配合 --dataset.no_stamp=true 保证目录名一致
  DATASET_NAME="b601_pi05_coffee_expo_$(date +%Y%m%d_%H%M%S)"
  DATA_DIR="$DATA_ROOT/$DATASET_NAME"
fi

# 是否推送到 Hugging Face Hub（当前环境外网不可达，建议保持 false，数据只存本地）
PUSH_TO_HUB=false
# 国内镜像源（配合上面开关；网络可用时生效）
export HF_ENDPOINT=https://hf-mirror.com

# 每个 episode 录制时长的安全上限（秒）；录制由按键结束，该值只是防无限录制的保险
EPISODE_TIME=600
RESET_TIME=5    # 重置环境时长（秒）；按 n / 右方向键可提前跳过等待
FPS=30

# ==================== 生成采集计划 ====================
PLAN_TASKS=()
PLAN_EPISODES=()
PLAN_SOURCES=()
add_plan() {  # task episodes source
  [ "${2:-0}" -le 0 ] && return 0
  PLAN_TASKS+=("$1")
  PLAN_EPISODES+=("$2")
  PLAN_SOURCES+=("${3:-}")
}

if [ "${#SINGLE_TASKS[@]}" -gt 0 ]; then
  for t in "${SINGLE_TASKS[@]}"; do
    add_plan "$t" "$EPISODES_PER_TASK" "$SOURCE_TAG"
  done
else
  HALF_HOT=$(( EPISODES_PER_TASK / 2 ))
  HALF_AMER=$(( EPISODES_PER_TASK - HALF_HOT ))
  case "$PLAN" in
    full)
      add_plan A1 "$EPISODES_PER_TASK" ""
      add_plan B1 "$EPISODES_PER_TASK" ""
      add_plan C1 "$HALF_HOT" hot_water
      add_plan C2 "$HALF_HOT" hot_water
      add_plan C3 "$HALF_HOT" hot_water
      add_plan B2 "$EPISODES_PER_TASK" ""
      add_plan C1 "$HALF_AMER" americano
      add_plan C2 "$HALF_AMER" americano
      add_plan C3 "$HALF_AMER" americano
      ;;
    hot_water)
      add_plan A1 "$EPISODES_PER_TASK" ""
      add_plan B1 "$EPISODES_PER_TASK" ""
      add_plan C1 "$HALF_HOT" hot_water
      add_plan C2 "$HALF_HOT" hot_water
      add_plan C3 "$HALF_HOT" hot_water
      ;;
    americano)
      add_plan A1 "$EPISODES_PER_TASK" ""
      add_plan B2 "$EPISODES_PER_TASK" ""
      add_plan C1 "$HALF_AMER" americano
      add_plan C2 "$HALF_AMER" americano
      add_plan C3 "$HALF_AMER" americano
      ;;
  esac
fi

PLAN_TOTAL=0
for n in "${PLAN_EPISODES[@]}"; do PLAN_TOTAL=$((PLAN_TOTAL + n)); done

# task ID → 本计划总集数（用于完成度统计）
declare -A PLANNED_PER_TASK=()
for i in "${!PLAN_TASKS[@]}"; do
  t="${PLAN_TASKS[$i]}"
  PLANNED_PER_TASK[$t]=$(( ${PLANNED_PER_TASK[$t]:-0} + PLAN_EPISODES[$i] ))
done

# --dry-run：只打印计划（不连硬件、不录制）
if [ "$DRY_RUN" = true ]; then
  echo ""
  echo "=========================================="
  echo "采集计划（dry-run，不连接硬件）"
  echo "=========================================="
  echo "数据集名称：$DATASET_NAME"
  echo "本地保存路径：$DATA_DIR"
  echo "本次计划总集数：$PLAN_TOTAL"
  echo ""
  for i in "${!PLAN_TASKS[@]}"; do
    t="${PLAN_TASKS[$i]}"
    src="${PLAN_SOURCES[$i]}"
    echo "  $t ${TASK_NAME[$t]}：${PLAN_EPISODES[$i]} 集${src:+（来源：$src）}"
    echo "    prompt: ${TASK_PROMPT[$t]}"
  done
  echo "=========================================="
  exit 0
fi

# ==================== 采集前检查 ====================
echo "1. 检查硬件..."
# B601 CAN：启动前一律复位接口并清空残留帧（与 01_teleop.sh / 02_record.sh 一致）
can_drain() {
  local port="$1"
  command -v candump >/dev/null 2>&1 || return 0
  timeout 0.3 candump "$port" >/dev/null 2>&1 || true
}

if ! ip link show "$B601_FOLLOWER_PORT" >/dev/null 2>&1; then
  echo "❌ $B601_FOLLOWER_PORT 不存在"; exit 1
fi

echo "♻️  复位 $B601_FOLLOWER_PORT（1Mbps）并清空残留 CAN 帧..."
sudo ip link set "$B601_FOLLOWER_PORT" down 2>/dev/null || true
sudo ip link set "$B601_FOLLOWER_PORT" type can bitrate 1000000 || {
  echo "❌ 配置 $B601_FOLLOWER_PORT 失败（是否被其它进程占用？）"; exit 1
}
sudo ip link set "$B601_FOLLOWER_PORT" up
sleep 0.5   # 给电机侧看门狗留时间丢掉上一条残留指令
can_drain "$B601_FOLLOWER_PORT"

if ! ip link show "$B601_FOLLOWER_PORT" 2>/dev/null | grep -q "state UP"; then
  echo "❌ $B601_FOLLOWER_PORT 未 UP —— 检查 PCAN-USB 是否插好、peak_usb 驱动是否加载"
  ip -details link show "$B601_FOLLOWER_PORT" 2>/dev/null | sed -n '1,6p'
  exit 1
fi
if ! ip -details link show "$B601_FOLLOWER_PORT" 2>/dev/null | grep -q "bitrate 1000000"; then
  echo "⚠️  $B601_FOLLOWER_PORT 的 bitrate 不是 1000000，请确认总线速率"
fi

# 主臂串口（自动探测：优先 by-id 稳定路径，且必须真正 ping 到舵机才算数）
if [ -z "$B601_LEADER_PORT" ]; then
  if [ -f "$LEADER_PORT_PROBE" ]; then
    echo "🔍 探测主臂串口（候选口逐个 ping 舵机 id 0~6）..."
    B601_LEADER_PORT="$(python "$LEADER_PORT_PROBE" || true)"
  else
    echo "⚠️  未找到探测脚本 $LEADER_PORT_PROBE，退回按设备名挑"
    for cand in /dev/serial/by-id/usb-1a86_USB_Serial*-if00-port0 /dev/ttyUSB* /dev/ttyACM*; do
      [ -e "$cand" ] && B601_LEADER_PORT="$cand" && break
    done
  fi
fi
if [ -z "$B601_LEADER_PORT" ] || [ ! -e "$B601_LEADER_PORT" ]; then
  echo "❌ 未找到主臂串口：现有候选串口都没有 StarArm102 舵机应答"
  echo "   当前串口设备：$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | tr '\n' ' ')"
  echo "   排查："
  echo "   1) 主臂 USB 线是否插好：journalctl -k | grep -E 'ch341|cdc_acm' | tail -3"
  echo "   2) 主臂舵机是否**上电**（12V 电源/开关）—— 舵机不上电时任何串口都探测不到"
  echo "   3) 逐个口看探测结果：python $LEADER_PORT_PROBE --list"
  echo "   4) 手动指定：B601_LEADER_PORT=/dev/ttyACM0 bash $0"
  exit 1
fi
[ -r "$B601_LEADER_PORT" ] || sudo chmod 666 "$B601_LEADER_PORT"
echo "✅ 硬件就绪（从臂 $B601_FOLLOWER_PORT 已复位并清空残留帧；主臂串口：$B601_LEADER_PORT）"

# 摄像头检查
echo ""
echo "2. 检查摄像头..."
for cam in $(echo "$CAMERAS" | grep -o 'index_or_path: [^,]*' | awk '{print $2}'); do
  [ -e "$cam" ] && echo "  ✅ $cam" || echo "  ❌ 摄像头不存在: $cam"
done

# 校准文件检查
echo ""
echo "3. 检查校准文件..."
CALIB_OK=true
for f in \
  "$HOME/.cache/huggingface/lerobot/calibration/robots/seeed_b601_rs_follower/${FOLLOWER_ID}.json" \
  "$HOME/.cache/huggingface/lerobot/calibration/teleoperators/rebot_102_leader/${LEADER_ID}.json"; do
  if [ -f "$f" ]; then
    echo "  ✅ $f"
  else
    echo "  ❌ 缺失：$f（请先运行 self_scripts/b601_common/02_follower_calibration.sh / 01_leader_calibration.sh）"
    CALIB_OK=false
  fi
done
if [ "$CALIB_OK" = false ]; then
  exit 1
fi

# 辅助脚本检查
echo ""
echo "4. 检查采集辅助脚本..."
for f in "$START_POSE_CHECK" "$VOICE_GUARD"; do
  if [ -f "$f" ]; then
    echo "  ✅ $f"
  else
    echo "  ❌ 缺失：$f"
    exit 1
  fi
done

# ==================== 数据集保存位置检查 ====================
echo ""
echo "5. 检查数据集保存位置..."
if [ -n "$RESUME_REPO_ID" ]; then
  if [ -d "$DATA_DIR/meta" ]; then
    EXISTING_EPISODES=$(python3 -c "import json; print(json.load(open('$DATA_DIR/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
    echo "✅ 续录目录已找到：$DATA_DIR"
    echo "   已有 $EXISTING_EPISODES 个 episode，本次计划追加 $PLAN_TOTAL 个"
  else
    echo "❌ 错误：续录目录不存在或缺少 meta/：$DATA_DIR"
    echo ""
    echo "   可用会话："
    ls -d "$DATA_ROOT"/b601_pi05_coffee_expo_* 2>/dev/null | sed 's|.*/||' | sort
    exit 1
  fi
else
  echo "✅ 数据根目录：$DATA_ROOT"
  echo "   本次新建数据集：$DATA_DIR"
fi

# ==================== 采集参数总览 ====================
echo ""
echo "=========================================="
echo "采集参数总览"
echo "=========================================="
echo "录制模式：$( [ -n "$RESUME_REPO_ID" ] && echo "续录（--resume）" || echo "新建数据集" )"
echo "数据集名称：$DATASET_NAME"
echo "本地保存路径：$DATA_DIR"
echo "计划：$PLAN（若指定了 --task/--tasks 则以指定技能为准）"
echo "本次计划总集数：$PLAN_TOTAL"
echo ""
for i in "${!PLAN_TASKS[@]}"; do
  t="${PLAN_TASKS[$i]}"
  src="${PLAN_SOURCES[$i]}"
  echo "  $t ${TASK_NAME[$t]}：${PLAN_EPISODES[$i]} 集${src:+（来源：$src）}"
  echo "    prompt: ${TASK_PROMPT[$t]}"
done
echo ""
echo "每个 Episode 时长：由按键结束（安全上限 ${EPISODE_TIME}秒）"
echo "重置时长：${RESET_TIME}秒（可按键提前跳过）"
echo "采集频率：${FPS} Hz"
echo "=========================================="
echo ""
echo -e "📝 按键控制（录制过程中随时可用，注意焦点保持在终端窗口）："
echo -e "   n / 右方向键 = 当前 episode 演示完成，结束录制并保存"
echo -e "   r / 左方向键 = 丢弃当前这一集（不保存）并重录"
echo -e "   q / Esc      = 停止整个录制（后续模块不再录）"
echo -e "   $HOME_KEY            = 取消跟随主臂，自动插值回零位 home pose，${HOME_HOLD}s 后自动结束并保存本集"
echo -e "                  （reset 阶段停在零位；本集数据写完后再用 ${HOME_REENGAGE}s 缓慢对齐回主臂）"
echo -e "   ⚠️ 每切换一个技能都会重启一次 lerobot-record（数据集用 --resume 续写），"
echo -e "      每次都要重新连接机械臂，属正常现象。"
echo ""

# ==================== 开始采集 ====================
# Rerun 缓冲/内存（解决 gRPC transport error 与 1000 帧限制）
export RERUN_FLUSH_NUM_BYTES=10000000
export LEROBOT_RERUN_MEMORY_LIMIT="30%"

LOG_DIR="$(dirname "$(readlink -f "$0")")/../../_logs"
mkdir -p "$LOG_DIR"

# 忽略 SIGPIPE：防止 Ctrl+C 时 tee 先退出导致管道破裂、lerobot-record 被
# SIGPIPE(141) 杀死而跳过断开清理（否则电机保持使能、从臂不释放）。
trap '' PIPE

total_episodes() {
  if [ -f "$DATA_DIR/meta/info.json" ]; then
    python3 -c "import json; print(json.load(open('$DATA_DIR/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

# 按 task 统计数据集内 episode 数。
# 新版 lerobot 把 episodes 元数据写成 meta/episodes/chunk-*/file-*.parquet（含 tasks 列），
# 老版本是 meta/episodes.jsonl；两种格式都兼容。
count_by_task() {
  python - "$DATA_DIR" <<'PY'
import collections, glob, json, os, sys

root = sys.argv[1]
counts = collections.Counter()

legacy = os.path.join(root, "meta", "episodes.jsonl")
if os.path.isfile(legacy) and os.path.getsize(legacy) > 0:
    with open(legacy, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                for t in json.loads(line).get("tasks", []) or []:
                    counts[t] += 1
else:
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "file-*.parquet")))
    try:
        import pyarrow.parquet as pq
        for path in files:
            for tasks in pq.read_table(path, columns=["tasks"]).column("tasks").to_pylist():
                for t in tasks or []:
                    counts[t] += 1
    except ImportError:
        import pandas as pd
        for path in files:
            for tasks in pd.read_parquet(path, columns=["tasks"])["tasks"]:
                for t in tasks or []:
                    counts[t] += 1

for t, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"  {n:4d}  {t}")
PY
}

declare -A SAVED_PER_TASK=()
TOTAL_SAVED=0

for idx in "${!PLAN_TASKS[@]}"; do
  task="${PLAN_TASKS[$idx]}"
  want="${PLAN_EPISODES[$idx]}"
  src="${PLAN_SOURCES[$idx]}"
  prompt="${TASK_PROMPT[$task]}"

  echo ""
  echo "=========================================="
  echo "▶ 模块 $task ${TASK_NAME[$task]}（目标 $want 集${src:+，来源 $src}）"
  echo "  prompt: $prompt"
  echo "=========================================="
  echo "准备环境："
  task_prep "$task"
  echo "每集流程："
  task_flow "$task"
  echo ""
  read -p "准备完成后按 ENTER 开始零位自检与录制（Ctrl+C 取消）..." dummy

  # 每个模块开录前做零位自检：主臂只读、只检测；从臂慢速归零、夹爪慢速合到 0°。
  # 自检通过后从臂停在零位，紧接着启动 lerobot-record，首帧不会跳变。
  echo ""
  echo "零位自检（从臂会自动慢速归零，请确认夹爪内没有物体、周围无障碍）..."
  set +e
  python "$START_POSE_CHECK" \
    --leader-port "$B601_LEADER_PORT" \
    --leader-id "$LEADER_ID" \
    --follower-port "$B601_FOLLOWER_PORT" \
    --follower-id "$FOLLOWER_ID" \
    --joint-directions "$B601_JOINT_DIRECTIONS"
  CHECK_RC=$?
  set -e
  if [ "$CHECK_RC" -ne 0 ]; then
    echo ""
    echo "❌ 零位自检未通过（exit=$CHECK_RC），已中止采集。"
    exit "$CHECK_RC"
  fi

  BEFORE_EPISODES=$(total_episodes)

  LOG_FILE="$LOG_DIR/coffee_expo_${task}${src:+_$src}_$(date +%Y%m%d_%H%M%S).log"
  echo "📝 日志：$LOG_FILE"
  echo ""

  RECORD_CMD="python \"$VOICE_GUARD\" \
    --home-key=$HOME_KEY \
    --home-seconds=$HOME_SECONDS \
    --home-hold=$HOME_HOLD \
    --reengage-seconds=$HOME_REENGAGE \
    -- \
    --robot.type=seeed_b601_rs_follower \
    --robot.id=$FOLLOWER_ID \
    --robot.port=$B601_FOLLOWER_PORT \
    --robot.can_adapter=socketcan \
    --robot.gravity_compensation=${B601_GRAVITY_COMPENSATION:-true} \
    --robot.max_relative_target=$MAX_RELATIVE_TARGET \
    --robot.joint_directions=\"$B601_JOINT_DIRECTIONS\" \
    --robot.cameras=\"$CAMERAS\" \
    --teleop.type=rebot_102_leader \
    --teleop.id=$LEADER_ID \
    --teleop.port=$B601_LEADER_PORT \
    --dataset.repo_id=$DATASET_NAME \
    --dataset.num_episodes=$want \
    --dataset.single_task=\"$prompt\" \
    --dataset.fps=$FPS \
    --dataset.episode_time_s=$EPISODE_TIME \
    --dataset.reset_time_s=$RESET_TIME \
    --dataset.video=true \
    --dataset.rgb_encoder.vcodec=h264 \
    --dataset.push_to_hub=$PUSH_TO_HUB \
    --display_data=$DISPLAY_DATA \
    --display_compressed_images=false"

  # 第一次 create 新数据集；之后（以及 --resume）都续写到同一目录。
  # root 被当作数据集目录本身，新建时必须传完整带时间戳路径并用 --no_stamp=true
  # 阻止框架再追加一次时间戳。
  if [ -f "$DATA_DIR/meta/info.json" ]; then
    RECORD_CMD="$RECORD_CMD --resume=true --dataset.root=$DATA_DIR"
  else
    RECORD_CMD="$RECORD_CMD --dataset.no_stamp=true --dataset.root=$DATA_DIR"
  fi

  # 非零退出时提示后继续统计，不直接中断（断开机械臂的清理错误很常见，数据通常没问题）
  set +e
  eval $RECORD_CMD 2>&1 | tee "$LOG_FILE"
  RECORD_EXIT=${PIPESTATUS[0]}
  set -e
  if [ $RECORD_EXIT -ne 0 ]; then
    echo ""
    echo "⚠️  lerobot-record 非零退出（exit=$RECORD_EXIT）"
    echo "    若日志显示 episode 已保存/视频已编码，通常是断开机械臂时的清理错误，数据不受影响。"
    echo "    机械臂可能需要断电重启以复位过载保护。"
  fi

  if [ ! -f "$DATA_DIR/meta/info.json" ]; then
    echo ""
    echo "❌ 数据集没有生成成功（$DATA_DIR/meta/info.json 缺失），中止后续模块，请查看日志："
    echo "   $LOG_FILE"
    exit 1
  fi

  AFTER_EPISODES=$(total_episodes)
  SAVED=$((AFTER_EPISODES - BEFORE_EPISODES))
  [ "$SAVED" -lt 0 ] && SAVED=0
  SAVED_PER_TASK[$task]=$(( ${SAVED_PER_TASK[$task]:-0} + SAVED ))
  TOTAL_SAVED=$((TOTAL_SAVED + SAVED))

  echo ""
  echo "✅ 模块 $task 本次保存 $SAVED 集（本次该技能累计 ${SAVED_PER_TASK[$task]} 集；数据集累计 $AFTER_EPISODES 集）"

  if [ "$SAVED" -lt "$want" ]; then
    echo "⚠️  本模块只保存了 $SAVED/$want 集（提前结束：按了 q / Esc 或录制中断），停止后续模块。"
    echo "    补录命令：bash self_scripts/b601_single/data_gen/01_record_coffee_expo.sh --resume $DATASET_NAME --task $task --episodes $((want - SAVED))${src:+ --source $src}"
    break
  fi
done

# ==================== 采集完成 ====================
echo ""
echo "=========================================="
echo "✅ 模块化咖啡展会技能数据采集结束"
echo "=========================================="
echo "数据集名称：$DATASET_NAME"
echo "本地保存路径：$DATA_DIR"
echo "本次保存：$TOTAL_SAVED 集（计划 $PLAN_TOTAL 集）"
echo ""
echo "按技能统计（本次运行）："
for t in A1 B1 B2 C1 C2 C3; do
  [ -n "${PLANNED_PER_TASK[$t]:-}" ] || continue
  printf "  %s %-15s 本次 %s 集 / 计划 %s 集\n" "$t" "${TASK_NAME[$t]}" "${SAVED_PER_TASK[$t]:-0}" "${PLANNED_PER_TASK[$t]}"
done

TOTAL_EPISODES=$(total_episodes)
echo ""
echo "数据集累计 episode：$TOTAL_EPISODES"
echo ""
echo "数据集内按 prompt 统计："
count_by_task

if [ "$PUSH_TO_HUB" = true ]; then
  echo ""
  echo "Hugging Face Hub 链接："
  echo "  https://huggingface.co/datasets/$DATASET_NAME"
else
  echo ""
  echo "上传状态：未推送到 Hub（PUSH_TO_HUB=false，数据仅保存在本地）"
fi
echo ""
echo "💡 提示："
echo "  1. 六个技能的 prompt 已写进同一份数据集的 meta/tasks.parquet；"
echo "     训练/推理时使用的 prompt 必须与脚本里逐字一致（含大小写和空格）。"
echo "  2. C1~C3 的 30 集由 15 集热水来源 + 15 集美式来源组成；"
echo "     若需补配额，用 --task C<N> --source hot_water|americano 续录。"
echo "  3. 展会运行时的组合方式：A1 + B_i + 外部等待 + C_j，共 2×3=6 种完整任务。"
echo "=========================================="
