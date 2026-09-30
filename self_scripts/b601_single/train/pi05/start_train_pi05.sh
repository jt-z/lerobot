#!/bin/bash
# 启动训练：单臂 seeed_b601_rs_follower 的 Pi0.5 LoRA 微调（make coffee 任务）
#
# 参照
# ----
#   1. so101_bimanual/train/start_train_pi05.sh + configs/pi05_train_config_20000steps.json
#      —— pi05 的 LoRA 写法（peft.LORA / freeze_vision_encoder / QUANTILES 归一化 /
#         input_features 用 base_0_rgb + left/right_wrist / rename_map 把数据集相机改名）
#   2. b601_single/train/smolvla/start_train_smolvla_v2.sh
#      —— 单臂 b601 的训练编排（数据完整性前置校验、传输静默期、按 pass 反算 steps、
#         按 UUID 选卡、CUDA 探针、--resume、训练后复查）
#
# 与 SmolVLA v2 的关系
# --------------------
# 数据换成 b601_20260910_164106_20260930（381 ep / 946,758 帧，比 v2 的 334 ep 更大），
# 策略换成 pi05（LoRA，只训适配器）。steps 同样**不写死**，按 NUM_PASSES 反算：
#
#   有效 batch = batch_size(8) × 卡数
#   8 卡（8×8=64）→ 2 pass × 946,758 ÷ 64 = 29,586 → 30,000 步
#   7 卡（8×7=56）→ 33,813 → 34,000 步  ← GPU0 掉总线时走这条
#
# NUM_PASSES 默认 2 的理由：pi05 是 3B 强预训练底座，旧仓 README_pi05.md 推荐微调区间为
# 「大数据集/复杂任务 10000~20000 步」，2 pass ≈ 30K 步已略高于该区间；而 SmolVLA v2 的
# 12 pass 是针对从头较充分微调的小模型，照搬到 pi05 上会远超推荐量、白烧时间。
#
# 两种给法，STEPS 优先于 NUM_PASSES：
#   STEPS=150000    直接指定绝对步数 —— 「只关心这段时间能跑多少」的场景（如 7 天假期）用它
#   NUM_PASSES=10   按数据集规模反算 —— 换数据集时不用重新拍步数
#
# ⚠️ scheduler_decay_steps 必须恒等于 steps
# ------------------------------------------
# pi05 默认 scheduler_decay_steps=30000。若 steps=34000 而不改它，余弦在 30K 步就走完，
# 剩下的步数 lr 会一直停在 decay_lr；反之 steps 小于默认值时 lerobot 会自动缩放，
# 但**变长不管**。所以脚本把推导出的 steps 同时传给 --steps 和
# --policy.scheduler_decay_steps，两者不可能再走散。
#
# 权重来源
# --------
# 用 --policy.pretrained_path 加载 lerobot/pi05_base（本地 modelscope 快照），
# 而不是 --policy.path。区别：pretrained_path 只加载权重，policy 的输入特征
# （本 config 显式声明的 base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb + 32 维 state）
# 按本 config 走；且 rename_map 只在有 pretrained_path 时生效（train.py validate 检查）。
#
# 相机映射（数据集 hand/front/top → pi05 base 期望的三个 key）
# ----------------------------------------------------------
#   front → observation.images.base_0_rgb       （主视角）
#   hand  → observation.images.left_wrist_0_rgb （腕部）
#   top   → observation.images.right_wrist_0_rgb（顶部）
# 见 config 的 rename_map；训练时 preprocessor 会用它改批次键与归一化统计键。
#
# 前置检查（任一不过就拒绝开训）
# ------------------------------
#   1. 数据集完整性  总行数 == meta total_frames（+ index 无重复 + episode 边界 + 视频不截断）
#   2. 传输静默期    数据集在 QUIET_MINS 分钟内没被写过
# 详见 smolvla/datastet_notes/DISTILLED.md：这个损坏训练时不报错，会**静默学错**，
# 所以不能靠「训练没崩」判断数据正常。
#
# 训练后复查
# ----------
# 钩子的守卫 3 是「有训练在跑就跳过」→ 训练期间源端若重传，数据会被静默改坏而没人修。
# 所以跑完会再数一遍总行数，对不上就明确报出来。
#
# 用法
# ----
#   bash self_scripts/b601_single/train/pi05/start_train_pi05.sh
#
#   ... --resume              # 从本配置自己的 last checkpoint 续训（沿用 ckpt 里的 steps）
#   ... --skip-data-check     # 跳过前置校验（不推荐）
#   ... --dry-run             # 只跑前置检查 + 打印将执行的命令
#
# 可调环境变量
#   STEPS        直接指定训练步数（优先于 NUM_PASSES）。例：STEPS=150000
#                （150k 步 ≈ 6.2 天，正好装进 7 天假期；200k 需 8.3 天，装不下）
#   NUM_PASSES   目标训练量（默认 2），steps = NUM_PASSES × total_frames ÷ (batch_size × 卡数)
#   GPU_LIST     用哪些卡（默认 auto）。auto = 用 nvidia-smi 能查到的全部卡（按 UUID）；
#                掉 PCIe 总线的卡会被自动排除；也可写死 "1,2,3,4,5,6,7" 或 UUID 列表
#   DS_ROOT      覆盖数据集根目录（默认从配置里读）
#   QUIET_MINS   传输静默期分钟数（默认 5）
#   PI05_BASE    预训练权重路径
#   VERIFY_SCRIPT 数据校验脚本路径（默认借用 smolvla/bench/verify_dataset.py）
#
# 退出码：0=训练正常结束  1=数据校验未通过  2=传输中/被守卫拦截  3=环境或用法错误

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
SELF_SCRIPTS_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
CONFIG="$SCRIPT_DIR/pi05_train_config.json"
VERIFY=${VERIFY_SCRIPT:-"$SCRIPT_DIR/../smolvla/bench/verify_dataset.py"}

PI05_BASE=${PI05_BASE:-/home/ksa/.cache/modelscope/models/lerobot--pi05_base/snapshots/master}
QUIET_MINS=${QUIET_MINS:-5}
NUM_PASSES=${NUM_PASSES:-2}
STEPS_REQUEST=${STEPS:-}     # 用户显式指定的绝对步数；空 = 回落到 NUM_PASSES 推导
GPU_LIST=${GPU_LIST:-auto}
RESUME=0; SKIP_CHECK=0; DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --resume)          RESUME=1; shift ;;
        --skip-data-check) SKIP_CHECK=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         sed -n '2,82p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数: $1（-h 看用法）" >&2; exit 3 ;;
    esac
done

die() { echo "✗ $*" >&2; exit "${EXIT_CODE:-3}"; }
step() { echo; echo "── $* ──"; }

# ---------------- 环境 ----------------
[ -f "$CONFIG" ] || die "找不到配置：$CONFIG"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "lerobot" ]; then
    echo "激活 lerobot conda 环境…"
    eval "$(conda shell.bash hook)" || die "conda 不可用"
    conda activate lerobot || die "conda activate lerobot 失败"
fi
PY=$(which python) || die "找不到 python"

# 使用 HF 国内镜像，避免 huggingface.co 网络不可达
export HF_ENDPOINT=https://hf-mirror.com

# output_dir 在配置里是相对路径（相对 self_scripts/），所以必须在这里跑
cd "$SELF_SCRIPTS_DIR" || die "cd $SELF_SCRIPTS_DIR 失败"

# ---------------- GPU ----------------
# 用 UUID 而不是序号来选卡：GPU0（0000:55:00.0）掉过 PCIe 总线（lspci 显示 rev ff），
# 这种情况下 CUDA 的序号可能整体前移（7 张卡是 0~6 而不是 1~7），写死序号会少用一张。
# 注意别用 `nvidia-smi -i <UUID>`：解析 UUID 要枚举全部卡，一撞到故障卡就报
# "Problem with GPU 0000:55:00.0: Unknown Error"，连 -i 都失效。不带 -i 的查询会把故障卡
# 报到 stderr、正常列出其余卡，所以用它取「可用卡集合」。
step "检查可用 GPU"
if [ "$GPU_LIST" = "auto" ]; then
    # 掉总线的卡 nvidia-smi 查不到，自然被排除
    GPU_LIST=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -)
fi
[ -n "$GPU_LIST" ] || die "nvidia-smi 没列出任何可用 GPU（驱动挂了？先复位故障卡）"
NPROC=$(awk -F, '{print NF}' <<<"$GPU_LIST")

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    /'
echo "  选中 $NPROC 张 → CUDA_VISIBLE_DEVICES=$GPU_LIST"

# CUDA 栈是否真的能用。一张卡掉 PCIe 总线会让 cuInit 直接失败，剩下健康的卡也一张都用不了；
# 所以在开训前先问一次驱动，避免白等。
CUDA_PROBE=$(CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PY" - <<'PY'
import ctypes
cu = ctypes.CDLL("libcuda.so.1")
n = ctypes.c_int(0)
rc = cu.cuInit(0)
if rc == 0:
    cu.cuDeviceGetCount(ctypes.byref(n))
print(rc, n.value)
PY
)
CUDA_RC=${CUDA_PROBE%% *}; CUDA_N=${CUDA_PROBE##* }
if [ "$CUDA_RC" != "0" ]; then
    if [ "$DRY_RUN" = 1 ]; then
        # dry-run 只提示不拦，好让 --dry-run 仍可用来核对启动计划
        echo "  ⚠ cuInit 失败（rc=$CUDA_RC）：真实开训会在这里被拦下，先复位故障卡"
    else
        EXIT_CODE=3 die "CUDA 驱动初始化失败（cuInit rc=$CUDA_RC）。一张卡掉总线会把整个 CUDA 栈拖死，
   此时健康卡也用不了，训练会在几分钟内就崩。请先复位故障卡：
     lspci | grep -i nvidia | grep 'rev ff'                        # 找出故障卡
     echo 1 | sudo tee /sys/bus/pci/devices/<PCI地址>/remove        # 从总线摘掉
     sudo nvidia-smi -r -i 0                                        # 或复位
     冷重启（关机断电再开）                                          # 最可靠"
    fi
else
    # 驱动 API 枚举是否受 CUDA_VISIBLE_DEVICES 过滤因版本而异，所以只提示不拦
    [ "$CUDA_N" = "$NPROC" ] || echo "  ⚠ CUDA 报 $CUDA_N 张、GPU_LIST 声明 $NPROC 张（序号可能因缺卡前移；确卡不对就用 GPU_LIST=auto）"
    echo "  ✓ CUDA 可用：$CUDA_N 张"
fi

# ---------------- 预训练权重 ----------------
step "检查预训练权重"
[ -d "$PI05_BASE" ] || die "找不到 pi05_base：$PI05_BASE
   （可用 PI05_BASE=/path/to/pi05_base 指定）"
[ -f "$PI05_BASE/model.safetensors" ] || die "$PI05_BASE 里没有 model.safetensors"
[ -f "$PI05_BASE/config.json" ] || die "$PI05_BASE 里没有 config.json"
echo "  ✓ $PI05_BASE"

# ---------------- 前置检查 1：数据集完整性 ----------------
DS_ROOT=${DS_ROOT:-$("$PY" -c "import json;print(json.load(open('$CONFIG'))['dataset']['root'])")}
[ -n "$DS_ROOT" ] || die "无法从配置里解析 dataset.root"
[ -d "$DS_ROOT" ] || die "数据集目录不存在：$DS_ROOT"
INFO_JSON="$DS_ROOT/meta/info.json"
[ -f "$INFO_JSON" ] || die "找不到 $INFO_JSON（不是 LeRobot 数据集目录？）"

ROWS_BEFORE=""
if [ "$SKIP_CHECK" = 1 ]; then
    echo "⚠ 已跳过数据校验（--skip-data-check）——数据若损坏将静默学错，风险自负"
else
    step "前置检查 1/2：数据集完整性"
    [ -f "$VERIFY" ] || die "找不到校验脚本：$VERIFY
   （可用 VERIFY_SCRIPT=/path/to/verify_dataset.py 指定）"

    VJSON=$(mktemp) || die "mktemp 失败"
    if ! "$PY" "$VERIFY" --root "$DS_ROOT" --json >"$VJSON" 2>&1; then
        echo "✗ 数据集校验未通过，拒绝开训。详细报告：" >&2
        echo >&2
        "$PY" "$VERIFY" --root "$DS_ROOT" >&2 || true
        echo >&2
        echo "修复：$PY $VERIFY --root $DS_ROOT --fix" >&2
        rm -f "$VJSON"
        EXIT_CODE=1 die "前置校验失败"
    fi

    REPORT=$("$PY" - "$VJSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  总行数        : {d['total_rows']:,}")
print(f"  meta 应有行数 : {d['meta_total_frames']:,}")
print(f"  数据文件      : {d['n_files']} 个")
print(f"  重复行        : {d['n_bogus_rows']}  ← 0 才算正常")
print(f"  错位行        : {d['n_misaligned_rows']}")
print("  ✓ 全部不变量成立")
print(d["total_rows"])
PY
)
    ROWS_BEFORE=$(printf '%s\n' "$REPORT" | tail -n 1)
    printf '%s\n' "$REPORT" | sed '$d'
    rm -f "$VJSON"
fi

# ---------------- 前置检查 2：传输静默期 ----------------
step "前置检查 2/2：传输静默期（${QUIET_MINS} 分钟）"
if pgrep -x rsync >/dev/null 2>&1 || pgrep -x scp >/dev/null 2>&1; then
    EXIT_CODE=2 die "检测到 rsync/scp 正在运行，传输可能未结束。稍后重试，或 QUIET_MINS=0 强开"
fi
recent=$(find "$DS_ROOT" -newermt "-${QUIET_MINS} minutes" -type f 2>/dev/null | head -1)
if [ -n "$recent" ]; then
    EXIT_CODE=2 die "数据集在 ${QUIET_MINS} 分钟内有写入（$(basename "$recent")），传输可能未结束。
   等一会儿再跑，或 QUIET_MINS=0 跳过本检查。"
fi
echo "  ✓ ${QUIET_MINS} 分钟内无写入"

# ---------------- 训练量：STEPS 绝对步数 / NUM_PASSES 按数据规模反算 ----------------
# 两条路都只为「定下 steps」，然后把同一值喂给 --steps 与 --policy.scheduler_decay_steps，
# 这样不可能再出现「steps 改了、decay_steps 没改」把后半程 lr 冻住的事故。
if [ -n "$STEPS_REQUEST" ]; then
    step "训练量：指定绝对步数 STEPS=$STEPS_REQUEST"
else
    step "训练量推导（${NUM_PASSES} pass × 数据集规模）"
fi
DERIVE=$("$PY" - "$INFO_JSON" "$CONFIG" "$NPROC" "$NUM_PASSES" "$STEPS_REQUEST" <<'PY'
import json, shutil, sys

info = json.load(open(sys.argv[1]))
cfg = json.load(open(sys.argv[2]))
nproc, passes = int(sys.argv[3]), float(sys.argv[4])
target = float(sys.argv[5]) if sys.argv[5].strip() else 0.0

episodes, frames = int(info["total_episodes"]), int(info["total_frames"])
bs, save_freq = int(cfg["batch_size"]), int(cfg["save_freq"])
sched = cfg["policy"]

if target > 0:
    steps = int(round(target / 1000.0)) * 1000
    origin = f"STEPS={target:,.0f} 指定"
else:
    raw = passes * frames / (bs * nproc)
    steps = int(round(raw / 1000.0)) * 1000
    origin = f"{passes:g} pass × {frames:,} ÷ {bs * nproc} = {raw:,.0f} 步"

n_ckpt = steps // save_freq + 1
need_gb = n_ckpt * 0.02  # 单个 pi05 LoRA ckpt 实测约 16 MB（只存适配器 + 优化器状态）
free_gb = shutil.disk_usage(".").free / 1e9
hours = steps * 3.6 / 3600

print(f"  数据集        : {episodes:,} episodes / {frames:,} 帧")
print(f"  有效 batch    : {bs} × {nproc} 卡 = {bs * nproc}")
print(f"  目标训练量    : {origin} → {steps:,} 步（{steps * bs * nproc / frames:.2f} pass）")
print(f"  lr schedule   : warmup {sched['scheduler_warmup_steps']} 步，"
      f"余弦 {sched['optimizer_lr']:g} → {sched['scheduler_decay_lr']:g}，"
      f"decay_steps = steps = {steps:,}")
print(f"  checkpoint    : save_freq={save_freq} → {n_ckpt} 个 ≈ {need_gb:.1f} GB"
      f"（输出盘剩余 {free_gb:.0f} GB）")
if free_gb < need_gb + 5:
    print(f"  ⚠ 磁盘余量偏紧：跑满后只剩约 {free_gb - need_gb:.0f} GB，建议先清旧 run 或调大 save_freq")
print(f"  预计墙钟      : {hours:.1f} h = {hours / 24:.2f} 天（按 8 卡 pi05 LoRA 实测 3.6 s/it 外推）")
print(f"steps={steps}")
PY
) || die "训练量推导失败（检查 $INFO_JSON 的 total_frames / 配置的 batch_size）"

STEPS=$(printf '%s\n' "$DERIVE" | sed -n 's/^steps=//p')
printf '%s\n' "$DERIVE" | sed '$d'   # 去掉末行的机器可读值
[ -n "$STEPS" ] || die "训练量推导失败：拿不到 steps"
[ "$STEPS" -gt 0 ] 2>/dev/null || die "推导出 steps=$STEPS（STEPS=$STEPS_REQUEST / NUM_PASSES=$NUM_PASSES 太小了）"

# 配置里那份 steps 只是「记录」，启动时一律以推导值为准（decay_steps 同理）。
JSON_STEPS=$("$PY" -c "import json;print(json.load(open('$CONFIG'))['steps'])")
if [ "$JSON_STEPS" != "$STEPS" ]; then
    echo "  ⚠ 配置里写的是 steps=$JSON_STEPS，本次按推导值 $STEPS 启动（配置值仅作记录）"
fi

# ---------------- 启动 ----------------
OUT_DIR=$("$PY" -c "import json;print(json.load(open('$CONFIG'))['output_dir'])")
CKPT_CFG="$SELF_SCRIPTS_DIR/$OUT_DIR/checkpoints/last/pretrained_model/train_config.json"
# ⚠️ 日志**不能**放在 output_dir 里面。lerobot 的 TrainPipelineConfig.validate() 在
# 「output_dir 已存在且非 resume」时会直接抛 FileExistsError —— 而我们要写日志就得先
# mkdir，那个 mkdir 会先把 output_dir 建出来，于是紧接着 lerobot 自己就把自己拦下了。
# 所以日志放到 output_dir 的**同级**目录：output_lerobot_train/logs/<run名>.log
LOG_FILE="$SELF_SCRIPTS_DIR/$(dirname "$OUT_DIR")/logs/$(basename "$OUT_DIR").log"

step "启动训练"

if [ "$RESUME" = 1 ] && [ -f "$CKPT_CFG" ]; then
    echo "从 checkpoint 续训：$CKPT_CFG"
    echo "（续训用 checkpoint 里的配置：里面的 steps / scheduler_decay_steps 是上次开工时算好的，"
    echo "  上面的推导值本次不生效，只是给你看当前数据集规模对应多少 pass）"
    LAUNCH=(accelerate launch --multi_gpu --num_processes="$NPROC" "$(which lerobot-train)"
            --config_path="$CKPT_CFG" --resume=true)
elif [ "$RESUME" = 1 ]; then
    die "--resume 指定了，但找不到 checkpoint：$CKPT_CFG"
else
    if [ -d "$SELF_SCRIPTS_DIR/$OUT_DIR" ]; then
        EXIT_CODE=3 die "输出目录已存在：$OUT_DIR
   换个配置（改 output_dir），或用 --resume 续训。"
    fi
    echo "从头训练（新 run，输出到 $OUT_DIR）"
    echo "日志：$LOG_FILE"
    LAUNCH=(accelerate launch --multi_gpu --num_processes="$NPROC" "$(which lerobot-train)"
            --policy.pretrained_path="$PI05_BASE"
            --config_path="$CONFIG"
            --policy.push_to_hub=false
            --steps="$STEPS"
            --policy.scheduler_decay_steps="$STEPS")
fi

echo "  ${LAUNCH[*]}"
echo

if [ "$DRY_RUN" = 1 ]; then
    echo "（--dry-run：前置检查已通过，不实际启动训练）"
    exit 0
fi

# 日志目录在这里才创建，有两层原因：
#   1. 必须在「输出目录是否存在」的判断之后 —— 否则 mkdir 会先把 output_dir 建出来，
#      紧接着的存在性检查就会把自己的创建动作当成冲突而误报。
#   2. 必须在 DRY_RUN 提前退出之后 —— 否则跑一次 --dry-run 就留下 output_dir，
#      真正开训时反而被「目录已存在」挡住。
mkdir -p "$(dirname "$LOG_FILE")" || die "无法创建日志目录"

# 训练输出同时落日志。用 PIPESTATUS[0] 取 lerobot-train 的退出码（而非 tee 的）
set -o pipefail
CUDA_VISIBLE_DEVICES="$GPU_LIST" "${LAUNCH[@]}" 2>&1 | tee -a "$LOG_FILE"
TRAIN_RC=${PIPESTATUS[0]}

# ---------------- 训练后复查：数据在训练期间被改过吗 ----------------
if [ -n "$ROWS_BEFORE" ]; then
    step "训练后复查：数据集是否在训练期间被改动"
    ROWS_AFTER=$("$PY" - "$DS_ROOT" <<'PY' 2>/dev/null
import glob, os, sys
import pyarrow.parquet as pq
files = sorted(glob.glob(os.path.join(sys.argv[1], "data/chunk-*/*.parquet")))
print(sum(pq.ParquetFile(f).metadata.num_rows for f in files))
PY
)
    if [ -z "$ROWS_AFTER" ]; then
        echo "  （无法复查，跳过）"
    elif [ "$ROWS_AFTER" = "$ROWS_BEFORE" ]; then
        echo "  ✓ 行数未变（$ROWS_BEFORE），训练期间数据没被改动"
    else
        echo "  ✗✗ 数据在训练期间被改动：$ROWS_BEFORE → $ROWS_AFTER 行"
        echo "    源端大概率在训练中途重传，而钩子的守卫 3 在训练期间会跳过修复。"
        echo "    这份 checkpoint 学到的是错配的数据，别直接上真机。"
        echo "    现在先跑：$PY $VERIFY --root $DS_ROOT --fix"
    fi
fi

exit "$TRAIN_RC"
