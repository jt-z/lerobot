#!/bin/bash
# 单 B601-RS ACT 推理（base 策略：纯推理，不建数据集、不录帧、不落盘）
# 任务：单臂端起纸杯放上咖啡机托盘、拿方块按按钮等 4 秒、方块放回桌面、再把杯子端回桌面
#       （任务文本/相机/fps/方向修正与 self_scripts/b601_single/02_record.sh 采集完全一致）
#
# 用法：
#   bash run_inference_ACT.sh                # 默认：180_new_datasets_act_model/100k（旧 100k 权重）
#   bash run_inference_ACT.sh a800           # 8×A800 训练 50k（462 集），不限时长
#   bash run_inference_ACT.sh A              # A/B 对比：全量 462 集 / 50k，限时 600 秒
#   bash run_inference_ACT.sh B              # A/B 对比：前 330 集 / 35.5k，限时 600 秒
#   bash run_inference_ACT.sh /path/to/pretrained_model     # 任意 ACT 权重目录
#
# 权重来源：
#   100k   /home/kf/dev/lerobot/model_weights/180_new_datasets_act_model/100k_pretrained_model
#   a800/A 2026-10-20 a800-55 训练的 8×A800 ACT 50k（462 集 clean，eff.batch 64，chunk 100）；
#          55 → z790 rsync 591M，双端 sha256 11/11 一致
#   B      /home/kf/LX/pai0/act_330_35k/checkpoints/035500（前 330 集）
# 训练配置要点（ACT）：chunk_size=100，n_action_steps=100，无 temporal ensemble（单次前向解码）；
#   输入 state(7) + hand/front/top(3x480x640)，MEAN_STD 归一化；训练数据 30Hz
#
# 需要把推理过程存成数据集时，改用 self_scripts/b601_single/04_dagger_collect.sh
# 环境/硬件检查/参数总览/命令拼装等公共逻辑见 _common.sh

set -e  # 遇到错误立即退出
set -o pipefail  # 管道中任一命令失败则整体失败（配合 tee 使用）

source "$(dirname "$(readlink -f "$0")")/_common.sh"

# ==================== 权重选择 ====================
CKPT_A800="/home/kf/LX/pai0/act_a800_8x_50k/checkpoints/050000/pretrained_model"
CKPT_330="/home/kf/LX/pai0/act_330_35k/checkpoints/035500/pretrained_model"
CKPT_100K="/home/kf/dev/lerobot/model_weights/180_new_datasets_act_model/100k_pretrained_model"

AB_TAG=""
RUN_TAG="make_coffee_act_50k"
TITLE="单臂推理（ACT 50k）- B601-RS（倒咖啡 / base 纯推理）"
case "${1:-default}" in
  default) MODEL_PATH="$CKPT_100K" ;;
  a800)    MODEL_PATH="$CKPT_A800" ;;
  A) MODEL_PATH="$CKPT_A800"; DURATION=600; AB_TAG="A_full50k"; RUN_TAG="AB_A_full50k"
     TITLE="单臂推理 A/B [A_full50k] 全量 462 集 / 50k 步 - B601-RS" ;;
  B) MODEL_PATH="$CKPT_330";  DURATION=600; AB_TAG="B_subset330"; RUN_TAG="AB_B_subset330"
     TITLE="单臂推理 A/B [B_subset330] 前 330 集 / 35.5k 步 - B601-RS" ;;
  -*) echo "❌ 未知参数：$1（见文件头用法）"; exit 1 ;;
  *)  MODEL_PATH="$1"; TITLE="单臂推理（ACT）- B601-RS（$1 / base 纯推理）" ;;
esac

# ACT 单次前向解码，无 num_steps / num_inference_steps 参数，留空即可
NUM_STEPS_ARG=""

# A/B 两次对比各打一份动作日志；日常跑则共用 action_test.log
if [ -n "$AB_TAG" ]; then
  ACTION_TRACE_LOG="$HOME/dev/lerobot/self_scripts/_logs/AB_${AB_TAG}_actions_$(date +%Y%m%d_%H%M%S).log"
fi

# ==================== ACT 内部注意力可视化 ====================
# 1 = 用 act_feature_viz.py 包装 lerobot-rollout（运行时打补丁，不改 lerobot / ACT 源码）：
#     抓最后一层 decoder 的交叉注意力（100 个动作 query → 902 token，其中 3×300 是
#     hand/front/top 的 15×20 feature map），按 head 聚合后还原成像素网格、上采样并叠在
#     相机原图上，写进 Rerun 的 act_viz/attn/<cam>。ACT 每 100 tick 才前向一次
#     （chunk_size=100 @30Hz ≈ 3.3s），所以缓存整段注意力，每 ACT_VIZ_EVERY_N 次取动作
#     换一帧显示"当前执行到 chunk 内第 k 步"的分布。底图是该 chunk 生成时的那帧相机图。
#     分析视图（ACT_VIZ_ANALYSIS=1）：attn_heads/<cam> 8 个 head 热图、timeline/<cam>
#     （横轴=chunk 内第 k 步）、mass/<cam>（各相机注意力质量）、selfattn（100×100 自注意力）。
# 与 ACTION_TRACE 可同时为 1：此时统一用 act_feature_viz.py 作入口，它会顺带装上动作追踪补丁。
ACT_VIZ=0
ACT_VIZ_WRAPPER="$_B601_INFER_DIR/../../tools/act_feature_viz.py"
ACT_VIZ_EVERY_N=1       # 每 N 次取动作刷一帧（1 = 30Hz；调大降帧率、省 CPU）
ACT_VIZ_ANALYSIS=1      # 1 = 额外输出四个分析视图与对应 MP4；0 = 只留主叠图
ACT_VIZ_HEADS_EVERY=10  # per-head 网格刷新间隔（tick，10 ≈ 3Hz）
ACT_VIZ_HEADS=mean      # head 聚合方式：mean | max
ACT_VIZ_ALPHA=0.55      # 热力图最大叠加强度 0~1
ACT_VIZ_BLUEPRINT=1     # 1 = 补发含 act_viz 视图的 Rerun blueprint（会覆盖 lerobot 默认布局）
# ---- 离线 MP4（默认开启，与 Rerun 是否可用无关；kill -9 会导致 MP4 不完整）----
ACT_VIZ_MP4=1           # 1 = 每帧同时写 MP4；0 = 只在线看
ACT_VIZ_MP4_DIR=""      # 输出目录；留空 = self_scripts/_logs/act_viz_<启动时间戳>/
ACT_VIZ_MP4_MODE=tiled  # tiled（hand/front/top 横拼 1920x480）| per_cam | both
ACT_VIZ_MP4_CRF=23      # x264 质量（越小越清晰、文件越大）

# ==================== 推理前检查 ====================
b601_require_lerobot
b601_banner "$TITLE"

# 1) 模型权重（先查，路径不对直接退出，避免误触硬件）
b601_check_model_files "$MODEL_PATH" config.json model.safetensors

# 2) 包装器
b601_check_trace_wrapper
if [ "$ACT_VIZ" = "1" ] && [ ! -f "$ACT_VIZ_WRAPPER" ]; then
  echo "❌ 缺少 ACT 可视化包装器：$ACT_VIZ_WRAPPER（或把 ACT_VIZ 设为 0）"
  exit 1
fi

# 3) 硬件
b601_check_can
b601_check_cameras
b601_check_calibration

# ==================== 推理参数总览 ====================
echo ""
echo "=========================================="
echo "推理参数总览"
echo "=========================================="
echo "模型路径：$MODEL_PATH"
echo "机器人类型：seeed_b601_rs_follower（单臂 B601-RS，can0）"
echo "策略：base（纯推理，不录制、不保存数据）"
echo "推理方式：sync（ACT 不支持 RTC）"
b601_print_run_params
b601_print_robot_notes
b601_print_trace_summary model/sent/cmd
if [ "$ACT_VIZ" = "1" ]; then
  echo "ACT 注意力可视化：✅ decoder 交叉注意力 → Rerun act_viz/attn/{hand,front,top}"
  echo "                    每 ${ACT_VIZ_EVERY_N} 次取动作换一帧，head 聚合=${ACT_VIZ_HEADS}，alpha=${ACT_VIZ_ALPHA}"
  if [ "$ACT_VIZ_ANALYSIS" = "1" ]; then
    echo "ACT 分析视图：✅ attn_heads/<cam>（${ACT_VIZ_HEADS_EVERY} tick 刷一次）、timeline/<cam>、mass/*、selfattn"
  else
    echo "ACT 分析视图：❌ 关闭（ACT_VIZ_ANALYSIS=0）"
  fi
  if [ "$ACT_VIZ_MP4" = "1" ]; then
    echo "ACT 注意力 MP4：✅ 模式=${ACT_VIZ_MP4_MODE}，crf=${ACT_VIZ_MP4_CRF}，输出=${ACT_VIZ_MP4_DIR:-self_scripts/_logs/act_viz_<时间戳>/}"
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

b601_confirm_zero
b601_start_run "开始单臂模型推理（ACT 50k）..." "$RUN_TAG"

b601_run_rollout \
  --strategy.type=base \
  --inference.type=sync \
  --policy.path="$MODEL_PATH" \
  $NUM_STEPS_ARG

# ==================== 推理完成 ====================
echo ""
echo "=========================================="
echo "✅ 单臂推理结束（ACT 50k / base 模式）"
echo "=========================================="
echo "本次未保存任何推理数据（base 策略不录制、不落盘）。"
echo "如需保存推理数据，请改用 self_scripts/b601_single/04_dagger_collect.sh"
echo "完整日志：$LOG_FILE"
echo "=========================================="
