#!/usr/bin/env bash
# b601 π0.5 离线动作 MAE 评测（与 A800 上 b601_eval_mae.py 完全同口径）
# 只在本机 4090 上跑策略前向，不接机器人、不发动作。
#
# 用法：
#   bash run_offline.sh 330 /home/kf/LX/pai0/b601_data/b601_v21            # 默认 ep 0 40 120
#   bash run_offline.sh 462 /home/kf/LX/pai0/b601_data/b601_v21 0 40 120 200
#   bash run_offline.sh 330 --anchors /home/kf/b601_pi05/anchors_6.npz     # 用锚点，不搬 22G 数据集
#
# 结果写到本目录 offline_<模型>.json（offline_330.json / offline_462.json）
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

MODEL=${1:?用法: bash run_offline.sh <330|462> <数据集目录|--anchors npz> [episodes...]}
case "$MODEL" in
  330|462) ;;
  *) echo "❌ 未知模型：$MODEL（只支持 330 / 462）"; exit 1 ;;
esac
shift

SRC=${1:?用法: bash run_offline.sh <330|462> <数据集目录|--anchors npz> [episodes...]}
if [ "$SRC" = "--anchors" ]; then
  ANCHORS=${2:?--anchors 后面要跟 npz 路径}
  shift 2
  SRC_ARG=(--anchors "$ANCHORS")
else
  shift
  SRC_ARG=(--dataset-root "$SRC")
fi
EPS=${*:-0 40 120}

$PY eval_offline.py \
  --ckpt "${B601_CKPT[$MODEL]}" \
  "${SRC_ARG[@]}" \
  --episodes $EPS \
  --out "offline_${MODEL}.json"
