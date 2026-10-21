#!/usr/bin/env bash
# 离线评测 330 集模型：bash run_offline_330.sh /path/to/b601_v21 [episodes...]
set -euo pipefail
cd "$(dirname "$0")"
source ./b601_pi05_env.sh
DS=${1:?用法: bash run_offline_330.sh <v2.1数据集目录> [episodes...]}
shift || true
EPS=${*:-0 40 120}
$PY infer_offline.py --model 330 --dataset-root "$DS" --episodes $EPS --out offline_330.json
