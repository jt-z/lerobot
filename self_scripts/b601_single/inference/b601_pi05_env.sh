#!/usr/bin/env bash
# b601 π0.5 推理公共环境（source 本文件）
# 用法： source b601_pi05_env.sh
export B601_HOME=/home/kf/b601_pi05
export B601_SFT_ROOT=$B601_HOME
export PYTHONPATH="$B601_HOME/scripts:$B601_HOME/openpi/src:$B601_HOME/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
unset B601_BASE  # 权重加载器只在训练用，推理不需要
export OPENPI_DATA_HOME=$B601_HOME/cache/openpi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export B601_REPO=b601/v21_clean
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=$B601_HOME/qgf_openpi_env/bin/python
# 两个 checkpoint 的短名
declare -A B601_CKPT=(
  [330]="$B601_HOME/checkpoints/330_24653"    # 前 330 集，24,653 步
  [462]="$B601_HOME/checkpoints/462_34750"    # 全 462 集，34,750 步
)
