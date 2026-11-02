#!/bin/bash
# B601 统一环境入口（2026-11-02 建立，取代原先每台机各写一份的做法）
#
# 用法：入口脚本里
#   _B601_ENV="$(dirname "$(readlink -f "$0")")/../b601_common/_env.sh"
#   [ -f "$_B601_ENV" ] && . "$_B601_ENV"
#
# 做的事：
#   1) 按主机名选设备档案 devices/<host>.sh（B601_DEVICE=... 可强制覆盖）
#   2) 解析主臂串口（档案给了就用，没给就按候选列表取第一个存在的）
#   3) 激活 python 环境（conda env 或 目录前缀两种），并统一导出 B601_ENV=<前缀>

_B601_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export B601_REPO_ROOT="${B601_REPO_ROOT:-$(cd "$_B601_ENV_DIR/../.." && pwd)}"

# ---- 1) 设备档案 ----
B601_DEVICE="${B601_DEVICE:-$(hostname -s)}"
case "$B601_DEVICE" in
  kfzbox|mineru-box|mineru_box) B601_DEVICE="kfzbox" ;;
  kf-z790|z790|kfz790)          B601_DEVICE="z790"   ;;
esac
export B601_DEVICE
. "$_B601_ENV_DIR/devices/_defaults.sh"
if [ -f "$_B601_ENV_DIR/devices/$B601_DEVICE.sh" ]; then
  . "$_B601_ENV_DIR/devices/$B601_DEVICE.sh"
else
  echo "⚠️  没有设备档案 devices/$B601_DEVICE.sh，只用通用默认值（相机/数据根可能不对）" >&2
fi

# ---- 2) 主臂串口 ----
if [ -z "${B601_LEADER_PORT:-}" ]; then
  for _cand in ${B601_LEADER_CANDIDATES:-$B601_LEADER_CANDIDATES_DEFAULT}; do
    if [ -e "$_cand" ]; then B601_LEADER_PORT="$_cand"; break; fi
  done
fi
export B601_LEADER_PORT
unset _cand

# ---- 3) python 环境 ----
case "${B601_ENV_KIND:-prefix}" in
  conda)
    if command -v conda >/dev/null 2>&1; then
      conda activate "${B601_CONDA_ENV:-lerobot}" 2>/dev/null || true
    else
      for _base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/dev/miniforge3" /opt/conda; do
        if [ -f "$_base/etc/profile.d/conda.sh" ]; then
          . "$_base/etc/profile.d/conda.sh"
          conda activate "${B601_CONDA_ENV:-lerobot}" 2>/dev/null || true
          break
        fi
      done
      unset _base
    fi
    export B601_ENV="${B601_ENV:-${CONDA_PREFIX:-}}"
    ;;
  prefix)
    if [ -d "${B601_ENV:-}" ]; then
      export PATH="$B601_ENV/bin:$PATH"
    else
      echo "⚠️  环境目录不存在: ${B601_ENV:-<未设置>}（设备档案 $B601_DEVICE）" >&2
    fi
    ;;
esac

export B601_DATA_ROOT="${B601_DATA_ROOT:-$HOME/b601_data}"
