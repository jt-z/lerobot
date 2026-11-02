#!/bin/bash
# kfzbox 本机环境入口（2026-10-29 建立）
# 目的：让 dev/lerobot 里的仓库脚本用 lerobot_b601 环境（0.6.2 整目录迁移版），
#      而不是本机旧的 lerobot 环境（0.4.4，SO-101 线），也不用每次手动 conda activate。
export B601_ENV="${B601_ENV:-$HOME/dev/miniforge3/envs/lerobot_b601}"
export PATH="$B601_ENV/bin:$PATH"
export B601_REPO_ROOT="${B601_REPO_ROOT:-$HOME/dev/lerobot}"
export B601_DATA_ROOT="${B601_DATA_ROOT:-$HOME/b601_data}"
