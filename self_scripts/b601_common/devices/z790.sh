#!/bin/bash
# 设备档案：z790（RTX 4090D，B601 采集主点）— 2026-11-02 建立
# 由 b601_common/_env.sh 按 hostname 自动 source
export B601_HOST_PROFILE="z790"

# ---- 环境：conda env `lerobot`（0.4.4 线，与 LX/pai0 运行时代码配套）----
export B601_ENV_KIND="conda"
export B601_CONDA_ENV="${B601_CONDA_ENV:-lerobot}"

# ---- 数据根：与代码仓分开（本机沿用旧路径）----
export B601_DATA_ROOT="${B601_DATA_ROOT:-$HOME/LX/pai0/b601_data}"

# ---- 主臂 StarArm102：本机固定 ttyUSB0（无 rebot_leader 别名）----
export B601_LEADER_PORT="${B601_LEADER_PORT:-/dev/ttyUSB0}"

# ---- 相机角色（2026-11-02 实测三路 JYU2C；另有一台 2609007 可作备用/第四路）----
export B601_CAM_HAND="${B601_CAM_HAND:-2608076}"
export B601_CAM_FRONT="${B601_CAM_FRONT:-2607031}"
export B601_CAM_TOP="${B601_CAM_TOP:-2607060}"
