#!/bin/bash
# 设备档案：kfzbox / 公司 165 机（RTX 3060，B601 第二采集点）— 2026-11-02 建立
# 由 b601_common/_env.sh 按 hostname 自动 source
export B601_HOST_PROFILE="kfzbox"

# ---- 环境：~/dev/miniforge3/envs/lerobot_b601（整目录迁移版，py3.12 / torch2.11+cu128 / lerobot 0.6.2）----
export B601_ENV_KIND="prefix"
export B601_ENV="${B601_ENV:-$HOME/dev/miniforge3/envs/lerobot_b601}"

# ---- 数据根：与代码仓分开 ----
export B601_DATA_ROOT="${B601_DATA_ROOT:-$HOME/b601_data}"

# ---- 主臂 StarArm102：CH340 无序列号，优先 udev 稳定别名；留空则按候选列表自动探测 ----
export B601_LEADER_PORT="${B601_LEADER_PORT:-}"
export B601_LEADER_CANDIDATES="${B601_LEADER_CANDIDATES:-/dev/rebot_leader /dev/serial/by-id/usb-1a86_USB_Serial*-if00-port0 /dev/ttyUSB* /dev/ttyACM*}"

# ---- 臂标定 id：与 z790 区分开（标定文件按 id 命名，两台必须各有一份零位）----
export B601_FOLLOWER_ID="${B601_FOLLOWER_ID:-follower_kfzbox}"
export B601_LEADER_ID="${B601_LEADER_ID:-leader_kfzbox}"

# ---- 相机（2026-11-02 实测：两路 JYU2C，镜头盖已摘，单开都 29.7fps）----
#   画面判定：2609010 能看到整个工作台/工具 → top；2610057 贴近机械臂本体 → hand
#   现场看一眼不对就改下面两行的序列号（只改本机档案，不动入口脚本）
#   本机只有两路，front 视角不采（角色列表里没有 front）
export B601_CAM_ROLES="${B601_CAM_ROLES:-hand top}"
export B601_CAM_HAND="${B601_CAM_HAND:-2610057}"
export B601_CAM_TOP="${B601_CAM_TOP:-2609010}"
