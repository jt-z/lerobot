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

# ---- 臂标定 id ----
# 实测两台的 B601 从臂 / StarArm102 主臂基本一致，且臂不需要每台各校一次：
# 两台共用一份默认标定（follower / leader），默认值在 devices/_defaults.sh + _env.sh 里。
# 若将来某台臂明显偏（换过电机/被拆装过），再取消下面两行注释、单独校一份。
# export B601_FOLLOWER_ID="${B601_FOLLOWER_ID:-follower_kfzbox}"
# export B601_LEADER_ID="${B601_LEADER_ID:-leader_kfzbox}"

# ---- 相机（2026-11-03 三路 JYU2C，镜头盖已摘，单开都 29.7fps）----
#   画面判定（2026-11-05 实测修正，此前 hand/top 接反）：2610057 能俯视整个工作台/工具 → top；
#             2609010 贴近机械臂本体/夹爪 → hand；2608006 侧面第三视角 → front
#   ⚠️ 三台 JYU2C 都是 USB2.0，每路固定预留 altsetting 11（约 24.5MB/s）。
#      若两路挂在同一个 USB hub 上，第三路会报 "Not enough bandwidth for altsetting 11"。
#      实测把 front 直接插到主机 USB 口（不经 hub）后，三路可同时 30fps 正常采集。
export B601_CAM_ROLES="${B601_CAM_ROLES:-hand front top}"
export B601_CAM_HAND="${B601_CAM_HAND:-2609010}"
export B601_CAM_FRONT="${B601_CAM_FRONT:-2608006}"
export B601_CAM_TOP="${B601_CAM_TOP:-2610057}"
