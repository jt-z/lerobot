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

# ---- 相机角色 ----
# 2026-11-02 实测本机只插了 2 路：2609010 / 2610057（第三路未接）。
# 角色（hand/front/top）尚未确认 → 留空 = 由 _defaults.sh 的兜底逻辑按现有相机顺序分配。
# 确认方法：python self_scripts/b601_common/08_cam_probe.py，看图后把序列号填到对应角色。
export B601_CAM_HAND="${B601_CAM_HAND:-}"
export B601_CAM_FRONT="${B601_CAM_FRONT:-}"
export B601_CAM_TOP="${B601_CAM_TOP:-}"
