# kfzbox（RTX 3060）B601 本机说明

> 2026-10-29 建立。本机所有 B601 操作**以 `~/dev/lerobot` 这个仓库为准**，
> `~/LX/pai0/tool` 那套是早期临时脚本，已作废（见 `~/LX/pai0/tool/_SUPERSEDED.md`）。

## 一、环境与路径

| 项 | 值 |
|---|---|
| 代码仓 | `~/dev/lerobot`（分支 `mywork`，与 z790 同一份：rsync 自 `/home/kf/dev/lerobot`，不含 `model_weights/`） |
| 运行环境 | `~/dev/miniforge3/envs/lerobot_b601`（python 3.12.13 / torch 2.11.0+cu128 / lerobot 0.6.2） |
| 环境入口 | `self_scripts/b601_common/_env_kfzbox.sh`（所有入口脚本自动 source，无需手动 conda activate） |
| 数据根 | `~/b601_data`（`B601_DATA_ROOT` 可覆盖；与代码仓分开，避免数据进 git） |
| 旧仓 | `~/dev/lerobot.dev-lerobot`（0.4.4 / `jt-z/lerobot-v0.4`，暂时改名保留，`mv` 回去即还原） |

⚠️ **运行时代码仍是 `~/LX/pai0/lerobot/src`**（`import lerobot` 指向那里）。仓库 `dev/lerobot` 目前只作为
“脚本 + 代码源”，与 z790 的分工一致（z790 也是脚本在 dev/lerobot、运行时在 LX 镜像）。

## 二、端口口径（本机实测）

| 设备 | 端口 | 说明 |
|---|---|---|
| 主臂 StarArm102 | `/dev/rebot_leader` → `ttyUSB0` | CH340 `1a86:7523`，**无序列号**，靠 `/etc/udev/rules.d/99-rebot-leader.rules` 建别名；1 Mbaud |
| 从臂 B601-RS | `can0` | PCAN-USB `0c72:000c`，SocketCAN，1 Mbps；7 个 RobStride id 1–7 |
| 相机 | `/dev/v4l/by-id/*video-index0` | 3 路，640×480 / 30fps / MJPG |

坑：Ubuntu 自带 **brltty** 会抢占 CH340，表现为 `ttyUSB0` 一出现就 disconnect。已屏蔽：
`systemctl mask brltty-udev brltty` + `/etc/udev/rules.d/85-brltty.rules -> /dev/null`。

## 三、命令（都在仓库根目录执行）

```bash
cd ~/dev/lerobot

bash self_scripts/b601_common/07_preflight.sh          # 体检：环境/can0/7电机/主臂/相机/标定
bash self_scripts/b601_common/01_leader_calibration.sh # 主臂校准（新主臂必做）
bash self_scripts/b601_common/02_follower_calibration.sh # 从臂零点（提示时输入 c 才会重写零点）
bash self_scripts/b601_single/01_teleop.sh             # 主从遥操作试动
bash self_scripts/b601_single/02_record.sh             # 采集
bash self_scripts/b601_single/02_record.sh --resume <repo_id> 10   # 续录
bash self_scripts/b601_single/03_replay.sh             # 回放
bash self_scripts/b601_single/04_dagger_collect.sh     # DAgger 纠错采集
python self_scripts/b601_common/08_cam_probe.py        # 抓帧判 hand/front/top
bash self_scripts/b601_common/09_gateway.sh start      # MotorBridge Gateway（Studio 网页后端）
```

## 四、本机做的适配（相对于 z790 原版）

改动集中在 6 个入口脚本，全部带 `# kfzbox 本机适配 (2026-10-29)` 注释：

1. `_env_kfzbox.sh` 新文件 + 每个入口脚本 `set -o pipefail` 后 source 它
   → 用 `lerobot_b601` 而不是旧的 `lerobot`（0.4.4）。
2. 主臂串口：`/dev/ttyUSB0` → 优先 `/dev/rebot_leader`，再退 by-id / ttyUSB* / ttyACM*。
3. `02_record.sh`：`DATA_ROOT` → `${B601_DATA_ROOT:-$HOME/b601_data}`；`--resume` 的会话列表同步改。
4. `02_record.sh`：若配置的三路 by-id 相机不在这台机上，自动按现有 3 路相机兜底并大声提示
   （顺序未必对，先跑 `08_cam_probe.py` 核对画面）。
5. 新增 `07_preflight.sh` / `08_cam_probe.py` / `09_gateway.sh`。

## 五、回滚

```bash
cd ~/dev/lerobot && git checkout -- self_scripts/        # 撤销本机适配（新增文件用 rm）
rm -rf ~/dev/lerobot && mv ~/dev/lerobot.dev-lerobot ~/dev/lerobot   # 整仓回到 0.4.4
```
