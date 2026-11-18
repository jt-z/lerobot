# B601 单臂笔记索引

本目录记录 **B601-RS 单臂 + 咖啡场景** 的数据集资产、采集脚本、pi05 模型推理与部署，按 `01_主题/文件.md` 分四类。

文件命名统一为 **`主题_类型_标题.md`**：
- 主题：`数据集` / `硬件` / `遥操作` / `采集` / `回放` / `pi05`
- 类型：`清单` / `指南` / `问题` / `分析` / `核对` / `机制`

这样 `ls` 和搜索能按主题聚在一起，类型只表示文档性质，不再和主题混在同一层。

## 维护约定（单一真源）

同一事实只在一处维护，其他文件引用而不复制：

| 事实 | 真源 |
|---|---|
| prompt 原文 | [本地数据集清单](01_数据集资产/数据集_清单_本地资产.md) §二 |
| 实测显存 / chunk 时延 | [pi05 时延与显存分析](04_推理部署/pi05_分析_时延与显存.md) |
| 推理链路与 lite 省显存机制 | [pi05 推理链路](03_模型架构/pi05_机制_推理链路.md) |
| CAN / 电机现场排查 | [CAN 与电机离线排查](02_数据采集/硬件_问题_CAN与电机离线.md) |
| 硬件参数（相机序列号、方向修正等） | `self_scripts/b601_common/devices/<host>.sh`，下方有 kfzbox 速查 |

目录边界：**03 放机制与一致性**（链路怎么走、输入输出是否对齐），**04 放运行与实测**（怎么跑、多快、怎么选、代码风险）。

## 01_数据集资产

### 1.1 [本地资产清单](01_数据集资产/数据集_清单_本地资产.md)
当前机器上有哪些数据集、各多少集/帧、放在哪、哪些可直接用于训练。
- 仓库根目录三份数据集（b601_1 / b601_2 / b601_mixed）
- Hugging Face 缓存下的完整数据集（16 份）
- 100 份 `rollout_*` 推理回放目录的用途与局限
- 4 条训练 prompt 原文（其他笔记引用此处的文本）
- 一条命令汇总所有数据集元信息

## 02_数据采集

### 2.1 [单臂脚本与遥操作链路](02_数据采集/遥操作_指南_单臂脚本与链路.md)
01~06 各脚本干什么、怎么用，以及遥操作的真实控制链路。
- 01 的执行顺序：设备档案 → CAN 复位 → 串口/校准检查 → ENTER → `lerobot-teleoperate`
- 60Hz 关节角直通（非 IK）：主臂 `-6` → `[-270,0]`，从臂 `-1` → `[0,270]`，夹爪方向不能改
- 无 `max_relative_target` = 无速度限幅；`safe_zero` 退出会自己回坐姿
- 02 按键 / 每集流程 / 默认参数 / 数据落点 / `--resume`
- 03 回放单集（`DATASET_ROOT` 必须显式传、`EPISODE` 默认 0）
- 变更记录：2026-11-17 02 的数据根改回 `B601_DATA_ROOT`

### 2.2 [采集脚本溯源](02_数据采集/采集_指南_脚本溯源.md)
已有数据集是哪个脚本采的、参数在哪里定义、如何原样重采或续录。
- b601_1 ← `02_record.sh`，b601_2 ← `05_record_place_cup.sh`，b601_mixed = 二者合并
- 公共参数层 `_env.sh` / `devices/<host>.sh`（相机、方向修正、数据根）
- 日志与数据集的对应关系
- 重采 / 续录命令与注意事项

### 2.3 [美式咖啡（第 4 条任务）采集](02_数据采集/采集_指南_美式咖啡第4任务.md)
`06_record_make_americano.sh`：任务文本、用法、与 05 的差异。
- 任务 prompt（训练/推理必须逐字一致）
- `--episodes` / `--resume` 用法
- 采集注意事项（咖啡机设置复位、按钮按压方式）
- **状态**：prompt 已定，尚未加入 `lx/pi05_inference.sh` 的 `TASKS`（计划作为 `TASK_INDEX=4`）

### 2.4 [episode 回放与键盘控制](02_数据采集/回放_指南_episode44与键盘控制.md)
把 `b601_1` 的 episode 44 固定成一条回放命令，并给回放加暂停与拖动。
- `lx/replay_task1_ep44.sh` → `03_replay.sh` → `06_replay_rerun.py`
- `p` 暂停、`r` 继续、`←/→` 按回放速度慢速 ±2s、`v` 开关画面
- `b601_1` / `b601_2` / `b601_mixed` 的 ep44 任务归属核对
- `USE_RERUN=0` 走原生 replay，没有键盘控制

### 2.5 [CAN 与电机离线](02_数据采集/硬件_问题_CAN与电机离线.md)
同一物理层的两类现场：遥操作/采集时电机 0/7 在线，以及推理启动时 `disable_all` 应答超时。
- 总线证据判读：`ERROR-ACTIVE` + `RX packets=0` + `candump` 空
- 先排除软件侧，再按急停 → 48V 断电重启 → 分配板 → CAN 线/终端电阻
- 已补启动前 can0 `down → 1Mbps → up` 并清空残留帧
- 修好后的确认顺序：`candump` 有帧 → `07_preflight.sh` 见 7 个 ID → 再启动

## 03_模型架构

### 3.1 [pi05 推理链路](03_模型架构/pi05_机制_推理链路.md)
`lx/` 下 π0.5 真机推理的调用链与 lite 包装器机制。
- `pi05_inference.sh` → `_env.sh` → `_common.sh` → `pi05_lite.py` → `lerobot-rollout`
- 12GB 装下 4.14B 模型的 5 个手段（bf16 构建、跳过 lm_head/expert 词嵌入等）
- 权重加载（strict、缺 key 直接报错，绝不退回随机权重）
- 观测→动作数据流、sync/rtc 在模型侧的差异
- 必须保持的不变量（prompt、相机键、方向修正、FPS）

### 3.2 [输入输出一致性](03_模型架构/pi05_核对_输入输出一致性.md)
上机前逐项比对推理输入与训练 / 数据集，确认坐标系没有错位。
- 4 条 prompt 逐字节 MATCH，最长 80 token < `max_length=96`
- state/action 名称、顺序、维度、单位、绝对/相对、归一化方式
- normalizer 的 q01/q99 与 `meta/stats.json` 逐值一致
- 方向修正 −1.0、重力补偿、标定 id、相机序列号、FPS
- 两个坑：默认任务是 0（不是放置任务）；`max_length` 96 的等价性未重跑

## 04_推理部署

### 4.1 [运行与任务切换](04_推理部署/pi05_指南_运行与任务切换.md)
启动命令、4 条子任务 prompt、与 ACT 单任务脚本的区别。
- `TASK_INDEX=0/1/2/3` 与 1/2/3 号取杯区的对应关系
- 区域 2 的两种写法（pi05 用任务下标，ACT 用 area 参数）
- 环境变量覆盖（`TASK_DESCRIPTION` / `INFERENCE_TYPE` / `LITE_*`）
- 前置检查、常见失败与日志统计

### 4.2 [时延与显存](04_推理部署/pi05_分析_时延与显存.md)
3060 12GB 上的实测数据与优化边界。
- 显存账 16.57GB → 6.97GB
- 锁定/解锁 × sync/rtc 的 chunk 时延（354 / 501ms）
- embed / prefix / denoise 分段归因
- rtc 预取预算、断流阈值、下一步优化方向

### 4.3 [sync 与 rtc 选型](04_推理部署/pi05_分析_sync与rtc选型.md)
为什么 sync 更贴、rtc 更顺但发黏，以及 3060 上怎么选。
- guided 修正覆盖 40% 步数、每段丢 15~17 步、视觉滞后 1.17s、宿主争用
- 本模型 `rtc_training_max_delay=0`，只能跑 guided，`trained` 直接 ValueError
- openpi 原生训练/推理都是同步分块（无 RTC）
- 3060 停 354ms（11 tick）vs 4090 估 150~220ms（5~7 tick）：机制相同，只是短到不可见
- 选型表与 `RTC_*` 参数约束、成功率对照实验方法

### 4.4 [代码风险清单](04_推理部署/pi05_问题_代码风险清单.md)
对推理脚本的代码走查结果，按严重度排序。
- R1 相机缺失只告警、模型会静默用空图补位
- R2 `ACTION_TRACE=1` 在 lite 路径下静默失效
- R3 `--bench` 默认 token 200、强制同步导致测速偏差
- R4~R9 行为一致性、文档与日志问题
- **R10（P0，已修）`TASK_DESCRIPTION` 被 `_common.sh` 覆盖 → 实际传 OOD prompt**
- **R11/R12（已修）`INFERENCE_TYPE` 默认 rtc 失效、启动前缺 can0 复位**
- 共同教训：`_common.sh` 无条件覆盖环境变量，默认值必须在 source 前快照

## 快速参考

参数默认值见 [pi05 推理运行 §二](04_推理部署/pi05_指南_运行与任务切换.md#二启动命令)，实测时延/显存见 [时延与显存分析](04_推理部署/pi05_分析_时延与显存.md)，不在这里维护第二份。

### 数据集 → 脚本

| 数据集 | 采集脚本 | prompt 条数 | 集数 |
|---|---|---|---|
| `b601_1` | `02_record.sh` | 1（取杯/接咖啡/按按钮） | 110 |
| `b601_2` | `05_record_place_cup.sh` | 3（1/2/3 号放杯区） | 110（40+40+30） |
| `b601_mixed` | 以上两者合并 | 4 | 220 |
| `b601_americano_<时间戳>` | `06_record_make_americano.sh` | 1（中杯 + 中等浓度） | 默认 120 |

### 关键路径

| 用途 | 路径 |
|---|---|
| 采集入口脚本 | `self_scripts/b601_single/0{1,2,3,4,5,6}_*.sh` |
| 设备/环境公共层 | `self_scripts/b601_common/_env.sh` |
| 设备档案 | `self_scripts/b601_common/devices/{_defaults,kfzbox,z790}.sh` |
| 采集日志 | `self_scripts/_logs/`（`record_b601_*` / `place_cup_r*` / `americano_*`） |
| 数据集（仓库根，已 gitignore） | `b601_1/`、`b601_2/`、`b601_mixed/` |
| 数据集（HF 缓存） | `~/.cache/huggingface/lerobot/<owner>/<name>/` |
| 推理脚本（TASKS 数组） | `lx/pi05_inference.sh` |
| pi05 权重 | `lx/output_pi05/007000/pretrained_model`（`003000` 截断，不可用） |
| pi05 省显存包装器 | `lx/tools/pi05_lite.py` |
| pi05 交接 / 时延文档 | `lx/PI05_HANDOFF.md`、`lx/PI05_LATENCY.md` |
| ACT 分区推理 | `.../inference/lerobot/run_act_place_area.sh <1\|2\|3>` |
| 推理日志 | `self_scripts/_logs/inference_b601_*.log` |

### pi05 子任务 → 命令

pi05 是**一个混合任务权重**，切换任务只换 prompt，不换权重。

| 子任务 | 命令 | 训练集集数 |
|---|---|---|
| 0 取杯/接咖啡/按按钮/回零位 | `bash lx/pi05_inference.sh` | 110 |
| 1 放 1 号取杯区 | `TASK_INDEX=1 bash lx/pi05_inference.sh` | 40 |
| 2 放 2 号取杯区 | `TASK_INDEX=2 bash lx/pi05_inference.sh` | 40 |
| 3 放 3 号取杯区 | `TASK_INDEX=3 bash lx/pi05_inference.sh` | 30 |
| 4 美式咖啡（中杯/中等） | 待接入：prompt 已定，尚未加入 `TASKS` | — |

ACT 单任务路线另有独立权重与脚本：`bash run_act_area2.sh`（区域 2）。

### 当前主机（kfzbox）关键参数

| 项 | 值 |
|---|---|
| 设备档案 | `devices/kfzbox.sh` |
| 从臂 | `seeed_b601_rs_follower` / `can0` / socketcan / 1Mbps / 重力补偿 |
| 主臂 | `rebot_102_leader` / `/dev/rebot_leader` |
| 相机 | hand=2609010、front=2608006、top=2610057（640×480 MJPG 30fps） |
| 方向修正 | 全 `-1.0`（与采集一致，改了就 state/action 不同坐标系） |
| 数据根 | `B601_DATA_ROOT=~/b601_data`（`05/06` 落在仓库根目录） |
| FPS | 30（训练数据 30Hz，不能改） |
