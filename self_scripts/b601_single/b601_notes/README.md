# B601 单臂笔记索引

本目录记录 **B601-RS 单臂 + 咖啡场景** 的数据集资产、采集脚本、pi05 模型推理与部署，按主题分类。
结构参照 `self_scripts/notes/`（`01_主题/文件.md` + 本索引）。

## 笔记列表

### 1. [本地数据集清单](01_数据集资产/数据集_本地数据集清单.md)
当前机器上有哪些数据集、各多少集/帧、放在哪、哪些可直接用于训练。
- 仓库根目录三份数据集（b601_1 / b601_2 / b601_mixed）
- Hugging Face 缓存下的完整数据集（16 份）
- 100 份 `rollout_*` 推理回放目录的用途与局限
- 一条命令汇总所有数据集元信息

### 2. [采集脚本溯源](02_数据采集/指南_采集脚本溯源.md)
已有数据集是哪个脚本采的、参数在哪里定义、如何原样重采或续录。
- b601_1 ← `02_record.sh`，b601_2 ← `05_record_place_cup.sh`，b601_mixed = 二者合并
- 公共参数层 `_env.sh` / `devices/<host>.sh`（相机、方向修正、数据根）
- 日志与数据集的对应关系
- 重采 / 续录命令与注意事项

### 3. [美式咖啡（第 4 条任务）采集](02_数据采集/指南_美式咖啡第4任务采集.md)
新增的 `06_record_make_americano.sh`：任务文本、用法、与 05 的差异。
- 任务 prompt（训练/推理必须逐字一致）
- `--episodes` / `--resume` 用法
- 采集注意事项（咖啡机设置复位、按钮按压方式）
- 与 `lx/pi05_inference.sh` 的 `TASKS` 数组同步

### 4. [单臂脚本用法与遥操作链路](02_数据采集/指南_单臂脚本用法与遥操作链路.md)
01~06 各脚本干什么、怎么用，以及遥操作的真实控制链路。
- 01 的执行顺序：设备档案 → CAN 复位 → 串口/校准检查 → ENTER → `lerobot-teleoperate`
- 60Hz 关节角直通（非 IK）：主臂 `-6` → `[-270,0]`，从臂 `-1` → `[0,270]`，夹爪方向不能改
- 无 `max_relative_target` = 无速度限幅；`safe_zero` 退出会自己回坐姿
- 02 按键 / 每集流程 / 默认参数 / 数据落点 / `--resume` / 环境变量覆盖
- 03 回放单集（`DATASET_ROOT` 必须显式传、`EPISODE` 默认 0）
- 变更记录：2026-11-17 02 的数据根改回 `B601_DATA_ROOT`

### 5. [episode 回放与键盘控制](02_数据采集/指南_episode回放与键盘控制.md)
把 `b601_1` 的 episode 44 固定成一条回放命令，并给回放加暂停与拖动。
- `lx/replay_task1_ep44.sh` → `03_replay.sh` → `06_replay_rerun.py`
- `p` 暂停、`r` 继续、`←/→` 按回放速度慢速 ±2s、`v` 开关画面
- `b601_1` / `b601_2` / `b601_mixed` 的 ep44 任务归属核对
- `USE_RERUN=0` 走原生 replay，没有键盘控制

### 6. [电机不在线排查](02_数据采集/问题_电机不在线排查.md)
从臂 0/7 电机在线时的排查记录（2026-11-17，未闭环）。
- 软件侧逐项排除：环境 / 插件 / 主臂 / 相机 / 校准 / CLI 参数
- 关键证据：发帧被 ACK（TEC=0）但 `rx_packets=0`、扫描全 `no reply`
- 结论：物理层通、至少一个节点带电，问题在电机侧不响应
- 物理排查顺序：急停 → 48V 断电重启（解"喂住"）→ 分配板 → 终端电阻
- 关 48V 再发帧的差分实验与复测命令

### 7. [pi05 推理链路](03_模型架构/模型_pi05推理链路.md)
`lx/` 下 π0.5 真机推理的调用链与省显存机制。
- `pi05_inference.sh` → `_env.sh` → `_common.sh` → `pi05_lite.py` → `lerobot-rollout`
- 12GB 装下 4.14B 模型的 5 个手段（bf16 构建、跳过 lm_head/expert 词嵌入等）
- 权重加载（strict、缺 key 直接报错，绝不退回随机权重）
- 观测→动作数据流、sync/rtc 在模型侧的差异
- 必须保持的不变量（prompt、相机键、方向修正、FPS）

### 8. [pi05 推理运行与任务切换](04_推理部署/指南_pi05推理运行与任务切换.md)
启动命令、4 条子任务 prompt、与 ACT 单任务脚本的区别。
- `TASK_INDEX=0/1/2/3` 与 1/2/3 号取杯区的对应关系
- 区域 2 的两种写法（pi05 用任务下标，ACT 用 area 参数）
- 环境变量覆盖（`TASK_DESCRIPTION` / `INFERENCE_TYPE` / `LITE_*`）
- 前置检查、常见失败与日志统计

### 9. [pi05 时延与显存分析](04_推理部署/分析_pi05时延与显存.md)
3060 12GB 上的实测数据与优化边界。
- 显存账 16.57GB → 6.97GB
- 锁定/解锁 × sync/rtc 的 chunk 时延（354 / 501ms）
- embed / prefix / denoise 分段归因
- rtc 预取预算、断流阈值、下一步优化方向

### 10. [pi05 推理代码风险清单](04_推理部署/问题_pi05推理代码风险清单.md)
对推理脚本的代码走查结果，按严重度排序。
- R1 相机缺失只告警、模型会静默用空图补位
- R2 `ACTION_TRACE=1` 在 lite 路径下静默失效
- R3 `--bench` 默认 token 200、强制同步导致测速偏差
- R4~R9 行为一致性、文档与日志问题
- **R10（P0，已修）`TASK_DESCRIPTION` 被 `_common.sh` 覆盖 → 实际传 OOD prompt**
- **R11/R12（已修）`INFERENCE_TYPE` 默认 rtc 失效、启动前缺 can0 复位**
- 共同教训：`_common.sh` 无条件覆盖环境变量，默认值必须在 source 前快照

### 11. [核对：pi05 输入输出一致性](03_模型架构/核对_pi05输入输出一致性.md)
上机前逐项比对推理输入与训练 / 数据集，确认坐标系没有错位。
- 4 条 prompt 逐字节 MATCH，最长 80 token < `max_length=96`
- state/action 名称、顺序、维度、单位、绝对/相对、归一化方式
- normalizer 的 q01/q99 与 `meta/stats.json` 逐值一致
- 方向修正 −1.0、重力补偿、标定 id、相机序列号、FPS
- 两个坑：默认任务是 0（不是放置任务）；`max_length` 96 的等价性未重跑

### 12. [分析：sync 与 rtc 的选型与动作差异](04_推理部署/分析_sync与rtc选型.md)
为什么 sync 更贴、rtc 更顺但发黏，以及 3060 上怎么选。
- guided 修正覆盖 40% 步数、每段丢 15~17 步、视觉滞后 1.17s、宿主争用
- 本模型 `rtc_training_max_delay=0`，只能跑 guided，`trained` 直接 ValueError
- openpi 原生训练/推理都是同步分块（无 RTC）
- 3060 停 354ms（11 tick）vs 4090 估 150~220ms（5~7 tick）：机制相同，只是短到不可见
- 选型表与 `RTC_*` 参数约束、成功率对照实验方法

### 13. [问题：推理启动时 CAN 应答超时](02_数据采集/问题_推理启动CAN应答超时.md)
`disable_all failed: control ack timeout: comm_type=4` 的定位与处理（两次均与从臂供电/接线有关）。
- 关键判据：`can0` 为 `ERROR-ACTIVE` 但 `RX packets = 0`（总线上没有任何电机报文）
- 与「电机不在线（0/7）」的区别与共同排查路径
- 已补：启动前 `down → 1Mbps → up` + 读空残留帧
- 修好后的确认顺序：`candump` 有帧 → `07_preflight.sh` 见 7 个 ID → 再推理

## 快速参考

### 数据集 → 脚本

| 数据集 | 采集脚本 | prompt 条数 | 集数 |
|---|---|---|---|
| `b601_1` | `02_record.sh` | 1（取杯/接咖啡/按按钮） | 110 |
| `b601_2` | `05_record_place_cup.sh` | 3（1/2/3 号放杯区） | 110（40+40+30） |
| `b601_mixed` | 以上两者合并 | 4 | 220 |
| `b601_americano_<时间戳>`（新） | `06_record_make_americano.sh` | 1（中杯 + 中等浓度） | 默认 120 |

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

ACT 单任务路线另有独立权重与脚本：`bash run_act_area2.sh`（区域 2）。

### pi05 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `chunk_size` / `n_action_steps` | 50 / 50 | 每 chunk 覆盖 50 tick @30Hz = 1667ms |
| `num_inference_steps` | 10 | 去噪步数 |
| 实际 token 长度 | 96 | 预处理器 `max_length`（config 里写的 200 不生效） |
| 视觉塔精度 | bf16 | `LITE_VISION_BF16=1`，精度 0.249° |
| 显存 | 6.97GB | 视觉 bf16；峰值约 7.8GB |
| chunk 时延 | sync 354ms / rtc 501ms | 解锁态实测 |
| `RTC_QUEUE_THRESHOLD` | 35 | 剩余步数触发预取 |
| `RTC_EXECUTION_HORIZON` | 20 | 与上一段动作重叠步数 |

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
