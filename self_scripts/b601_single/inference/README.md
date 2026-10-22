# b601 单臂推理包（z790 / 4090）

本目录按**技术栈**分成三个子目录，先确认要跑哪一个：

| 目录 | 用什么 | 权重格式 | 入口 |
|---|---|---|---|
| `lerobot/` | lerobot-rollout，直连 can0 从臂 | lerobot `pretrained_model/` | `run_act.sh` / `run_smolvla.sh` / `run_pi05_lora.sh` / `run_xvla.sh`（公共逻辑在 `_common.sh`） |
| `openpi/` | openpi / JAX policy server | openpi checkpoint（JAX，**不要**转成 lerobot 格式） | `serve_policy.py` + `run_rollout.sh` |
| `tools/` | 一次性诊断，不启动策略 | — | `capture_obs.py` / `query_chunk.py` / `watch_state.py` |

两个 pi05 不是一回事：`lerobot/run_pi05_lora.sh` 跑的是 lerobot LoRA adapter；
`openpi/serve_policy.py` 跑的是 openpi 全参 checkpoint。

## lerobot 路线（本机 4090 + can0 直连从臂）

    cd self_scripts/b601_single/inference/lerobot
    bash run_act.sh                # ACT：默认 100k 权重
    bash run_act.sh a800           # 8×A800 训练 50k（462 集），不限时长
    bash run_act.sh A              # A/B：全量 462 集 / 50k，600 秒
    bash run_act.sh B              # A/B：前 330 集 / 35.5k，600 秒
    bash run_smolvla.sh            # SmolVLA 90k
    bash run_pi05_lora.sh          # π0.5 LoRA，默认 r64 run 的 040000
    bash run_pi05_lora.sh 038000   # 同一个 run 的别的步数
    CKPT_ROOT=/home/kf/LX/pai0/<别的 run>/checkpoints bash run_pi05_lora.sh <步数>
    bash run_xvla.sh               # X-VLA

几个脚本依次做：环境激活 → 权重/包装器检查 → CAN/相机/校准检查 → 参数总览 → 人工确认 → `lerobot-rollout --strategy.type=base`。
共享的硬件端口、相机路径、任务文本、`joint_directions`、日志目录全部集中在 [`lerobot/_common.sh`](./lerobot/_common.sh)，
改硬件只改这一处。日志写在 `self_scripts/_logs/inference_b601_*.log`。

## π0.5 全参（openpi / JAX）

两个模型（JAX / openpi 格式，不是 lerobot 格式，**不需要也不建议转换**）：

| 短名 | 训练数据 | 步数 | 离线动作 MAE（训练分布内 / 未见过的 episode） |
|---|---|---|---|
| `330` | 前 330 集（ep 0–329） | 24,653 | 1.10° ／ **7.64°** |
| `462` | 全 462 集 | 34,750 | 1.15° ／ 1.12°（但那 13 集它训练时见过，属记忆） |

两个模型在**训练分布内**都很好（≈1.1°），但 `330` 对没见过的 episode 基本失效（7.64°）。所以真机验证**不要期待零样本泛化**，重点看它在训练场景下能不能把动作序列走对。

z790 上实测：稳态推理 **103 ms/次**（10 步去噪 × 3 相机，4090 24G），与 A800 的 98.7ms 相当。

离线动作 MAE（不接机器人、只跑策略前向）：

    cd self_scripts/b601_single/inference/openpi
    bash run_offline.sh 330 /home/kf/LX/pai0/b601_data/b601_v21          # 默认 ep 0 40 120
    bash run_offline.sh 462 --anchors /home/kf/b601_pi05/anchors_6.npz   # 用锚点，不搬 22G 数据集

## 目录

    /home/kf/b601_pi05/                    ← 权重与 JAX 环境（不在本仓库）
    ├── checkpoints/330_24653/      params + assets（12G）
    ├── checkpoints/462_34750/      同上
    ├── qgf_openpi_env/            JAX 0.5.3 venv（Python 3.11，已修软链）
    ├── qgf_py311/                 CPython 3.11.16 运行时
    ├── openpi/                    openpi 代码
    ├── anchors_6.npz              6 个评测锚点（11M，不用搬 22G 数据集）
    ├── offline_330.json / offline_462.json   离线评测结果
    ├── start_serve.sh             起 policy server（后台常驻）
    └── logs/                      server 日志 + rollout CSV

    /home/kf/dev/lerobot/self_scripts/b601_single/inference/   ← 本目录
    ├── README.md
    ├── lerobot/                   lerobot-rollout 栈（lerobot 格式权重）
    │   ├── _common.sh             公共库：环境/CAN/相机/校准检查、参数总览、命令拼装
    │   ├── run_act.sh             ACT（别名 default / a800 / A / B，或任意权重路径）
    │   ├── run_smolvla.sh         SmolVLA 90k
    │   ├── run_pi05_lora.sh       π0.5 LoRA（CKPT_ROOT + 步数）
    │   └── run_xvla.sh            X-VLA（Phase II 微调权重，sync 推理）
    ├── openpi/                    π0.5 全参（JAX，非 lerobot 格式）
    │   ├── env.sh                 环境变量 + B601_CKPT 短名表
    │   ├── serve_policy.py        起 websocket policy server
    │   ├── run_rollout.py         真机主循环（30Hz、每 5 步重观测、25° 限幅、CSV）
    │   ├── run_rollout.sh         主循环启动器
    │   ├── ws_client.py           客户端（含 --self-test 自检）
    │   ├── eval_offline.py        离线 MAE 实现（与 A800 完全同口径）
    │   ├── run_offline.sh         离线 MAE 启动器（330 / 462）
    │   └── infer_once.py          单帧观测 → 50 步动作块
    └── tools/                     一次性诊断
        ├── capture_obs.py         抓当前场景一帧观测存 npz（不动电机）
        ├── query_chunk.py         把一份观测发给 server，看 50 步动作块结构
        └── watch_state.py         只读关节角，核对符号/单位

## 路线 A：π0.5 全参真机验证（server + 控制循环）

一个终端起 server（JAX 环境、占 GPU），另一个终端跑控制循环（lerobot 环境、管机器人）。

    # 终端 1：起 server（换 462 就传 462）
    bash /home/kf/b601_pi05/start_serve.sh 330
    # 看到 READY {...} 且 ss -ltn | grep 8000 有监听即就绪

    # 终端 2：先干跑 5 秒（不接机器人，验证时序）
    cd /home/kf/dev/lerobot/self_scripts/b601_single/inference/openpi
    bash run_rollout.sh 330 --dry-run --duration 5

    # 终端 2：真机（会先要求确认零位/安全）
    bash run_rollout.sh 330 --duration 60

已实测的干跑结果：6 秒 180 步 = 严格 30Hz，36 次推理，平均 102.5 ms，CSV 落在
`/home/kf/b601_pi05/logs/rollout_pi05_ws_*.csv`（cmd_* 与 state_* 并排，方便核对）。

控制循环关键参数：`--execute-steps 5`（每块执行 5 步后重观测）、`--max-step-deg 25`
（单步限幅，0 关闭）、`--fps 30`（须与训练一致）、`--task` 用训练那条长文本。

## 输入输出约定（重要）

- `state`：7 维、度，顺序取自机器人 `action_features`（不是手写）
- 图像：`front`/`hand`/`top`，客户端统一转 **uint8 HWC**——与训练侧 `_parse_image` 口径一致
- 输出：50 步 × 7 维动作块，单位度
- **动作是数据集原有的"负号约定"（action ≈ −state）**，直接送 `robot.send_action`，
  `send_action` 内部会乘 `joint_directions=-1`，不要自己再取反
- 机器人侧相机 key 保持 hand/front/top，**不要**套 lerobot pi05 的 `base_0_rgb` 重命名（那是 LoRA 那条链路的要求）

## 已知坑（都踩过了）

1. **搬运会截断大 .so**：这次 `libnccl.so.2` 只传了 12.6MB（源 273MB），导致 GPU 侧
   一编译就 SIGBUS（`memset` 报 Bus error）。判据：`sha256sum` 两侧比对；`gdb` 看故障地址
   落在哪个映射即可定位到具体 .so。改完 243/244 文件一致后 GPU 立刻正常。
   以后搬环境必须做全量哈希比对，不能只信 rsync 的"完成"。
2. **数据集必须 v2.1 结构**；v3.0 需先用 A800 上的 `b601_convert.py` 转换。
   本机 `/home/kf/LX/pai0/b601_data/` 那份是 v3.0 531 集，与 462 集干净版 episode 编号对不上。
3. **不要和 rerun/firefox 抢内存**：z790 swap 只有 1G 且常年占满（Committed_AS≈71G，
   CommitLimit≈34G）。第一次 GPU 编译会吃几个 G 主机内存，跑前先关掉占用大的窗口。
4. 首次推理有 8–9 秒 JIT 编译，之后 ~103ms；server 起来后再跑控制循环，别让控制循环等编译。
5. `--fps` 必须 30（训练数据 30Hz）；改 fps 会让 flow matching 时序错位。
