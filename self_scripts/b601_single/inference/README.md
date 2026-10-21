# b601 单臂 π0.5 全参微调 —— z790 (4090) 推理验证包

两个模型（JAX / openpi 格式，不是 lerobot 格式，**不需要也不建议转换**）：

| 短名 | 训练数据 | 步数 | 离线动作 MAE（训练分布内 / 未见过的 episode） |
|---|---|---|---|
| `330` | 前 330 集（ep 0–329） | 24,653 | 1.10° ／ **7.64°** |
| `462` | 全 462 集 | 34,750 | 1.15° ／ 1.12°（但那 13 集它训练时见过，属记忆） |

两个模型在**训练分布内**都很好（≈1.1°），但 `330` 对没见过的 episode 基本失效（7.64°）。所以真机验证**不要期待零样本泛化**，重点看它在训练场景下能不能把动作序列走对。

z790 上实测：稳态推理 **103 ms/次**（10 步去噪 × 3 相机，4090 24G），与 A800 的 98.7ms 相当。

## 目录

    /home/kf/b601_pi05/
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
    ├── b601_pi05_env.sh           JAX 侧环境变量（server 用）
    ├── serve_policy.py            路线 A：把权重起成 websocket policy server
    ├── pi05_ws_client.py          路线 A：客户端（含 --self-test 自检）
    ├── run_pi05_ws_rollout.py     路线 A：真机主循环（30Hz、每 5 步重观测、25° 限幅、CSV）
    ├── run_pi05_ws_rollout.sh     路线 A：启动器
    ├── infer_offline.py           离线 MAE（与 A800 完全同口径）
    ├── infer_once.py              单帧观测 → 50 步动作块
    └── README.md

## 路线 A：真机验证（当前采用）

一个终端起 server（JAX 环境、占 GPU），另一个终端跑控制循环（lerobot 环境、管机器人）。

    # 终端 1：起 server（换 462 就传 462）
    bash /home/kf/b601_pi05/start_serve.sh 330
    # 看到 READY {...} 且 ss -ltn | grep 8000 有监听即就绪

    # 终端 2：先干跑 5 秒（不接机器人，验证时序）
    cd /home/kf/dev/lerobot/self_scripts/b601_single/inference
    bash run_pi05_ws_rollout.sh 330 --dry-run --duration 5

    # 终端 2：真机（会先要求确认零位/安全）
    bash run_pi05_ws_rollout.sh 330 --duration

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
