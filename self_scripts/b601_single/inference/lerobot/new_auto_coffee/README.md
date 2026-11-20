# new_auto_coffee —— coffee_expo 六技能调度与推理

数据源：`b601_pi05_coffee_expo_20261118_181510`（250 集 / 6 个任务）。
一个完整流程 = `A1 → B1/B2 → [人工确认萃取完成] → C1/C2/C3`，每个子任务可以独立选择
`act` / `replay` / `pi05` 三种执行方式，默认 `act`。

## 目录

| 文件 | 作用 |
|---|---|
| `coffee_task_table.sh` | 唯一任务表：prompt、ACT 权重、pi0.5 权重与 sha、replay episode 解析 |
| `run_coffee.sh` | 高层调度：`--drink hot_water\|americano --area 1\|2\|3` |
| `run_subtask.sh` | 单子任务入口（外部程序按顺序调用这个） |
| `subtask_supervisor.py` | 40s 超时、键 0/1、中止、归零等待、统一退出码 |
| `coffee_agent.py` | **常驻 agent**：机械臂 + 相机 + ACT/pi0.5 权重留在显存，socket 收指令 |
| `run_coffee_agent.sh` | 启动常驻 agent（`--self-test` 可离线验协议） |
| `agent_health.py` | 健康探测：健康退出 0，不可用退出 1（shell/语音共用） |
| `systemd/coffee-agent.service` | 用户级 systemd 单元：崩溃自动重启、开机自启 |
| `run_act_task.sh` | ACT 后端公共入口（读任务表） |
| `run_act_<ID>.sh` | 旧入口兼容层，等价于 `run_act_task.sh <ID>` |
| `run_pi05_coffee_250ep.sh` | pi0.5 后端入口：250 集 12k 权重 + 六任务 prompt 选择 |
| `run_act_coffee.sh` | 旧的单任务派发器，现在委托 `run_subtask.sh` |
| `_act_expo_common.sh` | ACT 推理参数与环境检查 |

## 怎么跑

```bash
cd ~/dev/lerobot/self_scripts/b601_single/inference/lerobot/new_auto_coffee

# 整杯流程（人工在旁边监控）
bash run_coffee.sh hot_water 2
bash run_coffee.sh americano 3 --policy act

# 只跑一个子任务（默认 act、40 秒上限）
bash run_subtask.sh A1
bash run_subtask.sh C2 --policy replay
bash run_subtask.sh B1 --policy pi05 --timeout 30

# 只看计划，不动机器人
bash run_coffee.sh hot_water 2 --dry-run
```

`run_coffee.sh` 会在开头做一次总确认；之后每个子任务用 `--yes` 跳过入口确认。
直接调 `run_subtask.sh` 时会在入口处按一次空格确认零位，加 `--yes` 可关闭；
非交互终端（外部程序调用）会自动跳过，请自行确保机械臂在零位。

## 子任务与 ACT 权重

| 任务 | 含义 | ACT 权重 | 状态 |
|---|---|---|---|
| A1 | 端空杯 → 咖啡机托盘 → 回零位 | `act_expo_A1_20261119/checkpoints/003100` | 已到位 |
| B1 | 选 Hot Water → Start → 回零位 | `act_expo_B1_20261119/checkpoints/002600` | 已到位 |
| B2 | 选 Americano → Start → 回零位 | `act_expo_B2_20261119/checkpoints/001800` | 已到位 |
| C1 | 满杯 → 杯位 1 → 回零位 | `act_expo_C1_6task/checkpoints/last` | 六任务重训占位 |
| C2 | 满杯 → 杯位 2 → 回零位 | `act_expo_C2_20261119/checkpoints/001800` | 已到位 |
| C3 | 满杯 → 杯位 3 → 回零位 | `act_expo_C3_6task/checkpoints/last` | 六任务重训占位 |

C1/C3 的占位路径在 `coffee_task_table.sh` 里。权重到位后二选一：

```bash
export B601_ACT_CKPT_C1=/path/to/pretrained_model   # 临时覆盖
# 或直接改 coffee_task_table.sh 里的默认路径
```

## 三种执行方式

| policy | 后端 | 说明 |
|---|---|---|
| `act`（默认） | 每个任务独立的 ACT 权重 | 走 `lerobot-rollout --strategy.type=base --inference.type=sync` |
| `replay` | 数据集 episode 原样回放 | 默认取该任务中间编号的 episode（B1 取中间集的前一集）；默认开启 rerun 可视化（三路相机），无相机/无显示时设 `B601_REPLAY_RERUN=0` |
| `pi05` | 单个 pi0.5 权重 + prompt | 默认 `$HOME/pi05_bf16/expo012000_250ep`（250 集 12k，六任务已训练）；换档走 `B601_PI05_MODEL` + `B601_PI05_EXPECT_SHA` |

replay 的 episode 默认取该任务所有 episode 的**中间编号**（中间批次通常更稳；B1 取中间集的前一集），
250 集版结果：A1=159、B1=46、B2=77、C1=136、C2=98、C3=245，时长都在 40 秒内。
需要指定某一条时用 `--episode N` 或 `B601_REPLAY_EPISODE_<ID>=N`。

### pi0.5 权重（250 集 run）

| 档位 | 目录 | sha256（前 16 位） | 说明 |
|---|---|---|---|
| 12k（默认） | `$HOME/pi05_bf16/expo012000_250ep` | `f70669263aed4ae1` | 6/6 任务已训练；同锚点对比 9k，5/6 任务更优，均值 2.416→2.319° |
| 9k | `$HOME/pi05_bf16/expo009000_250ep` | `a9803454d9746c94` | 6/6 任务已训练，上一档 |

⚠️ 带 `_250ep` 的是 250 集 run；102 集旧 run 的 `expo008000` / `expo012000`（无后缀）只训了 4 个任务，换档时别混。

直接用后端脚本（不经监督器，长时程调参时用）：

```bash
TASK=A1 DURATION=180 bash run_pi05_coffee_250ep.sh
MODEL_PATH=$HOME/pi05_bf16/expo009000_250ep \
  EXPECT_SHA256=a9803454d9746c94786b47b47dc54da53922fd75d60964e28518935e45ffe9db \
  TASK=C3 DURATION=120 bash run_pi05_coffee_250ep.sh
```

## 常驻 agent（Phase 1，推荐）

不起常驻服务时，每个子任务都要重新导入 lerobot（6–7s）、开三路相机（5s）、连 CAN（1–2s）、
加载权重（0.6–2s）。常驻 agent 把这些做一次，之后每个子任务只走 unix socket：

```bash
# 终端 A：起服务（机械臂 + 相机 + 权重常驻）
bash run_coffee_agent.sh

# 终端 B：照旧按子任务调用，会自动发现 socket 并改走 agent
bash run_subtask.sh A1 --policy act --yes
bash run_subtask.sh C2 --policy pi05 --yes
```

- 权重**懒加载**：第一次用到某个 ACT/pi0.5 才占显存，之后常驻、不回收
  （这张卡专职推理，不跑训练）。
- `PRELOAD=all bash run_coffee_agent.sh`：启动即加载 6 份 ACT + pi0.5（约 8.2GB），
  上机前先看 `nvidia-smi` 空闲 ≥ 9GB。
- `DISPLAY_DATA=0`：关 Rerun；`B601_AGENT_SOCKET=/path/x.sock`：换 socket 路径。
- socket 不在时 `run_subtask.sh` **自动回退**为原来的子进程方式；`--no-agent` 可强制回退。
- `replay` 也走 agent：agent 内直接按数据集 fps 重放录制动作（不加载权重、不建 inference engine），
  放完或被打断后同样慢速回零；agent 不在时才回退 `03_replay.sh` 子进程。
  适合演示时用回放兜底，同样省掉 14s 启动。

协议（NDJSON，一行一个 JSON；默认 socket `~/dev/lerobot/self_scripts/_logs/coffee_agent.sock`）：

```bash
{"op":"health"}
{"op":"status"}
{"op":"run","task":"A1","policy":"act","timeout":40,"prompt":"<与训练逐字一致>","ckpt":"/path/to/pretrained_model"}
{"op":"run","task":"C1","policy":"replay","timeout":40,"dataset":"/path/to/dataset","episode":12}
{"op":"switch","task":"A1","policy":"pi05","timeout":40,"prompt":"...","ckpt":"/path/to/model"}
{"op":"zero"}
{"op":"stop","reason":"ok|operator_fail|timeout|aborted"}
{"op":"shutdown"}
# → {"event":"accepted"...} / {"event":"switching"...} / {"event":"zeroed"...}
#   / {"event":"log","line":"..."} / {"event":"done","result":"ok","exit_code":0,...}
```

### 运行中热切换（q/w/e）

`switch` 会先以 `stop(switch)` 结束当前会话并等待归零，再在同一机械臂/相机上启动新策略；
旧会话的 `done` 被抑制，最终结果属于新策略。`zero` 只停并归零，不启动新策略。

语音程序 `demo/coffee_voice_trigger.py` 在主终端按：

| 键 | 动作 |
|---|---|
| `q` | 当前子任务切到 pi0.5 |
| `w` | 当前子任务切回 ACT |
| `e` | 当前子任务切到 replay（归零后回放该段） |

按键走 supervisor 的本地控制 socket（默认 `~/dev/lerobot/self_scripts/_logs/task_control.sock`，
可用 `B601_TASK_CONTROL_SOCKET` 覆盖）。切换前会检查目标权重/数据集；不可用时只回错误，
**不会打断当前任务**。replay 是重放该任务的数据集 episode，不是从当前帧继续。

退出码、按键 0/1、归零、`subtask_last.json` 契约全部不变（状态文件多一个 `"mode":"agent"`）。
agent 自身日志写到 `~/dev/lerobot/self_scripts/_logs/coffee_agent_<时间>.log`。

离线自检（不碰机械臂/相机/权重）：

```bash
bash run_coffee_agent.sh --self-test
# 另开一个终端
python - <<'PY'
import json, socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect("/home/jt/dev/lerobot/self_scripts/_logs/coffee_agent.sock")
s.sendall((json.dumps({"op": "health"}) + "\n").encode())
print(s.recv(4096).decode())
PY
```

### 健康探测

`run_subtask.sh` 不再只看 socket 文件是否存在，而是真发一条 `health`：

```bash
python agent_health.py                    # 打印 ok/busy/loaded/pid，健康退出 0，不可用退出 1
python agent_health.py --quiet            # 只给退出码（脚本用）
python agent_health.py --wait 15          # 最多等 15 秒，开机/部署后自检
```

- 探测失败 → `run_subtask.sh` 回退子进程（并在开头打印原因）。
- 探测健康但 `busy=true` → 仍然走 agent，由 agent 立刻返回 `busy` 错误（退出码 4），
  **不会**回退去抢相机。
- 语音端 `coffee_voice_trigger.py` 启动时也会探一次，日志里会写
  `[agent] 常驻服务在线/不可用`，不影响原有流程。

### 守护 / 开机自启（systemd 用户单元）

```bash
mkdir -p ~/.config/systemd/user
cp systemd/coffee-agent.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now coffee-agent
loginctl enable-linger "$USER"        # 未登录也随开机启动

systemctl --user status coffee-agent  # 看状态
journalctl --user -u coffee-agent -f  # 看日志
systemctl --user restart coffee-agent
```

- `Restart=on-failure` + `RestartSec=5`：崩溃/硬件初始化失败自动重启；
  120 秒内超过 5 次则停下，避免 CAN/相机没接好时无限重启。
- 正常 `shutdown`（退出码 0）不会重启。
- 需要 Rerun 时按单元文件里的注释打开 `DISPLAY_DATA=1` 与 `DISPLAY`，并在图形会话执行一次
  `systemctl --user import-environment DISPLAY`。
- 手动运行 `run_coffee_agent.sh` 与 systemd 二选一，别同时起（socket 会互相占用）。

⚠️ 真机项尚未验证：ACT 与 pi0.5 交替调用、切换任务是否零等待、每段结束是否回零位、
相机是否被第二个进程抢占。首次上机建议先用 `--no-agent` 跑一遍作对照。

## 超时、按键与归零

每个子任务默认最长 **40 秒**（`--timeout` 可改）：

- **键 0**：操作员判定本子任务成功 → 提前结束 → 慢速归零 → 退出码 `0`
- **键 1**：操作员判定失败 → 提前结束 → 慢速归零 → 退出码 `2`
- **Ctrl+C / SIGTERM**：中止 → 同样先归零 → 退出码 `130`
- **到 40 秒仍未按键**：ACT/pi0.5 视为超时（`3`）；replay 正常放完则算成功（`0`）

归零由后端自己完成，监督器只负责发信号/指令并等待：

- agent 模式：监督器发 `stop(reason)`，agent 结束控制环后同样走
  `return_to_initial_position`（约 3 秒插值）；机械臂和相机不真正 disconnect，
  下一个子任务继续用。
- ACT / pi0.5 子进程：`lerobot-rollout --duration=40` 到点退出，`return_to_initial_position`
  （约 3 秒插值）后 `disconnect()` 再走 `safe_zero()`。
- replay：监督器在超时后发 SIGINT，`06_replay_rerun.py` 的 `finally` 里断开并 `safe_zero()`。

键盘监听复用 `lerobot.utils.keyboard_input`；没有 TTY/显示器时自动降级为“只能等超时或 Ctrl+C”。

## 外部程序怎么调用

按子任务依次调用 `run_subtask.sh`，只认退出码：

```bash
bash run_subtask.sh A1 --yes || exit $?
bash run_subtask.sh B1 --yes || exit $?
# 外部自己控制萃取等待时间
bash run_subtask.sh C2 --yes || exit $?
```

退出码是稳定契约：

| 退出码 | 含义 |
|---|---|
| 0 | 成功（键 0，或 replay 正常放完） |
| 2 | 操作员判定失败（键 1） |
| 3 | 超时（跑满上限仍无人确认） |
| 4 | 后端/环境/权重错误 |
| 130 | 人为中止（Ctrl+C / SIGTERM） |

最后一次结果还会写到 `~/dev/lerobot/self_scripts/_logs/subtask_last.json`：

```json
{"task":"C2","policy":"act","result":"ok","exit_code":0,"trigger":"key_0","elapsed_s":18.4,"timeout_s":40}
```

可用 `--status-file` 改路径，方便外部程序读取。

## 口径与坑

1. prompt 必须与数据集 `meta/tasks.parquet` 逐字一致，统一放在 `coffee_task_table.sh`。
2. ACT 是整模型（不是 LoRA adapter），不要传 `--rename_map`；pi0.5 才需要相机 key 映射。
3. ACT 只能 `sync`，不能用 RTC。
4. 相机序列号来自 `b601_common/devices/<host>.sh`，换相机改档案，不要改这里。
5. replay 默认开启 rerun（`B601_REPLAY_RERUN=1`），会读三路相机并弹出 rerun 窗口；
   无相机或纯 SSH 无显示时设 `B601_REPLAY_RERUN=0`，退回原生 `lerobot-replay`。
6. 急停仍然用物理急停；键 0/1 只用于“人判断任务完成/失败”。

## 验证记录

- 2026-11-19：ACT A1/B1/B2/C2 权重落地，`sha256sum` 与训练端一致；`bash -n`、`py_compile` 通过。
- 2026-11-19：`run_coffee.sh --dry-run`、六任务 replay episode 自动解析、监督器超时/正常/错误退出码已离线验证。
- 2026-11-20：Phase 1 常驻 agent（`coffee_agent.py` + `run_coffee_agent.sh`）离线验证：
  socket 协议（health/run/stop/shutdown/单飞/坏参数）、supervisor 的 0/2/3/4/130 映射、
  ACT 与 pi0.5 的 `RolloutConfig` 装配；`run_offline_checks.sh` 32 项全过。
- 未做真机试跑（需要人在现场按空格 + 零位确认）；agent 的 ACT/pi0.5 交替调用也待真机确认。
