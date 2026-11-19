# new_auto_coffee —— coffee_expo 六技能调度与推理

数据源：`b601_pi05_coffee_expo_20261118_181510`（250 集 / 6 个任务）。
一个完整流程 = `A1 → B1/B2 → [人工确认萃取完成] → C1/C2/C3`，每个子任务可以独立选择
`act` / `replay` / `pi05` 三种执行方式，默认 `act`。

## 目录

| 文件 | 作用 |
|---|---|
| `coffee_task_table.sh` | 唯一任务表：prompt、ACT 权重、replay episode、是否已训练 |
| `run_coffee.sh` | 高层调度：`--drink hot_water\|americano --area 1\|2\|3` |
| `run_subtask.sh` | 单子任务入口（外部程序按顺序调用这个） |
| `subtask_supervisor.py` | 40s 超时、键 0/1、中止、归零等待、统一退出码 |
| `run_act_task.sh` | ACT 后端公共入口（读任务表） |
| `run_act_<ID>.sh` | 旧入口兼容层，等价于 `run_act_task.sh <ID>` |
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
直接调 `run_subtask.sh` 时会在入口处按一次 ENTER 确认零位，加 `--yes` 可关闭；
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
| `replay` | 数据集 episode 原样回放 | 默认取该任务中间编号的 episode；默认开启 rerun 可视化（三路相机），无相机/无显示时设 `B601_REPLAY_RERUN=0` |
| `pi05` | 单个 pi0.5 权重 + prompt | 当前 `expo008000` 只训了 4 个任务；按占位逻辑允许强行跑 C1/C3（结果不可信），新权重到位后替换 `B601_PI05_MODEL` 即可 |

replay 的 episode 默认取该任务所有 episode 的**中间编号**（中间批次通常更稳），
250 集版结果：A1=159、B1=47、B2=77、C1=136、C2=98、C3=245，时长都在 40 秒内。
需要指定某一条时用 `--episode N` 或 `B601_REPLAY_EPISODE_<ID>=N`。

## 超时、按键与归零

每个子任务默认最长 **40 秒**（`--timeout` 可改）：

- **键 0**：操作员判定本子任务成功 → 提前结束 → 慢速归零 → 退出码 `0`
- **键 1**：操作员判定失败 → 提前结束 → 慢速归零 → 退出码 `2`
- **Ctrl+C / SIGTERM**：中止 → 同样先归零 → 退出码 `130`
- **到 40 秒仍未按键**：ACT/pi0.5 视为超时（`3`）；replay 正常放完则算成功（`0`）

归零由后端子进程自己完成，监督器只负责发信号并等待：

- ACT / pi0.5：`lerobot-rollout --duration=40` 到点退出，`return_to_initial_position`
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
- 未做真机试跑（需要人在现场按 ENTER + 零位确认）。
