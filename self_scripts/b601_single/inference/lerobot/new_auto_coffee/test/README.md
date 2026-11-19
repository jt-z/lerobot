# coffee_expo 测试目录

给人和 AI 做固定回归，覆盖：

- 六子任务 CLI：`A1 / B1 / B2 / C1 / C2 / C3`
- 三种执行方式：`act / replay / pi05`
- replay：中间 episode、`rerun` 可视化、退出码
- 语音项目两阶段：饮品 → `A1 + B1/B2` → 自动 ARMED → 唤醒角度 → `C1/C2/C3`
- 安全：按键 `0/1`、`Ctrl+C`、语音停止、超时归零

相关路径：

| 内容 | 路径 |
|---|---|
| 子任务 CLI | `/home/jt/dev/lerobot/self_scripts/b601_single/inference/lerobot/new_auto_coffee` |
| 语音项目 | `/home/jt/dev/m260c-voice-assistant` |
| 语音主文件 | `/home/jt/dev/m260c-voice-assistant/demo/coffee_voice_trigger.py` |
| 数据集 | `/home/jt/b601_data/b601_pi05_coffee_expo_20261118_181510` |
| 运行环境 | `/home/jt/dev/miniforge3/envs/lerobot_b601/bin/python` |

## 快速开始

先跑离线自检（不动机器人）：

```bash
cd /home/jt/dev/lerobot/self_scripts/b601_single/inference/lerobot/new_auto_coffee/test
bash run_offline_checks.sh
```

然后按 [test_plan.md](test_plan.md) 从 `T0` 往后的顺序执行。会动机器人的用例必须有人在场。

## 给 AI 的硬规则

1. **🟢 安全用例可以自动执行**：`T0 / T1 / T2 / T11`。
2. **🟡 会动机器人的用例不要自动执行**：先打印命令和前置条件，等人类明确说“可以开始”再运行。
3. 同一时刻只允许一个机械臂任务；跑之前先检查：
   ```bash
   pgrep -af "subtask_supervisor.py|lerobot-rollout|lerobot-replay|run_subtask.sh"
   ```
   有输出就先停下，不要并发启动。
4. 每个动作用例结束后检查：
   - 机械臂回到零位
   - 上面的 `pgrep` 为空
   - 没有残留的 `gnome-terminal` 子任务窗口在跑
5. 失败时先看日志和退出码，不要连续重试同一个动作；把证据贴到结果记录里。
6. 不要用 `run.sh` 启动咖啡语音流程，它启动的是另一个 demo；用 `demo/coffee_voice_trigger.py`。

## 文件

| 文件 | 用途 |
|---|---|
| `test_plan.md` | 全部用例、命令、预期、排查 |
| `run_offline_checks.sh` | 一键离线回归（语法、任务表、CLI 契约、语音静态检查） |
| `check_voice_offline.py` | 语音调度层的离线断言（退出码映射、饮品解析、生成命令、shell 包装） |

## 当前已知状态（2026-11-19）

- replay 默认取该任务**中间 episode**：A1=159、B1=47、B2=77、C1=136、C2=98、C3=245，均 < 40s。
- replay 默认开启 `rerun`（三路相机）；无相机/无显示时设 `B601_REPLAY_RERUN=0`。
- replay 启动宽限为 10s，运动上限仍是 40s。
- ACT：A1 / B1 / B2 / C2 有权重；**C1 / C3 是占位**，会返回 `4`。
- 语音：启动问“要热水还是美式？”；阶段一 `A1 → B1/B2` 完成后自动 ARMED；RUNNING 时唤醒后说“停止”可中断。
- 语音主文件目前未提交。

## 结果记录模板

| 时间 | 用例 | 命令 | 退出码/结果 | 关键日志 | 备注 |
|---|---|---|---|---|---|
|  | T3 | `bash run_subtask.sh C2 --policy replay` |  |  |  |
|  | T5 | `coffee_voice_trigger.py --skip-phase1 ...` |  |  |  |
|  | T6 | `coffee_voice_trigger.py --policy replay ...` |  |  |  |

通过标准：用例的“预期”全部出现，退出码一致，机械臂归零，无残留进程。
