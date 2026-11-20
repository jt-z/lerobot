# coffee_expo 测试计划

标记说明：

- 🟢 安全：不动机器人，可自动执行
- 🟡 动机器人：必须人工在场、急停可达，AI 先给出命令并等待确认
- ⚪ 可选：耗时或依赖未就绪的权重

## 0. 通用前置

| 项 | 要求 |
|---|---|
| 机械臂 | 从臂在零位附近，周边清空，急停可用 |
| CAN | `can0` 已启动（1Mbps） |
| 相机 | hand / front / top 三路在线（replay 默认开 rerun） |
| 麦克风 | 降噪板串口可识别；必要时 `--port /dev/ttyACM0` |
| 环境 | `/home/jt/dev/miniforge3/envs/lerobot_b601/bin/python` |
| 显示 | 本地图形会话（会弹 gnome-terminal / rerun） |
| 数据集 | `/home/jt/b601_data/b601_pi05_coffee_expo_20261118_181510` |

通用变量：

```bash
NEW_AUTO=/home/jt/dev/lerobot/self_scripts/b601_single/inference/lerobot/new_auto_coffee
VOICE=/home/jt/dev/m260c-voice-assistant
PY=/home/jt/dev/miniforge3/envs/lerobot_b601/bin/python
```

跑任何动作前先确认没有残留进程：

```bash
pgrep -af "subtask_supervisor.py|lerobot-rollout|lerobot-replay|run_subtask.sh"
```

---

## T0 离线自检 🟢

**目的**：不动机器人，确认脚本语法、任务表、CLI 契约、语音静态逻辑。

```bash
cd "$NEW_AUTO/test"
bash run_offline_checks.sh
```

**预期**：

- 每一项 `[PASS]`
- 最后 `ALL OFFLINE CHECKS PASSED`
- 退出码 `0`

**失败处理**：先不要进入 T3 之后的动作测试，把第一个 `[FAIL]` 贴出来。

---

## T0b 常驻 agent 协议 🟢（离线）

**目的**：不碰机械臂/相机/权重，只验证 socket 协议与 supervisor 的结果映射。

```bash
cd "$NEW_AUTO"
PY=~/dev/miniforge3/envs/lerobot_b601/bin/python
SOCK=/tmp/coffee_agent_test.sock
$PY coffee_agent.py --socket "$SOCK" --self-test --display-data 0 --log-file /tmp/agent_test.log &
AGENT=$!
sleep 1

python - "$SOCK" <<'PY'
import json, socket, sys, time
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(sys.argv[1])
def send(o): s.sendall((json.dumps(o) + "\n").encode())
def recv():
    buf = b""
    while b"\n" not in buf: buf += s.recv(65536)
    return json.loads(buf.split(b"\n")[0])
send({"op": "health"}); print(recv())
send({"op": "run", "task": "A1", "policy": "act", "timeout": 2, "prompt": "p", "ckpt": "/tmp/fake"})
print(recv()); time.sleep(0.1)
send({"op": "stop", "reason": "ok"}); print(recv()); print(recv())
send({"op": "run", "task": "B1", "policy": "act", "timeout": 0.3, "prompt": "p", "ckpt": "/tmp/fake"})
print(recv()); print(recv())          # accepted → done(timeout)
send({"op": "shutdown"}); print(recv())
PY
wait $AGENT; echo "agent rc=$?"

# 监督器侧（超时 / 连不上 / 键 0 / 键 1）由 run_offline_checks.sh 的 agent 用例自动跑，
# 不需要人手按键；单独跑：
bash test/run_offline_checks.sh
```

**预期**：

| 操作 | 期望 |
|---|---|
| `health` | `ok=true, busy=false, self_test=true` |
| `run` + `stop(ok)` | `accepted` → `stop_ack(accepted=true)` → `done(result=ok, exit_code=0, trigger=key_0)` |
| `run` 不 stop（timeout=0.3） | `done(result=timeout, exit_code=3)` |
| `run` policy=replay 自然结束 | `done(result=ok, exit_code=0)` |
| 并发第二个 `run` | `error(busy)` |
| `stop(reason=nope)` | `error(bad_request)` |
| `shutdown` | `bye`，进程 rc=0，socket 文件被删除 |
| `agent_health.py`（健康 / 无 socket） | `rc=0` 且输出 `ok=true` / `rc=1` |
| supervisor 连不上 socket | `rc=4`，提示先启动 agent |
| supervisor 超时 / SIGTERM | `rc=3` / `rc=130`，状态文件 `mode=agent` |

**失败处理**：任何一条不符先查 `coffee_agent.py --self-test` 的日志，再查 supervisor 是否把 `trigger` 映射错。

---

## T1 CLI 契约 🟢

**目的**：确认错误输入不会启动机器人。

```bash
cd "$NEW_AUTO"

bash run_subtask.sh X9;                          echo "rc=$?"
bash run_subtask.sh A1 --policy nope;            echo "rc=$?"
bash run_subtask.sh;                             echo "rc=$?"
bash run_coffee.sh;                              echo "rc=$?"
bash run_coffee.sh hot_water 9;                  echo "rc=$?"
```

**预期**：

- 未知任务：`❌ 未知子任务：X9`，`rc=4`
- 未知策略：`❌ 未知策略：nope`，`rc=4`
- 缺参数：打印用法/错误，`rc=4`
- 区域 9：`❌ 未指定放置区域`，`rc=4`
- 全程不出现 `lerobot-rollout` / `subtask_supervisor.py`

---

## T2 任务表与中间 episode 🟢

**目的**：确认 replay 默认取中间 episode。

```bash
cd "$NEW_AUTO"
source ./coffee_task_table.sh
for t in A1 B1 B2 C1 C2 C3; do
  printf "%s prompt=%s episode=%s\n" \
    "$t" "$(coffee_task_prompt "$t")" "$(coffee_task_replay_episode "$t")"
done
```

**预期**：

| 任务 | 默认 episode | 时长 |
|---|---|---|
| A1 | 159 | 31.7s |
| B1 | 47 | 23.0s |
| B2 | 77 | 27.0s |
| C1 | 136 | 21.5s |
| C2 | 98 | 31.1s |
| C3 | 245 | 26.9s |

如果数据集换了，episode 会变；此时更新本表和 `README.md`，不要当成脚本 bug。

---

## T3 单子任务 replay（C2）🟡

**目的**：确认机器人、CAN、rerun、replay 单步链路。

```bash
cd "$NEW_AUTO"
bash run_subtask.sh C2 --policy replay
```

**前置**：从臂在零位，三路相机在线。

**预期**：

- 打印 `episode=98`、`rerun 显示：开启`
- 弹出 rerun 窗口
- 机械臂回放，结束后打印 `[SUBTASK] C2 RESULT=ok RC=0`
- 退出码 `0`

**证据**：

```bash
cat /home/jt/dev/lerobot/self_scripts/_logs/subtask_last.json
```

应包含 `"task": "C2"`、`"policy": "replay"`、`"result": "ok"`、`"exit_code": 0`。

**失败排查**：

| 现象 | 处理 |
|---|---|
| 没有 rerun 窗口 | 相机或显示问题；先 `B601_REPLAY_RERUN=0 ...` 验证纯 replay |
| 一开始就猛甩 | 从臂不在该 episode 首帧附近；回零后重试 |
| `RESULT=timeout` | episode 可能超 40s；确认 episode=98 |
| 没有 CAN | `candump can0` / 检查 `can0` 状态 |

---

## T4 整杯 replay（美式 → 2 区）🟡

**目的**：确认 A1 → B2 → 人工确认 → C2 的顺序和退出码。

先看计划（🟢）：

```bash
cd "$NEW_AUTO"
bash run_coffee.sh americano 2 --dry-run
```

**预期**：打印 `A1 → B2 → C2`，`rc=0`，不动机器人。

再执行：

```bash
bash run_coffee.sh americano 2 --policy replay
```

**预期**：

1. A1（episode 159，约 32s）→ `RC=0`
2. B2（episode 77，约 27s）→ `RC=0`
3. 停在人工确认，按空格后 C2（episode 98，约 31s）→ `RC=0`
4. 任一步非 0 立即停止后续步骤

**判定**：三步都 0，且咖啡机在 B2 时真的被按下 Start。

---

## T5 语音：角度 → C2 replay 🟡

**目的**：隔离验证麦克风、唤醒词、角度分区、语音层调用 replay。

```bash
cd "$VOICE"
$PY demo/coffee_voice_trigger.py \
  --policy replay --drink americano --skip-phase1 --no-tts \
  2>&1 | tee /tmp/coffee_voice_t5.log
```

**人工动作**：

1. 启动后确认日志出现 `跳过 ACT 默认推理，直接进入第二阶段`
2. 走到 2 区（240°~300°）
3. 说唤醒词 **“你好宽宽”**

**预期日志**：

```text
[唤醒] 唤醒：角度 2xx°，beam=..，score=.. -> 区域二
[任务] 终端已打开...
[任务] 步骤 1/1：C2 ...
[任务] 结束标记：退出码 0 -> ok
区域二任务已完成
```

**判定**：只有 C2 被启动，没有 A1/B。

**失败排查**：

| 现象 | 处理 |
|---|---|
| 没有唤醒 | 麦克风串口；加 `--port`；确认唤醒词 |
| 角度不对 | 看日志角度，调整站位或加 `--outside nearest` |
| 没弹终端 | 本地图形会话/DISPLAY |
| 启动了错误区域 | 检查 `ZONE_RANGES` 与站位 |

---

## T6 语音完整两阶段 replay 🟡

**目的**：验证饮品 → 阶段一 → 自动 ARMED → 阶段二。

```bash
cd "$VOICE"
$PY demo/coffee_voice_trigger.py \
  --policy replay --drink americano \
  2>&1 | tee /tmp/coffee_voice_t6.log
```

**人工动作**：

1. 阶段一窗口弹出后按一次空格
2. 走到 2 区，说“你好宽宽”

**预期**：

1. 日志 `阶段一: A1 空杯上托盘 -> B2 启动美式（成功自动进入第二阶段）`
2. A1（约 32s）→ B2（约 27s），每步 `RESULT=ok`
3. 日志 `阶段一完成（A1 + 饮品），自动进入第二阶段`
4. 唤醒后只启动 C2，播报“区域二任务已完成”

**判定**：阶段一结束后不需要按 r；唤醒后只跑 C2；全程无残留进程。

---

## T7 语音饮品问答 🟡

**目的**：验证启动时“热水/美式”识别。

```bash
cd "$VOICE"
$PY demo/coffee_voice_trigger.py \
  --policy replay --drink ask --no-tts \
  2>&1 | tee /tmp/coffee_voice_t7.log
```

**人工动作**：看到日志 `要热水还是美式？` 后，在 3 秒内说“美式”；再跑一次说“热水”。

**预期**：

- 美式：`[饮品] 识别为 美式（原文：'...'）`，阶段一执行 `A1 → B2`
- 热水：`[饮品] 识别为 热水（原文：'...'）`，阶段一执行 `A1 → B1`

**失败排查**：识别不对时先看原文；TTS 开启时有 1.4s 防自听等待。仍不稳定就用 `--drink americano` 跳过问答。

---

## T8 安全与中断 🟡

**目的**：确认操作员能随时停止并归零。

在 T3 或 T6 的任务运行中依次验证：

| 操作 | 位置 | 预期 |
|---|---|---|
| 按 `0` | 子任务终端 | 提前成功，归零，进入下一步 |
| 按 `1` | 子任务终端 | 操作员判失败，归零，后续停止 |
| `Ctrl+C` | 子任务终端 | 中止，归零，退出码 130 |
| 按 `x` | 语音终端 | 向 `subtask_supervisor.py` 发 SIGINT，归零，播报“已中断” |
| 说“你好宽宽，停止” | 任务 RUNNING 时 | 录音检查停止词并中断 |
| 说“你好宽宽，停止” | ARMED 时 | 取消本次任务，不启动 |

**证据**：

```bash
cat /home/jt/dev/lerobot/self_scripts/_logs/subtask_last.json
pgrep -af "subtask_supervisor.py|lerobot-rollout|lerobot-replay"
```

**判定**：退出码/`result` 与操作一致，机械臂归零，无残留进程。

---

## T9 超时 🟡

**目的**：确认 40s 上限和超时归零。

用一个 66 秒的 episode（A1 第 0 集）配合 10 秒上限：

```bash
cd "$NEW_AUTO"
bash run_subtask.sh A1 --policy replay --episode 0 --timeout 10
echo "rc=$?"
```

**预期**：

- 约 15 秒内（10 + 5 宽限）中断
- 打印 `RESULT=timeout`，`rc=3`
- 机械臂归零

**失败排查**：如果直接跑满 66 秒，检查 `--timeout` 是否被正确传入。

---

## T10 ACT 🟡

**目的**：确认 ACT 与 replay 使用同一调度接口。

单步：

```bash
cd "$NEW_AUTO"
bash run_subtask.sh C2 --policy act
```

整杯：

```bash
bash run_coffee.sh americano 2 --policy act
```

**预期**：A1/B2/C2 正常，退出码 0；接口、按键、超时行为与 replay 一致。

**注意**：ACT 到 40s 仍未结束会返回 `3`，这是策略本身没完成，不一定是脚本问题。

---

## T11 C1/C3 占位检查 🟢

**目的**：确认未就绪权重会在动机器人之前被拦住。

```bash
cd "$NEW_AUTO"
bash run_subtask.sh C1 --policy act --yes
echo "rc=$?"
```

**预期**：

- 打印 `❌ ACT 权重未就绪：/home/jt/LX/pai0/act_expo_C1_6task/...`
- `rc=4`
- 不启动 `subtask_supervisor.py`，不动机器人

C3 同理。权重到位后，把 `B601_ACT_CKPT_C1/C3` 指向真实目录，再按 T10 验证。

---

## T12 pi0.5 冒烟 ⚪

**目的**：确认 pi0.5 也能通过同一 CLI 调用。

```bash
cd "$NEW_AUTO"
bash run_subtask.sh C2 --policy pi05 --timeout 60
```

**预期**：能加载 12k 权重（`expo012000_250ep`）并执行；退出码 0 或 3（超时）。

**注意**：每个子任务都会重新加载 8.7GB 权重，整杯会很慢；现在 6 个任务都在 12k 的训练集内，
但 C3 只有 10 集，效果仍是最弱的一项。

---

## T13 收尾检查 🟢

```bash
pgrep -af "subtask_supervisor.py|lerobot-rollout|lerobot-replay|run_subtask.sh"
ls -l /home/jt/dev/lerobot/self_scripts/_logs/subtask_last.json
```

**预期**：

- `pgrep` 无输出
- 机械臂在零位
- 没有遗留的子任务终端窗口还在等待空格

---

## 全流程通过标准

1. T0、T1、T2 全过
2. T3、T4 的 replay 全过
3. T5 能按角度只启动对应 C 任务
4. T6 阶段一完成自动 ARMED，阶段二只跑一次
5. T7 能识别热水/美式
6. T8 的停止手段都生效且归零
7. T9 超时返回 3
8. T10 的 ACT 能跑通已有权重的任务
9. T11 的 C1/C3 被正确拦截为 4
