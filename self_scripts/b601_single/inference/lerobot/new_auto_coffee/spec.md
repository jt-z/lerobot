# 常驻推理 Agent（Phase 1）规格说明

## 1. 目标

把 `new_auto_coffee` 的每个子任务从「起一个进程、加载权重、连相机/CAN」改成「`run_subtask.sh` 只发一条 unix socket 指令」。常驻进程持有机械臂、三路相机和策略权重，ACT 与 pi0.5 都常驻显存，消除单任务 14–16s 启动开销。

**不是**每帧 RPC 推理。控制环仍在 agent 进程内用 lerobot 的 `BaseStrategy` 跑，保证时序与现有脚本完全一致。

## 2. 现状与约束（已核实）

- 运行时：`/home/jt/dev/miniforge3/envs/lerobot_b601`，lerobot 0.6.2 源码在 `/home/jt/LX/pai0/lerobot`。
- `lerobot.rollout.context` 顶层导入了 `make_robot_from_config`，并使用模块级 `_load_pretrained_policy`；两者都可 monkeypatch。
- `build_rollout_context(cfg, shutdown_event)` 会：
  - 加载策略并 `.to(device).eval()`；
  - `make_robot_from_config(cfg.robot)` → `robot.connect()`；
  - 用 `robot.get_observation()` 里 `*.pos` 作为 `initial_position`；
  - 构建 `ThreadSafeRobot`、processor、inference engine。
- `BaseStrategy.run` 的退出条件：`shutdown_event.is_set()` 或 `cfg.duration` 到时。
- `BaseStrategy.teardown` 调 `_teardown_hardware`：`engine.stop()` → 平滑回 `initial_position` → `robot.disconnect()`。
- 机器人配置字段（插件 `lerobot_robot_seeed_b601`）：`port` / `id` / `can_adapter` / `cameras: dict[str, CameraConfig]` / `joint_directions` / `gravity_compensation`。
- 相机：`OpenCVCameraConfig(index_or_path=Path(dev), fps=30, width=640, height=480, fourcc="MJPG")`。
- ACT：`n_action_steps=100`，`SyncInferenceConfig(seam_max_step_deg=3, seam_jump_threshold_deg=12)`，无 rename_map。
- pi0.5：`use_peft=False`、`dtype=bfloat16`、`num_inference_steps=10`、`n_action_steps=50`，rename_map `front→base_0_rgb / hand→left_wrist_0_rgb / top→right_wrist_0_rgb`；12k 权重默认 `$HOME/pi05_bf16/expo012000_250ep`。
- pi0.5 必须走 `lx/tools/pi05_lite.py` 的 `_install_lite_load()`（12GB 卡，跳过 lm_head）。
- 对外契约不可破坏：退出码 `0/2/3/4/130`，`subtask_last.json` 字段，按键 `0/1`、Ctrl+C 语义，语音项目按退出码判断。
- 显存预算：ACT 51.6M/197MB×6 ≈ 1.18GB；pi0.5 ≈ 7.0GB；3060 12GB 空闲约 11GB。

## 3. 架构

```
run_subtask.sh ──(参数/退出码不变)──> subtask_supervisor.py ──NDJSON over unix socket──> coffee_agent.py
                                                                                         ├─ 常驻 robot+cameras
                                                                                         ├─ 策略缓存 ACT×N / pi05×1
                                                                                         └─ 每个子任务：build_rollout_context → BaseStrategy.run → teardown(归零)
```

- agent 是唯一持有 V4L2 相机与 CAN 的进程；supervisor 不再 spawn 推理进程。
- 每次子任务仍新建 context / processor / inference engine，但复用同一个 policy 对象与同一个 robot。
- robot 通过代理对象交给 lerobot：`connect/disconnect` 为 no-op，其余属性与调用透传，避免 rollout 结束时关掉相机和 CAN。
- 归零目标为固定零位：首次 A1 起点（操作员确认的零位）记录一次，之后每段 teardown 与 replay 结束都回到该位姿，避免逐段链式漂移；A1 尚未跑过时回退为本段起点。

## 4. 协议（unix socket，NDJSON，UTF-8）

默认路径：`$B601_AGENT_SOCKET`，否则 `$HOME/dev/lerobot/self_scripts/_logs/coffee_agent.sock`。权限 0600。

请求：

| op | 字段 | 说明 |
| --- | --- | --- |
| `health` | — | 返回 `ok`、`busy`、`loaded`（已缓存权重）、`pid`、`uptime_s` |
| `run` | `task`, `policy`(`act`\|`pi05`), `timeout`(秒), `prompt`, `ckpt` | 单飞；忙时返回 `error: busy` |
| `stop` | `reason`(`ok`\|`operator_fail`\|`timeout`\|`aborted`) | 置停止事件；当前 `run` 走完归零后返回 `done` |
| `shutdown` | — | 结束当前 run（`aborted`）后退出进程 |

响应事件：

- `{"event":"accepted","task":...,"policy":...}`
- `{"event":"log","line":"..."}`（可选，转发 rollout 日志）
- `{"event":"done","task":...,"result":"ok|operator_fail|timeout|error|aborted","exit_code":N,"elapsed_s":x,"trigger":"natural|key_0|key_1|timeout|signal"}`
- `{"event":"error","error":"busy|bad_request|internal","message":"..."}`

结果映射：`ok→0`、`operator_fail→2`、`timeout→3`、`error→4`、`aborted→130`。无人 `stop` 且跑满 `duration` 视为 `timeout`（与现有 ACT/pi0.5 语义一致）。

## 5. 文件与改动

### 5.1 新增 `coffee_agent.py`

职责：

1. `register_third_party_plugins()`；必要时加载 pi05 lite 包装器（`LITE_LOAD=1`，默认 1）。
2. 构建并 `connect()` 常驻 `SeeedB601RSFollower`（相机来自环境变量给出的设备路径；`joint_directions` 全 `-1.0`，方向与 `_common.sh` 一致）。
3. 启动时 patch：
   - `lerobot.rollout.context.make_robot_from_config` → 返回 `_PersistentRobotProxy(robot)`；
   - `lerobot.rollout.context._load_pretrained_policy` → 按 `(type, pretrained_path)` 缓存 policy 对象。
4. 监听 unix socket；每个连接一个读循环线程，`run` 在独立 worker 线程执行，`stop` 可在 run 期间到达。
5. `_do_run(req)`：校验参数 → 组装 `RolloutConfig` → `build_rollout_context` → `create_strategy` → `setup/run/teardown` → 映射结果 → 回 `done`。
6. `--self-test`：不连硬件、不加载权重，只验证协议/单飞/stop/超时（按 `min(timeout, 0.2)` 秒模拟）。
7. 信号 `SIGINT/SIGTERM`：中断当前 run（`aborted`）并退出；退出时 `robot.disconnect()` 一次、删除 socket 文件。

关键实现细节：

- `_PersistentRobotProxy`：`__getattr__` 透传；`connect()` 记录并返回；`disconnect()` 只记录不真断；`is_connected` 透传真值。
- 每次 run 使用独立的 `threading.Event` 作为 `shutdown_event`；`stop` 写入 reason 后 `set()`。
- `RolloutConfig` 的 `robot` 传常驻配置（`__post_init__` 需要 `robot.type`），实际实例化被 patch 接管。
- ACT 参数：`policy.n_action_steps=100`、`SyncInferenceConfig(seam_max_step_deg=3, seam_jump_threshold_deg=12)`。
- pi05 参数：`policy.use_peft=False`、`policy.dtype="bfloat16"`、`policy.num_inference_steps=10`、`rename_map=...`、`n_action_steps` 保持 50。
- `--preload`：`all` 或逗号列表（`act:A1,B1` / `pi05`），启动后异步预加载；默认懒加载（首次使用才占显存）。
- `--display-data`：默认 1，启动时 `init_visualization` 一次，退出时 `shutdown_visualization`；失败只告警。
- 日志：写 `--log-file`，同时通过连接转发 `log` 事件。

### 5.2 新增 `run_coffee_agent.sh`

- `source ../_common.sh` 取相机设备路径、CAN 口、校准 id；把 `B601_CAM_HAND_DEV/FRONT/TOP` 导出给 Python。
- 前置 `lerobot_b601/bin` 到 PATH；`exec python coffee_agent.py --socket ... --log-file ...`。
- 支持 `B601_AGENT_SOCKET`、`PRELOAD`、`DISPLAY_DATA` 覆盖；`--self-test` 透传。

### 5.3 修改 `subtask_supervisor.py`

- 新增 `--agent-socket`、`--agent-prompt`、`--agent-ckpt`。
- agent 模式：连接 socket → 发 `run` → 循环收 `log`/`done`；按键 0/1、信号、软/硬超时改为发 `stop` 并带 reason；`done` 到达后用其结果与退出码；写 `subtask_last.json`（新增 `"mode":"agent"`，其余字段不变）。
- 连接失败 → 退出码 4，并提示先用 `run_coffee_agent.sh` 启动。
- 保留原有 spawn 路径，行为不变。

### 5.4 修改 `run_subtask.sh`

- 新增 `--no-agent`；`B601_AGENT_SOCKET` 或默认 socket 可连接时走 agent，否则保持现有 spawn 回退。
- 仍由本脚本解析 prompt / 默认 ACT 权重 / pi05 权重，并作为 `run` 请求字段传给 supervisor（任务表仍只有 `coffee_task_table.sh` 一处）。
- `replay` 不支持 agent，始终走 spawn。
- 开跑前检查（权重文件存在等）保留。

### 5.5 文档与离线检查

- `README.md`：新增「常驻 agent（Phase 1）」一节（启动、协议、回退、显存占用、未验证项）。
- `test/test_plan.md`：新增 agent 用例（health / run / 键 0 / 键 1 / 超时 / busy / 回退）。
- `test/run_offline_checks.sh`：新增 `bash -n`、`py_compile`、`--self-test` 往返、stop 三种 reason、超时、busy。

## 6. 验收标准

1. `bash -n` 所有改动脚本通过；`python -m py_compile coffee_agent.py subtask_supervisor.py` 通过。
2. `bash test/run_offline_checks.sh` 全绿（含新增 agent 自检）。
3. `run_coffee_agent.sh --self-test` 起服务后：
   - `health` 返回 `ok`；
   - `run` → `done(result=ok)` 对 `stop(reason=ok)`；
   - `stop(reason=operator_fail)` → `exit_code=2`；
   - 不 stop 且 `timeout` 小 → `result=timeout`；
   - 并发第二个 `run` → `error busy`。
4. 非 agent 模式下 `run_subtask.sh` 的退出码与 `subtask_last.json` 与改动前一致（回退路径不变）。
5. 真机项（不在离线验收内，需现场确认）：ACT 与 pi05 交替调用、切换任务无启动等待、每个子任务结束回零位、相机不被抢占。

## 7. 风险与缓解

- **显存**：6×ACT + pi05 ≈ 8.2GB。默认懒加载，`--preload all` 前先 `nvidia-smi` 确认空闲 ≥ 9GB。
- **policy 复用与 config 漂移**：缓存键含 `type+path`，命中时同步 `n_action_steps`；只支持 sync 推理。
- **相机独占**：代理只拦截 `connect/disconnect`，不复制相机对象。
- **异常残留**：`_do_run` 用 `try/finally` 保证 teardown、清 busy、重置停止事件；teardown 失败也返回 `error`。
- **契约破坏**：agent 模式仅是 supervisor 的一种后端，退出码/状态文件/按键语义在 supervisor 层统一。
