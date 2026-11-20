# 常驻推理 Agent（Phase 1）任务清单

> 规格见 `spec.md`。按顺序执行；每完成一项立即更新状态。

## T1 新增 `coffee_agent.py`

- [x] 参数：`--socket`、`--log-file`、`--preload`、`--display-data`、`--self-test`、`--device`。
- [x] `register_third_party_plugins()`；`LITE_LOAD=1` 时 `import pi05_lite; _install_lite_load(); _install_timing()`。
- [x] `_PersistentRobotProxy`：`connect/disconnect` no-op，其余透传。
- [x] patch `lerobot.rollout.context.make_robot_from_config` 与 `_load_pretrained_policy`（按 `type+path` 缓存）。
- [x] unix socket NDJSON 服务：`health/run/stop/shutdown`，单飞，worker 线程 + 读循环。
- [x] `_do_run`：组装 `RolloutConfig`（ACT / pi05 两套参数）→ `build_rollout_context` → `BaseStrategy` 生命周期 → 结果映射。
- [x] `--self-test`：不碰硬件/权重，可被 `stop` 打断。
- [x] 信号处理、退出清理（断机械臂、删 socket）。

验收：`py_compile` 通过；`--self-test` 下 health/run/stop/timeout/busy 全部符合协议（已跑，全 PASS）。

## T2 新增 `run_coffee_agent.sh`

- [x] `source ../_common.sh`，导出 `B601_CAM_HAND_DEV/FRONT/TOP`、`B601_FOLLOWER_PORT`、`FOLLOWER_ID`。
- [x] PATH 前置 `lerobot_b601/bin`；`exec python coffee_agent.py ...`。
- [x] 支持 `B601_AGENT_SOCKET / PRELOAD / DISPLAY_DATA / LITE_LOAD` 覆盖与 `--self-test` 透传。

验收：`bash -n` 通过；`--self-test` 能起服务并生成 socket 文件（已跑）。

## T3 修改 `subtask_supervisor.py`

- [x] 新增 `--agent-socket/--agent-prompt/--agent-ckpt`。
- [x] agent 模式：连接、发 `run`、收 `log/done`；键 0/1、信号、软/硬超时发 `stop(reason)`。
- [x] 结果与退出码复用现有映射；状态 JSON 增加 `"mode":"agent"`，其余字段不变。
- [x] connect 失败 → 4；保留 spawn 路径完全不变。

验收：agent 模式下 0/2/3/4/130 已验证（含 SIGTERM=130、busy→4）。

## T4 修改 `run_subtask.sh`

- [x] 新增 `--no-agent`；`B601_AGENT_SOCKET` 或默认 socket 可连时走 agent。
- [x] 传 `--agent-prompt`、`--agent-ckpt`（act 权重或 pi05 权重目录）。
- [x] `replay` 固定走 spawn；agent 不可用时自动回退 spawn。
- [x] 文件头用法与说明更新。

验收：`bash -n` 通过；端到端 `run_subtask.sh A1` 在 self-test agent 下走 agent 并返回 3。

## T5 文档

- [x] `README.md` 增「常驻 agent（Phase 1）」：启动/停止、协议示例、回退、显存预算、真机待验证项。
- [x] `test/test_plan.md` 增 agent 用例（T0b）。

验收：文档命令可直接复制执行。

## T6 离线回归

- [x] `test/run_offline_checks.sh` 增加 agent 自检（`bash -n` / `py_compile` / `check_agent_offline.py`）。
- [x] 跑一遍 `bash test/run_offline_checks.sh`：PASS=34 FAIL=0 WARN=0。
- [x] `bash -n` 所有改动脚本。

验收：离线检查全过；真机项在 README 标注「未验证」。

## T7 健康探测（新增需求）

- [x] 新增 `agent_health.py`（`--socket/--timeout/--wait/--quiet/--json`，健康 0、不可用 1）。
- [x] `run_subtask.sh` 路由前真发 health；失败才回退子进程，busy 仍走 agent（不抢相机）。
- [x] 语音 `coffee_voice_trigger.py` 启动时探测并打印 `[agent] 在线/不可用`。
- [x] `check_agent_offline.py` / `run_offline_checks.sh` 覆盖健康与不可用两种情况。

## T8 守护 / 开机自启（新增需求）

- [x] `systemd/coffee-agent.service`：用户级单元，`Restart=on-failure` + `RestartSec=5`，
      120s/5 次重启上限，`SIGTERM` + `TimeoutStopSec=45`。
- [x] README 安装步骤（含 `loginctl enable-linger`、Rerun 的 DISPLAY 说明）。
- [x] 离线检查校验单元文件的 ExecStart/Restart/WantedBy。

## T9 replay 接入 agent

- [x] `coffee_agent.py` 支持 `policy=replay`：校验 `dataset` + `episode`，按数据集 fps 逐帧重放，
      复用常驻机械臂/相机（不加载权重、不建 inference engine）。
- [x] 放完自然结束 = `ok`（与子进程 `--clean-exit-result ok` 一致）；超时 → `timeout`；键 0/1、Ctrl+C 照旧。
- [x] replay 结束用 `_smooth_home` 慢速回零（代理屏蔽了 `disconnect`，必须自己归零）。
- [x] supervisor 增加 `--agent-dataset/--agent-episode`；`run_subtask.sh` 的 replay 也走 agent，失败才回退。
- [x] 离线验证：replay 自然结束 → `ok/0`；数据侧 `select_columns(ACTION)` / fps / 动作名已用真实数据集核对
      （1979 帧 @30Hz，7 个关节键）。

## 明确不做

- 空闲卸载 / 显存回收：该卡专职推理，权重常驻。
- 训练与推理共用显卡时的保护：不适用于本机。
