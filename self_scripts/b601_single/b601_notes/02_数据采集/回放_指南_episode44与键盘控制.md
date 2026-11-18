# episode 回放与键盘控制（p / r / ← / → / v）

目的：把「回放某一份数据集的某一集」这件事做成一条固定命令，并在回放过程中支持暂停、按 2 秒粒度慢速倒退/前进、继续播放和开关画面。

适用机器：`kfzbox`（设备档案 `devices/kfzbox.sh`，从臂 `seeed_b601_rs_follower` / `can0`）。

## 一、为什么是 `b601_1` 的 episode 44

对话里说的 “task1” 对应仓库根目录的 `b601_1`，episode 44 属于它的唯一任务（按钮任务）：

| 数据集 | episode 44 的 task_index | 任务 | 长度 |
|---|---|---|---|
| `b601_1` | 0 | 取杯 → 接咖啡 → 按按钮 → 回零位（唯一任务） | 1434 帧 / 47.8s |
| `b601_2` | 1 | 放 2 号取杯区 | 660 帧 / 22.0s |
| `b601_mixed` | 0 | 同 `b601_1`（混合集里的任务 0） | 1434 帧 / 47.8s |

任务文本原文见 [本地数据集清单 §二](../01_数据集资产/数据集_清单_本地资产.md)，此处不重复。

容易踩的坑：同一个 `44` 在不同数据集里不是同一个任务。

- `b601_2` 三条任务的 episode 区间：区域 1 = 0~39、区域 2 = 40~79、区域 3 = 80~109，所以它的 ep44 是 `task_index=1`。
- `b601_mixed` 四条任务的区间：任务 0 = 0~109、任务 1 = 110~149、任务 2 = 150~189、任务 3 = 190~219。它的 ep44 是任务 0，任务 1 要看 ep110 之后。

## 二、入口命令

```bash
cd /home/jt/dev/lerobot

# 回放 b601_1 的 episode 44（默认带 rerun 画面 + 键盘控制）
bash lx/replay_task1_ep44.sh

# 回放同一数据集的其它集
bash lx/replay_task1_ep44.sh 45

# 纯回放：不读相机、不显示，也没有键盘控制
USE_RERUN=0 bash lx/replay_task1_ep44.sh
```

`lx/replay_task1_ep44.sh` 只是一层预设，实际执行链：

```
lx/replay_task1_ep44.sh
  └─ self_scripts/b601_single/03_replay.sh
       └─ self_scripts/b601_common/06_replay_rerun.py   # USE_RERUN=1（默认）
            └─ LeRobotDataset + make_robot_from_config + LeRobotReplay 工具链
```

脚本把两件事固定下来，避免每次手动传参：

| 变量 | 固定值 |
|---|---|
| `DATASET_ROOT` | 仓库根 `b601_1/`（`03_replay.sh` 自己的默认值还是 z790 的 `/home/kf/...`） |
| `EPISODE` | `44`（`03_replay.sh` 默认是 0） |

其余检查仍由 `03_replay.sh` 完成：设备档案 → CAN 复位 → 校准文件 → 打印首帧 state/action → 按 ENTER 确认 → 回放 → 退出 `safe_zero` 回零位。日志在 `self_scripts/_logs/replay_b601_*.log`。

## 三、键盘控制

焦点必须在**运行脚本的终端**（不是 rerun 窗口）。回放开始后：

| 键 | 行为 |
|---|---|
| `p` | 暂停。每个 tick 仍下发当前帧的目标角，机械臂保持当前姿态 |
| `r` | 从当前帧继续播放 |
| `←` / `→` | 暂停，并按正常回放速度**逐帧**倒退/前进 2 秒（30Hz 下 60 帧），走到目标帧后保持暂停；连按可继续累加 2 秒 |
| `v` | 开/关 rerun 画面刷新（不打断回放；窗口保留，关闭时只是停止刷新） |

拖动语义：

- 不是瞬移。按一次 `←` 会把目标帧设为 `当前帧 - 60`，之后每个 tick 走一帧，用 2 秒走完这 2 秒的轨迹。
- 拖动过程中再按方向键，会在当前目标上继续加减 60 帧，并自动裁剪到 `[0, num_frames-1]`。
- 按 `p` 会取消正在进行的拖动，停在当前帧；按 `r` 从当前帧恢复播放。
- 暂停和拖动时都在按数据集 fps 持续发送绝对目标角，机械臂不会因为“没发指令”而松掉。

## 四、实现方式

改动集中在 `self_scripts/b601_common/06_replay_rerun.py`：

1. 新增 `ReplayKeyboardController`（线程锁保护状态）：
   - `paused` / `target` / `idx` / `display_enabled` 四个状态。
   - `on_key()` 处理 `p` / `r` / `left` / `right` / `v`。
   - `next_action_index()` 返回本 tick 要下发的帧号；拖动时每 tick 只向目标移动一帧，播放时 `idx += 1`，到末尾返回 `None` 结束。
2. 回放主循环从 `for idx in range(dataset.num_frames)` 改成 `while True` + `next_action_index()`，仍然用 `CycleTimer(dataset.fps)` 的 `tick()/wait()` 保证 30Hz。
3. 键盘监听用 `lerobot.utils.keyboard_input.TerminalKeyListener`，只读当前终端。**没有**用 X11 全局钩子（`pynput`），避免在别的窗口按方向键误控机械臂。
4. `v` 的实现是切换 `display_enabled`：关闭时跳过 `log_visualization_data()`，保留 rerun 连接；再次打开直接恢复刷新。若启动参数是 `--display_data=false`，第一次按 `v` 才 `init_visualization()`。
5. 退出时依次 `keyboard.stop()` → `timer.log_run_summary()` → `robot.disconnect()` → `shutdown_visualization()`，`Ctrl+C` 也走同一 `finally`。

`USE_RERUN=0` 走的是原生 `lerobot-replay`，没有上述键盘控制；需要暂停/拖动时必须用默认的 rerun 路径。

## 五、安全注意事项

1. 回放下发的是**绝对目标角**。按 ENTER 前先把从臂摆到接近 episode 44 首帧（脚本会打印首帧 `state` / `action`），否则上电后会直接甩向首帧。
2. `←` 是沿着录制的动作**倒着走**，同样会真实驱动机械臂；第一次请在空旷姿态下短按验证方向。
3. `p` 只保持当前帧姿态，不是急停；真正的急停仍然是切断从臂 48V 电源。
4. `v` 只影响画面，不影响运动；关掉画面不能省掉机械臂动作。
5. 回放结束/`Ctrl+C` 后从臂会自己回零位再失能，退出时不要扶。

## 六、验证记录（2026-11-18）

未上真机执行，做了三项静态/逻辑验证：

```bash
bash -n lx/replay_task1_ep44.sh
/home/jt/dev/miniforge3/envs/lerobot_b601/bin/python -m py_compile \
  self_scripts/b601_common/06_replay_rerun.py
/home/jt/dev/miniforge3/envs/lerobot_b601/bin/python \
  self_scripts/b601_common/06_replay_rerun.py --help
```

并用 `ReplayKeyboardController(num_frames=200, fps=30)` 模拟按键，确认：

- 正常播放返回 `0,1,2...`；
- `p` 后连续返回同一帧；
- `←` 从 frame 3 用 60 个 tick 退到 frame 0 并停住；
- `r` 从 frame 0 恢复为 `0,1...`；
- `→` 从 frame 2 用 60 个 tick 前进到 frame 62；
- `v` 两次切换显示开关。

## 七、变更记录

**2026-11-18：新增 episode 44 回放入口与键盘控制**

- 新增 `lx/replay_task1_ep44.sh`：固定 `b601_1` + `EPISODE=44`，复用 `03_replay.sh` 的 CAN/校准/安全确认流程。
- `06_replay_rerun.py` 新增 `ReplayKeyboardController` 与 `while` 回放循环，支持 `p` / `r` / `←` / `→`（2s 慢速拖动）/ `v`（画面开关）。
- 键盘监听限定当前终端，不使用 X11 全局监听。
