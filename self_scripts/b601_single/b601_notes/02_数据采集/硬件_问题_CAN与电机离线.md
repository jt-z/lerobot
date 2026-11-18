# 问题：CAN 总线 / 电机离线排查

同一物理层的两类现场，排查路径一致：

| 现象 | 触发时机 | 关键证据 | 状态 |
|---|---|---|---|
| 电机 0/7 在线 | `01_teleop.sh` / `07_preflight.sh` 上机体检 | 扫描 id 1~7 全 `no reply`；发帧被 ACK（TEC=0）但 `rx_packets=0` | 2026-11-17，未闭环 |
| `disable_all failed: control ack timeout: comm_type=4` | `bash lx/pi05_inference.sh` 走到 `robot.connect()` | `can0` 为 `ERROR-ACTIVE` 但 `RX packets=0` | 2026-11-11 / 11-12 各一次，均为从臂供电/接线 |

推理启动时的完整报错：

```
motorbridge.errors.CallError: disable_all failed: control ack timeout: comm_type=4
  File ".../seeed_b601_follower.py", line 181, in connect      self.configure()
  File ".../seeed_b601_follower.py", line 369, in configure    self.bus.disable_all()
```

主机：`kfzbox`（RTX 3060 12GB），设备档案 `devices/kfzbox.sh`，环境 `~/dev/miniforge3/envs/lerobot_b601`。

## 一、先看总线证据

```bash
ip -details -statistics link show can0 | head -12
timeout 3 candump -T 3000 can0 | head
```

故障时典型输出：

```
can state ERROR-ACTIVE (berr-counter tx 0 rx 0)
bitrate 1000000 sample-point 0.750
RX:  bytes 0  packets 0
```

判读：

| 现象 | 含义 |
|---|---|
| `ERROR-ACTIVE` 且 error 计数全 0 | 适配器、驱动、波特率都正常，不是 bus-off |
| **`RX packets = 0`** | 总线上没有任何电机报文——电机 MCU 没在发帧 |
| `candump` 3 秒 0 帧 | 同上，二次确认 |
| 发帧（`cansend`）后 TX +1、**TEC 仍为 0** | 帧被 ACK 了 → 总线上至少有一个带电节点 |

`disable_all` 等 ACK 超时只是结果；根因是**总线上没有节点应答**。

电机 0/7 那次（2026-11-17）的完整证据：

| 检查 | 输出 | 推断 |
|---|---|---|
| `lsusb` | `0c72:000c PEAK-System PCAN-USB` | 适配器枚举正常 |
| 驱动 | `peak_usb` 已 attached，`can0` 存在 | 驱动正常 |
| 速率 | `bitrate 1000000`，sample-point 0.75 | 与电机一致 |
| 状态 | `ERROR-ACTIVE (berr-counter tx 0 rx 0)`，`bus-off=0`、`error-warn=0` | 从未出过错，不是线断/波特率错 |
| 监听 3s + `candump` 2s | `rx_packets = 0` | 总线上没有任何帧在发 |
| 主动发一帧 | `cansend can0 7FE#...`，TX 计数 +1，**TEC 仍为 0** | 帧被 ACK，至少一个节点带电 |
| 扫描 | `motorbridge-cli scan --start-id 1 --end-id 7`，逐个 `probe=0x01 no reply`（80ms 超时） | 电机一个都不回应 |
| 占用 / 其它口 | 无进程占用 can0；只有 can0 一个 CAN 接口 | 排除被别的程序或其它接口干扰 |

**关键判断**：如果 CAN 线断、A/B 接反、波特率不匹配，或整条臂完全没电，发帧时 TEC 会持续上涨并最终进 bus-off。现在 TEC 始终为 0、0 错误，说明物理层是通的、至少有一个节点带着电——问题不是「线断了」，而是**电机收到帧却不回应**。

## 二、先排除软件侧

| 检查项 | 结果 |
|---|---|
| 环境 | `lerobot_b601` 存在，py3.12 / torch 2.11 cu128 / lerobot 0.6.2，CUDA 可用 |
| 机器人插件 | `lerobot_robot_seeed_b601 1.0.3`，`seeed_b601_rs_follower` / `rebot_102_leader` 已注册 |
| 主臂 | `/dev/rebot_leader -> ttyUSB0`，7 个舵机都能 ping 到 |
| 相机 | 2609010 / 2608006 / 2610057 三路 by-id 都在 |
| 校准文件 | `seeed_b601_rs_follower/follower.json`、`rebot_102_leader/leader.json` 都在 |
| 图形 / sudo | `DISPLAY=:0` 可用；`sudo -n` 免密（脚本要 `sudo ip link`） |
| CLI 参数 | 用 draccus 离线解析 01 的完整参数，全部落地正确 |

结论：**脚本和环境都不是原因**，问题只在从臂这条 CAN 链路上。

## 三、按可能性排序

### 1. 急停按钮 / 从臂未上电（最符合现象）

如果急停只切电机驱动使能、不切 CAN 收发器供电，就会正好表现成「总线有 ACK、电机不响应」；推理那两次也是从臂 48V 没开/接线松。

- 旋出 / 拔出急停；
- 看电机指示灯与控制面板灯是否亮；
- 检查从臂 48V 电源是否打开。

### 2. 电机被残留帧「喂住」

`01_teleop.sh:108-112`、`02_record.sh:159-163` 都记录过这个坑：上次异常退出（Ctrl+C、异常崩溃）或急停断电时，PCAN 队列与电机侧可能残留半截指令帧，电机卡住后不再响应新帧，表现为 `control ack timeout: comm_type=4`。

- **48V 完全关掉，等 5~10 秒再开**；
- 不要热插拔 CAN 线；
- 再跑扫描确认。

### 3. 48V 分配中间断

「有一个节点 ACK、7 个电机全哑」也符合只有第一个电机（或某块分配板）有电的情况。

- 逐个关节看电机指示灯是否都亮；
- 检查 48V 菊花链 / 分配板的连接。

### 4. CAN 线 / 终端电阻（当前可能性最低）

- 断电后用万用表量 `CANH-CANL`，两端各一个 120Ω 并联应约 **60Ω**；
- 检查 PCAN DB9 与从臂端插头是否插到底；
- 确认总线上没有挂其它设备。

## 四、能直接定位的差分实验

把从臂 48V 电源关掉，再发一帧：

```bash
ip -details link show can0 | grep "can state"
cansend can0 7FE#0001020304050607
sleep 0.5
ip -details link show can0 | grep "can state"
```

- 若 TEC 上涨 / 进 bus-off → 之前的 ACK 就来自从臂，说明**线与供电都正常**，问题纯粹在电机状态（急停 / 喂住 / ID）。
- 若仍被 ACK → 总线上还有别的设备，先把它找出来（实验后需 `sudo ip link set can0 down && up` 复位，或在脚本里让它自动复位）。

## 五、已做的修复：启动前复位 can0

`02_record.sh` 一直有这一步，推理脚本原先缺失（只判断接口 `UP`），现已补齐（`lx/pi05_inference.sh` 的 CAN 段）：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
sleep 0.5
timeout 0.3 candump can0 >/dev/null 2>&1   # 读空残留帧
```

- 配置失败（被其他进程占用）直接 `exit 1`，不会等模型加载 20s 后才报错
- bitrate 不是 1Mbps 时打印警告
- 启动时会打印 `♻️ 复位 can0（1Mbps）并清空残留 CAN 帧...`

⚠️ 复位只能清理**主机侧**队列；如果电机本身没上电（`RX=0`），复位无效，必须查物理层。

## 六、修好后的确认顺序

```bash
# 1. 上电后总线应开始有帧
timeout 3 candump -T 3000 can0 | head

# 2. 体检：应能看到 7 个电机 ID（1~7）
bash self_scripts/b601_common/07_preflight.sh

# 3. 再启动推理 / 遥操作
bash lx/pi05_inference.sh
```

单独复测扫描：

```bash
cd /home/jt/dev/lerobot
timeout 40 ~/dev/miniforge3/envs/lerobot_b601/bin/motorbridge-cli scan \
  --vendor robstride --channel can0 --start-id 1 --end-id 7
```

看到 `scan done: 7 motor(s) found` 再启动 `01_teleop.sh` / `02_record.sh`。电机没上线时按 ENTER 也只会在 `connect()` 阶段以 `ack timeout` 失败。

## 七、附带现象

- `[gripper 调试] 使能前原始角度: NA（失能无反馈，属正常；无法读取）`：正常，失能态读不到角度
- `terminate called without an active exception`（1~3 行）：异常后后台线程析构的连带打印，不是独立故障

## 八、状态

- 2026-11-11 / 11-12 的推理启动超时：两次均由从臂供电/接线引起，已通过上电 + 启动前 can0 复位解决。
- 2026-11-17 的电机 0/7：软件侧已排除，**未闭环**，等待现场按急停 → 48V 断电重启 → 分配板 → CAN 线/终端电阻的顺序复测。
