# 问题：推理启动时 CAN 应答超时

现象：`bash lx/pi05_inference.sh` 走到 `robot.connect()` 立刻退出，模型、相机都正常。

```
motorbridge.errors.CallError: disable_all failed: control ack timeout: comm_type=4
  File ".../seeed_b601_follower.py", line 181, in connect      self.configure()
  File ".../seeed_b601_follower.py", line 369, in configure    self.bus.disable_all()
```

发生记录：2026-11-11 17:19、2026-11-12 15:42 各一次，**两次都与从臂侧供电/接线有关**，与代码、模型无关。

## 一、先看总线证据

```bash
ip -details -statistics link show can0 | head -12
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
| **`RX packets = 0`** | 总线上没有任何电机报文——电机的 MCU 没在发帧 |
| `candump -T 3000 can0` 3 秒 0 帧 | 同上，可作二次确认 |

`disable_all` 等 ACK 超时只是结果；根因是**总线上没有节点应答**。

## 二、与「电机不在线（0/7）」的区别

[电机不在线排查](../02_数据采集/问题_电机不在线排查.md) 记录的是遥控/采集链路里**扫描不到电机**；本问题是同一物理层的另一种表现：

- 那次：发帧被 ACK（TEC=0），但 `rx_packets=0`、扫描全 `no reply`
- 本次：`RX=0` + 首帧控制指令就超时，启动即失败

两者排查路径一致：**先急停、再 48V 断电重启、再分配板、最后 CAN 线/终端电阻**。

## 三、按可能性排序

1. **从臂 48V 电源未开 / 急停被按下**（两次都是这个）——看电机指示灯是否亮
2. **CAN 线两端松动**（PCAN → 从臂底座），拔插一次
3. **电机被残留帧「喂住」**：上次异常退出（Ctrl+C 打断、异常崩溃）后控制器停在半状态，需断 48V 再上电
4. **CAN 线 A/B 接反 / 缺少 120R 终端电阻**（`ERROR-ACTIVE` 且能发帧但无回应的场景才怀疑）

## 四、已做的修复：启动前复位 can0

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

## 五、修好后的确认顺序

```bash
# 1. 上电后总线应开始有帧
timeout 3 candump -T 3000 can0 | head

# 2. 体检：应能看到 7 个电机 ID（1~7）
bash self_scripts/b601_common/07_preflight.sh

# 3. 再启动推理
bash lx/pi05_inference.sh
```

## 六、附带现象

- `[gripper 调试] 使能前原始角度: NA（失能无反馈，属正常；无法读取）`：正常，失能态读不到角度
- `terminate called without an active exception`（1~3 行）：异常后后台线程析构的连带打印，不是独立故障
