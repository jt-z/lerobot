# USB 摄像头带宽问题解决方案

## 问题诊断

当前所有三个摄像头都连接在 **USB 2.0 Bus 01 (480Mbps)** 上，导致带宽不足，`/dev/video4` 无法读取数据。

### 当前 USB 连接状态

```
Bus 01 (USB 2.0, 480Mbps) - 拥挤
├─ Port 11: /dev/video0 (icspring 左手) - 640x480@30fps YUYV
├─ Port 3.1: /dev/video2 (icspring 主摄像头) - 640x480@30fps MJPG
└─ Port 3.3: /dev/video4 (JYU2C-2083 右手) - 无法读取 ❌

Bus 02 (USB 3.0, 10Gbps) - 几乎空闲
└─ 仅有一个 Hub，无摄像头
```

### 带宽计算

USB 2.0 理论带宽：480 Mbps = 60 MB/s

单个摄像头带宽需求：
- YUYV 640x480@30fps ≈ 17.6 MB/s
- MJPG 640x480@30fps ≈ 2-5 MB/s (压缩)

**三个摄像头同时运行超出了 USB 2.0 总线的实际可用带宽。**

## 解决方案

### 方案 1：重新分配 USB 端口（推荐）

**将摄像头分配到不同的 USB 总线：**

1. **识别主板上的 USB 3.0 端口**（通常是蓝色）
   - 这些端口连接到 Bus 02 (USB 3.0)
   - 即使是 USB 2.0 摄像头，插在 USB 3.0 端口上也能获得独立带宽

2. **建议的连接方式：**
   ```
   USB 3.0 端口 (Bus 02):
   ├─ /dev/video4 (JYU2C-2083) - 移到这里 ✓
   └─ /dev/video2 (icspring 主摄像头) - 可选
   
   USB 2.0 端口 (Bus 01):
   └─ /dev/video0 (icspring 左手)
   ```

3. **操作步骤：**
   - 拔掉 `/dev/video4` 的摄像头
   - 插入主板后面的 **蓝色 USB 3.0 端口**
   - 运行 `lsusb -t` 确认它现在在 Bus 02 上
   - 重新测试遥操作脚本

### 方案 2：使用两摄像头配置（当前可用）

如果无法重新分配端口，使用已经测试通过的版本：

```bash
./self_scripts/teleoperate_dual_so101_no_right_camera.sh
```

此版本使用：
- ✅ 左臂手部摄像头 (`/dev/video0`)
- ✅ 主摄像头顶部 (`/dev/video2`)

### 方案 3：降低分辨率/帧率（不推荐）

理论上可以降低所有摄像头的参数，但测试显示效果不佳。

## 验证步骤

重新分配 USB 端口后，运行以下命令验证：

```bash
# 1. 检查 USB 总线分布
lsusb -t

# 2. 确认摄像头在不同总线上
# 理想状态：至少一个摄像头在 Bus 02

# 3. 测试多摄像头读取
python3 ~/test_cameras.py

# 4. 运行完整的遥操作脚本
./self_scripts/teleoperate_dual_so101.sh
```

## 技术背景

- USB 2.0 理论带宽：480 Mbps
- USB 2.0 实际可用带宽：约 60-70% (考虑协议开销)
- USB 3.0 理论带宽：5 Gbps (10倍于 USB 2.0)
- UVC (USB Video Class) 摄像头即使是 USB 2.0 设备，插在 USB 3.0 端口上也能获得更好的带宽分配

## 相关文件

- 完整版遥操作脚本：`./self_scripts/teleoperate_dual_so101.sh`
- 两摄像头版本：`./self_scripts/teleoperate_dual_so101_no_right_camera.sh`
- 测试脚本：`~/test_cameras.py`
