# GPU故障处理

## 问题表现

### nvidia-smi输出

```
Unable to determine the device handle for GPU0: 0000:55:00.0: Unknown Error
Wed Aug  5 17:48:26 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|=========================================+========================+======================|
|   1  NVIDIA GeForce RTX 3090        Off |   00000000:56:00.0 Off |                  N/A |
|   2  NVIDIA GeForce RTX 3090        Off |   00000000:59:00.0 Off |                  N/A |
|   3  NVIDIA GeForce RTX 3090        Off |   00000000:5D:00.0 Off |                  N/A |
|   4  NVIDIA GeForce RTX 3090        Off |   00000000:88:00.0 Off |                  N/A |
|   5  NVIDIA GeForce RTX 3090        Off |   00000000:89:00.0 Off |                  N/A |
|   6  NVIDIA GeForce RTX 3090        Off |   00000000:8C:00.0 Off |                  N/A |
|   7  NVIDIA GeForce RTX 3090        Off |   00000000:90:00.0 Off |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

**关键发现**：
- GPU 0 (PCI 0000:55:00.0) **完全缺失**
- 只列出了GPU 1-7（7张卡）
- 所有可见GPU状态正常（温度33-36°C，功耗正常）

### PyTorch错误

```
/home/ksa/.conda/envs/lerobot/lib/python3.12/site-packages/torch/cuda/__init__.py:789: 
UserWarning: Can't initialize NVML

CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, 
e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. 
Setting the available devices to be zero.

WARNING:lerobot.configs.policies:Device 'cuda' is not available. Switching to 'cpu'.
```

**影响**：
- 所有8个训练进程都无法访问CUDA
- 进程退回到CPU模式
- 配置显示 `'device': 'cpu'`

---

## 问题诊断

### 可能的原因

| 原因 | 可能性 | 依据 |
|------|-------|------|
| **CUDA kernel崩溃后遗症** | ⭐⭐⭐ 高 | 发生在NaN导致的CUDA错误之后 |
| **GPU硬件故障** | ⭐⭐ 中 | GPU 0单独失效，其他GPU正常 |
| **PCIe连接问题** | ⭐⭐ 中 | 物理连接松动或故障 |
| **驱动状态异常** | ⭐⭐⭐ 高 | 驱动进入不一致状态 |
| **供电问题** | ⭐ 低 | 单卡问题，不太可能是供电 |
| **温度过高** | ✗ 否 | 所有GPU温度正常 |

### 时间线关联

```
17:34:57 - NaN导致CUDA崩溃
           ↓
17:46:00 - 重启训练，发现GPU 0失效
```

**相隔约11分钟**，期间可能发生：
- CUDA驱动状态未完全恢复
- GPU 0进入错误保护模式
- PCI总线重新枚举失败

---

## 立即应对方案

### 方案1：绕过故障GPU（推荐）⭐

**使用CUDA_VISIBLE_DEVICES排除GPU 0**：

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes=7 \
    $(which lerobot-train) --config_path=train_config_act_stable.json
```

**优点**：
- 立即可用，无需等待
- 7张3090仍有强大算力
- 不依赖管理员权限

**缺点**：
- 有效批次从128降至112
- 损失12.5%的计算资源

**启动脚本**：
```bash
cd /home/ksa/lerobot/self_scripts
bash start_train_act_stable_7gpu.sh
```

---

### 方案2：重置NVIDIA驱动

**步骤1：杀死所有GPU进程**
```bash
pkill -9 -f python
```

**步骤2：重置GPU**
```bash
sudo nvidia-smi --gpu-reset
# 或针对GPU 0
sudo nvidia-smi -i 0 -r
```

**步骤3：验证GPU恢复**
```bash
nvidia-smi
```

**步骤4：测试PyTorch是否能访问**
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

**优点**：
- 如果成功，恢复所有8卡
- 不需要重启服务器

**缺点**：
- 需要sudo权限
- 不保证成功（如果是硬件问题）

**辅助脚本**（reset_gpu_and_train.sh）：
```bash
#!/bin/bash
echo "1. 杀死所有Python进程..."
pkill -9 -f python
sleep 3

echo "2. 重置NVIDIA驱动..."
sudo nvidia-smi --gpu-reset

echo "3. 验证GPU状态..."
nvidia-smi

echo "4. 测试PyTorch CUDA..."
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

---

### 方案3：重启服务器（最彻底）

```bash
sudo reboot
```

**优点**：
- 最彻底的状态重置
- 通常能解决驱动状态异常

**缺点**：
- 需要停止所有工作
- 如果是硬件问题，重启后仍会失效

**适用场景**：
- 方案1和2都失败
- 或者服务器已运行多日，需要维护窗口

---

## 长期解决方案

### 检查GPU健康状态

**步骤1：运行内存测试**
```bash
# 如果有cuda-memtest工具
cuda-memtest --device 0

# 或使用pytorch测试
python3 << EOF
import torch
device = torch.device('cuda:0')
x = torch.randn(10000, 10000, device=device)
y = torch.mm(x, x)
print("GPU 0 memory test passed")
EOF
```

**步骤2：检查PCIe链路**
```bash
nvidia-smi -q -i 0 | grep -E "PCI|Link"
```

查看：
- PCIe Gen（应该是Gen 3或4）
- Link Width（应该是x16或x8）

**步骤3：查看系统日志**
```bash
sudo dmesg | grep -i nvidia
sudo dmesg | grep -i pci
```

查找：
- PCIe错误
- GPU reset事件
- 硬件错误报告

---

### 可能需要的硬件操作

| 检查项 | 操作 |
|--------|------|
| **PCIe连接** | 重新插拔GPU，确保接触良好 |
| **供电线缆** | 检查GPU供电线是否松动 |
| **灰尘清理** | 清理PCIe插槽和GPU金手指 |
| **散热检查** | 确认风扇工作，散热片无堵塞 |
| **更换插槽** | 尝试将GPU 0移到其他PCIe插槽 |

---

## 预防措施

### 1. 启用持久化模式

```bash
sudo nvidia-smi -pm 1
```

**作用**：
- 保持NVIDIA驱动常驻内存
- 减少GPU初始化开销
- 提高错误恢复能力

### 2. 定期重启

对于长时间运行的训练服务器：
- 每周或每月计划性重启
- 清理驱动和内核状态
- 避免累积的状态异常

### 3. 监控GPU健康

**安装监控工具**：
```bash
# nvitop（已在使用）
pip install nvitop

# gpustat
pip install gpustat
```

**定期检查**：
- GPU温度趋势
- ECC错误计数
- PCIe带宽
- 功耗异常

### 4. 训练时的保护

**在训练脚本中添加异常处理**：
```python
try:
    train()
except RuntimeError as e:
    if "CUDA" in str(e):
        # 保存当前状态
        torch.save(checkpoint, "emergency_checkpoint.pth")
        # 记录错误
        with open("cuda_error.log", "w") as f:
            f.write(str(e))
    raise
```

---

## GPU故障判断流程

```
GPU不可见
    │
    ├─ nvidia-smi能否看到？
    │   ├─ 能看到 → 驱动OK，检查PyTorch
    │   │   └─ torch.cuda.is_available()
    │   │       ├─ True → 环境变量问题
    │   │       └─ False → CUDA安装问题
    │   │
    │   └─ 看不到 → 检查nvidia-smi输出
    │       ├─ 所有GPU都看不到 → 驱动崩溃，重启
    │       └─ 只有某个GPU看不到 → 本次情况
    │           ├─ 重置GPU → sudo nvidia-smi -i X -r
    │           ├─ 仍失败 → 检查dmesg日志
    │           └─ 有PCIe错误 → 硬件问题
    │
    └─ 硬件问题判断
        ├─ 温度正常？
        ├─ 功耗正常？
        ├─ PCIe链路正常？
        └─ 系统日志有错误？
```

---

## 7卡训练的影响分析

### 计算能力

| 指标 | 8卡 | 7卡 | 差异 |
|------|-----|-----|------|
| GPU总数 | 8 | 7 | -12.5% |
| 有效批次 | 128 | 112 | -12.5% |
| 单步时间 | t | ~0.99t | +1% (通信略少) |
| **总训练时间** | **T** | **~1.13T** | **+13%** |

### 模型性能影响

- **收敛质量**：无明显影响（批次112仍足够大）
- **最终精度**：应该相同（总训练样本数不变）
- **训练稳定性**：可能略好（更小批次的正则化效果）

### 成本效益

**12.5%的时间延长 vs 立即继续训练**：
- 等待GPU修复：时间不确定（可能数小时到数天）
- 用7卡训练：立即开始，可控的时间成本

**建议**：先用7卡完成这轮训练，硬件维护可并行进行

---

## 如何判断是临时故障还是永久损坏

### 临时故障的特征

- ✓ 重启后恢复
- ✓ nvidia-smi --gpu-reset后恢复
- ✓ 无PCIe错误日志
- ✓ 温度和功耗正常

### 永久损坏的特征

- ✗ 重启后仍失效
- ✗ dmesg中有大量PCIe错误
- ✗ GPU风扇异常或不转
- ✗ 移到其他PCIe插槽仍失效

### 本次情况的初步判断

**倾向于：临时驱动状态异常**

依据：
1. 发生在CUDA kernel崩溃之后
2. 其他GPU正常工作
3. GPU 0之前工作正常（训练刚开始时可用）

**建议操作顺序**：
1. 先用7卡继续训练（方案1）
2. 训练空闲时尝试重启服务器（方案3）
3. 如果重启后仍失效，安排硬件检查

---

## 总结

### 当前推荐方案

**立即执行**：
```bash
cd /home/ksa/lerobot/self_scripts
bash start_train_act_stable_7gpu.sh
```

**原因**：
- 7卡足够完成训练
- 时间成本可控（+13%）
- 避免等待不确定的修复时间

### 后续处理

**训练间隙**：
- 尝试重启服务器
- 运行GPU健康检查
- 查看系统日志

**如果GPU 0频繁出问题**：
- 考虑RMA（保修）
- 或永久配置为7卡环境
