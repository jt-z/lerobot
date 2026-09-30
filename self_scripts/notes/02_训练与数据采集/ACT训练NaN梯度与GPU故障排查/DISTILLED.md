# ACT训练NaN梯度与GPU故障排查（精炼版）

## 核心结论

VAE的 `exp(log_sigma_x2_hat)` 计算在 `kl_weight=10.0` 时数值溢出产生NaN，降至1.0并配合小批次(16)、低学习率(5e-5)、简化架构(2层VAE)后稳定。

---

## 关键知识点

### 1. NaN产生机制

**公式**：
```
KL散度 = -0.5 * (1 + log_σ² - μ² - exp(log_σ²))
总损失 = L1损失 + kl_weight × KL散度
```

**数值危险点**：
- `kl_weight=10.0` 使KL损失主导训练
- 网络为降低KL散度让 `log_σ²` 剧烈变化
- 当 `log_σ² > 20` 时，`exp(log_σ²)` > 5×10⁸ → inf → NaN
- CUDA异步执行，错误在 `.item()` 同步时才抛出

### 2. 关键配置参数

| 参数 | 危险值 | 安全值 | 作用机制 |
|------|--------|--------|----------|
| `kl_weight` | 10.0 | **1.0** | 直接控制VAE训练强度 |
| `batch_size` | 32(×8=256) | **16(×7=112)** | 减小梯度方差，降低异常样本影响 |
| `optimizer_lr` | 8e-5 | **5e-5** | 减小参数单步变化，避免跳入危险区 |
| `n_vae_encoder_layers` | 4 | **2** | 减少非线性累积，降低表示空间复杂度 |
| `grad_clip_norm` | 10.0 | **5.0** | 更早截断异常梯度 |

### 3. 为什么在step 200出现

训练初期参数接近初始化范围，暂时稳定。Step 200附近：
1. VAE学会调整方差参数
2. 某batch触发极端预测
3. `log_σ²` 突破安全阈值 → 链式崩溃

**本质**：VAE训练是在优化一个potentially unbounded的目标

### 4. GPU故障的根因

**表现**：GPU 0 (0000:55:00.0) 从 nvidia-smi 消失

**可能原因**：
- CUDA kernel崩溃后驱动进入异常状态（高概率）
- 或硬件故障（中概率）

**应对**：使用 `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7` 绕过，有效批次128→112（-12.5%）

---

## 工作机制

### VAE训练的数值平衡

```
重建精度 ←→ 潜在空间正则化
    ↑            ↑
  L1损失      KL散度
             (×kl_weight)
```

**kl_weight的作用**：
- 过小(0.1)：VAE退化为普通自编码器，泛化能力差
- 适中(1.0)：平衡重建和正则化，**推荐**
- 过大(10.0)：网络冒险调整方差参数，数值爆炸

### CUDA异步执行与错误延迟

```
时刻1: exp(log_σ²) 计算 → GPU产生inf/NaN
时刻2: Python继续执行，不知道错误已发生
时刻3: l1_loss.item() 触发同步 → 检测到之前的kernel失败
```

**误导性**：错误堆栈指向 `.item()`，但真正错误在更早的 `exp()` 计算

---

## 重要决策与原因

| 决策点 | 选择 | 原因 |
|--------|------|------|
| VAE配置 | 降低kl_weight而非禁用VAE | VAE能提高泛化，问题在配置而非架构本身 |
| GPU方案 | 7卡训练而非等待修复 | 时间成本可控(+13%)，避免不确定的等待 |
| 梯度裁剪 | 5.0而非10.0 | 配合低学习率，形成多层保护 |
| 批次大小 | 16而非32 | 平衡稳定性和效率，16×GPU数是经验甜点 |

---

## 踩坑精华

| 问题 | 根因 | 解决方案本质 |
|------|------|------------|
| NaN在step 200出现 | VAE训练的delayed instability | 从源头降低KL权重，而非事后补救 |
| 错误指向.item() | CUDA异步执行机制 | 用CUDA_LAUNCH_BLOCKING=1定位真正错误 |
| 调试模式卡住 | 同步执行导致10-100倍性能下降 | 只用于错误定位，立即关闭 |
| GPU 0消失 | CUDA崩溃后驱动状态异常 | 用环境变量隔离故障硬件，保持训练连续性 |

---

## 可复用的规律

### 1. VAE/CVAE训练的黄金法则

**起始配置**：
```python
kl_weight = 1.0  # 不要用10.0
batch_size = 16  # 每GPU
learning_rate = 5e-5  # 保守
```

**渐进优化**：训练稳定后，每次只调整一个参数，观察效果

### 2. 多GPU训练的容错设计

**好的做法**：
```bash
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 ...
```

**效果**：部分GPU失效时可继续训练，而非全盘崩溃

### 3. 数值不稳定的诊断流程

```
梯度NaN → 检查损失项 → 识别无界操作(exp/div/log) → 
添加数值保护(clamp/eps) 或 降低相关权重
```

**本质**：NaN通常来自unbounded operation，而非bounded的L1/L2损失

### 4. 调试模式的正确用法

```
CUDA_LAUNCH_BLOCKING=1  # 仅用于定位错误
    ↓
定位到具体代码行
    ↓
关闭调试模式，用配置修复根因
```

**永远不要在调试模式下长时间训练**

---

## 关联知识

### 相关模型
- Diffusion Policy（也用VAE，同样需要保守的kl_weight）
- CVAE（条件VAE，数值稳定性原理相同）
- β-VAE（kl_weight即β参数，过大同样有风险）

### 相关技术
- KL散度的数值稳定实现（log-sum-exp trick）
- 梯度裁剪的原理与局限
- 分布式训练的故障恢复

### 可迁移场景
- 其他机器人学习任务（单臂、移动机器人）
- 多模态学习（视觉+触觉）
- 任何使用VAE正则化的监督学习

---

## 一句话记忆

**VAE训练时kl_weight要像学习率一样谨慎调整，10.0对大多数任务都过于激进，从1.0开始是更安全的选择。**

---

## 快速参考

### 立即可用的命令

```bash
# 7卡稳定训练（推荐）
cd /home/ksa/lerobot/self_scripts
bash start_train_act_stable_7gpu.sh

# 数据验证
python check_dataset_validity.py

# GPU状态检查
nvidia-smi

# PyTorch CUDA测试
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### 配置文件速查

- **稳定配置**：`train_config_act_stable.json` (kl=1.0, batch=16)
- **无VAE配置**：`train_config_act_no_vae.json` (最后手段)
- **7卡启动**：`start_train_act_stable_7gpu.sh`

### 关键数值对比

| 项目 | 原始 | 稳定 | 变化 |
|------|------|------|------|
| kl_weight | 10.0 | 1.0 | **↓90%** |
| 有效批次 | 256 | 112 | ↓56% |
| 参数量 | 52M | 43M | ↓17% |
| 训练时长 | T | ~2.1T | +110% |
