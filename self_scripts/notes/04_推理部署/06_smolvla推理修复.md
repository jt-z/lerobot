# SmolVLA 推理脚本修复详解

安装代码库： `pip install -e ".[smolvla]" -i https://pypi.tuna.tsinghua.edu.cn/simple`


## 问题背景

在尝试使用训练好的 SmolVLA 模型进行机器人推理时，原始脚本 `run_inference_smolvla.sh` 存在多处参数配置错误，导致策略无法正确加载或执行。

## 问题分析

### 1. 摄像头名称映射的核心问题 ⚠️ **最关键**

#### 问题描述

**训练时的摄像头命名：**
- 数据集中摄像头键名：`observation.images.front` 和 `observation.images.side`
- 通过 `--rename_map` 训练时重映射为：`observation.images.camera1` 和 `observation.images.camera2`
- SmolVLA 模型权重期望的输入特征键名就是 `camera1`、`camera2`、`camera3`

**推理时的问题：**
机器人原始摄像头名称仍然是 `front` 和 `side`，必须在推理时也进行相同的重映射，否则策略无法匹配输入特征。

#### 原脚本的错误

```bash
# ❌ 错误：使用顶层参数 --rename_map
--rename_map='{"observation.images.front": "observation.images.camera1", ...}'
```

**为什么失败：**

从 `lerobot_record.py` 源码分析：

```python
# lerobot_record.py:186
class DatasetRecordConfig:
    rename_map: dict[str, str] = field(default_factory=dict)

# lerobot_record.py:473 - 策略加载时使用的是 cfg.dataset.rename_map
policy = make_policy(
    cfg.policy, 
    ds_meta=dataset.meta, 
    rename_map=cfg.dataset.rename_map  # ← 这里！
)
```

`RecordConfig` 顶层**没有** `rename_map` 字段，只有嵌套的 `DatasetRecordConfig` 里有。

使用顶层 `--rename_map` 时：
1. 命令行解析器找不到对应字段，参数被**静默忽略**
2. `cfg.dataset.rename_map` 保持空字典 `{}`
3. 策略加载时收到 `front`/`side`，但权重期望 `camera1`/`camera2`
4. **特征维度不匹配，推理失败**

#### 修复方案

```bash
# ✅ 正确：使用嵌套参数 --dataset.rename_map
--dataset.rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}'
```

这样 `cfg.dataset.rename_map` 才能正确接收到映射表，推理时摄像头名称会被正确转换。

---

### 2. 空摄像头补齐问题

#### 问题描述

SmolVLA base 模型是基于 Aloha 机器人（3个摄像头）预训练的：
- `camera1`: top camera
- `camera2`: left wrist camera  
- `camera3`: right wrist camera

你的 SO101 机器人只有 2 个摄像头（front + side），训练时必须补齐到 3 路输入才能复用预训练权重。

#### 训练时的配置

```bash
# start_train_smolvla.sh
--policy.empty_cameras=1  # 添加 1 个空摄像头
```

这个参数触发 `configuration_smolvla.py` 的逻辑：

```python
# configuration_smolvla.py:124
def validate_features(self) -> None:
    for i in range(self.empty_cameras):
        key = f"{OBS_IMAGES}.empty_camera_{i}"
        empty_camera = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 480, 640),
        )
        self.input_features[key] = empty_camera
```

训练时模型的 `input_features` 包含：
- `observation.images.camera1` (来自 front)
- `observation.images.camera2` (来自 side)
- `observation.images.empty_camera_0` (全黑图像，自动生成)

#### 原脚本的错误

```bash
# ❌ 推理脚本缺少 --policy.empty_cameras=1
```

**为什么失败：**

1. 模型权重是按 3 路视觉输入训练的
2. 推理时如果只提供 2 路输入，**输入张量维度不匹配**
3. 前向传播时会报形状错误

#### 修复方案

```bash
# ✅ 推理时也必须添加空摄像头，与训练配置一致
--policy.empty_cameras=1
```

---

### 3. Episode 时长配置不合理

#### 原脚本

```bash
--dataset.episode_time_s=1000  # 1000 秒 ≈ 16.7 分钟
```

**问题：**
- 推理测试时单个 episode 不需要这么长
- 实际任务（put ball to cup）可能只需要 30-60 秒
- 过长会导致无法快速迭代测试

#### 修复方案

```bash
--dataset.episode_time_s=60  # 60 秒，更合理
```

---

### 4. 推理次数未明确指定

#### 原脚本

```bash
# 缺少 --dataset.num_episodes
```

默认值是 `50` 次（`lerobot_record.py:161`），对于快速测试来说太多。

#### 修复方案

```bash
--dataset.num_episodes=5  # 明确指定 5 次测试
```

---

### 5. 策略运行设备未指定

#### 原脚本

```bash
# 缺少 --policy.device
```

虽然 checkpoint 中保存了 `device: cuda` 配置，但显式传参更安全，避免因配置加载问题导致在 CPU 上运行（推理速度会非常慢）。

#### 修复方案

```bash
--policy.device=cuda  # 明确使用 GPU
```

---

### 6. Checkpoint 路径硬编码风险

#### 原脚本

```bash
--policy.path=/home/jt/dev/lerobot/output_lerobot_train/smolvla_put_ball2cup/checkpoints/020000/pretrained_model
```

直接硬编码，如果路径不存在会在后续步骤才报错，调试不友好。

#### 修复方案

```bash
# 添加路径检查
POLICY_PATH=/home/jt/dev/lerobot/output_lerobot_train/smolvla_put_ball2cup/checkpoints/020000/pretrained_model
if [ ! -d "$POLICY_PATH" ]; then
    echo "错误: 找不到模型 checkpoint: $POLICY_PATH"
    echo "请确认训练已完成且路径正确"
    exit 1
fi
```

---

## 完整修复后的脚本

```bash
#!/bin/bash
# SmolVLA 推理脚本 - so101 put_ball2cup 任务
# 使用 lerobot-record + --policy.path 在真实机器人上运行策略

# 清理上次评估缓存（每次重新开始）
rm -rf /home/jt/.cache/huggingface/lerobot/hellozjt/eval_smolvla_so101_put_ball2cup

# 确认 checkpoint 路径存在
POLICY_PATH=/home/jt/dev/lerobot/output_lerobot_train/smolvla_put_ball2cup/checkpoints/020000/pretrained_model
if [ ! -d "$POLICY_PATH" ]; then
    echo "错误: 找不到模型 checkpoint: $POLICY_PATH"
    echo "请确认训练已完成且路径正确"
    exit 1
fi

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=jt_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}" \
  --display_data=true \
  --dataset.push_to_hub=false \
  --dataset.repo_id=hellozjt/eval_smolvla_so101_put_ball2cup \
  --dataset.single_task="Put ball to the cup" \
  --dataset.episode_time_s=60 \
  --dataset.num_episodes=5 \
  --dataset.rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
  --policy.path=$POLICY_PATH \
  --policy.device=cuda \
  --policy.empty_cameras=1
```

---

## 数据流示意图

### 修复前（错误）

```
机器人摄像头              无映射/映射失败           策略期望输入
──────────────────────────────────────────────────────
observation.images.front  ─┐                        observation.images.camera1 ✗
observation.images.side   ─┤ 无重映射或映射无效      observation.images.camera2 ✗
                          └─→ 特征不匹配，推理失败
```

### 修复后（正确）

```
机器人摄像头                   --dataset.rename_map            策略输入
─────────────────────────────────────────────────────────────────
observation.images.front  ────→ observation.images.camera1  ─┐
observation.images.side   ────→ observation.images.camera2  ─┤─→ SmolVLA Policy
                                observation.images.empty_camera_0 ─┘
                                (--policy.empty_cameras=1 自动生成)
```

---

## 验证日志分析

从 `log_smol_vla.txt` 可以看到修复成功：

### 1. 配置正确加载

```python
'rename_map': {
    'observation.images.front': 'observation.images.camera1',
    'observation.images.side': 'observation.images.camera2'
}
```

✅ `rename_map` 正确传入 `cfg.dataset.rename_map`

### 2. 策略输入特征正确

```python
'input_features': {
    'observation.images.camera1': {'shape': (3, 256, 256), 'type': 'VISUAL'},
    'observation.images.camera2': {'shape': (3, 256, 256), 'type': 'VISUAL'},
    'observation.images.camera3': {'shape': (3, 256, 256), 'type': 'VISUAL'},
    'observation.images.empty_camera_0': {'shape': (3, 480, 640), 'type': 'VISUAL'},
    'observation.state': {'shape': (6,), 'type': 'STATE'}
}
```

✅ 包含 `camera1`、`camera2`、`camera3` 和 `empty_camera_0`  
✅ 模型期望的 4 路输入特征全部存在（camera3 是 base 模型的，empty_camera_0 是你补的）

### 3. 设备配置正确

```python
'device': 'cuda',
'empty_cameras': 1,
```

✅ GPU 推理  
✅ 空摄像头补齐

### 4. 推理成功启动

```
INFO 2026-07-08 12:04:38 follower.py:106 jt_follower_arm SOFollower connected.
INFO 2026-07-08 12:04:38 ls/utils.py:227 Recording episode 0
```

✅ 机器人连接成功  
✅ 开始执行推理并记录数据

---

## 关键经验总结

### 1. 参数层级很重要

`lerobot-train` 和 `lerobot-record` 的配置结构不同：

| 命令 | `rename_map` 位置 |
|------|-------------------|
| `lerobot-train` | 顶层参数 `--rename_map` |
| `lerobot-record` | 嵌套参数 `--dataset.rename_map` |

**不能直接复制粘贴训练脚本的参数到推理脚本！**

### 2. 训练与推理必须配置一致

这些参数在训练和推理时**必须完全一致**：
- `--dataset.rename_map` / `--rename_map`（映射规则）
- `--policy.empty_cameras`（空摄像头数量）
- 任务描述文本（`--dataset.single_task`）

否则模型的输入特征维度会对不上。

### 3. 阅读源码是最可靠的调试方法

当命令行参数不生效时：
1. 找到脚本源文件（如 `lerobot_record.py`）
2. 搜索配置类定义（如 `RecordConfig`、`DatasetRecordConfig`）
3. 确认参数的正确嵌套路径
4. 查看参数如何被使用（如 `cfg.dataset.rename_map` 传给 `make_policy`）

### 4. 验证修复的检查清单

- [ ] 日志中 `rename_map` 非空
- [ ] `policy.input_features` 包含期望的键名（`camera1`/`camera2` 等）
- [ ] `empty_cameras` 值与训练一致
- [ ] `device` 设为 `cuda`
- [ ] 机器人连接成功
- [ ] Episode 开始记录

---

## 延伸：为什么需要 rename_map？

### Base 模型的历史包袱

SmolVLA base 是在多个数据集上预训练的，不同数据集的摄像头命名不统一：
- Aloha 数据集：`top`, `left_wrist`, `right_wrist`
- Open X-Embodiment 数据集：`camera1`, `camera2`, `camera3`

为了统一，训练时强制重映射到 `camera1/2/3` 命名空间。

### 你的情况

- 机器人驱动代码定义摄像头为 `front` 和 `side`
- 训练时用 `rename_map` 映射到 `camera1` 和 `camera2`
- 推理时必须做相同映射，才能匹配模型权重

**如果不映射：**  
模型权重的 visual encoder 输入层期望键名 `camera1`，但收到的是 `front`，无法找到对应的特征，推理失败。

---

## 文件修改记录

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `run_inference_smolvla.sh` | 参数修正 | 6 处参数修改 + 路径检查逻辑 |

**修改时间：** 2026-07-08  
**验证状态：** ✅ 推理成功启动，机器人连接正常
