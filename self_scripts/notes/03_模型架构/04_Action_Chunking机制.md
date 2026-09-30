# Action Chunking 机制详解

## 一、基本概念

### 什么是 Action Chunking？

Action Chunking（动作分块）是一种机器人学习中的推理策略：

- **模型一次推理输出多个连续的动作**（一个动作块/chunk）
- **每帧只执行一个动作**，其余动作缓存到队列中
- 队列空了才触发下一次模型推理

### 为什么要用 Action Chunking？

1. **减少推理调用频率** - 模型推理慢，不需要每帧都调用
2. **动作更平滑连贯** - 模型一次看到完整的动作序列，规划更合理
3. **实时性更好** - 大部分帧直接从队列取动作，延迟很低

---

## 二、SmolVLA 中的配置

在 `configuration_smolvla.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 50 | 模型一次输出的动作序列长度 |
| `n_action_steps` | 50 | 每次推理后从 chunk 中取多少步放入队列 |

在你的配置中，`chunk_size = n_action_steps = 50`，说明每次推理的结果全部用完才进行下一次推理。

---

## 三、核心实现：select_action

位置: `src/lerobot/policies/smolvla/modeling_smolvla.py`

```python
def select_action(self, batch, noise=None, **kwargs):
    """Select a single action given environment observations.

    包装 select_actions，每次只返回一个动作。
    通过队列管理，队列空时才调用模型推理。
    """
    self.eval()
    batch = self._prepare_batch(batch)
    self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
    
    # 队列为空时才调用模型推理
    if self._check_get_actions_condition():  # len(queue) == 0
        actions = self._get_action_chunk(batch, noise)
        # actions shape: (batch_size, n_action_steps, action_dim)
        # 转置后放入队列: (n_action_steps, batch_size, action_dim)
        self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
    
    # 每次只返回一个动作
    return self._queues[ACTION].popleft()
```

### 队列初始化

```python
def reset(self):
    """环境重置时调用"""
    self._queues = {
        ACTION: deque(maxlen=self.config.n_action_steps),
    }
```

使用 `collections.deque` 作为队列，最大长度为 `n_action_steps`。

### 检查是否需要推理

```python
def _check_get_actions_condition(self) -> bool:
    return len(self._queues[ACTION]) == 0
```

只有当动作队列完全空了，才会调用模型生成新的动作块。

---

## 四、时间计算

### 你的配置

| 参数 | 值 |
|------|-----|
| chunk_size | 50 |
| n_action_steps | 50 |
| fps | 30 |

### 计算

```
推理间隔 = n_action_steps / fps
         = 50 / 30
         ≈ 1.67 秒
```

也就是说，大约每 **1.67 秒** 才调用一次完整的模型推理。

### 时间线示意

```
时间轴 (秒):
0s ──── 推理 1 ────────────────────────────── 1.67s ──── 推理 2 ──── ...
          ↓                                            ↑
          ├─ 动作 0 (第 0 帧)                          │
          ├─ 动作 1 (第 1 帧)                          │
          ├─ 动作 2 (第 2 帧)                          │
          ...                                        从队列取
          └─ 动作 49 (第 49 帧) ──────────────────────┘
```

---

## 五、完整流程

### 帧 0（队列为空，触发推理）

```
1. 机器人采集观测
2. 准备输入数据
3. 调用模型 → 输出 50 个动作 [a0, a1, a2, ..., a49]
4. 50 个动作全部放入队列
5. 从队列取出 a0 执行
6. 队列剩余: [a1, a2, ..., a49] (49 个)
```

### 帧 1 ~ 帧 49（队列非空，直接取）

```
1. 机器人采集观测
2. 准备输入数据
3. 检查队列 → 非空，跳过模型推理
4. 从队列取出下一个动作执行
5. 队列长度 -1
```

### 帧 50（队列再次为空，触发推理）

```
1. 机器人采集观测
2. 准备输入数据
3. 检查队列 → 空，调用模型推理
4. 输出新的 50 个动作
5. 放入队列，取出第一个执行
... 循环往复
```

---

## 六、动作块的生成：_get_action_chunk

```python
def _get_action_chunk(self, batch, noise=None, **kwargs):
    # 1. 准备图像
    images, img_masks = self.prepare_images(batch)
    
    # 2. 准备状态
    state = self.prepare_state(batch)
    
    # 3. 语言 tokens
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
    
    # 4. 调用模型采样动作序列
    actions = self.model.sample_actions(
        images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
    )
    # actions shape: (batch_size, chunk_size, max_action_dim)
    
    # 5. 去掉 padding 维度
    original_action_dim = self.config.action_feature.shape[0]
    actions = actions[:, :, :original_action_dim]
    
    return actions
```

### 输出形状

- `chunk_size = 50`: 动作序列长度
- `max_action_dim = 32`: 动作维度（padding 后）
- 原始动作维度根据机器人而定（如 SO-101 是 6 个关节 + 1 个夹爪 = 7 维）

---

## 七、与 Flow Matching 的关系

SmolVLA 使用 Flow Matching 生成动作序列，整个过程需要多步迭代：

```
sample_actions():
    1. 生成纯噪声 x_T (shape: [B, 50, action_dim])
    2. 计算前缀 KV cache (图像+语言+状态)
    3. 循环 num_steps=10 次:
       ├── 计算当前时间步的去噪方向 v_t
       └── 更新 x_t = x_t + dt * v_t
    4. 返回最终的 x_0 (50 个动作)
```

**注意**: 整个 10 步去噪都在一次 `sample_actions` 调用中完成，一次性生成全部 50 个动作。

---

## 八、优缺点分析

### 优点

1. **降低平均延迟**
   - 大部分帧直接从队列取动作，延迟 < 1ms
   - 只有每 50 帧才有一次推理延迟

2. **动作更平滑**
   - 模型一次看到完整的动作序列
   - 相邻动作之间更连贯

3. **更长的规划视野**
   - 模型可以规划未来 1.67 秒的动作
   - 比单步预测更有前瞻性

### 缺点

1. **对环境变化反应慢**
   - 如果环境突然变化，后面的动作可能过时
   - 最坏情况需要等 1.67 秒才能重新规划

2. **误差累积**
   - 后面的动作是基于当前观测预测的
   - 随着时间推移，误差可能累积

### 改进思路

实际应用中，通常会让 `n_action_steps < chunk_size`，比如：
- `chunk_size = 50`（模型输出 50 个动作）
- `n_action_steps = 10`（只执行前 10 个就重新推理）

这样可以在保持平滑性的同时，提高对环境变化的响应速度。

---

## 九、相关代码文件索引

| 文件 | 作用 |
|------|------|
| `src/lerobot/policies/smolvla/modeling_smolvla.py` | select_action, _get_action_chunk |
| `src/lerobot/policies/smolvla/configuration_smolvla.py` | chunk_size, n_action_steps 配置 |
| `src/lerobot/utils/control_utils.py` | predict_action 调用入口 |
