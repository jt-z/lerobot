# SmolVLA 模型架构详解

## 一、整体架构图

```
SmolVLA
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   输入: 图像 + 语言 + 状态                                   │
│        │                                                     │
│        ▼                                                     │
│   ┌──────────────────────────────────────────────────┐      │
│   │              embed_prefix (前缀嵌入)             │      │
│   │  图像 → SigLIP 视觉编码器 → 图像 embeddings      │      │
│   │  语言 → Tokenizer + Embedding → 语言 embeddings  │      │
│   │  状态 → Linear 投影 → 状态 embedding             │      │
│   └───────────────────────┬──────────────────────────┘      │
│                           │                                 │
│                           ▼                                 │
│   ┌──────────────────────────────────────────────────┐      │
│   │           VLM 主干 (SmolVLM2)                    │      │
│   │  计算前缀的 KV cache (只计算一次)                 │      │
│   └───────────────────────┬──────────────────────────┘      │
│                           │ KV cache                         │
│                           ▼                                 │
│   ┌──────────────────────────────────────────────────┐      │
│   │         Flow Matching 去噪 (10 步)               │      │
│   │  ┌──────────────────────────────────────────┐    │      │
│   │  │  第 t 步:                                 │    │      │
│   │  │  1. embed_suffix(动作噪声 + 时间步)       │    │      │
│   │  │  2. Action Expert + Cross Attention       │    │      │
│   │  │     (关注 VLM 的 KV cache)                 │    │      │
│   │  │  3. action_out_proj → 去噪方向 v_t        │    │      │
│   │  │  4. x_t = x_t + dt * v_t (欧拉更新)       │    │      │
│   │  └──────────────────────────────────────────┘    │      │
│   └───────────────────────┬──────────────────────────┘      │
│                           │                                 │
│                           ▼                                 │
│                    输出: 动作序列 (50 步)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、核心组件

### VLAFlowMatching 模型

位置: `src/lerobot/policies/smolvla/modeling_smolvla.py`

这是 SmolVLA 的核心模型类，负责整个推理流程。

```python
class VLAFlowMatching(nn.Module):
    def __init__(self, config, rtc_processor=None):
        super().__init__()
        self.config = config
        
        # VLM + Action Expert 双路模型
        self.vlm_with_expert = SmolVLMWithExpertModel(...)
        
        # 状态投影层
        self.state_proj = nn.Linear(max_state_dim, hidden_size)
        
        # 动作输入/输出投影层
        self.action_in_proj = nn.Linear(max_action_dim, expert_hidden_size)
        self.action_out_proj = nn.Linear(expert_hidden_size, max_action_dim)
        
        # 时间步 MLP
        self.action_time_mlp_in = nn.Linear(...)
        self.action_time_mlp_out = nn.Linear(...)
```

### SmolVLMWithExpertModel - 双路架构

位置: `src/lerobot/policies/smolvla/smolvlm_with_expert.py`

```
                    ┌─────────────────┐
前缀 (图像+语言) →  │   VLM 主干     │ → KV cache
                    │  (SmolVLM2)    │
                    └────────┬────────┘
                             │ cross attention
                             ▼
                    ┌─────────────────┐
  后缀 (动作) →    │  Action Expert   │ → 动作预测
                    │  (较小的 Transformer)
                    └─────────────────┘
```

**设计思路**:
- VLM 主干处理多模态输入（图像+语言），提取丰富的语义信息
- Action Expert 专门处理动作序列，通过 cross-attention 关注 VLM 的输出
- 这种设计类似于"编码器-解码器"架构，但 VLM 同时承担了编码器的角色

---

## 三、前缀嵌入：embed_prefix

位置: `modeling_smolvla.py`

### 功能

将图像、语言、状态三种模态转换为统一的嵌入序列，拼接在一起。

### 序列结构

```
[图像 embedding] + [语言 embedding] + [状态 embedding]
     ↓                 ↓                  ↓
   N_img            48 tokens          1 token
```

### 各模态处理

#### 1. 图像嵌入

```python
img_emb = self.vlm_with_expert.embed_image(img)
# 内部流程:
# image → SigLIP 视觉编码器 → connector 投影 → 图像 embeddings
```

**图像预处理**（在 `prepare_images` 中）:
- Resize + Padding 到 512×512（保持宽高比）
- 归一化: [0, 1] → [-1, 1]（SigLIP 要求）

**嵌入缩放**:
```python
img_emb = img_emb * sqrt(img_emb_dim)
```

#### 2. 语言嵌入

```python
lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
# 内部: text_model.get_input_embeddings()(tokens)
```

**嵌入缩放**:
```python
lang_emb = lang_emb * sqrt(lang_emb_dim)
```

#### 3. 状态嵌入

```python
state_emb = self.state_proj(state)
state_emb = state_emb[:, None, :]  # 增加序列维度
```

状态先 padding 到 `max_state_dim=32`，再通过线性层投影到 hidden_size。

### 注意力掩码设计

```python
att_masks = [0] * num_img_embs     # 图像: 前缀
att_masks += [0] * num_lang_embs    # 语言: 前缀
att_masks += [1] * states_seq_len   # 状态: 因果分界
```

**含义**:
- `att_mask = 0`: 前缀部分，token 之间可以互相注意
- `att_mask = 1`: 因果注意力的分界点，后面的 token 只能看前面的

通过 `make_att_2d_masks()` 生成 2D 注意力掩码矩阵。

---

## 四、后缀嵌入：embed_suffix

位置: `modeling_smolvla.py`

### 功能

将带噪声的动作序列和时间步编码为嵌入向量，作为 Action Expert 的输入。

### 输入

- `noisy_actions`: 带噪声的动作，形状 `(B, chunk_size, action_dim)`
- `timestep`: 当前时间步，标量

### 处理流程

```python
def embed_suffix(self, noisy_actions, timestep):
    # 1. 动作投影
    action_emb = self.action_in_proj(noisy_actions)  # (B, 50, expert_hidden)
    
    # 2. 时间步编码 (正弦位置编码)
    time_emb = create_sinusoidal_pos_embedding(timestep, ...)
    time_emb = time_emb[:, None, :].expand_as(action_emb)
    
    # 3. 拼接动作 + 时间
    action_time_emb = torch.cat([action_emb, time_emb], dim=2)
    
    # 4. MLP 融合
    action_time_emb = self.action_time_mlp_in(action_time_emb)
    action_time_emb = F.silu(action_time_emb)
    action_time_emb = self.action_time_mlp_out(action_time_emb)
    
    return action_time_emb, pad_masks, att_masks
```

### 时间步编码

使用正弦-余弦位置编码：

```python
def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    fraction = torch.linspace(0.0, 1.0, dimension // 2, ...)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb
```

- `min_period = 4e-3`
- `max_period = 4.0`
- 时间范围 [0, 1] 映射到不同频率的正弦波

---

## 五、Flow Matching 采样：sample_actions

位置: `modeling_smolvla.py`

### 整体流程

```python
def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
    bsize = state.shape[0]
    device = state.device
    
    # 步骤 1: 生成初始噪声
    if noise is None:
        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device)
    
    # 步骤 2: 嵌入前缀 + 计算 KV cache (只算一次)
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    
    _, past_key_values = self.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=self.config.use_cache,
        fill_kv_cache=True,  # 只填充 KV cache，不计算后缀输出
    )
    
    # 步骤 3: Flow Matching 迭代去噪
    num_steps = self.config.num_steps  # 10
    dt = -1.0 / num_steps
    x_t = noise
    
    for step in range(num_steps):
        time = 1.0 + step * dt  # 从 1.0 逐步降到 0.0
        time_tensor = torch.tensor(time, ...).expand(bsize)
        
        # 单步去噪
        v_t = self.denoise_step(x_t, prefix_pad_masks, past_key_values, time_tensor)
        
        # 欧拉步更新
        x_t = x_t + dt * v_t
    
    return x_t  # 最终的动作序列
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_steps` | 10 | 去噪步数 |
| `dt` | -1/10 = -0.1 | 每步的时间增量 |
| 起始时间 | 1.0 | 纯噪声 |
| 结束时间 | 0.0 | 干净的动作 |

---

## 六、单步去噪：denoise_step

位置: `modeling_smolvla.py`

```python
def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
    # 1. 嵌入后缀（动作噪声 + 时间步）
    suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)
    
    # 2. 构建注意力掩码
    # 后缀可以关注前缀 + 之前的后缀 token
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    
    # 3. 计算位置 ID
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    
    # 4. 通过 VLM + Expert
    outputs_embeds, _ = self.vlm_with_expert.forward(
        attention_mask=full_att_2d_masks,
        position_ids=position_ids,
        past_key_values=past_key_values,  # 复用前缀的 KV cache
        inputs_embeds=[None, suffix_embs],  # 前缀为空，后缀有输入
        use_cache=self.config.use_cache,
        fill_kv_cache=False,  # 不更新 KV cache
    )
    
    # 5. 投影到动作空间
    suffix_out = outputs_embeds[1][:, -self.config.chunk_size :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    v_t = self.action_out_proj(suffix_out)
    
    return v_t
```

### 注意力模式

```
前缀 token (图像+语言+状态)  ←───  后缀 token (动作)
     固定的 KV cache          cross attention
                              因果自注意力
```

- 后缀 token 通过 cross-attention 关注前缀的语义信息
- 后缀 token 之间是因果自注意力（只能看前面的动作）

---

## 七、SmolVLMWithExpert 详细结构

位置: `smolvlm_with_expert.py`

### VLM 主干

```python
self.vlm = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    ...
)
```

包含：
- 视觉编码器: SigLIP
- 连接器 (connector): 视觉特征投影
- 文本模型: Gemma 风格的 Transformer

### Action Expert

```python
lm_expert_config = copy.deepcopy(config.text_config)
lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)
lm_expert_config.num_hidden_layers = num_expert_layers
self.lm_expert = AutoModel.from_config(lm_expert_config)
```

- 比 VLM 小（宽度 `expert_width_multiplier = 0.75`）
- 层数可以少于 VLM（`num_expert_layers` 配置）
- 没有 `embed_tokens`（直接接收嵌入向量）

### Cross Attention 模式

当 `attention_mode = "cross_attn"` 时：
- Expert 的 K 和 V 来自 VLM 的输出（cross attention）
- Expert 的 Q 来自动作嵌入
- 每隔 `self_attn_every_n_layers` 层加一层自注意力

```python
if "cross" in attention_mode:
    for layer_idx in range(len(self.lm_expert.layers)):
        if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
            continue  # 这层是自注意力
        # 修改 KV 投影层，接受 VLM 的输出维度
        self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
            vlm_hidden_size, expert_hidden_size, ...
        )
        self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
            vlm_hidden_size, expert_hidden_size, ...
        )
```

---

## 八、训练 vs 推理

### 训练前向传播

```python
def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
    # 1. 采样噪声和时间
    noise = self.sample_noise(actions.shape, actions.device) if noise is None
    time = self.sample_time(actions.shape[0], actions.device) if time is None
    
    # 2. 构造带噪声的动作
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions  # 目标: 预测噪声方向
    
    # 3. 嵌入前缀 + 后缀
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
    suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)
    
    # 4. 拼接前+后缀，一次前向传播
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    
    (_, suffix_out), _ = self.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        fill_kv_cache=False,
    )
    
    # 5. 计算 MSE 损失
    v_t = self.action_out_proj(suffix_out)
    losses = F.mse_loss(u_t, v_t, reduction="none")
    
    return losses
```

### 训练 vs 推理的区别

| 方面 | 训练 | 推理 |
|------|------|------|
| 前向传播 | 前缀+后缀一次通过 | 前缀算 KV cache，后缀迭代 10 次 |
| 输出 | 去噪方向 (单次) | 完整动作序列 (10 步迭代) |
| 损失 | MSE(u_t, v_t) | 无损失，只采样 |
| KV cache | 不使用 | 使用，加速推理 |

---

## 九、相关代码文件索引

| 文件 | 作用 |
|------|------|
| `src/lerobot/policies/smolvla/modeling_smolvla.py` | VLAFlowMatching 核心模型 |
| `src/lerobot/policies/smolvla/smolvlm_with_expert.py` | VLM + Action Expert 双路模型 |
| `src/lerobot/policies/smolvla/configuration_smolvla.py` | 模型配置 |
| `src/lerobot/policies/smolvla/processor_smolvla.py` | 数据预处理 |
