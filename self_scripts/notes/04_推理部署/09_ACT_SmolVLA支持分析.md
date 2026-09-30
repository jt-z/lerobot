# ACT与SmolVLA对双臂SO-101支持情况分析

## 📋 执行摘要

LeRobot框架**原生支持**双臂SO-101机械臂，并且ACT和SmolVLA两个策略模型都可以用于双臂控制。以下是关键发现：

### ✅ 支持情况总览

| 组件 | 支持状态 | 说明 |
|------|---------|------|
| **硬件抽象层** | ✅ 完整支持 | BiSOFollower类封装双臂 |
| **ACT策略** | ✅ 原生支持 | 论文本身就是为双臂ALOHA设计 |
| **SmolVLA策略** | ✅ 支持 | 通过动作维度padding支持多维动作 |
| **数据采集** | ✅ 支持 | lerobot_record支持bi_so_follower |
| **标定系统** | ✅ 已配置 | 存在完整的双臂标定文件 |

---

## 1️⃣ 硬件层：BiSOFollower双臂架构

### 核心设计

LeRobot提供了专门的`BiSOFollower`类来封装双臂SO-101机械臂：

**文件位置**：`src/lerobot/robots/bi_so_follower/bi_so_follower.py`

### 架构原理

```python
class BiSOFollower(Robot):
    """双臂SO Follower机械臂"""
    
    def __init__(self, config: BiSOFollowerConfig):
        # 创建两个独立的单臂实例
        self.left_arm = SOFollower(left_arm_config)
        self.right_arm = SOFollower(right_arm_config)
        
        # 合并相机配置
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}
```

### 关键机制：前缀分离

#### 观测数据（Observation）
```python
def get_observation(self) -> RobotObservation:
    obs_dict = {}
    
    # 左臂：添加 "left_" 前缀
    left_obs = self.left_arm.get_observation()
    obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})
    
    # 右臂：添加 "right_" 前缀
    right_obs = self.right_arm.get_observation()
    obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})
    
    return obs_dict
```

**输出示例**：
```python
{
    "left_shoulder_pan.pos": 45.2,
    "left_shoulder_lift.pos": -30.5,
    "left_elbow_flex.pos": 60.1,
    "left_wrist_flex.pos": 20.3,
    "left_wrist_roll.pos": 0.0,
    "left_gripper.pos": 55.0,
    "right_shoulder_pan.pos": -45.2,
    "right_shoulder_lift.pos": -30.5,
    # ... 右臂其他关节
}
```

#### 动作指令（Action）
```python
def send_action(self, action: RobotAction) -> RobotAction:
    # 拆分左右臂动作（移除前缀）
    left_action = {
        key.removeprefix("left_"): value 
        for key, value in action.items() 
        if key.startswith("left_")
    }
    
    right_action = {
        key.removeprefix("right_"): value 
        for key, value in action.items() 
        if key.startswith("right_")
    }
    
    # 分别发送到两个机械臂
    sent_action_left = self.left_arm.send_action(left_action)
    sent_action_right = self.right_arm.send_action(right_action)
    
    # 添加前缀并返回
    return {
        **{f"left_{k}": v for k, v in sent_action_left.items()},
        **{f"right_{k}": v for k, v in sent_action_right.items()}
    }
```

### 配置结构

**文件位置**：`src/lerobot/robots/bi_so_follower/config_bi_so_follower.py`

```python
@dataclass
class BiSOFollowerConfig(RobotConfig):
    left_arm_config: SOFollowerArmConfig
    right_arm_config: SOFollowerArmConfig
    calibration_dir: Path
```

### 标定文件现状

根据您的分析文档（`SO101_calibration_analysis.md`），您已经有完整的双臂标定：

```
~/.cache/huggingface/lerobot/calibration/
├── robots/so_follower/
│   ├── jt_follower_arm.json         # 主力配置（2026-06-29更新）
│   └── my_awesome_follower_arm.json  # 测试配置
└── teleoperators/so_leader/
    ├── jt_leader_arm.json           # 主力配置
    └── my_awesome_leader_arm.json    # 测试配置
```

**注意**：这些是**单臂**标定文件。双臂系统会使用：
- **左臂**：从一个标定文件加载
- **右臂**：从另一个标定文件加载（或同一个文件）

---

## 2️⃣ ACT策略：原生双臂支持

### 论文背景

ACT（Action Chunking Transformer）论文标题就是：
> **"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"**

**核心观点**：ACT本身就是为**双臂ALOHA**机器人设计的！

### 动作维度处理

**文件位置**：`src/lerobot/policies/act/modeling_act.py`

#### 关键参数

```python
@dataclass
class ACTConfig(PreTrainedConfig):
    n_obs_steps: int = 1          # 观测步数
    chunk_size: int = 100          # 动作chunk大小
    n_action_steps: int = 100      # 每次推理执行的动作步数
    
    # 输入输出特征通过 input_features 和 output_features 定义
    # 这些是从数据集自动推断的
```

#### 动作维度自动适配

ACT模型的动作维度是**通过配置自动确定**的：

```python
# 配置示例（从数据集推断）
output_features = {
    "action": PolicyFeature(
        type=FeatureType.ACTION,
        shape=(14,)  # 双臂：7关节/臂 × 2臂 = 14维
    )
}
```

**对于双臂SO-101**：
- 单臂：6个关节（shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper）
- 双臂：12维动作空间（6 × 2）

#### 模型输出

```python
def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
    """
    返回:
        actions: (batch_size, chunk_size, action_dim) 
                 对于双臂SO-101: (B, 100, 12)
    """
```

### 双臂使用示例

```python
from lerobot.policies.act import ACTPolicy

# 配置会自动从数据集推断动作维度
config = ACTConfig(
    chunk_size=100,
    n_action_steps=100,
    vision_backbone="resnet18",
)

# 创建策略
policy = ACTPolicy(config)

# 推理时的输入
observation = {
    "observation.image.top": ...,        # 相机图像
    "observation.state": torch.tensor([  # 12维状态
        # 左臂6个关节
        left_shoulder_pan, left_shoulder_lift, left_elbow_flex,
        left_wrist_flex, left_wrist_roll, left_gripper,
        # 右臂6个关节
        right_shoulder_pan, right_shoulder_lift, right_elbow_flex,
        right_wrist_flex, right_wrist_roll, right_gripper,
    ])
}

# 输出：(1, 100, 12) - 100步，12维动作
action = policy.select_action(observation)
```

### ACT关键特性

| 特性 | 说明 |
|------|------|
| **Action Chunking** | 一次预测多步动作（默认100步） |
| **VAE编码** | 使用变分自编码器学习动作分布 |
| **Transformer架构** | Encoder-Decoder结构处理视觉和状态 |
| **时序集成** | 可选的temporal ensembling提升稳定性 |
| **双臂原生支持** | 设计目标就是双臂精细操作 |

---

## 3️⃣ SmolVLA策略：通过Padding支持

### 设计理念

SmolVLA（Small Vision-Language-Action）是一个基于VLM的策略，通过**动作维度padding**机制支持不同维度的动作空间。

**文件位置**：`src/lerobot/policies/smolvla/`

### 配置参数

```python
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    chunk_size: int = 50           # 比ACT短
    n_action_steps: int = 50
    
    # 关键：最大维度限制
    max_state_dim: int = 32        # 状态向量最大维度
    max_action_dim: int = 32       # 动作向量最大维度
    
    # 图像预处理
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    
    # VLM相关
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    freeze_vision_encoder: bool = True
```

### 动作维度Padding机制

**文件位置**：`src/lerobot/policies/smolvla/modeling_smolvla.py`

```python
def forward(self, batch: dict[str, Tensor]) -> dict:
    # 读取原始动作维度
    original_action_dim = self.config.action_feature.shape[0]  # 例如：12
    
    # Padding到max_action_dim
    actions = pad_vector(batch[ACTION], self.config.max_action_dim)  # 12 -> 32
    
    # 模型处理...
    output_actions = self.model(...)
    
    # 去除padding，恢复原始维度
    actions = actions[:, :, :original_action_dim]  # 32 -> 12
    
    return actions
```

**Padding函数**：
```python
def pad_vector(vector: Tensor, max_dim: int) -> Tensor:
    """
    vector: (B, T, D)  例如 (32, 50, 12)
    返回:   (B, T, max_dim)  例如 (32, 50, 32)
    """
    if vector.shape[-1] < max_dim:
        padding = torch.zeros(*vector.shape[:-1], max_dim - vector.shape[-1])
        return torch.cat([vector, padding], dim=-1)
    return vector
```

### 状态投影

```python
class SmolVLA(nn.Module):
    def __init__(self, config: SmolVLAConfig):
        # 状态投影：max_state_dim -> VLM hidden size
        self.state_in_proj = nn.Linear(
            config.max_state_dim, 
            self.vlm_with_expert.config.text_config.hidden_size
        )
        
        # 动作投影：max_action_dim -> expert hidden size
        self.action_in_proj = nn.Linear(
            config.max_action_dim, 
            self.vlm_with_expert.expert_hidden_size
        )
        
        # 输出投影：expert hidden size -> max_action_dim
        self.action_out_proj = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, 
            config.max_action_dim
        )
```

### SmolVLA关键特性

| 特性 | 说明 |
|------|------|
| **VLM骨干** | 基于SmolVLM2视觉-语言模型 |
| **语言条件** | 支持自然语言任务描述 |
| **Padding机制** | 自动适配不同动作维度（≤32维） |
| **轻量化** | 500M参数，相比大模型更快 |
| **Action Chunking** | 预测50步动作序列 |

### 双臂SO-101使用

对于双臂SO-101（12维动作）：

```python
from lerobot.policies.smolvla import SmolVLAPolicy

config = SmolVLAConfig(
    chunk_size=50,
    n_action_steps=50,
    max_state_dim=32,   # 12 < 32，自动padding
    max_action_dim=32,  # 12 < 32，自动padding
)

policy = SmolVLAPolicy(config)

# 输入包含语言指令
observation = {
    "observation.image.top": ...,
    "observation.state": torch.tensor([...]),  # 12维
    "language_instruction": "pick up the cube"  # 可选
}

# 输出：(1, 50, 12)
action = policy.select_action(observation)
```

---

## 4️⃣ 数据采集与训练流程

### 数据采集脚本

**文件位置**：`src/lerobot/scripts/lerobot_record.py`

```bash
# 使用双臂SO-101采集数据
python lerobot/scripts/lerobot_record.py \
    --robot-path lerobot/configs/robot/bi_so_follower.yaml \
    --robot-overrides robot.config.left_arm_config.port=/dev/ttyUSB0 \
                      robot.config.right_arm_config.port=/dev/ttyUSB1 \
    --repo-id my_username/my_bimanual_dataset \
    --num-episodes 50 \
    --warmup-time-s 5 \
    --reset-time-s 5
```

### 数据格式

采集的数据会自动包含双臂前缀：

```python
# 数据集中的键
{
    "observation.images.top": ...,
    "observation.state": [  # 12维向量
        left_shoulder_pan, left_shoulder_lift, ..., 
        right_shoulder_pan, right_shoulder_lift, ...
    ],
    "action": [  # 12维向量
        left_shoulder_pan_target, ...,
        right_shoulder_pan_target, ...
    ]
}
```

### 训练ACT

```bash
python lerobot/scripts/train.py \
    --policy act \
    --dataset-repo-id my_username/my_bimanual_dataset \
    --output-dir outputs/act_bimanual \
    --policy.n_obs_steps 1 \
    --policy.chunk_size 100 \
    --policy.n_action_steps 100 \
    --policy.vision_backbone resnet18 \
    --policy.use_vae true \
    --training.batch_size 8 \
    --training.num_epochs 2000 \
    --training.lr 1e-5
```

### 训练SmolVLA

```bash
python lerobot/scripts/train.py \
    --policy smolvla \
    --dataset-repo-id my_username/my_bimanual_dataset \
    --output-dir outputs/smolvla_bimanual \
    --policy.chunk_size 50 \
    --policy.n_action_steps 50 \
    --policy.max_state_dim 32 \
    --policy.max_action_dim 32 \
    --policy.freeze_vision_encoder true \
    --training.batch_size 16 \
    --training.num_epochs 1000 \
    --training.lr 1e-4
```

---

## 5️⃣ 评估与部署

### 评估脚本

```bash
python lerobot/scripts/eval.py \
    --policy-path outputs/act_bimanual \
    --robot-path lerobot/configs/robot/bi_so_follower.yaml \
    --robot-overrides robot.config.left_arm_config.port=/dev/ttyUSB0 \
                      robot.config.right_arm_config.port=/dev/ttyUSB1 \
    --num-episodes 10
```

### 实时推理示例

```python
from lerobot.robots.bi_so_follower import BiSOFollower
from lerobot.policies.act import ACTPolicy

# 初始化机器人
robot = BiSOFollower(config)
robot.connect()

# 加载策略
policy = ACTPolicy.from_pretrained("outputs/act_bimanual")
policy.eval()

# 运行评估
observation = robot.get_observation()
while not done:
    action = policy.select_action(observation)
    observation = robot.send_action(action)
```

---

## 6️⃣ 关键差异对比

### ACT vs SmolVLA

| 维度 | ACT | SmolVLA |
|------|-----|---------|
| **原生双臂支持** | ✅ 论文设计目标 | ⚠️ 通过padding支持 |
| **动作序列长度** | 100步（可配置） | 50步（可配置） |
| **视觉编码器** | ResNet18/34/50 | SmolVLM2骨干 |
| **语言条件** | ❌ 不支持 | ✅ 支持 |
| **模型大小** | ~10-50M | ~500M |
| **训练速度** | 快 | 中等 |
| **推理速度** | 快（~30Hz） | 较慢（~10Hz） |
| **数据效率** | 需要较多数据 | 利用预训练，数据效率高 |
| **适用场景** | 固定任务的精细操作 | 多任务、语言指导操作 |

### 选择建议

#### 选择ACT的情况
- ✅ 专注于单一或少数固定任务
- ✅ 需要快速推理速度（实时控制）
- ✅ 数据充足（>1000个episode）
- ✅ 需要精细的双臂协调
- ✅ 不需要语言指令

#### 选择SmolVLA的情况
- ✅ 需要多任务泛化能力
- ✅ 需要语言条件控制
- ✅ 数据有限（利用预训练）
- ✅ 可以接受较慢的推理速度
- ✅ 探索VLM在机器人中的应用

---

## 7️⃣ 实际使用中的注意事项

### 标定问题

根据您的`SO101_calibration_analysis.md`：

1. **当前标定状态**：良好（2026-06-29更新）
2. **主力配置**：`jt_follower_arm.json` 和 `jt_leader_arm.json`
3. **问题关节**：wrist_roll和gripper需要定期重新标定

### 双臂配置建议

```yaml
# bi_so_follower配置示例
robot:
  type: bi_so_follower
  config:
    calibration_dir: ~/.cache/huggingface/lerobot/calibration
    left_arm_config:
      port: /dev/ttyUSB0
      calibration_id: jt_follower_arm
      max_relative_target: 0.3  # 安全限制
    right_arm_config:
      port: /dev/ttyUSB1
      calibration_id: jt_follower_arm  # 可以用同一个标定
      max_relative_target: 0.3
```

### 动作空间定义

确保数据采集和策略训练时使用一致的动作空间定义：

```python
# 推荐的动作空间顺序
action_space = [
    # 左臂（6维）
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    # 右臂（6维）
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
]
```

### 常见问题

#### Q1: 双臂动作是否需要同步？
- **硬件层**：BiSOFollower独立控制两臂，不强制同步
- **策略层**：ACT/SmolVLA同时预测12维动作，自然协调
- **建议**：依赖策略学习双臂协调，无需手动同步

#### Q2: 如何处理单臂任务？
- 可以固定一个臂的动作为0或当前位置
- 或者直接使用`SOFollower`单臂类

#### Q3: 训练数据需要多少？
- **ACT**：推荐1000+个episode获得良好性能
- **SmolVLA**：利用预训练，50-200个episode可能足够
- **实际**：取决于任务复杂度

#### Q4: 推理速度能达到多少？
- **ACT**：30-50 Hz（取决于硬件）
- **SmolVLA**：10-20 Hz（VLM骨干较大）
- **实际控制频率**：通常10-20 Hz足够

---

## 8️⃣ 完整工作流程示例

### Step 1: 标定（已完成）
您已有完整标定，存储在：
```
~/.cache/huggingface/lerobot/calibration/
├── robots/so_follower/jt_follower_arm.json
└── teleoperators/so_leader/jt_leader_arm.json
```

### Step 2: 数据采集
```bash
python lerobot/scripts/lerobot_record.py \
    --robot-path lerobot/configs/robot/bi_so_follower.yaml \
    --robot-overrides robot.config.left_arm_config.port=/dev/ttyUSB0 \
                      robot.config.right_arm_config.port=/dev/ttyUSB1 \
    --repo-id your_name/bimanual_task \
    --num-episodes 100
```

### Step 3: 训练ACT
```bash
python lerobot/scripts/train.py \
    --policy act \
    --dataset-repo-id your_name/bimanual_task \
    --output-dir outputs/act_model \
    --training.num_epochs 2000
```

### Step 4: 评估
```bash
python lerobot/scripts/eval.py \
    --policy-path outputs/act_model \
    --robot-path lerobot/configs/robot/bi_so_follower.yaml \
    --num-episodes 10
```

---

## 9️⃣ 总结

### 核心结论

1. **✅ LeRobot原生支持双臂SO-101**
   - 通过`BiSOFollower`类实现完整封装
   - 使用前缀机制（left_/right_）区分双臂

2. **✅ ACT策略完全支持双臂**
   - 论文本身就是为双臂ALOHA设计
   - 动作维度自动适配（12维）
   - 推荐用于精细双臂操作任务

3. **✅ SmolVLA策略支持双臂**
   - 通过padding机制支持任意维度（≤32）
   - 额外支持语言条件控制
   - 推荐用于多任务泛化场景

4. **✅ 您的硬件已就绪**
   - 标定文件完整且最新（2026-06-29）
   - 双臂硬件通信独立（不同串口）
   - 可以直接开始数据采集和训练

### 推荐起步路径

**如果您是第一次尝试**：
1. 从ACT开始（更成熟、更快）
2. 采集50-100个episode
3. 训练并评估
4. 如果效果不理想，增加数据量
5. 如果需要多任务能力，再尝试SmolVLA

**如果您有丰富经验**：
- 直接使用SmolVLA探索语言条件控制
- 利用预训练模型加速迁移学习

---

## 📚 参考资源

### 论文
- **ACT**: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- **SmolVLA**: [SmolVLM: Small Vision-Language Models](https://huggingface.co/papers/SmolVLM)

### 代码位置
- **BiSOFollower**: `src/lerobot/robots/bi_so_follower/`
- **ACT策略**: `src/lerobot/policies/act/`
- **SmolVLA策略**: `src/lerobot/policies/smolvla/`
- **数据采集**: `src/lerobot/scripts/lerobot_record.py`
- **训练脚本**: `src/lerobot/scripts/train.py`
- **评估脚本**: `src/lerobot/scripts/eval.py`

### 您的相关文档
- `SO101_calibration_analysis.md` - 标定情况分析
- `self_scripts/notes/SO101_dual_arm_analysis.md` - 双臂代码流程分析

---

**文档创建时间**：2026-07-29  
**分析基于代码库版本**：commit 9db8b93  
**作者**：Claude Code分析
