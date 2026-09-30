# 双臂 SO-101 + ACT 训练完整操作指南

## 📋 目录

1. [硬件准备](#1-硬件准备)
2. [数据采集](#2-数据采集)
3. [ACT 训练](#3-act-训练)
4. [模型评估](#4-模型评估)
5. [常见问题](#5-常见问题)

---

## 1. 硬件准备

### 1.1 确认硬件连接

你需要 **4 个独立的串口连接**：

```bash
# 查看连接的串口设备
ls -l /dev/ttyUSB* /dev/ttyACM*

# 预期输出（示例）：
# /dev/ttyUSB0  <- 左臂 Follower
# /dev/ttyUSB1  <- 右臂 Follower
# /dev/ttyUSB2  <- 左臂 Leader（遥操作）
# /dev/ttyUSB3  <- 右臂 Leader（遥操作）
```

### 1.2 确认摄像头

```bash
# 查看可用摄像头
v4l2-ctl --list-devices

# 或者用 Python 测试
python -c "
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
"
```

记录下可用的摄像头索引（例如：0, 1, 2, 3）

### 1.3 校准检查

你已经有完整的校准文件（根据你的分析文档）：

```bash
# 查看校准文件
ls -la ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
ls -la ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/

# 确认存在：
# - jt_follower_arm.json
# - jt_leader_arm.json
```

**如果需要重新校准**：参考你的 `SO101_calibration_analysis.md` 文档

---

## 2. 数据采集

### 2.1 基本采集命令

```bash
lerobot-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/ttyLeftFollower \
  --robot.right_arm_config.port=/dev/ttyRightFollower \
  --robot.id=jt_follower_arm \
  --robot.left_arm_config.cameras='{"hand": {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30}}' \
  --robot.right_arm_config.cameras='{"hand": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}' \
  --robot.left_arm_config.max_relative_target=20.0 \
  --robot.right_arm_config.max_relative_target=20.0 \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/ttyLeftLeader \
  --teleop.right_arm_config.port=/dev/ttyRightLeader \
  --teleop.id=jt_leader_arm \
  --dataset.repo_id=jt-z/bimanual-task-dataset \
  --dataset.num_episodes=50 \
  --dataset.single_task="双臂协同抓取和传递物体" \
  --dataset.fps=30 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=20 \
  --dataset.video=true \
  --dataset.vcodec=h264 \
  --dataset.push_to_hub=true \
  --display_data=true
```

### 2.2 参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--robot.type` | 机器人类型 | `bi_so_follower` |
| `--robot.left_arm_config.port` | 左臂从动臂串口 | `/dev/ttyLeftFollower` 或实际端口 |
| `--robot.right_arm_config.port` | 右臂从动臂串口 | `/dev/ttyRightFollower` 或实际端口 |
| `--robot.id` | 机器人标识符（用于校准） | `jt_follower_arm`（你的校准文件名前缀）|
| `--robot.*_arm_config.max_relative_target` | 最大相对移动量（度） | 20.0（防止抖动）|
| `--teleop.type` | 遥操作类型 | `bi_so_leader` |
| `--teleop.left_arm_config.port` | 左臂主动臂串口 | `/dev/ttyLeftLeader` 或实际端口 |
| `--teleop.right_arm_config.port` | 右臂主动臂串口 | `/dev/ttyRightLeader` 或实际端口 |
| `--teleop.id` | 遥操作标识符（用于校准） | `jt_leader_arm`（你的校准文件名前缀）|
| `--dataset.repo_id` | 数据集名称（HF Hub） | `你的用户名/数据集名` |
| `--dataset.num_episodes` | 采集的 episode 数量 | 50-200（看任务复杂度）|
| `--dataset.single_task` | 任务描述 | 清晰描述任务内容 |
| `--dataset.fps` | 数据采集频率 | 30 Hz |
| `--dataset.episode_time_s` | 每个 episode 录制时长 | 30 秒 |
| `--dataset.reset_time_s` | 重置环境时长 | 20 秒 |
| `--dataset.video` | 是否保存视频 | `true` |
| `--dataset.vcodec` | 视频编码格式 | `h264`（推荐）或 `libx264` |
| `--dataset.push_to_hub` | 是否上传到 HF Hub | `true` |
| `--resume` | 从中断处继续采集 | `true`（可选）|

### 2.3 摄像头配置选项

#### 最小配置（每臂一个腕部相机）
```bash
--robot.left_arm_config.cameras='{"hand": {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30}}' \
--robot.right_arm_config.cameras='{"hand": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}'
```

#### 推荐配置（左臂三相机 + 右臂一相机）
```bash
--robot.left_arm_config.cameras='{
  hand: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30},
  top: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30, fourcc: MJPG}
}' \
--robot.right_arm_config.cameras='{
  hand: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'
```

**摄像头配置说明**：
- **相机名称**：使用 `hand`（手部）、`top`（顶部）、`front`（前视）等描述性名称
- **index_or_path**：可以使用数字索引（0, 1, 2）或设备路径（`/dev/video0`）
- **fourcc**：视频编码格式，`MJPG` 可以提高某些摄像头的性能和兼容性
- **分辨率**：640×480 是推荐的平衡配置，可根据需要调整

### 2.4 采集流程

1. **启动命令**后，系统会：
   - 连接到双臂硬件
   - 检查校准（首次会提示校准）
   - 连接摄像头
   - 准备数据采集

2. **录制 episode**：
   - 系统提示 "Recording episode X"
   - 使用主动臂（Leader）控制从动臂（Follower）执行任务
   - 默认录制时长可通过 `--dataset.episode_time_s` 设置（推荐 30 秒）
   - 按快捷键可提前结束或重录

3. **重置环境**：
   - 系统提示 "Reset the environment"
   - 将物体和机械臂恢复到初始状态
   - 重置时长可通过 `--dataset.reset_time_s` 设置（推荐 20 秒）
   - 准备下一个 episode

4. **完成采集**：
   - 达到设定的 episode 数量后自动停止
   - 数据保存到本地并上传到 Hugging Face Hub（如果设置了 `--dataset.push_to_hub=true`）

5. **中断恢复**：
   - 如果采集过程中断，可以添加 `--resume=true` 参数继续
   - 系统会自动从上次中断的 episode 继续采集

### 2.5 采集脚本示例

创建一个脚本便于重复使用（基于实际使用的脚本）：

```bash
#!/bin/bash
# 文件名：collect_bimanual_data.sh
# 数据采集脚本：双臂协同任务
#
# 支持中断继续录制：
#   如果录制过程中断，可以添加 --resume 参数继续录制
#   脚本会自动从上次中断的 episode 继续

set -e  # 遇到错误立即退出

# ==================== 命令行参数 ====================
# 检查是否传入 --resume 参数
RESUME_MODE=false
if [ "$1" == "--resume" ]; then
  RESUME_MODE=true
  echo "🔄 恢复模式：将从上次中断处继续录制"
fi

echo "=========================================="
echo "双臂数据采集 - 协同抓取任务"
echo "=========================================="
echo ""

# ==================== 硬件配置 ====================
LEFT_FOLLOWER_PORT="/dev/ttyLeftFollower"
RIGHT_FOLLOWER_PORT="/dev/ttyRightFollower"
LEFT_LEADER_PORT="/dev/ttyLeftLeader"
RIGHT_LEADER_PORT="/dev/ttyRightLeader"

# ==================== 摄像头配置 ====================
# 左臂：3个摄像头（手部、顶部、前视）
LEFT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30},
  top: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG},
  front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# 右臂：1个摄像头（手部）
RIGHT_CAMERAS='{
  hand: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'

# ==================== 数据集配置 ====================
DATASET_NAME="jt-z/bimanual-pick-and-place"
TASK_DESCRIPTION="双臂协同抓取物体并放置到目标位置"
NUM_EPISODES=100
EPISODE_TIME=30  # 每个 episode 录制时长（秒）
RESET_TIME=20    # 重置环境时长（秒）
FPS=30

# ==================== 采集前检查 ====================
echo "1. 检查硬件连接..."

# 检查串口
for port in $LEFT_FOLLOWER_PORT $RIGHT_FOLLOWER_PORT $LEFT_LEADER_PORT $RIGHT_LEADER_PORT; do
  if [ ! -e "$port" ]; then
    echo "❌ 错误：串口不存在 $port"
    echo "请检查硬件连接和串口映射"
    exit 1
  else
    echo "✅ $port 已连接"
  fi
done

# 检查摄像头
echo ""
echo "2. 检查摄像头..."
for video in /dev/video0 /dev/video2 /dev/video4 /dev/video6; do
  if [ ! -e "$video" ]; then
    echo "❌ 警告：摄像头不存在 $video"
  else
    echo "✅ $video 已连接"
  fi
done

# 检查校准文件
echo ""
echo "3. 检查校准文件..."
CALIB_FOLLOWER_DIR="$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower"
CALIB_LEADER_DIR="$HOME/.cache/huggingface/lerobot/calibration/teleoperators/so_leader"

if [ -d "$CALIB_FOLLOWER_DIR" ] && [ -d "$CALIB_LEADER_DIR" ]; then
  echo "✅ 校准文件目录存在"
  echo "   Follower: $(ls $CALIB_FOLLOWER_DIR | grep jt_follower_arm | wc -l) 个文件"
  echo "   Leader: $(ls $CALIB_LEADER_DIR | grep jt_leader_arm | wc -l) 个文件"
else
  echo "⚠️  校准文件目录不完整，首次运行时会提示校准"
fi

# ==================== 采集参数总览 ====================
echo ""
echo "=========================================="
echo "采集参数总览"
echo "=========================================="
echo "数据集名称：$DATASET_NAME"
echo "任务描述：$TASK_DESCRIPTION"
echo "Episode 数量：$NUM_EPISODES"
echo "每 Episode 时长：${EPISODE_TIME}秒"
echo "重置时长：${RESET_TIME}秒"
echo "采集频率：${FPS} Hz"
echo "预计总时长：约 $((($EPISODE_TIME + $RESET_TIME) * $NUM_EPISODES / 60)) 分钟"
echo "=========================================="
echo ""

# 确认开始
read -p "按 ENTER 开始数据采集，按 Ctrl+C 取消..." dummy

# ==================== 开始采集 ====================
echo ""
echo "🚀 开始数据采集..."
echo ""

# 构建命令
RECORD_CMD="lerobot-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=$LEFT_FOLLOWER_PORT \
  --robot.right_arm_config.port=$RIGHT_FOLLOWER_PORT \
  --robot.id=jt_follower_arm \
  --robot.left_arm_config.cameras=\"$LEFT_CAMERAS\" \
  --robot.right_arm_config.cameras=\"$RIGHT_CAMERAS\" \
  --robot.left_arm_config.max_relative_target=20.0 \
  --robot.right_arm_config.max_relative_target=20.0 \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=$LEFT_LEADER_PORT \
  --teleop.right_arm_config.port=$RIGHT_LEADER_PORT \
  --teleop.id=jt_leader_arm \
  --dataset.repo_id=$DATASET_NAME \
  --dataset.num_episodes=$NUM_EPISODES \
  --dataset.single_task=\"$TASK_DESCRIPTION\" \
  --dataset.fps=$FPS \
  --dataset.episode_time_s=$EPISODE_TIME \
  --dataset.reset_time_s=$RESET_TIME \
  --dataset.video=true \
  --dataset.vcodec=h264 \
  --dataset.push_to_hub=true \
  --display_data=true"

# 如果是恢复模式，添加 --resume 参数
if [ "$RESUME_MODE" = true ]; then
  RECORD_CMD="$RECORD_CMD --resume=true"
fi

# 执行命令
eval $RECORD_CMD

# ==================== 采集完成 ====================
echo ""
echo "=========================================="
echo "✅ 数据采集完成！"
echo "=========================================="
echo "数据集名称：$DATASET_NAME"
echo "Episode 数量：$NUM_EPISODES"
echo "Hugging Face Hub 链接："
echo "  https://huggingface.co/datasets/$DATASET_NAME"
echo ""
echo "💡 提示："
echo "  如果录制过程中断，可以使用以下命令继续："
echo "  ./collect_bimanual_data.sh --resume"
echo ""
echo "下一步："
echo "  1. 访问上述链接查看数据集"
echo "  2. 检查数据质量"
echo "  3. 开始训练 ACT 模型"
echo "=========================================="
```

**使用方法**：

```bash
# 首次采集
chmod +x collect_bimanual_data.sh
./collect_bimanual_data.sh

# 如果中断后继续采集
./collect_bimanual_data.sh --resume
```

**脚本特点**：
- ✅ 自动检查硬件连接（串口、摄像头、校准文件）
- ✅ 支持中断恢复（`--resume` 参数）
- ✅ 清晰的采集参数总览和预估时长
- ✅ 完整的错误检查和友好提示
- ✅ 多摄像头配置支持
- ✅ 自动上传到 Hugging Face Hub

---

## 3. ACT 训练

### 3.1 基本训练命令

```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=jt-z/bimanual-pick-and-place \
  --policy.input_features.observation.state.dtype=float32 \
  --policy.output_features.action.dtype=float32 \
  --training.output_dir=outputs/act_bimanual \
  --training.num_epochs=2000 \
  --training.batch_size=8 \
  --training.lr=1e-5 \
  --training.save_freq=500 \
  --training.eval_freq=500 \
  --device=cuda
```

### 3.2 ACT 关键参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--policy` | 策略类型 | - | `act` |
| `--dataset.repo_id` | 数据集位置 | - | 你的数据集 |
| `--policy.chunk_size` | 动作序列长度 | 100 | 100 |
| `--policy.n_action_steps` | 执行的动作步数 | 100 | 100 |
| `--policy.n_obs_steps` | 观测历史步数 | 1 | 1 |
| `--policy.vision_backbone` | 视觉编码器 | `resnet18` | `resnet18` 或 `resnet34` |
| `--policy.use_vae` | 使用 VAE | `true` | `true` |
| `--training.num_epochs` | 训练轮数 | 1000 | 2000-3000 |
| `--training.batch_size` | 批次大小 | 8 | 8-16（看显存）|
| `--training.lr` | 学习率 | 1e-4 | 1e-5 到 1e-4 |
| `--device` | 设备 | `cuda` | `cuda` 或 `cpu` |

### 3.3 完整训练命令（推荐配置）

```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=jt-z/bimanual-pick-and-place \
  --training.output_dir=outputs/act_bimanual_$(date +%Y%m%d_%H%M%S) \
  --training.num_epochs=2000 \
  --training.batch_size=8 \
  --training.lr=1e-5 \
  --training.gradient_accumulation_steps=1 \
  --training.save_freq=500 \
  --training.eval_freq=500 \
  --training.log_freq=100 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.n_obs_steps=1 \
  --policy.vision_backbone=resnet18 \
  --policy.use_vae=true \
  --policy.kl_weight=10.0 \
  --device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot-bimanual \
  --wandb.run_name=act_bimanual_training
```

### 3.4 训练脚本示例

```bash
#!/bin/bash
# 文件名：train_act_bimanual.sh

DATASET_REPO_ID="jt-z/bimanual-pick-and-place"
OUTPUT_DIR="outputs/act_bimanual_$(date +%Y%m%d_%H%M%S)"
NUM_EPOCHS=2000
BATCH_SIZE=8
LEARNING_RATE=1e-5

echo "开始训练 ACT 模型..."
echo "数据集: $DATASET_REPO_ID"
echo "输出目录: $OUTPUT_DIR"

lerobot-train \
  --policy=act \
  --dataset.repo_id=$DATASET_REPO_ID \
  --training.output_dir=$OUTPUT_DIR \
  --training.num_epochs=$NUM_EPOCHS \
  --training.batch_size=$BATCH_SIZE \
  --training.lr=$LEARNING_RATE \
  --training.save_freq=500 \
  --training.eval_freq=500 \
  --training.log_freq=100 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.n_obs_steps=1 \
  --policy.vision_backbone=resnet18 \
  --policy.use_vae=true \
  --device=cuda

echo "训练完成！"
echo "模型保存在: $OUTPUT_DIR"
```

### 3.5 监控训练进度

训练过程中会输出：

```
step: 0     loss: 2.456    lr: 1e-05
step: 100   loss: 1.234    lr: 1e-05
step: 200   loss: 0.987    lr: 1e-05
...
Saving checkpoint at step 500...
Running evaluation...
```

**使用 WandB 可视化**（推荐）：
```bash
# 如果启用了 --wandb.enable=true
# 访问：https://wandb.ai/your-username/lerobot-bimanual
```

### 3.6 训练时长估算

| 数据量 | Batch Size | GPU | 预估时长 |
|--------|-----------|-----|---------|
| 50 episodes | 8 | RTX 3090 | 2-4 小时 |
| 100 episodes | 8 | RTX 3090 | 4-8 小时 |
| 200 episodes | 8 | RTX 3090 | 8-16 小时 |
| 50 episodes | 8 | RTX 4090 | 1-2 小时 |

---

## 4. 模型评估

### 4.1 基本评估命令

```bash
lerobot-eval \
  --policy.path=outputs/act_bimanual_20260729_123456 \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/ttyUSB0 \
  --robot.right_arm_config.port=/dev/ttyUSB1 \
  --robot.id=jt_bimanual_follower \
  --robot.left_arm_config.cameras='{"wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
  --robot.right_arm_config.cameras='{"wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
  --eval.n_episodes=10 \
  --display_data=true
```

### 4.2 评估脚本示例

```bash
#!/bin/bash
# 文件名：eval_act_bimanual.sh

MODEL_PATH="outputs/act_bimanual_20260729_123456"
NUM_EPISODES=10

echo "开始评估模型..."
echo "模型路径: $MODEL_PATH"

lerobot-eval \
  --policy.path=$MODEL_PATH \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/ttyUSB0 \
  --robot.right_arm_config.port=/dev/ttyUSB1 \
  --robot.id=jt_bimanual_follower \
  --robot.left_arm_config.cameras='{"wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
  --robot.right_arm_config.cameras='{"wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
  --eval.n_episodes=$NUM_EPISODES \
  --display_data=true

echo "评估完成！"
```

### 4.3 Python 代码评估

如果需要更细粒度的控制：

```python
#!/usr/bin/env python
import torch
from lerobot.robots.bi_so_follower import BiSOFollower
from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig
from lerobot.policies.act import ACTPolicy

# 配置机器人
config = BiSOFollowerConfig(
    left_arm_config=SOFollowerConfig(
        port="/dev/ttyUSB0",
        id="jt_bimanual_follower",
        cameras={"wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}
    ),
    right_arm_config=SOFollowerConfig(
        port="/dev/ttyUSB1",
        id="jt_bimanual_follower",
        cameras={"wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}
    )
)

# 初始化机器人
robot = BiSOFollower(config)
robot.connect()

# 加载策略
policy = ACTPolicy.from_pretrained("outputs/act_bimanual_20260729_123456")
policy.eval()
policy.to("cuda")

# 运行评估
num_episodes = 10
success_count = 0

for episode in range(num_episodes):
    print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
    
    # 获取初始观测
    observation = robot.get_observation()
    
    done = False
    step = 0
    max_steps = 500
    
    while not done and step < max_steps:
        # 预测动作
        with torch.no_grad():
            action = policy.select_action(observation)
        
        # 执行动作
        observation = robot.send_action(action)
        
        step += 1
    
    # 评估是否成功（需要根据任务定义）
    success = input("任务是否成功? (y/n): ").strip().lower() == 'y'
    if success:
        success_count += 1
    
    print(f"Steps taken: {step}")
    print(f"Success so far: {success_count}/{episode + 1}")

# 断开连接
robot.disconnect()

# 输出结果
success_rate = success_count / num_episodes * 100
print(f"\n=== Evaluation Results ===")
print(f"Total episodes: {num_episodes}")
print(f"Successful: {success_count}")
print(f"Success rate: {success_rate:.1f}%")
```

---

## 5. 常见问题

### 5.1 数据采集问题

#### Q1: 串口连接失败
```
Error: Could not open port /dev/ttyUSB0
```

**解决方案**：
```bash
# 1. 检查设备是否存在
ls -l /dev/ttyUSB*

# 2. 检查权限
sudo chmod 666 /dev/ttyUSB0
# 或者添加用户到 dialout 组
sudo usermod -a -G dialout $USER
# 然后重新登录

# 3. 检查是否被其他程序占用
lsof /dev/ttyUSB0
```

#### Q2: 摄像头无法打开
```
Error: Cannot open camera 0
```

**解决方案**：
```bash
# 1. 检查摄像头是否被占用
lsof /dev/video0

# 2. 测试摄像头
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# 3. 检查权限
sudo chmod 666 /dev/video*
```

#### Q3: 校准文件不匹配
```
Warning: Calibration mismatch
```

**解决方案**：
```bash
# 重新校准
# 按照提示移动机械臂到中间位置，然后移动各关节到极限
# 校准文件会保存到 ~/.cache/huggingface/lerobot/calibration/
```

### 5.2 训练问题

#### Q4: CUDA 内存不足
```
RuntimeError: CUDA out of memory
```

**解决方案**：
```bash
# 减小 batch size
lerobot-train ... --training.batch_size=4

# 或使用梯度累积
lerobot-train ... --training.batch_size=4 --training.gradient_accumulation_steps=2
```

#### Q5: 损失不下降
```
loss: 2.456 (保持不变多个 epoch)
```

**可能原因和解决方案**：
1. **学习率太小**：尝试 `--training.lr=1e-4`
2. **数据量不足**：采集更多 episodes（至少 50-100）
3. **数据质量差**：检查遥操作是否稳定，重新采集高质量数据
4. **模型配置问题**：检查 `input_features` 和 `output_features` 是否正确

#### Q6: 训练速度太慢

**优化方案**：
```bash
# 1. 使用更小的视觉编码器
--policy.vision_backbone=resnet18  # 而不是 resnet50

# 2. 减少图像分辨率
--robot.left_arm_config.cameras='{"wrist": {"width": 320, "height": 240, ...}}'

# 3. 使用混合精度训练
--training.use_amp=true
```

### 5.3 评估问题

#### Q7: 模型表现不稳定

**可能原因**：
1. **训练不充分**：继续训练更多 epochs
2. **环境变化**：确保评估环境与训练时一致（光照、物体位置等）
3. **Action Chunking 问题**：尝试调整 `n_action_steps`

**改进方案**：
```bash
# 使用 temporal ensembling
--policy.temporal_ensemble_coeff=0.01
```

#### Q8: 双臂不协调

**可能原因**：
1. **数据质量**：采集时遥操作不够流畅
2. **训练数据不足**：双臂协调需要更多数据
3. **动作空间定义**：确认左右臂动作顺序正确

**改进方案**：
- 采集更多展示双臂协调的 episodes
- 增加训练时间
- 检查数据集中的动作维度顺序

### 5.4 硬件问题

#### Q9: 机械臂抖动

**可能原因**：
1. **控制频率不匹配**：采集和推理频率不一致
2. **动作变化太大**：`max_relative_target` 设置过大
3. **PID 参数不佳**：电机控制参数需要调整

**解决方案**：
```bash
# 1. 限制最大相对目标
--robot.left_arm_config.max_relative_target=0.2 \
--robot.right_arm_config.max_relative_target=0.2

# 2. 调整推理频率
--eval.fps=20  # 降低控制频率
```

#### Q10: 夹爪力度不对

**可能原因**：
- 夹爪扭矩限制太低或太高

**解决方案**：
检查 `so_follower.py` 中的夹爪配置：
```python
# 第 168-170 行
self.bus.write("Max_Torque_Limit", "gripper", 500)  # 调整这个值
self.bus.write("Protection_Current", "gripper", 250)
```

---

## 6. 性能优化建议

### 6.1 数据采集优化

1. **保持遥操作流畅**：避免突然的动作变化
2. **环境一致性**：确保每个 episode 的初始状态相似
3. **数据多样性**：在不同位置、角度采集数据
4. **质量控制**：采集后立即回放检查，删除失败的 episode

### 6.2 训练优化

1. **数据增强**：
   ```bash
   --training.use_augmentation=true
   --training.augmentation_type=random_crop
   ```

2. **学习率调度**：
   ```bash
   --training.lr_scheduler=cosine
   --training.warmup_steps=1000
   ```

3. **正则化**：
   ```bash
   --training.weight_decay=1e-4
   --policy.dropout=0.1
   ```

### 6.3 推理优化

1. **使用 TensorRT**（高级）：
   ```python
   # 导出为 ONNX 后转换为 TensorRT
   ```

2. **批处理推理**（如果有多个机器人）
3. **模型量化**（降低精度以提速）

---

## 7. 完整工作流程总结

```bash
# Step 1: 确认硬件连接
ls -l /dev/ttyUSB*
v4l2-ctl --list-devices

# Step 2: 数据采集（50-200 episodes）
bash collect_bimanual_data.sh

# Step 3: 训练 ACT 模型（2000 epochs）
bash train_act_bimanual.sh

# Step 4: 评估模型性能（10 episodes）
bash eval_act_bimanual.sh

# Step 5: 根据结果迭代
# - 如果性能不佳：采集更多数据或调整超参数
# - 如果性能良好：部署到实际应用
```

---

## 8. 参考资源

- **你的分析文档**：`ACT_SmolVLA_BiSOFollower_Support_Analysis.md`
- **校准分析**：`SO101_calibration_analysis.md`
- **ACT 论文**：https://arxiv.org/abs/2304.13705
- **LeRobot 文档**：https://huggingface.co/docs/lerobot
- **中文教程**：https://zihao-ai.feishu.cn/wiki/space/7589642043471924447

---

**创建时间**：2026-07-29  
**作者**：Claude Code  
**适用于**：双臂 SO-101 + ACT 策略训练
