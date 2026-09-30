# SmolVLA 推理脚本修复总结

## 问题概述

在使用 `lerobot-record` 运行 SmolVLA 模型推理时，出现了以下错误：

```
ValueError: Feature mismatch between dataset/environment and policy config.
- Missing features: ['observation.images.camera1', 'observation.images.camera2', 'observation.images.camera3', 'observation.images.empty_camera_0']
- Extra features: ['observation.images.front', 'observation.images.side']
```

## 根本原因分析

### 1. 摄像头命名不匹配

- **机器人端**：摄像头命名为 `front` 和 `side`
- **模型端**：期望的摄像头命名为 `camera1`、`camera2`、`camera3`

虽然 `policy_preprocessor.json` 中配置了 `rename_observations_processor` 步骤来处理重命名，但验证阶段在处理器运行之前就执行了。

### 2. 验证逻辑缺陷（核心问题）

在 `lerobot_record.py` 调用 `make_policy` 时，**没有传递 `rename_map` 参数**：

```python
# 原始代码（lerobot_record.py 第 473 行）
policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
```

而 `make_policy` 函数中，只有当 `rename_map` 不为空时才会跳过验证：

```python
# factory.py 第 525-527 行
if not rename_map:
    validate_visual_features_consistency(cfg, features)
```

因此即使命令行传入了 `--dataset.rename_map`，验证阶段仍然会执行并报错。

## 修复方案

### 修复 1：修改 `lerobot_record.py`（本地补丁）

**⚠️ 重要警告：** 此修改是对框架核心文件的本地补丁，执行 `git pull` 或重新安装包后会丢失。建议提交上游 PR。

**文件**：`/home/jt/dev/lerobot/src/lerobot/scripts/lerobot_record.py`

**修改内容**：将 `make_policy` 调用改为传递 `rename_map` 参数

```python
# 修改前（第 473 行）
policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

# 修改后
policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.dataset.rename_map)
```

**修改原因**：`make_policy` 函数已经支持 `rename_map` 参数（用于跳过验证），但 `lerobot_record.py` 从未传递它。

### 修复 2：创建推理脚本

**文件**：`/home/jt/dev/lerobot/self_scripts/run_inference_smolvla.sh`

```bash
#!/bin/bash
# SmolVLA 推理脚本 - so101 put_ball2cup 任务
# 使用 lerobot-record + --policy.path 在真实机器人上运行策略
# 
# ⚠️ 依赖条件：
# 1. lerobot_record.py 已本地修改（见 notes/smolvla_inference_fix_summary.md）
# 2. SmolVLA 依赖已安装：pip install -e ".[smolvla]"

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

## 关键配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--robot.cameras` | `front`、`side` | 机器人实际摄像头名称 |
| `--dataset.rename_map` | `front→camera1`, `side→camera2` | 映射到模型期望的命名 |
| `--policy.path` | `output_lerobot_train/smolvla_put_ball2cup/checkpoints/020000/pretrained_model` | SmolVLA 训练 checkpoint |
| `--dataset.single_task` | `"Put ball to the cup"` | 任务描述，需与训练时一致 |
| `--policy.empty_cameras` | `1` | 自动填充 1 个空摄像头 |

## 数据流说明

```
机器人摄像头 (front, side)
       ↓
   [--dataset.rename_map]
       ↓
   验证阶段跳过（因为 rename_map 不为空）
       ↓
   [rename_observations_processor]
       ↓
模型输入 (camera1, camera2)
       ↓
   [empty_cameras=1]
       ↓
模型处理 (camera1, camera2, camera3*, empty_camera_0*)
```

- `camera3` 和 `empty_camera_0` 由模型自动生成（填充全 -1 张量）
- 重命名在预处理阶段（`rename_observations_processor`）执行

## 相关文件

| 文件 | 路径 | 作用 |
|------|------|------|
| 训练脚本 | `/home/jt/dev/lerobot/self_scripts/start_train_smolvla.sh` | 启动 SmolVLA 训练 |
| 训练配置 | `/home/jt/dev/lerobot/self_scripts/train_config_smolvla.json` | 训练参数配置 |
| 模型配置 | `output_lerobot_train/smolvla_put_ball2cup/checkpoints/020000/pretrained_model/config.json` | 模型架构配置 |
| 预处理配置 | `output_lerobot_train/smolvla_put_ball2cup/checkpoints/020000/pretrained_model/policy_preprocessor.json` | 预处理步骤配置 |
| 推理脚本 | `/home/jt/dev/lerobot/self_scripts/run_inference_smolvla.sh` | 运行推理 |

## 执行步骤

1. **安装 SmolVLA 依赖**：
   ```bash
   pip install -e ".[smolvla]"
   ```

2. **确认本地补丁已应用**：
   ```bash
   grep -n "rename_map=cfg.dataset.rename_map" /home/jt/dev/lerobot/src/lerobot/scripts/lerobot_record.py
   ```
   预期输出：`473:    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.dataset.rename_map)`

3. **运行推理脚本**：
   ```bash
   bash /home/jt/dev/lerobot/self_scripts/run_inference_smolvla.sh
   ```

## 注意事项

1. **`--dataset.rename_map` 参数**：必须与训练时使用的映射一致（`train_config_smolvla.json` 中的 `rename_map`）
2. **任务描述**：`--dataset.single_task` 必须与训练数据集中的描述一致
3. **Checkpoint 路径**：需指向实际训练完成的模型目录
4. **本地补丁维护**：执行 `git pull` 后需重新应用补丁
5. **上游 PR**：建议将 `lerobot_record.py` 的修改提交为上游 PR

## 常见问题

### Q: 为什么 `--dataset.rename_map` 不生效？
A: 因为 `lerobot_record.py` 调用 `make_policy` 时没有传递 `rename_map` 参数，导致验证阶段在重命名之前执行。

### Q: 为什么需要 `--policy.empty_cameras=1`？
A: 模型期望 3 个摄像头（camera1, camera2, camera3），但机器人只有 2 个。`empty_cameras=1` 告诉模型自动创建 1 个空摄像头。

### Q: 如果执行 `git pull` 后脚本失效怎么办？
A: 需要重新应用 `lerobot_record.py` 的补丁，或者等待上游合并 PR。