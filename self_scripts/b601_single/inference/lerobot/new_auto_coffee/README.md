# new_auto_coffee —— coffee_expo 四个技能各自独立的 ACT 权重与推理脚本

数据源：232 `b601_pi05_coffee_expo_20261118_181510`（102 集 / 92,239 帧 / 4 个任务）
训练：232 8×3090，按 `--dataset.episodes` 白名单把每个任务单独切出来训（不是物理复制数据集），
每段 ≈13 passes、有效 batch 128（batch 16/卡 × 8 卡）。

| 任务 | 内容 | 权重（kfzbox 路径） | 步数 | 训练帧数 | model.safetensors sha256 |
|---|---|---|---|---|---|
| A1 | 端空杯 → 咖啡机下方银色接水托盘 → 回零位 | `~/LX/pai0/act_expo_A1_20261119/checkpoints/003100/pretrained_model` | 3,100 | 30,633（30 集） | `1aca66f6…dae7919` |
| B1 | 选 Hot Water → 按 Start 出热水 → 回零位 | `~/LX/pai0/act_expo_B1_20261119/checkpoints/002600/pretrained_model` | 2,600 | 25,810（30 集） | `a0301f1e…032e0f05` |
| B2 | 选 Americano → 按 Start 冲煮 → 回零位 | `~/LX/pai0/act_expo_B2_20261119/checkpoints/001800/pretrained_model` | 1,800 | 18,078（21 集） | `4f93154a…aa0c99ef` |
| C2 | 满杯 → 放到杯位 2 → 回零位 | `~/LX/pai0/act_expo_C2_20261119/checkpoints/001800/pretrained_model` | 1,800 | 17,718（21 集） | `bc581551…7d0d5515` |

权重都是 `checkpoints/last`（即各段最后一步），每个 `pretrained_model/` 198 MB / 7 个文件；
传输时按目录 `pretrained_model/` 整体 rsync，落地后已用 `sha256sum` 双端比过。

## 怎么跑

```bash
cd ~/dev/lerobot/self_scripts/b601_single/inference/lerobot/new_auto_coffee
bash run_act_coffee.sh A1          # 端空杯上托盘，默认 300 秒
bash run_act_coffee.sh B1 600      # 热水，600 秒
bash run_act_coffee.sh B2          # 美式，300 秒
bash run_act_A1.sh <权重目录> 0     # 换权重 / 不限时（0 = 一直跑到 Ctrl+C）
```

脚本会依次自检：lerobot 环境（必须 0.6.2 的 `lerobot_b601`，不是 0.4.4）→ 权重文件 → 动作追踪包装器
→ can0 → 三路相机 → 标定文件，然后要求「机械臂在零位、夹爪闭合」按 ENTER 才开跑。

## 口径与坑

1. **任务文本必须与数据集 `meta/tasks.parquet` 逐字一致**（脚本里已按 task_index 抄好）；ACT 基本不吃语言，
   但 lerobot-rollout 要求有 `--task`，且换错文本会让日志/回放对不上。
2. **ACT 是整模型，不是 LoRA adapter**：权重目录里是 `model.safetensors` + `config.json`（没有 adapter_*）。
3. **不需要 `--rename_map`**：ACT 的输入特征就是 `observation.images.{hand,front,top}`，与机器人侧同名
   （pi05 那套才要做 front→base_0_rgb 的映射）。
4. **只能 sync**：ACT 不支持 RTC，别传 `--inference.type=rtc`。
5. 相机序列号来自设备档案（`b601_common/devices/kfzbox.sh`，经 `_common.sh` 注入），换相机改档案，不要改这里。
6. 重新训一轮之后：改三个 `run_act_*.sh` 里的 `DEFAULT_CKPT` + `STEP_INFO`，并把新 sha256 写进本文件。

## 做过的验证（2026-11-19）

- 三个 ckpt 在 kfzbox 上落地，`sha256sum` 与 232 端逐字节一致（各 198 MB / 7 文件）
- 三份脚本 `bash -n` 通过；`run_act_coffee.sh` 无参数时打印任务表
- `ACTPolicy.from_pretrained` 能加载，输入特征为三路相机 + `observation.state`(7)，输出 `action`(7)
- 未做真机试跑（需要人守在现场按 ENTER + 零位确认）
