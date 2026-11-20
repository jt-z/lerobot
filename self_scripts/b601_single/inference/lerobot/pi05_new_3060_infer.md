3060 上现在用的是 232 训练的 pi05_coffee_expo ckpt 008000，已经合并成完整 bf16 权重（没有 int4）：

| 项   | 值                                                                                                                            |
|------|-------------------------------------------------------------------------------------------------------------------------------|
| 权重 | kfzbox:~/pi05_bf16/expo008000/model.safetensors（7.73 GiB，812 张量）                                                         |
| 来源 | 232 output_lerobot_train/pi05_coffee_expo_181510_r64_lr1e3/checkpoints/008000（LoRA r=64/α=8 合并进 pi05_base）               |
| 校验 | sha256 ddda200f0beaeaa4cb9023899d0ae6d173fc14d87c59fdbd5dcd557c7857fe78（与 232 一致）                                        |
| 配套 | config.json / extras.pt（non-persistent buffer）/ batch_expo.pt（8 帧验证输入）/ adapter_model.safetensors（原始 LoRA，留档） |

执行方式（bf16，一条命令）

bash
ssh kfzbox
cd ~/pi05_int4

8 帧一批（默认）
~/pi05venv/bin/python verify_pi05.py \
  --ckpt expo008000 \
  --batch expo008000/batch_expo.pt \
  --noise expo008000/batch_expo_noise.pt \
  --reps 3 --report results/verify_expo008000.json

单帧（刚才也验过：569 ms/次，峰值 8.95 GiB）
~/pi05venv/bin/python verify_pi05.py --ckpt expo008000 \
  --batch expo008000/batch_expo.pt --noise expo008000/batch_expo_noise.pt \
  --bs 1 --reps 3


实测（3060 12GB，bf16）：

| 批量 | 时延   | 推理峰值 |
|------|--------|----------|
| 1 帧 | 569 ms | 8.95 GiB |
| 8 帧 | 2.99 s | 9.97 GiB |

脚本会打印：相机键（base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb）、chunk 50、10 步去噪、完整 prompt、与 GT 的逐关节角度 MAE、块行程/平滑度、时延、峰值显存，并检查同噪声两次是否逐位一致。

跑之前不需要装 lerobot——依赖只有 ~/pi05venv（torch 2.13 + transformers 5.5.4）+ 同目录的 pi05_min.py / int4lib.py。

一句提醒：--bs 1 那条的 gt_mae_deg 是 24.0°（8 帧全跑是 31.6°），这个离线偏差仍未定性（合并 vs ckpt 本身），我上一轮说的 peft 等价性核对还没做——你要跑真机前建议先定这件事。
