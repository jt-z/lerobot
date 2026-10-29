# LingBot-VA（`lingbot_va`）单臂推理

B601-RS 单臂的 **LingBot-VA** 推理入口。权重是 **A800-57 上微调**的 b601 版本：Wan2.2 视频扩散栈上的
**视频-动作世界模型**（dual-stream transformer，边预测未来视频 latent 边出动作），微调方式是 **LoRA**
（r=16 / alpha=32 / 13 个线性模块 / 41,925,568 个可训练参数）。

这里的脚本只做**纯推理**（`--strategy.type=base`：不建数据集、不录帧、不落盘）。

---

## 1. 文件

| 文件 | 作用 |
|---|---|
| `env.sh` | 三段权重路径、注意力后端、UMT5 线程数、离线 HF、任务文本、相机 rename_map。**所有脚本的唯一配置源** |
| `_env.py` | `env.sh` 的 Python 侧读取器：允许 `preflight.py` / `smoke_load.py` 单独运行（不必先 `source`） |
| `preflight.py` | 预检（秒级、不占显存）：三段权重是否齐、checkpoint 内路径是否还是训练机的、rename_map 是否一致、超参、依赖 |
| `smoke_load.sh` | 加载自检启动器（`smoke_load.py` 必须用 lerobot env 的解释器跑，直接 `./smoke_load.py` 会用系统 python3） |
| `smoke_load.py` | **加载权重 + 跑 N 次 `select_action`**，打印每块重建耗时、折合每动作耗时/慢放倍数、动作块统计、显存峰值。不碰 CAN/相机 |
| `run_lingbot.sh` | 真机推理：预检 → 参数总览 → ENTER 确认 → `lerobot-rollout`（带 `action_trace.py` 动作追踪）→ 日志落 `self_scripts/_logs/` |

公共硬件逻辑（CAN/相机/校准/参数总览/命令拼装）复用 `../lerobot/_common.sh`，与 ACT / SmolVLA / π0.5 三个脚本同源。

---

## 2. 权重三段（缺一不可）

| # | 路径 | 内容 | 谁在用 |
|---|---|---|---|
| ① | `/home/kf/LX/pai0/lingbot/050000/pretrained_model` | LoRA adapter（7 关键文件 / 160 MB，**没有 model.safetensors**） | 本脚本的 `--policy.path` |
| ② | `/home/kf/LX/pai0/lingbot/lingbot_va_base` | LeRobot 格式基座：只有 5B transformer 的 `model.safetensors`（9.5 GiB） | `adapter_config.json` 的 `base_model_name_or_path` |
| ③ | `/home/kf/LX/pai0/lingbot/lingbot-va-base` | diffusers 风格冻结模块：`vae/`(2.8 GB) + `text_encoder/`(11.3 GB UMT5-XXL) + `tokenizer/` | `config.json` 的 `wan_pretrained_path` |

细节：

- 加载语义（`rollout/context.py::_load_pretrained_policy`）：`use_peft=True` → 先按 `adapter_config.json` 的
  `base_model_name_or_path` 起基座，再 `PeftModel.from_pretrained` 挂 adapter。所以脚本**不需要** `--policy.type`。
- **UMT5-XXL 跑在 CPU 上**（`text_encoder_device=cpu`，省 11 GB 显存），每集只编码一次任务文本；
  线程数必须给足（见 §5 坑 4）。
- `lingbot-va-base/transformer/`（10 GB，上游 diffusers 权重）LeRobot 这条链路**不读**，只有 `vae/ text_encoder/ tokenizer/` 有用。
- **已做的路径改写**（2026-10-23）：checkpoint 从 57 拷来时 3 个 json 里写死 `/home/ubuntu/lxx/pai0/models/...`，
  已 sed 成本机路径，原件留 `*.bak-20261023`（`adapter_config.json` / `config.json` / `train_config.json`）。
  换新 checkpoint 时先跑 `./preflight.py`，它会把仍然指向训练机的路径拦下来并打印该跑的 sed。

---

## 3. 上机四步

```bash
cd /home/kf/dev/lerobot/self_scripts/b601_single/inference/lingbot

# ① 预检（秒级，不占显存）：权重/路径/rename_map/超参/依赖
./preflight.py                        # 期望最后一行 PREFLIGHT_OK

# ② 加载自检（约 1~2 分钟；要 ~14 GB 空闲显存）：权重能拼起来 + 每块重建多久 + 动作像不像样
./smoke_load.sh --steps 48            # 无实拍帧时加 --obs none；期望最后一行 LOADTEST_OK

# ③ 空跑（不碰硬件，验 flag 全链路）：策略加载完成后会停在连接 can99，属正常
DRY=1 DURATION=10 ./run_lingbot.sh    # 期望：Loading policy from ... → Policy loaded → Connecting robot(...)
                                      #       → add_robstride_motor: if_nametoindex failed for can99

# ④ 真机（第一次只跑 20 秒，前 3 秒盯方向）
DURATION=20 ./run_lingbot.sh
```

`③` 之前必须先跑 `②`：`③` 只验证 flag/加载链路，**检不出**视觉特征与延迟问题。

---

## 4. 与 π0.5 / SmolVLA 的关键差异（照抄别的脚本会白跑）

| 项 | π0.5（LoRA）/ SmolVLA | **LingBot-VA** |
|---|---|---|
| 输入是否含 state | 有 | **没有**（`input_features` 只有 3 路相机，`observation.state` 被忽略） |
| 动作语义 | 关节角（`action ≈ -state`） | 同左（本 ckpt 直接拿 b601 关节标签微调，7 维 = 前 7 个动作通道） |
| 相机 | `base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb` | `cam_high / cam_left_wrist / cam_right_wrist`（**front→cam_high, hand→cam_left_wrist, top→cam_right_wrist**） |
| 每块动作数 | 50（`n_action_steps`） | **32**（`frame_chunk_size=2 × action_per_frame=16`）；**首块只有 16 个**（第 0 帧是条件帧） |
| 推理后端 | 可 `rtc` | **只能 `sync`**（`predict_action_chunk` 不接受 RTC 参数，传 rtc 会在 context 构建阶段直接报错） |
| 注意力 | 默认 | ckpt 里存的是 `flex`（**训练专用**），推理必须 `--policy.attn_mode=torch`（脚本已默认覆盖） |
| 依赖 | 无需 diffusers | 需要 `diffusers`（0.39.0，已装进 `~/miniconda3/envs/lerobot`，`--no-deps` 装的，没动别的包） |
| 显存 | ~6–11 GB | **默认 CFG 约 22 GB（24 G 卡装不下）/ 关 CFG 约 17.5 GB**，且必须独占（见坑 1） |
| 每步开销 | 每步 ~100 ms | **每步几乎免费，但每 32 步要重建一次块，开销全在重建上**（见 §5） |

---

## 5. 延迟：开销在"换块"，不在"每步"

`select_action` 平时只是从队列里弹一个动作（微秒级），队列空了才跑一次完整重建。
**每块的前向次数**（决定真机放慢倍数）是：

```
前向次数 ≈ (num_inference_steps + 1) + (action_num_inference_steps + 1)   ×  (CFG>1 ? 双 batch : 1)
```

**4090D 实测（2026-10-23，`GUIDANCE=1.0`，视频 25 / 动作 50 步，实拍帧）**：

| 事件 | 耗时 | 说明 |
|---|---|---|
| 权重加载 | 20 s | 5B transformer + VAE 进卡，约 13 GB |
| **首块**（第 1~16 个动作） | **57.2 s** | UMT5-XXL 首次文本编码（CPU）+ VAE + 首块 77 次前向，**一次性** |
| 换块（第 17、49 步） | **4.07 / 4.16 s** | 稳态就是它：32 个动作 / 块 |
| 每步（队列弹出） | 4 ms（p50） | 几乎免费 |
| 显存峰值 | **19.0 GiB**（保留 20.1） | `guidance=1.0` 才压得进 24 G |

⇒ 稳态 **128 ms/动作**，30 Hz（33.3 ms/动作）下约 **3.8× 慢放**（可接受）；首块那 57 s 是每集一次的开销。
⇒ **首块没算进"慢放"**：`smoke_load` 打印的 `682 ms/动作 / 20.5×` 是把首块 57 s 平均进去的，看稳态请用 4.1 s/块。

默认配置 `num_inference_steps=25 / action_num_inference_steps=50 / guidance_scale=5.0`：

- 前向次数 = 26 + 51 = **77 次**，每次都是 5B transformer **双 batch**（CFG 开着）；
- 参照量级：57 上 5B 单次前向 ≈ 0.18 s（A800、batch 2、~1024 token），4090D 只会更慢；
  ⇒ 每块粗估 **15–25 s**，而 32 个动作在 30 Hz 下只对应 **1.07 s** 的轨迹
  ⇒ **真机会是十几倍慢放**（臂能动，但一格一格挪）。

想快就动这三个开关（脚本都支持环境变量覆盖，**动作质量需自己用闭环验**）：

| 开关 | 默认 | 快档 | 效果 |
|---|---|---|---|
| `VIDEO_STEPS` | 25 | 5 | 前向 26→6 |
| `ACTION_STEPS` | 50 | 10 | 前向 51→11 |
| `GUIDANCE` | 5.0 | **1.0（24 G 卡上必须）** | 关 CFG：前向从双 batch 变单 batch（≈再快一倍），**且 KV cache 减半 —— 24 G 卡上这步是必须的，不只是提速** |

```bash
VIDEO_STEPS=5 ACTION_STEPS=10 GUIDANCE=1.0 DURATION=20 ./run_lingbot.sh   # 前向 77→17，预计每块 2~3 s
./smoke_load.sh --video-steps 5 --action-steps 10 --guidance 1.0          # 先在离线看延迟，不上机
```

`--fps=30` 保持不动：动作是逐帧绝对目标角，控制环跟不上只会让动作变慢，不会改变目标本身。

---

## 6. 已知坑

1. **显存是第一硬约束：24 G 单卡必须独占，且默认配置装不下**（2026-10-23 实测两轮 OOM）。
   KV cache 是主因，`create_empty_cache` 给 30 个 block 各开 k/v：

   ```
   total_token_len = (attn_window/2) × (latent_token/chunk + action_token/chunk) = 36 × (640 + 32) ≈ 24.2k
   KV ≈ 2(k,v) × 24192 × 24 heads × 128 dim × 2 B × 30 blocks × batch
      batch=2（CFG 开，guidance 5.0）→ 8.9 GB ；batch=1（guidance 1.0）→ 4.45 GB
   ```

   | 配置 | 5B transformer | VAE | KV | 合计（+激活） | 24 G 卡 |
   |---|---|---|---|---|---|
   | `guidance=5.0`（ckpt 原值） | 10.2 G | 2.8 G | 8.9 G | **≈22 G+** | ❌ 实测 OOM（独占整卡也差） |
   | `GUIDANCE=1.0` | 10.2 G | 2.8 G | 4.45 G | ≈17.5 G | ✅ 可跑 |
   | `GUIDANCE=1.0` + 与 π0.5 server 共存 | — | — | — | 17.5 + 8.4 = 25.9 G | ❌ 装不下 |

   ⇒ **4090 上 lingbot 与 8000 端口的 π0.5 server 互斥**，跑之前先停：
   ```bash
   for p in $(pgrep -f 'openpi_env/bin/python .*serve_polic[y]\.py'); do kill $p; done   # 停 π0.5 server
   pkill -f 'rerun_cli/rerun' || true                                                    # 停可视化残留（可选）
   ```
   起回来按 π0.5 那边的启动脚本（`/home/kf/b601_pi05/start_serve.sh` 或 README（openpi 目录））。
2. **`attn_mode` 不能照抄 ckpt**：训练用 `flex`（需要 torch flex-attention + compile），推理用 `torch`(SDPA)。
   脚本默认已覆盖；若手改 `--policy.attn_mode=flex` 报 flex 相关错误，改回去。
3. **首块只有 16 个动作**，之后每块 32 个。判读动作序列长度时别当成"模型输出不齐"。
4. **UMT5 的 CPU 线程**：`OMP_NUM_THREADS=1` 时一次文本编码要几分钟（57 实测 >240 s vs 16 线程 21.9 s）。
   `env.sh` 已设 16；在别的终端里 `OMP_NUM_THREADS=1` 手工起脚本会踩到。
5. **`DRY=1` 停在 can99 是正常的**：`add_robstride_motor: if_nametoindex failed for can99` 说明 flag 全过了。
6. **动作约等于 `-state`**：`|action + state|` 小 + 行程小 = 模型在输出"保持不动"（数据集 `joint_directions` 全 -1 所致），
   不是链路坏了。判"在动"看 **32 步块的行程**。
7. **训练机代码与本机差一个补丁**：57 的 `modeling_lingbot_va.py` 比本机多一处 UMT5 **prompt embeds 缓存**
   （纯训练提速，50k 步从 625 h 降到 10 h）。推理每集只编码一次，**不需要**这个补丁，别把 57 的文件拷回来覆盖。
8. **首块 57 s：`DURATION` 不要给 20 秒**。控制环从起跑开始计时，第一个 `select_action` 要阻塞 57 s
   （UMT5 CPU 编码 + 首块），`DURATION=20` 会在出第一个动作之前就到点退出，看起来像"啥也没干"。
   第一次上机用 `DURATION=180`，或先在离线跑一次 `smoke_load.sh` 让 UMT5 预热（同一进程内才有效）。
9. **`--policy.save_predicted_video=true` 会 VAE 解码预测视频**（世界模型的"想象画面"），多花时间和显存；
   要可视化再加，默认关。

---

## 7. 已验证 / 未验证（2026-10-23）

已验证（本机、实测）：

- `preflight.py` 全绿：16 项通过（1 条警告 = ckpt 里 `attn_mode=flex`，推理已覆盖为 `torch`）。
- 权重能拼起来：`LOAD_OK PeftModel`，20–21 s，`lora_params=41,925,568`，device=cuda:0；
  基座 9.5 GiB + VAE 进显存后进程约 13 GB。
- **`GUIDANCE=1.0` 跑通 64 步**：`LOADTEST_OK`，显存峰值 19.0 GiB，稳态 4.1 s/块（128 ms/动作，3.8× 慢放），
  动作有意图（shoulder_lift 行程 46°、elbow 37°、gripper 82°，不是"保持不动"）。
- 两次 OOM 实测：与 π0.5 server 共存时 13.2 G 触顶；独占整卡、`guidance=5.0` 时 21.7 G 触顶
  （都在 `create_empty_cache` → `init_kv_cache`）⇒ 见坑 1 的显存表。
- pre/post processor（含 `rename_map`）能按 rollout 的方式装载。
- `diffusers 0.39.0` 已装（与本机 `torch 2.11.0+cu128 / transformers 5.5.4 / peft 0.20.0` 兼容）。
- `DRY=1 DURATION=10 ./run_lingbot.sh` 全 flag 链通：预检 → 参数总览 → `Loading policy` →
  **`Policy loaded: type=lingbot_va, device=cuda`（21 s）** → `Connecting robot(...)` →
  `if_nametoindex failed for can99`（预期终止，物理上动不到臂）。日志：
  `self_scripts/_logs/inference_b601_make_coffee_lingbot_050000_20261023_141234.log`。
- `DURATION` / `FPS` / `ACTION_TRACE` 三个环境变量能从命令行直接覆盖（脚本已避开 `_common.sh` 硬赋值的坑）。

未验证（需空闲显存 / 需真机）：

- **每块重建的实际耗时**（`smoke_load.sh` 会打印 p50/max、换块次数、折合 ms/动作、慢放倍数）。
- 真机行为：起步、方向、跟随、抖动（`action_trace` 日志里有 `model/sent/cmd` 三列可对比）。

建议验证顺序：腾卡 → `./smoke_load.sh --steps 48` 拿到延迟 → `./smoke_load.sh --video-steps 5 --action-steps 10 --guidance 1.0`
对比 → 选一组能接受的上机 `DRY=1` → `DURATION=20` 真机。
