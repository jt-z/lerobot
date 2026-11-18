# 分析：sync 与 rtc 的选型与动作差异

目的：解释「为什么 sync 动作更贴、rtc 更顺但有时发黏」，给出 3060 上的选型依据。
相关：[时延与显存](pi05_分析_时延与显存.md)、[推理链路](../03_模型架构/pi05_机制_推理链路.md)。

## 一、一句话结论

sync 是 pi05 的原生推理方式，输出 100% 是模型采样结果，代价是每 1.667s 硬停一次；
rtc 用后台线程预取消除停顿，但在这个**未做 RTC 训练**的 checkpoint 上只能用 `guided` 近似，属于推理期补偿，会改变输出分布。

## 二、机制对比

| | sync | rtc（guided） |
|---|---|---|
| 调度 | 每段播完阻塞推理，算完再播 | 队列剩 35 步时后台预取，不阻塞控制环 |
| 停顿 | chunk 时延（3060 解锁态 354ms） | 0 |
| 输出 | 模型原始采样 | 在去噪速度场里被上一段计划引导 |
| 每段实际执行 | 完整 50 步 | **丢弃前 `real_delay`（实测 15~17）步，只播 33~35 步** |
| 视觉滞后 | 观测到执行约 0.35s | 观测到接管约 1.17s |
| 3060 实测 chunk | 353.9ms | 501.3ms |

`rtc` 的两种模式：`trained`（训练时见过延迟条件）与 `guided`（免训练近似）。
本模型 `rtc_training_max_delay = 0`，强行 `--inference.rtc.mode=trained` 会直接
`ValueError: RTC mode='trained' requires a checkpoint trained with policy.rtc_training_max_delay > 0`，
**可运行的只有 guided**。

## 三、为什么 rtc 的动作观感更「黏」

1. **guided 修正覆盖 40% 的步数**：`execution_horizon=20`，新 chunk 前 20 步被朝上一段剩余计划拉（`linear` 权重、`max_guidance_weight=10`）。两者不一致时输出是折中轨迹，不是任何一段的原始意图。
2. **模型规划的后段被截掉**：每段丢前 15~17 步，只播 33~35 步，模型在 50 步里规划的减速/收尾/松爪阶段基本不执行，于是更「急」、更易冲过头。
3. **反馈更旧**：预取用的是「还剩 35 步」时的观测，新 chunk 接管时视觉信息已过去约 1.17s。丢步只对齐了时间轴，没有对齐感知。对按按钮这类需要视觉闭环的动作最不利。
4. **宿主争用让 tick 不均**：后台推理线程与 30Hz 主循环 / 三路 MJPG 解码 / CAN 通信抢 CPU，denoise 从 199ms 涨到 302ms，控制节拍更抖。

落地经验：**搬运类（任务 1/2/3）对上述差异不敏感，按按钮类（任务 0）最敏感**。

## 四、openpi 原生就是同步

- 训练：flow matching 对整段 50 步 chunk 做速度场回归，**没有延迟条件、没有 RTC 掩码**
- 推理：`action_chunk = policy.infer(example)["actions"]`，一次请求返回整段，客户端播完再请求下一段——没有后台线程、队列、预取
- RTC 是 PI 后续单独发布的研究成果，不在 openpi 默认路径里，是 lerobot 以 `inference.type=rtc` 接进来的附加能力

所以「openpi 原生 = sync」成立；也解释了为什么这个 checkpoint 只能跑 guided。

## 五、为什么 3060 上停顿明显，4090 上「像没停」

停顿机制相同，只是时长不同；感知差异来自 tick 数：

| | 3060（实测） | 4090（估算） |
|---|---|---|
| embed | 38ms | 25~35ms（含 CPU 预处理，不线性缩放） |
| prefix_fwd | 116ms | 70~90ms |
| denoise | 199ms | 60~80ms |
| **停顿** | **354ms（11 tick）** | **约 150~220ms（5~7 tick）** |

- 354ms = 11 个 tick 完全不动，肉眼可见
- 150~200ms = 5~6 个 tick，通常只表现为一次轻微顿挫
- 因此 4090 不是「连续」，而是停顿短到不容易察觉；机制仍是同步

## 六、选型建议

| 场景 | 选择 | 理由 |
|---|---|---|
| 验证策略能力、成功率优先 | **sync** | 原始输出，无 guided 修正、无截断 |
| 演示场合不能有停顿 | rtc | 停顿彻底消失 |
| rtc 但担心滞后 | rtc + `RTC_QUEUE_THRESHOLD=24` | 滞后 1.17s → 0.8s，代价是余量剩约 6 步 |
| 换 4090/5090 | sync | 停顿压到 150~220ms，无需引入 rtc 风险 |
| 想让 sync 停顿更短 | 压 denoise（compile / 8 步） | 最多压到约 400ms 级，无法归零 |

参数约束：`RTC_EXECUTION_HORIZON` 必须 ≥ 实测 `real_delay`（约 16），当前 20 合适，**不要调到 24 以上**，否则「黏」感更强。

## 七、怎么验证（不要只看顺不顺）

1. 固定一个子任务（建议任务 1，4/4 已成功），sync / rtc 各跑 5~10 次，只改 `INFERENCE_TYPE`
2. 记录：成功与否、失败模式、chunk 数、时延 p50/p95、`real_delay` 最小值
3. 判定：rtc 成功率持平 → 切 rtc；出现 ≥2 次「犹豫/放偏」→ 保持 sync

命令：

```bash
TASK_INDEX=1 INFERENCE_TYPE=sync bash lx/pi05_inference.sh
TASK_INDEX=1 bash lx/pi05_inference.sh          # 默认 rtc
```
