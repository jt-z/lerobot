# b601 单臂 makecoffee 数据集笔记

主题：`/data/share/b601_20260910_164106`（单臂 `seeed_b601_rs_follower`，7 维 action/state，
3 路相机 hand/front/top，30 fps）的**源端增量传输状态**、**`file-009.parquet` 损坏复现与修复**、
**校验方法**。

配套训练工程（ACT 训练配置、100K run 分析、多卡扩展性）见上级目录
[`../README.md`](../README.md)，本目录不重复那部分内容。

## 文件索引

| 文件 | 内容 |
|---|---|
| [`00-项目元数据.md`](00-项目元数据.md) | 基本信息、完整时间线、问题与解决、关键决策、路径速查 |
| [`01-数据集现状与传输状态.md`](01-数据集现状与传输状态.md) | 当前规模、增量批次、传输是否完成的判断方法 |
| [`02-损坏机理-物理行号等于index.md`](02-损坏机理-物理行号等于index.md) | 根因、为什么必须在数据 parquet 层修、影响面 |
| [`03-file009-修复操作.md`](03-file009-修复操作.md) | 可复现的修复步骤与逐项校验门槛 |
| [`04-校验方法与结果.md`](04-校验方法与结果.md) | 结构校验 + 解码校验的方法与实测结果 |
| [`05-预防与治本方案.md`](05-预防与治本方案.md) | 为什么每次重传都会复现、两个治本选项 |
| [`06-源端修复记录.md`](06-源端修复记录.md) | **治本选项 A 已执行**：源端修复全过程、证据链、跨机器验证、回滚方式 |
| [`DISTILLED.md`](DISTILLED.md) | 蒸馏版：规律、决策、踩坑精华 |

## 脚本

| 脚本 | 用途 | 用法 |
|---|---|---|
| [`scripts/fix_file009.py`](scripts/fix_file009.py) | 幂等修复：丢弃 file-009 末尾多的 11,005 行 row group，先备份再原子替换 | `python scripts/fix_file009.py --root /data/share/b601_20260910_164106`（加 `--dry-run` 只检查） |
| [`scripts/verify_dataset.py`](scripts/verify_dataset.py) | 结构校验（不需要 GPU）；`--decode` 追加 `LeRobotDataset` 实拉解码校验 | `python scripts/verify_dataset.py --root ... [--decode]` |

### 自动钩子（已安装）

[`../bench/post_sync_check.sh`](../bench/post_sync_check.sh) + [`../bench/install_hook.sh`](../bench/install_hook.sh)
—— 每 10 分钟轮询，检出损坏即自动修复并记日志，**不需要人工记得跑**。
三重守卫（单实例 / 数据静默 ≥5 分钟 / 无训练在跑）+ 指纹短路（没变就零成本退出）。
详见 [`05-预防与治本方案.md`](05-预防与治本方案.md) 的「已实现」小节。

```bash
bash bench/install_hook.sh status     # 看状态与最近日志
bash bench/install_hook.sh run        # 立即跑一次
bash bench/install_hook.sh uninstall  # 卸载
```

### 与 `bench/verify_dataset.py` 的分工

仓库里有**两份** `verify_dataset.py`，刻意分工，**不要互相覆盖**：

| | 本目录 `scripts/` | [`../bench/verify_dataset.py`](../bench/verify_dataset.py) |
|---|---|---|
| 定位 | **人读报告**：5 项编号检查，便于人工判读 | **CI / 同步钩子** |
| 改数据 | ❌ 只读 | ✅ `--fix`（落盘前先验证） |
| 视频检查 | ❌ | ✅ `--video`（ffprobe 容器帧数 vs meta） |
| 机器可读 | ❌ | ✅ `--json`（stdout 纯 JSON） |
| 检测判据 | 文件边界 `物理起始行号 == index`（**症状级**） | `index` 值**重复**（**根因级**，精确定位 1 个 row group） |

> ⚠️ **改判据时两份都要同步。** 另外注意：本目录那份的判据 2 是**症状级** ——
> 那个残留块会把 file-010 之后**所有**文件整体推移，所以「错位文件数」会远多于 1，
> **不要据此决定删除范围**。真正多余的只有一处残留 row group，其 `index` 值
> 与后面合法行的 `index` 一一重复（`bench/` 那份用这个判据定位）。

> ⚠️ 已知本目录名 `datastet_notes` 是 `dataset_notes` 的笔误（按用户给定路径原样创建）。

## 快速自检

```bash
# 一行判断数据集有没有坏（不等就说明重传又把损坏带回来了）
python scripts/verify_dataset.py --root /data/share/b601_20260910_164106
```

## 一句话结论

**源端每次重传都会把 `data/chunk-000/file-009.parquet` 覆盖回损坏版（末尾多 11,005 行）**；
`LeRobotDataset` 靠「物理行号 == `index` 值」定位 episode，这个块会让 **ep162 之后约
112,698 帧（≈21%）** 静默错配——训练不报错、loss 曲线正常，但学的是错的。
**同步后先跑 `verify_dataset.py`，`总行数 == meta total_frames` 成立再开训。**
