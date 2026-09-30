# SmolVLA 推理笔记索引

本目录包含 SmolVLA 模型推理过程的详细分析笔记，按主题分类。

## 笔记列表

### 1. [推理流程总览](01_推理流程总览.md)
整体架构概览，从脚本入口到模型输出的完整流程。
- 入口脚本与命令结构
- 推理循环主流程
- 预处理/后处理流水线
- 一次完整推理的时间线

### 2. [参数配置处理](02_参数配置处理.md)
命令行参数如何解析、传递并最终影响模型行为。
- 参数传递路径总览
- 配置类层次结构
- from_pretrained 加载流程
- 脚本参数到配置的完整映射表

### 3. [文字描述处理流程](03_文字描述处理流程.md)
`--dataset.single_task` 中的文本如何从字符串转换为模型输入。
- 文本传递的完整路径
- TokenizerProcessorStep 详解
- 48 个 token 的限制说明
- 注意力掩码设计

### 4. [Action Chunking 机制](04_Action_Chunking机制.md)
为什么模型一次输出 50 个动作，而每帧只执行一个。
- 基本概念与动机
- select_action 核心实现
- 时间计算与时间线
- 优缺点分析

### 5. [SmolVLA 模型架构](05_SmolVLA模型架构.md)
模型内部结构，从嵌入到 Flow Matching 采样的完整过程。
- 整体架构图
- 前缀嵌入 (embed_prefix)
- 后缀嵌入 (embed_suffix)
- Flow Matching 采样
- 单步去噪 (denoise_step)
- SmolVLMWithExpert 双路结构
- 训练 vs 推理对比

## 快速参考

### 关键配置参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `chunk_size` | 50 | 模型一次输出的动作数 |
| `n_action_steps` | 50 | 每次推理后执行的步数 |
| `num_steps` | 10 | Flow Matching 去噪步数 |
| `tokenizer_max_length` | 48 | 文本最大 token 数 |
| `empty_cameras` | 0 | 空相机填充数量 |
| `max_state_dim` | 32 | 状态向量最大维度 |
| `max_action_dim` | 32 | 动作向量最大维度 |

### 关键代码位置

| 功能 | 文件 | 函数/类 |
|------|------|---------|
| 推理入口 | `lerobot_record.py` | `record_loop()` |
| 核心推理 | `control_utils.py` | `predict_action()` |
| 动作选择 | `modeling_smolvla.py` | `SmolVLAPolicy.select_action()` |
| 模型推理 | `modeling_smolvla.py` | `VLAFlowMatching.sample_actions()` |
| 前缀嵌入 | `modeling_smolvla.py` | `VLAFlowMatching.embed_prefix()` |
| 文本分词 | `tokenizer_processor.py` | `TokenizerProcessorStep` |
| 模型工厂 | `factory.py` | `make_policy()` |
