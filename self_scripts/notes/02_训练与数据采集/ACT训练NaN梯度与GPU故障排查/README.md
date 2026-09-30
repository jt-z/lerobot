# ACT训练NaN梯度与GPU故障排查

**一句话概述**：LeRobot ACT模型训练时出现NaN梯度导致CUDA崩溃，排查发现VAE配置过激进，同时GPU 0硬件失效，最终调整为7卡+稳定配置方案。

---

## 文件索引

| 文件名 | 内容 |
|--------|------|
| `00-项目元数据.md` | 时间线、问题记录、关键决策 |
| `01-问题诊断与根因分析.md` | NaN梯度问题的完整分析 |
| `02-稳定配置方案.md` | 修改的训练配置及原理 |
| `03-GPU故障处理.md` | GPU 0失效问题的诊断与应对 |
| `04-最终结论与最佳实践.md` | 总结和可复用经验 |
| `DISTILLED.md` | 精炼版知识点 |

---

## 脚本文件

| 脚本 | 用途 | 使用方式 |
|------|------|---------|
| `train_config_act_stable.json` | 稳定训练配置 | 用作 `--config_path` 参数 |
| `start_train_act_stable_7gpu.sh` | 7卡训练启动脚本 | `bash start_train_act_stable_7gpu.sh` |
| `check_dataset_validity.py` | 数据集验证工具 | `python check_dataset_validity.py` |
| `reset_gpu_and_train.sh` | GPU重置+训练脚本 | `bash reset_gpu_and_train.sh` |

---

## 快速结论

### 问题本质
NaN梯度来源于VAE的KL散度计算中 `exp(log_sigma_x2_hat)` 数值爆炸

### 关键修改
| 参数 | 原始值 | 稳定值 | 原因 |
|------|--------|--------|------|
| kl_weight | 10.0 | **1.0** | 降低VAE训练强度 |
| batch_size | 32 | **16** | 减小梯度方差 |
| optimizer_lr | 8e-5 | **5e-5** | 更保守的更新 |
| n_vae_encoder_layers | 4 | **2** | 简化架构 |
| grad_clip_norm | 10.0 | **5.0** | 更严格限制 |

### 硬件问题
- GPU 0 (0000:55:00.0) 完全失效
- 改用 `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7` 继续训练
- 有效批次大小：128 (16×8) → 112 (16×7)

### 推荐方案
```bash
cd /home/ksa/lerobot/self_scripts
bash start_train_act_stable_7gpu.sh
```
