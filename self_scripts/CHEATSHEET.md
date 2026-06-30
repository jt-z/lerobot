# LeRobot 环境速查手册

> 机器：`kfzbox`  |  用户：`jt`  |  Conda 环境：`lerobot` (Python 3.12)

---

## 1. 关键目录

| 路径 | 说明 |
|------|------|
| `~/dev/` | 开发根目录（含 `lerobot/`、`miniforge3/`、`test_cuda.py`） |
| `~/dev/lerobot/` | LeRobot 代码仓库（git repo） |
| `~/dev/lerobot/self_scripts/` | 自己的脚本/笔记目录（本文件所在处） |
| `~/dev/miniforge3/` | Conda 安装目录，环境在 `envs/lerobot/` |
| `~/dev/shared_weights/` | 训练好的模型权重 |
| `~/.cache/huggingface/lerobot/` | HF 缓存：`calibration/`（标定）、`TommyZihao/`（数据集） |
| `~/.cache/huggingface/token` | HF 登录 token 保存位置 |

### 已有模型权重 (`~/dev/shared_weights/`)
- `zihao_shake_pi0_50k_pretrained_model`  — Pi0，50k step，握手任务
- `zihao_shake_act_30k_pretrained_model`  — ACT，30k step，握手任务
- `zihao_orange_pretrained_model`         — 橙子任务

### 单个模型目录内容（标准结构）
```
config.json                                                  # 策略结构配置
model.safetensors                                            # 模型权重
train_config.json                                            # 训练配置
policy_preprocessor.json                                     # 预处理流水线
policy_preprocessor_step_5_normalizer_processor.safetensors  # 输入归一化参数
policy_postprocessor.json                                    # 后处理流水线
policy_postprocessor_step_0_unnormalizer_processor.safetensors  # 输出反归一化参数
```

---

## 2. 设备 (机械臂串口)

```
/dev/ttyACM0   # 机械臂 1（owner root:dialout）
/dev/ttyACM1   # 机械臂 2
```

查看串口设备：
```bash
ls -lah /dev/ttyA*
```

授权（每次插拔/重启后端口权限会复位）：
```bash
sudo chmod 666 /dev/ttyA*
# 一劳永逸：把用户加入 dialout 组（需重新登录生效）
sudo usermod -aG dialout $USER
```

---

## 3. 常用命令

### 环境
```bash
conda activate lerobot
cd ~/dev/lerobot && git pull          # 更新代码
```

### Hugging Face
```bash
hf auth login                          # 登录（旧 huggingface-cli login 已废弃）
hf auth whoami
hf download <repo>                     # 下载
```
> 当前 token 名：`lerobot`，权限 fineGrained。
> 注意：`git-credential` helper 未配置，push 时可能需重新认证。如需保存：
> `git config --global credential.helper store`

### 清理数据集缓存
```bash
sudo rm -rf ~/.cache/huggingface/lerobot/TommyZihao/eval_lerobot_zihao_dataset_shake_hands
```

---

## 4. ⚠️ 依赖版本坑（重要）

**不要随意 `pip install --upgrade transformers huggingface_hub`！**

升级后会把 `huggingface_hub` 拉到 `1.20.1`，与 LeRobot 冲突：

```
lerobot 0.4.4 requires huggingface-hub<0.36.0,>=0.34.2,
but you have huggingface-hub 1.20.1 which is incompatible.
```

连锁反应：升级常常会把 `numpy`、`fsspec`、`packaging` 也一起拉到过新版本，导致多个冲突。

✅ 一键修复（锁回兼容版本，已验证 `pip check` 通过）：
```bash
pip install "huggingface_hub>=0.34.2,<0.36.0" \
            "numpy>=2,<2.3.0" \
            "fsspec[http]>=2023.5.0,<=2025.9.0" \
            "packaging>=24.2,<26.0"
```

验证：
```bash
pip check            # 应输出 No broken requirements found
python -c "import lerobot; print('ok')"
```

> 已验证可用基线：lerobot 0.4.4 / hf_hub 0.35.3 / transformers 4.53.3 /
> tokenizers 0.21.4 / numpy 2.2.6 / fsspec 2025.9.0 / packaging 25.0

---

## 5. pip 镜像源

当前使用清华源：
```
https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```
