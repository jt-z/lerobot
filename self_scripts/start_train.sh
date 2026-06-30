#!/bin/bash
# 启动训练
cd /home/ksa/lerobot/self_scripts/
accelerate launch   --multi_gpu   --num_processes=8   $(which lerobot-train)   --config_path=train_config.json