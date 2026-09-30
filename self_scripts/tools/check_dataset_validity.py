#!/usr/bin/env python3
"""检查数据集中是否存在NaN或无穷值"""
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

dataset_repo = "hellozjt/demo_data_dual"
print(f"加载数据集: {dataset_repo}")

dataset = LeRobotDataset(dataset_repo)

print(f"\n数据集信息:")
print(f"  - Episodes: {dataset.num_episodes}")
print(f"  - Frames: {dataset.num_frames}")
print(f"  - Features: {list(dataset.features.keys())}")

# 检查前100个样本
print("\n检查前100个样本是否存在异常值...")
issues = []

for idx in range(min(100, len(dataset))):
    item = dataset[idx]

    # 检查动作
    if 'action' in item:
        action = item['action']
        if torch.isnan(action).any():
            issues.append(f"Sample {idx}: action contains NaN")
        if torch.isinf(action).any():
            issues.append(f"Sample {idx}: action contains Inf")
        if (torch.abs(action) > 1e6).any():
            issues.append(f"Sample {idx}: action has extreme values: {action}")

    # 检查图像
    for key in item.keys():
        if 'image' in key.lower() or 'observation' in key.lower():
            img = item[key]
            if torch.isnan(img).any():
                issues.append(f"Sample {idx}: {key} contains NaN")
            if torch.isinf(img).any():
                issues.append(f"Sample {idx}: {key} contains Inf")

if issues:
    print("\n⚠️ 发现数据问题:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\n✓ 前100个样本未发现NaN或Inf异常值")

# 检查统计信息
print("\n数据统计信息:")
if hasattr(dataset, 'stats'):
    print(f"Stats keys: {dataset.stats.keys()}")
    for key, value in dataset.stats.items():
        if isinstance(value, dict) and 'mean' in value:
            mean = value['mean']
            std = value.get('std', None)
            if isinstance(mean, (list, np.ndarray, torch.Tensor)):
                mean = np.array(mean) if not isinstance(mean, np.ndarray) else mean
                print(f"  {key}:")
                print(f"    mean: {mean}")
                if std is not None:
                    std = np.array(std) if not isinstance(std, np.ndarray) else std
                    print(f"    std: {std}")
                    if (std == 0).any():
                        print(f"    ⚠️ WARNING: Zero std detected!")
