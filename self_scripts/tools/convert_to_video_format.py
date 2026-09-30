#!/usr/bin/env python3
"""
将 image 格式的数据集转换为 video 格式（分离存储）
这会大幅减小数据集大小（从 3.8GB 降到约 600MB）
"""

from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import convert_image_to_video_dataset

# 配置参数
INPUT_DATASET = "hellozjt/demo_data_dual"  # 输入数据集
OUTPUT_DIR = Path.home() / ".cache/huggingface/lerobot/hellozjt/demo_data_dual_video"  # 输出目录
OUTPUT_REPO_ID = "hellozjt/demo_data_dual_video"  # 输出数据集 ID

# 视频编码参数（可调整）
VIDEO_CODEC = "libsvtav1"  # 视频编码器 (libsvtav1/h264/hevc)
CRF = 30  # 压缩质量 (18-35, 越低质量越好但文件越大)
NUM_WORKERS = 4  # 并行处理线程数

def main():
    print(f"加载数据集: {INPUT_DATASET}")
    dataset = LeRobotDataset(INPUT_DATASET)

    print(f"\n数据集信息:")
    print(f"  - 总 episodes: {dataset.meta.total_episodes}")
    print(f"  - 总帧数: {dataset.meta.total_frames}")
    print(f"  - 机器人类型: {dataset.meta.robot_type}")
    print(f"  - 图像 keys: {dataset.meta.image_keys}")
    print(f"  - FPS: {dataset.meta.fps}")

    print(f"\n开始转换为视频格式...")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  视频编码: {VIDEO_CODEC}, CRF={CRF}")

    # 执行转换
    new_dataset = convert_image_to_video_dataset(
        dataset=dataset,
        output_dir=OUTPUT_DIR,
        repo_id=OUTPUT_REPO_ID,
        vcodec=VIDEO_CODEC,
        pix_fmt="yuv420p",
        g=2,  # GOP size
        crf=CRF,
        fast_decode=0,
        num_workers=NUM_WORKERS,
        episode_indices=None,  # None = 转换所有 episodes
    )

    print(f"\n✅ 转换完成!")
    print(f"新数据集路径: {OUTPUT_DIR}")

    # 检查大小
    import subprocess
    result = subprocess.run(
        ["du", "-sh", str(OUTPUT_DIR)],
        capture_output=True,
        text=True
    )
    print(f"新数据集大小: {result.stdout.strip().split()[0]}")

    print(f"\n使用方法:")
    print(f"  dataset = LeRobotDataset('{OUTPUT_REPO_ID}')")

if __name__ == "__main__":
    main()
