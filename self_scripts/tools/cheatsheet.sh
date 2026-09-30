#!/bin/bash
# self_scripts 目录速查表（纯注释，不执行任何命令）
#
# 目录按【硬件配置】分组，组内编号即执行顺序：
#   b601_common/           B601 通用：校准 / 串口 / 相机 / 夹爪 / 回放可视化（任何 B601 方案共用）
#   b601_single/           单 B601-RS 臂（从臂 can0 + 主臂 StarArm102）
#   b601_so101_bimanual/   B601-RS 左臂 + SO-101 右臂（异构双臂）
#   so101_single/          单 SO-101 臂（从臂 + 主臂；黄香蕉分拣任务）
#   so101_bimanual/        SO-101 + SO-101 双臂（含老数据集推理）
#   tools/                 通用诊断/可视化工具
#   notes/                 现象记录（延迟问题、端口号映射）
#   _archive/              历史版本（不要再用）
#   _logs/                 运行日志（已 gitignore）
#
# 统一在仓库根目录执行：cd /home/kf/dev/lerobot
#
# ============================ b601_common ============================
# bash self_scripts/b601_common/01_leader_calibration.sh      # 主臂 StarArm102 校准
# bash self_scripts/b601_common/02_follower_calibration.sh    # 从臂 B601-RS 校准（SocketCAN）
# python self_scripts/b601_common/03_gripper_zero_probe.py --phase 1|2   # 夹爪零点漂移判别实验
# python self_scripts/b601_common/04_leader_port.py           # 探测主臂串口号（02_record.sh 内部会调用）
# python self_scripts/b601_common/05_camera_align.py [--cams hand top] [--rebuild-ref]  # 相机对齐检查
# python self_scripts/b601_common/06_replay_rerun.py          # 数据集回放 + rerun 可视化
# self_scripts/b601_common/pr/                                # seeed_b601 单圈夹爪 PR 材料（说明 + patch）
#
# ============================ b601_single ============================
# bash self_scripts/b601_single/01_teleop.sh                  # 单臂遥操作测试（不录数据）
# bash self_scripts/b601_single/02_record.sh                  # 单臂采集（现用版；3 路 JYU2C）
# bash self_scripts/b601_single/02_record.sh --resume <repo_id> [追加集数]   # 续录
# bash self_scripts/b601_single/03_replay.sh                  # 回放已采集数据（可选 rerun 窗口）
# bash self_scripts/b601_single/04_dagger_collect.sh          # DAgger 扩充采集（ACT 自主 + 人工纠错）
# bash self_scripts/b601_single/inference/run_inference_ACT.sh        # 单臂 ACT 推理
# bash self_scripts/b601_single/inference/run_inference_smolvla.sh    # 单臂 SmolVLA 推理
# bash self_scripts/b601_single/inference/pretrained_models/run_inference_single_b601_make_coffee_xvla.sh  # 单臂 X-VLA 推理
# bash self_scripts/b601_single/train/xvla/train_xvla_single_b601.sh [session 名]   # 单臂 X-VLA 微调
# bash self_scripts/b601_single/train/smolvla/start_train_smolvla_v2.sh [--resume|--dry-run|--skip-data-check]  # 单臂 SmolVLA 训练（含数据校验/GPU 健康检查/按 pass 反算 steps；bench/verify_dataset.py + datastet_notes 配套）
#
# ========================= b601_so101_bimanual =======================
# bash self_scripts/b601_so101_bimanual/01_teleop_test.sh     # 异构双臂遥操作测试
# bash self_scripts/b601_so101_bimanual/02_record.sh          # 异构双臂采集（--resume 同单臂）
# python self_scripts/b601_so101_bimanual/03_monitor_motors.py                # 电机状态监视
# bash self_scripts/b601_so101_bimanual/inference/run_inference_pi0.sh        # 异构双臂 pi0 推理
#
# ============================ so101_single ===========================
# bash self_scripts/so101_single/calibrate_101.sh [follower|leader]  # SO-101 单臂校准（默认两条臂都校准）
# bash self_scripts/so101_single/teleop_101.sh                # SO-101 单臂遥操作（不录数据）
# bash self_scripts/so101_single/record_101.sh                # SO-101 单臂采集（2 路 JYU2C；--resume 同单臂）
# bash self_scripts/so101_single/act_inference_101.sh [base]  # SO-101 单臂 ACT 推理（默认边推理边录数据）
# bash self_scripts/so101_single/pi05_inference_101.sh [direct|direct-check|save]   # SO-101 单臂 pi0.5 推理
# python self_scripts/so101_single/pi05_direct_inference_101.py   # 上面 direct 后端的直接控制环（含 shoulder_lift 口径映射）
#
# ========================== so101_bimanual ===========================
# bash self_scripts/so101_bimanual/01_calibrate.sh            # SO-101 双臂校准
# bash self_scripts/so101_bimanual/02_teleoperate.sh          # SO-101 双臂遥操作
# bash self_scripts/so101_bimanual/03_collect_make_coffee.sh [--resume <repo_id> [追加集数]]   # 双臂采集
# bash self_scripts/so101_bimanual/train/start_train_act.sh             # 双臂 make_coffee ACT 训练
# bash self_scripts/so101_bimanual/train/start_train_pi05.sh            # 双臂 cap_pen PI05 LoRA 微调（configs/pi05_train_config_20000steps.json）
# bash self_scripts/so101_bimanual/inference/run_inference_make_coffee_ACT.sh       # 新数据集 ACT 推理
# bash self_scripts/so101_bimanual/inference/run_inference_make_coffee_smolvla.sh   # 新数据集 SmolVLA 推理
# self_scripts/so101_bimanual/inference/old_dataset/          # 老数据集（笔帽盖笔）推理：
#     run_inference_cap_pen_ACT_300k.sh / run_inference_cap_pen_smolvla.sh <4k|30k|110k> / PI0 / PI05
#
# ============================== tools ================================
# python self_scripts/tools/act_feature_viz.py                # ACT 输入/输出特征可视化
# python self_scripts/tools/action_trace.py                   # 推理动作追踪（推理脚本内部调用）
# python self_scripts/tools/plot_action_diag_aligned.py       # 动作对齐诊断图（读 _logs/inference_logs/）
# python self_scripts/tools/check_dataset_validity.py         # 检查数据集 NaN/Inf/极端值（改脚本顶部的 dataset_repo）
# python self_scripts/tools/convert_to_video_format.py        # image 格式数据集 → video 格式（约缩小 6 倍；改脚本顶部配置）
