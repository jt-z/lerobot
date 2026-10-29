#!/usr/bin/env bash
# 四个 ckpt × 两个种子，同一 prime、同一时长，跑完输出对比表
# 用法: bash sweep_ckpts.sh [时长秒] [种子列表]
#   默认: 25 秒, 种子 7 99
set -uo pipefail
cd ~/dev/lerobot/self_scripts/b601_single/inference/openpi
DUR=${1:-25}
SEEDS=${2:-"7 99"}
PRIME=3.67,1.47,1.6,0.04,12.88,3.82,0.2      # ep100 示范首帧
OUT=~/b601_pi05/logs/sweep_$(date +%H%M%S).txt
CKPTS="lr24k lr36k lr48k lr60k"

echo "扫描: $CKPTS | 时长 ${DUR}s | 种子 $SEEDS | prime=$PRIME" | tee "$OUT"
echo "开始 $(date +%H:%M:%S)" | tee -a "$OUT"

for ck in $CKPTS; do
  for sd in $SEEDS; do
    LOG=$(ls -t ~/b601_pi05/logs/ab_D_*.log | head -1)
    echo "--- $ck seed=$sd ---" | tee -a "$OUT"
    CKPT=$ck DUR=$DUR SEED=$sd PRIME=$PRIME bash ab_test.sh D > /tmp/sweep_run.txt 2>&1
    NEW=$(ls -t ~/b601_pi05/logs/ab_D_*.log | head -1)
    # 抽取关键行
    grep -E "实际步数|跟随|shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_yaw|wrist_roll|gripper|边界方向反转率|判定" "$NEW" | tee -a "$OUT"
    echo "" | tee -a "$OUT"
  done
done

echo "=== 汇总 ===" | tee -a "$OUT"
for f in $(ls -t ~/b601_pi05/logs/ab_D_*.log | head -8 | tac); do
  echo "## $(basename $f)" | tee -a "$OUT"
  grep -E "CKPT|净=|判定" "$f" | head -9 | tee -a "$OUT"
done
echo "SWEEP_DONE $OUT" | tee -a "$OUT"
