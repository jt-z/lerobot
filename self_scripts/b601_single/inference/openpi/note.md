cd /home/kf/dev/lerobot/self_scripts/b601_single/inference/openpi                                                      
bash /home/kf/b601_pi05/start_serve.sh trim330        # 起 8000（会先杀掉旧 server，日志 logs/serve_trim330_16942.log）
bash run_rollout.sh 330 --dry-run --duration 5        # 干跑，看节拍 ≈33.3ms                                                                                              
bash run_rollout.sh 330 --duration 60                 # 真机，不要带 --kick-deg