EXTRA='--prime-state -5.5,0.5,1.4,-0.4,8.1,-2.7,0.0' bash openpi/ab_test.sh D

CKPT=trim2pass CHECK=1 DUR=45 SEED=42 PRIME=-1.6,0.9,0.3,1.9,-1.7,1.3,2.6 bash ab_test.sh D

CKPT=trim2pass DUR=20 SEED=42 PRIME=-1.6,0.9,0.3,1.9,-1.7,1.3,359.0 EXTRA="--max-step-deg 12" bash ab_test.sh D

CKPT=trim2pass DUR=60 SEED=42 EXECN=50 PRIME=-1.6,0.9,0.3,1.9,-1.7,1.3,359.0   EXTRA="--max-step-deg 12" bash ab_test.sh D

CKPT=trim2pass DUR=20 SEED=99 PRIME=-1.6,0.9,0.3,1.9,-1.7,1.3,359.0     EXTRA="--max-step-deg 12" bash ab_test.sh D


# --- 10.26 --- new --- 

# 1) 起 server
cd /home/kf/b601_pi05 && bash start_serve.sh lr36k 8000          
# 2) 真机                                                                                                                                                                                 
cd /home/kf/dev/lerobot/self_scripts/b601_single/inference/openpi                                                                                                                         
CKPT=lr36k DUR=20 bash ab_test.sh D   

# 8090 起 server
http://asuslaptop.golden-fir.ts.net:9119/chat?resume=20261020_110049_e04501
504b-630f-b7ef

主页（左右对照）： 192.168.60.183
对齐检查（会实时刷新分数）： 192.168.60.183/era



episode： 31-45  hand摄像头全都歪了20度以上

很多帧的首帧不是初始姿态，首帧是被截断了的后面的夹取杯子的动作

episode 122之后： front摄像头又被碰歪了


# --- 可视化
启动

~/b601_pi05/start_scene.sh 100 20-120 8090

参数依次为：参考集、目标机位区间、端口。它会自动停掉占用 8090 的实例，并预热 11 个参照帧。
如果想在前台运行并查看日志，直接这样执行；如果想挂到后台，就在命令末尾加 &，或另开一个 tmux 窗口。

终止 fuser -k -n tcp 8090