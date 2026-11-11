

kfzbox 上启动 ACT 推理（三条命令，按顺序）：

```bash
ssh kfzbox
cd /home/jt/b601_ops

# ① 离线验证（不开机器人，~30 秒）：确认权重能加载、动作合理
bash act_verify.sh 015000

# ② 真机推理（默认限时 300 秒，会先自动跑 ①，过了才启动）
bash act_robot.sh 015000 300
# 提示 "输入 go 回车" 时输 go 才开始动
# 想换权重就换 step：bash act_robot.sh 010000 300
```

前置条件（脚本会先检查并提示）：机械臂上电、PEAK 的 CAN 盒插好（can0）、三路相机插好（hand/front/top，走 /dev/v4l/by-id 固定路径）。推理只用从臂，主臂不用接。

```bash
# 想绕过我的包装、直接用你们原来的脚本也行（权重路径就是普通 ACT ckpt）：
bash ~/dev/lerobot/self_scripts/b601_single/inference/lerobot/run_act.sh \
    /home/jt/LX/pai0/act_b601_20261105_130447/checkpoints/015000/pretrained_model
```



日志落在 ~/b601_ops/logs/（离线验证 verify_act_<step>.json，真机 robot_act_<step>_<时间>.log）。

