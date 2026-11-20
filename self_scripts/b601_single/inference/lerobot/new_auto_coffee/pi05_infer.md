在 kfzbox 上依次跑（每段之间要人工复原现场，所以别用无停顿的 for 循环）。
默认权重是 250 集 run 的 12k（`expo012000_250ep`，6 任务已训练）：

    ssh kfzbox
    cd ~/dev/lerobot/self_scripts/b601_single/inference/lerobot/new_auto_coffee

    TASK=A1 DURATION=180 ./run_pi05_coffee_250ep.sh     # 空杯 → 接水盘
    TASK=C2 DURATION=120 ./run_pi05_coffee_250ep.sh     # 满杯 → 放置区 2
    TASK=C1 DURATION=120 ./run_pi05_coffee_250ep.sh     # 满杯 → 放置区 1
    TASK=B2 DURATION=90  ./run_pi05_coffee_250ep.sh     # 按 Americano
    TASK=B1 DURATION=90  ./run_pi05_coffee_250ep.sh     # 按 Hot Water
    TASK=C3 DURATION=60  ./run_pi05_coffee_250ep.sh     # 满杯 → 放置区 3（只 10 集，最不可靠）

每段的执行过程都一样：打印参数总览 → 提示时把臂摆到零位并按回车 → 跑够时长自动退 → 停在原地（不会自己回零，需要你手动复位再进下一段）。日志在 ~/dev/lerobot/self_scripts/_logs/inference_b601_*_<时间>.log，动作追踪另有一份 action_test.log。

