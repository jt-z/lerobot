#!/usr/bin/env python3
"""b601 π0.5 真机推理主循环（路线 A：4090 上的 JAX policy server + 本机 lerobot 控制）。

交互模型（与训练口径一致）：
  1. 读观测：7 关节角（度）+ 三路相机 front/hand/top
  2. 把观测发给 server，拿回一个 50 步动作块（receding horizon：只执行前 execute_steps 步就重规划）
  3. 按 30Hz 逐帧 robot.send_action()
  4. **异步预取**：在动作块快用完（还剩 --prefetch-lead 步）时就后台发起下一次请求，
     主循环绝不等待推理，动作不会中断（推理 ~100-150ms ≈ 3-5 帧，同步做会每块卡顿一次）

安全默认：
  - 开跑前必须按 ENTER 确认（可 --yes 跳过）
  - 每步目标相对当前位姿的跳变默认限 25°（--max-step-deg 0 关闭）
  - Ctrl+C 立即停发动作并断开
  - --task 文本必须与训练时完全一致，否则语言条件偏掉

先干跑（不接机器人，验证 server + 主循环时序）：
  python run_rollout.py --dry-run --duration 5

真机：
  bash run_rollout.sh 330
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws_client import CAMS, JOINTS, Pi05WSClient  # noqa: E402

TASK_DEFAULT = ("Pick up the paper cup, place it on the silver tray of the coffee machine, pick up the cube, "
                "press the button with the cube (red light on), wait about 4 seconds, release the button "
                "(red light off), put the cube on the table first, then move the cup from the coffee machine "
                "to the table")

CAM_DEFAULT = {
    'hand': '/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0',
    'front': '/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0',
    'top': '/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0',
}
DIRS_DEFAULT = {j: -1.0 for j in JOINTS}


def build_robot(can_port: str, cameras: dict):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot_robot_seeed_b601.config_seeed_b601_rs_follower import SeeedB601RSFollowerConfig

    cam_cfg = {name: OpenCVCameraConfig(index_or_path=path, width=640, height=480, fps=30, fourcc='MJPG')
               for name, path in cameras.items()}
    cfg = SeeedB601RSFollowerConfig(
        port=can_port, can_adapter='socketcan', id='follower',
        cameras=cam_cfg, joint_directions=DIRS_DEFAULT, gravity_compensation=True)
    from lerobot.robots import make_robot_from_config
    return make_robot_from_config(cfg)


def joint_order(robot) -> list[str]:
    """从 action_features 里取关节顺序，避免手写顺序和机器人不一致。"""
    keys = [k for k in robot.action_features if k.endswith('.pos')]
    names = [k[:-4] for k in keys]
    if len(names) != 7:
        print(f'⚠️ action_features 关节数 {len(names)} != 7，用内置顺序', file=sys.stderr)
        return list(JOINTS)
    return names


def state_vector(obs: dict, order: list[str]) -> np.ndarray:
    return np.asarray([float(obs[f'{j}.pos']) for j in order], dtype=np.float32)


def image_dict(obs: dict) -> dict:
    """不同驱动版本的相机键不一致：有的给 observation.images.<cam>，有的给裸名 <cam>。两种都收。"""
    out = {}
    for c in CAMS:
        for key in (f'observation.images.{c}', c):
            if key in obs:
                out[c] = obs[key]
                break
        else:
            raise KeyError(f'观测里没有相机 {c}（试过 observation.images.{c} 和 {c}），实际 keys={list(obs)}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--can', default='can0')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--chunk', type=int, default=50)
    ap.add_argument('--execute-steps', type=int, default=5, help='每个动作块执行前几步再重规划')
    ap.add_argument('--prefetch-lead', type=int, default=3, help='提前几步发起下一次推理')
    ap.add_argument('--duration', type=float, default=0.0, help='0 = 不限，跑到 Ctrl+C')
    ap.add_argument('--task', default=TASK_DEFAULT)
    ap.add_argument('--max-step-deg', type=float, default=25.0, help='单步目标相对当前位姿的限幅，0=关闭')
    ap.add_argument('--log-dir', default='/home/kf/b601_pi05/logs')
    ap.add_argument('--dry-run', action='store_true', help='不接机器人，用合成观测跑主循环')
    ap.add_argument('--yes', action='store_true', help='跳过 ENTER 确认')
    ap.add_argument('--kick-deg', type=float, default=0.0,
                    help='开跑前给机械臂一个初始推力（度，状态坐标；正=该关节角度增大），默认 0 关闭')
    ap.add_argument('--kick-joint', default='shoulder_lift', help='受力的关节')
    ap.add_argument('--kick-steps', type=int, default=30, help='推力用几步平滑完成（30 步=1 秒）')
    ap.add_argument('--hold-break-deg', type=float, default=0.0,
                    help='卡住自救：若连续 3 个动作块行程都 <2°，就再推一次这么多度（默认 0 关闭）')
    ap.add_argument('--prime-state', default='',
                    help='开跑前把整臂摆到该 state 姿态（7 个逗号分隔的度，state 坐标）')
    ap.add_argument('--prime-steps', type=int, default=60, help='摆姿态平滑用几步（60 步=2 秒）')
    ap.add_argument('--force-prime', action='store_true', help='允许单关节变化 >30° 的摆位')
    a = ap.parse_args()

    cli = Pi05WSClient(a.host, a.port)
    print('server metadata:', cli.metadata)

    robot = None
    order = list(JOINTS)
    if not a.dry_run:
        robot = build_robot(a.can, CAM_DEFAULT)
        print('连接机器人 ...')
        robot.connect()
        order = joint_order(robot)
        print('关节顺序:', order)
        obs = robot.get_observation()
        print('当前 state(度):', np.round(state_vector(obs, order), 1).tolist())
        if not a.yes:
            input('确认从臂处于零位、周边安全，按 ENTER 开始（Ctrl+C 取消）...')

    Path(a.log_dir).mkdir(parents=True, exist_ok=True)
    log = Path(a.log_dir) / f"rollout_pi05_ws_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    fh = open(log, 'w', newline='')
    w = csv.writer(fh)
    w.writerow(['t', 'period_ms', 'chunk_id', 'step_in_chunk', 'late']
                + [f'cmd_{j}' for j in order] + [f'state_{j}' for j in order])

    # 初始推力：把某个关节平滑推 kick_deg 度（状态坐标），越过"零位=不动"的分岔点
    # 动作约定 action = -state，所以发下去的值取负
    def prime(robot, order, joint, delta_deg, steps, label=''):
        if delta_deg == 0 or robot is None:
            return
        obs = robot.get_observation()
        s0 = state_vector(obs, order)
        i = order.index(joint) if joint in order else 3
        goal = s0.copy()
        goal[i] += delta_deg
        print(f'[prime{label}] {joint} {s0[i]:.1f}° → {goal[i]:.1f}°（{delta_deg:+.1f}°，{steps} 步）')
        for k in range(1, steps + 1):
            s = s0 + (goal - s0) * (k / steps)
            robot.send_action({f'{j}.pos': float(-s[m]) for m, j in enumerate(order)})
            time.sleep(1.0 / a.fps)

    if not a.dry_run and a.kick_deg:
        prime(robot, order, a.kick_joint, a.kick_deg, a.kick_steps, ' kick')
        time.sleep(0.5)

    # 整臂摆到训练起点姿态（治"全零姿态落在训练分布外 → 策略选待机模式"）
    if not a.dry_run and a.prime_state:
        tgt = np.array([float(x) for x in a.prime_state.split(',')], dtype=np.float32)
        assert len(tgt) == len(order), f'prime-state 需要 {len(order)} 个值，收到 {len(tgt)}'
        obs = robot.get_observation()
        s0 = state_vector(obs, order)
        delta = tgt - s0
        print(f'[prime-state] 当前 {np.round(s0, 1).tolist()}')
        print(f'[prime-state] 目标 {np.round(tgt, 1).tolist()}  最大变化 {np.abs(delta).max():.1f}°')
        if np.abs(delta).max() > 30.0 and not a.force_prime:
            raise SystemExit('[prime-state] 单关节变化 >30°，为安全拒绝执行；确认无误请加 --force-prime')
        for k in range(1, a.prime_steps + 1):
            s = s0 + delta * (k / a.prime_steps)
            robot.send_action({f'{j}.pos': float(-s[m]) for m, j in enumerate(order)})
            time.sleep(1.0 / a.fps)
        time.sleep(0.4)
        s1 = state_vector(robot.get_observation(), order)
        print(f'[prime-state] 到位 {np.round(s1, 1).tolist()}  残差 {np.abs(tgt - s1).max():.2f}°')

    rng = np.random.default_rng(0)
    ex = ThreadPoolExecutor(max_workers=1)
    t_start = time.perf_counter()
    n = 0
    chunk = None
    chunk_id = 0
    idx = 0
    fut = None
    late_steps = 0
    prev_t = t_start
    period_ms = 0.0
    hold_chunks = 0
    breaks = 0

    print('开始推理（Ctrl+C 停止）')
    try:
        while True:
            now = time.perf_counter()
            if a.duration and now - t_start >= a.duration:
                print('到达 --duration，停止')
                break

            # 1) 读观测
            if a.dry_run:
                state = rng.normal(0, 20, 7).astype(np.float32)
                images = {c: (rng.random((480, 640, 3)) * 255).astype(np.uint8) for c in CAMS}
            else:
                obs = robot.get_observation()
                state = state_vector(obs, order)
                images = image_dict(obs)

            # 2) 预取：块快用完时后台发起下一次推理，主循环不等
            if chunk is not None and fut is None and idx >= max(1, a.execute_steps - a.prefetch_lead):
                fut = ex.submit(cli.infer, state, images, a.task)

            # 3) 换块：结果就绪才切，绝不阻塞
            if fut is not None and fut.done() and idx >= a.execute_steps:
                chunk = fut.result()
                fut = None
                chunk_id += 1
                idx = 0
                travel = float(np.max(chunk.max(0) - chunk.min(0)))
                if travel < 2.0:
                    hold_chunks += 1
                else:
                    hold_chunks = 0
                if a.hold_break_deg and hold_chunks >= 3:
                    print(f'[hold-break] 连续 {hold_chunks} 个块行程 {travel:.1f}° < 2° → 推 '
                          f'{a.hold_break_deg}° 自救（第 {breaks + 1} 次）')
                    prime(robot, order, a.kick_joint, a.hold_break_deg, 12, ' break')
                    hold_chunks = 0
                    breaks += 1
                    chunk = None

            # 4) 首个块（只同步一次）
            if chunk is None:
                chunk = cli.infer(state, images, a.task)
                idx = 0
                print(f'首个动作块 {chunk.shape}，延迟 {cli.avg_latency_ms:.0f} ms（含 JIT 编译）')

            # 5) 执行；块跑完但新块没到，就停在最后一步（动作不中断）
            step = min(idx, len(chunk) - 1)
            if idx >= a.execute_steps:
                late_steps += 1        # 说明预取没赶上，用旧块续命
            target = chunk[step]
            if a.max_step_deg > 0:
                target = state + np.clip(target - state, -a.max_step_deg, a.max_step_deg)

            if not a.dry_run:
                robot.send_action({f'{j}.pos': float(target[i]) for i, j in enumerate(order)})

            period_ms = (now - prev_t) * 1000.0
            prev_t = now
            w.writerow([f'{now - t_start:.4f}', f'{period_ms:.1f}', chunk_id, step,
                        int(idx >= a.execute_steps)]
                       + np.round(target, 2).tolist() + np.round(state, 2).tolist())
            idx += 1
            n += 1

            # 30Hz 节拍（按绝对时刻排程，避免累计漂移）
            sleep = t_start + n / a.fps - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print('\nCtrl+C，停止')
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
        fh.close()
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as e:  # noqa: BLE001
                print('断开时异常:', e, file=sys.stderr)
        print(f'共 {n} 步 / {chunk_id + 1} 个块，{cli.calls} 次推理，平均延迟 {cli.avg_latency_ms:.1f} ms')
        print(f'预取未赶上（用旧块续命）的步数: {late_steps}   自救推力次数: {breaks}')
        print('轨迹:', log)


if __name__ == '__main__':
    main()
