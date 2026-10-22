#!/usr/bin/env python3
"""只读关节状态（不动电机）——用来核对实机读数与训练数据是否同符号/同单位。

用法（lerobot 环境）：
  /home/kf/miniconda3/envs/lerobot/bin/python watch_state.py --seconds 20

会每 0.2 秒打印一次 7 个关节角，并给出相对第一次读数的变化量。
你手动把某个关节抬高/放下，就能看出这个关节读数是变大还是变小，
再和训练数据的方向对照（数据集里 shoulder_lift 从 0 → 180 表示"抬起来"）。
"""
import argparse
import time

import numpy as np

CAM_DEFAULT = {
    'hand': '/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0',
    'front': '/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0',
    'top': '/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0',
}
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--can', default='can0')
    a = ap.parse_args()

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config
    from lerobot_robot_seeed_b601.config_seeed_b601_rs_follower import SeeedB601RSFollowerConfig

    cam_cfg = {n: OpenCVCameraConfig(index_or_path=p, width=640, height=480, fps=30, fourcc='MJPG')
               for n, p in CAM_DEFAULT.items()}
    cfg = SeeedB601RSFollowerConfig(port=a.can, can_adapter='socketcan', id='follower', cameras=cam_cfg,
                                    joint_directions={j: -1.0 for j in JOINTS}, gravity_compensation=True)
    robot = make_robot_from_config(cfg)
    robot.connect()
    order = [k[:-4] for k in robot.action_features if k.endswith('.pos')] or list(JOINTS)
    print('关节顺序:', order)
    print('训练数据参考（度）：起始位 shoulder_lift≈0、elbow≈0；任务中 shoulder_lift→180、elbow→143')
    print('手动抬高 shoulder_lift，读数应变大（正方向）；若变小说明该关节符号与训练数据相反\n')

    base = None
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < a.seconds:
            obs = robot.get_observation()
            st = np.array([float(obs[f'{j}.pos']) for j in order], dtype=np.float32)
            if base is None:
                base = st.copy()
            d = st - base
            print('state ' + ' '.join(f'{j[:6]}={v:7.1f}' for j, v in zip(order, st))
                  + ' | Δ ' + ' '.join(f'{v:+6.1f}' for v in d), flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        print('\n已断开（未发送任何动作）')


if __name__ == '__main__':
    main()
