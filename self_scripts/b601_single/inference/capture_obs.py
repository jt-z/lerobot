#!/usr/bin/env python3
"""抓取当前场景的一帧观测（state + 三路图像）存成 npz，不发送任何动作。"""
import argparse

import numpy as np

CAMS = {
    'hand': '/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2608076-video-index0',
    'front': '/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0',
    'top': '/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/kf/b601_pi05/logs/live_obs.npz')
    ap.add_argument('--can', default='can0')
    a = ap.parse_args()

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config
    from lerobot_robot_seeed_b601.config_seeed_b601_rs_follower import SeeedB601RSFollowerConfig

    cam_cfg = {n: OpenCVCameraConfig(index_or_path=p, width=640, height=480, fps=30, fourcc='MJPG')
               for n, p in CAMS.items()}
    cfg = SeeedB601RSFollowerConfig(port=a.can, can_adapter='socketcan', id='follower', cameras=cam_cfg,
                                    gravity_compensation=True)
    robot = make_robot_from_config(cfg)
    robot.connect()
    try:
        obs = robot.get_observation()
        order = [k[:-4] for k in robot.action_features if k.endswith('.pos')]
        st = np.array([float(obs[f'{j}.pos']) for j in order], np.float32)
        imgs = {}
        for c in ('front', 'hand', 'top'):
            img = obs.get(f'observation.images.{c}', obs.get(c))
            arr = np.asarray(img)
            if np.issubdtype(arr.dtype, np.floating):
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            if arr.shape[0] == 3 and arr.shape[-1] != 3:
                arr = np.transpose(arr, (1, 2, 0))
            imgs[f'img_{c}'] = arr
        np.savez_compressed(a.out, state=st, **imgs)
        print('关节顺序:', order)
        print('state:', np.round(st, 1).tolist())
        for c, v in imgs.items():
            print(f'  {c}: {v.shape} {v.dtype} mean {np.round(v.reshape(-1,3).mean(0),1).tolist()}')
        print('saved:', a.out)
    finally:
        robot.disconnect()


if __name__ == '__main__':
    main()
