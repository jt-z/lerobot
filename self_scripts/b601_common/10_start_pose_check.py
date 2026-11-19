#!/usr/bin/env python
"""启动零位自检 + 从臂慢速归零（给 b601_single/02/05/06 采集脚本用）。

为什么需要它
------------
主臂 StarArm102 是"只读"的：configure() 把 7 个舵机全部 unlock，类里没有
send_action，软件无法把它驱动回零位。所以主臂只能"读出来检测 + 提示人工摆正"。
从臂 B601-RS 可以驱动，这里用 safe_zero() 做插值归零，再把夹爪缓慢合到 0°。

流程
----
1. 连接主臂（只读）和从臂（不接相机，避免多开一路 USB 带宽）
2. 读两侧 7 个关节角（统一到策略坐标，单位度）
3. 从臂偏离零位超阈值 -> safe_zero() 慢速把 6 个臂关节归零，再把夹爪慢速合到 0°
4. 重新读一遍，打印主从对照表
5. 主臂偏离零位超阈值 -> 列出偏差最大的关节，提示人工摆正，回车后重新检测
6. 两侧都进阈值 -> 退出码 0（此时从臂停在零位，采集脚本随后立刻启动）

退出码：0 通过；1 出错；2 非交互环境下主臂仍不达标

用法（一般由采集脚本调用，不需要手动执行）：
  python self_scripts/b601_common/10_start_pose_check.py \
    --leader-port /dev/rebot_leader --follower-port can0 \
    --joint-directions "{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, \
wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}"
"""

import argparse
import json
import logging
import sys
import time

# 单位换算/预览用；真正连接失败时的报错已经足够清楚，这里只关掉框架的 INFO 刷屏
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_DIRECTIONS = (
    "{shoulder_pan: -1.0, shoulder_lift: -1.0, elbow_flex: -1.0, "
    "wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: -1.0, gripper: -1.0}"
)
GRIPPER = "gripper"


def parse_directions(text: str) -> dict[str, float]:
    """解析 `{shoulder_pan: -1.0, ...}` 或标准 JSON 两种写法。"""
    text = (text or "").strip()
    try:
        data = json.loads(text)
        return {str(k): float(v) for k, v in data.items()}
    except Exception:
        pass

    inner = text.strip().strip("{}")
    out: dict[str, float] = {}
    for part in inner.split(","):
        if not part.strip():
            continue
        key, _, value = part.partition(":")
        if not value:
            raise ValueError(f"无法解析关节方向：{part!r}")
        out[key.strip().strip("\"'")] = float(value.strip())
    return out


def fmt(value: float | None) -> str:
    """把角度格式化成固定宽度，缺读数时显示 --。"""
    return "  --  " if value is None else f"{value:7.2f}"


def deviation(pose: dict[str, float], joint: str) -> float | None:
    """返回该关节相对零位的绝对偏差（度），没有读数时返回 None。"""
    value = pose.get(f"{joint}.pos")
    return None if value is None else abs(value)


def print_table(leader: dict[str, float], follower: dict[str, float], joints: list[str]) -> None:
    """打印主臂/从臂逐关节角度与零位偏差对照表。"""
    print()
    print(f"  {'关节':<14}{'主臂(°)':>10}{'从臂(°)':>10}{'主臂偏差':>10}{'从臂偏差':>10}")
    print("  " + "-" * 54)
    for joint in joints:
        l_val = leader.get(f"{joint}.pos")
        f_val = follower.get(f"{joint}.pos")
        print(
            f"  {joint:<14}{fmt(l_val):>10}{fmt(f_val):>10}"
            f"{fmt(deviation(leader, joint)):>10}{fmt(deviation(follower, joint)):>10}"
        )
    print()


def joint_tolerance(joint: str, tolerance: float, gripper_tolerance: float) -> float:
    """夹爪用单独容差（主臂夹爪方向系数是 -6，会放大读数），其余关节用通用容差。"""
    return gripper_tolerance if joint == GRIPPER else tolerance


def over_tolerance(
    pose: dict[str, float], joints: list[str], tolerance: float, gripper_tolerance: float
) -> list[tuple[str, float, float]]:
    """返回 [(关节, 偏差, 阈值)]，只包含超差关节。"""
    bad = []
    for joint in joints:
        dev = deviation(pose, joint)
        if dev is None:
            continue
        limit = joint_tolerance(joint, tolerance, gripper_tolerance)
        if dev > limit:
            bad.append((joint, dev, limit))
    bad.sort(key=lambda item: item[1], reverse=True)
    return bad


def ramp_joint(robot, joint: str, start: float, target: float, hold: dict[str, float], seconds: float) -> None:
    """把单个关节从 start 缓慢插值到 target，其余关节保持在 hold 当前值。"""
    steps = max(1, int(seconds * 50))
    for i in range(1, steps + 1):
        ratio = i / steps
        action = dict(hold)
        action[joint] = start + (target - start) * ratio
        robot.send_action({f"{name}.pos": value for name, value in action.items()})
        time.sleep(1.0 / 50.0)


def align_follower(robot, pose: dict[str, float], joints: list[str], args) -> dict[str, float]:
    """从臂慢速归零：6 个臂关节用 safe_zero()，夹爪单独插值到 0°。"""
    arm_bad = [j for j, dev, _ in over_tolerance(pose, joints, args.tolerance, args.gripper_tolerance) if j != GRIPPER]

    if arm_bad:
        print(f"  🔧 从臂偏离零位（{', '.join(arm_bad)}），safe_zero() 慢速归零 ...")
        robot.safe_zero(step_interval_s=0.02, exit_on_complete=False)
        pose = robot.get_observation()

    gripper_dev = deviation(pose, GRIPPER)
    if gripper_dev is not None and gripper_dev > args.gripper_tolerance:
        start = pose[f"{GRIPPER}.pos"]
        hold = {joint: pose.get(f"{joint}.pos", 0.0) for joint in joints}
        print(f"  🔧 从臂夹爪 {start:.1f}° -> 0°（{args.align_seconds:.1f}s 慢速合拢）...")
        ramp_joint(robot, GRIPPER, start, 0.0, hold, args.align_seconds)
        pose = robot.get_observation()

    return pose


def main() -> int:
    parser = argparse.ArgumentParser(description="主从臂启动零位自检 + 从臂慢速归零")
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--leader-id", default="leader")
    parser.add_argument("--follower-port", default="can0")
    parser.add_argument("--follower-id", default="follower")
    parser.add_argument("--joint-directions", default=DEFAULT_DIRECTIONS,
                        help="从臂关节方向，与 --robot.joint_directions 保持一致")
    parser.add_argument("--tolerance", type=float, default=5.0, help="臂关节零位容差（度）")
    parser.add_argument("--gripper-tolerance", type=float, default=20.0,
                        help="夹爪零位容差（度）；主臂夹爪方向系数是 -6，会放大读数")
    parser.add_argument("--align-seconds", type=float, default=2.0, help="夹爪合拢耗时（秒）")
    parser.add_argument("--no-align", action="store_true", help="只检测，不驱动从臂")
    parser.add_argument("--max-retries", type=int, default=10, help="主臂不达标时最多重新检测次数")
    args = parser.parse_args()

    from lerobot.teleoperators.rebot_102_leader import RebotArm102Leader, RebotArm102LeaderTeleopConfig
    from lerobot_robot_seeed_b601 import SeeedB601RSFollower, SeeedB601RSFollowerConfig

    directions = parse_directions(args.joint_directions)
    joints = list(directions.keys())

    leader = RebotArm102Leader(RebotArm102LeaderTeleopConfig(
        port=args.leader_port, id=args.leader_id
    ))
    follower = SeeedB601RSFollower(SeeedB601RSFollowerConfig(
        port=args.follower_port,
        id=args.follower_id,
        can_adapter="socketcan",
        gravity_compensation=True,
        joint_directions=directions,
    ))

    leader_connected = False
    try:
        print("🔌 连接主臂 ...")
        leader.connect()
        leader_connected = True

        print("🔌 连接从臂（不接相机）...")
        follower.connect()

        # 先确认通信真的通：get_action()/get_observation() 在读取失败时会退回
        # 全 0，看起来"刚好在零位"，必须提前把掉线的关节打出来。
        print("🔎 主从臂舵机应答检查 ...")
        for name, servo_id in leader.config.joint_ids.items():
            if not leader.bus.ping(servo_id):
                raise RuntimeError(f"主臂舵机无应答：{name} (id={servo_id})")
        for motor in follower.motors.values():
            motor.request_feedback()
        follower.bus.poll_feedback_once()
        offline = [name for name, motor in follower.motors.items() if motor.get_state() is None]
        if offline:
            raise RuntimeError(f"从臂电机无反馈：{', '.join(offline)}")

        attempt = 0
        while True:
            attempt += 1
            leader_pose = leader.get_action()
            follower_pose = follower.get_observation()

            print(f"\n第 {attempt} 次检测：")
            print_table(leader_pose, follower_pose, joints)

            if not args.no_align:
                if over_tolerance(follower_pose, joints, args.tolerance, args.gripper_tolerance):
                    follower_pose = align_follower(follower, follower_pose, joints, args)
                    print("归零后从臂：")
                    print_table(leader_pose, follower_pose, joints)

            follower_bad = over_tolerance(follower_pose, joints, args.tolerance, args.gripper_tolerance)
            if follower_bad:
                print("❌ 从臂仍未回到零位：")
                for joint, dev, limit in follower_bad:
                    print(f"   {joint}: 偏差 {dev:.1f}° > {limit:.1f}°")
                print("   请检查该关节是否卡住/限位，或先断电手动回到零位。")
                return 1

            leader_bad = over_tolerance(leader_pose, joints, args.tolerance, args.gripper_tolerance)
            if not leader_bad:
                print("✅ 主从臂均在零位，可以开始采集。")
                return 0

            print("⚠️  主臂不在零位（被碰歪或未摆正）：")
            for joint, dev, limit in leader_bad:
                print(f"   {joint}: 偏差 {dev:.1f}° > {limit:.1f}°")

            if not sys.stdin.isatty() or attempt > args.max_retries:
                print("   非交互环境或重试次数已用尽，放弃。")
                return 2

            input("   请把主臂摆到零位（各关节≈0°、夹爪合拢），按 ENTER 重新检测（Ctrl+C 取消）...")
    except KeyboardInterrupt:
        print("\n已取消。")
        return 1
    except Exception as exc:  # 连接/通信失败
        print(f"❌ 零位自检失败：{exc}")
        return 1
    finally:
        try:
            if follower.is_connected:
                follower.disconnect()  # 内部会 safe_zero，把从臂留在零位
        finally:
            if leader_connected and leader.is_connected:
                leader.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
