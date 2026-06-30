#!/usr/bin/env python
"""
Diagnostic tool to compare calibration values between motors and saved calibration file.
"""

import json
import logging
from pathlib import Path

from lerobot.motors import MotorCalibration, Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose motor calibration mismatch")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--robot-id", type=str, default="jt_follower_arm", help="Robot ID")
    args = parser.parse_args()

    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }

    bus = FeetechMotorsBus(port=args.port, motors=motors)

    calibration_fpath = Path(f"~/.cache/huggingface/lerobot/calibration/robots/so_follower/{args.robot_id}.json").expanduser()

    print("=" * 80)
    print("Motor Calibration Diagnostic Tool")
    print("=" * 80)

    try:
        bus.connect(handshake=True)
        print(f"\n✓ Connected to motors on port {args.port}")

        print("\n" + "=" * 80)
        print("1. Reading calibration values FROM MOTORS:")
        print("=" * 80)
        motors_calibration = bus.read_calibration()
        for motor_name, cal in motors_calibration.items():
            print(f"\n  Motor: {motor_name}")
            print(f"    ID: {cal.id}")
            print(f"    Drive Mode: {cal.drive_mode}")
            print(f"    Homing Offset: {cal.homing_offset}")
            print(f"    Range Min: {cal.range_min}")
            print(f"    Range Max: {cal.range_max}")

        print("\n" + "=" * 80)
        print("2. Reading calibration values FROM FILE:")
        print("=" * 80)
        print(f"   Calibration file path: {calibration_fpath}")

        file_calibration = {}
        if calibration_fpath.is_file():
            print("   ✓ Calibration file found")
            with open(calibration_fpath) as f:
                raw_data = json.load(f)
                for motor_name, data in raw_data.items():
                    file_calibration[motor_name] = MotorCalibration(
                        id=data["id"],
                        drive_mode=data["drive_mode"],
                        homing_offset=data["homing_offset"],
                        range_min=data["range_min"],
                        range_max=data["range_max"],
                    )

            for motor_name, cal in file_calibration.items():
                print(f"\n    Motor: {motor_name}")
                print(f"      ID: {cal.id}")
                print(f"      Drive Mode: {cal.drive_mode}")
                print(f"      Homing Offset: {cal.homing_offset}")
                print(f"      Range Min: {cal.range_min}")
                print(f"      Range Max: {cal.range_max}")
        else:
            print("   ✗ Calibration file NOT found")

        print("\n" + "=" * 80)
        print("3. Comparison Results:")
        print("=" * 80)

        if not file_calibration:
            print("   Cannot compare: no calibration file found")
            print("\n   Suggestion: Run calibration first or check the robot-id")
        else:
            mismatched = []
            for motor_name in motors:
                if motor_name not in motors_calibration:
                    mismatched.append(f"  - {motor_name}: NOT FOUND in motors")
                    continue
                if motor_name not in file_calibration:
                    mismatched.append(f"  - {motor_name}: NOT FOUND in calibration file")
                    continue

                m_cal = motors_calibration[motor_name]
                f_cal = file_calibration[motor_name]

                differences = []
                if m_cal.homing_offset != f_cal.homing_offset:
                    differences.append(f"Homing_Offset: {m_cal.homing_offset} vs {f_cal.homing_offset}")
                if m_cal.range_min != f_cal.range_min:
                    differences.append(f"Range_Min: {m_cal.range_min} vs {f_cal.range_min}")
                if m_cal.range_max != f_cal.range_max:
                    differences.append(f"Range_Max: {m_cal.range_max} vs {f_cal.range_max}")

                if differences:
                    mismatched.append(f"  - {motor_name}: {', '.join(differences)}")

            if mismatched:
                print("   ✗ MISMATCH FOUND:")
                for m in mismatched:
                    print(m)
                print("\n   Suggestion: Press ENTER when prompted to use the saved calibration file")
                print("               This will write the file values back to the motors")
            else:
                print("   ✓ All calibration values match!")
                print("\n   Note: If you're still getting calibration prompts, there might be another issue")
                print("         Check: protocol version, motor model, or communication errors")

        bus.disconnect(disable_torque=False)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()