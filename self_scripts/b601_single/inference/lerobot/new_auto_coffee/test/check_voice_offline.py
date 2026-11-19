#!/usr/bin/env python3
"""离线检查语音调度层：不加载音频/串口/机器人。

覆盖：
- 退出码 -> 状态映射
- 热水/美式解析
- run_subtask 命令生成
- 区域任务命令
- _task_shell 的 DONE 行为（用 true/false 模拟）

用法：
    B601_VOICE_DIR=/home/jt/dev/m260c-voice-assistant \
    /home/jt/dev/miniforge3/envs/lerobot_b601/bin/python check_voice_offline.py
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import types


def _load_voice(voice_dir):
    stubs = {
        "audio_capture": ["AudioCapture"],
        "mic_serial": ["MicSerial", "MSG_SHAKE", "MSG_AIUI"],
        "voice_interact_test": [
            "ala_card_ids", "find_angles", "find_values",
            "has_key", "pulse_source_for_xfm",
        ],
    }
    for name, attrs in stubs.items():
        mod = types.ModuleType(name)
        for attr in attrs:
            setattr(mod, attr, None)
        sys.modules[name] = mod

    path = os.path.join(voice_dir, "demo", "coffee_voice_trigger.py")
    spec = importlib.util.spec_from_file_location("coffee_voice_trigger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice-dir", default=os.environ.get(
        "B601_VOICE_DIR", "/home/jt/dev/m260c-voice-assistant"))
    args = parser.parse_args()

    failures = []

    def check(name, cond, detail=""):
        if cond:
            print("[PASS] %s" % name)
        else:
            print("[FAIL] %s %s" % (name, detail))
            failures.append(name)

    cv = _load_voice(args.voice_dir)

    # 1. 退出码映射
    mapping = {0: "ok", 2: "rejected", 3: "timeout",
               4: "unavailable", 130: "interrupted", 1: "failed"}
    for rc, want in mapping.items():
        check("rc %s -> %s" % (rc, want), cv._rc_to_status(rc) == want,
              "got %r" % cv._rc_to_status(rc))

    # 2. 饮品解析
    check("parse 美式", cv.parse_drink("来一杯美式") == "americano")
    check("parse 热水", cv.parse_drink("我要热水") == "hot_water")
    check("parse 无关", cv.parse_drink("随便") is None)

    # 3. 单子任务命令
    cmd = cv.subtask_cmd("C2")
    for token in ("run_subtask.sh C2", "--policy act", "--timeout 40",
                  "--yes", "--status-file /tmp/m260c_C2.json"):
        check("subtask_cmd 包含 %s" % token, token in cmd, cmd)

    # 4. 区域任务
    ns = argparse.Namespace(policy="act", subtask_timeout=40,
                            zone1_cmd="", zone2_cmd="", zone3_cmd="")
    tasks = cv.build_zone_tasks(ns)
    for zone, task_id in ((1, "C1"), (2, "C2"), (3, "C3")):
        check("区域 %d -> %s" % (zone, task_id),
              task_id in tasks[zone].steps[0].cmd,
              tasks[zone].steps[0].cmd)

    # 5. 阶段一标题
    phase1 = cv.ZoneTask(0, "ack", [cv.Step("A1", "true")], name="阶段一")
    check("phase1 title", phase1.title == cv.PHASE1_TITLE, phase1.title)

    # 6. _task_shell 的 DONE 行为（true/false 模拟，不动机器人）
    for name, steps, want in (
        ("shell 全成功", ["true", "true"], "DONE 0"),
        ("shell 第二步失败", ["true", "false"], "DONE 1"),
    ):
        cv.RESULT_FILE = tempfile.mktemp(prefix="m260c_offline_")
        task = cv.ZoneTask(0, "ack",
                           [cv.Step("s%d" % i, cmd) for i, cmd in enumerate(steps, 1)],
                           name="offline")
        script = cv._task_shell(task)
        subprocess.run(["bash", "-c", script], input="\n" * 4,
                       text=True, capture_output=True, timeout=10)
        data = ""
        if os.path.exists(cv.RESULT_FILE):
            with open(cv.RESULT_FILE) as fh:
                data = fh.read()
        check(name, want in data, repr(data))

    if failures:
        print("VOICE OFFLINE CHECKS FAILED: %s" % ", ".join(failures))
        return 1
    print("ALL VOICE OFFLINE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
