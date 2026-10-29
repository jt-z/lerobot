"""env.sh 的 Python 侧读取器（preflight.py / smoke_load.py 共用）。

允许脚本**单独运行**，不必先 `source env.sh`：环境变量没设就去同目录 env.sh 里取字面值，
再退到本文件里的默认值。只解析 `export NAME=值` 这种单行赋值，不执行任何 shell。
"""

from __future__ import annotations

import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_SH = os.path.join(_HERE, "env.sh")

DEFAULT_ROOT = "/home/kf/LX/pai0/lingbot"
DEFAULT_PY = "/home/kf/miniconda3/envs/lerobot/bin/python"
DEFAULT_TASK = (
    "Pick up the paper cup, place it on the silver tray of the coffee machine, pick up the cube, "
    "press the button with the cube (red light on), wait about 4 seconds, release the button "
    "(red light off), put the cube on the table first, then move the cup from the coffee machine "
    "to the table"
)
DEFAULT_RENAME_MAP = {
    "observation.images.front": "observation.images.cam_high",
    "observation.images.hand": "observation.images.cam_left_wrist",
    "observation.images.top": "observation.images.cam_right_wrist",
}


def _from_env_sh(name: str) -> str | None:
    try:
        with open(_ENV_SH, encoding="utf-8") as f:
            for line in f:
                m = re.match(rf"\s*export\s+{re.escape(name)}=(.*)$", line.rstrip("\n"))
                if not m:
                    continue
                val = m.group(1).strip()
                # 先剥引号，再判断是不是 "${OTHER:-x}" 这类引用（是就交给调用方默认值）
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                if "$" in val:
                    return None
                return val
    except OSError:
        pass
    return None


def var(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v:
        return v
    return _from_env_sh(name) or default


def root() -> str:
    return var("B601_LINGBOT_ROOT") or DEFAULT_ROOT


def step() -> str:
    return var("LINGBOT_STEP") or "050000"


def ckpt() -> str:
    return var("LINGBOT_CKPT") or os.path.join(root(), step(), "pretrained_model")


def base() -> str:
    return var("LINGBOT_BASE") or os.path.join(root(), "lingbot_va_base")


def wan() -> str:
    return var("LINGBOT_WAN") or os.path.join(root(), "lingbot-va-base")


def py() -> str:
    return var("LINGBOT_PY") or DEFAULT_PY


def task() -> str:
    return var("LINGBOT_TASK") or DEFAULT_TASK


def attn_mode() -> str:
    return var("LINGBOT_ATTN_MODE") or "torch"


def rename_map() -> dict[str, str]:
    raw = var("LINGBOT_CAM_RENAME_MAP")
    if not raw:
        return dict(DEFAULT_RENAME_MAP)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return dict(DEFAULT_RENAME_MAP)
