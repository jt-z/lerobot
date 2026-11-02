#!/bin/bash
# 兼容壳（2026-11-02）：历史上入口脚本写的是 _env_kfzbox.sh。
# 口径已统一到 b601_common/_env.sh + devices/<host>.sh，这个文件只为老脚本/老文档不破。
B601_DEVICE=kfzbox
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
