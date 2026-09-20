#!/bin/bash

# ========================================
# 双臂 SO-101 遥操作脚本
# ========================================
# 此脚本用于通过双臂 Leader 控制双臂 Follower
# ========================================

set -e  # 遇到错误立即退出

# ========================================
# 配置区域 - 根据你的实际串口修改
# ========================================
# 端口映射（2026-08-26 实测，与 self_scripts/so101_bimanual/inference/old_dataset/run_inference_cap_pen_PI05.sh 一致）：
#   左从臂 = USB 序列号 5C82108837（当前枚举为 ttyACM2）
#   右从臂 = USB 序列号 5B61034841（当前枚举为 ttyACM3）
#   左主臂 = USB 序列号 5C82106862（当前枚举为 ttyACM0）
#   右主臂 = USB 序列号 5B61034865（当前枚举为 ttyACM1）
# 注意：ttyACM 编号随插拔顺序变化，故全部使用 /dev/serial/by-id 稳定路径，
#       只要适配器与机械臂的物理接线不变就不会变。
# 左/右判定依据：各臂 EEPROM 中的 homing_offset 与左右校准文件匹配
#   （5C82108837 与 5C82106862 为左构型，5B61034841 与 5B61034865 为右构型）。

# Follower 臂串口
LEFT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82108837-if00"
RIGHT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00"

# Leader 臂串口
LEFT_LEADER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82106862-if00"
RIGHT_LEADER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034865-if00"

# 校准文件 ID（必须与校准时使用的 ID 一致）
FOLLOWER_ID="jt_follower_arm"
LEADER_ID="jt_leader_arm"

# 控制频率（Hz）
# 注意：60Hz 下四臂 + 四摄像头的总线/CPU 负载较高，日志（teleopereate.log）显示
#       60Hz 下启动约 0.3s 后从臂总线即掉线。改为 30Hz（LeRobot 遥操作默认频率），
#       总线与 CPU 负载减半，可显著降低掉线风险，且不影响操作手感。
FPS=30

# 是否显示数据（true/false）
DISPLAY_DATA=true

# ========================================
# 摄像头配置
# ========================================
# 摄像头映射（2026-08-26 实测，与 self_scripts/so101_bimanual/inference/old_dataset/run_inference_cap_pen_PI05.sh 一致）：
#   左臂手部 = icSpring 无序列号
#   左臂顶部 = icSpring 202404160005
#   右臂手部 = JYU2C-2083 2607060
#   前视     = JYU2C-2083 2607031（新增）
# 注意：/dev/videoN 编号随插拔顺序变化，故使用 /dev/v4l/by-id 稳定路径。

# 左臂摄像头（hand camera + main camera）
LEFT_HAND_CAMERA="/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0"               # 左臂手部
LEFT_MAIN_CAMERA="/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0"  # 左臂顶部
# 右臂摄像头（hand camera）
RIGHT_HAND_CAMERA="/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607060-video-index0"  # 右臂手部
# 前视摄像头（front camera）
FRONT_CAMERA="/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607031-video-index0"        # 前视（新增）

# 摄像头分辨率和帧率
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30

# ========================================
# 颜色输出
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================
# 辅助函数
# ========================================

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ========================================
# 检查串口设备
# ========================================

check_ports() {
    print_header "检查串口设备"

    local all_ports_ok=true

    for port in "$LEFT_FOLLOWER_PORT" "$RIGHT_FOLLOWER_PORT" "$LEFT_LEADER_PORT" "$RIGHT_LEADER_PORT"; do
        if [ -e "$port" ]; then
            print_success "找到设备: $port"
        else
            print_error "设备不存在: $port"
            all_ports_ok=false
        fi
    done

    if [ "$all_ports_ok" = false ]; then
        echo -e "\n${YELLOW}提示：运行以下命令查看可用串口：${NC}"
        echo "  ls -l /dev/ttyUSB* /dev/ttyACM* /dev/tty*"
        exit 1
    fi

    echo -e "\n${GREEN}所有串口设备检查通过！${NC}"
}

# ========================================
# 检查摄像头设备
# ========================================

check_cameras() {
    print_header "检查摄像头设备"

    local all_cameras_ok=true

    for camera in "$LEFT_HAND_CAMERA" "$LEFT_MAIN_CAMERA" "$RIGHT_HAND_CAMERA" "$FRONT_CAMERA"; do
        if [ -e "$camera" ]; then
            print_success "找到摄像头: $camera"
        else
            print_warning "摄像头不存在: $camera"
            all_cameras_ok=false
        fi
    done

    if [ "$all_cameras_ok" = false ]; then
        echo -e "\n${YELLOW}警告：部分摄像头未找到，将继续运行但可能无法显示图像${NC}"
        echo -e "${YELLOW}运行以下命令查看可用摄像头：${NC}"
        echo "  ls -l /dev/video*"
        echo ""
        echo -e "${YELLOW}是否继续？(y/n)${NC}"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "取消遥操作"
            exit 0
        fi
    else
        echo -e "\n${GREEN}所有摄像头设备检查通过！${NC}"
    fi
}

# ========================================
# 检查校准文件
# ========================================

check_calibration() {
    print_header "检查校准文件"

    local all_calib_ok=true

    # 检查 Follower 校准文件
    local follower_dir=~/.cache/huggingface/lerobot/calibration/robots/so_follower
    for side in left right; do
        local calib_file="${follower_dir}/${FOLLOWER_ID}_${side}.json"
        if [ -f "$calib_file" ]; then
            print_success "找到 Follower 校准文件: ${FOLLOWER_ID}_${side}.json"
        else
            print_error "缺少 Follower 校准文件: $calib_file"
            all_calib_ok=false
        fi
    done

    # 检查 Leader 校准文件
    local leader_dir=~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader
    for side in left right; do
        local calib_file="${leader_dir}/${LEADER_ID}_${side}.json"
        if [ -f "$calib_file" ]; then
            print_success "找到 Leader 校准文件: ${LEADER_ID}_${side}.json"
        else
            print_error "缺少 Leader 校准文件: $calib_file"
            all_calib_ok=false
        fi
    done

    if [ "$all_calib_ok" = false ]; then
        echo -e "\n${YELLOW}请先运行校准脚本：${NC}"
        echo "  ./self_scripts/so101_bimanual/01_calibrate.sh"
        exit 1
    fi

    echo -e "\n${GREEN}所有校准文件检查通过！${NC}"
}

# ========================================
# 主流程
# ========================================

main() {
    print_header "双臂 SO-101 遥操作脚本"

    echo "配置信息："
    echo "  Follower 左臂: $LEFT_FOLLOWER_PORT"
    echo "  Follower 右臂: $RIGHT_FOLLOWER_PORT"
    echo "  Leader 左臂: $LEFT_LEADER_PORT"
    echo "  Leader 右臂: $RIGHT_LEADER_PORT"
    echo "  Follower ID: ${FOLLOWER_ID}_left / ${FOLLOWER_ID}_right"
    echo "  Leader ID: ${LEADER_ID}_left / ${LEADER_ID}_right"
    echo "  控制频率: ${FPS} Hz"
    echo "  显示数据: $DISPLAY_DATA"
    echo ""
    echo "摄像头配置："
    echo "  左臂手部摄像头: $LEFT_HAND_CAMERA"
    echo "  主摄像头（顶部）: $LEFT_MAIN_CAMERA"
    echo "  右臂手部摄像头: $RIGHT_HAND_CAMERA"
    echo "  前视摄像头: $FRONT_CAMERA"
    echo "  分辨率: ${CAMERA_WIDTH}x${CAMERA_HEIGHT} @ ${CAMERA_FPS}fps"
    echo ""

    # 检查串口
    check_ports

    # 检查摄像头
    check_cameras

    # 检查校准文件
    check_calibration

    # 重要提示
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}⚠️  重要提示${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "${YELLOW}开始遥操作前，请确保：${NC}"
    echo "  1. 所有机械臂已正确连接"
    echo "  2. Follower 与 Leader 初始姿态尽量一致（差距过大 → 启动全速追赶 → 易掉总线）"
    echo "  3. Follower 机械臂处于安全的初始位置"
    echo "  4. Leader 机械臂处于舒适的操作位置"
    echo "  5. 周围环境安全，无障碍物"
    echo "  6. 准备好随时按 ${RED}Ctrl+C${NC} 紧急停止"
    echo ""

    # 询问用户是否继续
    echo -e "${YELLOW}是否开始遥操作？(y/n)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "取消遥操作"
        exit 0
    fi

    echo ""
    print_header "启动遥操作"
    echo -e "${GREEN}按 Ctrl+C 停止遥操作${NC}\n"

    # 执行遥操作（带摄像头配置）
    # 注意：所有摄像头均明确指定 MJPG 格式（与推理脚本一致）
    # front 摄像头挂载到 left_arm，在观测中会显示为 left_front
    # max_relative_target 限制每个控制周期内从臂追赶主臂的幅度：
    #   40°/tick 远超电机物理速度（60Hz 下约 6°/tick），起不到限速作用。
    #   若启动时从臂与主臂姿态差距过大，从臂会全速追赶、电流骤增导致总线掉线
    #   （见 teleopereate.log 中 "There is no status packet!" 与 Clamp 警告）。
    #   10°/tick @ 30Hz ≈ 300°/s，可平滑追赶，且不影响正常操作速度。
    lerobot-teleoperate \
        --robot.type=bi_so_follower \
        --robot.left_arm_config.port="$LEFT_FOLLOWER_PORT" \
        --robot.right_arm_config.port="$RIGHT_FOLLOWER_PORT" \
        --robot.id="$FOLLOWER_ID" \
        --robot.left_arm_config.max_relative_target=10.0 \
        --robot.right_arm_config.max_relative_target=10.0 \
        --robot.left_arm_config.cameras="{
            hand: {type: opencv, index_or_path: $LEFT_HAND_CAMERA, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $CAMERA_FPS, fourcc: MJPG},
            top: {type: opencv, index_or_path: $LEFT_MAIN_CAMERA, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $CAMERA_FPS, fourcc: MJPG},
            front: {type: opencv, index_or_path: $FRONT_CAMERA, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $CAMERA_FPS, fourcc: MJPG}
        }" \
        --robot.right_arm_config.cameras="{
            hand: {type: opencv, index_or_path: $RIGHT_HAND_CAMERA, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $CAMERA_FPS, fourcc: MJPG}
        }" \
        --teleop.type=bi_so_leader \
        --teleop.left_arm_config.port="$LEFT_LEADER_PORT" \
        --teleop.right_arm_config.port="$RIGHT_LEADER_PORT" \
        --teleop.id="$LEADER_ID" \
        --fps="$FPS" \
        --display_data="$DISPLAY_DATA"

    # 检查结果
    if [ $? -eq 0 ]; then
        echo ""
        print_success "遥操作正常结束"
    else
        echo ""
        print_error "遥操作异常退出"
        exit 1
    fi
}

# 运行主函数
main
