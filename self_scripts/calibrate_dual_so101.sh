#!/bin/bash

# ========================================
# 双臂 SO-101 校准脚本
# ========================================
# 此脚本用于重新校准双臂 Follower（从臂）与双臂 Leader（主臂）
# 底层调用 LeRobot 框架的 lerobot-calibrate（src/lerobot/scripts/lerobot_calibrate.py），
# 通过 bi_so_follower / bi_so_leader 双机械臂类型依次校准左、右臂。
# ========================================

set -e  # 遇到错误立即退出

# ========================================
# 配置区域 - 根据你的实际串口修改
# ========================================
# 端口映射（2026-08-26 实测，与 teleoperate_dual_so101.sh 一致）：
#   左从臂 = USB 序列号 5C82108837（当前枚举为 ttyACM2）
#   右从臂 = USB 序列号 5B61034841（当前枚举为 ttyACM3）
#   左主臂 = USB 序列号 5B61034865（当前枚举为 ttyACM1）
#   右主臂 = USB 序列号 5C82106862（当前枚举为 ttyACM0）
# 注意：ttyACM 编号随插拔顺序变化，故全部使用 /dev/serial/by-id 稳定路径，
#       只要适配器与机械臂的物理接线不变就不会变。

# Follower 臂串口
LEFT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82108837-if00"
RIGHT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034841-if00"

# Leader 臂串口
LEFT_LEADER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034865-if00"
RIGHT_LEADER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82106862-if00"

# 校准文件 ID（必须与遥操作时使用的 ID 一致；校准时自动生成 _left/_right 两个文件）
FOLLOWER_ID="jt_follower_arm"
LEADER_ID="jt_leader_arm"

# 校准文件存放目录（LeRobot 默认缓存路径，一般无需修改）
CALIB_DIR="$HOME/.cache/huggingface/lerobot/calibration"

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
# 检查已有校准文件
# ========================================

check_calibration() {
    print_header "检查已有校准文件"

    local follower_dir="${CALIB_DIR}/robots/so_follower"
    local leader_dir="${CALIB_DIR}/teleoperators/so_leader"
    local all_calib_ok=true

    for side in left right; do
        local calib_file="${follower_dir}/${FOLLOWER_ID}_${side}.json"
        if [ -f "$calib_file" ]; then
            print_success "Follower 校准文件: $calib_file"
        else
            print_warning "缺少 Follower 校准文件: $calib_file"
            all_calib_ok=false
        fi
    done

    for side in left right; do
        local calib_file="${leader_dir}/${LEADER_ID}_${side}.json"
        if [ -f "$calib_file" ]; then
            print_success "Leader 校准文件: $calib_file"
        else
            print_warning "缺少 Leader 校准文件: $calib_file"
            all_calib_ok=false
        fi
    done

    if [ "$all_calib_ok" = false ]; then
        echo -e "\n${YELLOW}部分校准文件缺失（首次校准属正常现象），本次校准将补全。${NC}"
    fi
}

# ========================================
# 校准一个双机械臂设备（依次校准左臂、右臂）
# ========================================
# $1 = 配置前缀（robot / teleop）
# $2 = 设备类型（bi_so_follower / bi_so_leader）
# $3 = ID（如 jt_follower_arm）
# $4 = 左臂串口
# $5 = 右臂串口
# $6 = 设备中文名（如 "Follower（从臂）"）

calibrate_device() {
    local prefix="$1"
    local device_type="$2"
    local id="$3"
    local left_port="$4"
    local right_port="$5"
    local device_name="$6"

    print_header "校准 $device_name（$device_type）"
    echo "  左臂串口: $left_port"
    echo "  右臂串口: $right_port"
    echo "  校准 ID: ${id}_left / ${id}_right"
    echo ""
    echo -e "${YELLOW}说明：每支臂需要两步手动操作：${NC}"
    echo "  1. 把机械臂移到其运动范围的中点（homing 位置），然后按回车"
    echo "  2. 依次缓慢转动除 wrist_roll 外的所有关节，走完整个行程，然后按回车"
    echo ""
    echo -e "${RED}重要：${NC}"
    echo -e "${RED}如果已存在校准文件，程序会询问是否使用旧校准：${NC}"
    echo -e "${RED}  直接按回车 = 使用旧校准文件（本次不重新校准）${NC}"
    echo -e "${RED}  输入 c 再按回车 = 重新校准${NC}"
    echo -e "${YELLOW}如需重新校准，请务必输入 c 再按回车！${NC}"
    echo ""

    lerobot-calibrate \
        "--${prefix}.type=${device_type}" \
        "--${prefix}.id=${id}" \
        "--${prefix}.left_arm_config.port=${left_port}" \
        "--${prefix}.right_arm_config.port=${right_port}"

    if [ $? -eq 0 ]; then
        echo ""
        print_success "$device_name 校准完成"
    else
        echo ""
        print_error "$device_name 校准失败"
        exit 1
    fi
}

# ========================================
# 主流程
# ========================================

main() {
    print_header "双臂 SO-101 校准脚本"

    echo "配置信息："
    echo "  Follower 左臂: $LEFT_FOLLOWER_PORT"
    echo "  Follower 右臂: $RIGHT_FOLLOWER_PORT"
    echo "  Leader 左臂: $LEFT_LEADER_PORT"
    echo "  Leader 右臂: $RIGHT_LEADER_PORT"
    echo "  Follower ID: ${FOLLOWER_ID}_left / ${FOLLOWER_ID}_right"
    echo "  Leader ID: ${LEADER_ID}_left / ${LEADER_ID}_right"
    echo "  校准文件目录: $CALIB_DIR"
    echo ""

    # 检查串口
    check_ports

    # 检查已有校准文件
    check_calibration

    # 重要提示
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}⚠️  重要提示${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "${YELLOW}校准前请确保：${NC}"
    echo "  1. 所有机械臂已正确连接且供电正常"
    echo "  2. 机械臂周围环境安全，无障碍物"
    echo "  3. 校准过程中电机力矩释放，请用手扶持机械臂再操作"
    echo "  4. 校准过程为交互式，需按终端提示逐步操作"
    echo "  5. 校准顺序：先 Follower（从臂）左/右，再 Leader（主臂）左/右"
    echo "  6. 校准完成后会生成/覆盖校准文件，遥操作脚本会自动使用"
    echo "  7. 过程中随时可按 ${RED}Ctrl+C${NC} 中止（已写入的校准会保留）"
    echo ""

    # 询问用户是否继续
    echo -e "${YELLOW}是否开始校准？(y/n)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "取消校准"
        exit 0
    fi

    # 校准 Follower（从臂，左/右）
    calibrate_device robot bi_so_follower "$FOLLOWER_ID" "$LEFT_FOLLOWER_PORT" "$RIGHT_FOLLOWER_PORT" "Follower（从臂）"

    # 校准 Leader（主臂，左/右）
    calibrate_device teleop bi_so_leader "$LEADER_ID" "$LEFT_LEADER_PORT" "$RIGHT_LEADER_PORT" "Leader（主臂）"

    echo ""
    print_header "校准完成"
    print_success "所有机械臂校准完成！"
    echo "校准文件位置："
    echo "  ${CALIB_DIR}/robots/so_follower/${FOLLOWER_ID}_{left,right}.json"
    echo "  ${CALIB_DIR}/teleoperators/so_leader/${LEADER_ID}_{left,right}.json"
    echo ""
    echo -e "${YELLOW}现在可以运行遥操作脚本：${NC}"
    echo "  ./self_scripts/teleoperate_dual_so101.sh"
}

# 运行主函数
main
