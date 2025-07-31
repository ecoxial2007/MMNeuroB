#!/bin/bash

# --- 1. 配置区域 ---
# 在这里修改你的路径和设置

# 定义要执行的Python脚本的名称
PYTHON_SCRIPT="03_concat_tiles_feature.py"

# 定义要遍历的所有尺寸
SIZES=("360" "720" "1440")


# 定义输入和输出目录的路径模板
FEATURES_ROOT_TPL="./samples/03_tile_features/{size}"
ORDER_ROOT_TPL="./samples/02_tiles/{size}"
OUTPUT_ROOT_TPL="./samples/04_wsi_features/{size}"


# --- 2. 执行区域 ---
# 一般无需修改以下内容

echo "=================================================="
echo "开始批量执行 Python 特征合并脚本..."
echo "=================================================="

# 循环遍历每个尺寸并执行Python脚本
for size_val in "${SIZES[@]}"; do

    # 根据模板和当前尺寸构建完整的输入和输出路径
    current_features_root="${FEATURES_ROOT_TPL/\{method\}/$METHOD}"
    current_features_root="${current_features_root/\{size\}/$size_val}"

    current_order_root="${ORDER_ROOT_TPL/\{size\}/$size_val}"

    current_output_root="${OUTPUT_ROOT_TPL/\{method\}/$METHOD}"
    current_output_root="${current_output_root/\{size\}/$size_val}"

    echo "--------------------------------------------------"
    echo "正在为尺寸 [${size_val}] 执行任务..."

    # 执行Python脚本，并通过命令行参数传入所有配置
    python3 "${PYTHON_SCRIPT}" \
        --features_root "${current_features_root}" \
        --order_root "${current_order_root}" \
        --output_root "${current_output_root}"

    # 检查上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "尺寸 [${size_val}] 的任务成功完成。"
    else
        echo "错误：尺寸 [${size_val}] 的任务执行失败。"
        # 如果希望在遇到错误时停止整个脚本，可以取消下面的注释
        # exit 1
    fi
done

echo "=================================================="
echo "所有合并任务执行完毕。"
