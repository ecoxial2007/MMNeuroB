#!/bin/bash

# --- 1. 配置区域 ---

# 定义要执行的Python脚本
PYTHON_SCRIPT="02_extract_moco_feature.py"

# 定义模型和检查点的路径
MOCO_CHECKPOINT="./moco/checkpoint_moco704_multi-level.pth.tar"
CONCH_CHECKPOINT="conch/pytorch_model.bin"

# 定义模型和处理参数
MODEL_NAME="conch_ViT-B-16"
BATCH_SIZE=256
INPUT_SIZE=448

# 定义输入和输出目录的路径模板
# 脚本会自动将下面的 {size} 替换为 SIZES 数组中的每个值
IMAGE_ROOT_TPL="samples/02_tiles/{size}"
FEATURES_ROOT_TPL="samples/03_tile_features/{size}"

# 定义要遍历的所有尺寸 (注意：修正了数组的定义方式)
SIZES=("360" "720" "1440")

# --- 2. 执行区域 ---

echo "=================================================="
echo "开始批量执行 Python 特征提取脚本..."
echo "=================================================="

# 循环遍历每个尺寸并执行Python脚本
# 关键修正：使用 "${SIZES[@]}" 来正确遍历数组中的所有元素
for size_val in "${SIZES[@]}"; do

    # 根据模板和当前尺寸构建完整的输入和输出路径
    current_image_root="${IMAGE_ROOT_TPL/\{size\}/$size_val}"
    current_features_root="${FEATURES_ROOT_TPL/\{size\}/$size_val}"

    echo "--------------------------------------------------"
    echo "正在为尺寸 [${size_val}] 执行任务..."
    echo "输入 (Image Root): ${current_image_root}"
    echo "输出 (Features Root): ${current_features_root}"

    # 执行Python脚本，并通过命令行参数传入所有配置
    python3 "${PYTHON_SCRIPT}" \
        --image_root "${current_image_root}" \
        --features_root "${current_features_root}" \
        --moco_checkpoint "${MOCO_CHECKPOINT}" \
        --conch_checkpoint "${CONCH_CHECKPOINT}" \
        --batch_size ${BATCH_SIZE} \
        --input_size ${INPUT_SIZE} \
        --model_name "${MODEL_NAME}"

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
echo "所有脚本任务执行完毕。"