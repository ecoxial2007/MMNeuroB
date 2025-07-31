#!/bin/bash

# --- 1. 参数配置 (请根据需要修改) ---

# 必需参数
CHECKPOINT_PATH="NeuroBClassification/checkpoint_anno4_head8.pth"
ANNO_PATH="samples/Annotations/"
FEATURE_PATH="./samples/04_wsi_features"
SPLIT="test"

# 可选参数
OUTPUT_DIR="./results"
DEVICE="cuda:0"

# --- 2. 执行评估命令 ---

# 打印将要使用的配置
echo "🚀 开始执行模型评估..."
echo "==================================="
echo "  检查点: ${CHECKPOINT_PATH}"
echo "  标注文件: ${ANNO_PATH}"
echo "  特征路径: ${FEATURE_PATH}"
echo "  数据划分: ${SPLIT}"
echo "  设备: ${DEVICE}"
echo "==================================="
echo ""

# 调用 Python 脚本
# 我们假设您的 Python 文件名为 evaluate.py
python 04_test.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --anno_path "${ANNO_PATH}" \
    --feature_path "${FEATURE_PATH}" \
    --split "${SPLIT}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --visible

# 检查上一个命令的退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 评估成功完成！"
    echo "📄 结果已保存至目录: ${OUTPUT_DIR}"
else
    echo ""
    echo "❌ 评估过程中发生错误。"
fi