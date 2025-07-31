#!/bin/bash

# 定义输入目录和输出的基础目录
input_dir="./samples/01_svs"
output_dir_base="./samples/02_tiles"

# 遍历指定的 tile_size 值
for tile_size in 360 720 1440; do

  # 为当前 tile_size 创建一个特定的输出目录
  current_output_dir="${output_dir_base}/${tile_size}"
  mkdir -p "$current_output_dir" # -p 选项确保在目录不存在时创建它，且不会报错

  echo "========================================="
  echo "Processing with tile_size: $tile_size"
  echo "Outputting to: $current_output_dir"
  echo "========================================="

  # 遍历指定目录下的所有 .svs 文件
  for svs_file in "$input_dir"/*.svs; do
    if [ -f "$svs_file" ]; then
      echo "Processing file: $svs_file"
      
      # 运行 Python 预处理脚本
      python 01_preprocess.py \
        --input_slide "$svs_file" \
        --output_dir "$current_output_dir" \
        --tile_size "$tile_size"
    fi
  done

done

echo "All tasks completed."