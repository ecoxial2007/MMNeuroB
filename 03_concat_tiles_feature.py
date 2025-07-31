import os
import re
import h5py
import argparse
from tqdm import tqdm
import numpy as np

def concat_features_for_folder(folder_name, args):
    """
    为单个文件夹（通常对应一个WSI）处理特征合并。

    Args:
        folder_name (str): 当前要处理的子文件夹名称。
        args (argparse.Namespace): 包含所有路径配置的命令行参数。
    """
    try:
        # --- 1. 构建当前文件夹所需的完整路径 ---
        # 用于确定文件顺序的原始 tile 目录
        tile_order_dir = os.path.join(args.order_root, folder_name, 'tiles')
        # 包含单个特征 .h5 文件的目录
        feature_source_dir = os.path.join(args.features_root, folder_name, 'tiles')
        # 最终合并后的 .h5 文件的完整输出路径
        output_h5_path = os.path.join(args.output_root, f"{folder_name}.h5")

        # --- 2. 检查路径是否存在 ---
        if not os.path.isdir(tile_order_dir):
            print(f"Warning: Tile order directory not found, skipping: {tile_order_dir}")
            return
        if not os.path.isdir(feature_source_dir):
            print(f"Warning: Feature source directory not found, skipping: {feature_source_dir}")
            return

        # --- 3. 获取并过滤文件列表以确定顺序 ---
        file_list = []
        # 正则表达式，用于匹配 "x_y.jpg" 格式的文件名
        pattern = re.compile(r"^(-?\d+)_(-?\d+)\.jpg$")

        for filename in os.listdir(tile_order_dir):
            match = pattern.match(filename)
            if match:
                x, y = map(int, match.groups())
                # 过滤掉坐标为负数的 tile
                if x >= 0 and y >= 0:
                    file_list.append((x, y, filename))

        if not file_list:
            print(f"Warning: No valid tile files found in {tile_order_dir}")
            return

        # --- 4. 按坐标排序 (x优先, y其次) ---
        sorted_files = sorted(file_list, key=lambda item: (item[0], item[1]))

        # --- 5. 按顺序读取并合并H5特征数据 ---
        merged_data = []
        valid_source_files = []
        for x, y, jpg_filename in sorted_files:
            h5_filename = jpg_filename.replace('.jpg', '.h5')
            h5_filepath = os.path.join(feature_source_dir, h5_filename)

            if os.path.exists(h5_filepath):
                try:
                    with h5py.File(h5_filepath, "r") as f:
                        # 假设特征数据存储在名为 "features" 的数据集中
                        dataset = f["features"][:]
                        merged_data.append(dataset)
                        valid_source_files.append(jpg_filename)
                except Exception as e:
                    print(f"Error reading {h5_filepath}: {e}")
            else:
                print(f"Warning: Feature file not found, skipping: {h5_filepath}")

        if not merged_data:
            print(f"Error: No features could be read for folder {folder_name}. Skipping.")
            return

        # --- 6. 垂直堆叠所有特征向量 ---
        final_vector = np.vstack(merged_data)

        # --- 7. 保存合并后的结果 ---
        with h5py.File(output_h5_path, "w") as f:
            f.create_dataset("features", data=final_vector)
            # 将处理过的源文件名列表存入属性，便于追溯
            f.attrs.create("source_files", [fn for fn in valid_source_files])

    except Exception as e:
        print(f"An unexpected error occurred while processing {folder_name}: {e}")

def main():
    """
    主执行函数，负责解析参数和遍历文件夹。
    """
    parser = argparse.ArgumentParser(description="Concatenate individual H5 tile features into a single H5 file per folder.")

    # --- 定义命令行参数 ---
    parser.add_argument('--features_root', type=str, required=True,
                        help='Root directory containing the individual .h5 feature files (e.g., .../moco_features_1440).')
    parser.add_argument('--order_root', type=str, required=True,
                        help='Root directory with original .jpg tiles, used to determine concatenation order (e.g., .../tiles_1440).')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Root directory to save the concatenated .h5 files (e.g., .../moco_concat_features_1440).')

    args = parser.parse_args()

    print("--- Feature Concatenation Script Started ---")
    print(f"Feature Source: {args.features_root}")
    print(f"Order Source:   {args.order_root}")
    print(f"Output Target:  {args.output_root}")
    print("---------------------------------------------")

    # --- 确保输出目录存在 ---
    os.makedirs(args.output_root, exist_ok=True)

    # --- 遍历所有待处理的文件夹 ---
    # 我们以 order_root 中的文件夹为准
    try:
        folder_names = os.listdir(args.order_root)
        if not folder_names:
            print(f"Error: No subdirectories found in order_root: {args.order_root}")
            return

        for folder_name in tqdm(folder_names, desc="Processing folders"):
            concat_features_for_folder(folder_name, args)

        print("\n--- All folders processed. ---")

    except FileNotFoundError:
        print(f"Error: The specified order_root directory does not exist: {args.order_root}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
