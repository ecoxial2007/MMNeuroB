import argparse
import torch
from PIL import Image
import h5py
import glob
import os
from tqdm import tqdm
from open_clip_custom import create_model_from_pretrained
import torchvision.transforms as transforms
import numpy as np


# 已将 input_size 和 conch_model_name 添加到函数参数中
def load_moco_conch_model(moco_checkpoint_path, original_conch_checkpoint_path, conch_model_name, input_size):
    """
    加载经过MoCo预训练的Conch模型。
    此函数现在会创建并返回MoCo训练时对应的推理预处理流程。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        # --- Step 1: 构建基础模型结构 ---
        print(f"=> Building model structure for '{conch_model_name}' from '{original_conch_checkpoint_path}'")
        if not os.path.isfile(original_conch_checkpoint_path):
            raise FileNotFoundError(f"Original Conch model not found at {original_conch_checkpoint_path}")

        model, _ = create_model_from_pretrained(
            conch_model_name,
            checkpoint_path=original_conch_checkpoint_path
        )
        print("Base model structure created.")

        # --- A. MoCo Inference Preprocessing ---
        # 使用来自 argparse 的 input_size
        preprocess = transforms.Compose([
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"Using MoCo inference preprocessing ({input_size}x{input_size}, ImageNet norm).")

        # --- Step 2: 加载 MoCo 权重 ---
        print(f"=> Loading MoCo checkpoint from '{moco_checkpoint_path}'")
        if not os.path.isfile(moco_checkpoint_path):
            raise FileNotFoundError(f"MoCo checkpoint not found at {moco_checkpoint_path}")

        checkpoint = torch.load(moco_checkpoint_path, map_location="cpu")
        state_dict = checkpoint['state_dict']

        # --- Step 3: 清理权重键名 ---
        prefix_to_remove = 'module.base_encoder.'
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix_to_remove):
                new_key = k[len(prefix_to_remove):]
                new_state_dict[new_key] = v

        if not new_state_dict:
            raise ValueError(f"Could not extract weights with prefix '{prefix_to_remove}'. Check checkpoint keys.")

        # --- Step 4: 加载权重到模型的视觉部分 ---
        print("Loading MoCo weights into the model's visual backbone...")
        msg = model.visual.load_state_dict(new_state_dict, strict=False)
        print(f"Weight loading complete. Info: {msg}")

        # --- Step 5: 完成并返回 ---
        model.eval()
        model.to(device)
        print("MoCo-trained Conch model loaded successfully.")
        return model, preprocess

    except Exception as e:
        print(f"Model loading failed: {e}")
        return None, None


def process_images_batch(image_paths, preprocess):
    """
    批量处理图片 (无需改动)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processed_images = []
    valid_image_paths = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            image = preprocess(image).unsqueeze(0)
            processed_images.append(image)
            valid_image_paths.append(image_path)
        except Exception as e:
            print(f"Image {image_path} processing failed: {e}")
            continue
    if processed_images:
        return torch.cat(processed_images, dim=0).to(device), valid_image_paths
    else:
        return None, []


def extract_features_batch(model, images):
    """
    批量提取图片特征 (无需改动)
    """
    try:
        with torch.inference_mode():
            image_embs = model.encode_image(images, proj_contrast=False, normalize=False)
        return image_embs.cpu().numpy()
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None


def save_features_to_h5(features, h5_file_path):
    """
    将特征保存为H5文件 (无需改动)
    """
    root, _ = os.path.split(h5_file_path)
    if not os.path.exists(root):
        os.makedirs(root)
    try:
        with h5py.File(h5_file_path, 'w') as hf:
            hf.create_dataset('features', data=features)
    except Exception as e:
        print(f"Saving features failed: {e}")


def main():
    """
    主执行函数
    """
    # --- 1. 定义和解析命令行参数 ---
    parser = argparse.ArgumentParser(description="Extract features from image tiles using a MoCo-trained Conch model.")

    # 路径参数
    parser.add_argument('--image_root', type=str, required=True,
                        help='Root directory of the input images (e.g., /path/to/tiles_360).')
    parser.add_argument('--features_root', type=str, required=True,
                        help='Root directory to save the output .h5 feature files.')
    parser.add_argument('--moco_checkpoint', type=str, required=True,
                        help='Path to the MoCo model checkpoint (.pth.tar file).')
    parser.add_argument('--conch_checkpoint', type=str, required=True,
                        help='Path to the original Conch model definition (pytorch_model.bin).')

    # 模型和处理参数
    parser.add_argument('--model_name', type=str, default='conch_ViT-B-16',
                        help='Name of the Conch model architecture.')
    parser.add_argument('--input_size', type=int, default=448,
                        help='The input image size (resize and crop) for the model.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of images to process in a single batch.')

    args = parser.parse_args()

    # --- 2. 使用参数加载模型 ---
    model, preprocess = load_moco_conch_model(
        moco_checkpoint_path=args.moco_checkpoint,
        original_conch_checkpoint_path=args.conch_checkpoint,
        conch_model_name=args.model_name,
        input_size=args.input_size
    )

    if model is None or preprocess is None:
        print("Exiting due to model loading failure.")
        exit(1)

    # --- 3. 查找所有要处理的图片 ---
    # 假设文件结构是 {image_root}/{slide_id}/tiles/*.jpg
    image_paths_pattern = os.path.join(args.image_root, '*', 'tiles', '*.jpg')
    image_paths_to_process = glob.glob(image_paths_pattern)

    print(f"Found {len(image_paths_to_process)} images to process from pattern: {image_paths_pattern}")

    if not image_paths_to_process:
        print("Warning: No images found. Please check the --image_root and your directory structure.")
        exit(0)

    # --- 4. 批量处理图片并提取/保存特征 ---
    for i in tqdm(range(0, len(image_paths_to_process), args.batch_size), desc="Extracting Features"):
        batch_image_paths = image_paths_to_process[i:i + args.batch_size]

        processed_images_tensor, valid_paths_in_batch = process_images_batch(batch_image_paths, preprocess)

        if processed_images_tensor is not None:
            features_batch = extract_features_batch(model, processed_images_tensor)

            if features_batch is not None:
                # 逐个保存批次中每个图片的特征
                for j, single_image_features in enumerate(features_batch):
                    original_image_path = valid_paths_in_batch[j]
                    # 构建输出路径，将 image_root 替换为 features_root
                    h5_file_path = original_image_path.replace(args.image_root, args.features_root).replace('.jpg',
                                                                                                            '.h5')
                    save_features_to_h5(single_image_features, h5_file_path)

    print("Processing complete.")


if __name__ == "__main__":
    main()