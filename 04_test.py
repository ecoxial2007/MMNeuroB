import os
import json
import argparse
from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchmetrics.classification import MulticlassAUROC
from tqdm import tqdm

from NeuroBClassification.datasets.child_path_4cls_multi_level import WSLTilesDataset
from NeuroBClassification.model_multi import EffConfig, balance_accuracy, EfficientModel, downstream_task_forward
from NeuroBClassification.hiera_path_loss import HierarchicalLoss
from NeuroBClassification.metrics import calculate_micro_auroc_with_torchmetrics


def process_batch(batch, set_to_device=None, replace_empty_with_none=False):
    """
    一个更稳健的函数，用于递归地处理一个批次的数据。
    - 可以将所有嵌套的 Tensor 移动到指定设备。
    - 可以将所有嵌套的空容器（列表、字典、Tensor）替换为 None。
    """

    # 递归地对数据结构中的每个元素应用一个函数
    def _recursive_apply(data, func):
        if torch.is_tensor(data):
            return func(data)
        elif isinstance(data, Mapping):  # 字典类
            return {key: _recursive_apply(value, func) for key, value in data.items()}
        elif isinstance(data, Sequence) and not isinstance(data, str):  # 列表/元组类
            return [_recursive_apply(item, func) for item in data]
        else:
            return func(data)

    # 1. 如果需要，移动所有 Tensor 到指定设备
    if set_to_device is not None:
        def move_to_device_func(item):
            return item.to(set_to_device) if torch.is_tensor(item) else item

        batch = _recursive_apply(batch, move_to_device_func)

    # 2. 如果需要，将空元素替换为 None
    if replace_empty_with_none:
        def _recursive_replace(data):
            if isinstance(data, Mapping):
                if not data: return None
                return {k: _recursive_replace(v) for k, v in data.items()}
            elif isinstance(data, Sequence) and not isinstance(data, str):
                if not data: return None
                return [_recursive_replace(item) for item in data]
            elif torch.is_tensor(data) and data.nelement() == 0:
                return None
            else:
                return data

        batch = _recursive_replace(batch)

    return batch


def main(args):
    """
    主评估函数
    """
    # 1. 环境与设备设置
    if 'cuda' in args.device and not torch.cuda.is_available():
        print(f"❌ CUDA device '{args.device}' not available, falling back to CPU.")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"🚀 Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 模型和损失函数初始化
    config = EffConfig.from_args(args)
    loss_fn = HierarchicalLoss()
    model = EfficientModel(config, device).to(device)

    print(f"📂 Loading checkpoint from: {args.checkpoint_path}")
    checkpoints = torch.load(args.checkpoint_path, map_location=device, weights_only=True)


    model.load_state_dict(state_dict=checkpoints, strict=True)
    model.eval()



    # 3. 数据集与 DataLoader 设置
    dset_test = WSLTilesDataset(args, split=args.split)
    dldr_test = DataLoader(dset_test,
                           batch_size=args.batch_size,
                           shuffle=False,  # 在评估时应为 False
                           pin_memory=args.pin_memory,
                           drop_last=args.drop_last,
                           num_workers=args.num_workers,
                           collate_fn=default_collate)
    print(f"📊 Evaluating on '{args.split}' split with {len(dset_test)} samples.")

    # 4. 评估循环
    val_y_preds, val_y_gts, val_file_ids = [], [], []

    for batch in tqdm(dldr_test, desc=f"Evaluating {args.split} split"):
        with torch.no_grad():
            batch = process_batch(batch, set_to_device=device, replace_empty_with_none=args.replace_empty_with_none)
            loss, accs, selected_frames, y_pred, y_gt, file_id = downstream_task_forward(model, batch, args, loss_fn)

            val_y_preds.append(y_pred)
            val_y_gts.append(y_gt)
            val_file_ids.extend(file_id)  # 使用 extend 添加批次中的所有 file_id

    val_y_preds = torch.cat(val_y_preds, dim=0)
    val_y_gts = torch.cat(val_y_gts, dim=0)

    # 5. 计算评估指标
    num_classes = dset_test.num_classes
    auroc_metric_macro = MulticlassAUROC(num_classes=num_classes, average="macro", thresholds=None)
    auroc_metric_per_class = MulticlassAUROC(num_classes=num_classes, average=None, thresholds=None)

    auroc_metric_macro.update(val_y_preds, val_y_gts)
    auroc_metric_per_class.update(val_y_preds, val_y_gts)

    metrics = {
        'acc_micro': (torch.argmax(val_y_preds, dim=-1) == val_y_gts).float().mean().item(),
        'acc_macro': balance_accuracy(val_y_preds, val_y_gts).item(),
        'auc_micro': calculate_micro_auroc_with_torchmetrics(val_y_preds, val_y_gts, num_classes),#.item(),
        'auc_macro': auroc_metric_macro.compute().item(),
        'auc_per_class': auroc_metric_per_class.compute().tolist()
    }

    # 6. 保存结果
    predictions = {}
    for fild_id, y_pred, y_gt in zip(val_file_ids, val_y_preds, val_y_gts):
        y_pred_id = torch.argmax(y_pred, dim=-1)
        y_prob = torch.softmax(y_pred, dim=-1)
        pred_item = {
            'y_pred': dset_test.label2answer[y_pred_id.item()],
            'y_pred_prob': float(y_prob[y_pred_id].item()),
            'y_gt': dset_test.label2answer[y_gt.item()],
            'y_gt_prob': float(y_prob[y_gt].item()),
            'result': bool(y_pred_id.item() == y_gt.item()),
        }
        predictions[fild_id] = pred_item
        if args.visible:
            print(f"ID: {fild_id} -> {pred_item}")

    final_results = {
        'metadata': {
            'checkpoint_path': args.checkpoint_path,
            'split': args.split,
            'num_samples': len(dset_test),
        },
        'overall_metrics': metrics,
        'per_file_predictions': predictions
    }

    ckpt_name = os.path.basename(args.checkpoint_path).replace('.pth', '')
    save_path = os.path.join(args.output_dir, f'{ckpt_name}_{args.split}_results.json')

    with open(save_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    print("-" * 50)
    print(f"✅ Results saved to: {save_path}")
    print(f"📋 Overall Metrics for split '{args.split}':")
    print(f"  Accuracy (micro): {metrics['acc_micro']:.4f}")
    print(f"  Accuracy (macro): {metrics['acc_macro']:.4f}")
    print(f"  AUC (micro):      {metrics['auc_micro']:.4f}")
    print(f"  AUC (macro):      {metrics['auc_macro']:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified script for model evaluation.")

    # --- I/O and Path Parameters ---
    io_group = parser.add_argument_group('I/O and Path Parameters')
    io_group.add_argument('--anno_path', type=str, required=True, help='Path to annotation file.')
    io_group.add_argument('--feature_path', type=str, required=True,
                          help='Path to directory with pre-extracted features.')
    io_group.add_argument('--checkpoint_path', type=str, required=True,
                          help='Path to the model checkpoint (.pth) file.')
    io_group.add_argument('--output_dir', type=str, default='./results',
                          help="Directory to save the output JSON results.")
    io_group.add_argument("--split", type=str, default='test')
    io_group.add_argument('--use_mki', action='store_true')

    # --- Dataset and DataLoader Parameters ---
    data_group = parser.add_argument_group('Dataset and DataLoader Parameters')
    data_group.add_argument('--batch_size', default=1, type=int, help="Batch size for evaluation.")
    data_group.add_argument('--num_workers', default=4, type=int, help="Number of workers for data loading.")
    data_group.add_argument('--pin_memory', action='store_true',
                            help="Use pinned memory for DataLoader for faster CPU to GPU transfer.")
    data_group.set_defaults(pin_memory=False, drop_last=True)

    # --- Model Hyperparameters ---
    model_group = parser.add_argument_group('Model Hyperparameters')
    model_group.add_argument('--n_layers', default=1, type=int, help="Number of layers in the model (see EffConfig).")
    model_group.add_argument('--n_heads', default=8, type=int, help="Number of attention heads (see EffConfig).")
    model_group.add_argument('--enc_dropout', default=0.1, type=float, help="Dropout rate (see EffConfig).")
    model_group.add_argument('--d_input', default=512, type=int, help="Input dimension of features (see EffConfig).")
    model_group.add_argument('--n_tiles', default=0, type=int, help="Number of frames/tiles sampled for input.")
    model_group.add_argument('--tao', default=0.01, type=float, help="Temperature parameter for scaling logits (see EffConfig).")

    # --- Runtime and Evaluation Parameters ---
    runtime_group = parser.add_argument_group('Runtime and Evaluation Parameters')
    runtime_group.add_argument('--device', default='cuda:0', type=str,
                               help="Device for computation (e.g., 'cuda:0', 'cpu').")
    runtime_group.add_argument('--replace_empty_with_none', default=True, help="In batch processing, replace empty tensors/lists with None.")
    runtime_group.add_argument('--visible', action='store_true',
                               help="Print prediction details for each sample to the console.")

    args = parser.parse_args()
    main(args)