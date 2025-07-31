import torch
import torch.nn.functional as F
import torchmetrics  # 确保您已安装 torchmetrics
from typing import Optional


def calculate_micro_auroc_with_torchmetrics(
        val_y_preds: torch.Tensor,  # 预测概率, 形状 (B, C)
        val_y_gts: torch.Tensor,  # 真实标签, 形状 (B,)
        num_classes: int,  # 总类别数 C
        class_to_ignore: Optional[int] = None  # 要忽略的类别索引, 例如 0
) -> float:
    """
    使用 torchmetrics 计算微平均 AUROC 分数，可以选择忽略指定的类别。

    参数:
        val_y_preds (torch.Tensor): 模型的预测概率 (例如 softmax 输出)。形状为 (B, C)，
                                    其中 B 是批量大小，C 是类别数。
        val_y_gts (torch.Tensor): 真实的类别标签 (整数)。形状为 (B,)。
        num_classes (int): 数据集中的总类别数 (C)。
        class_to_ignore (Optional[int]): 要从计算中忽略的类别索引。
                                         如果是 None，则包含所有类别。默认为 None。

    返回:
        float: 计算得到的微平均 AUROC 分数。如果 AUROC 未定义（例如，由于数据不足或
               所有类别都被忽略），则可能返回 float('nan')。
    """

    # --- 输入验证 ---
    if not isinstance(val_y_preds, torch.Tensor) or not isinstance(val_y_gts, torch.Tensor):
        raise TypeError("输入 val_y_preds 和 val_y_gts 必须是 PyTorch Tensors。")
    if val_y_preds.ndim != 2:
        raise ValueError(f"val_y_preds 应为2维 (B, C)，得到 {val_y_preds.ndim} 维。")
    if val_y_gts.ndim != 1:
        raise ValueError(f"val_y_gts 应为1维 (B,)，得到 {val_y_gts.ndim} 维。")
    if val_y_preds.shape[0] != val_y_gts.shape[0]:
        raise ValueError(f"val_y_preds ({val_y_preds.shape[0]}) 和 val_y_gts ({val_y_gts.shape[0]}) 的批量大小不匹配。")
    if val_y_preds.shape[1] != num_classes:
        raise ValueError(f"val_y_preds 中的类别数 ({val_y_preds.shape[1]}) 与 num_classes ({num_classes}) 不匹配。")

    if class_to_ignore is not None:
        if not (0 <= class_to_ignore < num_classes):
            raise ValueError(f"class_to_ignore ({class_to_ignore}) 超出范围 [0, {num_classes - 1}]。")

    # --- 1. 将真实标签转换为 One-Hot 编码 ---
    try:
        # 确保标签在有效范围内 [0, num_classes-1]
        if torch.any(val_y_gts < 0) or torch.any(val_y_gts >= num_classes):
            raise ValueError(f"val_y_gts 中的标签必须在 [0, {num_classes - 1}] 范围内。")
        y_true_one_hot = F.one_hot(val_y_gts, num_classes=num_classes)  # 形状: (B, num_classes)
    except RuntimeError as e:
        raise ValueError(
            f"One-hot 编码时出错。请确保 val_y_gts 中的标签在 [0, {num_classes - 1}] 范围内。原始错误: {e}"
        )

    # --- 2. (可选) 移除要忽略的类别的列 ---
    if class_to_ignore is not None:
        columns_to_keep = [c for c in range(num_classes) if c != class_to_ignore]

        if not columns_to_keep:
            # 如果 num_classes 为 1 且该类被忽略，则列表为空
            print(f"警告: 考虑到 class_to_ignore={class_to_ignore} 后，没有类别可用于计算。AUROC 未定义，返回 NaN。")
            return float('nan')

        y_true_selected_classes = y_true_one_hot[:, columns_to_keep]
        y_preds_selected_classes = val_y_preds[:, columns_to_keep]
    else:
        y_true_selected_classes = y_true_one_hot
        y_preds_selected_classes = val_y_preds

    # --- 3. 展平标签和预测 ---
    # target_flat 需要是 long/int 类型，因为 AUROC 指标通常期望如此
    target_flat = y_true_selected_classes.reshape(-1).long()
    preds_flat = y_preds_selected_classes.reshape(-1)

    if target_flat.numel() == 0:
        # 例如，如果批量大小 B 为 0，或者所有类别都被过滤掉了
        print("警告: 展平后的目标张量元素数量为零。AUROC 未定义，返回 NaN。")
        return float('nan')

    # --- 4. 使用 torchmetrics.AUROC (binary) 计算微平均 AUROC ---
    try:
        # task="binary" 因为我们已将问题转换为等效的二分类形式
        micro_auroc_metric = torchmetrics.AUROC(task="binary", thresholds=None)
        # preds 在前，target 在后
        micro_auroc_metric.update(preds_flat, target_flat)
        micro_auc_score_tensor = micro_auroc_metric.compute()
    except Exception as e:
        # 处理 AUROC 计算中可能出现的错误 (例如，所有目标值相同导致 AUROC 未定义)
        print(f"警告: torchmetrics.AUROC 计算过程中出错: {e}。AUROC 可能未定义，返回 NaN。")
        return float('nan')

    return micro_auc_score_tensor.item()  # .item() 从单元素张量中获取 Python float 值


if __name__ == '__main__':
    # --- 示例用法 ---
    B = 100  # 批量大小
    C = 7  # 类别数

    # 模拟预测和标签
    example_preds = F.softmax(torch.rand(B, C), dim=1)
    example_gts = torch.randint(0, C, (B,))

    print("示例数据:")
    print(f"  预测形状: {example_preds.shape}")
    print(f"  标签形状: {example_gts.shape}")
    print("-" * 30)

    # 情况 1: 忽略类别 0
    ignore_c_0 = 0
    micro_auc_ignored_0 = calculate_micro_auroc_with_torchmetrics(
        example_preds, example_gts, num_classes=C, class_to_ignore=ignore_c_0
    )
    print(f"微平均 AUROC (忽略类别 {ignore_c_0}): {micro_auc_ignored_0:.4f}")

    # 情况 2: 忽略类别 3
    ignore_c_3 = 3
    micro_auc_ignored_3 = calculate_micro_auroc_with_torchmetrics(
        example_preds, example_gts, num_classes=C, class_to_ignore=ignore_c_3
    )
    print(f"微平均 AUROC (忽略类别 {ignore_c_3}): {micro_auc_ignored_3:.4f}")

    # 情况 3: 不忽略任何类别 (class_to_ignore=None)
    micro_auc_all_classes = calculate_micro_auroc_with_torchmetrics(
        example_preds, example_gts, num_classes=C, class_to_ignore=None
    )
    print(f"微平均 AUROC (所有类别): {micro_auc_all_classes:.4f}")

    # 情况 4: 只有一个类别，且被忽略 (应该返回 NaN)
    print("-" * 30)
    print("测试边缘情况：单类别且被忽略")
    single_class_preds = F.softmax(torch.rand(5, 1), dim=1)
    single_class_gts = torch.zeros(5, dtype=torch.long)  # 所有标签都是0
    auc_single_ignored = calculate_micro_auroc_with_torchmetrics(
        single_class_preds, single_class_gts, num_classes=1, class_to_ignore=0
    )
    print(f"微平均 AUROC (单类别被忽略): {auc_single_ignored}")

    # 情况 5: 标签越界 (应该抛出 ValueError)
    print("-" * 30)
    print("测试边缘情况：标签越界")
    try:
        invalid_gts = torch.tensor([0, 1, C])  # C 是一个无效标签
        calculate_micro_auroc_with_torchmetrics(
            example_preds, invalid_gts, num_classes=C, class_to_ignore=None
        )
    except ValueError as e:
        print(f"捕获到预期的错误: {e}")