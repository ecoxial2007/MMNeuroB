from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce_loss(loss: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """根据指定的 reduction 方法缩减损失。"""
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    """带有标签平滑的交叉熵损失。"""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, target: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1)
        nll = F.nll_loss(log_preds, target, reduction='none')
        return reduce_loss(alpha * loss / n + (1 - alpha) * nll, self.reduction)


class HierarchicalLoss(nn.Module):
    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing_ce = LabelSmoothingCrossEntropy()


    def forward(self,
                image_features: torch.Tensor,
                text_features_dict: Dict[str, torch.Tensor],
                true_labels_dict: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, torch.Tensor]:
        """
        计算总损失。

        Args:
            image_features (torch.Tensor): 图像聚合特征, shape: [B, 512]。
            text_features_dict (Dict[str, torch.Tensor]): 包含文本特征的字典。
                'candidate': 叶子节点（主要任务）的文本特征。
                'mki' (optional): MKI（辅助任务）的文本特征。
            true_labels_dict (Dict[str, torch.Tensor]): 包含真实标签的字典。
        Returns:
            一个元组，包含总损失和各类别的预测概率。
        """
        text_features = text_features_dict.get('candidate')
        mki_feature = text_features_dict.get('mki', None)

        # 1. 计算主要任务的预测和标准交叉熵损失
        # 根据输入特征的维度进行处理
        if len(text_features.shape) == 4:
            leaf_text_embeds = text_features.mean(dim=2)
        else:  # shape is 3
            leaf_text_embeds = text_features

        y_pred = F.cosine_similarity(image_features.unsqueeze(1), leaf_text_embeds, dim=-1)
        leaf_probs = torch.softmax(y_pred / self.temperature, dim=-1)
        y_labels = true_labels_dict.get('candidate')
        norm_ce_loss = self.label_smoothing_ce(y_pred / self.temperature, y_labels)

        # 2. 如果存在 MKI 辅助任务，计算其损失并返回
        if mki_feature is not None:
            y_mki_pred = F.cosine_similarity(image_features.unsqueeze(1), mki_feature, dim=-1)
            mki_probs = torch.softmax(y_mki_pred / self.temperature, dim=-1)
            mki_labels = true_labels_dict.get('mki')
            norm_ce_mki_loss = self.label_smoothing_ce(y_mki_pred / self.temperature, mki_labels)
            # 返回包含主任务和辅助任务总损失的元组
            return norm_ce_loss + norm_ce_mki_loss, leaf_probs, mki_probs

        return norm_ce_loss, leaf_probs
