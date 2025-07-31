from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from einops import rearrange
from NeuroBClassification.utils.attn import LayerNorm


@dataclass
class EffConfig:
    """
    模型配置的数据类。
    """
    n_layers: int = 1
    n_heads: int = 12
    enc_dropout: float = 0.1
    d_input: int = 768
    tao: float = 1.0

    # 以下参数主要由 Dataset 或其他模块使用
    n_tiles: int = 16
    split: str = 'train'
    use_text_query: bool = True
    use_text_cands: bool = True
    n_cands: int = 4

    @classmethod
    def from_args(cls, args):
        """从 argparse 的参数创建配置实例。"""
        # 只提取此类中定义的字段
        return cls(**{k: v for k, v in vars(args).items() if k in cls.__annotations__})


class QuickGELU(nn.Module):
    """GELU 激活函数的一种快速近似实现。"""

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualCrossAttentionBlock(nn.Module):
    """
    带有残差连接的交叉注意力模块。
    """

    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=False)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q, k, v 的形状应为 (sequence_length, batch_size, embedding_dim)
        context, _ = self.attn(self.ln_1(q), self.ln_1(k), self.ln_1(v))
        q = q + context
        q = q + self.mlp(self.ln_2(q))
        return q


class EfficientModel(nn.Module):
    """
    一个高效的多尺度视觉-语言聚合模型。

    该模型接收多尺度的视觉Tile特征和一组文本候选答案的特征，
    通过交叉注意力机制进行融合，并使用注意力池化得到最终的聚合特征。
    """

    def __init__(self, config: EffConfig, device):
        super().__init__()
        self.config = config
        self.device = device

        self.vision_text_attention = ResidualCrossAttentionBlock(
            d_model=config.d_input, n_head=config.n_heads, dropout=config.enc_dropout
        )

        # 注意力池化层，用于聚合多尺度特征
        self.attention_pooling = nn.Sequential(
            nn.Linear(config.d_input, config.d_input // 2),
            nn.Tanh(),
            nn.Linear(config.d_input // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_vis_seq: dict, x_txt_cands: dict):
        """
        模型的前向传播。

        Args:
            x_vis_seq (dict): 一个字典，包含多尺度的视觉特征。
                              e.g., {'360': {'features': tensor}, '720': ...}
            x_txt_cands (dict): 一个字典，包含文本候选答案的特征。
                                e.g., {'candidate': tensor, 'mki': tensor}
        """
        # 提取并重塑文本候选答案特征 (作为Query)
        # 形状: (num_candidates, batch_size, embed_dim)
        txt_features = x_txt_cands.get('candidate')
        if len(txt_features.shape) == 4:
            txt_features = torch.mean(txt_features, dim=2)
        txt_features = rearrange(txt_features, 'b t c -> t b c')

        # 遍历多尺度的视觉特征
        processed_vis_features = []
        for scale in x_vis_seq:
            # 重塑视觉特征 (作为Key和Value)
            # 形状: (num_tiles, batch_size, embed_dim)
            vis_features_scale = rearrange(x_vis_seq[scale]['features'], 'b t c -> t b c')

            # 执行视觉-文本交叉注意力
            attended_features = self.vision_text_attention(txt_features, vis_features_scale, vis_features_scale)
            processed_vis_features.append(attended_features)

        # 将所有尺度处理后的特征拼接起来
        # 形状: (total_tiles, batch_size, embed_dim) -> (batch_size, total_tiles, embed_dim)
        combined_features = torch.cat(processed_vis_features, dim=0)
        combined_features = rearrange(combined_features, 't b d -> b t d')

        # 使用注意力池化进行加权求和
        attention_weights = self.attention_pooling(combined_features)
        aggregated_feature = torch.sum(combined_features * attention_weights, dim=1)

        return aggregated_feature


### --- 下游任务和评估函数 ---

def downstream_task_forward(model, batch, args, loss_fn):
    """
    执行一个下游分类任务的前向传播（主要任务）。
    """
    x_vis_seq, x_txt_cands, y_gt_dict, file_id, _ = batch

    # 1. 获取模型输出的聚合特征
    aggregated_feature = model(x_vis_seq, x_txt_cands)

    # 2. 使用传入的损失函数计算损失和预测
    # loss_fn 内部处理了相似度计算等逻辑
    result = loss_fn(aggregated_feature, x_txt_cands, y_gt_dict)
    loss = result[0]
    y_pred = result[1]  # 主要任务的预测结果

    # 3. 计算准确率
    y_gt = y_gt_dict.get('candidate')
    accs = (y_pred.argmax(dim=-1) == y_gt).float()

    return loss, accs, aggregated_feature, y_pred, y_gt, file_id


def downstream_task_forward_mki(model, batch, args, loss_fn):
    """
    执行 MKI 辅助任务的前向传播。
    """
    x_vis_seq, x_txt_cands, y_gt_dict, file_id, _ = batch

    # 1. 获取模型输出的聚合特征
    aggregated_feature = model(x_vis_seq, x_txt_cands)

    # 2. 使用损失函数计算损失和预测
    result = loss_fn(aggregated_feature, x_txt_cands, y_gt_dict, args.loss)
    loss = result[0]
    y_pred = result[2]  # MKI 任务的预测结果

    # 3. 计算准确率
    y_gt = y_gt_dict.get('mki')
    accs = (y_pred.argmax(dim=-1) == y_gt).float()

    return loss, accs, aggregated_feature, y_pred, y_gt, file_id


def balance_accuracy(y_pred, y_gt):
    """
    计算平衡准确率（宏平均准确率）。
    """
    class_accs = []
    num_classes = y_pred.shape[1]  # 从预测的维度获取类别数

    for class_id in range(num_classes):
        mask = (y_gt == class_id)
        if mask.sum() == 0:
            continue  # 跳过数据中不存在的类别

        class_correct = (y_pred.argmax(dim=-1)[mask] == y_gt[mask]).float().sum()
        class_acc = class_correct / mask.sum()
        class_accs.append(class_acc)

    if not class_accs:
        return torch.tensor(0.0)  # 如果没有任何类别的样本，则返回0

    return sum(class_accs) / len(class_accs)