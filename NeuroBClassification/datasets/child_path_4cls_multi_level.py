import os
import json
import h5py
import numpy as np
import torch
from torch.utils import data
from collections import Counter

class WSLTilesDataset(data.Dataset):
    """
    用于WSI病理图像分类的数据集类。
    它负责加载多尺度的图像tile特征、文本特征以及对应的标签，
    并支持 MKI 指数作为辅助任务。
    """
    def __init__(self, args, split="train"):
        super().__init__()
        self.anno_path = args.anno_path
        self.feature_path = args.feature_path
        self.split = split
        self.use_mki = args.use_mki

        # 1. 加载标注文件
        with open(os.path.join(self.anno_path, f'{split}.json'), 'r') as jf:
            annotations = json.load(jf)

        # 2. 加载和处理特征 (专为 conch 特征优化)
        print("INFO: Loading features for 'conch' model setup.")
        # 假设所有h5特征文件都在标注文件目录下
        text_h5_path = os.path.join(self.anno_path, 'hiera_conch_answer_feature.h5')
        
        with h5py.File(text_h5_path, 'r') as hf:
            text_cands_features = torch.tensor(np.array(hf['features']), dtype=torch.float32)

        # 将7分类的文本特征重映射为4分类
        self.text_cands_features = self.transform_features_torch(text_cands_features)


        # 按需加载 MKI 特征
        self.mki_features = None
        if self.use_mki:
            print("INFO: Loading MKI features for auxiliary task.")
            mki_h5_path = os.path.join(self.anno_path, 'mki_feature.h5')
            with h5py.File(mki_h5_path, 'r') as hf:
                self.mki_features = torch.tensor(np.array(hf['features']), dtype=torch.float32)

        # 3. 解析和处理元数据
        self.metadata = []
        for value in annotations:
            # 将7个原始类别ID映射到4个新类别ID
            class_id = int(value['histological_type'])
            if class_id in [1, 2, 3]: new_class_id = 0
            elif class_id in [4, 5]: new_class_id = 1
            elif class_id == 6: new_class_id = 2
            elif class_id == 7: new_class_id = 3
            else: continue # 如果有其他类别，则跳过

            new_value = {
                'FileName': value['patient_id'],
                'PathologyNumber': new_class_id,
                'PathologyType': value['description'],
                'MKI': value["MKI"]
            }
            self.metadata.append(new_value)

        # 4. 定义标签映射
        self.answer2label = {
            'Neuroblastoma': 0,
            'Ganglioneuroblastoma, nodular type': 1,
            'Ganglioneuroblastoma, mixed type': 2,
            'Ganglioneuroma': 3
        }
        self.label2answer = {v: k for k, v in self.answer2label.items()}
        self.num_classes = len(self.answer2label)
        self.mki_str_to_label = {'<2%': 0, '2-4%': 1, '>4%': 2}


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        """
        获取单个数据样本，包括多尺度图像特征、候选答案特征和标签。
        """
        f = self.metadata[index]
        file_id = f['FileName']
        
        # 准备标签
        labels = {
            'candidate': torch.tensor(int(f['PathologyNumber']), dtype=torch.long),
            'mki': torch.tensor(int(self.mki_str_to_label[f["MKI"]]), dtype=torch.long)
        }

        # 加载多尺度图像特征
        scales = ['360', '720', '1440']
        multi_scale_features = {}
        for scale in scales:
            tiles_feature_path = os.path.join(self.feature_path, scale, f'{file_id}.h5')
            with h5py.File(tiles_feature_path, 'r') as hf:
                features_tensor = torch.tensor(np.array(hf['features']), dtype=torch.float32)
                multi_scale_features[scale] = {'features': features_tensor}
        
        # 准备候选答案/文本特征
        answer_features = {'candidate': self.text_cands_features}
        if self.use_mki and self.mki_features is not None:
            answer_features['mki'] = self.mki_features

        # 返回统一格式的数据
        return multi_scale_features, answer_features, labels, file_id, 0


    def transform_features_torch(self, text_cands_features):
        """
        将7分类的特征张量通过取均值等方式重映射为4分类。
        
        Args:
            text_cands_features (torch.Tensor): 输入的特征张量, shape (7, ...)。

        Returns:
            torch.Tensor: 转换后的特征张量, shape (4, ...)。
        """
        if text_cands_features.shape[0] != 7:
            raise ValueError("Input feature dimension must be 7 for transformation.")
        
        # 如果特征有额外的维度 (e.g., shape 7,3,512), 则先处理
        if len(text_cands_features.shape) == 4:
            text_cands_features = text_cands_features.mean(dim=-2)

        # 初始化4分类的新特征张量
        new_shape = (4,) + text_cands_features.shape[1:]
        transformed_features = torch.zeros(new_shape,
                                           dtype=text_cands_features.dtype,
                                           device=text_cands_features.device)

        # 应用映射逻辑：
        # 新类别 0 <- 原类别 1, 2, 3 (索引 0, 1, 2)
        transformed_features[0] = torch.mean(text_cands_features[[0, 1, 2]], dim=0)
        # 新类别 1 <- 原类别 4, 5 (索引 3, 4)
        transformed_features[1] = torch.mean(text_cands_features[[3, 4]], dim=0)
        # 新类别 2 <- 原类别 6 (索引 5)
        transformed_features[2] = text_cands_features[5]
        # 新类别 3 <- 原类别 7 (索引 6)
        transformed_features[3] = text_cands_features[6]

        return transformed_features

    def get_sample_weights(self):
        """
        计算用于 WeightedRandomSampler 的样本权重，以处理类别不平衡问题。
        权重 = 1 / (该类别的样本总数)。
        """
        class_labels = [item['PathologyNumber'] for item in self.metadata]
        if not class_labels:
            print("WARNING: No labels found, returning uniform weights.")
            return torch.ones(len(self.metadata), dtype=torch.double)

        class_counts = Counter(class_labels)
        print(f"INFO: Class counts for weighting: {class_counts}")
        
        sample_weights = [1.0 / class_counts[label] for label in class_labels]
        return torch.tensor(sample_weights, dtype=torch.double)