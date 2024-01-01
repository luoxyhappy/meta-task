import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torch.nn as nn
def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target%num_classes, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, custom_labels=None):
        super(CustomImageFolder, self).__init__(root, transform, target_transform)
        
        # 自定义标签列表
        self.custom_labels = custom_labels

    def __getitem__(self, index):
        path, _ = self.samples[index]
        target = self.custom_labels[self.targets[index]]  # 使用自定义标签

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
class DataSplit:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        dataset_idx = self.indices[idx]
        return self.dataset[dataset_idx]

    def __len__(self):
        return len(self.indices)
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss

def update_tensor(tensor, i):
    tensor2 = tensor.clone()  # 克隆一个新的张量，以保留原始数据不变

    # 将不为i的值改为0
    tensor2[tensor != i] = 0

    # 将为i的值改为1
    tensor2[tensor == i] = 1

    return tensor2
