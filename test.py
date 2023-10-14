import torch
from utils import label_to_onehot
# 生成长度为10的随机整数Tensor
label = torch.randint(low=0, high=10, size=(10,))
print(label)
onehot_label = label_to_onehot(label,num_classes=10)
print(onehot_label)
