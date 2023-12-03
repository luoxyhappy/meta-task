#SMOTE下的二分类
import torch
from models.convNet3 import ConvNet
import torch.nn as nn
import torch.optim as optim
import torchvision
import random
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
from utils import label_to_onehot
from models.resnet import ResNet18
from torchvision.datasets import ImageFolder
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
epoch_num = 100
device  = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

num_classes = 10
train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ColorJitter(brightness=0.24705882352941178),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

dataset_train = datasets.CIFAR10('/home/demo2/FL_data/cifar-10/cifar-10-batches-py',train = True, transform = train_transform)
dataset_test = datasets.CIFAR10('/home/demo2/FL_data/cifar-10/cifar-10-batches-py',train = False, transform = transform)

testloader = torch.utils.data.DataLoader(dataset_test, batch_size=64,
                                         shuffle=False, num_workers=2)
index = [[] for i in range(10)] #获取每个图片的索引
for i in range(len(dataset_train)):
    index[dataset_train[i][1]].append(i)
num = 5000 // 9

all_acc = []
for classid in range(1):
    SMOTE_data = CustomImageFolder('/home/demo2/Slaugfl_results/oversample/{}'.format(classid), transform = train_transform,custom_labels=[classid])
    indice = []
    for i,(img,target) in enumerate(dataset_train):
        if target != classid:
            indice.append(i)
    dataset_train = DataSplit(dataset_train,indice)
    combined_dataset = ConcatDataset([dataset_train, SMOTE_data])
    # indice = []
    # for i,(img,target) in enumerate(dataset_train):
    #     if target == classid:
    #         indice.append(i)
    # subset = DataSplit(dataset_train,indice)
    # combined_dataset = ConcatDataset([dataset_train, subset,subset,subset,subset,subset,subset,subset,subset,subset])
    #统计每个类的数量
    class_num = [0 for i in range(10)]
    for data in combined_dataset:
        class_num[data[1]] = class_num[data[1]]+1
    print("每个class的数量",class_num)
    
    trainloader = torch.utils.data.DataLoader(combined_dataset, batch_size=64,
                                            shuffle=True, num_workers=2)
    net = ResNet18(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    parameters_to_optimize = net.parameters()
    optimizer = optim.SGD(parameters_to_optimize, lr=0.01)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
    acc = []
    for epoch in range(epoch_num):  # 迭代10个epoch
        total = 0
        correct = 0
        class_correct = [0 for i in range(num_classes)]
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device),data[1].to(device)
            optimizer.zero_grad()
            output = net(inputs)
            labels = update_tensor(labels,classid)#将对应的class置为1，其余置为0
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for pred,true_label in zip(predicted,labels):
                if pred == true_label:
                    class_correct[pred] += 1

            running_loss += loss.item()
            if i % 200 == 199:    # 每200个batch打印一次损失
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        scheduler.step()
        print('Finished Training')
        print('Accuracy on the train set: %.2f'%(100 * correct / total))
        print(class_correct)

        #test
        net.eval()
        total = 0
        correct = 0
        class_correct = [0 for i in range(num_classes)]
        for i,data in enumerate(testloader):
            images, labels = data[0].to(device),data[1].to(device)
            output = net(images)
            labels = update_tensor(labels, classid)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for pred,true_label in zip(predicted,labels):
                if pred == true_label:
                    class_correct[pred] += 1
        print('Accuracy on the test set: %.2f'%(100 * correct / total))
        print(class_correct)
        acc.append(100 * correct / total)
    print(acc)
    all_acc.append(acc)
    torch.save(net.state_dict(),'/home/demo2/Code/meta-task/checkpoint/SMOTEmodels/'+str(classid)+'.pt')
print(all_acc)