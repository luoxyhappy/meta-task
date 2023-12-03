#融合所有模型的输出，看表现是否超过某一单一模型

import torch
from models.convNet3 import ConvNet
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from models.resnet import ResNet18
from utils import CustomImageFolder,DataSplit,update_tensor,label_to_onehot
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch_num = 10
num_classes = 10
transfertask = 0 #所有模型transferlearning的任务类别
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
nets = []
for i in range(num_classes):
    nets.append(ResNet18(num_classes=2).to(device))
#加载模型并测试
for classid in range(num_classes):
    nets[classid].load_state_dict(torch.load('/home/demo2/Code/meta-task/checkpoint/SMOTEmodels_trans/'+str(classid)+'.pt'))
    nets[classid].eval()
    total = 0
    correct = 0
    class_correct = [0 for i in range(num_classes)]
    for i,data in enumerate(testloader):
        images, labels = data[0].to(device),data[1].to(device)
        output = nets[classid](images)
        labels = update_tensor(labels, transfertask)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for pred,true_label in zip(predicted,labels):
            if pred == true_label:
                class_correct[pred] += 1
    print('Accuracy on the test set: %.2f'%(100 * correct / total))
    print(class_correct)

#融合各个模型的预测
fusion_matrix = nn.Linear(10,2).to(device)
# criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0,9.0]).to(device))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(fusion_matrix.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
SMOTE_data = CustomImageFolder('/home/demo2/Slaugfl_results/oversample/{}'.format(transfertask), transform = train_transform,custom_labels=[transfertask])
combined_dataset = ConcatDataset([dataset_train, SMOTE_data])
# combined_dataset = dataset_train
#统计每个类的数量
class_num = [0 for i in range(10)]
for data in combined_dataset:
    class_num[data[1]] = class_num[data[1]]+1
print("每个class的数量",class_num)
trainloader = torch.utils.data.DataLoader(combined_dataset, batch_size=64,
                                            shuffle=True, num_workers=2)
testacc = []
for epoch in range(epoch_num):
    total = 0
    correct = 0
    class_correct = [0 for i in range(num_classes)]
    fusion_matrix.train()
    print(epoch)
    running_loss = 0.0
    for i in range(num_classes):
        nets[i].train()
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device),data[1].to(device)
        labels = update_tensor(labels,transfertask)
        outputs = []
        optimizer.zero_grad()
        for net in nets:
            outputs.append(net(inputs)[:,1])#长度为10的list,每个元素为一个batch_size*2的tensor
        # predictions = torch.cat(outputs, dim=1)
        predictions = torch.stack(outputs, dim=1)
        output = fusion_matrix(predictions)
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
    fusion_matrix.eval()
    total = 0
    correct = 0
    class_correct = [0 for i in range(num_classes)]
    for i,data in enumerate(testloader):
        images, labels = data[0].to(device),data[1].to(device)
        labels = update_tensor(labels, transfertask)
        outputs = []
        for net in nets:
            outputs.append(net(images)[:,1])#长度为10的list,每个元素为一个batch_size*2的tensor
        # predictions = torch.cat(outputs, dim=1)
        predictions = torch.stack(outputs, dim=1)
        output = fusion_matrix(predictions)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for pred,true_label in zip(predicted,labels):
            if pred == true_label:
                class_correct[pred] += 1
    print('Accuracy on the test set: %.2f'%(100 * correct / total))
    print(class_correct)
    testacc.append(100 * correct / total)


for name, param in fusion_matrix.named_parameters():
    print(name,param)
print(testacc)