#训练二分类模型
import torch
from models.convNet3 import ConvNet
import torch.nn as nn
import torch.optim as optim
import torchvision
import random
from torchvision import datasets, transforms
from utils import label_to_onehot
from models.resnet import ResNet18
from utils import DataSplit,FocalLoss,update_tensor


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
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=64,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(dataset_test, batch_size=64,
                                         shuffle=False, num_workers=2)
index = [[] for i in range(10)] #获取每个图片的索引
for i in range(len(dataset_train)):
    index[dataset_train[i][1]].append(i)
num = 5000 // 9

all_acc = []
for classid in range(num_classes):
    net = ResNet18(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0,9.0]).to(device))
    parameters_to_optimize = net.parameters()
    optimizer = optim.SGD(parameters_to_optimize, lr=0.01)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
    acc = []
    for epoch in range(epoch_num):  # 迭代10个epoch
        # sample_index = []
        # for i in range(num_classes):
        #     if i == classid:
        #         sample_index.extend(index[i])
        #     else:
        #         sample_index.extend(random.sample(index[i],num))
        # sample_dataset = DataSplit(dataset_train,sample_index)
        # trainloader = torch.utils.data.DataLoader(sample_dataset, batch_size=64,
        #                                 shuffle=True, num_workers=2)
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
    torch.save(net.state_dict(),'./models_balanceloss_100epoch/'+str(classid)+'.pt')
print(all_acc)