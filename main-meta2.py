#每个网络输出为单个值，不为二分类
import torch
from models.convNet3 import ConvNet
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from utils import label_to_onehot
from models.modified_resnet import ResNet18
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

nets = []
for i in range(num_classes):
    nets.append(ResNet18(num_classes=1).to(device))
criterion = nn.CrossEntropyLoss()
parameters_to_optimize = list(nets[0].parameters()) + list(nets[1].parameters()) + list(nets[2].parameters()) + list(nets[3].parameters()) + list(nets[4].parameters()) + list(nets[5].parameters()) + list(nets[6].parameters()) + list(nets[7].parameters()) + list(nets[8].parameters()) + list(nets[9].parameters())
optimizer = optim.SGD(parameters_to_optimize, lr=0.01)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
acc = []
for epoch in range(epoch_num):  # 迭代10个epoch
    print(epoch)
    total = 0
    correct = 0
    class_correct = [0 for i in range(num_classes)]
    running_loss = 0.0
    for i in range(num_classes):
        nets[i].train()
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device),data[1].to(device)
        outputs = []
        optimizer.zero_grad()
        for net in nets:
            outputs.append(net(inputs)) #10 * 64 *1
        predictions = torch.cat(outputs, dim=1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(predictions.data, 1)
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
    for i in range(num_classes):
        nets[i].eval()
    total = 0
    correct = 0
    class_correct = [0 for i in range(num_classes)]
    for i,data in enumerate(testloader):
        images, labels = data[0].to(device),data[1].to(device)
        outputs = []
        for net in nets:
            outputs.append(net(images)) #10*64*1
        outputs =  torch.cat(outputs, dim=1) #64*10
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for pred,true_label in zip(predicted,labels):
            if pred == true_label:
                class_correct[pred] += 1
    print('Accuracy on the test set: %.2f'%(100 * correct / total))
    print(class_correct)
    acc.append(100 * correct / total)
print(acc)

