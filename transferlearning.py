#训练好的二分类模型重新训练任务
import torch
from models.convNet3 import ConvNet
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from models.resnet import ResNet18
from utils import CustomImageFolder,DataSplit,update_tensor
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch_num=20
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

#train
# models = []
# for i in range(num_classes):
#     models.append(convNet)
all_acc = []
for classid in range(1):
    SMOTE_data = CustomImageFolder('/home/demo2/Slaugfl_results/oversample/{}'.format(transfertask), transform = train_transform,custom_labels=[transfertask])
    indice = []
    for i,(img,target) in enumerate(dataset_train):
        if target != transfertask:
            indice.append(i)
    dataset_train = DataSplit(dataset_train,indice)
    combined_dataset = ConcatDataset([dataset_train, SMOTE_data])
    #统计每个类的数量
    class_num = [0 for i in range(10)]
    for data in combined_dataset:
        class_num[data[1]] = class_num[data[1]]+1
    print("每个class的数量",class_num)
    trainloader = torch.utils.data.DataLoader(combined_dataset, batch_size=64,
                                            shuffle=True, num_workers=2)
    # net = ResNet18(num_classes = 2).to(device)
    # net.load_state_dict(torch.load('/home/demo2/Code/meta-task/checkpoint/SMOTEmodels/'+str(classid)+'.pt'))
    net = ResNet18(num_classes = 2).to(device)
    net.load_state_dict(torch.load('/home/demo2/Code/meta-task/checkpoint/SMOTEmodels/0.pt'))

    #锁定parameters
    for name, param in net.named_parameters():
        if 'linear' not in name:
            param.requires_grad = False
    
    #重新建立线性层
    net.linear = nn.Linear(512, 2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.linear.parameters(), lr=0.001)
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
            labels = update_tensor(labels,transfertask)#将对应的class置为1，其余置为0
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
            labels = update_tensor(labels, transfertask)
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
    torch.save(net.state_dict(),'/home/demo2/Code/meta-task/checkpoint/SMOTEmodels_trans/0.pt')
print(all_acc)