import torch
from models.convNet3 import ConvNet
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from utils import label_to_onehot
from models.modified_resnet import ResNet18
epoch_num=100
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

# nets = []
# for i in range(num_classes):
#     nets.append(ResNet18(num_classes=2).to(device))
net = ResNet18(num_classes=2).to(device)
param_groups = net.get_parameter_groups()

learning_rates = [0.01, 0.01] #前面共同部分学习率更低
weights = torch.tensor([1, 9], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight = weights)
# parameters_to_optimize = list(nets[0].parameters()) + list(nets[1].parameters()) + list(nets[2].parameters()) + list(nets[3].parameters()) + list(nets[4].parameters()) + list(nets[5].parameters()) + list(nets[6].parameters()) + list(nets[7].parameters()) + list(nets[8].parameters()) + list(nets[9].parameters())
optimizer = optim.SGD([
    {'params': param_groups[0], 'lr': learning_rates[0]},
    {'params': param_groups[1], 'lr': learning_rates[1]}
])
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
acc_train = []
acc_test = []
for epoch in range(epoch_num):  # 迭代10个epoch
    total = 0
    correct = 0
    class_correct = [0 for i in range(num_classes)]
    print(epoch)
    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device),data[1].to(device)
        onehot_labels = label_to_onehot(labels,num_classes=10) #64*10
        optimizer.zero_grad()
        outputs = net(inputs)
        predictions = torch.stack(outputs, dim=1) #64*10*2
        predictions = predictions.view(-1, 2)
        onehot_labels = onehot_labels.view(-1)
        onehot_labels = onehot_labels.long()
        # print(onehot_labels)
        # print(predictions)
        # print(predictions.shape)
        # print(onehot_labels.shape)
        loss = criterion(predictions, onehot_labels)
        loss.backward()
        optimizer.step()
        #统计正确率
        for classid in range(10):
            outputs[classid] = outputs[classid][:,1] #10*64*1
        outputs =  torch.stack(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
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
    print('Accuracy on the test set: %d %%' % (100 * correct / total))
    acc_train.append(100 * correct / total)
    print(class_correct)
    print('Finished Training')

    #test
    net.eval()
    total = 0
    correct = 0
    class_correct = [0 for i in range(num_classes)]
    for i,data in enumerate(testloader):
        images, labels = data[0].to(device),data[1].to(device)
        # outputs = []
        # for net in nets:
        #     outputs.append(net(images)[:,1]) #10*64*2
        outputs = net(images)
        for classid in range(10):
            outputs[classid] = outputs[classid][:,1] #10*64*1
        outputs =  torch.stack(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for pred,true_label in zip(predicted,labels):
            if pred == true_label:
                class_correct[pred] += 1
    print('Accuracy on the test set: %d %%' % (100 * correct / total))
    print(class_correct)
    acc_test.append(100 * correct / total)
torch.save(net.state_dict(),'/home/demo2/Code/meta-task/checkpoint/fusionmodel.pt')
print('train acc:',acc_train)
print('test acc:',acc_test)

