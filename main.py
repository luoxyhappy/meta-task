import torch
from models.convNet3 import ConvNet
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from models.resnet import ResNet18
import matplotlib.pyplot as plt
device  = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
num_classes = 10
def test(net,testloader):
    #test
    net.eval()
    total = 0
    correct = 0
    class_correct = [0 for i in range(num_classes)]
    for i,data in enumerate(testloader):
        images, labels = data[0].to(device),data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for pred,true_label in zip(predicted,labels):
            if pred == true_label:
                class_correct[pred] += 1
    acc = round(100*correct / total , 2)
    net.train()
    return acc,class_correct
def plot(iter,acc,loss):
    plt.plot(iter, acc)
    plt.xlabel('iter')
    plt.ylabel('acc')
    plt.title('acc-iter')
    plt.grid(True)
    plt.savefig('iter-acc.pdf', format='pdf')
    plt.close()
    plt.plot(iter, loss)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title('loss-iter')
    plt.grid(True)
    plt.savefig('loss-acc.pdf', format='pdf')
    plt.close()
def train():
    epoch_num=150
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
    #train
    # models = []
    # for i in range(num_classes):
    #     models.append(convNet)
    net = ResNet18().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,  momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
    acc = []
    loss_ = []
    iter = []
    iteration = 0
    running_loss = 0.0
    for epoch in range(epoch_num):  # 迭代10个epoch
        total = 0
        correct = 0
        class_correct = [0 for i in range(num_classes)]
        current_lr = optimizer.param_groups[0]['lr']
        print(epoch,current_lr)
        
        net.train()
        for i, data in enumerate(trainloader):
            iteration = iteration + 1
            inputs, labels = data[0].to(device),data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            # for pred,true_label in zip(predicted,labels):
            #     if pred == true_label:
            #         class_correct[pred] += 1
            if iteration % 200 == 0:    # 每200个batch打印一次损失
                iter.append(iteration)
                test_acc,_ = test(net,testloader)
                acc.append(test_acc)
                print('[%d, %5d] loss: %.3f, acc:%.2f' %
                    (epoch + 1, iteration, running_loss / 200, test_acc))
                loss_.append(running_loss)
                running_loss = 0.0
        scheduler.step()
        # print('Accuracy on the test set: %.2f'%(100 * correct / total))
        # print(class_correct)
        # print('Finished Training')

    plot(iter,acc,loss_)
    torch.save(net.state_dict(),'./checkpoint/onemodel.pt')
if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    net = ResNet18(num_classes=10).to(device)
    net.load_state_dict(torch.load('/home/demo2/Code/meta-task/checkpoint/onemodel.pt'))
    dataset_test = datasets.ImageFolder('/home/demo2/Slaugfl_results/selected_diffusion_data',transform = transform)
    dataset_test = datasets.CIFAR10('/home/demo2/FL_data/cifar-10/cifar-10-batches-py',train = False, transform = transform)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=64,
                                            shuffle=False, num_workers=2)
    net.eval()
    mo1 = 0
    mo2 = 0
    mo3 = 0
    mo4 = 0
    mo5 = 0
    for i, data in enumerate(testloader):
        inputs, labels = data[0].to(device),data[1].to(device)
        outputs,feature1,feature2,feature3,feature4,feature5 = net(inputs,out_feature = True)
        
        mod_tensor = torch.abs(feature1.detach())  # 取绝对值
        mod_tensor = torch.remainder(mod_tensor, 2)  # 对每个元素取模2
        mean_mod = torch.sum(mod_tensor)
        mo1 = mo1 + mean_mod

        mod_tensor = torch.abs(feature2.detach())  # 取绝对值
        mod_tensor = torch.remainder(mod_tensor, 2)  # 对每个元素取模2
        mean_mod = torch.sum(mod_tensor)
        mo2 = mo2 + mean_mod

        mod_tensor = torch.abs(feature3.detach())  # 取绝对值
        mod_tensor = torch.remainder(mod_tensor, 2)  # 对每个元素取模2
        mean_mod = torch.sum(mod_tensor)
        mo3 = mo3 + mean_mod

        mod_tensor = torch.abs(feature4.detach())  # 取绝对值
        mod_tensor = torch.remainder(mod_tensor, 2)  # 对每个元素取模2
        mean_mod = torch.sum(mod_tensor)
        mo4 = mo4 + mean_mod

        mod_tensor = torch.abs(feature5.detach())  # 取绝对值
        mod_tensor = torch.remainder(mod_tensor, 2)  # 对每个元素取模2
        mean_mod = torch.sum(mod_tensor)
        mo5 = mo5 + mean_mod
        
    mo1 = mo1/10000
    mo2 = mo2/10000
    mo3 = mo3/10000
    mo4 = mo4/10000
    mo5 = mo5/10000

    print(float(mo1),float(mo2),float(mo3),float(mo4),float(mo5))
    # acc,class_correct = test(net,testloader)
    # print(acc)
    # print(class_correct)