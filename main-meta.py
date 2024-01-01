import torch
from models.convNet3 import ConvNet
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from utils import label_to_onehot,DataSplit
from models.modified_resnet import ResNet18
import matplotlib.pyplot as plt
import copy
from utils2.tsne_visualize import *
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = 10
def custom_hotlabel(labels,num_classes=10):
    true_labels = label_to_onehot(labels,num_classes)
    weight = torch.ones_like(true_labels)
    weight = weight + true_labels*8
    for i in range(1,3):
        true_labels = true_labels + label_to_onehot(labels+i,num_classes)
    
    return true_labels.data, weight.data
def dataset_downsample(dataset,downsamplenum):
    index = [[] for i in range(num_classes)] #获取每个图片的索引
    for i in range(len(dataset)):
        index[dataset[i][1]].append(i)
    #图片采样
    slice = []
    for i in range(num_classes):
        slice.extend(random.sample(index[i],downsamplenum))
    sample_dataset = DataSplit(dataset,slice)
    return sample_dataset
def plot(iter,acc,loss,filename):
    plt.plot(iter, acc)
    plt.xlabel('iter')
    plt.ylabel('acc')
    plt.title('acc-iter')
    plt.grid(True)
    plt.savefig('./results/iter/iter-acc-{}.pdf'.format(filename), format='pdf')
    plt.close()
    plt.plot(iter, loss)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title('loss-iter')
    plt.grid(True)
    plt.savefig('./results/iter/iter-loss-{}.pdf'.format(filename), format='pdf')
    plt.close()
def test(net,testloader):
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
    acc = round(100 * correct / total,2)
    net.train()
    return acc , class_correct
    # print('Accuracy on the test set: %d %%' % (100 * correct / total))
    # print(class_correct)
    # acc_test.append(100 * correct / total)
def lossfusion_train(filename):
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

    # nets = []
    # for i in range(num_classes):
    #     nets.append(ResNet18(num_classes=2).to(device))
    net = ResNet18(num_classes=2).to(device)
    param_groups = net.get_parameter_groups()

    learning_rates = [0.01, 0.01] #前面共同部分学习率更低
    weights = torch.tensor([1, 9], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight = weights)
    criterion2 = nn.CrossEntropyLoss(reduction='none')
    # parameters_to_optimize = list(nets[0].parameters()) + list(nets[1].parameters()) + list(nets[2].parameters()) + list(nets[3].parameters()) + list(nets[4].parameters()) + list(nets[5].parameters()) + list(nets[6].parameters()) + list(nets[7].parameters()) + list(nets[8].parameters()) + list(nets[9].parameters())
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': learning_rates[0]},
        {'params': param_groups[1], 'lr': learning_rates[1]}
    ], momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

    acc_test = []
    iter = []
    loss_ = []
    running_loss = 0.0
    iteration = 0
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
            onehot_labels,my_weight = custom_hotlabel(labels,num_classes=10) #64*10
            # onehot_labels = label_to_onehot(labels,num_classes=10)
            optimizer.zero_grad()
            outputs = net(inputs)
            predictions = torch.stack(outputs, dim=1) #64*10*2
            predictions = predictions.view(-1, 2)
            onehot_labels = onehot_labels.view(-1)
            my_weight = my_weight.view(-1)
            onehot_labels = onehot_labels.long()
            loss = criterion2(predictions, onehot_labels)
            loss = torch.mul(loss,my_weight).sum()/my_weight.sum()
            # loss = torch.nn.functional.cross_entropy(predictions, onehot_labels, reduction='none')
            # loss = torch.mul(loss,weights)
            # loss = torch.mean(loss)
            # loss += criterion(predictions,label_to_onehot(labels+1,num_classes=10).view(-1).long())*0.2 #loss加权,weight=[1,0.1],0.1是真实label前面的一个label
            loss.backward()
            optimizer.step()



 
            running_loss += loss.item()
            if iteration % 400 == 0:    # 每200个batch打印一次损失
                acc,c_acc = test(net,testloader)
                print('[%d, %5d] loss: %.3f acc: %.2f' %
                    (epoch + 1, iteration, running_loss / 400,acc))
                acc_test.append(acc)
                loss_.append(running_loss)
                iter.append(iteration)
                running_loss = 0.0

        scheduler.step()

    plot(iter,acc_test,loss_,filename)
    torch.save(net.state_dict(),'/home/demo2/Code/meta-task/checkpoint/{}.pt'.format(filename))
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
    ], momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

    acc_test = []
    iter = []
    loss_ = []
    running_loss = 0.0
    iteration = 0
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
            onehot_labels = label_to_onehot(labels,num_classes=10) #64*10
            optimizer.zero_grad()
            outputs = net(inputs)
            predictions = torch.stack(outputs, dim=1) #64*10*2
            predictions = predictions.view(-1, 2)
            onehot_labels = onehot_labels.view(-1)
            onehot_labels = onehot_labels.long()
            loss = criterion(predictions, onehot_labels)
            loss.backward()
            optimizer.step()

            #统计正确率
            # for classid in range(10):
            #     outputs[classid] = outputs[classid][:,1] #10*64*1
            # outputs =  torch.stack(outputs, dim=1)
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            # for pred,true_label in zip(predicted,labels):
            #     if pred == true_label:
            #         class_correct[pred] += 1

            running_loss += loss.item()
            if iteration % 200 == 0:    # 每200个batch打印一次损失
                acc = test(net,testloader)
                print('[%d, %5d] loss: %.3f acc: %.2f' %
                    (epoch + 1, iteration, running_loss / 200,acc))
                acc_test.append(acc)
                loss_.append(running_loss)
                iter.append(iteration)
                running_loss = 0.0

        scheduler.step()

    plot(iter,acc_test,loss_)
    torch.save(net.state_dict(),'/home/demo2/Code/meta-task/checkpoint/fusionmodel_2.pt')
def prune_network(model,classid):
    for name,param in model.named_parameters():
        if 'layer4.'+str(classid) in name or 'linear.'+str(classid) in name:
            param.data[torch.abs(param.data) < 0.01] = 0
    return model
def get_parameter_names(model):
    parameter_names = []
    for name, _ in model.named_parameters():
        parameter_names.append(name)
    return parameter_names
def featurefusion_pass(feature_list,net,weight):
    feature_list = get_featurefusion(feature_list,weight)

    # for i in range(10):
    #     feature_list[i] = feature_list[1]
    output = [net.linear[i](feature_list[i]) for i in range(10)]
    return output
def get_featurefusion(feature_list,weight):
    #融合feature，并传入网络获得输出
    updated_tensors = [torch.zeros_like(tensor).to(device) for tensor in feature_list]
    # 遍历列表中的每个张量
    for i, tensor in enumerate(feature_list):
        # 遍历其他九个张量并进行加权相加
        for j, other_tensor in enumerate(feature_list):
            if j != i:  # 跳过当前张量
                updated_tensors[i] += weight[1] * other_tensor

        # 添加当前张量的更新部分
        updated_tensors[i] += weight[0] * tensor
        # 将更新后的张量赋值给当前张量，以更新它
        feature_list[i] = updated_tensors[i]
    return feature_list
def fusion_test(net,testloader,weight):
    #融合feature测试网络，weight=[x,x]，第一个表示本tensor权重，第二个表示其它tensor权重
    net.eval()
    total = 0
    correct = 0
    class_correct = [0 for i in range(num_classes)]
    for i,data in enumerate(testloader):
        images, labels = data[0].to(device),data[1].to(device)
        # outputs = []
        # for net in nets:
        #     outputs.append(net(images)[:,1]) #10*64*2
        _,feature_list,_,_,_,_ = net(images,out_feature=True)
        outputs = featurefusion_pass(feature_list,net,weight)
        for classid in range(10):
            outputs[classid] = outputs[classid][:,1] #10*64*1
        outputs =  torch.stack(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for pred,true_label in zip(predicted,labels):
            if pred == true_label:
                class_correct[pred] += 1
    acc = round(100 * correct / total,2)
    net.train()
    return acc , class_correct
def update_structure_parameters(model,weight):
    num_structures = len(model.layer4)
    # layer4_params = [struct.parameters() for struct in model.layer4]

    # for params in zip(*layer4_params):
    #     weighted_params = []
    #     for i, param in enumerate(params):
    #         other_params = [p for j, p in enumerate(params) if j != i]
    #         other_params_sum = sum(other_params)
    #         weighted_param = param * weight[0] + other_params_sum * weight[1]
    #         weighted_params.append(weighted_param)

    #     for i, weighted_param in enumerate(weighted_params):
    #         params[i].data = weighted_param
    
    linear_params = [struct.parameters() for struct in model.linear]

    for params in zip(*linear_params):
        weighted_params = []
        for i, param in enumerate(params):
            other_params = [p for j, p in enumerate(params) if j != i]
            other_params_sum = sum(other_params)
            weighted_param = param * weight[0] + other_params_sum * weight[1]
            weighted_params.append(weighted_param)

        for i, weighted_param in enumerate(weighted_params):
            params[i].data = weighted_param

    return model
def Tsneplot(net,dataset,weight,sample_num,withdevice):
    #画tsne图 net:网络 dataset:测试集, weight:feature权重, sample_num：每个类downsample数量
    sample_set = dataset_downsample(dataset,sample_num)
    # sample_set = dataset
    testloader = torch.utils.data.DataLoader(sample_set, batch_size=64,
                                            shuffle=False, num_workers=2)
    plot_features = [[] for i in range(10)]
    plot_targets = [[] for i in range(10)]
    # plot_targets = []
    devices = []
    for m,data in enumerate(testloader):
        images, labels = data[0].to(device),data[1].to(device)
        _,features,_,_,_,_ = net(images,out_feature=True)
        fusion_features = get_featurefusion(features,weight)
        fusion_features = features
        for i in range(10):
            plot_features[i].append(fusion_features[i].data)
            plot_targets[i].append(labels.data)
    for classid in range(10):
        plot_features[classid] = torch.cat(plot_features[classid],dim=0)
        plot_targets[classid] = torch.cat(plot_targets[classid],dim=0)
        dd = torch.tensor([classid]*plot_targets[classid].shape[0])
        devices.append(dd)

    if withdevice == False:
    #withoutdevice
        for classid in range(10):
            # plot_feature = torch.cat(plot_feature[classid],dim=0)
            x_np = np.array(plot_features[classid].cpu())
            y_np = np.array(plot_targets[classid].cpu())
            tsne = TSNE(n_components=2, random_state=33)
            x_tsne = tsne.fit_transform(x_np)
            # Tsne = Dim_Reducer(device, 64)
            # x_tsne = Tsne.unsupervised_reduce(data_input=x_np) 
            y_tsne = y_np.reshape((-1,1))
            fig_name = '/home/demo2/Code/meta-task/results/fusionmodel/最后一层fusion[0.55,0.05]/'+str(classid)+'.pdf'
            tsne_plot(x_tsne,y_tsne,fig_name=fig_name)

    else:

    #with device
        plot_feature = plot_feature[:2]
        plot_targets = plot_targets[:2]
        devices = devices[:2]
        feature_cat = torch.cat(plot_feature).to('cpu')
        targets_cat = torch.cat(plot_targets).to('cpu')
        d_cat = torch.cat(devices).to('cpu')
        x_np = feature_cat.numpy()
        y_np = targets_cat.numpy()
        d_np = d_cat.numpy()
        print(x_np.shape,y_np.shape,d_np.shape)
        tsne = TSNE(n_components=2, random_state=33)
        x_tsne = tsne.fit_transform(x_np)
        y_tsne = y_np.reshape((-1, 1))
        d_tsne = d_np.reshape((-1, 1))
        print(x_tsne.shape,y_tsne.shape,d_tsne.shape)
        tsne_plot(x_tsne, y_tsne, d=d_tsne, fig_name="feature_fusion_tsne_weight:[{},{}].pdf".format(weight[0],weight[1]))
def Tsneplot2(net,dataset,weight,sample_num):
    #画tsne图 net:网络 dataset:测试集, weight:feature权重, sample_num：每个类downsample数量
    sample_set = dataset_downsample(dataset,sample_num)
    # sample_set = dataset
    testloader = torch.utils.data.DataLoader(sample_set, batch_size=64,
                                            shuffle=False, num_workers=2)
    plot_features = [[] for i in range(10)]
    plot_targets = [[] for i in range(10)]
    # plot_targets = []
    devices = []
    for m,data in enumerate(testloader):
        images, labels = data[0].to(device),data[1].to(device)
        _,features,_,_,_,_ = net(images,out_feature=True)
        fusion_features = get_featurefusion(features,weight=weight[0])
        fusion_features = features
        plot_features[0].append(fusion_features[0].data)
        plot_targets[0].append(labels.data)
    for m,data in enumerate(testloader):
        images, labels = data[0].to(device),data[1].to(device)
        _,features,_,_,_,_ = net(images,out_feature=True)
        fusion_features = get_featurefusion(features,weight=weight[1])
        fusion_features = features
        plot_features[1].append(fusion_features[0].data)
        plot_targets[1].append(labels.data)

    for classid in range(2):
        plot_features[classid] = torch.cat(plot_features[classid],dim=0)
        plot_targets[classid] = torch.cat(plot_targets[classid],dim=0)
        dd = torch.tensor([classid]*plot_targets[classid].shape[0])
        devices.append(dd)

    plot_feature = plot_features[:2]
    plot_targets = plot_targets[:2]
    devices = devices[:2]
    feature_cat = torch.cat(plot_feature).to('cpu')
    targets_cat = torch.cat(plot_targets).to('cpu')
    d_cat = torch.cat(devices).to('cpu')
    x_np = feature_cat.numpy()
    y_np = targets_cat.numpy()
    d_np = d_cat.numpy()
    print(x_np.shape,y_np.shape,d_np.shape)
    tsne = TSNE(n_components=2, random_state=33)
    x_tsne = tsne.fit_transform(x_np)
    y_tsne = y_np.reshape((-1, 1))
    d_tsne = d_np.reshape((-1, 1))
    print(x_tsne.shape,y_tsne.shape,d_tsne.shape)
    tsne_plot_withd(x_tsne, y_tsne, d=d_tsne, fig_name="feature_fusion_tsne_weight:[{},{}][{},{}].pdf".format(weight[0][0],weight[0][1],weight[1][0],weight[1][1],))



if __name__ == '__main__':
    lossfusion_train(filename='fusion[1,0.1,0.1]')
    # transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
    #                                         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    # net = ResNet18(num_classes=2).to(device)
    # net.load_state_dict(torch.load('/home/demo2/Code/meta-task/checkpoint/fusionmodel_2.pt'))
    # # dataset_test = datasets.ImageFolder('/home/demo2/Slaugfl_results/selected_diffusion_data',transform = transform)
    # dataset_test = datasets.CIFAR10('/home/demo2/FL_data/cifar-10/cifar-10-batches-py',train = False, transform = transform)
    # testloader = torch.utils.data.DataLoader(dataset_test, batch_size=64,
    #                                         shuffle=False, num_workers=2)
    # net.eval()

    # #测试参数加权
    # acc,class_acc = test(net,testloader)
    # print(acc,class_acc)
    # test_acc = []
    # for c in range(150,170,2):
    #     s = c/10000
    #     weight = [1-s*9,s]
    #     updated_net = update_structure_parameters(copy.deepcopy(net),weight)
    #     acc,class_acc = test(updated_net,testloader)
    #     print(s,acc,class_acc)
    #     test_acc.append(acc)
    # print(test_acc)
    # # print(net)

    # # 测试feature加权
    # acc_list = []
    # for i in range(100,110):
    #     # weight = [1-9*i/1000,i/1000]
    #     weight = [0,1/9]
    #     print(weight)
    #     # updated_net = update_structure_parameters(copy.deepcopy(net),weight)
    #     acc,class_acc = fusion_test(net,testloader,weight)
    #     print(acc,class_acc)
    #     acc_list.append(acc)
    # print(acc_list)

    #画feature tsne图
    # Tsneplot(net,dataset_test,[1,0],1000,withdevice=True)
    # Tsneplot2(net,dataset_test,[[1,0],[0.55,0.05]],200)


   