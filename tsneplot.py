import torch

from models.convNet3 import ConvNet
import torch.nn as nn
import torch.optim as optim
from utils import CustomImageFolder,DataSplit,update_tensor
import torchvision
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
from utils import label_to_onehot
from utils2.tsne_visualize import *

def update_tensor(tensor, i):
    tensor2 = tensor.clone()  # 克隆一个新的张量，以保留原始数据不变

    # 将不为i的值改为0
    tensor2[tensor != i] = 0

    # 将为i的值改为1
    tensor2[tensor == i] = 1

    return tensor2
class DataSplit:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        dataset_idx = self.indices[idx]
        return self.dataset[dataset_idx]

    def __len__(self):
        return len(self.indices)
def onemodeltsne():
    from models.resnet import ResNet18
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
    SMOTE_data = CustomImageFolder('/home/demo2/Slaugfl_results/oversample/{}'.format(0), transform = train_transform,custom_labels=[0])
    indice = []
    for i,(img,target) in enumerate(dataset_train):
        if target != 0:
            indice.append(i)
    dataset_train = DataSplit(dataset_train,indice)
    dataset_train = ConcatDataset([dataset_train, SMOTE_data])
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=64,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=64,
                                            shuffle=False, num_workers=2)
    index = [[] for i in range(num_classes)] #获取每个图片的索引
    for i in range(len(dataset_test)):
        index[dataset_test[i][1]].append(i)

    #图片采样
    slice = []
    for i in range(num_classes):
        slice.extend(random.sample(index[i],1000))


    sample_dataset = DataSplit(dataset_test,slice)

    index = [[] for i in range(num_classes)] #获取每个图片的索引
    for i in range(len(dataset_train)):
        index[dataset_train[i][1]].append(i)

    #图片采样
    slice = []
    for i in range(num_classes):
        slice.extend(random.sample(index[i],1000))


    sample_dataset_train = DataSplit(dataset_train,slice)


    nets = []
    #加载模型
    for i in range(10):
        nets.append(ResNet18(num_classes=2).to(device))
        nets[i].load_state_dict(torch.load('/home/demo2/Code/meta-task/checkpoint/SMOTEmodels_trans/'+str(i)+'.pt'))
        nets[i].eval()

    #加载单一模型
    # nets.append(ResNet18(num_classes=10).to(device))
    # nets[0].load_state_dict(torch.load('/home/demo2/Code/meta-task/checkpoint/onemodel.pt'))
    # nets[0].eval()

    #画tsne图
    for classid in range(10):
        plot_feature = []
        plot_targets = []
        sampleloader = torch.utils.data.DataLoader(sample_dataset, batch_size=64,
                                            shuffle=False, num_workers=2)
        for m,(images,labels) in enumerate(sampleloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs,_,_,_,_,features = nets[classid](images,out_feature = True)
            plot_feature.append(features.data)
            plot_targets.append(labels.data)
        plot_feature = torch.cat(plot_feature,dim=0)
        plot_targets = torch.cat(plot_targets,dim=0)
        d1 = torch.tensor([1]*plot_targets.shape[0])
        x_np = np.array(plot_feature.cpu())
        y_np = np.array(plot_targets.cpu())
        Tsne = Dim_Reducer(device, 64)
        x_tsne = Tsne.unsupervised_reduce(data_input=x_np) 
        y_tsne = y_np.reshape((-1,1))
        fig_name = '/home/demo2/Code/meta-task/results/SMOTEtrans/'+str(classid)+'_5.pdf'
        tsne_plot(x_tsne,y_tsne,fig_name=fig_name)


        # plot_feature_2 = []
        # plot_targets_2 = []
        # sampleloader = torch.utils.data.DataLoader(sample_dataset_train, batch_size=64,
        #                                      shuffle=False, num_workers=2)
        # for m,(images,labels) in enumerate(sampleloader):
        #     images = images.to(device)
        #     labels = labels.to(device)
        #     outputs,features,_,_,_,_ = nets[classid](images,out_feature = True)
        #     plot_feature_2.append(features.data)
        #     plot_targets_2.append(labels.data)
        # plot_feature_2 = torch.cat(plot_feature_2,dim=0)
        # plot_targets_2 = torch.cat(plot_targets_2,dim=0)
        # d2 = torch.tensor([2]*plot_targets_2.shape[0])

        # x_cat = torch.cat((plot_feature,plot_feature_2))
        # y_cat = torch.cat((plot_targets,plot_targets_2))
        # d_cat = torch.cat((d1,d2))
        # x_np = np.array(x_cat.cpu())
        # y_np = np.array(y_cat.cpu())
        # d_np = np.array(d_cat.cpu())
        # tsne = TSNE(n_components=2, random_state=33)
        # x_tsne = tsne.fit_transform(x_np)
        # y_tsne = y_np.reshape((-1, 1))
        # d_tsne = d_np.reshape((-1, 1))
        # print(x_tsne.shape,y_tsne.shape,d_tsne.shape)
        # tsne_plot(x_tsne, y_tsne, d=d_tsne, fig_name="/home/demo2/Code/meta-task/results/onemodel/00.pdf")

        # x_np = np.array(plot_feature.cpu())
        # y_np = np.array(plot_targets.cpu())
        # Tsne = Dim_Reducer(device,64)
        # x_tsne = Tsne.unsupervised_reduce(data_input=x_np) 
        # y_tsne = y_np.reshape((-1, 1))
        # fig_name = '/home/demo2/Code/meta-task/results/SMOTEtrans/'+str(classid)+'_3.pdf'
        # fig_name ='/home/demo2/Code/meta-task/results/onemodel/0.pdf'
        # tsne_plot(x_tsne,y_tsne,fig_name = fig_name)

    #     total = 0
    #     correct = 0
    #     class_correct = [0 for i in range(2)]
    #     for i,data in enumerate(testloader):
    #         images, labels = data[0].to(device),data[1].to(device)
    #         output = nets[classid](images)
    #         labels = update_tensor(labels, classid)
    #         _, predicted = torch.max(output.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         for pred,true_label in zip(predicted,labels):
    #             if pred == true_label:
    #                 class_correct[pred] += 1
    #     print('Accuracy on the test set: %.2f'%(100 * correct / total))
    #     print(class_correct)

    # for i in range(num_classes):
    #     nets[i].eval()
    # total = 0
    # correct = 0
    # class_correct = [0 for i in range(num_classes)]
    # for i,data in enumerate(testloader):
    #     images, labels = data[0].to(device),data[1].to(device)
    #     outputs = []
    #     for net in nets:
    #         outputs.append(net(images)[:,1]) #10*64*2
    #     outputs =  torch.stack(outputs, dim=1)
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()
    #     for pred,true_label in zip(predicted,labels):
    #         if pred == true_label:
    #             class_correct[pred] += 1
    # print('Accuracy on the test set: {:.2f} '.format(100 * correct / total))
    # print(class_correct)

def fusionmodeltsne():
    from models.modified_resnet import ResNet18
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
    indice = []
 
    # dataset_train = ConcatDataset([dataset_train, SMOTE_data])
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=64,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=64,
                                            shuffle=False, num_workers=2)
    index = [[] for i in range(num_classes)] #获取每个图片的索引
    for i in range(len(dataset_test)):
        index[dataset_test[i][1]].append(i)

    #图片采样
    slice = []
    for i in range(num_classes):
        slice.extend(random.sample(index[i],1000))


    sample_dataset = DataSplit(dataset_test,slice)



    # nets = []
    # #加载模型
    # for i in range(10):
    #     nets.append(ResNet18(num_classes=2).to(device))
    #     nets[i].load_state_dict(torch.load('/home/demo2/Code/meta-task/checkpoint/fusionmodel.pt'))
    #     nets[i].eval()

    net = ResNet18(num_classes=2).to(device)
    net.load_state_dict(torch.load('/home/demo2/Code/meta-task/checkpoint/fusionmodel_2.pt'))
    net.eval()
    #加载单一模型
    # nets.append(ResNet18(num_classes=10).to(device))
    # nets[0].load_state_dict(torch.load('/home/demo2/Code/meta-task/checkpoint/onemodel.pt'))
    # nets[0].eval()

    #画tsne图
    plot_features = [[] for i in range(10)]
    plot_targets = []
    sampleloader = torch.utils.data.DataLoader(sample_dataset, batch_size=64,
                                        shuffle=False, num_workers=2)
    for m,(images,labels) in enumerate(sampleloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs,_,_,_,_,features = net(images,out_feature = True)
        # for i in range(10):
        #     plot_features[i].append(features[i].data)
        plot_features[0].append(features.data)
        plot_targets.append(labels.data)
    plot_targets = torch.cat(plot_targets,dim=0)
    for classid in range(1):
        plot_feature = torch.cat(plot_features[classid],dim=0)
        x_np = np.array(plot_feature.cpu())
        y_np = np.array(plot_targets.cpu())
        # tsne = TSNE(n_components=2, random_state=33)
        # x_tsne = tsne.fit_transform(x_np)
        Tsne = Dim_Reducer(device, 64)
        x_tsne = Tsne.unsupervised_reduce(data_input=x_np) 
        y_tsne = y_np.reshape((-1,1))
        fig_name = '/home/demo2/Code/meta-task/results/fusionmodel/倒数第五层/'+str(classid)+'.pdf'
        tsne_plot(x_tsne,y_tsne,fig_name=fig_name)

if __name__ == '__main__':
    fusionmodeltsne()