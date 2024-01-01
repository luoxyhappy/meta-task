from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
# from options import args_parser
import torch
import random
from sklearn.datasets import load_iris,load_digits
import random
from torch.utils.data import DataLoader
import matplotlib
def tsne_plot(x, y,fig_name=""):
    fig = plt.figure(figsize=(4, 3))
    print("Begin visualization ...")
    markers = ['.', 'o', 'v', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    # colors = ['red', 'green', 'blue', 'gold', 'lightcyan', 'lavender', 'yellowgreen', 'lavenderblush', 'thistle', 'aquamarine']
    # colors_1 = ['#000000', '#FF0000', '#FF8C00', 'gold', 'lightseagreen', 'royalblue', 'sage', 'palevioletred', 'darkviolet', 'g']
    colors = ['#000000', 'peru', '#FF8C00', 'gold', 'lightseagreen', 'royalblue', 'darkseagreen', 'violet', 'palevioletred', 'g']

    Label_Com = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 7
             }

    # S_data = np.hstack((x, y, d))
    S_data = np.hstack((x, y))
    # S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2], 'device': S_data[:,3]})
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    # for class_index in range(args.num_classes):
    #     for device_index in range(args.num_users):
    #         X = S_data.loc[(S_data['label'] == class_index) & (S_data['device'] == device_index)]['x']
    #         Y = S_data.loc[(S_data['label'] == class_index) & (S_data['device'] == device_index)]['y']
    #         # plt.scatter(X, Y, cmap='brg', s=100, marker='.', c='', edgecolors=colors[index], alpha=0.65)
    #         # plt.scatter(X, Y, marker='.', color=colors[class_index], alpha=0.6)
    #         plt.scatter(X, Y, marker=markers[device_index], c=colors[class_index])

    for class_index in range(10):
        X = S_data.loc[S_data['label'] == class_index]['x']
        Y = S_data.loc[S_data['label'] == class_index]['y']
        # plt.scatter(X, Y, cmap='brg', s=100, marker='.', c='', edgecolors=colors[index], alpha=0.65)
        # alpha set: fedavg-0.1, fedsem-0.08, fedper-0.03, hetfel-0.03
        plt.scatter(X, Y, marker='.', color=colors[class_index], alpha=0.5)  ##alpha透明度选项0-1 0表示完全透明


    # for class_index in range(args.num_classes):
    #     # global prototype
    #     X = S_data.loc[(S_data['label'] == class_index)]['x']
    #     Y = S_data.loc[(S_data['label'] == class_index)]['y']
    #     x_avg = np.average(X)
    #     y_avg = np.average(Y)
    #     plt.scatter(x_avg, y_avg, marker='o', c=colors[class_index], s=50)


    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值

    # plt.title(args.alg, fontsize=14, fontweight='normal', pad=20)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
    # plt.legend(scatterpoints=1, labels=Label_Com, loc='best', labelspacing=0.4, columnspacing=0.4, markerscale=2,
    #            bbox_to_anchor=(0.9, 0), ncol=12, prop=font1, handletextpad=0.1)
    plt.legend(scatterpoints=1, labels=Label_Com, loc='upper right', labelspacing=1, columnspacing=0.5,
               bbox_to_anchor=(1.0,1.0), ncol=2, prop=font1, handletextpad=0.1,fontsize='medium')

    # fig.show()
    plt.savefig(fig_name, format='pdf', dpi=600)

def tsne_plot_withd(x, y, d, num_classes=10,selected_clients=[0,1],fig_name=""):
    plt.rcParams['pdf.use14corefonts'] = True
    # plt.switch_backend('agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(8, 6))
    print("Begin visualization ...")
    # markers = ['.', 'o', 'v', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    markers = ['o', 'v', '*', 's', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    # colors = ['red', 'green', 'blue', 'gold', 'lightcyan', 'lavender', 'yellowgreen', 'lavenderblush', 'thistle', 'aquamarine']
    # colors_1 = ['#000000', '#FF0000', '#FF8C00', 'gold', 'lightseagreen', 'royalblue', 'sage', 'palevioletred', 'darkviolet', 'g']
    colors = ['#000000', 'peru', '#FF8C00', 'gold', 'lightseagreen', 'royalblue', 'darkseagreen', 'violet', 'palevioletred', 'g']
    Label_Com = ['0 ([1,0])', '0 ([0.55,0.05])','1 ([1,0])', '1 ([0.55,0.05])', '2 ([1,0])', '2 ([0.55,0.05])', '3 ([1,0])', '3 ([0.55,0.05])', '4 ([1,0])', '4 ([0.55,0.05])', '5 ([1,0])', '5 ([0.55,0.05])', '6 ([1,0])', '6 ([0.55,0.05])', '7 ([1,0])', '7 ([0.55,0.05])', '8 ([1,0])', '8 ([0.55,0.05])', '9 ([1,0])', '9 ([0.55,0.05])']
    # Label_Com = ['0 ([1,0])', '1 ([1,0])',  '2 ([1,0])',  '3 ([1,0])',  '4 ([1,0])', '5 ([1,0])', '6 ([1,0])', '7 ([1,0])', '8 ([1,0])', '9 ([1,0])', '0 ([0.55,0.05])','1 ([0.55,0.05])','2 ([0.55,0.05])','3 ([0.55,0.05])','4 ([0.55,0.05])', '5 ([0.55,0.05])', '6 ([0.55,0.05])','7 ([0.55,0.05])', '8 ([0.55,0.05])',  '9 ([0.55,0.05])']
    font1 = {'family': 'Helvetica',
             'size': 12
             }
 
    # S_data = np.hstack((x, y))  ##(bs,3)  xdim=2,ydim=1
    S_data = np.hstack((x, y, d))  ##(bs,4)  xdim=2,ydim=1,devicedim=1
    print(S_data.shape)
    # raise KeyError
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2], 'device': S_data[:,3]})
    # S_data = np.hstack((x, y))
    # S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    for class_index in range(num_classes):
        for i, device_index in enumerate(selected_clients):
            X = S_data.loc[(S_data['label'] == class_index) & (S_data['device'] == device_index)]['x']
            Y = S_data.loc[(S_data['label'] == class_index) & (S_data['device'] == device_index)]['y']
            # plt.scatter(X, Y, cmap='brg', s=100, marker='.', c='', edgecolors=colors[index], alpha=0.65)
            # plt.scatter(X, Y, marker='.', color=colors[class_index], alpha=0.6)
            # print("---------")
            # print(X)
            plt.scatter(X, Y, marker=markers[i], c=colors[class_index])

    # for class_index in range(num_classes):
    #     X = S_data.loc[S_data['label'] == class_index]['x']
    #     Y = S_data.loc[S_data['label'] == class_index]['y']
    #     # plt.scatter(X, Y, cmap='brg', s=100, marker='.', c='', edgecolors=colors[index], alpha=0.65)
    #     # alpha set: fedavg-0.1, fedsem-0.08, fedper-0.03, hetfel-0.03
    #     plt.scatter(X, Y, marker='.', color=colors[class_index], alpha=0.5)  ##alpha透明度选项0-1 0表示完全透明

    # for class_index in range(args.num_classes):
    #     # global prototype
    #     X = S_data.loc[(S_data['label'] == class_index)]['x']
    #     Y = S_data.loc[(S_data['label'] == class_index)]['y']
    #     x_avg = np.average(X)
    #     y_avg = np.average(Y)
    #     plt.scatter(x_avg, y_avg, marker='o', c=colors[class_index], s=50)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值

    # plt.title(args.alg, fontsize=14, fontweight='normal', pad=20)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
    # plt.legend(scatterpoints=1, labels=Label_Com, loc='best', labelspacing=0.4, columnspacing=0.4, markerscale=2,
    #            bbox_to_anchor=(0.9, 0), ncol=12, prop=font1, handletextpad=0.1)
    plt.legend(scatterpoints=1, labels=Label_Com, loc='upper right', labelspacing=0.1, columnspacing=0.0,
               bbox_to_anchor=(1.0,1.0), ncol=2, prop=font1, handletextpad=0.0,fontsize='large')

    # fig.show()
    plt.savefig(fig_name, format='pdf', dpi=300)



class Dim_Reducer(object):
    def __init__(self, device,batch_size):
        # self.args = args
        self.device = device
        self.batch_size = batch_size

    def unsupervised_reduce(self, data_input=None, reduce_method="tSNE"):
        if reduce_method == "tSNE":
            __reduce_method = TSNE(n_components=2, random_state=33)
            other_params = {}
        else:
            raise NotImplementedError

        if data_input is not None:
            data_tsne = __reduce_method.fit_transform(data_input)
       
        return data_tsne


    def get_features(self, model=None, test_data_by_class=None, num_points=1000):
        if model is not None:
            model.eval()
            model = model.to(self.device)
            with torch.no_grad():
                if test_data_by_class is not None:
                    feat_list_allclass = []
                    labels_list_allclass = []
                    for label, data_arranged in test_data_by_class.items():
                        class_data_loader = DataLoader(data_arranged, self.batch_size, drop_last=False)
                        feat_list = []
                        labels_list = []
                        # loaded_num_points = 0
                        for i, batch_data in enumerate(class_data_loader):
                            data, labels = batch_data
                            data = data.to(self.device)
                            output,feat, _,_,_,_, = model(data,True)
                            feat_list.append(feat)
                            labels_list.append(labels)
                            # loaded_num_points += data.shape[0]
                            # if num_points < loaded_num_points:
                            #     print("nm error")
                            #     break
                        # print(torch.cat(feat_list, dim=0).shape)
                        # feat = torch.cat(feat_list, dim=0)[:num_points].to('cpu')
                        # labels = torch.cat(labels_list, dim=0)[:num_points]
                        # print("all feature size:")
                        # print(torch.cat(feat_list, dim=0).shape)
                        feat_list_allclass.append(torch.cat(feat_list, dim=0)[:num_points].to('cpu'))
                        labels_list_allclass.append(torch.cat(labels_list, dim=0)[:num_points])
                    feat = torch.cat(feat_list_allclass,dim=0)
                    labels = torch.cat(labels_list_allclass,dim=0)

                # print("after selected feature size:")
                # print(feat.shape)
                data_input = feat
                # data_tsne = __reduce_method.fit_transform(data_input, **other_params)
                model.to("cpu")
        else:
            raise NotImplementedError

        return data_input, labels

    # def setup_tSNE_path(self, save_checkpoints_config,
    #         extra_name=None, epoch="init", postfix=None, file_format=".jpg"):

    #     postfix_str = "-" + postfix if postfix is not None else ""

    #     if extra_name is not None:
    #         save_activation_path = save_checkpoints_config["checkpoint_root_path"] \
    #             + "tSNE-" + extra_name + "-" + save_checkpoints_config["checkpoint_file_name_prefix"] \
    #             + "-epoch-"+str(epoch) + postfix_str + file_format
    #     else:
    #         save_activation_path = save_checkpoints_config["checkpoint_root_path"] \
    #             + "tSNE-" + save_checkpoints_config["checkpoint_file_name_prefix"] \
    #             + "-epoch-"+str(epoch) + postfix_str + file_format
    #     return save_activation_path

if __name__=='__main__':
    #用法1
    device=torch.device ( "cuda:0" if torch.cuda.is_available () else "cpu")
    temp_list = [x for x in range(10)]
    x1 = torch.randn(100,512)
    y1 = torch.tensor(np.random.choice(temp_list,100))
    Tsne = Dim_Reducer(device, 64)
    x_tsne = Tsne.unsupervised_reduce(data_input=x1) 
    y_tsne = y1.reshape((-1,1))
    tsne_plot(x_tsne,y_tsne,fig_name='test.pdf')
    #用法2
    # temp_list = [x for x in range(10)]
    # x1 = torch.randn(100,512)
    # y1 = torch.tensor(np.random.choice(temp_list,100))
    # d1 = torch.tensor([1]*100)
    # x2 = torch.randn(100,512)
    # y2 = torch.tensor(np.random.choice(temp_list,100))
    # d2 = torch.tensor([2]*100)
    # x_cat = torch.cat((x1,x2))
    # y_cat = torch.cat((y1,y2))
    # d_cat = torch.cat((d1,d2))
    # x_np = x_cat.numpy()
    # y_np = y_cat.numpy()
    # d_np = d_cat.numpy()
    # print(x_np.shape,y_np.shape,d_np.shape)
    # tsne = TSNE(n_components=2, random_state=33)
    # x_tsne = tsne.fit_transform(x_np)
    # y_tsne = y_np.reshape((-1, 1))
    # d_tsne = d_np.reshape((-1, 1))
    # print(x_tsne.shape,y_tsne.shape,d_tsne.shape)
    # tsne_plot(x_tsne, y_tsne, d=d_tsne, fig_name="../test_tsne.pdf")


