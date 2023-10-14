import torch
import torch.nn as nn
from .BasicModule import BasicModule
import torch.nn.functional as F
from torch.nn import Parameter
import math

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


''' ConvNet '''
class ConvNet(BasicModule):
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='groupnorm', net_pooling='avgpooling', im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.model_name = "ConvNet"
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x, out_feature = False):
        # out = self.features(x)  #共有12层
        # import pdb; pdb.set_trace()
        fourth_middle = self.features[:-9](x)
        third_middle = self.features[-9: -5](fourth_middle)
        second_middle = self.features[-5: -1](third_middle)
        out = self.features[-1:](second_middle)
        # second_middle = self.features[:-3](x)
        out_middle = out
        out = out.view(out.size(0), -1)
        out_final = self.classifier(out)
        if out_feature == True:
            return out_final, out_middle.view(out_middle.size(0), -1), second_middle.view(second_middle.size(0), -1), third_middle.view(third_middle.size(0), -1), fourth_middle.view(fourth_middle.size(0), -1),0
        else:
            return out_final

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


if __name__ == '__main__':
    net = ConvNet()
    outputs_T1, out_middle_feature1, second_middle_feature1, third_middle_feature1, fourth_middle_feature1 = net(torch.randn(20, 3, 32, 32),True)

    print(out_middle_feature1.shape, second_middle_feature1.shape, third_middle_feature1.shape, fourth_middle_feature1.shape) ##torch.Size([20, 2048]) torch.Size([20, 8192]) torch.Size([20, 32768]) torch.Size([20, 131072])
    # for x in y:
    #     print(x.size())
    # print(y.size())