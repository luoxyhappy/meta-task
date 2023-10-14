#coding:utf8
import torch as t
import torch.nn.init as init
import time


class BasicModule(t.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        if name is None:
            prefix = 'modelstorage/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name


class Flatten(t.nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flatten, self).__init__()
        #self.size = size

    def forward(self, x):
        # print(x.size())
        return x.view(x.size(0), -1)

class Unflatten(t.nn.Module):
    def __init__(self, channel, height, width):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


def print_network(model, _print):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # _print(name)
        _print("The number of parameters of this model: {}".format(num_params))
        _print(model)

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, t.nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, t.nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, t.nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)