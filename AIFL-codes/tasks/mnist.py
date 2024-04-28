from __future__ import print_function

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from dataloader import *


class Net(nn.Module):  #定义一个名为net的神经网络模型，它继承自‘nn.Module'类
    '''
    LeNet

    retrieved from the pytorch tutorial
    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

    '''

    def __init__(self): #定义了net类的初始化方法，用于定义神经网络的各个层次。
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): #定义了net类的前向传播方法，描述了数据再网络中的流动过程。
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):  #定义了一个辅助方法，用于计算输入张量的总特征值
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def getDataset(): #定义了一个函数，用于获取MNIST数据集。
    dataset = datasets.MNIST('./data',
                             train=True,
                             download=True,
                             transform=transforms.Compose([transforms.Resize((32, 32)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,), (0.3081,))]))
    return dataset


def basic_loader(num_clients, loader_type):  #定义了一个函数，用于创建基本的数据加载器。
    dataset = getDataset()
    return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk'):
    #定义一个函数，用于训练数据加载器的创建
    assert loader_type in ['iid', 'byLabel', 'dirichlet'], 'Loader has to be either \'iid\' or \'non_overlap_label \''
    if loader_type == 'iid':
        loader_type = iidLoader
    elif loader_type == 'byLabel':
        loader_type = byLabelLoader
    elif loader_type == 'dirichlet':
        loader_type = dirichletLoader

    if store:
        try:
            with open(path, 'rb') as handle:
                loader = pickle.load(handle)
        except:
            print('Loader not found, initializing one')
            loader = basic_loader(num_clients, loader_type)
    else:
        print('Initialize a data loader')
        loader = basic_loader(num_clients, loader_type)
    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)

    return loader


def test_dataloader(test_batch_size): #定义了一个函数，用于测试数据加载器的创建
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True,
                                                             transform=transforms.Compose(
                                                                 [transforms.Resize((32, 32)), transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,), (0.3081,))])),
                                              batch_size=test_batch_size, shuffle=True)
    return test_loader


if __name__ == '__main__':  #若这个脚本被直接运行，而不是被其他脚本导入，则执行以下代码块。
    from torchsummary import summary #导入summary函数，用于显示模型摘要信息。

    print("#Initialize a network")
    net = Net()
    summary(net.cuda(), (1, 32, 32))

    print("\n#Initialize dataloaders")
    loader_types = ['iid', 'byLabel', 'dirichlet']
    for i in range(len(loader_types)):
        loader = train_dataloader(10, loader_types[i], store=False)
        print(f"Initialized {len(loader)} loaders (type: {loader_types[i]}), each with batch size {loader.bsz}.\
        \nThe size of dataset in each loader are:")
        print([len(loader[i].dataset) for i in range(len(loader))])
        print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    print("\n#Feeding data to network")
    x = next(iter(loader[i]))[0].cuda()
    y = net(x)
    print(f"Size of input:  {x.shape} \nSize of output: {y.shape}")
