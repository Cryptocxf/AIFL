from __future__ import print_function

import os
import pickle

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models.resnet import resnet18

from dataloader import *

def Net( ):
    num_classes = 200
    model = resnet18(pretrained=True)
    n = model.fc.in_features
    model.fc = nn.Linear(n, num_classes)
    return model

def getDataset(root_dir='./data/tiny-imagenet', train=True):
    if train:
        train_dir = os.path.join(root_dir, 'train')
        return datasets.ImageFolder(train_dir, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    else:
        test_dir = os.path.join(root_dir, 'val')
        return datasets.ImageFolder(test_dir, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))

def basic_loader(num_clients, loader_type, **kwargs):
    dataset = getDataset(**kwargs)
    return loader_type(num_clients, dataset)

def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk', **kwargs):
    assert loader_type in ['iid', 'byLabel', 'dirichlet'], 'Loader has to be one of the  \'iid\',\'byLabel\',\'dirichlet\''
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
            print('loader not found, initialize one')
            loader = basic_loader(num_clients, loader_type, **kwargs)
    else:
        print('initialize a data loader')
        loader = basic_loader(num_clients, loader_type, **kwargs)
    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)

    return loader

def test_dataloader(test_batch_size, **kwargs):
    test_dataset = getDataset(train=False, **kwargs)
    return torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

if __name__ == '__main__':
    from torchsummary import summary

    print("# Initialize a network")
    net = Net()
    summary(net.cuda(), (3, 64, 64))

    print("\n# Initialize dataloaders")
    loader_types = ['iid', 'dirichlet']
    for loader_type in loader_types:
        loader = train_dataloader(100, loader_type, store=False)
        print(f"Initialized {len(loader)} loaders (type: {loader_type}), each with batch size {loader.bsz}.")
        print(f"The size of dataset in each loader are: {[len(loader[i].dataset) for i in range(len(loader))]}")
        print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    print("\n# Feeding data to network")
    x = next(iter(loader[i]))[0].cuda()
    y = net(x)
    print(f"Size of input:  {x.shape} \nSize of output: {y.shape}")
