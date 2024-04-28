from __future__ import print_function

import torch
import torch.nn.functional as F

from clients import *



class Attacker_LabelFlipping1to7(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss, device='cpu', inner_epochs=1):
        super(Attacker_LabelFlipping1to7, self).__init__(cid, model, dataLoader, optimizer, criterion, device,
                                                         inner_epochs)

    def data_transform(self, data, target):
        target_ = torch.tensor(list(map(lambda x: 7 if x == 1 else x, target)))
        assert target.shape == target_.shape, "Inconsistent target shape"
        return data, target_




class Attacker_LabelFlipping01swap(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss, device='cpu', inner_epochs=1):
        super(Attacker_LabelFlipping01swap, self).__init__(cid, model, dataLoader, optimizer, criterion, device,
                                                           inner_epochs)

    def data_transform(self, data, target):
        target_ = torch.tensor(list(map(lambda x: 1 - x if x in [0, 1] else x, target)))
        #print('tar=',target,target_)
        assert target.shape == target_.shape, "Inconsistent target shape"
        return data, target_
    '''
    提升5倍攻击效果
    '''
    def update(self):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState = self.model.state_dict()
        for param in self.originalState:
            self.stateChange[param] = newState[param] - self.originalState[param]
            self.stateChange[param] = self.stateChange[param]*5
        self.isTrained = False
