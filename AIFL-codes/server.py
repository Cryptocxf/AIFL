from __future__ import print_function

from copy import deepcopy

import torch
import torch.nn.functional as F

from utils import utils
import time


class Server():
    def __init__(self, model, dataLoader, criterion=F.nll_loss, device='cpu'):
        # nll_loss负对数似然损失函数，用于处理多分类问题，输入是对数化的概率值。
        self.clients = []
        self.model = model
        self.dataLoader = dataLoader  # 数据加载
        self.device = device  # 设备
        self.emptyStates = None
        self.init_stateChange()
        self.Delta = None
        self.iter = 0
        self.AR = self.FedAvg
        self.func = torch.mean
        self.isSaveChanges = False
        self.savePath = './AggData'
        self.criterion = criterion
        self.path_to_aggNet = ""

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.emptyStates = states

    def attach(self, c):
        self.clients.append(c)

    # 分发模型
    # 设定client的模型
    def distribute(self):
        for c in self.clients:
            c.setModelParameter(self.model.state_dict())

    def test(self):
        print("[Server] Start testing")
        self.model.to(self.device)  # GPU
        self.model.eval()
        test_loss = 0
        correct = 0
        count = 0
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                if output.dim() == 1:
                    pred = torch.round(torch.sigmoid(output))
                else:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += pred.shape[0]
        test_loss /= count
        accuracy = 100. * correct / count
        self.model.cpu()  ## avoid occupying gpu when idle
        print(
            '[Server] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, count,
                                                                                          accuracy))
        return test_loss, accuracy

    # 训练
    # 从服务端控制和模拟client的训练，参数为训练的client集合
    def train(self, group):
        selectedClients = [self.clients[i] for i in group]
        for c in selectedClients:
            c.train()
            c.update()

        if self.isSaveChanges:
            self.saveChanges(selectedClients)

        tic = time.perf_counter()
        Delta = self.AR(selectedClients)
        self.Delta = Delta
        toc = time.perf_counter()
        print(f"[Server] The aggregation takes {toc - tic:0.6f} seconds.\n")

        for param in self.model.state_dict():
            self.model.state_dict()[param] += Delta[param]
        self.iter += 1

    def saveChanges(self, clients):

        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]

        param_trainable = utils.getTrainableParameters(self.model)

        param_nontrainable = [param for param in Delta.keys() if param not in param_trainable]
        for param in param_nontrainable:
            del Delta[param]
        print(f"[Server] Saving the model weight of the trainable paramters:\n {Delta.keys()}")
        for param in param_trainable:
            ##stacking the weight in the innerest dimension
            param_stack = torch.stack([delta[param] for delta in deltas], -1)
            shaped = param_stack.view(-1, len(clients))
            Delta[param] = shaped

        saveAsPCA = False
        saveOriginal = True
        if saveAsPCA:
            from utils import convert_pca
            proj_vec = convert_pca._convertWithPCA(Delta)
            savepath = f'{self.savePath}/pca_{self.iter}.pt'
            torch.save(proj_vec, savepath)
            print(
                f'[Server] The PCA projections of the update vectors have been saved to {savepath} (with shape {proj_vec.shape})')
        #             return
        if saveOriginal:
            savepath = f'{self.savePath}/{self.iter}.pt'

            torch.save(Delta, savepath)
            print(f'[Server] Update vectors have been saved to {savepath}')

    ## Aggregation functions ##
    # 设置聚合方案
    def set_AR(self, ar):
        if ar == 'fedavg':
            self.AR = self.FedAvg
        elif ar == 'krum':
            self.AR = self.krum
        elif ar == 'newkrum':
            self.AR = self.newkrum
        else:
            raise ValueError("Not a valid aggregation rule or aggregation rule not implemented")

    # 联邦平均
    def FedAvg(self, clients):
        # torch.mean求平均
        # dim指维度，输入为(m,n,k)，dim=-1，则输出(m,n,1)或(m,n)，keepdim选择是否需要保持结构，即1
        out = self.FedFuncWholeNet(clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out

    # 中值算法
    def FedMedian(self, clients):
        # torch.median
        out = self.FedFuncWholeNet(clients, lambda arr: torch.median(arr, dim=-1, keepdim=True)[0])
        return out

    # krum
    def krum(self, clients):
        from rules.newkrum import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net('krum').cpu()(arr.cpu()))
        return out

    # newkrum
    def newkrum(self, clients):
        from rules.newkrum import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net('newkrum').cpu()(arr.cpu()))
        return out

    def FedFuncWholeNet(self, clients, func):
        '''
        The aggregation rule views the update vectors as stacked vectors (1 by d by n).
        '''
        # 将更新向量视为堆叠向量（单行，d维，n个）
        # deepcopy()可以复制的列表里包含子列表，但copy()不可以
        Delta = deepcopy(self.emptyStates)
        # 获得所有客户端的模型更新
        deltas = [c.getDelta() for c in clients]
        # 模型更新转化为一维向量
        vecs = [utils.net2vec(delta) for delta in deltas]
        # isfinite返回一个带有布尔元素的新张量，表示每个元素是否是有限的。
        # 当实数值不是 NaN、负无穷或无穷大时，它们是有限的。当复数值的实部和虚部都是有限的时，复数值是有限的。
        # tensor.all 如果张量tensor中所有元素都是True, 才返回True; 否则返回False
        # tensor.item 该方法的功能是以标准的Python数字的形式来返回这个张量的值。这个方法只能用于只包含一个元素的张量。
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        # stack() 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
        # unsqueeze在指定的位置插入一个维度，有两个参数，input是输入的tensor,dim是要插到的维度
        result = func(torch.stack(vecs, 1).unsqueeze(0))  # input as 1 by d by n
        # torch.view(x, -1) & torch.view(-1)将原 tensor 以参数 x 设置第一维度重排，第二维度自动补齐；当没有参数 x 时，直接重排为一维的 tensor
        result = result.view(-1)
        utils.vec2net(result, Delta)
        return Delta

    def FedFuncWholeStateDict(self, clients, func):
        '''
        The aggregation rule views the update vectors as a set of state dict.
        '''
        # 不同于上一个函数，该函数没有转化为一维向量
        # 保留字典的格式，从而实现分层
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        # sanity check, remove update vectors with nan/inf values
        deltas = [delta for delta in deltas if torch.isfinite(utils.net2vec(delta)).all().item()]

        resultDelta = func(deltas)

        Delta.update(resultDelta)
        return Delta
