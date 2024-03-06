import torch
import torch.nn as nn
import math

'''
Krum aggregation
- find the point closest to its neignborhood

Reference:
Blanchard, Peva, Rachid Guerraoui, and Julien Stainer. "Machine learning with adversaries: Byzantine tolerant gradient descent." Advances in Neural Information Processing Systems. 2017.
`https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf`

'''
r_all = torch.zeros(10)


def MinNormalS(data):  #分数归一化处理
    min = data.min()
    max = data.max()
    if min != max:
        normalized = (max - data) / (max - min)
        total = torch.sum(normalized)
        p = normalized / total
        # print('total', total)
    else:
        normalized = data
        total = torch.sum(normalized)
        p = normalized / total
    sum = 0
    for i in range(4):
        if p[0, i] != 0:
            sum += p[0, i] * torch.log(p[0, i])
    E_s = (-1 / (math.log(4))) * sum
    return normalized, E_s


def MinNormalR(data):  #信誉归一化处理
    min = data.min()
    max = data.max()
    if min != max:
        normalized = (data - min) / (max - min)
        total = torch.sum(normalized)
        q = normalized / total
    else:
        normalized = data
        total = torch.sum(normalized)
        q = normalized / total
    sum = 0
    for i in range(4):
        if q[0, i] != 0:
            sum += q[0, i] * torch.log(q[0, i])
    E_r = (-1 / (math.log(4))) * sum
    return normalized, E_r


def combine(data):
    return torch.stack(data, dim=0)


def getKrum(input):
    '''
    compute krum or multi-krum of input. O(dn^2)

    input : batchsize* vector dimension * n

    return
        krum : batchsize* vector dimension * 1
        mkrum : batchsize* vector dimension * 1
    '''

    n = input.shape[-1]  # 最后一维在长度，n=10
    #print('input=',input)
    f = n // 2  # worse case 50% malicious points
    k = n - f - 2  # k=3

    # collection distance, distance from points to points
    x = input.permute(0, 2, 1)  # 每一块的行与列进行交换，即每块做转置
    cdist = torch.cdist(x, x, p=2)  # 计算x在每一行肯x每一行之间在距离，即欧几里得公式
    #print('cdist=', cdist)
    # find the k+1 nbh of each point
    nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)  # 输入数组cdist中距离最小在k+1个点，以及相应在索引地址

    #print('nbh=', nbh)
    # the point closest to its nbh
    i_star = torch.argmin(nbhDist.sum(2))#计算nbhDist每一行的和的最小值的索引
    #print('i_star=',i_star)

    #_, ii_star = torch.topk(nbhDist.sum(2), 4, largest=False)  # ii_star最好的k+1个更新的索引
    ii_star = nbh[:,i_star,:]
    #print('ii_stat=',ii_star)
    #print('nbhDist=', nbhDist)
    sum_nbhDist = nbhDist.sum(2)
    score1 = nbhDist[:, (ii_star[0, 0], ii_star[0, 1], ii_star[0, 2], ii_star[0, 3]), :]
    score = score1.sum(2)
    # print('score=', score)

    nor_s, Es = MinNormalS(score)
    # print('nor_s=',nor_s)
    # print('Es=', Es)

    for i in range(n):
        if -1 < r_all[i] and r_all[i] < 1:
            if (ii_star[0,0] == i) or (ii_star[0,1] == i) or (ii_star[0,2] == i) or (ii_star[0,3] == i):
                r_all[i] += 0.05
            else:
                r_all[i] = r_all[i] - (sum_nbhDist[0,i] / 4)*0.1
    #print('r_all', r_all)
    reputation = r_all[torch.tensor([[ii_star[0, 0], ii_star[0, 1], ii_star[0, 2], ii_star[0, 3]]], dtype=torch.long)]
    nor_r, Er = MinNormalR(reputation)
    # print('nor_r=',nor_r)
    # print('Er=',Er)
    alpha = ((1 - Es) / (2 - Es - Er))
    beta = ((1 - Er) / (2 - Es - Er))
    # print('alpha=',alpha)
    # print('beta=', beta)
    xi = []
    sum_xi = 0
    for i in range(4):
        sum_xi += (alpha * score[0, i] + beta * reputation[0, i])
    for i in range(4):
        xi.append(((alpha * score[0, i] + beta * reputation[0, i])) / sum_xi)
    #print('xi=',xi)

    # krum
    krum = input[:, :, [i_star]]
    #print('krum=', krum)

    # Multi-Krum
    #print('mkrum=',input[:, :, nbh[:, ii_star[0,0], :].view(-1)])
    #print('nbh[:,i_star,:].view(-1)=',nbh[:,ii_star[0,0],:].view(-1))
    # mkrum = input[:, :, nbh[:, i_star, :].view(-1)].mean(2, keepdims=True) #view（-1）作用将多维变成一维，例如a=[[1,2,3],[4,5,6]],a.view(-1)=tensor([1,2,3,4,5,6]);mean(2,keepdims=True)求第2维中元素在平均值

    # new_krum
    n_krum = input[:, :, [ii_star[0, 0], ii_star[0, 1], ii_star[0, 2], ii_star[0, 3]]]  # 找到k+1个最近的更新
    #print('n_krum=',n_krum)

    for i in range(4):
        #print('n_krum[0,:,i]=',n_krum[0, :, i])
        n_krum[0, :, i] *= xi[i]
        #print('n_krum[0,:,i]11=', n_krum[0, :, i])
    new_krum = torch.sum(n_krum, dim=2).unsqueeze(-1)
    #print('new_krum=',new_krum)

    return krum, new_krum


class Net(nn.Module):
    def __init__(self, mode='mkrum'):
        super(Net, self).__init__()
        assert (mode in ['krum', 'mkrum'])
        self.mode = mode

    def forward(self, input):
        #         print(input.shape)
        '''
        input: batchsize* vector dimension * n

        return 
            out : batchsize* vector dimension * 1
        '''
        krum, mkrum = getKrum(input)

        out = krum if self.mode == 'krum' else mkrum

        return out
