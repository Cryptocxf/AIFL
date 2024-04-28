from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from clients_attackers import *
from server import Server


def main(args):
    print('#####################')
    print('#####################')
    print('#####################')
    print(f'Aggregation Rule:\t{args.AR}\nData distribution:\t{args.loader_type}\nAttacks:\t{args.attacks} ')
    print('#####################')
    print('#####################')
    print('#####################')

    # 设置 (CPU) 生成随机数的种子，并返回一个torch.Generator对象。
    # 设置种子的用意是一旦固定种子，后面依次生成的随机数其实都是固定的。
    torch.manual_seed(args.seed)

    # 设备
    device = args.device

    # 攻击模式
    attacks = args.attacks

    # 实验记录输出文件
    writer = SummaryWriter(f'./logs/{args.output_folder}/{args.experiment_name}')

    # 加载数据集
    if args.dataset == 'mnist':
        from tasks import mnist
        trainData = mnist.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                           store=False)
        testData = mnist.test_dataloader(args.test_batch_size)
        Net = mnist.Net
        # 评价标准，交叉熵
        criterion = F.cross_entropy
    elif args.dataset == 'cifar':
        from tasks import cifar
        trainData = cifar.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                           store=False)
        testData = cifar.test_dataloader(args.test_batch_size)
        Net = cifar.Net
        criterion = F.cross_entropy
    elif args.dataset == 'cifar100':
        from tasks import cifar100
        trainData = cifar100.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                              store=False)
        testData = cifar100.test_dataloader(args.test_batch_size)
        Net = cifar100.Net
        criterion = F.cross_entropy
    elif args.dataset == 'imagenet':
        from tasks import imagenet
        trainData = imagenet.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                          store=False)
        testData = imagenet.test_dataloader(args.test_batch_size)
        Net = imagenet.Net
        criterion = F.cross_entropy

    # create server instance
    # 服务器模型
    model0 = Net()
    server = Server(model0, testData, criterion, device)
    # 设定聚合规则
    server.set_AR(args.AR)
    # If you choose the aggregation rule --AR to be attention or mlp ,
    # then you will also need to specify the location of the model parameters in --path_to_aggNet
    server.path_to_aggNet = args.path_to_aggNet
    # 保存模型权重
    if args.save_model_weights:
        server.isSaveChanges = True
        server.savePath = f'./AggData/{args.loader_type}/{args.dataset}/{args.attacks}/{args.AR}'
        from pathlib import Path
        Path(server.savePath).mkdir(parents=True, exist_ok=True)
        '''
        honest clients are labeled as 1, malicious clients are labeled as 0
        '''
        # 用用户标签向量标注良性和恶意
        # 返回一个全为1 的张量，形状由可变参数sizes定义
        # 在这里是num_clients长度的向量
        label = torch.ones(args.num_clients)
        print('label==',label)
        for i in args.attacker_list_labelFlipping:
            print('i=',i)
            label[i] = 0
        for i in args.attacker_list_labelFlippingDirectional:
            label[i] = 0
        for i in args.attacker_list_omniscient:
            label[i] = 0
        for i in args.attacker_list_backdoor:
            label[i] = 0
        for i in args.attacker_list_semanticBackdoor:
            label[i] = 0

        # 存储在结果路径中label.pt
        # .pt是pytorch模型文件,保存了权重及结构
        torch.save(label, f'{server.savePath}/label.pt')
        # 重新加载模型信息
        # checkpoint = torch.load(dir)
        # model.load_state_dict(checkpoint[‘net’])
        # optimizer.load_state_dict(checkpoint[‘optimizer’])
        # start_epoch = checkpoint[‘epoch’] + 1

    # create clients instance
    # 各类攻击者的list
    attacker_list_labelFlipping = args.attacker_list_labelFlipping
    # 遍历客户端，进行攻击设置
    for i in range(args.num_clients):
        model = Net()
        # 配置优化器
        if args.optimizer == 'SGD':
            # 可以调整随机梯度下降的动量系数
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # 实例化攻击者
        if i in attacker_list_labelFlipping:
            #print('label=',attacker_list_labelFlipping)
            client_i = Attacker_LabelFlipping01swap(i, model, trainData[i], optimizer, criterion, device,
                                                    args.inner_epochs)
            #print('label-1=',client_i)
        # 不进行攻击的诚实客户端
        else:
            client_i = Client(i, model, trainData[i], optimizer, criterion, device, args.inner_epochs)
        # 相当于收集上传阶段
        server.attach(client_i)

    loss, accuracy = server.test()
    steps = 0
    # 存储损失和准确率
    writer.add_scalar('test/loss', loss, steps)
    # add_scalar
    # 功能：将标量添加到 summary
    # 参数：
    # tag (string)：数据标识符
    # scalar_value (float or string/blobname)：要保存的数值
    # global_step (int)：全局步值
    # walltime (float)：可选参数，用于记录发生的时间，默认为 time.time()
    # 使用tensorboard读取
    # tensorboard --logdir=runs/flower_experiment
    writer.add_scalar('test/accuracy', accuracy, steps)


    # 训练轮次
    for j in range(args.epochs):
        steps = j + 1

        print('\n\n########EPOCH %d ########' % j)
        print('###Model distribution###\n')
        # 模型分发
        server.distribute()
        #         group=Random().sample(range(5),1)# 随机选取客户端集合
        # 训练的客户端集合
        group = range(args.num_clients)
        # 服务器模拟客户端训练
        server.train(group)
        #         server.train_concurrent(group)# 不清楚用处？

        # 测试损失及准确率
        loss, accuracy = server.test()

        # 记录各轮的损失及准确率
        writer.add_scalar('test/loss', loss, steps)
        writer.add_scalar('test/accuracy', accuracy, steps)

        if 'BACKDOOR' in args.attacks.upper():
            if 'SEMANTIC' in args.attacks.upper():
                loss, accuracy, bdata, bpred = server.test_semanticBackdoor()
            else:
                loss, accuracy = server.test_backdoor()

            writer.add_scalar('test/loss_backdoor', loss, steps)
            writer.add_scalar('test/backdoor_success_rate', accuracy, steps)

    writer.close()
