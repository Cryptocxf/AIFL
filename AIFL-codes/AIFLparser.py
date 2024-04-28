import argparse
import torch

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)# 训练的批次大小
    parser.add_argument("--test_batch_size", type=int, default=64)# 测试的批次大小
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default='SGD')
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate of models")
    parser.add_argument("--momentum", type=float, default=0.5)# 动量参数
    parser.add_argument("--seed", type=int, default=0)# 种子
    parser.add_argument("--log_interval", type=int, default=2)# 日志输出频率
    parser.add_argument("-n", "--num_clients", type=int, default=10)# 客户端数量
    parser.add_argument("--output_folder", type=str, default="experiments",# 输出文件夹的名称
                        help="path to output folder, e.g. \"experiments\"")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar", "cifar100", "imagenet"], default="mnist")
    parser.add_argument("--loader_type", type=str, choices=["iid", "byLabel", "dirichlet"], default="iid")# 数据集加载模式，iid、按标签、迪利克雷分布划分非iid
    parser.add_argument("--loader_path", type=str, default="./data/loader.pk", help="where to save the data partitions")# 存储划分的数据集
    parser.add_argument("--AR", type=str, help="aggregation rule")# 聚合规则？
    parser.add_argument("--n_attacker_labelFlipping", type=int, default=0)# 标签反转攻击者数量
    parser.add_argument("--n_attacker_labelFlippingDirectional", type=int, default=0)# 标签反转攻击者数量 Directional？
    parser.add_argument("--attacks", type=str, help="if contains \"backdoor\", activate the corresponding tests")# 攻击模式。后面用做了种子，通过几个关键字选择攻击模式[是否有'backdoor']['RANDOM'或'CUSTOM']['SEMANTIC']
    parser.add_argument("--save_model_weights", action="store_true",# 是否保存模型权重
                        help="If --save_model_weights is specified, the local models and their label (benign or malicious) will be saved under ./AggData directory. This is how we can collect empirical update vectors.")
    parser.add_argument("--experiment_name", type=str)# 实验名称
    parser.add_argument("--path_to_aggNet", type=str,# 模型参数？
                        help="If you choose the aggregation rule --AR to be attention or mlp , then you will also need to specify the location of the model parameters in --path_to_aggNet")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default='cuda' if torch.cuda.is_available() else "cpu")# 使用设备
    parser.add_argument("--inner_epochs", type=int, default=1)# 内部轮次？

    args = parser.parse_args()
    # 客户端数量
    n = args.num_clients

    # 标签反转攻击者
    m = args.n_attacker_labelFlipping
    args.attacker_list_labelFlipping = np.random.permutation(list(range(n)))[:m]

    return args

if __name__ == "__main__":

    import _main

    args = parse_args()
    print("#" * 64)
    for i in vars(args):
        print(f"#{i:>40}: {str(getattr(args, i)):<20}#")
    print("#" * 64)
    _main.main(args)
