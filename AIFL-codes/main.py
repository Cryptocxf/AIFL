from utils.allocateGPU import *

# allocate_gpu()

import AIFLparser
import _main

# no attacker
# python main.py --dataset mnist --AR fedavg --loader_type dirichlet  --epochs 30 --attacks 'noattack'

# labelFlipping
# python main.py --dataset mnist --AR fedavg --loader_type dirichlet  --epochs 30 --attacks 'labelflipping' --n_attacker_labelFlipping 2

# labelFlippingDirectional
# python main.py --dataset mnist --AR fedavg --loader_type dirichlet  --epochs 30 --attacks 'labelflippingDirectional' --n_attacker_labelFlippingDirectional 2



if __name__ == "__main__":
    args = AIFLparser.parse_args()# 解析参数
    print("#" * 64)
    for i in vars(args):
        print(f"#{i:>40}: {str(getattr(args, i)):<20}#")
    print("#" * 64)
    _main.main(args)# 传参
