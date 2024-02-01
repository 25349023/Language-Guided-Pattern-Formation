import os
import random

import numpy as np
import torch

from eval_real import eval_model_real
from utils import get_args


def main():
    args = get_args()

    if args.exp_name is None:
        args.exp_name = args.scenario + '_' + args.critic_type + '_' + args.target_update_mode + '_hiddensize' \
                        + str(args.hidden_size) + '_' + str(args.seed)
    print("=================Arguments==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    torch.set_num_threads(1)

    dev_str = "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    device = torch.device(dev_str)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    eval_agent = torch.load(os.path.join(exp_save_dir, 'agents_best.ckpt'), map_location=device)['agents']
    eval_agent.device = dev_str
    print(f'device: {eval_agent.device}')

    eval_model_real(args, eval_agent)


if __name__ == '__main__':
    main()
