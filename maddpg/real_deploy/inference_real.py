import os
import random
import time

import numpy as np
import torch

from eval_seq import eval_model_seq
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

    total_numsteps = 0
    exp_save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    start_time = time.time()
    eval_agent = torch.load(os.path.join(exp_save_dir, 'agents_best.ckpt'), map_location=device)['agents']
    eval_agent.device = dev_str
    print(f'device: {eval_agent.device}')

    tr_log = {'num_adversary': 0,
              'best_good_eval_reward': 0,
              'best_adversary_eval_reward': 0,
              'exp_save_dir': exp_save_dir, 'total_numsteps': total_numsteps,
              'value_loss': 0, 'policy_loss': 0,
              'i_episode': 0, 'start_time': start_time}
    eval_model_seq(args, eval_agent, tr_log, False)


if __name__ == '__main__':
    main()
