import dataclasses
import glob
import os
import pprint
import random

import numpy as np
import torch

import metrics
from eval_seq import eval_model_seq
from utils import get_args


def load_model(ckpt_dir, dev_str, suffix):
    device = torch.device(dev_str)
    agent_path = os.path.join(ckpt_dir, f'agents{suffix}.ckpt')
    eval_agent = torch.load(agent_path, map_location=device)['agents']
    eval_agent.device = dev_str
    return eval_agent


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

    ckpt_suffix = '_best' if args.ckpt_type == 'best' else ''

    metric_results = []
    if args.num_seeds == -1:
        for dirname in glob.iglob(os.path.join(args.save_dir, args.exp_name, '*seed*')):
            args.seed = random.randrange(50000)
            try:
                eval_agent = load_model(dirname, dev_str, ckpt_suffix)
            except FileNotFoundError:
                continue
            print(f'Running Evaluation for {os.path.basename(dirname)}')
            met, keep_dists = eval_model_seq(args, eval_agent)
            metric_results.append(met)
            np.save(os.path.join(dirname, 'keep_dists.npy'), keep_dists)
    else:
        ckpt_dir = os.path.join(args.save_dir, args.exp_name)
        eval_agent = load_model(ckpt_dir, dev_str, ckpt_suffix)
        met, keep_dists = eval_model_seq(args, eval_agent)
        metric_results.append(met)
        np.save(os.path.join(ckpt_dir, 'keep_dists.npy'), keep_dists)

    pprint.pprint(metric_results)
    print()
    for field in dataclasses.fields(metrics.MetricRecord):
        print(f'Avg {field.name}: {np.mean([getattr(m, field.name) for m in metric_results])}')


if __name__ == '__main__':
    main()
