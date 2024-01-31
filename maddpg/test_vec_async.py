import argparse
import asyncio
import os
import random
import time

import numpy as np
import torch

from eval import eval_model_async


# from multiprocessing import Queue
# from multiprocessing.sharedctypes import Value
# import torch.multiprocessing as mp


def get_args():
    parser = argparse.ArgumentParser(description='PIC MARL Algorithm')
    parser.add_argument('--scenario', required=True,
                        help='name of the environment to run')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.01, metavar='G',
                        help='discount factor for model (default: 0.001)')
    parser.add_argument('--ou_noise', type=bool, default=True)
    parser.add_argument('--param_noise', type=bool, default=False)
    parser.add_argument('--train_noise', default=False, action='store_true')
    parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                        help='initial noise scale (default: 0.3)')
    parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                        help='final noise scale (default: 0.3)')
    parser.add_argument('--exploration_end', type=int, default=60000, metavar='N',
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=9, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='batch size (default: 128)')
    parser.add_argument('--num_steps', type=int, default=25, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=60000, metavar='N',
                        help='number of episodes (default: 1000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                        help='number of episodes (default: 128)')
    parser.add_argument('--updates_per_step', type=int, default=8, metavar='N',
                        help='model updates per simulator step (default: 5)')
    parser.add_argument('--critic_updates_per_step', type=int, default=8, metavar='N',
                        help='model updates per simulator step (default: 5)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--actor_lr', type=float, default=1e-2,
                        help='(default: 1e-4)')
    parser.add_argument('--critic_lr', type=float, default=1e-2,
                        help='(default: 1e-3)')
    parser.add_argument('--fixed_lr', default=False, action='store_true')
    parser.add_argument('--num_eval_runs', type=int, default=20, help='number of runs per evaluation (default: 5)')
    parser.add_argument("--exp_name", type=str, help="name of the experiment")
    parser.add_argument("--save_dir", type=str, default="./ckpt_plot",
                        help="directory in which training state and model should be saved")
    parser.add_argument('--static_env', default=False, action='store_true')
    parser.add_argument('--critic_type', type=str, default='mlp', help="Supports [mlp, gcn_mean, gcn_max]")
    parser.add_argument('--actor_type', type=str, default='mlp', help="Supports [mlp, gcn_max]")
    parser.add_argument('--critic_dec_cen', default='cen')
    parser.add_argument("--env_agent_ckpt", type=str, default='ckpt_plot/simple_tag_v5_al0a10_4/agents.ckpt')
    parser.add_argument('--shuffle', default=None, type=str, help='None|shuffle|sort')
    parser.add_argument('--episode_per_update', type=int, default=4, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--episode_per_actor_update', type=int, default=4)
    parser.add_argument('--episode_per_critic_update', type=int, default=4)
    parser.add_argument('--steps_per_actor_update', type=int, default=100)
    parser.add_argument('--steps_per_critic_update', type=int, default=100)
    # parser.add_argument('--episodes_per_update', type=int, default=4)
    parser.add_argument('--target_update_mode', default='soft', help='soft | hard | episodic')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument("--display", action="store_true", default=False)
    return parser.parse_args()


async def main():
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

    test_q = asyncio.Queue()
    done_training = asyncio.Event()
    t = asyncio.create_task(eval_model_async(test_q, done_training, args, False))

    tr_log = {'num_adversary': 0,
              'best_good_eval_reward': 0,
              'best_adversary_eval_reward': 0,
              'exp_save_dir': exp_save_dir, 'total_numsteps': total_numsteps,
              'value_loss': 0, 'policy_loss': 0,
              'i_episode': 0, 'start_time': start_time}
    await test_q.put([eval_agent, tr_log])

    await asyncio.sleep(5)
    done_training.set()
    await t


if __name__ == '__main__':
    asyncio.run(main())
