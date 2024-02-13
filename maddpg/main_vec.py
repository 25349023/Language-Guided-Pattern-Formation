import dataclasses
import json
import os
import pprint
import random
import time
from multiprocessing import Queue
from multiprocessing.sharedctypes import Value

import numpy as np
import torch
import torch.multiprocessing as mp

import metrics
import multiagent.scenarios as scenarios
from ddpg_vec import DDPG
from ddpg_vec import hard_update
from ddpg_vec_hetero import DDPGH
from eval import eval_model_q
from replay_memory import ReplayMemory, Transition
from utils import *


def run_experiment(exp_name, args, test_q, metric_q):
    exp_save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(exp_save_dir, exist_ok=args.force)
    with open(f'{exp_save_dir}/train_args.json', 'w') as f:
        json.dump(vars(args), f)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(args.scenario, args)
    n_agents = env.n

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_act = n_actions(env.action_space)
    obs_dims = [env.observation_space[i].shape[0] for i in range(n_agents)]
    obs_dims.insert(0, 0)
    if 'hetero' in args.scenario:
        groups = scenarios.load(args.scenario + ".py").Scenario().group
        agent = DDPGH(args.gamma, args.tau, args.hidden_size,
                      env.observation_space[0].shape[0], n_act[0], n_agents, obs_dims, 0,
                      args.actor_lr, args.critic_lr,
                      args.fixed_lr, args.critic_type, args.train_noise, args.num_episodes,
                      args.num_steps, args.critic_dec_cen, args.target_update_mode, device, groups=groups)
        eval_agent = DDPGH(args.gamma, args.tau, args.hidden_size,
                           env.observation_space[0].shape[0], n_act[0], n_agents, obs_dims, 0,
                           args.actor_lr, args.critic_lr,
                           args.fixed_lr, args.critic_type, args.train_noise, args.num_episodes,
                           args.num_steps, args.critic_dec_cen, args.target_update_mode, 'cpu', groups=groups)
    else:
        agent = DDPG(args.gamma, args.tau, args.hidden_size,
                     env.observation_space[0].shape[0], n_act[0], n_agents, obs_dims, 0,
                     args.actor_lr, args.critic_lr,
                     args.fixed_lr, args.critic_type, args.actor_type, args.train_noise, args.num_episodes,
                     args.num_steps, args.critic_dec_cen, args.target_update_mode, device)
        eval_agent = DDPG(args.gamma, args.tau, args.hidden_size,
                          env.observation_space[0].shape[0], n_act[0], n_agents, obs_dims, 0,
                          args.actor_lr, args.critic_lr,
                          args.fixed_lr, args.critic_type, args.actor_type, args.train_noise, args.num_episodes,
                          args.num_steps, args.critic_dec_cen, args.target_update_mode, 'cpu')

    memory = ReplayMemory(args.replay_size)

    def evaluate_signal(metric=False):
        tr_log = {'num_adversary': 0,
                  'best_good_eval_reward': best_good_eval_reward,
                  'best_adversary_eval_reward': best_adversary_eval_reward,
                  'exp_save_dir': exp_save_dir, 'total_numsteps': total_numsteps,
                  'value_loss': value_loss, 'policy_loss': policy_loss,
                  'i_episode': i_episode, 'start_time': start_time}
        copy_actor_policy(agent, eval_agent)
        test_q.put([eval_agent, tr_log, metric])

    rewards = []
    total_numsteps = 0
    updates = 0
    best_eval_reward, best_good_eval_reward, best_adversary_eval_reward = -1000000000, -1000000000, -1000000000
    start_time = time.time()
    copy_actor_policy(agent, eval_agent)
    torch.save({'agents': eval_agent}, os.path.join(exp_save_dir, 'agents_best.ckpt'))

    for i_episode in range(args.num_episodes):
        obs_n = env.reset()
        episode_reward = 0
        episode_step = 0
        agents_rew = [[] for _ in range(n_agents)]
        while True:
            action_n = agent.select_action(torch.Tensor(obs_n).to(device), action_noise=True,
                                           param_noise=False).squeeze().cpu().numpy()
            next_obs_n, reward_n, done_n, info = env.step(action_n)
            total_numsteps += 1
            episode_step += 1
            terminal = (episode_step >= args.num_steps)

            action = torch.Tensor(action_n).view(1, -1)
            mask = torch.Tensor([[not done for done in done_n]])
            next_x = torch.Tensor(np.concatenate(next_obs_n, axis=0)).view(1, -1)
            reward = torch.Tensor([reward_n])
            x = torch.Tensor(np.concatenate(obs_n, axis=0)).view(1, -1)
            memory.push(x, action, mask, next_x, reward)
            for i, r in enumerate(reward_n):
                agents_rew[i].append(r)
            episode_reward += np.sum(reward_n)
            obs_n = next_obs_n

            if len(memory) > args.batch_size:
                if total_numsteps % args.steps_per_actor_update == 0:
                    for _ in range(args.updates_per_step):
                        transitions = memory.sample(args.batch_size)
                        batch = Transition(*zip(*transitions))
                        policy_loss = agent.update_actor_parameters(batch, i, args.shuffle)
                        updates += 1
                    # print(f'episode {i_episode}, p loss {policy_loss}, p_lr {agent.actor_lr}')

                if total_numsteps % args.steps_per_critic_update == 0:
                    value_losses = []
                    for _ in range(args.critic_updates_per_step):
                        transitions = memory.sample(args.batch_size)
                        batch = Transition(*zip(*transitions))
                        value_losses.append(agent.update_critic_parameters(batch, i, args.shuffle))
                        updates += 1
                    value_loss = torch.tensor(value_losses).mean().item()
                    # print(f'episode {i_episode}, q loss {value_loss}, '
                    #       f'q_lr {agent.critic_optim.param_groups[0]["lr"]}')
                    if args.target_update_mode == 'episodic':
                        hard_update(agent.critic_target, agent.critic)

            if done_n[0] or terminal:
                if (i_episode + 1) % args.eval_freq == 0:
                    print('train epidoe reward', episode_reward)
                break

        if not args.fixed_lr:
            agent.adjust_lr(i_episode)

        # writer.add_scalar('reward/train', episode_reward, i_episode)
        rewards.append(episode_reward)
        if (i_episode + 1) % args.eval_freq == 0:
            evaluate_signal()

    env.close()
    evaluate_signal(True)
    eval_metrics = metric_q.get()

    return eval_metrics


if __name__ == '__main__':
    args = get_args()

    if args.exp_name is None:
        args.exp_name = args.scenario + '_' + args.critic_type + '_' + args.target_update_mode + '_hiddensize' \
                        + str(args.hidden_size) + '_' + str(args.seed)
    print("=================Arguments==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    torch.set_num_threads(1)

    # for mp test
    test_q = Queue()
    metric_q = Queue()
    done_training = Value('i', False)
    p = mp.Process(target=eval_model_q, args=(test_q, done_training, args), kwargs={'metric_q': metric_q})
    p.start()

    metric_results = []

    if args.num_seeds > 1:
        exp_root = os.path.join(args.save_dir, args.exp_name)
        os.makedirs(exp_root, exist_ok=args.force)

        for i in range(args.num_seeds):
            args.seed = random.randrange(1000000)
            exp_name = os.path.join(args.exp_name, f'seed{args.seed}')
            metric_results.append(run_experiment(exp_name, args, test_q, metric_q))
    else:
        metric_results.append(run_experiment(args.exp_name, args, test_q, metric_q))

    print(f'======== Evaluation Results ========')
    pprint.pprint(metric_results)
    print()
    for field in dataclasses.fields(metrics.MetricRecord):
        print(f'Avg {field.name}: {np.mean([getattr(m, field.name) for m in metric_results])}')

    # time.sleep(5)
    done_training.value = True

    p.join()
