import os
import random
import time
from multiprocessing import Queue
from multiprocessing.sharedctypes import Value

import numpy as np
import torch
import torch.multiprocessing as mp

import multiagent.scenarios as scenarios
from ddpg_vec import DDPG
from ddpg_vec import hard_update
from ddpg_vec_hetero import DDPGH
from eval import eval_model_q
from replay_memory import ReplayMemory, Transition
from utils import *


def run_experiment(exp_name, test_q, args):
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
    feat_dims = []
    for i in range(n_agents):
        feat_dims.append(env.observation_space[i].shape[0])

    rewards = []
    total_numsteps = 0
    updates = 0
    exp_save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
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
                    print('episode {}, p loss {}, p_lr {}'.
                          format(i_episode, policy_loss, agent.actor_lr))

                if total_numsteps % args.steps_per_critic_update == 0:
                    value_losses = []
                    for _ in range(args.critic_updates_per_step):
                        transitions = memory.sample(args.batch_size)
                        batch = Transition(*zip(*transitions))
                        value_losses.append(agent.update_critic_parameters(batch, i, args.shuffle))
                        updates += 1
                    value_loss = torch.tensor(value_losses).mean().item()
                    print('episode {}, q loss {},  q_lr {}'.
                          format(i_episode, value_loss, agent.critic_optim.param_groups[0]['lr']))
                    if args.target_update_mode == 'episodic':
                        hard_update(agent.critic_target, agent.critic)

            if done_n[0] or terminal:
                print('train epidoe reward', episode_reward)
                break

        if not args.fixed_lr:
            agent.adjust_lr(i_episode)

        # writer.add_scalar('reward/train', episode_reward, i_episode)
        rewards.append(episode_reward)
        if (i_episode + 1) % args.eval_freq == 0:
            tr_log = {'num_adversary': 0,
                      'best_good_eval_reward': best_good_eval_reward,
                      'best_adversary_eval_reward': best_adversary_eval_reward,
                      'exp_save_dir': exp_save_dir, 'total_numsteps': total_numsteps,
                      'value_loss': value_loss, 'policy_loss': policy_loss,
                      'i_episode': i_episode, 'start_time': start_time}
            copy_actor_policy(agent, eval_agent)
            test_q.put([eval_agent, tr_log])

    env.close()
    return episode_reward


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
    done_training = Value('i', False)
    p = mp.Process(target=eval_model_q, args=(test_q, done_training, args))
    p.start()

    train_rewards = []
    for i in range(args.num_seeds):
        if args.num_seeds > 1:
            args.seed = random.randrange(1000000)
            exp_name = args.exp_name + f'_seed{args.seed}'
        else:
            exp_name = args.exp_name

        tr_rw = run_experiment(exp_name, test_q, args)
        train_rewards.append(tr_rw)

    print(f'training rewards for each seed = {train_rewards}')
    time.sleep(5)
    done_training.value = True

    p.join()
