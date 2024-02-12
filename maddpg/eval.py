import contextlib
import os
import time

import numpy as np
import torch

from metrics import MetricRecord, completion_rate
from utils import make_env, dict2csv


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def eval_model_q(test_q, done_training, args, save=True, metric_q=None):
    eval_env = make_env(args.scenario, args, benchmark=True)

    while True:
        if not test_q.empty():
            print('=================== start eval ===================')
            plot = {'good_rewards': [], 'adversary_rewards': [], 'rewards': [], 'steps': [], 'q_loss': [],
                    'gcn_q_loss': [], 'p_loss': [], 'final': [], 'abs': []}
            best_eval_reward = -100000000

            eval_env.seed(args.seed + 10)
            eval_rewards = []
            comp_rates = []
            agent, tr_log, ret_metric = test_q.get()
            with temp_seed(args.seed):
                for n_eval in range(args.num_eval_runs):
                    obs_n = eval_env.reset()
                    episode_reward = 0
                    episode_step = 0
                    while True:
                        action_n = agent.select_action(torch.Tensor(obs_n), action_noise=True,
                                                       param_noise=False).squeeze().cpu().numpy()
                        next_obs_n, reward_n, done_n, info_n = eval_env.step(action_n)
                        # TODO: redesign the get_info method to collect collision info
                        episode_step += 1
                        terminal = (episode_step >= args.num_steps)
                        episode_reward += np.sum(reward_n)
                        obs_n = next_obs_n

                        time.sleep(0.02)
                        eval_env.render()

                        if done_n[0] or terminal:
                            eval_rewards.append(episode_reward)
                            if n_eval % 100 == 0:
                                print('test reward', episode_reward)
                            comp_rates.append(completion_rate(eval_env))
                            break

            mean_reward = np.mean(eval_rewards)
            if save and mean_reward > best_eval_reward:
                best_eval_reward = mean_reward
                torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents_best.ckpt'))

            plot['rewards'].append(mean_reward)
            plot['steps'].append(tr_log['total_numsteps'])
            plot['q_loss'].append(tr_log['value_loss'])
            plot['p_loss'].append(tr_log['policy_loss'])
            print("========================================================")
            print(f"Episode: {tr_log['i_episode']}, total numsteps: {tr_log['total_numsteps']}, "
                  f"{args.num_eval_runs} eval runs, total time: {time.time() - tr_log['start_time']} s")
            print(f"reward: avg {mean_reward} std {np.std(eval_rewards)}, "
                  f"best reward {best_eval_reward}")

            # plot['final'].append(np.mean(plot['rewards'][-10:]))
            # plot['abs'].append(best_eval_reward)
            dict2csv(plot, os.path.join(tr_log['exp_save_dir'], 'train_curve.csv'))

            metric_record = MetricRecord(mean_reward, np.mean(comp_rates), 0)

            if ret_metric:
                metric_q.put(metric_record)

            if save:
                torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents.ckpt'))
        if done_training.value and test_q.empty():
            break

    eval_env.close()


async def eval_model_async(test_q, done_training, args, save=True):
    plot = {'good_rewards': [], 'adversary_rewards': [], 'rewards': [], 'steps': [], 'q_loss': [], 'gcn_q_loss': [],
            'p_loss': [], 'final': [], 'abs': []}
    best_eval_reward = -100000000
    eval_env = make_env(args.scenario, args)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    while True:
        agent, tr_log = await test_q.get()
        print('=================== start eval ===================')
        eval_env.seed(args.seed + 10)
        eval_rewards = []
        good_eval_rewards = []
        with temp_seed(args.seed):
            for n_eval in range(args.num_eval_runs):
                obs_n = eval_env.reset()
                episode_reward = 0
                episode_step = 0
                n_agents = eval_env.n
                agents_rew = [[] for _ in range(n_agents)]
                while True:
                    action_n = agent.select_action(torch.Tensor(obs_n).to(device), action_noise=True,
                                                   param_noise=False).squeeze().cpu().numpy()
                    next_obs_n, reward_n, done_n, _ = eval_env.step(action_n)
                    episode_step += 1
                    terminal = (episode_step >= args.num_steps)
                    episode_reward += np.sum(reward_n)
                    for i, r in enumerate(reward_n):
                        agents_rew[i].append(r)
                    obs_n = next_obs_n

                    time.sleep(0.1)
                    eval_env.render()

                    if done_n[0] or terminal:
                        eval_rewards.append(episode_reward)
                        agents_rew = [np.sum(rew) for rew in agents_rew]
                        good_reward = np.sum(agents_rew)
                        good_eval_rewards.append(good_reward)
                        if n_eval % 100 == 0:
                            print('test reward', episode_reward)
                        break
            if save and np.mean(eval_rewards) > best_eval_reward:
                best_eval_reward = np.mean(eval_rewards)
                torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents_best.ckpt'))

            plot['rewards'].append(np.mean(eval_rewards))
            plot['steps'].append(tr_log['total_numsteps'])
            plot['q_loss'].append(tr_log['value_loss'])
            plot['p_loss'].append(tr_log['policy_loss'])
            print("========================================================")
            print(
                "Episode: {}, total numsteps: {}, {} eval runs, total time: {} s".
                format(tr_log['i_episode'], tr_log['total_numsteps'], args.num_eval_runs,
                       time.time() - tr_log['start_time']))
            print("GOOD reward: avg {} std {}, average reward: {}, best reward {}".format(np.mean(eval_rewards),
                                                                                          np.std(eval_rewards),
                                                                                          np.mean(plot['rewards'][
                                                                                                  -10:]),
                                                                                          best_eval_reward))
            plot['final'].append(np.mean(plot['rewards'][-10:]))
            plot['abs'].append(best_eval_reward)
            dict2csv(plot, os.path.join(tr_log['exp_save_dir'], 'train_curve.csv'))
            eval_env.close()
        if done_training.is_set() and test_q.empty():
            if save:
                torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents.ckpt'))
            break
