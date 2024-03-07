import contextlib
import os
import time

import numpy as np
import torch

from metrics import MetricRecord, completion_rate, distance_to_landmark
from utils import make_env, dict2csv


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def reset_evaluation():
    best_eval_reward = -100000000
    plot = {'good_rewards': [], 'adversary_rewards': [], 'rewards': [], 'steps': [], 'q_loss': [],
            'gcn_q_loss': [], 'p_loss': [], 'comp_rate': []}
    return best_eval_reward, plot


def eval_model_q(test_q, done_training, args, save=True, metric_q=None):
    eval_env = make_env(args.scenario, args, benchmark=True)
    best_eval_reward, plot = reset_evaluation()

    while True:
        if not test_q.empty():
            print('=================== start eval ===================')
            eval_env.seed(args.seed + 10)
            eval_rewards = []
            comp_rates = []
            collisions = []
            total_dists = []
            keep_dists = []
            agent, tr_log, ret_metric = test_q.get()

            with temp_seed(args.seed):
                for n_eval in range(args.num_eval_runs):
                    obs_n = eval_env.reset()
                    episode_reward = 0
                    episode_step = 0
                    episode_collisions = 0
                    dis_bw_agents = [1000] * args.num_agents

                    while True:
                        obs_n = np.asarray(obs_n)
                        action_n = agent.select_action(torch.Tensor(obs_n), action_noise=True,
                                                       param_noise=False).squeeze().cpu().numpy()
                        next_obs_n, reward_n, done_n, info_n = eval_env.step(action_n)
                        col, min_d = zip(*info_n)
                        episode_reward += np.sum(reward_n)
                        episode_collisions += sum(col) // 2
                        dis_bw_agents = [min(old, new) for old, new in zip(dis_bw_agents, min_d)]

                        episode_step += 1
                        terminal = (episode_step >= args.num_steps)
                        obs_n = next_obs_n

                        time.sleep(0.02)
                        eval_env.render()

                        if done_n[0] or terminal:
                            eval_rewards.append(episode_reward)
                            comp_rates.append(completion_rate(eval_env))
                            collisions.append(episode_collisions)
                            total_dists.append(distance_to_landmark(eval_env))
                            keep_dists.append(np.mean(dis_bw_agents))
                            if n_eval % 10 == 0:
                                print(f'test reward: {episode_reward}, total collision: {episode_collisions}, '
                                      f'completion rate: {comp_rates[-1]}')
                            break

            mean_reward = np.mean(eval_rewards)
            mean_cr = np.mean(comp_rates)

            if save and mean_reward > best_eval_reward:
                best_eval_reward = mean_reward
                torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents_best.ckpt'))

            plot['rewards'].append(mean_reward)
            plot['steps'].append(tr_log['total_numsteps'])
            plot['q_loss'].append(tr_log['value_loss'])
            plot['p_loss'].append(tr_log['policy_loss'])
            plot['comp_rate'].append(mean_cr)
            print("========================================================")
            print(f"Episode: {tr_log['i_episode']}, total numsteps: {tr_log['total_numsteps']}, "
                  f"{args.num_eval_runs} eval runs, total time: {time.time() - tr_log['start_time']} s")
            print(f"reward: avg {mean_reward} std {np.std(eval_rewards)}, "
                  f"best reward {best_eval_reward}")

            dict2csv(plot, os.path.join(tr_log['exp_save_dir'], 'train_curve.csv'))

            if ret_metric:
                metric_record = MetricRecord(mean_reward, mean_cr, np.mean(collisions),
                                             np.mean(total_dists), np.mean(keep_dists))
                metric_q.put(metric_record)
                best_eval_reward, plot = reset_evaluation()

            if save:
                torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents.ckpt'))

        if done_training.value and test_q.empty():
            break

    eval_env.close()
