import contextlib
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from metrics import MetricRecord, completion_rate, distance_to_landmark
from utils import make_env


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def save_frame(env):
    os.makedirs('frames', exist_ok=True)
    fig = env.get_frame()
    plt.imsave(f'frames/lastframe_{datetime.now():%y%m%d-%H%M%S}.png', fig)


def eval_model_seq(args, agent, ckpt_dir):
    eval_env = make_env(args.scenario, args, benchmark=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    print('=================== start eval ===================')
    eval_env.seed(args.seed + 10)
    eval_rewards = []
    comp_rates = []
    collisions = []
    total_dists = []
    keep_dists = []

    with temp_seed(args.seed):
        for n_eval in range(args.num_eval_runs):
            obs_n = eval_env.reset()
            episode_reward = 0
            episode_step = 0
            episode_collisions = 0
            keep_d = []

            while True:
                obs_n = np.asarray(obs_n)
                action_n = agent.select_action(torch.Tensor(obs_n).to(device), action_noise=True,
                                               param_noise=False).squeeze().cpu().numpy()
                next_obs_n, reward_n, done_n, info_n = eval_env.step(action_n)
                col, nearest_dist = zip(*info_n)
                episode_reward += np.sum(reward_n)
                episode_collisions += sum(col) // 2
                keep_d.append(nearest_dist)

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
                    # keep_dists.append(np.mean(dis_bw_agents))
                    keep_dists.append(keep_d)
                    print(f'test reward: {episode_reward}, total collision: {episode_collisions}, '
                          f'completion rate: {comp_rates[-1]}')

                    if args.save_last_frame:
                        save_frame(eval_env)
                    break

    mean_reward = np.mean(eval_rewards)

    print("========================================================")
    print(f"reward: avg {mean_reward} std {np.std(eval_rewards)}")

    eval_env.close()
    eval_env.close_renderer()

    metric_record = MetricRecord(mean_reward, np.mean(comp_rates), np.mean(collisions),
                                 np.mean(total_dists), 0)
    return metric_record, np.asarray(keep_dists)
