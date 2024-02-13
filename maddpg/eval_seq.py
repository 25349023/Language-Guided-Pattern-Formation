import contextlib
import time

import numpy as np
import torch

from metrics import MetricRecord, completion_rate
from utils import make_env


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def eval_model_seq(args, agent):
    eval_env = make_env(args.scenario, args, benchmark=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    print('=================== start eval ===================')
    eval_env.seed(args.seed + 10)
    eval_rewards = []
    comp_rates = []
    collisions = []
    with temp_seed(args.seed):
        for n_eval in range(args.num_eval_runs):
            obs_n = eval_env.reset()
            episode_reward = 0
            episode_step = 0
            episode_collisions = 0

            while True:
                action_n = agent.select_action(torch.Tensor(obs_n).to(device), action_noise=True,
                                               param_noise=False).squeeze().cpu().numpy()
                next_obs_n, reward_n, done_n, info_n = eval_env.step(action_n)
                episode_reward += np.sum(reward_n)
                episode_collisions += sum(info_n) // 2

                episode_step += 1
                terminal = (episode_step >= args.num_steps)
                obs_n = next_obs_n

                time.sleep(0.02)
                eval_env.render()

                if done_n[0] or terminal:
                    eval_rewards.append(episode_reward)
                    print(f'test reward: {episode_reward}, total collision: {episode_collisions}')
                    comp_rates.append(completion_rate(eval_env))
                    collisions.append(episode_collisions)
                    break

    mean_reward = np.mean(eval_rewards)

    print("========================================================")
    print(f"reward: avg {mean_reward} std {np.std(eval_rewards)}")

    eval_env.close()
    eval_env.close_renderer()

    metric_record = MetricRecord(mean_reward, np.mean(comp_rates), np.mean(collisions))
    return metric_record
