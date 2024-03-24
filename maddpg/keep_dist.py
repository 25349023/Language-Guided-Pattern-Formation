import argparse
import glob
import os

import numpy as np
import seaborn

seaborn.set_theme()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, default="ckpt_plot",
                        help="directory from which the reward info should be load")
    parser.add_argument("--exp_names", type=str, nargs='+', help="name of the experiment")
    parser.add_argument("-q", type=float, default=0.25, help="The quantile value")
    return parser.parse_args()


def all_kept_distance(load_dir, exp_names, q=0.25):
    for exp_name in exp_names:
        kd = kept_distances(load_dir, exp_name, q)
        print(f'{exp_name}: {kd}')


def kept_distances(load_dir, exp_name, q=0.25):
    kept_dists = []

    for dirname in glob.iglob(os.path.join(load_dir, exp_name, '*seed*')):
        kdist = np.load(os.path.join(dirname, 'keep_dists.npy'))  # [episodes x episode_len x agents]
        quantile_per_agent = np.quantile(kdist, q, axis=1)  # [episodes x agents]
        median_of_qt_per_episode = np.quantile(quantile_per_agent, 0.5, axis=1)  # [episodes]
        # mean_of_qt_per_episode = quantile_per_agent.min(axis=1)  # [episodes]
        kept_dists.append(np.median(median_of_qt_per_episode))

    return np.mean(kept_dists)


if __name__ == '__main__':
    args = get_args()
    all_kept_distance(args.load_dir, args.exp_names, args.q)
