import argparse
import contextlib
import csv
import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn

seaborn.set_theme()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, default="ckpt_plot",
                        help="directory from which the reward info should be load")
    parser.add_argument("--exp_names", type=str, nargs='+', help="name of the experiment")
    parser.add_argument("--save", action='store_true', default=False, help="save the chart")
    return parser.parse_args()


def plot_experiments(load_dir, exp_names, save):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for exp_name in exp_names:
        plot_one_exp(ax, load_dir, exp_name)

    ax.legend(loc='best')
    if save:
        os.makedirs('reward_plots', exist_ok=True)
        fig.savefig(f'reward_plots/plot_{datetime.now():%y%m%d-%H%M%S}.pdf')
    plt.show()


def plot_one_exp(ax, load_dir, exp_name):
    reward_series = []
    steps = []

    for dirname in glob.iglob(os.path.join(load_dir, exp_name, '*seed*')):
        print(dirname)
        try:
            trlog_file = open(os.path.join(dirname, 'train_curve.csv'), newline='')
        except FileNotFoundError:
            continue

        with contextlib.closing(trlog_file):
            reader = csv.reader(trlog_file)
            for row in reader:
                if row[0] == 'rewards':
                    reward_series.append([float(r) for r in row[1:]])
                elif row[0] == 'steps':
                    steps = [int(s) for s in row[1:]]

    reward_series = np.asarray(reward_series)
    mean = reward_series.mean(axis=0)
    std = reward_series.std(axis=0)
    steps = np.asarray(steps)

    ax.plot(steps, mean, label=exp_name)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2)


if __name__ == '__main__':
    args = get_args()
    plot_experiments(args.load_dir, args.exp_names, args.save)
    # plot_experiments('ckpt_plot', ['normal_n10'])
