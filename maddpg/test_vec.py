import os
import random
import time
from multiprocessing import Queue
from multiprocessing.sharedctypes import Value

import numpy as np
import torch
import torch.multiprocessing as mp

from eval import eval_model_q
from utils import *

args = get_args()

if args.exp_name is None:
    args.exp_name = args.scenario + '_' + args.critic_type + '_' + args.target_update_mode + '_hiddensize' \
                    + str(args.hidden_size) + '_' + str(args.seed)
print("=================Arguments==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

torch.set_num_threads(1)
device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

# env = make_env(args.scenario, None)
# n_agents = env.n
# env.seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
num_adversary = 0

rewards = []
total_numsteps = 0
updates = 0
exp_save_dir = os.path.join(args.save_dir, args.exp_name)
os.makedirs(exp_save_dir, exist_ok=True)
best_eval_reward, best_good_eval_reward, best_adversary_eval_reward = -1000000000, -1000000000, -1000000000
start_time = time.time()
eval_agent = torch.load(os.path.join(exp_save_dir, 'agents_best.ckpt'))['agents']

test_q = Queue()
done_training = Value('i', False)
p = mp.Process(target=eval_model_q, args=(test_q, done_training, args, False))
p.start()

tr_log = {'num_adversary': 0,
          'best_good_eval_reward': 0,
          'best_adversary_eval_reward': 0,
          'exp_save_dir': exp_save_dir, 'total_numsteps': total_numsteps,
          'value_loss': 0, 'policy_loss': 0,
          'i_episode': 0, 'start_time': start_time}
test_q.put([eval_agent, tr_log])

# env.close()
time.sleep(5)
done_training.value = True
