import argparse
import csv
import inspect


def adjust_learning_rate(optimizer, steps, max_steps, start_decrease_step, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if steps > start_decrease_step:
        lr = init_lr * (1 - ((steps - start_decrease_step) / (max_steps - start_decrease_step)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def dict2csv(output_dict, f_name):
    with open(f_name, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=",")
        for k, v in output_dict.items():
            v = [k] + v
            writer.writerow(v)


def n_actions(action_spaces):
    """
    :param action_space: list
    :return: n_action: list
    """
    n_actions = []
    from gym import spaces
    from multiagent.environment import MultiDiscrete
    for action_space in action_spaces:
        if isinstance(action_space, spaces.discrete.Discrete):
            n_actions.append(action_space.n)
        elif isinstance(action_space, MultiDiscrete):
            total_n_action = 0
            one_agent_n_action = 0
            for h, l in zip(action_space.high, action_space.low):
                total_n_action += int(h - l + 1)
                one_agent_n_action += int(h - l + 1)
            n_actions.append(one_agent_n_action)
        else:
            raise NotImplementedError
    return n_actions


def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    sig = inspect.signature(scenario.make_world)
    if 'args' in sig.parameters:
        world = scenario.make_world(args=arglist)
    else:
        world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius)
    return env


def make_env_vec(scenario_name, arglist, benchmark=False):
    from multiagent.environment_vec import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius)
    return env


def copy_actor_policy(s_agent, t_agent):
    if hasattr(s_agent, 'actors'):
        for i in range(s_agent.n_group):
            state_dict = s_agent.actors[i].state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            t_agent.actors[i].load_state_dict(state_dict)
        t_agent.actors_params, t_agent.critic_params = None, None
    else:
        state_dict = s_agent.actor.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        t_agent.actor.load_state_dict(state_dict)
        t_agent.actor_params, t_agent.critic_params = None, None


def get_args():
    parser = argparse.ArgumentParser(description='PIC MARL Algorithm')
    # env settings
    parser.add_argument('--scenario', required=True, help='name of the environment to run')
    parser.add_argument("--num_agents", type=int, default=10)
    parser.add_argument("--agent_rad", type=float, default=0.15)
    parser.add_argument("--world_rad", type=float, default=3.0)
    parser.add_argument("--n_others", type=int, default=5)

    # Algorithm parameters
    parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
                        help='discount factor for reward (default: 0.95)')
    parser.add_argument('--tau', type=float, default=0.01, metavar='G',
                        help='soft update factor for target network (default: 0.01)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='batch size (default: 1024)')
    parser.add_argument('--num_steps', type=int, default=25, metavar='N',
                        help='max episode length (default: 25)')
    parser.add_argument('--num_episodes', type=int, default=60000, metavar='N',
                        help='number of episodes (default: 60000)')
    parser.add_argument('--param_noise', type=bool, default=False)
    parser.add_argument('--train_noise', default=False, action='store_true')
    parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                        help='initial noise scale (default: 0.3)')
    parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                        help='final noise scale (default: 0.3)')
    parser.add_argument('--exploration_end', type=int, default=60000, metavar='N',
                        help='number of episodes with noise (default: 60000)')

    # Optimization parameters
    parser.add_argument('--actor_lr', type=float, default=1e-2, help='(default: 1e-2)')
    parser.add_argument('--critic_lr', type=float, default=1e-2, help='(default: 1e-2)')
    parser.add_argument('--fixed_lr', default=False, action='store_true')
    parser.add_argument('--updates_per_step', type=int, default=8, metavar='N',
                        help='# of actor updates each time (default: 8)')
    parser.add_argument('--critic_updates_per_step', type=int, default=8, metavar='N',
                        help='# of critic updates each time (default: 8)')
    parser.add_argument('--steps_per_actor_update', type=int, default=100)
    parser.add_argument('--steps_per_critic_update', type=int, default=100)
    parser.add_argument('--shuffle', default=None, type=str, help='None|shuffle|sort')
    parser.add_argument('--target_update_mode', default='soft', help='soft | hard | episodic')

    # Network parameters
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                        help='hidden size (default: 128)')
    parser.add_argument('--critic_type', type=str, default='mlp', help="Supports [mlp, gcn_mean, gcn_max]")
    parser.add_argument('--actor_type', type=str, default='mlp', help="Supports [mlp, gcn_max]")
    parser.add_argument('--critic_dec_cen', default='cen')
    parser.add_argument("--activation", type=str, default='relu', help='relu|leaky_relu')
    parser.add_argument('--cuda', default=False, action='store_true')

    # Evaluation parameters
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--save_last_frame", action="store_true", default=False)
    parser.add_argument('--num_eval_runs', type=int, default=20,
                        help='number of runs per evaluation (default: 20)')
    parser.add_argument("--shape", type=str, default='circle',
                        help='refer to real_deploy/tools.py for avalible shapes')
    parser.add_argument("--retain_pos", action="store_true", default=False,
                        help="retain agents' positions between episodes")
    parser.add_argument("--ckpt_type", type=str, default="best", help='best | last')

    # Experiment parameters
    parser.add_argument('--seed', type=int, default=9, metavar='N',
                        help='random seed (default: 9)')
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--exp_name", type=str, help="name of the experiment")
    parser.add_argument("--force", action="store_true", default=False, help='overwrite the previous ckpt')
    parser.add_argument("--save_dir", type=str, default="ckpt_plot",
                        help="directory in which training state and model should be saved")

    return parser.parse_args()
