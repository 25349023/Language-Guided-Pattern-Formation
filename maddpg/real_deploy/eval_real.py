import contextlib
import time

import numpy as np
import torch

from tools import sim2real_coord, get_landmarks


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_direction(velocity, eps=1e-6):
    def quantize(v):
        if abs(v) < eps:
            return 0.0
        return 1.0 if v > 0 else -1.0

    direction = np.array([quantize(v) for v in velocity])
    return direction


def single_agent_observation(agents, i, landmarks):
    """
    :param agents: list of all agents
    :param i: index of target agent
    :param landmarks: list of all landmarks

    :return: obs: [26] np array
        [0-1] self_agent direction (sign of velocity)
        [2-3] self_agent location
        [4-15] landmarks location
        [16-25] other agents' relative location
    """

    def distance(positions):
        return np.sqrt(np.sum(np.square(np.array(positions) - curr_agent.position), axis=1))

    # TODO 2-1: Construct agent observations, note the position should be in simulator coord
    #           You can use utility functions `real2sim_coord` and `sim2real_coord`
    curr_agent = agents[i]
    n_others = 5

    # positions of all landmarks relative to the curr_agent itself
    landmark_pos = []
    for landmark in landmarks:
        landmark_pos.append(landmark.position - curr_agent.position)

    # positions of other agents relative to the curr_agent itself
    other_pos = []
    for other in agents:
        if other is curr_agent:
            continue
        other_pos.append(other.position - curr_agent.position)

    # select k+1 nearest landmark positions as observation
    landmark_dist = distance(landmark_pos)
    landmark_dist_idx = np.argsort(landmark_dist)
    landmark_pos = [landmark_pos[i] for i in landmark_dist_idx[:n_others + 1]]

    # select k nearest agent positions as observation
    other_dist = distance(other_pos)
    dist_idx = np.argsort(other_dist)
    other_pos = [other_pos[i] for i in dist_idx[:n_others]]

    # TODO 2-2: depends on different checkpoints, the [0-1] observation will differ
    obs = np.concatenate([get_direction(curr_agent.velocity, eps=0.1)] +
                         [curr_agent.position] + landmark_pos + other_pos)
    # obs = np.concatenate([np.array([0.0, 0.0])] +
    #                      [curr_agent.position] + landmark_pos + other_pos)
    return obs


def all_agent_observation(agents, landmarks):
    """ Observations should be in simulator coordinates """
    return [single_agent_observation(agents, i, landmarks)
            for i in range(len(agents))]


def eval_model_real(args, agent):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    # TODO 1: Build / Initialize real robot agents
    agents = ...
    n_agents = args.num_agents  # default is 10 agents
    landmarks = get_landmarks(n_agents)

    print('=================== start eval ===================')
    with temp_seed(args.seed):
        for n_eval in range(args.num_eval_runs):

            # TODO 2-3: Check this function call can give the proper observation
            obs_n = all_agent_observation(agents, landmarks)
            episode_step = 0
            while True:
                action_n = agent.select_action(
                    torch.Tensor(obs_n).to(device), action_noise=True,
                    param_noise=False).squeeze().cpu().numpy()

                # TODO 5: tweak the goal position calculation,
                #         since in the simulator it is computed based on force and velocity
                goal_pos = []
                for agent, act in zip(agents, action_n):
                    idle, right, left, up, down = act
                    dx = (right - left) * 0.1
                    dy = (up - down) * 0.1
                    goal = (agent.position[0] + dx, agent.position[1] + dy)
                    goal_pos.append(sim2real_coord(*goal))

                # TODO 3: Send the goal positions to real robots and then update agents information
                # do something here...

                episode_step += 1
                terminal = (episode_step >= args.num_steps)

                # TODO 4: get the next observation
                obs_n = all_agent_observation(agents, landmarks)

                time.sleep(0.1)
                # eval_env.render()

                if terminal:
                    break

    print("========================================================")
