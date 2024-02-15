import os
import contextlib
import time

import numpy as np
import torch

import matplotlib.pyplot as plt
from datetime import datetime

from tools import sim2real_coord, real2sim_coord, get_landmarks
from osxDriver import osx001Driver


class Agent:
    def __init__(self, position, velocity, robot_id):
        self.position = position
        self.velocity = velocity
        self.robot_id = robot_id

def get_agents(robot_id_list, robot_pos_dict, pos_prev_dict):
    agents = []
    for robot_id in robot_id_list:
        current_pos = np.array(real2sim_coord(*robot_pos_dict[robot_id]))
        prev_pos = np.array(real2sim_coord(*pos_prev_dict[robot_id]))
        agents.append(Agent(current_pos, current_pos - prev_pos, robot_id))
    return agents

def is_goal_close_to_landmarks(goal, landmarks, threshold=25.0):
    for landmark in landmarks:
        distance = np.sqrt((goal[0] - landmark[0])**2 + (goal[1] - landmark[1])**2)
        if distance <= threshold:
            print(f'distance: {distance}')
            return True
    return False

def get_robot_state(robot_id_list, driver1, driver2):
    # receive the position of the robots
    robot_pos_dict = {}
    robot_degree_dict = {}
    while True:
        # Get the initial position of the robots from both drivers
        robot_degree_driver1, robot_pos_driver1 = driver1.get_id_position()
        robot_degree_driver2, robot_pos_driver2 = driver2.get_id_position()
        # Update the robot_pos_dict with the new positions
        robot_pos_dict.update(robot_pos_driver1)
        robot_pos_dict.update(robot_pos_driver2)
        robot_degree_dict.update(robot_degree_driver1)
        robot_degree_dict.update(robot_degree_driver2)
        # Check if we have received positions for all robot IDs
        if all(robot_id in robot_pos_dict for robot_id in robot_id_list):
            print("Received positions for all robots. Moving to the next part of the code.")
            break
    return robot_degree_dict, robot_pos_dict

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
        landmark_pos.append(landmark - curr_agent.position)

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


def eval_model_real(args, policy, update_interval=1):
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
        save_dir = f'real_deploy/data/{datetime.now().strftime("%Y%m%d%H%M%S")}'
        save_obs_dir = f'{save_dir}/obs'
        os.makedirs(save_obs_dir, exist_ok=True)
        save_goal_dir = f'{save_dir}/goals'
        os.makedirs(save_goal_dir, exist_ok=True)

        # set robot id
        # robot_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        robot_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        pos_prev_dict = {i: None for i in robot_id_list}

        # open port
        driver1 = osx001Driver('COM3')
        driver2 = osx001Driver('COM4')
        driver1.flush_buffer()
        driver2.flush_buffer()
        
        # driver2.writeMotorSpeed(15, 80, 80)
        # import pdb;pdb.set_trace()
        

        # set parameter
        delta_end = 1
        for robot_id in robot_id_list:
            if robot_id <= 9:
                driver1.set_parameter(robot_id, delta_end=delta_end)
            else:
                driver2.set_parameter(robot_id, delta_end=delta_end)

        n_agents = args.num_agents  # default is 10 agents
        landmarks = get_landmarks(n_agents, args.shape)
        
        # visualize the landmarks
        real_landmarks = np.array([sim2real_coord(x, y) for x, y in landmarks])
        # Plotting
        plt.figure(figsize=(12, 6))
        # Plot for landmarks
        plt.subplot(1, 2, 1)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], color='blue', label='Landmarks')
        for i, robot_id in enumerate(robot_id_list):
            plt.annotate(str(robot_id), (landmarks[i, 0], landmarks[i, 1]))
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.title('Landmarks')
        plt.xlabel('x')
        plt.ylabel('y')
        # Plot for transferred positions
        plt.subplot(1, 2, 2)
        plt.scatter(real_landmarks[:, 0], real_landmarks[:, 1], color='red', label='Transferred Positions')
        for i, robot_id in enumerate(robot_id_list):
            plt.annotate(str(robot_id), (real_landmarks[i, 0], real_landmarks[i, 1]))
        plt.xlim(-255, 255)
        plt.ylim(-255, 255)
        plt.title('Transferred Positions')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        # Display the plots
        plt.tight_layout()
        plt.savefig(f'{save_dir}/landmarks.png')

        print('=================== start eval ===================')
        with temp_seed(args.seed):
            for n_eval in range(args.num_eval_runs):
                episode_step = 0
                while True:
                    # receive the position of the robots
                    robot_degree_dict, robot_pos_dict = get_robot_state(robot_id_list, driver1, driver2)

                    # set previous position to the initial position
                    # agents = get_agents(robot_id_list, robot_pos_dict, robot_pos_dict)
                    if episode_step == 0:
                        agents = get_agents(robot_id_list, robot_pos_dict, robot_pos_dict)
                    else:
                        agents = get_agents(robot_id_list, robot_pos_dict, pos_prev_dict)
                    
                    obs_n = all_agent_observation(agents, landmarks)
                    np.save(f'{save_obs_dir}/obs_{episode_step}.npy', np.array(obs_n))
                    action_n = policy.select_action(
                        torch.Tensor(obs_n).to(device), action_noise=True,
                        param_noise=False).squeeze().cpu().numpy()

                    goal_pos = []
                    for agent, act in zip(agents, action_n):
                        idle, right, left, up, down = act
                        dx = (right - left) * 0.05
                        dy = (up - down) * 0.05
                        delta = sim2real_coord(*(dx, dy))
                        delta_clipped = tuple(max(min(d, 1.5), -1.5) for d in delta)
                        current_pos = sim2real_coord(*agent.position)
                        print(f'Robot {agent.robot_id} - delta: {delta}')
                        print(f'Robot {agent.robot_id} - delta_clipped: {delta_clipped}')
                        print(f'Robot {agent.robot_id} - current_pos: {current_pos}')
                        goal_pos.append(tuple(c + d for c, d in zip(current_pos, delta_clipped)))

                    # visualize goal pos
                    x_coords, y_coords = zip(*goal_pos)
                    plt.figure(figsize=(10, 6))
                    plt.scatter(x_coords, y_coords, color='blue', marker='o')
                    for robot_id, goal in zip(robot_id_list, goal_pos):
                        plt.annotate(robot_id, goal, textcoords="offset points", xytext=(0,10), ha='center')
                    plt.title('Goal Positions')
                    plt.xlabel('X Coordinate')
                    plt.ylabel('Y Coordinate')
                    plt.xlim(-255, 255)
                    plt.ylim(-255, 255)
                    plt.grid(True)
                    plt.savefig(f'{save_goal_dir}/goal_positions_{episode_step}.png')

                    if episode_step % update_interval == 0:
                        for robot_id, goal in zip(robot_id_list, goal_pos):
                            if not is_goal_close_to_landmarks(goal, real_landmarks):
                                if robot_id <= 9:
                                    # driver1.resetIMU(robot_id)
                                    driver1.setTargetPosition(robot_id, int(goal[0]), int(goal[1]))
                                else:
                                    # driver2.resetIMU(robot_id)
                                    driver2.setTargetPosition(robot_id, int(goal[0]), int(goal[1]))
                        move_start_time = time.time()
                        while(True):
                            driver1.checkStatusPacket()
                            driver2.checkStatusPacket()
                            if time.time() - move_start_time > 0.1:
                                driver1.flush_buffer()
                                driver2.flush_buffer()
                                break
                    
                    # update robot prev position
                    for robot_id in robot_id_list:
                        pos_prev_dict[robot_id] = robot_pos_dict[robot_id]

                    episode_step += 1
                    terminal = (episode_step >= args.num_steps)

                    # time.sleep(0.3)

                    if terminal:
                        break
    except Exception as e:
        print(e)
    finally:
        driver1.flush_buffer()
        driver2.flush_buffer()

    # print("========================================================")
