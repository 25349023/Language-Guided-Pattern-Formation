from dataclasses import dataclass

import numpy as np


# TODO: implement different metrics

@dataclass
class MetricRecord:
    eval_reward: float
    completion_rate: float
    avg_collision_count: float
    remaining_dist: float
    keeping_dist: float


def completion_rate(eval_env):
    world = eval_env.world
    landmarks = np.array([l.state.p_pos for l in world.landmarks])
    agents = np.array([a.state.p_pos for a in world.agents])
    agent_radius = world.agents[0].size

    cover_cnt = 0
    # simple (n^2) solution
    for landmark in landmarks:
        distances = np.sqrt(np.square(landmark - agents).sum(axis=1))
        covered = distances.min() < agent_radius
        if covered:
            cover_cnt += 1

    return cover_cnt / len(landmarks)


def distance_to_landmark(eval_env):
    world = eval_env.world
    landmarks = np.array([l.state.p_pos for l in world.landmarks])
    agents = np.array([a.state.p_pos for a in world.agents])

    total_dist = 0
    for agent in agents:
        distances = np.sqrt(np.square(agent - landmarks).sum(axis=1))
        total_dist += distances.min()

    return total_dist
