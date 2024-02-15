from dataclasses import dataclass

import numpy as np


# TODO: implement different metrics

@dataclass
class MetricRecord:
    eval_reward: float
    completion_rate: float
    collision_avg_count: float


def completion_rate(eval_env):
    world = eval_env.world
    landmarks = np.array([l.state.p_pos for l in world.landmarks])
    agents = np.array([a.state.p_pos for a in world.agents])
    agent_diameter = world.agents[0].size * 2

    cover_cnt = 0
    # simple (n^2) solution
    for landmark in landmarks:
        distances = np.sqrt(np.square(landmark - agents).sum(axis=1))
        covered = distances.min() < agent_diameter
        if covered:
            cover_cnt += 1

    return cover_cnt / len(landmarks)
