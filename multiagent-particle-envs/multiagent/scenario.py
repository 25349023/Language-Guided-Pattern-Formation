import numpy as np

# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()


class CollisionBenchmarkMixin:
    def benchmark_data(self, agent, world):
        all_agents = np.array([a.state.p_pos for a in world.agents])
        agent_diameter = self.agent_size * 2

        distances = np.sqrt(np.square(agent.state.p_pos - all_agents).sum(axis=1))
        collide_cnt = (distances < agent_diameter).sum() - 1

        return collide_cnt
