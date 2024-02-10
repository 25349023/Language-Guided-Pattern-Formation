from dataclasses import dataclass


# TODO: implement different metrics

@dataclass
class MetricRecord:
    eval_reward: float
    completion_rate: float
    collision_avg_count: float


def completion_rate():
    # TODO: implement this metric
    pass
