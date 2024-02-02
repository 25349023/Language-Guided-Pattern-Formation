# PIC-Lang

## Installation & Setup

### Platform and Dependencies: 
* Ubuntu 20.04 
* Python 3.7
* Pytorch 1.12.1
* OpenAI gym 0.26.2 (https://github.com/openai/gym)

### Install the improved MPE:
    cd multiagent-particle-envs
    pip install -e .

### Install other required packages
    pip install -r requirements.txt

## Training

### Simulation
    cd maddpg
    python inference.py --exp_name coop_navigation_n20 --scenario simple_spread_custom_n20 --critic_type gcn_max --cuda --num_steps 100

### Real robots
    cd maddpg
    python inference_real.py --exp_name <exp_name> --critic_type gcn_max --cuda --num_steps 100 --num_agents 20

, where exp_name can be `coop_navigation_dire_n20`, `coop_navigation_dire_n20_thres0p1`, `coop_navigation_vel0_n20`, or
`coop_navigation_novel_n20`.

The differences between these setups are: (see TODO 2-2 in eval_real.py)
- `coop_navigation_dire_n20` use the `get_direction(curr_agent.velocity)` to compute the first two element of the observation
- `coop_navigation_dire_n20_thres0p1` use the `get_direction(curr_agent.velocity, eps=0.1)` to compute the first two element of the observation
- `coop_navigation_vel0_n20` use the `np.array([0.0, 0.0])` as the first two element of the observation
- `coop_navigation_novel_n20` doesn't include the direction/velocity info in the agent's observation

## Acknowledgement
The MADDPG code is based on the DDPG implementation of https://github.com/ikostrikov/pytorch-ddpg-naf

The improved MPE code is based on the MPE implementation of https://github.com/openai/multiagent-particle-envs

The GCN code is based on the implementation of https://github.com/tkipf/gcn

The PIC code is based on the implementation of https://github.com/IouJenLiu/PIC

## License
PIC-Lang is licensed under the MIT License

