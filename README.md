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

## Model Evaluation

### Simulation
    cd maddpg
    python inference.py --save_dir <ckpt_dir> --exp_name coop_navigation_n20 --scenario simple_spread_custom_n20 --critic_type gcn_max --cuda --num_steps 100

### Real robots
    cd maddpg
    export PYTHONPATH=<path to maddpg directory>
    python real_deploy/inference_real.py --save_dir <ckpt_dir> --exp_name <exp_name> --scenario test --critic_type gcn_max --cuda --num_steps 100 --num_agents 10 --shape <shape>

- `<ckpt_dir>` should be `exp_ckpt` if you are using the checkpoints provided by this repo
- `<exp_name>` can be `coop_navigation_dire_n10`, `coop_navigation_vel0_n10`
- The available options for `<shape>` can be found in `maddpg/real_deploy/tools.py`

The differences between these setups are: (see TODO 2-2 in eval_real.py)
- `coop_navigation_dire_n10` use the `get_direction(curr_agent.velocity)` to compute the first two element of the observation
- `coop_navigation_vel0_n10` use the `np.array([0.0, 0.0])` as the first two element of the observation

## Acknowledgement
The MADDPG code is based on the DDPG implementation of https://github.com/ikostrikov/pytorch-ddpg-naf

The improved MPE code is based on the MPE implementation of https://github.com/openai/multiagent-particle-envs

The GCN code is based on the implementation of https://github.com/tkipf/gcn

The PIC code is based on the implementation of https://github.com/IouJenLiu/PIC

## License
PIC-Lang is licensed under the MIT License

