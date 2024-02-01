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
    python inference.py --exp_name coop_navigation_n20 --scenario simple_spread_custom_n20  --critic_type gcn_max  --cuda --num_steps 100

### Real robots
    cd maddpg
    python inference_real.py --exp_name coop_navigation_n20 --scenario simple_spread_custom_n20  --critic_type gcn_max  --cuda --num_steps 100

## Acknowledgement
The MADDPG code is based on the DDPG implementation of https://github.com/ikostrikov/pytorch-ddpg-naf

The improved MPE code is based on the MPE implementation of https://github.com/openai/multiagent-particle-envs

The GCN code is based on the implementation of https://github.com/tkipf/gcn

The PIC code is based on the implementation of https://github.com/IouJenLiu/PIC

## License
PIC-Lang is licensed under the MIT License

