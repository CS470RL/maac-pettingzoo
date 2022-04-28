import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer
import csv
import matplotlib.pyplot as plt

from algorithms.attention_sac import AttentionSAC
from pettingzoo.magent import tiger_deer_v3

def run(config):
    f = open(f"test{config.testnum}.csv", 'w')
    write = csv.writer(f, delimiter='\t')

    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1

    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(run_num)
    np.random.seed(run_num)

    # make a parallel Pettingzoo environment
    env = tiger_deer_v3.parallel_env(
                                map_size=45, 
                                minimap_mode=False, 
                                tiger_step_recover=-0.1, 
                                deer_attacked=-0.1, 
                                max_cycles=500, 
                                extra_features=False
                                )

    model = AttentionSAC.init_from_env(
                                    env,
                                    tau=config.tau,
                                    pi_lr=config.pi_lr,
                                    q_lr=config.q_lr,
                                    gamma=config.gamma,
                                    pol_hidden_dim=config.pol_hidden_dim,
                                    critic_hidden_dim=config.critic_hidden_dim,
                                    attend_heads=config.attend_heads,
                                    reward_scale=config.reward_scale
                                    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
                             
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--testnum", default=1, type=int)

    config = parser.parse_args()

    run(config)



