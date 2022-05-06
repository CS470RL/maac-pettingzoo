import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer

from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.make_env import make_env
from algorithms.attention_sac import AttentionSAC

import csv
import matplotlib.pyplot as plt

from pettingzoo.magent import tiger_deer_v3
from gym.spaces.utils import flatten, flatdim

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

    env = tiger_deer_v3.parallel_env(
                                map_size=45, 
                                minimap_mode=False, 
                                tiger_step_recover=-0.1, 
                                deer_attacked=-0.1, 
                                max_cycles=500, 
                                extra_features=False
                                )

    model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)

    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [flatdim(obsp) for obsp in env.observation_spaces.values()],
                                 [acsp.n for acsp in env.action_spaces.values()])

    print('SETTING FINISHED!!!')
    
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor([val.flatten()]), requires_grad=False)
                         for val in obs.values()]

            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            
            # rearrange actions to be per environment
            actions = {}
            for i in range(config.n_rollout_threads):
                for name_i, ac in zip(obs.keys(), agent_actions):
                    actions[name_i] = np.where(ac[i] == 1)[0][0]

            next_obs, rewards, dones, infos = env.step(actions)

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads

            if (len(replay_buffer) >= config.batch_size and (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')

                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')

        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads
        )
 
        reward_sum = 0
        tiger_sum = 0
        deer_sum = 0

        for a_i, a_ep_rew in enumerate(ep_rews):
            rew_i = a_ep_rew * config.episode_length
            reward_sum += rew_i

            if a_i < 101:
                deer_sum += rew_i
            else:
                tiger_sum += rew_i

            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew * config.episode_length, ep_i)

        logger.add_scalar('tiger_rewards_sum', tiger_sum, ep_i)
        logger.add_scalar('deer_rewards_sum', deer_sum, ep_i)
        logger.add_scalar('reward_sum', reward_sum, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

        avg_rew = np.array(ep_rews) * config.episode_length
        print(f'avg_rew: {avg_rew}')

        # record reward in every episode
        # format : (1, 8) python list - [agent1 reward, agent2, ..., agent8]
        write.writerow(avg_rew) 

    f.close()

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


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
