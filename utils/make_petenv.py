"""
Code for making a PettingZoo magent envrionment (e.g., Tiger-Deer)
For more info, see https://www.pettingzoo.ml/magent.

Added by Yuseung Lee (https://github.com/phillipinseoul)
"""
from gym.spaces.utils import flatten, flatdim
from pettingzoo.magent import tiger_deer_v3

# env = tiger_deer_v3.env(
#                     map_size=45, 
#                     minimap_mode=False, 
#                     tiger_step_recover=-0.1, 
#                     deer_attacked=-0.1, 
#                     max_cycles=500, 
#                     extra_features=False
#                     )

env = tiger_deer_v3.parallel_env(
                    map_size=45, 
                    minimap_mode=False, 
                    tiger_step_recover=-0.1, 
                    deer_attacked=-0.1, 
                    max_cycles=500, 
                    extra_features=False
                    )

# x = flatdim(env.observation_spaces['tiger_0'])
# y = flatten(env.observation_spaces['tiger_0'])
# print(x)
# print(y)

print('env.action_spaces[tiger_0]')
print(env.action_spaces['tiger_0'])
print('env.action_spaces[tiger_0].n')
print(env.action_spaces['tiger_0'].n)
print('env.action_spaces[tiger_0].shape')
print(env.action_spaces['tiger_0'].shape)

# print('env.observation_spaces[tiger_0]')
# print(env.observation_spaces['tiger_0'])
# print('env.observation_spaces[tiger_0]')
# print(env.observation_spaces['tiger_0'])
# print('env.observation_spaces[tiger_0].shape')
# print(env.observation_spaces['tiger_0'].shape)




# for acsp in env.action_spaces:
    # print(acsp)
# print(env.action_space.n)
# print(env.observation_spaces['tiger_19'])
# print(env.observation_spaces.shape)
# print(env.observation_space.shape)
# print(env.agents)
# print(env)
# print(tg_env)
# tg_env.seed(100)
# print(tg_env.step)

# env.reset()

# for agent in env.agent_iter():
#     observation, reward, done, info = env.last()
#     action = policy(observation, agent)
#     env.step(action)



# Make the Tiger-Deer environment from PettingZoo
def make_td_env():
    from pettingzoo.magent import tiger_deer_v3







