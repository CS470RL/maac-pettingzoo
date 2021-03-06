<<<<<<< HEAD
# Multi-Actor-Attention-Critic
Save average rewards for each agent per episode, and plot 'avg rewards per episode'
<br>

* Paper [*Actor-Attention-Critic for Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/1810.02912) (Iqbal and Sha, ICML 2019)
* Original code repo : https://github.com/shariqiqbal2810/MAAC
<br>
<br>

## How to run MAAC code & plot
`python main.py fullobs_collect_treasure mytest1 --n_episodes 10000 --n_rollout_threads 1 --testnum 1`  

`python plot.py --input test1.csv --which 0`
<br>
- main.py `--testnum`
'testnum' option in main.py MAAC code saves the rewards to `test{testnum}.csv`  
plot.py takes 'test{testnum}' or 'test{testnum}.csv' as an input using `--input` option.

- main.py `--n_rollout_threads 1`  
🚨 currently only supports single process

- plot.py `--which` agent to plot  
plot.py takes the agent number to plot with `--which` option.  
0 to plot every agents, or 1~NUMOFAGENTS to plot individual agent 
(NUMOFAGENTS in MAAC paper is 8)

## Experiment Results
* env: fullobs_collect_treasure
* n_episodes: 10,000
* n_rollout_threads: 10

#### Training statistics for a single agnet (agent0)
![agent0](./imgs/agent0.png)

#### Total sum of rewards in each episode
![reward_sum](./imgs/reward_sum.png)
=======
# maac-pettingzoo
Applying the MAAC algorithm onto Pettingzoo Environments 
>>>>>>> 06f27286cc120afd90dcf43d223ab0a9b844643f
