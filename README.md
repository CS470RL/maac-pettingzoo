# Multi-Actor-Attention-Critic
Save average rewards for each agent per episode, and plot 'avg rewards per episode'
<br>

* Paper [*Actor-Attention-Critic for Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/1810.02912) (Iqbal and Sha, ICML 2019)
* Original code repo : https://github.com/shariqiqbal2810/MAAC
<br>
<br>

### How to run MAAC code & plot
python main.py fullobs_collect_treasure mytest1 --n_episodes 10000 --n_rollout_threads 1 --testnum 1
python plot.py --input test1.csv --which 0
<br>
- main.py `--testnum`
'testnum' option in MAAC code saves the rewards to 'test{testnum}.csv'
`plot.py` takes 'test{testnum}' or 'test{testnum}.csv' as an input with `--input` option.
<br>
- main.py `--n_rollout_threads 1`
🚨 currently only supports single process
<br>
- plot.py `--which` agent to plot
`plot.py` takes the agent number to plot with '--which' option.
0 to plot every agents, or 1~NUMOFAGENTS to plot individual agent 
(NUMOFAGENTS in MAAC paper is 8)
