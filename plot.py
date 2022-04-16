import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse

NUMOFAGENTS = 8 # Refer to the paper

def plot(plotWhich):
    PLOTWHICH = args.which
    FILENAME = args.input.strip('.csv')

    rewardPerAgent = [[] for _ in range(NUMOFAGENTS)]

    f = open(f'{FILENAME}.csv')
    reader = csv.reader(f, delimiter='\t')
    for episode, line in enumerate(reader):
        epi_reward = list(map(float, line))
        for n, indiv_reward in enumerate(epi_reward):
            rewardPerAgent[n].append(indiv_reward)
    else:
        TOTAL_EPISODES = episode + 1

    res = np.array(rewardPerAgent)
    # res = np.array([np.array(row) for row in rewardPerAgent])
    # print(res.shape, res[1], res[1].shape)

    # Plot all agents
    if PLOTWHICH == 0:   
        for n in range(1, NUMOFAGENTS + 1):
            plt.scatter(np.arange(0, TOTAL_EPISODES), rewardPerAgent[n-1], label=f"agent{n}", s=1)

    # Plot single agent
    elif 1 <= PLOTWHICH <= NUMOFAGENTS:
            plt.scatter(np.arange(0, TOTAL_EPISODES), rewardPerAgent[PLOTWHICH], label=f"agent{PLOTWHICH}", s=1)
    else:
        raise Exception('Invalid agent number')

    plt.xlabel('episode')
    plt.ylabel('average rewards per agent')
    plt.legend()
    plt.savefig(f'{FILENAME}.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="(required) tab delimited csv file containing (episodes x NUMOFAGENTS) rewards")
    parser.add_argument("--which", default=0, type=int, help="Select which agent to plot : 0 to plot all agents, 1~8 to plot one individual agent")
    args = parser.parse_args()
    plot(args)
