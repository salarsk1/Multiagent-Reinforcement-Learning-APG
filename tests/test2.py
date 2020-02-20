from apg import *
import numpy as np
import torch
import collections
from collections import namedtuple
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

#hyper parameters
batch_size = 64
n_episodes = 40_000
steps_per_episode = 5
window_size = 100


n_agents = 2
env = Env(0.01, n_agents)
cost = [0.05, 0.05]
agent_types = [0, 0]
n_types = len(set(cost))
type_dict = defaultdict()
for i in range(n_agents):
    type_dict[i] = agent_types[i]


cost_funcs = [CostFunction(i) for i in cost]


market_min = 2.0
market_worth = 2.0
percentage_market_min = .1
q_min = (1.0 + percentage_market_min) * market_min

value = RPIValueFunction(market_min, market_worth, 0.2, q_min)

#define the variables fo the reward
reward_win                  = 0.0
reward_lose                 = 0.0
principal_select_reward     = 0.0
principal_no_select_reward  = -1.0
winner_prime_penalty        = -2.0
agent_fail_penalty          = 0.0
principal_fail_penalty      = -2.0
principal_percentage        = 0.5

reward_process = IncentiveCostNotCoveredReward(value, reward_win, reward_lose, 
                                               principal_select_reward, 
                                               principal_no_select_reward, 
                                               winner_prime_penalty, 
                                               agent_fail_penalty, 
                                               principal_fail_penalty,
                                               principal_percentage)
#Define the RL agent types

agents = [A2CAgent(3, [100, 300, 100], 42,
                    learning_rate = 3.0e-5, gamma = .5) for _ in range(n_agents)]

principal = A2CAgent(2, [10, 10], 21, learning_rate=5e-4, gamma=1.0)

winner = [A2CAgent(2, [100, 300, 100], 40, learning_rate=1e-4, gamma=1.) for _ in range(n_types)]

# agents = [DQNAgent(2, [200, 100], 41,
#                     learning_rate = 0.00001, gamma = 0.95, min_eps = 0.01, 
#                     epsilon_decay = 0.9995, max_memory_size = 50_000) for _ in range(n_agents)]

# principal = DQNAgent(n_agents, [100, 200, 100], n_agents, learning_rate = 0.00001, 
#                      gamma = .99, min_eps = 0.01, epsilon_decay = 0.999, max_memory_size = 5_000)

# winner = DQNAgent(3, [100, 300, 200], 40, learning_rate = 0.0003, gamma = 0.95,\
#                 min_eps = 0.01, epsilon_decay = 0.999, max_memory_size = 10_000)

if __name__ == "__main__":

    run(env, n_episodes, steps_per_episode, n_agents, cost_funcs, type_dict,
        value, reward_process, agents, principal, winner, min_bid = 1.0, 
        max_bid = 3.0, write_output_every=5_000, 
        path_to_directory = "/Users/salarsk/developments/phd/nsf/rl_acq/code/")

