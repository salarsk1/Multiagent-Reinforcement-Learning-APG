import os
from apg import *
import numpy as np
import torch
import collections
from collections import namedtuple
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

path_to_directory = os.getcwd() + "/"
#hyper parameters
batch_size = 64
n_episodes = 80_000
steps_per_episode = 5
window_size = 100

n_agents = 2
env = Env(0.01, n_agents)
cost = [0.15, 0.15]
agent_types = [0, 0]
n_types = len(set(cost))
type_dict = defaultdict()
for i in range(n_agents):
    type_dict[i] = agent_types[i]


cost_funcs = [CostFunction(i) for i in cost]


market_min = 2.0
market_worth = 10.0
percentage_market_min = 0.0
q_min = (1.0 + percentage_market_min) * market_min

value = RBValueFunction(market_min, market_worth, q_min)

#define the variables fo the reward
reward_win                  = 0.0
reward_lose                 = 0.0
principal_select_reward     = 0.0
principal_no_select_reward  = -1.0
winner_prime_penalty        = -3.0 * market_worth
agent_fail_penalty          = -3.0 * market_worth
principal_fail_penalty      = -market_worth
prepayment_percentage       = .2
cost_premium                = 2.0

reward_process = CostNotCoveredReward(value, reward_win, reward_lose, 
                                      principal_select_reward, 
                                      principal_no_select_reward, 
                                      winner_prime_penalty, 
                                      agent_fail_penalty, 
                                      principal_fail_penalty,
                                      prepayment_percentage,
                                      cost_premium)
#Define the RL agent types

agents = [A2CAgent(2, [300, 300, 300, 200], 22,
                    learning_rate = 5.0e-6, gamma = 1.0) for _ in range(n_agents)]

principal = A2CAgent(2, [10, 10], 21, learning_rate = 5e-4, gamma = 1.0)

winner = [A2CAgent(3, [100, 100, 40], 20, learning_rate = 6.0e-5, gamma = 1.0) for _ in range(n_types)]

# agents = [DQNAgent(2, [300, 300, 300, 200], 42,
#                     learning_rate = 0.0001, gamma = 1.0, min_eps = 0.01, 
#                     epsilon_decay = 0.999, max_memory_size = 50_000) for _ in range(n_agents)]

# principal = DQNAgent(n_agents, [100, 200, 100], 21, learning_rate = 0.00001, 
#                      gamma = .99, min_eps = 0.01, epsilon_decay = 0.999, max_memory_size = 5_000)

# winner = [DQNAgent(3, [200, 300, 200], 40, learning_rate = 0.0001, gamma = 1.0,\
#                 min_eps = 0.01, epsilon_decay = 0.999, max_memory_size = 50_000) for _ in range(n_types)]

if __name__ == "__main__":

    run(env, n_episodes, steps_per_episode, n_agents, cost_funcs, type_dict,
        value, reward_process, agents, principal, winner, min_bid = 1.0, 
        max_bid = 5.0, write_output_every=40_000, 
        path_to_directory = path_to_directory)

