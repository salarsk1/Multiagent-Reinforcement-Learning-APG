from apg import *
import numpy as np
import torch
import collections
from collections import namedtuple
import matplotlib.pyplot as plt
import pickle

q_min = 1.8

n_agents = 2

env = Env(0.1, n_agents)

cost = [0.07, 0.07]

batch_size = 64

# agents = [DQNAgent(2, [100, 200, 200], 41,
#                     learning_rate = 0.0001, gamma = 0.95, min_eps = 0.01, 
#                     epsilon_decay = 0.999, max_memory_size = 2_00) for _ in range(n_agents)]

principal = DQNAgent(n_agents, [100, 200, 100], n_agents, learning_rate = 0.00001, 
                     gamma = .99, min_eps = 0.01, epsilon_decay = 0.999, max_memory_size = 5_000)

principal = A2CAgent(n_agents, [100, 200, 100], n_agents, learning_rate=1e-4, gamma=0.95)

winner = DQNAgent(3, [100, 300, 200], 40, learning_rate = 0.0003, gamma = 0.95,\
                min_eps = 0.01, epsilon_decay = 0.999, max_memory_size = 10_000)

n_episodes = 50_000

steps_per_episode = 4

principal_epsiode_rew = []
agent1_epsiode_rew = []
agent2_epsiode_rew = []

principal_update_target_freq = 10

window_size = 200

epsilon = 0.0

did_win = [0, 0]

for i in range(n_agents):
    agents[i].action = np.array([1,1])

principal_state = np.hstack([agents[i].action[0]*0.05+1.0 for i in range(n_agents)]).flatten()

# principal_action = principal.get_action(torch.FloatTensor(principal_state), 10.0, 0)[0]
principal_action = principal.get_action(torch.FloatTensor(principal_state))[0]

selectee = 0

max_try = 1

q = 1

success = 0

winner_state = np.hstack([agents[selectee].action[0]*0.05+1.0, cost[0]])
for i in range(n_agents):
    agents[i].state = np.hstack([agents[i].action[0], did_win[i]]).flatten()
    agents[i].new_state = agents[i].state


agent1_b_history = []
agent1_q_history = []
agent2_b_history = []
agent2_q_history = []

agent1_b_avg = []
agent1_q_avg = []
agent2_b_avg = []
agent2_q_avg = []

delivered_q_history = [[], []]

deliverd_q_avg = [[], []]

try_history = [[], []]

try_history_avg = [[], []]

selectee = n_agents

done_prev = False

winner.state = np.hstack([0, q_min, cost[0]])

# winner.action = winner.get_action(winner.state, 10., 0)
winner.action = winner.get_action_e_greedy(winner.state, 0)



real_bid_history = [[], []]
real_quality_history = [[], []]
real_try_history = [[], []]




for episode in range(n_episodes):
    done = False
    for i in range(n_agents):
        agents[i].reward = 0
    principal_reward = 0
    winner.reward = 0
    agent1_q = 0
    agent1_b = 0
    agent2_q = 0
    agent2_b = 0
    q_avg    = [0, 0]
    count_win = [0, 0]
    try_avg = [0, 0]
    trajectory = []

    bid_episode = [[] for i in range(n_agents)]
    quality_episode = [[] for i in range(n_agents)]
    try_episode = [[] for i in range(n_agents)]
    # r_ind = np.random.randint(0, 100_000)
    # if r_ind == 72891 or r_ind == 12511:
    #     r_ind = 12510

    for step in range(steps_per_episode):
        r_ind = np.random.randint(0, 100_000)
        if r_ind == 72891 or r_ind == 12511:
            r_ind = 12510
        # if step == steps_per_episode - 1:
        #     done = True
        for i in range(n_agents):
            # agents[i].action = agents[i].get_action(agents[i].state, 10.0, episode)
            agents[i].action = agents[i].get_action_e_greedy(agents[i].state, episode)

        principal_new_state = np.hstack([agents[i].action[0]*0.05+1.0 for i in range(n_agents)]).flatten()
        # principal_action = principal.get_action(principal_state, 10., episode)
        principal_action = principal.get_action(principal_state)
        selectee = principal_action[0]
        
        did_win = [0]*n_agents

        if selectee != n_agents:
            did_win[selectee] = 1
        if selectee == 0:
            agents[0].reward += 0.0
            agents[1].reward += -1.0
            principal_reward += 0.0
        elif selectee == 1:
            agents[0].reward += -1.0
            agents[1].reward += 0.0
            principal_reward += 0.0
        else:
            agents[0].reward += -1.0
            agents[1].reward += -1.0
            principal_reward += -1.0

        if selectee == n_agents:
            
            for i in range(n_agents):

                agents[i].replay.push(agents[i].state, agents[i].action[0], 
                            agents[i].reward, agents[i].new_state, done)
            # principal.replay.push(principal_state, principal_action[0], 
            #             principal_reward, principal_new_state, done)
            trajectory.append([principal_state, principal_action[0], 
                        principal_reward, principal_new_state, done])
            
            for i in range(n_agents):
                agents[i].state = agents[i].new_state
            principal_state = principal_new_state

            for i in range(n_agents):
                if len(agents[i].replay) > batch_size:
                    agents[i].update(batch_size, episode)
            # if len(principal.replay) > batch_size:
            #     principal.update(batch_size, episode)

            print("####################")
            print("episode {}, step {}".format(episode+1, step+1))
            for i in range(n_agents):
                print("agent {}: b = {}".format(i+1, agents[i].action[0]*0.05+1.0))
                # print("agent {}: b = {}".format(2, agents[1].action[0]*0.05+1.0))
            print(principal_action[1])
            print("NO WINNER")
            print("####################")
            continue
        count_win[selectee] += 1

        winner.new_state = np.hstack([agents[selectee].action[0]*0.05+1.0, q_min, cost[selectee]])

        winner.replay.push(winner.state, winner.action[0], 
                            winner.reward, winner.new_state, 
                            done_prev)

        # winner.action = winner.get_action(winner.new_state, 10., episode)
        winner.action = winner.get_action_e_greedy(winner.new_state, episode)
        max_try = winner.action[0]
        try_avg[selectee] += max_try
        opt_result = env.step(r_ind)
        q = opt_result[max_try]

        for i in range(n_agents):
            agents[i].new_state = np.hstack([agents[i].action[0]*0.05+1.0, did_win[i]]).flatten()


        if q + epsilon < q_min:
            agents[selectee].reward += -2.0
            principal_reward += -2.0
            winner.reward += agents[selectee].reward
            success = 0
            # done = True
        else:
            winner.reward += (agents[selectee].action[0]*0.05+1.0 - 0.5*(max_try + 1) * cost[selectee])
            principal_reward += (2.*q - (agents[selectee].action[0]*0.05+1.0) - 0.5*(max_try + 1) * cost[selectee])
            agents[selectee].reward += winner.reward
            success = 1

        for i in range(n_agents):
            agents[i].replay.push(agents[i].state, agents[i].action[0], 
                            agents[i].reward, agents[i].new_state, done)

        # principal.replay.push(principal_state, principal_action[0], 
        #             principal_reward, principal_new_state, done)
        trajectory.append([principal_state, principal_action[0], 
                        principal_reward, principal_new_state, done])

        # winner.replay.push(winner.state, winner.action[0], 
        #                             winner.reward, winner.new_state, done_prev)
            
        for i in range(n_agents):
            agents[i].state = agents[i].new_state

        principal_state = principal_new_state
        winner.state = winner.new_state

        for i in range(n_agents):
            if len(agents[i].replay) > batch_size:
                agents[i].update(batch_size, episode)

        # if len(principal.replay) > batch_size:
        #     principal.update(batch_size, episode)

        if len(winner.replay) > batch_size: 
            winner.update(batch_size, episode)

        if selectee == 0:
            agent1_b += agents[0].action[0]*0.05+1.0
        elif selectee == 1:
            agent2_b += agents[1].action[0]*0.05+1.0

        bid_episode[selectee].append(agents[selectee].action[0]*0.05+1.0)
        try_episode[selectee].append(max_try)


        print("####################")
        print("episode {}, step {}".format(episode+1, step+1))
        for i in range(n_agents):
            print("agent {}: b = {}".format(i+1, agents[i].action[0]*0.05+1.0))
        # print("agent {}: b = {}".format(2, agents[1].action[0]*0.05+1.0))
        print("winner: agent {}, deliverd quality = {}".format(selectee+1, q))
        print("winner effort level = {}".format(max_try+1))
        print(principal_action[1])
        print(agents[selectee].action[1])
        print(winner.action[1])
        print("####################")
        q_avg[selectee] += q
        done_prev = done
        # if done:
        #     break
    for i in range(n_agents):
        if len(bid_episode) > 0:
            real_bid_history[i].append(bid_episode[i])
            real_try_history[i].append(try_episode[i])

    if (episode + 1) % principal_update_target_freq == 0:
        for i in range(n_agents):
            agents[i].target.load_state_dict(agents[i].policy.state_dict())
        # principal.target.load_state_dict(principal.policy.state_dict())
        winner.target.load_state_dict(winner.policy.state_dict())
    principal.update(trajectory)

    for i in range(n_agents):
        agents[i].reward_history.append(agents[i].reward)
    winner.reward_history.append(winner.reward)
    principal.reward_history.append(principal_reward)
    
    
    if count_win[0] > 0:
        agent1_b_history.append(agent1_b / (count_win[0]))
        delivered_q_history[0].append(q_avg[0] / count_win[0])
        try_history[0].append(try_avg[0] / count_win[0])

    if count_win[1] > 0:
        agent2_b_history.append(agent2_b / (count_win[1]))
        delivered_q_history[1].append(q_avg[1] / count_win[1])
        try_history[1].append(try_avg[1] / count_win[1])

    if count_win[0]>0 and count_win[1]>0:
        print('----------------------------------------------------------')
        for i in range(n_agents):
            print("episode {}, agent {} reward is: {}".format(episode+1, i+1, agents[i].reward))
        # print("episode {}, agent {} reward is: {}".format(episode+1, 2, agents[1].reward))
        print("episode {}, principal reward is {}".format(episode+1, principal_reward))
        print("episode {}, winner reward is {}".format(episode+1, winner.reward))
        # print("episode {}, winner {} reward is {}".format(episode+1, 2, winner[1].reward))

        print("episode {}, agent 1 average b is {}".format(episode+1, agent1_b_history[-1]))
        print("episode {}, agent 2 average b is {}".format(episode+1, agent2_b_history[-1]))
        print("episode {}, delivered q average by winner {} is {}".format(episode+1, 1, delivered_q_history[0][-1]))
        print("episode {}, delivered q average by winner {} is {}".format(episode+1, 2, delivered_q_history[1][-1]))
        print("episode {}, max try average for winner {} is {}".format(episode+1, 1, try_history[0][-1]))
        print("episode {}, max try average for winner {} is {}".format(episode+1, 2, try_history[1][-1]))
        print('----------------------------------------------------------')

    if (episode+1) > window_size:
        for i in range(n_agents):
            agents[i].average_reward.append(np.mean(agents[i].reward_history[-window_size:]))
        winner.average_reward.append(np.mean(winner.reward_history[-window_size:]))
        principal.average_reward.append(np.mean(principal.reward_history[-window_size:]))
        agent1_b_avg.append(np.mean(agent1_b_history[-window_size:]))
        agent2_b_avg.append(np.mean(agent2_b_history[-window_size:]))
        for i in range(n_agents):
            deliverd_q_avg[i].append(np.mean(delivered_q_history[i][-window_size:]))
            try_history_avg[i].append(np.mean(try_history[i][-window_size:]))
    

        print("\n")
        print('*********************************')
        for i in range(n_agents):
            print("episode {}, agent {} average reward is: {}".format(episode+1, i+1, agents[i].average_reward[-1]))
        # print("episode {}, agent {} average reward is: {}".format(episode+1, 2, agents[1].average_reward[-1]))
        print("episode {}, principal average reward is {}:".format(episode+1, principal.average_reward[-1]))
        print("episode {}, agent 1 average b is {}:".format(episode+1, agent1_b_avg[-1]))
        print("episode {}, agent 2 average b is {}:".format(episode+1, agent2_b_avg[-1]))
        for i in range(n_agents):
            print("episode {}, delivered q average by winner {} is {}:".format(episode+1, i+1, deliverd_q_avg[i][-1]))
        # print("episode {}, delivered q average by winner {} is {}:".format(episode+1, 2, deliverd_q_avg[1][-1]))
        for i in range(n_agents):
            print("episode {}, max try average by winner {} is {}:".format(episode+1, i+1, try_history_avg[i][-1]))
        # print("episode {}, max try average by winner {} is {}:".format(episode+1, 2, try_history_avg[1][-1    ]))
        print('*********************************')

    if (episode+1) % 5000 == 0:
        plt.title('reward history')
        for i in range(n_agents):
            plt.plot(agents[i].reward_history, label = 'agent'+str(i+1))
        plt.plot(principal.reward_history, label = 'principal')
        plt.plot(winner.reward_history, label = 'winner')
        # plt.plot(winner[1].reward_history, label = 'winner 2')
        plt.savefig('rewad_history.png', dpi = 300)
        plt.legend()




        plt.figure()
        plt.title('average reward history')
        for i in range(n_agents):
            plt.plot(agents[i].average_reward, label = 'agent'+str(i+1))
        plt.plot(principal.average_reward, label = 'principal')
        plt.plot(winner.average_reward, label = 'winner')
        plt.savefig('avg_rewad_history.png', dpi = 300)
        plt.legend()




        plt.figure()
        plt.title('average b for agent 1')
        plt.plot(agent1_b_avg, label = 'agent 1 b average')
        plt.plot(deliverd_q_avg[0], label = 'delivered q average')
        plt.savefig('avg_b_1.png', dpi = 300)
        plt.legend()




        plt.figure()
        plt.title('average b for agent 2')
        plt.plot(agent2_b_avg, label = 'agent 2 b average')
        plt.plot(deliverd_q_avg[1], label = 'delivered q average')
        plt.savefig('avg_b_2.png', dpi = 300)
        plt.legend()




        plt.figure()
        plt.title('max try average by agent 1')
        plt.plot(try_history_avg[0], label = 'max try average')
        plt.savefig('max_try_1.png', dpi = 300)
        plt.legend()



        plt.figure()
        plt.title('max try average by agent 2')
        plt.plot(try_history_avg[1], label = 'max try average')
        plt.savefig('max_try_2.png', dpi = 300)
        plt.legend()

        with open('bid.out', 'wb') as f:
            pickle.dump(real_bid_history, f)


        with open('try.out', 'wb') as f:
            pickle.dump(real_try_history, f)

        plt.show()
