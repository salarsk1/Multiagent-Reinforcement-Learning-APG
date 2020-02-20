from apg import *
import numpy as np
import torch
import collections
from collections import namedtuple
import matplotlib.pyplot as plt
import pickle

q_max = 0.0

n_agents = 2

env = Env(0.1, n_agents)

cost = [0.05, 0.1]

batch_size = 64

agents = [DDPGAgent(3, [500, 200, 500], 2, actor_learning_rate = 1.e-4, 
          critic_learning_rate = 1.e-3, gamma=1.0, max_memory_size = 10_000) for _ in range(n_agents)]

principal = DQNAgent(n_agents*2, [200, 100, 200], n_agents+1, learning_rate = 0.0001, 
                     gamma = 1.0, min_eps = 0.0, epsilon_decay = 0.999, max_memory_size = 10_000)

winner = DQNAgent(3, [200, 100, 200], 40, learning_rate = 0.0001, gamma = 1.0,\
                min_eps = 0.0, epsilon_decay = 0.999, max_memory_size = 10_000)

n_episodes = 4_000

steps_per_episode = 10

agent_selections_probs = [0 for _ in range(n_agents)]

noise = OUNoise(env.action_space, mu=.0, theta=0.1, max_sigma=.3, min_sigma=.3)


principal_epsiode_rew = []
agent1_epsiode_rew = []
agent2_epsiode_rew = []

principal_update_target_freq = 4

window_size = 50

epsilon = 0.0

did_win = [0, 0]

for i in range(n_agents):
    agents[i].action = env.take_random_action()

principal_state = np.concatenate([*agents[0].action, *agents[1].action]).flatten()

principal_action = principal.get_action(torch.FloatTensor(principal_state), 1.0, 0)[0]

selectee = 0

max_try = 1

q = 1

success = 0

winner_state = np.hstack([*agents[selectee].action[0], cost[0]])
for i in range(n_agents):
    agents[i].state = np.hstack([*agents[i].action, did_win[i]]).flatten()
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

winner.state = np.hstack([0, 0, cost[0]])

winner.action = winner.get_action(winner.state, 10.0, 0)



real_bid_history = [[], []]
real_quality_history = [[], []]
real_try_history = [[], []]




for episode in range(n_episodes):
    noise.reset()
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


    bid_episode = [[], []]
    quality_episode = [[], []]
    try_episode = [[], []]

    for step in range(steps_per_episode):
        r_ind = np.random.randint(0, 100_000)
        if r_ind == 72891 or r_ind == 12511:
            r_ind = 12510
        # if step == steps_per_episode - 1:
        #     done = True
        for i in range(n_agents):
            agents[i].action = agents[i].get_action(agents[i].state)
            agents[i].action = noise.get_action(agents[i].action, step)
        principal_new_state = np.concatenate([agents[0].action, agents[1].action]).flatten()
        principal_action = principal.get_action(principal_state, 1.0, episode)
        selectee = principal_action[0]
        print('selectee', selectee + 1)
        
        did_win = [0]*n_agents

        if selectee != n_agents:
            did_win[selectee] = 1
        # for i in range(n_agents):
        #     # agent_selections_probs[i] = principal_action[1][0,i]
        #     # agents[i].new_state = np.hstack([*agents[i].action, agent_selections_probs[i].numpy()]).flatten()
        #     agents[i].new_state = np.hstack([*agents[i].action, did_win[i]]).flatten()
        if selectee == 0:
            agents[0].reward += 0.0
            agents[1].reward += 0.0
            principal_reward += -1.0
        elif selectee == 1:
            agents[0].reward += 0.0
            agents[1].reward += 0.0
            principal_reward += -1.0
        else:
            agents[0].reward += -1.0
            agents[1].reward += -1.0
            principal_reward += -2.0

        if selectee == n_agents:
            
            for i in range(n_agents):
                agents[i].replay.push(agents[i].state, agents[i].action, 
                            agents[i].reward, agents[i].new_state, done)
            principal.replay.push(principal_state, principal_action[0], 
                        principal_reward, principal_new_state, done)
            
            for i in range(n_agents):
                agents[i].state = agents[i].new_state
            principal_state = principal_new_state

            for i in range(n_agents):
                if len(agents[i].replay) > batch_size:
                    agents[i].update(batch_size)
            if len(principal.replay) > batch_size:
               principal.update(batch_size, episode)

            print("####################")
            print("episode {}, step {}".format(episode+1, step+1))
            print("agent {}: b = {}, q = {}".format(1, agents[0].action[0], agents[0].action[1]))
            print("agent {}: b = {}, q = {}".format(2, agents[1].action[0], agents[1].action[1]))
            print(principal_action[1])
            print("NO WINNER")
            print("####################")
            continue

        count_win[selectee] += 1

        winner.new_state = np.hstack([*agents[selectee].action, cost[selectee]])

        winner.replay.push(winner.state, winner.action[0], 
                                    winner.reward, winner.new_state, 
                                    done_prev)

        winner.action = winner.get_action(winner.new_state, 10.0, episode)
        max_try = winner.action[0]
        try_avg[selectee] += max_try
        opt_result = env.step(r_ind)
        q = opt_result[max_try]

        for i in range(n_agents):
            # agent_selections_probs[i] = principal_action[1][0,i]
            # agents[i].new_state = np.hstack([*agents[i].action, agent_selections_probs[i].numpy()]).flatten()
            agents[i].new_state = np.hstack([*agents[i].action, did_win[i]]).flatten()

        if q + epsilon < agents[selectee].action[1]:
            agents[selectee].reward += -4.0 + (q - agents[selectee].action[1])
            principal_reward += -3.0
            winner.reward += -4.0 + (q - agents[selectee].action[1])
            success = 0
            # done = True
        else:
            winner.reward += (1.0*agents[selectee].action[0] +(q - agents[selectee].action[1]) - cost[selectee] * (max_try+1))
            principal_reward += (2.*q - agents[selectee].action[0])
            agents[selectee].reward += winner.reward
            success = 1
            # q_max = q
            # winner[selectee].reward += agents[selectee].action[0]
            # agents[selectee].reward += winner[selectee].reward
            # principal_reward += (1.*q - agents[selectee].action[0] - (max_try+1)*cost[selectee])
            # success = 1
        # done = True

        for i in range(n_agents):
            agents[i].replay.push(agents[i].state, agents[i].action, 
                            agents[i].reward, agents[i].new_state, done)

        principal.replay.push(principal_state, principal_action[0], 
                    principal_reward, principal_new_state, done)

        winner.replay.push(winner.state, winner.action[0], 
                                    winner.reward, winner.new_state, done)
            
        for i in range(n_agents):
            agents[i].state = agents[i].new_state
        principal_state = principal_new_state
        winner.state = winner.new_state

        for i in range(n_agents):
            if len(agents[i].replay) > batch_size:
                agents[i].update(batch_size)

        if len(principal.replay) > batch_size:
            principal.update(batch_size, episode)

        if len(winner.replay) > batch_size:
            winner.update(batch_size, episode)

        if selectee == 0:
            agent1_b += agents[0].action[0]
            agent1_q += agents[0].action[1]
        elif selectee == 1:
            agent2_b += agents[1].action[0]
            agent2_q += agents[1].action[1]

        bid_episode[selectee].append(agents[selectee].action[0])
        quality_episode[selectee].append(agents[selectee].action[1])
        try_episode[selectee].append(max_try)


        print("####################")
        print("episode {}, step {}".format(episode+1, step+1))
        print("agent {}: b = {}, q = {}".format(1, agents[0].action[0], agents[0].action[1]))
        print("agent {}: b = {}, q = {}".format(2, agents[1].action[0], agents[1].action[1]))
        print("winner: agent {}, deliverd quality = {}".format(selectee+1, q))
        print("winner effort level = {}".format(max_try+1))
        print(principal_action[1])
        print(winner.action[1])
        print("####################")
        q_avg[selectee] += q
        done_prev = done
        if done:
            break
    for i in range(n_agents):
        if len(bid_episode) > 0:
            real_bid_history[i].append(bid_episode[i])
            real_quality_history[i].append(quality_episode[i])
            real_try_history[i].append(try_episode[i])

    if (episode + 1) % principal_update_target_freq == 0:
        principal.target.load_state_dict(principal.policy.state_dict())
        winner.target.load_state_dict(winner.policy.state_dict())

    for i in range(n_agents):
        agents[i].reward_history.append(agents[i].reward)
    winner.reward_history.append(winner.reward)
    principal.reward_history.append(principal_reward)
    
    
    if count_win[0] > 0:
        agent1_b_history.append(agent1_b / (count_win[0]))
        agent1_q_history.append(agent1_q / (count_win[0]))
        delivered_q_history[0].append(q_avg[0] / count_win[0])
        try_history[0].append(try_avg[0] / count_win[0])

    if count_win[1] > 0:
        agent2_b_history.append(agent2_b / (count_win[1]))
        agent2_q_history.append(agent2_q / (count_win[1]))
        delivered_q_history[1].append(q_avg[1] / count_win[1])
        try_history[1].append(try_avg[1] / count_win[1])

    if count_win[0]>0 and count_win[1]>0:
        print('----------------------------------------------------------')
        print("episode {}, agent {} reward is: {}".format(episode+1, 1, agents[0].reward))
        print("episode {}, agent {} reward is: {}".format(episode+1, 2, agents[1].reward))
        print("episode {}, principal reward is {}".format(episode+1, principal_reward))
        print("episode {}, winner reward is {}".format(episode+1, winner.reward))
        # print("episode {}, winner {} reward is {}".format(episode+1, 2, winner[1].reward))

        print("episode {}, agent 1 average b is {}".format(episode+1, agent1_b_history[-1]))
        print("episode {}, agent 1 average q is {}".format(episode+1, agent1_q_history[-1]))
        print("episode {}, agent 2 average b is {}".format(episode+1, agent2_b_history[-1]))
        print("episode {}, agent 2 average q is {}".format(episode+1, agent2_q_history[-1]))
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
        agent1_q_avg.append(np.mean(agent1_q_history[-window_size:]))
        agent2_b_avg.append(np.mean(agent2_b_history[-window_size:]))
        agent2_q_avg.append(np.mean(agent2_q_history[-window_size:]))
        for i in range(n_agents):
            deliverd_q_avg[i].append(np.mean(delivered_q_history[i][-window_size:]))
            try_history_avg[i].append(np.mean(try_history[i][-window_size:]))
    

        print("\n")
        print('*********************************')
        print("episode {}, agent {} average reward is: {}".format(episode+1, 1, agents[0].average_reward[-1]))
        print("episode {}, agent {} average reward is: {}".format(episode+1, 2, agents[1].average_reward[-1]))
        print("episode {}, principal average reward is {}:".format(episode+1, principal.average_reward[-1]))
        print("episode {}, agent 1 average b is {}:".format(episode+1, agent1_b_avg[-1]))
        print("episode {}, agent 1 average q is {}:".format(episode+1, agent1_q_avg[-1]))
        print("episode {}, agent 2 average b is {}:".format(episode+1, agent2_b_avg[-1]))
        print("episode {}, agent 2 average q is {}:".format(episode+1, agent2_q_avg[-1]))
        print("episode {}, delivered q average by winner {} is {}:".format(episode+1, 1, deliverd_q_avg[0][-1]))
        print("episode {}, delivered q average by winner {} is {}:".format(episode+1, 2, deliverd_q_avg[1][-1]))
        print("episode {}, max try average by winner {} is {}:".format(episode+1, 1, try_history_avg[0][-1]))
        print("episode {}, max try average by winner {} is {}:".format(episode+1, 2, try_history_avg[1][-1]))
        print('*********************************')

# print("\n")
# print("reward histories")
# print('-------------------------------')
# print("agent {} reward history is {}:".format(1, agents[0].reward_history))
# print("agent {} reward history is {}:".format(2, agents[1].reward_history))
# print("agent {} average reward is {}:".format(1, agents[0].average_reward))
# print("agent {} average reward is {}:".format(2, agents[1].average_reward))

# print("princiapl reward history is {}:".format(principal.reward_history))
# print("principal average reward is {}:".format(agents[i].average_reward))
# print('-------------------------------')



plt.title('reward history')
for i in range(n_agents):
    plt.plot(agents[i].reward_history, label = 'agent'+str(i+1))
plt.plot(principal.reward_history, label = 'principal')
plt.plot(winner.reward_history, label = 'winner')
# plt.plot(winner[1].reward_history, label = 'winner 2')
plt.legend()




plt.figure()
for i in range(n_agents):
    plt.plot(agents[i].average_reward, label = 'agent'+str(i+1))
plt.title('average reward history')
plt.plot(principal.average_reward, label = 'principal')
plt.plot(winner.average_reward, label = 'winner')
# plt.plot(winner[1].average_reward, label = 'winner 2')
plt.savefig('avg_rewad_history.png', dpi = 300)
plt.legend()




plt.figure()
plt.title('average b and q for agent 1')
plt.plot(agent1_b_avg, label = 'agent 1 b average')
plt.plot(agent1_q_avg, label = 'agent 1 q average')
plt.plot(deliverd_q_avg[0], label = 'delivered q average')
plt.savefig('avg_b_q_1.png', dpi = 300)
plt.legend()




plt.figure()
plt.title('average b and q for agent 2')
plt.plot(agent2_b_avg, label = 'agent 2 b average')
plt.plot(agent2_q_avg, label = 'agent 2 q average')
plt.plot(deliverd_q_avg[1], label = 'delivered q average')
plt.savefig('avg_b_q_2.png', dpi = 300)
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

with open('qual.out', 'wb') as f:
    pickle.dump(real_quality_history, f)

with open('try.out', 'wb') as f:
    pickle.dump(real_try_history, f)

plt.show()
