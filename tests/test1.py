from apg import *
import numpy as np
import torch
import collections
from collections import namedtuple
import matplotlib.pyplot as plt

quality_min = 2.45

n_agents = 2

env = Env(0.1, n_agents)

cost = [0.07, 0.07]

batch_size = 64

agents = [DDPGAgent(4, [128, 128, 128, 64],2, actor_learning_rate = 1.e-4, 
          critic_learning_rate = 1.e-3, max_memory_size = 50_000) for _ in range(n_agents)]
    
principal = DQNAgent(n_agents*2, [128, 128, 128, 64], n_agents+1, learning_rate = 0.001, 
                     min_eps = 0.03, epsilon_decay = 0.005, max_memory_size = 50_000)

winner = DQNAgent(2, [128, 128, 128, 64], 40, learning_rate = 0.001, 
                     min_eps = 0.01, epsilon_decay = 0.005, max_memory_size = 50_000)

n_episodes = 12_000

steps_per_episode = 10

agent_selections_probs = [0 for _ in range(n_agents)]

noise = OUNoise(env.action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3)


principal_epsiode_rew = []
agent1_epsiode_rew = []
agent2_epsiode_rew = []

principal_update_target_freq = 32

window_size = 50

epsilon = 0.0

did_win = [0, 0]

for i in range(n_agents):
    agents[i].action = env.take_random_action()
principal_state = np.concatenate([*agents[0].action, *agents[1].action]).flatten()
principal_action = principal.get_action(torch.FloatTensor(principal_state), 0)[1]

for i in range(n_agents):
    agent_selections_probs[i] = principal_action[0,i]
    # agents[i].state = np.hstack([*agents[i].action, agent_selections_probs[i].numpy()]).flatten()
    agents[i].state = np.hstack([*agents[i].action, did_win[i]]).flatten()

agent1_b_history = []
agent1_q_history = []
agent2_b_history = []
agent2_q_history = []

agent1_b_avg = []
agent1_q_avg = []
agent2_b_avg = []
agent2_q_avg = []

delivered_q_history = []
deliverd_q_avg = []

selectee = n_agents
for episode in range(n_episodes):
    noise.reset()
    # env.set_env()
    # for i in range(n_agents):
    #     agents[i].action = env.take_random_action()
    # principal_state = np.concatenate([*agents[0].action, *agents[1].action]).flatten()
    # principal_action = principal.get_action(torch.FloatTensor(principal_state), episode)[1]

    # for i in range(n_agents):
    #     agent_selections_probs[i] = principal_action[0,i]
    #     agents[i].state = np.hstack([*agents[i].action, agent_selections_probs[i].numpy()]).flatten()

    done = False
    # r_ind = np.random.randint(0, 100_000)
    for i in range(n_agents):
        agents[i].reward = 0
    principal_reward = 0
    agent1_q = 0
    agent1_b = 0
    agent2_q = 0
    agent2_b = 0
    q_avg    = 0
    for step in range(steps_per_episode):
        r_ind = np.random.randint(0, 100_000)
        if r_ind == 12511:
            r_ind = 12510
        if step == steps_per_episode - 1:
            done = True
        for i in range(n_agents):
            agents[i].action = agents[i].get_action(agents[i].state)
            agents[i].action = noise.get_action(agents[i].action, step)
        principal_new_state = np.concatenate([agents[0].action, agents[1].action]).flatten()
        principal_action = principal.get_action(principal_state, episode)
        selectee = principal_action[0].numpy()
        
        did_win = [0]*n_agents

        if selectee != n_agents:
            did_win[selectee] = 1
        for i in range(n_agents):
            # agent_selections_probs[i] = principal_action[1][0,i]
            # agents[i].new_state = np.hstack([*agents[i].action, agent_selections_probs[i].numpy()]).flatten()
            agents[i].new_state = np.hstack([*agents[i].action, did_win[i]]).flatten()
        if selectee == 0:
            agents[0].reward += 1.0
            agents[1].reward += 0.0
            principal_reward += 1.0
        elif selectee == 1:
            agents[0].reward += 0.0
            agents[1].reward += 1.0
            principal_reward += 1.0
        else:
            agents[0].reward += 0.0
            agents[1].reward += 0.0
            principal_reward += 0.0

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

        opt_result = env.step(r_ind)
        max_try = int(agents[selectee].action[0] / cost[selectee])
        if max_try > 0:
            max_try -= 1
        max_try = min(len(opt_result)-1, max_try)
        max_try = 39
        if opt_result[max_try] + epsilon < quality_min:
            q = opt_result[max_try]
            agents[selectee].reward += -3.0
            principal_reward += -3.0

        else:
            itr = np.where(opt_result + epsilon >= agents[selectee].action[1])[0]
            if itr.any():
                itr = itr[0]
            else:
                itr = max_try

            max_try = min(itr, max_try)
            q = opt_result[max_try]
            agents[selectee].reward += (agents[selectee].action[0] - cost[selectee] * (max_try+1))
            principal_reward += (2.*q - agents[selectee].action[0])
            # agents[selectee].reward = agents[selectee].action[0] + 2.*(q - quality_min)
            # principal_reward += (5.*q - agents[selectee].action[0] - (max_try+1)*cost[selectee])

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

        agent1_b += agents[0].action[0]
        agent1_q += agents[0].action[1]
        agent2_b += agents[1].action[0]
        agent2_q += agents[1].action[1]

        print("####################")
        print("episode {}, step {}".format(episode+1, step+1))
        print("agent {}: b = {}, q = {}".format(1, agents[0].action[0], agents[0].action[1]))
        print("agent {}: b = {}, q = {}".format(2, agents[1].action[0], agents[1].action[1]))
        print("winner: agent {}, deliverd quality = {}".format(selectee+1, q))
        print(principal_action[1])
        print("####################")
        q_avg += q

    if (episode + 1) % principal_update_target_freq == 0:
        principal.target.load_state_dict(principal.policy.state_dict())

    for i in range(n_agents):
        agents[i].reward_history.append(agents[i].reward)
    principal.reward_history.append(principal_reward)
    agent1_b_history.append(agent1_b / steps_per_episode)
    agent1_q_history.append(agent1_q / steps_per_episode)
    agent2_b_history.append(agent2_b / steps_per_episode)
    agent2_q_history.append(agent2_q / steps_per_episode)
    delivered_q_history.append(q_avg / steps_per_episode)

    print('----------------------------------------------------------')
    print("episode {}, agent {} reward is: {}".format(episode+1, 1, agents[0].reward))
    print("episode {}, agent {} reward is: {}".format(episode+1, 2, agents[1].reward))
    print("episode {}, principal reward is {}".format(episode+1, principal_reward))
    print("episode {}, agent 1 average b is {}".format(episode+1, agent1_b / steps_per_episode))
    print("episode {}, agent 1 average q is {}".format(episode+1, agent1_q / steps_per_episode))
    print("episode {}, agent 2 average b is {}".format(episode+1, agent2_b / steps_per_episode))
    print("episode {}, agent 2 average q is {}".format(episode+1, agent2_q / steps_per_episode))
    print("episode {}, delivered q average is {}".format(episode+1, delivered_q_history[-1]))
    print('----------------------------------------------------------')

    if (episode+1) > window_size:
        for i in range(n_agents):
            agents[i].average_reward.append(np.mean(agents[i].reward_history[-window_size:]))
        principal.average_reward.append(np.mean(principal.reward_history[-window_size:]))
        agent1_b_avg.append(np.mean(agent1_b_history[-window_size:]))
        agent1_q_avg.append(np.mean(agent1_q_history[-window_size:]))
        agent2_b_avg.append(np.mean(agent2_b_history[-window_size:]))
        agent2_q_avg.append(np.mean(agent2_q_history[-window_size:]))
        deliverd_q_avg.append(np.mean(delivered_q_history[-window_size:]))
    

        print("\n")
        print('*********************************')
        print("episode {}, agent {} average reward is: {}".format(episode+1, 1, agents[0].average_reward[-1]))
        print("episode {}, agent {} average reward is: {}".format(episode+1, 2, agents[1].average_reward[-1]))
        print("episode {}, principal average reward is {}:".format(episode+1, principal.average_reward[-1]))
        print("episode {}, agent 1 average b is {}:".format(episode+1, agent1_b_avg[-1]))
        print("episode {}, agent 1 average q is {}:".format(episode+1, agent1_q_avg[-1]))
        print("episode {}, agent 2 average b is {}:".format(episode+1, agent2_b_avg[-1]))
        print("episode {}, agent 2 average q is {}:".format(episode+1, agent2_q_avg[-1]))
        print("episode {}, delivered q average is {}:".format(episode+1, deliverd_q_avg[-1]))
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

plt.legend()

plt.figure()
for i in range(n_agents):
    plt.plot(agents[i].average_reward, label = 'agent'+str(i+1))
plt.title('average reward history')
plt.plot(principal.average_reward, label = 'principal')
plt.legend()

plt.figure()
plt.title('average b and q')
plt.plot(agent1_b_avg, label = 'agent 1 b average')
plt.plot(agent1_q_avg, label = 'agent 1 q average')
plt.plot(agent2_b_avg, label = 'agent 2 b average')
plt.plot(agent2_q_avg, label = 'agent 2 q average')
plt.plot(deliverd_q_avg, label = 'delivered q average')

plt.legend()
plt.show()




