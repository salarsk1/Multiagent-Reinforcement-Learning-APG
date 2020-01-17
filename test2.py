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

agents = [DDPGAgent(env, [256, 256, 128], actor_learning_rate = 5.e-5, 
          critic_learning_rate = 5.e-4, max_memory_size = 50_000) for _ in range(n_agents)]
    
principal = DQNAgent(n_agents*2, [256, 256, 128], n_agents, learning_rate = 0.001, 
                     min_eps = 0.01, epsilon_decay = 0.9995, max_memory_size = 50_000)

winner = DQNAgent(3, [256, 256, 128], 40, learning_rate = 0.0001, 
                     min_eps = 0.01, epsilon_decay = 0.9995, max_memory_size = 50_000)

n_episodes = 8_000

steps_per_episode = 10

agent_selections_probs = [0 for _ in range(n_agents)]

noise = OUNoise(env.action_space, mu=0.0, theta=0.0, max_sigma=0.5, min_sigma=0.5)


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
principal_action = principal.get_action(torch.FloatTensor(principal_state), 0)[0]
selectee = principal_action.numpy()
max_try = 1
q = 1
success = 0

winner_state = np.hstack([*agents[selectee].action[0], selectee])
for i in range(n_agents):
    agents[i].state = np.hstack([*agents[i].action, 1, did_win[i]]).flatten()

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

try_history = []
try_history_avg = []

selectee = n_agents
done_prev = False
winner_state = np.hstack([0, 0, 0])
winner_action = winner.get_action(winner_state, 0)
for episode in range(n_episodes):
    noise.reset()
    done = False
    for i in range(n_agents):
        agents[i].reward = 0
    principal_reward = 0
    winner_reward = 0
    agent1_q = 0
    agent1_b = 0
    agent2_q = 0
    agent2_b = 0
    q_avg    = 0
    count_win = 0
    try_avg = 0
    for step in range(steps_per_episode):
        r_ind = np.random.randint(0, 100_000)
        if r_ind == 72891:
            r_ind = 1223
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
        # for i in range(n_agents):
        #     # agent_selections_probs[i] = principal_action[1][0,i]
        #     # agents[i].new_state = np.hstack([*agents[i].action, agent_selections_probs[i].numpy()]).flatten()
        #     agents[i].new_state = np.hstack([*agents[i].action, did_win[i]]).flatten()
        if selectee == 0:
            agents[0].reward += 1.0
            agents[1].reward += -1.0
            principal_reward += 1.0
        elif selectee == 1:
            agents[0].reward += -1.0
            agents[1].reward += 1.0
            principal_reward += 1.0
        # else:
        #     agents[0].reward += -2.0
        #     agents[1].reward += -2.0
        #     principal_reward += -2.0

        # if selectee == n_agents:

        #     done = True
            
        #     for i in range(n_agents):
        #         agents[i].replay.push(agents[i].state, agents[i].action, 
        #                     agents[i].reward, agents[i].new_state, done)
        #     principal.replay.push(principal_state, principal_action[0], 
        #                 principal_reward, principal_new_state, done)
            
        #     for i in range(n_agents):
        #         agents[i].state = agents[i].new_state
        #     principal_state = principal_new_state

        #     for i in range(n_agents):
        #         if len(agents[i].replay) > batch_size:
        #             agents[i].update(batch_size)
        #     if len(principal.replay) > batch_size:
        #        principal.update(batch_size, episode)

        #     print("####################")
        #     print("episode {}, step {}".format(episode+1, step+1))
        #     print("agent {}: b = {}, q = {}".format(1, agents[0].action[0], agents[0].action[1]))
        #     print("agent {}: b = {}, q = {}".format(2, agents[1].action[0], agents[1].action[1]))
        #     print(principal_action[1])
        #     print("NO WINNER")
        #     print("####################")
        #     break


        count_win += 1

        winner_new_state = np.hstack([*agents[selectee].action, selectee])


        winner.replay.push(winner_state, winner_action[0], winner_reward, winner_new_state, done_prev)


        winner_action = winner.get_action(winner_new_state, episode)
        max_try = winner_action[0].numpy()
        try_avg += max_try
        opt_result = env.step(r_ind)
        q = opt_result[max_try]

        for i in range(n_agents):
            # agent_selections_probs[i] = principal_action[1][0,i]
            # agents[i].new_state = np.hstack([*agents[i].action, agent_selections_probs[i].numpy()]).flatten()
            agents[i].new_state = np.hstack([*agents[i].action, q, did_win[i]]).flatten()

        if q + epsilon < agents[selectee].action[1]:
            agents[selectee].reward += -(max_try + 1) * cost[selectee]
            principal_reward += -agents[selectee].action[0]
            winner_reward += -(max_try + 1) * cost[selectee]
            success = 0
            done = True
        else:
            winner_reward += (1.3*agents[selectee].action[0] - cost[selectee] * (max_try+1))
            principal_reward += (1.4*q - agents[selectee].action[0])
            agents[selectee].reward += winner_reward
            success = 1
            # winner_reward += agents[selectee].action[0]
            # agents[selectee].reward += winner_reward
            # principal_reward += (3.*q - agents[selectee].action[0] - (max_try+1)*cost[selectee])
            # success = 1

        for i in range(n_agents):
            agents[i].replay.push(agents[i].state, agents[i].action, 
                            agents[i].reward, agents[i].new_state, done)

        principal.replay.push(principal_state, principal_action[0], 
                    principal_reward, principal_new_state, done)

        winner.replay.push(winner_state, winner_action[0], winner_reward, winner_new_state, done)
            
        for i in range(n_agents):
            agents[i].state = agents[i].new_state
        principal_state = principal_new_state
        winner_state = winner_new_state

        for i in range(n_agents):
            if len(agents[i].replay) > batch_size:
                agents[i].update(batch_size)

        if len(principal.replay) > batch_size:
            principal.update(batch_size, episode)

        if len(winner.replay) > batch_size:
            winner.update(batch_size, episode)

        agent1_b += agents[0].action[0]
        agent1_q += agents[0].action[1]
        agent2_b += agents[1].action[0]
        agent2_q += agents[1].action[1]

        print("####################")
        print("episode {}, step {}".format(episode+1, step+1))
        print("agent {}: b = {}, q = {}".format(1, agents[0].action[0], agents[0].action[1]))
        print("agent {}: b = {}, q = {}".format(2, agents[1].action[0], agents[1].action[1]))
        print("winner: agent {}, deliverd quality = {}".format(selectee+1, q))
        print("winner effort level = {}".format(max_try+1))
        print(principal_action[1])
        print("####################")
        q_avg += q
        done_prev = done
        if done:
            break

    if (episode + 1) % principal_update_target_freq == 0:
        principal.target.load_state_dict(principal.policy.state_dict())
        winner.target.load_state_dict(winner.policy.state_dict())

    for i in range(n_agents):
        agents[i].reward_history.append(agents[i].reward)
    principal.reward_history.append(principal_reward)
    winner.reward_history.append(winner_reward)
    if count_win != 0:
        agent1_b_history.append(agent1_b / count_win)
        agent1_q_history.append(agent1_q / count_win)
        agent2_b_history.append(agent2_b / count_win)
        agent2_q_history.append(agent2_q / count_win)
        delivered_q_history.append(q_avg / count_win)
        try_history.append(try_avg / count_win)

    print('----------------------------------------------------------')
    print("episode {}, agent {} reward is: {}".format(episode+1, 1, agents[0].reward))
    print("episode {}, agent {} reward is: {}".format(episode+1, 2, agents[1].reward))
    print("episode {}, principal reward is {}".format(episode+1, principal_reward))
    print("episode {}, winner reward is {}".format(episode+1, winner_reward))
    print("episode {}, agent 1 average b is {}".format(episode+1, agent1_b_history[-1]))
    print("episode {}, agent 1 average q is {}".format(episode+1, agent1_q_history[-1]))
    print("episode {}, agent 2 average b is {}".format(episode+1, agent2_b_history[-1]))
    print("episode {}, agent 2 average q is {}".format(episode+1, agent2_q_history[-1]))
    print("episode {}, delivered q average is {}".format(episode+1, delivered_q_history[-1]))
    print("episode {}, max try average is {}".format(episode+1, try_history[-1]))
    print('----------------------------------------------------------')

    if (episode+1) > window_size:
        for i in range(n_agents):
            agents[i].average_reward.append(np.mean(agents[i].reward_history[-window_size:]))
        principal.average_reward.append(np.mean(principal.reward_history[-window_size:]))
        winner.average_reward.append(np.mean(winner.reward_history[-window_size:]))
        agent1_b_avg.append(np.mean(agent1_b_history[-window_size:]))
        agent1_q_avg.append(np.mean(agent1_q_history[-window_size:]))
        agent2_b_avg.append(np.mean(agent2_b_history[-window_size:]))
        agent2_q_avg.append(np.mean(agent2_q_history[-window_size:]))
        deliverd_q_avg.append(np.mean(delivered_q_history[-window_size:]))
        try_history_avg.append(np.mean(try_history[-window_size:]))
    

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
        print("episode {}, max try average is {}:".format(episode+1, try_history_avg[-1]))
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

plt.legend()

plt.figure()
for i in range(n_agents):
    plt.plot(agents[i].average_reward, label = 'agent'+str(i+1))
plt.title('average reward history')
plt.plot(principal.average_reward, label = 'principal')
plt.plot(winner.average_reward, label = 'winner')
plt.legend()

plt.figure()
plt.title('average b and q')
plt.plot(agent1_b_avg, label = 'agent 1 b average')
plt.plot(agent1_q_avg, label = 'agent 1 q average')
plt.plot(agent2_b_avg, label = 'agent 2 b average')
plt.plot(agent2_q_avg, label = 'agent 2 q average')
plt.plot(deliverd_q_avg, label = 'delivered q average')
plt.legend()

plt.figure()
plt.title('max try average')
plt.plot(try_history_avg, label = 'max try average')
plt.legend()
plt.show()
