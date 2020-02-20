import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from ._utils import *

__all__ = ["run"]

def run(env, n_episodes, steps_per_episode, n_agents, cost_funcs, type_dict,
        value, reward_process, agents, principal, winner, 
        min_bid = 1.0, max_bid = 3.0, batch_size = 64, 
        window_size = 200, write_output_every = 5000, path_to_directory = "/Users/salarsk/developments/phd/nsf/rl_acq/code/"):
    principal_epsiode_rew = []
    agent1_epsiode_rew = []
    agent2_epsiode_rew = []
    n_types = len(winner)

    principal_update_target_freq = 30


    did_win = [0 for _ in range(n_agents)]

    for i in range(n_agents):
        agents[i].action = np.array([1,1])

    principal_state = np.hstack([0, value.q_min])

    # principal_action = principal.get_action(torch.FloatTensor(principal_state), 10.0, 0)[0]
    principal_action = principal.get_action(torch.FloatTensor(principal_state),0)[0]

    selectee = 0

    max_try = 1

    q = 1

    success = 0

    winner_state = np.hstack([1.0, value.q_min, cost_funcs[0].c])
    for i in range(n_agents):
        agents[i].state = np.hstack([agents[i].action[0], did_win[i]]).flatten()
        agents[i].new_state = agents[i].state


    agents_b_history = [[] for _ in range(n_agents)]
    agents_q_history = [[] for _ in range(n_agents)]

    agents_b_avg = [[] for _ in range(n_agents)]
    agents_q_avg = [[] for _ in range(n_agents)]

    delivered_q_history = [[] for _ in range(n_agents)]

    deliverd_q_avg = [[] for _ in range(n_agents)]

    try_history = [[] for _ in range(n_agents)]

    try_history_avg = [[] for _ in range(n_agents)]

    selectee = n_agents

    done_prev = False

    
    for i in range(n_types):
        winner[i].state = np.hstack([2.0, value.q_min, cost_funcs[i].c])

    # winner.action = winner.get_action(winner.state, 10., 0)
    # winner.action = winner.get_action_e_greedy(winner.state, 0)
    for i in range(n_types):
        winner[i].action = winner[i].get_action(winner[i].state, 0)

    real_bid_history = [[] for _ in range(n_agents)]
    real_quality_history = [[] for _ in range(n_agents)]
    real_try_history = [[] for _ in range(n_agents)]
    q = 0
    for episode in range(n_episodes):
        done = False
        for i in range(n_agents):
            agents[i].reward = 0
        for i in range(n_types):
            winner[i].reward = 0
        principal_reward = 0
        agents_q = [0 for _ in range(n_agents)]
        agents_b = [0 for _ in range(n_agents)]
        agent2_q = 0
        agent2_b = 0
        q_avg    = [0 for _ in range(n_agents)]
        count_win = [0 for _ in range(n_agents)]
        try_avg = [0 for _ in range(n_agents)]
        trajectory = []

        trajectory_agents = [[] for _ in range(n_agents)]
        trajectory_winner = [[] for _ in range(n_types)]

        bid_episode = [[] for i in range(n_agents)]
        quality_episode = [[] for i in range(n_agents)]
        try_episode = [[] for i in range(n_agents)]

        for step in range(steps_per_episode):
            r_ind = np.random.randint(0, 100_000)
            if r_ind == 72891 or r_ind == 12511:
                r_ind = 12510

            principal_action = principal.get_action(principal_state, episode)
            q_min_action = principal_action[0]
            for i in range(n_agents):
                agents[i].state = np.hstack([map_nn_output(agents[i].action[0], \
                                            agents[i].num_actions-1, min_bid, max_bid),\
                                            did_win[i]]).flatten()
            for i in range(n_agents):
                agents[i].action = agents[i].get_action(agents[i].state, episode)
            check_participaion = False
            for i in range(n_agents):
                if agents[i].action[0] != agents[i].num_actions-1:
                    check_participaion = True
                    break
            if not check_participaion:
                for i in range(n_agents):
                    agents[i].new_state = np.hstack([0.0, 0.0]).flatten()
                    trajectory_agents[i].append([agents[i].state, agents[i].action[0], \
                                                 0, agents[i].new_state, \
                                                 done])
                    agents[i].update(trajectory_agents[i])
                    # agents[i].replay.push(agents[i].state, agents[i].action[0], 
                    #             agents[i].reward, agents[i].new_state, done)

                    agents[i].state = agents[i].new_state
                # trajectory_winner.append([winner.state, winner.action[0], \
                #           winner.reward, winner.new_state, \
                #           done_prev])

                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                print("episode {}, step {}: NO PARTICIPATION".format(episode+1, step+1))
                print("requested minimum quality = {}".format(value.q_min))
                # q_avg[selectee] += 0
                done_prev = done
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                for i in range(n_agents):
                    bid_episode[i].append(0.0)
                    # try_episode[i].append(0.0)
                    quality_episode[i].append(0.0)
                continue
            bb = np.array([agents[i].action[0] for i in range(n_agents)])
            bb = np.where(bb == bb.min())[0]
            index = np.random.choice(bb.shape[0])
            selectee = bb[index]
            did_win = [0]*n_agents
            selectee_type = type_dict[selectee]

            for i in range(n_agents):
                agents[i].new_state = np.hstack([map_nn_output(agents[i].action[0], \
                                                agents[i].num_actions-1, min_bid, max_bid),\
                                                did_win[i]]).flatten()

            did_win[selectee] = 1
            for i in range(n_agents):
                if i==selectee:
                    agents[i].reward += reward_process.reward_winning(map_nn_output(agents[selectee].action[0], \
                                                                    agents[selectee].num_actions-1, min_bid, max_bid))
                    principal_reward += reward_process.principal_selecting_reward()
                else:
                    agents[i].reward += reward_process.reward_losing()

            agents[selectee].new_state = np.hstack([map_nn_output(agents[selectee].action[0], \
                                        agents[selectee].num_actions-1, min_bid, max_bid), \
                                        did_win[selectee]]).flatten()
            count_win[selectee] += 1


            winner[selectee_type].new_state = np.hstack([map_nn_output(agents[selectee].action[0],
                                                                       agents[selectee].num_actions-1, min_bid, max_bid),
                                                                       value.q_min, cost_funcs[selectee].c])

            # winner[selectee_type].replay.push(winner[selectee_type].state, winner[selectee_type].action[0],
            #                                 winner[selectee_type].reward, winner[selectee_type].new_state, 
            #                                 done_prev)
            trajectory_winner[selectee_type].append([winner[selectee_type].state, winner[selectee_type].action[0], \
                                      winner[selectee_type].reward, winner[selectee_type].new_state, \
                                      done_prev])

            winner[selectee_type].action = winner[selectee_type].get_action(winner[selectee_type].new_state, episode)
            max_try = int(map_nn_output(winner[selectee_type].action[0], winner[selectee_type].num_actions, 2, 40))
            agent_cost = cost_funcs[selectee](max_try)

            try_avg[selectee] += max_try
            opt_result = env.step(r_ind)
            q = opt_result[max_try-1]
            winner[selectee_type].reward += reward_process.winner_prime_reward(q, map_nn_output(agents[selectee].action[0], \
                                            agents[selectee].num_actions-1, min_bid, max_bid), \
                                            cost_funcs[selectee](max_try))
            agents[selectee].reward +=reward_process.winner_agent_reward(q, \
                                        map_nn_output(agents[selectee].action[0], \
                                        agents[selectee].num_actions-1, min_bid, max_bid),\
                                        cost_funcs[selectee](max_try))

            principal_reward += reward_process.principal_reward(q, 
                                map_nn_output(agents[selectee].action[0],
                                agents[selectee].num_actions-1, min_bid, max_bid),
                                cost_funcs[selectee](max_try))
            if q < value.q_min:
                success = 0
            else:
                success = 1

            
            # Set the principal's trajectiry and state

            principal_new_state = np.hstack([value.q_min, success])
            


            principal_state = principal_new_state

            for i in range(n_agents):
                trajectory_agents[i].append([agents[i].state, agents[i].action[0], \
                                             agents[i].reward, agents[i].new_state, \
                                             done])

            # for i in range(n_agents):
            #     agents[i].replay.push(agents[i].state, agents[i].action[0], 
            #                     agents[i].reward, agents[i].new_state, done)

            # winner.replay.push(winner.state, winner.action[0], 
            #                             winner.reward, winner.new_state, done_prev)
                
            for i in range(n_agents):
                agents[i].state = agents[i].new_state

            # principal_state = principal_new_state
            winner[selectee_type].state = winner[selectee_type].new_state

            # for i in range(n_agents):
            #     if len(agents[i].replay) > batch_size:
            #         agents[i].update(batch_size, episode)

            # if len(principal.replay) > batch_size:
            #     principal.update(batch_size, episode)

            # for i in range(n_types):
            #     if len(winner[i].replay) > batch_size: 
            #         winner[i].update(batch_size, episode)

            for i in range(n_agents):
                if selectee == i:
                    agents_b[i] += map_nn_output(agents[i].action[0], agents[i].num_actions-1, 
                                                 min_bid, max_bid)
                    break

            bid_episode[selectee].append(map_nn_output(agents[selectee].action[0], \
                                    agents[selectee].num_actions-1, min_bid, max_bid))
            try_episode[selectee].append(max_try)
            quality_episode[selectee].append(q)

            print("####################")
            print("episode {}, step {}".format(episode+1, step+1))
            for i in range(n_agents):
                print("agent {}: b = {}".format(i+1, map_nn_output(agents[i].action[0], \
                                                agents[i].num_actions-1, min_bid, max_bid)))
            print("winner: agent {}, deliverd quality = {}".format(selectee+1, q))
            print("winner {} effort level = {}".format(selectee+1, max_try))
            print("requested minimum quality = {}".format(value.q_min))
            # print(principal_action[1])
            # print(agents[selectee].action[1])
            # print(winner.action[1])
            print("####################")
            q_avg[selectee] += q
            done_prev = done

        for i in range(n_agents):
            if len(bid_episode) > 0:
                real_bid_history[i].append(bid_episode[i])
                real_try_history[i].append(try_episode[i])
                real_quality_history[i].append(quality_episode[i])

        # if (episode + 1) % principal_update_target_freq == 0:
        #     for i in range(n_agents):
        #         agents[i].target.load_state_dict(agents[i].policy.state_dict())
        #     principal.target.load_state_dict(principal.policy.state_dict())
        #     for i in range(n_types):
        #         winner[i].target.load_state_dict(winner[i].policy.state_dict())
        for i in range(n_types):
            if len(trajectory_winner[i]) > 0:
                winner[i].update(trajectory_winner[i])

        for i in range(n_agents):
            agents[i].update(trajectory_agents[i])

        for i in range(n_agents):
            agents[i].reward_history.append(agents[i].reward)
        # winner[i].reward_history.append(winner[i].reward)
        principal.reward_history.append(principal_reward)
        
        for i in range(n_agents):
            if count_win[i] > 0:
                agents_b_history[i].append(agents_b[i] / (count_win[i]))
                delivered_q_history[i].append(q_avg[i] / count_win[i])
                try_history[i].append(try_avg[i] / count_win[i])

        if episode +1 > 10:
            print('----------------------------------------------------------')
            for i in range(n_agents):
                print("episode {}, agent {} reward is: {}".format(episode+1, i+1, agents[i].reward))
                # print("episode {}, agent {} reward is: {}".format(episode+1, i+1, winner[type_dict[i]].reward))
            print("episode {}, principal reward is {}".format(episode+1, principal_reward))
            # print("episode {}, winner reward is {}".format(episode+1, winner.reward))
            for i in range(n_agents):
                print("episode {}, agent 1 average b is {}".format(episode+1, agents_b_history[i][-1]))
            # for i in range(n_types):
            #     print("episode {}, delivered q average by winner type {} is {}".format(episode+1, i, delivered_q_history[i][-1]))
            for i in range(n_types):
                print("episode {}, max try average for winner type {} is {}".format(episode+1, i, try_history[i][-1]))
            print('----------------------------------------------------------')

        if (episode+1) > window_size:
            for i in range(n_agents):
                agents[i].average_reward.append(np.mean(agents[i].reward_history[-window_size:]))
            for i in range(n_types):
                winner[i].average_reward.append(np.mean(winner[i].reward_history[-window_size:]))
            principal.average_reward.append(np.mean(principal.reward_history[-window_size:]))
            for i in range(n_agents):
                agents_b_avg[i].append(np.mean(agents_b_history[i][-window_size:]))
            for i in range(n_agents):
                deliverd_q_avg[i].append(np.mean(delivered_q_history[i][-window_size:]))
                try_history_avg[i].append(np.mean(try_history[i][-window_size:]))
        

            print("\n")
            print('*********************************')
            for i in range(n_agents):
                print("episode {}, agent {} average reward is: {}".format(episode+1, i+1, agents[i].average_reward[-1]))
            print("episode {}, principal average reward is {}:".format(episode+1, principal.average_reward[-1]))
            for i in range(n_agents):
                print("episode {}, agent {} average b is {}:".format(episode+1, i+1, agents_b_avg[i][-1]))
            for i in range(n_agents):
                print("episode {}, delivered q average by winner {} is {}:".format(episode+1, i+1, deliverd_q_avg[i][-1]))
            for i in range(n_agents):
                print("episode {}, max try average by winner {} is {}:".format(episode+1, i+1, try_history_avg[i][-1]))
            print('*********************************')

        if (episode+1) % write_output_every == 0:
            plt.title('reward history')
            for i in range(n_agents):
                plt.plot(agents[i].reward_history, label = 'agent'+str(i+1))
                # plt.plot(winner[i].reward_history, label = 'winner')
            plt.plot(principal.reward_history, label = 'principal')
            # plt.plot(winner.reward_history, label = 'winner')
            plt.savefig(path_to_directory+"rewad_history.png", dpi = 300)
            plt.legend()

            plt.figure()
            plt.title('average reward history')
            for i in range(n_agents):
                plt.plot(agents[i].average_reward, label = 'agent'+str(i+1))
                # plt.plot(winner[i].average_reward, label = 'winner')
            plt.plot(principal.average_reward, label = 'principal')
            # plt.plot(winner.average_reward, label = 'winner')
            plt.savefig(path_to_directory+"avg_rewad_history.png", dpi = 300)
            plt.legend()

            for i in range(n_agents):
                plt.figure()
                plt.title('average b for agent '+str(i+1))
                plt.plot(agents_b_avg[i], label = 'agent ' + str(i+1)+ 'b average')
                plt.plot(deliverd_q_avg[i], label = 'delivered q average')
                plt.savefig(path_to_directory+"avg_b_"+str(i+1)+'.png', dpi = 300)
                plt.legend()

            for i in range(n_agents):
                plt.figure()
                plt.title('max try average by agent '+str(i+1))
                plt.plot(try_history_avg[i], label = 'max try average')
                plt.savefig(path_to_directory+'max_try_'+str(i+1)+'.png', dpi = 300)
                plt.legend()

            with open(path_to_directory+'bid.out', 'wb') as f:
                pickle.dump(real_bid_history, f)

            with open(path_to_directory + 'try.out', 'wb') as f:
                pickle.dump(real_try_history, f)

            with open(path_to_directory + 'qual.out', 'wb') as f:
                pickle.dump(real_quality_history, f)

            with open(path_to_directory + 'principal_reward.out', 'wb') as f:
                pickle.dump(principal.reward_history, f)

            for i in range(n_agents):
                with open(path_to_directory + "agent_" + str(i+1) + "_reward.out", "wb") as f:
                    pickle.dump(agents[i].reward_history, f)
            plt.show()


