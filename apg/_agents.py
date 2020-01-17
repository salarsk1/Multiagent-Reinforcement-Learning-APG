import torch
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import numpy as np
from _utils import *
from _environment import *
import torch.nn.functional as F
import random
from _model import *
from collections import deque
import math
import gym


__all__ = ["DDPGAgent", "DQNAgent"]


class DDPGAgent(object):
    def __init__(self, input_size, hidden_layers, output_size, actor_learning_rate=1e-4, 
                critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, 
                max_memory_size=30_000):
        # Params
        # self.num_states = env.observation_space.shape[0]
        # self.num_actions = env.action_space.shape[0]
        self.num_states = input_size
        self.num_actions = output_size
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(self.num_states, hidden_layers, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_layers, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_layers, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_layers, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.replay = ReplayBuffer(max_memory_size)
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.state = None
        self.action = None
        self.reward = None
        self.new_state = None

        self.reward_history = []
        self.average_reward = []
    
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0,0]
        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.replay.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

class DQNAgent(object):
    def __init__(self, input_size, hidden_layers, output_size, learning_rate = 0.001, 
                gamma=0.99, epsilon_decay=0.01, min_eps = 0.01, target_update_freq = 32,
                max_memory_size = 30_000):

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.min_eps = min_eps
        self.target_update_freq = target_update_freq
        self.max_memory_size = max_memory_size
        self.replay = ReplayBuffer(self.max_memory_size)
        self.policy = DQN(self.input_size, self.hidden_layers, self.output_size)
        self.target = DQN(self.input_size, self.hidden_layers, self.output_size)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = learning_rate)

        self.reward_history = []
        self.average_reward = []

        self.state = None
        self.action = None
        self.reward = None
        self.new_state = None

    def get_epsilon(self, episode):
        # explore_rate = self.min_eps + (1.0 - self.min_eps) * \
        #         math.exp(-episode * self.epsilon_decay)
        explore_rate = self.min_eps + self.epsilon_decay**episode
        return explore_rate


    def get_action(self, state, episode):
        explore_rate = self.get_epsilon(episode)
        p = random.random()
        if p < explore_rate:
            # ret = torch.tensor(np.random.randint(self.output_size))
            # ret = torch.tensor(np.random.uniform(size=(1,self.output_size)))
            ret = torch.tensor(np.random.uniform(size=(1,self.output_size)))

            return ret.argmax(dim=1).squeeze(0), ret
        else:
            with torch.no_grad():
                # ret = self.policy(state).argmax(dim = 1).squeeze(0)
                ret = self.policy(state.reshape(1, -1))
                return ret.argmax(dim = 1).squeeze(0), ret

    def update(self, batch_size, episode):

        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)

        current_qs = self.policy(torch.FloatTensor(states)).gather(dim=1, 
                                index = torch.tensor(actions).unsqueeze(-1))

        next_q_values = self.target(torch.FloatTensor(next_states)).gather(dim=1, 
                                    index = torch.tensor(actions).unsqueeze(-1))

        for i in range(batch_size):
            if dones[i] == True:
                next_q_values[i] = 0

        target_qs = torch.FloatTensor(rewards).unsqueeze(dim=1) + self.gamma * next_q_values

        loss = F.mse_loss(current_qs, target_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class A2C(object):
    def __init__(self, input_size, hidden_layers, output_size, actor_learning_rate=1e-4, 
                critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, 
                max_memory_size=30_000):
        # Params
        # self.num_states = env.observation_space.shape[0]
        # self.num_actions = env.action_space.shape[0]
        self.num_states = input_size
        self.num_actions = output_size
        self.gamma = gamma

        # Networks
        self.actor = Actor(self.num_states, hidden_layers, self.num_actions)
        
        self.critic = Critic(self.num_states + self.num_actions, hidden_layers, self.num_actions)
        
        
        # Training
        self.replay = ReplayBuffer(max_memory_size)
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.state = None
        self.action = None
        self.reward = None
        self.new_state = None

        self.reward_history = []
        self.average_reward = []











if __name__ == "__main__":
    import gym
    import sys
    import matplotlib.pyplot as plt
    env1 = Env(0.05, 2)
    agent1 = DDPGAgent(env1)


    # env = NormalizedEnv(gym.make("Pendulum-v0"))
    # env = gym.make("Pendulum-v0")
    # agent = DDPGAgent(env)
    # noise = OUNoise(env.action_space)
    # batch_size = 128
    # rewards = []
    # avg_rewards = []

    # for episode in range(100):
    #     state = env.reset()
    #     noise.reset()
    #     episode_reward = 0
        
    #     for step in range(500):
    #         action = agent.get_action(state)
    #         action = noise.get_action(action, step)
    #         new_state, reward, done, _ = env.step(action)
    #         print(new_state)
    #         quit()
    #         agent.replay.push(state, action, reward, new_state, done)
            
    #         if len(agent.replay) > batch_size:
    #             agent.update(batch_size)        
            
    #         state = new_state
    #         episode_reward += reward

    #         if done:
    #             print("episode: {}, reward: {}, average _reward: {} \n".format(episode, 
    #                     np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
    #             break

    #     rewards.append(episode_reward)
    #     avg_rewards.append(np.mean(rewards[-10:]))

    # plt.plot(rewards)
    # plt.plot(avg_rewards)
    # plt.plot()
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.show()



#The following is to test the DQN agent 

    env = gym.make("CartPole-v0")
    agent = DQNAgent(env.observation_space.shape[0], 200, 200, 
                    env.action_space.n, learning_rate=0.01)

    total_reward = []
    
    episode = 0

    mean_reward = deque(maxlen=20000)
    
    window = 20
    
    num_episodes = 500
    
    num_max_steps = 2000
    
    batch_size = 4

    for episode in range(num_episodes):
        tot_rew = 0
        state = env.reset()
        done  = False
        count_steps = 0
        
        while not done and count_steps < num_max_steps:
            action = agent.get_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0), episode)[0]

            new_state, reward, done, info = env.step(action.item())
            
            tot_rew += reward
            
            agent.replay.push(state, action, reward, new_state, done)
            
            state = new_state
            
            if len(agent.replay) >= batch_size:
                agent.update(batch_size, episode+1)
            
            count_steps += 1
        
        total_reward.append(tot_rew)
        
        if (episode+1)%100 == 0:    
            print(episode+1)
            print("episode: ", episode + 1, "mean reward: ", np.mean(total_reward[-100:]))
        # count += 1
        if (episode + 1) % agent.target_update_freq == 0:
            print(episode+1)
            agent.target.load_state_dict(agent.policy.state_dict())

    print(total_reward)

    env.close()