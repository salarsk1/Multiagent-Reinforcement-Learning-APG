import torch
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import numpy as np
from ._utils import *
from ._environment import *
import torch.nn.functional as F
import random
from ._model import *
from collections import deque
import math
import gym
from torch.distributions import Categorical


__all__ = ["DDPGAgent", "DQNAgent", "A2CAgent"]


class DQNAgent(object):
    def __init__(self, input_size, hidden_layers, output_size, learning_rate = 0.001, 
                gamma=0.99, epsilon_decay=0.01, min_eps = 0.01, target_update_freq = 32,
                max_memory_size = 30_000):

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_actions = output_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.min_eps = min_eps
        self.target_update_freq = target_update_freq
        self.max_memory_size = max_memory_size
        self.replay = ReplayBuffer(self.max_memory_size)
        self.policy = DQN(self.input_size, self.hidden_layers, self.num_actions)
        self.target = DQN(self.input_size, self.hidden_layers, self.num_actions)
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
        explore_rate = max(self.min_eps, self.epsilon_decay**episode)
        return explore_rate

    def get_action(self, state, episode):
        explore_rate = self.get_epsilon(episode)
        self.policy.eval()
        p = random.random()
        if p < explore_rate:
            # ret = torch.tensor(np.random.randint(self.output_size))
            # ret = torch.tensor(np.random.uniform(size=(1,self.output_size)))
            ret = torch.tensor(np.random.uniform(size=(1,self.num_actions)))

            return ret.argmax(dim=1).squeeze(0), ret
        else:
            with torch.no_grad():
                ret = self.policy(state).reshape(1,-1)
                return ret.argmax(dim=1).squeeze(0), ret

    def update(self, batch_size, episode):

        self.policy.train()
        self.target.eval()
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


class A2CAgent(object):
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=1e-4, 
                gamma=0.99):
        # Params
        # self.num_states = env.observation_space.shape[0]
        # self.num_actions = env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_states = input_size
        self.num_actions = output_size
        self.gamma = gamma

        # Networks
        self.actor = Actor(self.num_states, hidden_layers, self.num_actions)
        
        self.critic = Actor(self.num_states, hidden_layers, 1)
        
        # Training
        self.critic_criterion  = nn.MSELoss()
        self.a2c_optimizer  = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         lr=learning_rate)

        self.state = None
        self.action = None
        self.reward = None
        self.new_state = None

        self.reward_history = []
        self.average_reward = []

    def get_action(self, state, episode = None):
        # state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        # policy_dist = F.softmax(self.actor.forward(state), dim=1).detach().numpy()
        # print(policy_dist)
        # action = np.random.choice(self.num_actions, p = np.squeeze(policy_dist))
        # return action, policy_dist
        self.actor.eval()
        p = random.random()
        if p < 0.01:
            ret = torch.tensor(np.random.uniform(size=(1,self.num_actions)))
            return ret.argmax(dim=1).squeeze(0), ret


        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            logits = self.actor.forward(state)
        dist = F.softmax(logits, dim=0)
        # for i in range(len(dist)):
        #     if dist[i] < 1.e-10:
        #         dist[i] = 0.0
        #     if dist[i] > 1.0-1.e-10:
        #         dist[i] = dist[i] = 1.0
        probs = Categorical(dist)
        return probs.sample().cpu().detach().item(), dist


    def compute_loss(self, trajectory):
        states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        actions = torch.Tensor([sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
        next_states = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
        dones = torch.Tensor([sars[4] for sars in trajectory]).view(-1, 1).to(self.device)

        # compute discounted rewards
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))])\
             * rewards[j:]) for j in range(rewards.size(0))]
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)

        logits = self.actor.forward(states)
        values = self.critic.forward(states)
        dists = F.softmax(logits, dim=1)
        probs = Categorical(dists)

        # compute value loss
        value_loss = F.mse_loss(values, value_targets.detach())
        
        
        # compute entropy bonus
        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist+1.e-7)))
        entropy = torch.stack(entropy).sum()
        
        # compute policy loss
        advantage = value_targets - values
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        policy_loss = policy_loss.mean()
        
        total_loss = policy_loss + value_loss - 0.001 * entropy 
        return total_loss
        
    def update(self, trajectory):
        self.actor.train()
        loss = self.compute_loss(trajectory)

        self.a2c_optimizer.zero_grad()
        loss.backward()
        self.a2c_optimizer.step()

if __name__ == "__main__":
    import gym
    import sys
    import matplotlib.pyplot as plt

    env = gym.make("CartPole-v0")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    MAX_EPISODE = 1500
    MAX_STEPS = 500

    lr = 1e-4
    gamma = 0.99

    agent = A2CAgent(4, [256, 256, 128], 2, learning_rate = 1.e-4)

    def run():
        for episode in range(MAX_EPISODE):
            state = env.reset()
            trajectory = []
            episode_reward = 0
            for steps in range(MAX_STEPS):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                trajectory.append([state, action, reward, next_state, done])
                episode_reward += reward

                if done:
                    break
                state = next_state
            if episode % 10 == 0:
                print("Episode " + str(episode) + ": " + str(episode_reward))
            agent.update(trajectory)

    run()

    quit()


    env = NormalizedEnv(gym.make("Pendulum-v0"))
    env = gym.make("Pendulum-v0")
    agent = DDPGAgent(env)
    noise = OUNoise(env.action_space)
    batch_size = 128
    rewards = []
    avg_rewards = []

    for episode in range(100):
        state = env.reset()
        noise.reset()
        episode_reward = 0
        
        for step in range(500):
            action = agent.get_action(state)
            action = noise.get_action(action, step)
            new_state, reward, done, _ = env.step(action)
            print(new_state)
            quit()
            agent.replay.push(state, action, reward, new_state, done)
            
            if len(agent.replay) > batch_size:
                agent.update(batch_size)        
            
            state = new_state
            episode_reward += reward

            if done:
                print("episode: {}, reward: {}, average _reward: {} \n".format(episode, 
                        np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
                break

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()



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