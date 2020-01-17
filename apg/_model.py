import torch
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from _utils import *
from _environment import *

__all__ = ["Critic", "Actor", "DQN"]

class Critic(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        
        super(Critic, self).__init__()
        
        self.hidden_layers = hidden_layers

        self.linear = nn.ModuleList()
        
        self.linear.append(nn.Linear(input_size, self.hidden_layers[0]))
        
        for i in range(1, len(self.hidden_layers)):
            self.linear.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))

        self.linear.append(nn.Linear(self.hidden_layers[-1], output_size))

    def forward(self, state, action):
        """
        state and action parameters are torch tensors
        """
        x = torch.cat([state, action], 1)

        for layer in range(len(self.linear)-1):
            x = F.relu(self.linear[layer](x))

        x = self.linear[-1](x)

        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, learning_rate = 3e-4):
        
        super(Actor, self).__init__()

        self.hidden_layers = hidden_layers

        self.linear = nn.ModuleList()
        
        self.linear.append(nn.Linear(input_size, self.hidden_layers[0]))
        
        for i in range(1, len(self.hidden_layers)):
            self.linear.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))

        self.linear.append(nn.Linear(self.hidden_layers[-1], output_size))
        
    def forward(self, x):
        """
        Param state is a torch tensor
        """
        for layer in range(len(self.linear)-1):
            x = F.relu(self.linear[layer](x))

        x = 3.0 * F.sigmoid(self.linear[-1](x))
        return x

class DQN(nn.Module):

    def __init__(self, input_size, hidden_layers, output_size):
        
        super(DQN, self).__init__()
        
        self.input_size = input_size
        
        self.hidden_layers = hidden_layers

        self.output_size = output_size

        self.linear = nn.ModuleList()

        self.linear.append(nn.Linear(self.input_size, self.hidden_layers[0]))
        
        for i in range(1, len(self.hidden_layers)):
            self.linear.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))

        self.linear.append(nn.Linear(self.hidden_layers[-1], self.output_size))

    def forward(self, state):

        state = torch.FloatTensor(state)
        for layer in range(len(self.linear)-1):
            state = F.tanh(self.linear[layer](state))

        # return F.softmax(self.linear[-1](state), dim=1)
        return self.linear[-1](state)

if __name__ == "__main__":

    dqn = DQN(2)
    print(dqn(torch.randn(1,4)))

    

