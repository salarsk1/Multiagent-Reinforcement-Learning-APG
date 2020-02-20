import numpy as np
from ._environment import *
from collections import deque
import random
import matplotlib.pyplot as plt

__all__ = ["OUNoise", "ReplayBuffer", "RewardProcess", "CostNotCoveredReward", 
            "CostPlusReward", "ValueFunction", "CostFunction", "map_nn_output",
            "IncentiveCostNotCoveredReward", "IncentiveCostPlusReward",
            "RBValueFunction", "RPIValueFunction"]

class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

class RewardProcess(object):

    def __init__(self, value, reward_win, reward_lose, 
                principal_select_reward, principal_no_select_reward, 
                winner_prime_penalty, agent_fail_penalty, principal_fail_penalty,
                prepayment_percentage, cost_premimum):

        self.value                      = value
        self.reward_win                 = reward_win
        self.reward_lose                = reward_lose
        self.principal_select_reward    = principal_select_reward
        self.principal_no_select_reward = principal_no_select_reward
        self.winner_prime_penalty       = winner_prime_penalty
        self.agent_fail_penalty         = agent_fail_penalty
        self.principal_fail_penalty     = principal_fail_penalty
        self.prepayment_percentage      = prepayment_percentage
        self.cost_premimum              = cost_premimum

    def reward_winning(self, agent_bid):
        return self.prepayment_percentage * agent_bid

    def reward_losing(self):
        return self.reward_lose

    def principal_selecting_reward(self):
        return self.principal_select_reward

    def principal_not_selecting_reward(self):
        return self.principal_no_select_reward

    def winner_agent_reward(self):
        pass

    def principal_reward(self):
        pass

    def winner_prime_reward(self, q, agent_bid, agent_cost):

        if q < self.value.q_min:
            return self.reward_winning(agent_bid) - agent_cost + self.winner_prime_penalty
        else:
            return (1.0 - self.prepayment_percentage) * agent_bid - agent_cost


class CostNotCoveredReward(RewardProcess):

    def __init__(self, value, reward_win, reward_lose, 
                principal_select_reward, principal_no_select_reward, 
                winner_prime_penalty, agent_fail_penalty, principal_fail_penalty,
                prepayment_percentage, cost_premimum):
        super(CostNotCoveredReward, self).__init__(value, reward_win, 
                            reward_lose, principal_select_reward, principal_no_select_reward,
                            winner_prime_penalty, agent_fail_penalty, principal_fail_penalty,
                            prepayment_percentage, cost_premimum)

    def principal_reward(self, q, agent_bid, agent_cost = None):
        # if q < self.value.q_min:
        if q < self.value.market_min:
            return self.principal_fail_penalty
        else:
            return self.value(q) - agent_bid

    def winner_agent_reward(self, q, agent_bid, agent_cost):
        if q < self.value.q_min:
            return self.agent_fail_penalty
        return (1.0 - self.prepayment_percentage) * agent_bid - agent_cost


class CostPlusReward(RewardProcess):

    def __init__(self, value, reward_win, reward_lose, 
                principal_select_reward, principal_no_select_reward, 
                winner_prime_penalty, agent_fail_penalty, principal_fail_penalty,
                prepayment_percentage, cost_premimum):
        super(CostPlusReward, self).__init__(value, reward_win, 
                            reward_lose, principal_select_reward, principal_no_select_reward,
                            winner_prime_penalty, agent_fail_penalty, principal_fail_penalty,
                            prepayment_percentage, cost_premimum)

    def principal_reward(self, q, agent_bid, agent_cost):
        # if q < self.value.q_min:
        if q < self.value.market_min:
            return self.principal_fail_penalty
        else:
            return self.value(q) - agent_bid - agent_cost

    def winner_agent_reward(self, q, agent_bid, agent_cost = None):
        if q < self.value.q_min:
            return self.agent_fail_penalty
        return (1.0 - self.prepayment_percentage) * agent_bid - max(0.0, agent_cost - self.cost_premimum)

    def winner_prime_reward(self, q, agent_bid, agent_cost):
        if q < self.value.q_min:
            return self.reward_winning(agent_bid) - agent_cost + self.winner_prime_penalty
        else:
            return (1.0 - self.prepayment_percentage) * agent_bid - agent_cost

class IncentiveCostNotCoveredReward(CostNotCoveredReward):

    def __init__(self, value, reward_win, reward_lose, 
                principal_select_reward, principal_no_select_reward, 
                winner_prime_penalty, agent_fail_penalty, principal_fail_penalty,
                prepayment_percentage, cost_premimum):
        super(IncentiveCostNotCoveredReward, self).__init__(value, reward_win, reward_lose, 
                    principal_select_reward, principal_no_select_reward, 
                    winner_prime_penalty, agent_fail_penalty, principal_fail_penalty,
                    prepayment_percentage, cost_premimum)

    def winner_agent_reward(self, q, agent_bid, agent_cost):
        if q < self.value.q_min:
            return self.agent_fail_penalty
        return (1.0 - self.prepayment_percentage) * agent_bid - agent_cost + (q - self.value.q_min)

    def winner_prime_reward(self, q, agent_bid, agent_cost):
        if q < self.value.q_min:
            return self.reward_winning(agent_bid) - agent_cost + self.winner_prime_penalty
        else:
            return (q - self.value.q_min) + (1.0 - self.prepayment_percentage) * agent_bid - agent_cost

class IncentiveCostPlusReward(IncentiveCostNotCoveredReward):

    def __init__(self, value, reward_win, reward_lose, 
                principal_select_reward, principal_no_select_reward, 
                winner_prime_penalty, agent_fail_penalty, principal_fail_penalty,
                prepayment_percentage, cost_premimum):
        super(IncentiveCostPlusReward, self).__init__(value, reward_win, reward_lose, 
                principal_select_reward, principal_no_select_reward, 
                winner_prime_penalty, agent_fail_penalty, principal_fail_penalty,
                prepayment_percentage, cost_premimum)

    def winner_agent_reward(self, q, agent_bid, agent_cost = None):
        if q < self.value.q_min:
            return self.agent_fail_penalty
        else:
            return (1.0 - self.prepayment_percentage) * agent_bid + (q - self.value.q_min) - max(0.0, agent_cost - self.cost_premimum)
    def winner_prime_reward(self, q, agent_bid, agent_cost):
        if q < self.value.q_min:
            return self.reward_winning(agent_cost) - max(0.0, agent_cost-self.cost_premimum) + self.winner_prime_penalty
        else:
            return (q - self.value.q_min) + (1.0 - self.prepayment_percentage) * agent_bid - max(0.0, agent_cost-self.cost_premimum)

class ValueFunction(object):

    def __init__(self, market_min, market_worth, c):
        self.c = c
        self.market_min = market_min
        self.market_worth = market_worth

class RPIValueFunction(ValueFunction):

    def __init__(self, market_min, market_worth, c, q_min = None):
        super(RPIValueFunction, self).__init__(market_min, market_worth, c)
        self.q_min = q_min

    def __call__(self, q):
        if q < self.market_min:
            return 0
        else:
            return self.market_worth + self.c * (q - self.market_min)

class RBValueFunction(ValueFunction):

    def __init__(self, market_min, market_worth, q_min=None):
        super(RBValueFunction, self).__init__(market_min, market_worth, 0)
        self.q_min = q_min

    def __call__(self, q):
        if q < self.market_min:
            return 0
        else:
            return self.market_worth

class CostFunction(object):
    
    def __init__(self, c):
        self.c = c

    def __call__(self, e):
        return self.c * e

def map_nn_output(output, n_discret, min_val, max_val):
    step = (max_val - min_val) / (n_discret - 1)
    return min_val + output * step

if __name__ == "__main__":
    
    value = RPIValueFunction(2.0, 10.0, 1.)
    cost = CostFunction(0.05)

    h = []
    x = np.linspace(0.0, 3.0, 1000)
    for i in x: 
        h.append(value(i))
    plt.plot(x, h)
    plt.show()
    



