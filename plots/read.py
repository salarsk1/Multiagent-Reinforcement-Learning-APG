import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
with open('0.01.out', 'rb') as f:
    z = pickle.load(f)
a = []
for i in range(len(z)):
    for j in range(len(z[i])):
        if len(z[i][j]) == 40:
            a.append(z[i][j])

        # h.append(a[i][18])
# plt.hist(h)
# plt.show()

if __name__ == "__main__":
    # with open('try.out', 'rb') as f:
    #     z = pickle.load(f)

    # agent1_b_mean = [[] for _ in range(len(z))]
    # agent1_b_up   = [[] for _ in range(len(z))]
    # agent1_b_down = [[] for _ in range(len(z))]
    # data = []
    # # print(len(z[0::2]))
    # print(len(z))
    # for k in range(len(z)):
    #     print(k)
    #     for i in z[k]:
    #         if len(i) > 0:
    #             agent1_b_mean[k].append(np.mean(i))
    #             agent1_b_up[k].append(np.percentile(i, 97.5))
    #             agent1_b_down[k].append(np.percentile(i, 2.5))

    #     df_m = pd.DataFrame(agent1_b_mean[k])
    #     df_up = pd.DataFrame(agent1_b_up[k])
    #     df_down = pd.DataFrame(agent1_b_down[k])
    #     roll_m = df_m.rolling(200).mean()
    #     roll_up = df_up.rolling(200).mean()
    #     roll_down = df_down.rolling(200).mean()
    #     plt.plot(roll_m)
    # plt.fill_between(list(np.arange(len(agent1_b_down))),roll_down.to_numpy().flatten(), roll_up.to_numpy().flatten(), alpha=0.5)
    # plt.hist(data[-10000:], bins=20)
    # plt.show()

    for k in range(2):
        with open('agent_'+str(k+1)+'_reward.out', 'rb') as f:
            z = pickle.load(f)
        df_m = pd.DataFrame(z)
        plt.plot(df_m.rolling(200).mean(), label='agent'+str(k+1)+'reward')
    with open('principal_reward.out', 'rb') as f:
        z = pickle.load(f)
    df_m = pd.DataFrame(z)
    plt.plot(df_m.rolling(200).mean(), label = 'principal reward')
    plt.legend()
    plt.show()

