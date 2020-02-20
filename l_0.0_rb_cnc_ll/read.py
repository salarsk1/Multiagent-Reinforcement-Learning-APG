import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
# with open('0.02.out', 'rb') as f:
#     z = pickle.load(f)
# a = []
# for i in range(len(z)):
#     for j in range(len(z[i])):
#         a.append(z[i][j])
# M = []
# c = 0
# h = []

        # h.append(a[i][18])
# plt.hist(h)
# plt.show()

if __name__ == "__main__":
    with open('bid.out', 'rb') as f:
        z = pickle.load(f)
    agents_b_mean = [[] for _ in range(len(z))]
    agents_b_up   = [[] for _ in range(len(z))]
    agents_b_down = [[] for _ in range(len(z))]
    # print(len(z[0::2]))

    for k in range(len(z)):
        for i in z[k]:
            if len(i) > 0:
                agents_b_mean[k].append(np.mean(i))
                agents_b_up[k].append(np.percentile(i, 97.5))
                agents_b_down[k].append(np.percentile(i, 2.5))

    for i in range(len(z)):
        df_m = pd.DataFrame(agents_b_mean[i])
        df_up = pd.DataFrame(agents_b_up[i])
        df_down = pd.DataFrame(agents_b_down[i])
        roll_m = df_m.rolling(100).mean()
        print("agent {} average bid is {}".format(i+1, roll_m.iloc[-100:]))

    with open('qual.out', 'rb') as f:
        z = pickle.load(f)
    agents_b_mean = [[] for _ in range(len(z))]
    agents_b_up   = [[] for _ in range(len(z))]
    agents_b_down = [[] for _ in range(len(z))]
    # print(len(z[0::2]))

    for k in range(len(z)):
        for i in z[k]:
            if len(i) > 0:
                agents_b_mean[k].append(np.mean(i))
                agents_b_up[k].append(np.percentile(i, 97.5))
                agents_b_down[k].append(np.percentile(i, 2.5))

    for i in range(len(z)):
        df_m = pd.DataFrame(agents_b_mean[i])
        df_up = pd.DataFrame(agents_b_up[i])
        df_down = pd.DataFrame(agents_b_down[i])
        roll_m = df_m.rolling(100).mean()
        print("agent {} delivered quality is {}".format(i+1, roll_m.iloc[-100:]))

    with open('try.out', 'rb') as f:
        z = pickle.load(f)
    agents_b_mean = [[] for _ in range(len(z))]
    agents_b_up   = [[] for _ in range(len(z))]
    agents_b_down = [[] for _ in range(len(z))]
    # print(len(z[0::2]))

    for k in range(len(z)):
        for i in z[k]:
            if len(i) > 0:
                agents_b_mean[k].append(np.mean(i))
                agents_b_up[k].append(np.percentile(i, 97.5))
                agents_b_down[k].append(np.percentile(i, 2.5))

    for i in range(len(z)):
        df_m = pd.DataFrame(agents_b_mean[i])
        df_up = pd.DataFrame(agents_b_up[i])
        df_down = pd.DataFrame(agents_b_down[i])
        roll_m = df_m.rolling(100).mean()
        print("agent {} effort level is {}".format(i+1, roll_m.iloc[-100:]))

        # roll_up = df_up.rolling(100).mean()
        # roll_down = df_down.rolling(100).mean()
        # plt.plot(roll_m, label = "agent "+str(i+1)+"bid")
    #     plt.fill_between(list(np.arange(len(agents_b_down[i]))),roll_down.to_numpy().flatten(), roll_up.to_numpy().flatten(), alpha=0.5)
    # plt.show()

