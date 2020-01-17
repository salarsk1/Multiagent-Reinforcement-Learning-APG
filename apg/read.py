import pickle
import numpy as np
import time
with open('0.005.out', 'rb') as f:
    z = pickle.load(f)
a = []
for i in range(len(z)):
    for j in range(len(z[i])):
        a.append(z[i][j])
M = []
c = 0
h = []
for i in range(0,100000):
    if len(a[i]) != 40:
        print(i)

