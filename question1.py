import numpy as np
import matplotlib.pyplot as plt
from grid import Grid

env = Grid()
num_steps = 100
Q = np.zeros((25,4))
gamma = .9
'''
s: state of the environment
a: current action
r: reward for performing action a in state s
sPrime: next state
aPrime: next action
'''
s = env.reset()
a = np.random.randint(4)
for i in range(num_steps):
    sPrime,r,term = env.step(a)
    if term:
        sPrime = env.reset()
    aPrime = np.random.randint(4)
    #value function update here
    Q[s,a] = r + (not term)*gamma*Q[sPrime,aPrime]
    Q[s,a] = r + (not term)*gamma*np.max(Q[sPrime])
    s = sPrime
    a = aPrime



