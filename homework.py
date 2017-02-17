'''
s: state of the environment
a: current action
r: reward for performing action a in state s
sPrime: next state
aPrime: next action
gamma: discount rate
alpha: learning rate
epsilon: % random actions
'''
import numpy as np
import matplotlib.pyplot as plt
from grid import Grid
plt.ion()

#feel free to modify these variables!
control = False
onpolicy = False
epsilon = .2
gamma = .9
alpha = .1
num_steps = 100000
refresh = 1000

env = Grid()
Q = np.random.rand(25,4)/100
s = env.reset()
a = np.random.randint(4)
visits = np.zeros((25,))
goals = 0.0
lava = 0.0
for i in range(num_steps):
    #------main loop start----------
    sPrime,r,term = env.step(a)
    if term:
        sPrime = env.reset()
    if not control or np.random.rand() < epsilon:
        aPrime = np.random.randint(4)
    else:
        aPrime = Q[sPrime].argmax()
    #value function update here
    if onpolicy:
        target = r + (not term)*gamma*Q[sPrime,aPrime]
    else:
        target = r + (not term)*gamma*np.max(Q[sPrime])
    Q[s,a] = (1-alpha)*Q[s,a]+alpha*target

    s = sPrime
    a = aPrime
    #-----main loop end-----------
    #this is all to draw pretty pictures -- you can ignore everything below here
    visits[s]+=1
    if r > 0:
        goals+=1
    elif r < 0:
        lava+=1
    if i % refresh == 0:
        print('total steps: ',i,' avg steps per reward: ',refresh//(goals+1e-6),'avg steps per death: ', refresh//(lava+1e-6))
        goals = 0.0
        lava = 0.0
        plt.figure(1)
        plt.clf()

        ax = plt.subplot(121)
        ax.set_title('max values')
        plt.imshow(Q.max(1).reshape([5,5]))

        ax = plt.subplot(122)
        ax.set_title('state visits')
        plt.imshow(visits.reshape([5,5]))

        plt.figure(2)
        plt.clf()
        
        ax = plt.subplot(332)
        ax.set_title('up values')
        plt.imshow(Q[:,1].reshape([5,5]))
        plt.clim(Q.min(),Q.max())

        ax = plt.subplot(334)
        ax.set_title('left values')
        plt.imshow(Q[:,2].reshape([5,5]))
        plt.clim(Q.min(),Q.max())

        ax = plt.subplot(336)
        ax.set_title('right values')
        plt.imshow(Q[:,0].reshape([5,5]))
        plt.clim(Q.min(),Q.max())
        
        ax = plt.subplot(338)
        ax.set_title('down values')
        plt.imshow(Q[:,3].reshape([5,5]))
        plt.clim(Q.min(),Q.max())
        input('Press Enter to Continue...')
        plt.pause(.001)
        visits[:] = 0.0
plt.ioff()
plt.show()
