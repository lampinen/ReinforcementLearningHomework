import numpy as np
class MountainCar(object):
    def __init__(self):
        self.min_x = -1.2
        self.max_x = .5
        self.min_v = -.07
        self.max_v = .07
        self.reset()
    def reset(self):
        self.x = np.random.rand()*(self.max_x-self.min_x)+self.min_x
        self.v = np.random.rand()*(self.max_v-self.min_v)+self.min_v
        return np.asarray([self.x,self.v])
    def step(self,action):
        self.v = self.v + .001*(action-1) - .0025*np.cos(3*self.x)
        self.v = max(self.min_v,min(self.v,self.max_v))
        self.x = self.x + self.v
        self.x = max(self.min_x,min(self.x,self.max_x))
        reward = 0.0
        term = False
        if self.x == self.max_x:
            reward = 1.0
            term = True
        return np.asarray([self.x,self.v]),reward,term 
if __name__ == "__main__":
    def policy(s):
        x = s[0]
        v = s[1]
        if v < 0 and x != -1.2:
            action = 0
        else:
            action = 2
        return action
    import matplotlib.pyplot as plt
    plt.ion()
    env = MountainCar()
    s = env.reset()
    for i in range(1000):
        #a = np.random.randint(3)
        a = policy(s)
        s,r,term = env.step(a)
        print(s)
        plt.clf()
        s_pos = np.linspace(env.min_x,env.max_x)
        plt.plot(s_pos,np.sin(3*s_pos)/3)
        plt.scatter(s[0],np.sin(3*s[0])/3)
        plt.pause(.01)
        if term:
            print('win')
            s = env.reset()

