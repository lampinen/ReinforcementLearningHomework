import numpy as np
import time
import tensorflow as tf
def linear(in_,out_dim,name,activation_fn=None,reuse=False):
    in_dim = in_.get_shape()[1]
    with tf.variable_scope(name,reuse=reuse):
        W = tf.get_variable('W',[in_dim,out_dim],tf.float32)
        b = tf.get_variable('b',[out_dim],tf.float32)
        out = tf.matmul(in_,W)+b
        if activation_fn != None:
            out = activation_fn(out)
    return out
def basic_encoder(in_,hid_dim,out_dim,scope='encoder',reuse=False,layers=1):
    with tf.variable_scope(scope,reuse=reuse):
        hid = in_
        for i in range(layers):
            hid = linear(hid,hid_dim,'hid'+str(i),tf.nn.relu)
        last_hid = linear(hid,out_dim,'hid'+str(layers))
    return last_hid
class DQN:
    def __init__(self,env):
        self.n_actions = env.n_actions
        self.s_dim = env.s_dim
        self.env = env
        self.replay_dim = int(1e6)
        self.burn_in = int(1e4)
        self.epsilon_end = .1
        self.anneal_frac = .2
        self.gamma = .99
        self.lr = 1e-4
        self.mb_dim = 300
        self.hid_dim = 1000
        self.layers = 1
        self.build_graph()
        self.setup_session()
    def build_graph(self):
        self.s = tf.placeholder(tf.float32,shape=(None,self.s_dim))
        self.a = tf.placeholder(tf.int64,shape=(None,))
        self.r = tf.placeholder(tf.float32,shape=(None,))
        self.nt = tf.placeholder(tf.float32,shape=(None,))
        self.sPrime = tf.placeholder(tf.float32,shape=(None,self.s_dim))
        self.q = basic_encoder(self.s,self.hid_dim,self.n_actions,layers = self.layers)
        selected_q = tf.reduce_sum(self.q*tf.one_hot(self.a,self.n_actions,1.0,0.0),-1)
        '''loss'''
        q_prime = tf.stop_gradient(tf.reduce_max(basic_encoder(self.sPrime,self.hid_dim,self.n_actions,reuse=True,layers=self.layers),1))
        self.loss = tf.reduce_mean(tf.square(selected_q - (self.r+ self.gamma*self.nt*q_prime)))
        tf.summary.scalar('net loss',self.loss)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    def setup_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
    def eval(self,steps=100000,timeout = 500):
        s = self.env.reset()
        cum_reward = 0.0
        time = 0
        for i in range(steps):
            a = np.argmax(self.sess.run(self.q,feed_dict={self.s:[s]})[0])
            sPrime,r,term = self.env.step(a)
            time+=1
            cum_reward += r
            if term or time >= timeout:
                s = self.env.reset()
                time = 0
            else:
                s = sPrime
        return cum_reward/steps
    def train(self,steps=1000000,refresh=10000,timeout = 1000):
        cur_time = time.clock()
        avg_loss = 0
        avg_reward = 0
        rho = .99
        S = np.zeros((self.replay_dim,self.s_dim))
        A = np.zeros((self.replay_dim),dtype=np.int64)
        R = np.zeros((self.replay_dim,))
        NT = np.zeros((self.replay_dim,))
        SPrime = np.zeros((self.replay_dim,self.s_dim))

        replay_ind = 0
        epsilon = np.linspace(1,self.epsilon_end,int(steps*self.anneal_frac))
        S[replay_ind] = self.env.reset()
        episode_steps = 0
        for i in range(steps):
            if np.random.rand() < epsilon[min(i,len(epsilon)-1)]:
                A[replay_ind] = np.random.randint(self.n_actions)
            else:
                A[replay_ind] = np.argmax(self.sess.run(self.q,feed_dict={self.s:[S[replay_ind]]})[0])
            SPrime[replay_ind],R[replay_ind],term = self.env.step(A[replay_ind])
            if i < steps-1:
                if term or episode_steps > timeout:
                    S[replay_ind+1] = self.env.reset()
                    NT[replay_ind] = 0
                    episode_steps = 0
                else:
                    S[replay_ind+1] = SPrime[replay_ind].copy()
                    NT[replay_ind] = 1
            avg_reward = rho*avg_reward + (1-rho)*R[replay_ind]
            replay_ind = (replay_ind + 1) % self.replay_dim
            episode_steps+=1
            if i > self.burn_in:
                samples = np.random.randint(max(i,self.replay_dim),size=self.mb_dim)
                _,loss = self.sess.run([self.train_step,self.loss],feed_dict={self.s:S[samples],self.a:A[samples],self.r:R[samples],self.sPrime:SPrime[samples],self.nt:NT[samples]})
                avg_loss = rho*avg_loss + (1-rho)*loss
                if i % refresh == 0:
                    print('iter: ',i,' loss: ',avg_loss,' cur epsilon: ',epsilon[min(i,len(epsilon)-1)],' avg reward per timestep: ',avg_reward,' time: ',time.clock()-cur_time)
                    cur_time = time.clock()
                    loss_hist = []


if __name__ == "__main__":
    import mountain_car
    e = mountain_car.MountainCar()
    dqn = DQN(e)
    dqn.train()
