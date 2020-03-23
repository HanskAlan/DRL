import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from collections import deque, namedtuple
import random
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")



env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

episode_num=20000

N_F = env.observation_space.shape[0]
N_A = env.action_space.n
L_A=0.001
lr=0.01
discount_factor=0.99
target_replace_iter=200
batch_size=32

np.random.seed(2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_F, 40)
        self.fc2 = nn.Linear(40, N_A)
       # self.fc3 = nn.Linear(10, N_A)

    def forward(self, x):
        x = F.relu(self.fc1(x))
       # x = F.relu(self.fc2(x))
        output = F.softmax(self.fc2(x), dim=0)
        return output
class DQN():
    def __init__(self):
        self.q_network,self.target_network=Net(),Net()
        self.learning_step_counter=0
        self.loss_func=nn.MSELoss()
        self.replay_memory=[]
        self.replay_memory_size=4000
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory_count=0


    def learning(self):
        if self.learning_step_counter%target_replace_iter==0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        samples_index = np.random.choice(range(self.replay_memory_size), batch_size)
        samples=np.array(self.replay_memory)[samples_index,:]


        states_batch=torch.FloatTensor(np.stack(samples[:,0],axis=0))
        #states_batch=[torch.FloatTensor(s) for s in states_batch]
        action_batch=torch.LongTensor(np.stack(samples[:,1],axis=0).reshape(-1,1))
        reward_batch=torch.FloatTensor(np.stack(samples[:,2],axis=0))
        next_states_batch=torch.FloatTensor(np.stack(samples[:,3],axis=0))
        #next_states_batch=np.array([torch.FloatTensor(s) for s in next_states_batch])
        #next_states_batch=torch.cat(next_states_batch,0)
        done_mask_batch=torch.FloatTensor(np.invert(np.stack(samples[:,4],axis=0)))#真正的done_batch取反



        q_next = self.target_network(next_states_batch).detach()
        q_eval_temp=self.q_network(states_batch)
        q_eval=q_eval_temp.gather(1,action_batch)
        temp=q_next.max(1)
        temp1=q_next.max(1)[0].reshape(batch_size,1)
        q_target=discount_factor*q_next.max(1)[0].reshape(batch_size,1)*done_mask_batch.reshape(batch_size,1)\
                 +reward_batch.reshape(batch_size,1)
        loss = self.loss_func(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learning_step_counter += 1

    def store_transaction(self,transaction):
        if self.memory_count<self.replay_memory_size:
            self.replay_memory.append(transaction)
            self.memory_count+=1
        else:
            self.replay_memory.pop(0)
            self.replay_memory.append(transaction)

dqn=DQN()
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

#方便绘图
x_axis=np.arange(0,episode_num)
y_reward_axis=np.arange(0,episode_num)
y_episode_length_axis=np.arange(0,episode_num)

for i in range(episode_num):
    state = env.reset()
    state = torch.from_numpy(np.asarray(state)).type('torch.FloatTensor')
    eps_length=0
    eps_reward=0
    while True:
        #env.render()
        with torch.no_grad():
            action_prob=dqn.q_network(state)
        sampler=Categorical(action_prob)
        action=sampler.sample().numpy()
        next_state, reward, done, _ = env.step(action)
        x, x_dot, theta, theta_dot = next_state

        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        eps_reward+=reward
        experience = Transition( state, action, reward, next_state, done)
        dqn.store_transaction(experience)

        if dqn.memory_count==dqn.replay_memory_size:
            dqn.learning()
        if done:
            break
        state=next_state
        eps_length+=1
        state = torch.from_numpy(np.asarray(state)).type('torch.FloatTensor')
    y_reward_axis[i]=eps_reward
    y_episode_length_axis[i]=eps_length
    print("episode %d reward:\n"%i, eps_reward)
    print("episode %d length:\n"%i, eps_length)

plt.title("reward")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x_axis,y_reward_axis)
plt.show()

plt.title("episode_length")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x_axis,y_episode_length_axis)
plt.show()





