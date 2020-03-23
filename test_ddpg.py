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

env=gym.make("Pendulum-v0")
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high
max_action = float(env.action_space.high[0])

discount_factor=0.99
episode_num=100000
lr_actor=0.001
lr_critic=0.001
batch_size=64
tau=0.005
var=0.1
max_step_num=2000
RENDER=False
#update_time=2
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class Actor_Net(nn.Module):
    def __init__(self,state_dim):
        super(Actor_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(400, 100)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(100, action_dim)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = torch.tanh(self.fc3(x))
        return output*max_action

class Critic_Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic_Net, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l1.weight.data.normal_(0, 0.1)
        self.l2 = nn.Linear(400, 300)
        self.l2.weight.data.normal_(0, 0.1)
        self.l3 = nn.Linear(300, 1)
        self.l3.weight.data.normal_(0, 0.1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Memory():
    def __init__(self):
        self.memory=[]
        self.memory_size=50000
        self.current_mem_count=0
    def sample(self):
        sample_index=np.random.choice(range(self.memory_size),batch_size)
        samples=np.array(self.memory)[sample_index,:]
        return samples
    def store_experience(self,transition):
        if self.current_mem_count<self.memory_size:
            self.memory.append(transition)
            self.current_mem_count+=1
        else:
            self.memory.pop(0)
            self.memory.append(transition)
class DDPG():
    def __init__(self):
        self.critic_eval_network,self.critic_target_network=Critic_Net(state_dim,action_dim),Critic_Net(state_dim,action_dim)
        self.critic_target_network.load_state_dict(self.critic_eval_network.state_dict())
        self.actor_u_network,self.actor_target_network=Actor_Net(state_dim),Actor_Net(state_dim)
        self.actor_target_network.load_state_dict(self.actor_u_network.state_dict())
        self.learning_step=0
        self.buffer=Memory()
        self.critic_loss_func=nn.MSELoss()
        self.critic_opt=optim.Adam(self.critic_eval_network.parameters(),lr=lr_critic)
        self.actor_opt=optim.Adam(self.actor_u_network.parameters(),lr=lr_actor)

    def choose_action(self,state):
        state=torch.FloatTensor(state)
        return self.actor_u_network(state).detach().numpy()



    def learning(self):
        #print("learningfirst")
        samples=np.array(self.buffer.sample())
        states_batch = torch.FloatTensor(np.stack(samples[:, 0], axis=0))
        action_batch = torch.FloatTensor(np.stack(samples[:, 1], axis=0).reshape(-1, 1))
        reward_batch = torch.FloatTensor(np.stack(samples[:, 2], axis=0).reshape(-1, 1))
        next_states_batch = torch.FloatTensor(np.stack(samples[:, 3], axis=0))
        done_mask_batch = torch.FloatTensor(np.invert(np.stack(samples[:, 4], axis=0)).reshape(-1, 1))  # 真正的done_batch取反
        # temp=self.actor_target_network(next_states_batch)
        # temp=self.critic_target_network(next_states_batch,temp)
       # print("learningsecond")
        y=reward_batch+discount_factor*self.critic_target_network(next_states_batch,self.actor_target_network(next_states_batch)).detach()
        q_current=self.critic_eval_network(states_batch,action_batch)

        critic_loss=self.critic_loss_func(y,q_current)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
       # print("learningthird")
        actor_loss=-torch.mean(self.critic_eval_network(states_batch,self.actor_u_network(states_batch)))
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
       # print("learningforth")
        for param, target_param in zip(self.critic_eval_network.parameters(), self.critic_target_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor_u_network.parameters(), self.actor_target_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        self.learning_step+=1
        # print("learning fifth")
        # print(self.learning_step)
ddpg=DDPG()
x_axis=np.arange(0,episode_num)
y_reward_axis=np.arange(0,episode_num)
#y_episode_length_axis=np.arange(0,episode_num)
for i in range(episode_num):
    state = env.reset()
    state = torch.from_numpy(np.asarray(state)).type('torch.FloatTensor')
    eps_length = 0
    eps_reward = 0
    for j in range(max_step_num):
        # if RENDER:
        #     env.render()
        action=ddpg.choose_action(state)
        #action = np.clip(np.random.normal(action, var), -2, 2)
        action = (action + np.random.normal(0, var, size=env.action_space.shape[0])).clip(
            env.action_space.low, env.action_space.high)
        next_state, reward, done, _ = env.step(action)
        experience=Transition( state, action, reward, next_state, done)

        ddpg.buffer.store_experience(experience)
        #print(1)
        if ddpg.buffer.current_mem_count>=ddpg.buffer.memory_size:
            #var*=0.9999
            ddpg.learning()

        state = next_state
        eps_reward+=reward
        state = torch.from_numpy(np.asarray(state)).type('torch.FloatTensor')
        if j == max_step_num - 1:
            print('Episode:', i, ' Reward: %i' % int(eps_reward), 'Explore: %.2f' % var, )
            # if eps_reward > -300:
            #     RENDER = True
            break
    y_reward_axis[i] = eps_reward

    print("episode %d reward:\n" % i, eps_reward)


plt.title("reward")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x_axis, y_reward_axis)
plt.show()
