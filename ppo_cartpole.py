from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

gamma = 0.99
render = False
seed = 1
log_interval = 10

env = gym.make('CartPole-v0').unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(seed)
env.seed(seed)
batch_size=32
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value
# class Memory():
#     def __init__(self):
#         self.memory=[]
#         self.memory_size=1000
#         self.current_mem_count=0
#     def sample(self):
#         sample_index=np.random.choice(range(self.current_mem_count),batch_size)
#         #samples=np.array(self.memory)[sample_index,:]
#         samples=[self.memory[x] for x in sample_index]
#         return samples,sample_index
#     def store_experience(self,transition):
#         if self.current_mem_count<self.memory_size:
#             self.memory.append(transition)
#             self.current_mem_count+=1
#         else:
#             self.memory.pop(0)
#             self.memory.append(transition)
#     def detete_memory(self):
#         self.memory=[]
class PPO():
    def __init__(self,actor_lr=0.01,critic_lr=0.01):
        self.actor=Actor()
        self.critic=Critic()
        self.buffer=[]
        self.actor_lr=actor_lr
        self.critic_lr=critic_lr
        self.actor_opt=optim.Adam(self.actor.parameters(),lr=self.actor_lr)
        self.critic_opt=optim.Adam(self.critic.parameters(),lr=self.critic_lr)
        self.ppo_update_time=10
        self.clip_param=0.2
        self.max_grad_norm = 0.5
        self.training_step=0
        self.buf_size=0
    def choose_action(self,state):
        state = torch.from_numpy(np.asarray(state)).type('torch.FloatTensor').unsqueeze(0)
        with torch.no_grad():
            action_probs=self.actor(state)
        m = Categorical(action_probs)
        action=m.sample().squeeze().numpy()
        action_probs=action_probs[:,action]
        return action,action_probs
    def get_value(self,state):
        state = torch.from_numpy(np.asarray(state)).type('torch.FloatTensor')

        value=self.critic(state)
        return value
    def store_experience(self,trans):
        self.buffer.append(trans)
        self.buf_size+=1

    def learning(self):
        # state= torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        # action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        # reward = [t.reward for t in self.buffer]
#        old_action_log_prob =torch.from_numpy(np.asarray([t.a_log_prob for t in self.buffer.memory])).type('torch.FloatTensor')
        #old_action_log_prob = torch.FloatTensor([t.a_log_prob for t in self.buffer.memory])
        reward = [t.reward for t in self.buffer]
        #action = [t.action for t in self.buffer]
        G=[]
        R=0
        for r in reward[::-1]:
            R=r+gamma*R
            G.append(R)
        G.reverse()
        G=np.array(G)
        for i in range(self.ppo_update_time):
            # samples,sample_index=self.buffer.sample()
            #Gt_batch=torch.G[sample_index]
            sample_index = np.random.choice(len(self.buffer), batch_size).tolist()
            samples=[self.buffer[x] for x in sample_index]
            Gt_batch=torch.from_numpy(np.asarray(G[sample_index])).type('torch.FloatTensor').view(-1,1)
            old_a_log_prob_batch=torch.from_numpy(np.asarray([t.a_log_prob.numpy() for t in samples])).type('torch.FloatTensor')
            state_batch = torch.from_numpy(np.asarray([t.state for t in samples])).type('torch.FloatTensor')
            action_batch=torch.from_numpy(np.asarray([t.action for t in samples])).type('torch.LongTensor').view(-1,1)
            value_batch=self.get_value(state_batch)
            advantage=(Gt_batch-value_batch).detach()
           # sample_index=sample_index.reshape(-1,1)
            #sample_index=np.array((sample_index)).reshape(-1,1)
            temp=self.actor(state_batch)
            action_prob=self.actor(state_batch).gather(1,action_batch)
            ratio=action_prob/old_a_log_prob_batch
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

            action_loss = -torch.min(surr1, surr2).mean()
            self.actor_opt.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()

            value_loss = F.mse_loss(Gt_batch, value_batch)
            self.critic_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_opt.step()

            self.training_step += 1
        del self.buffer[:]

if __name__=="__main__":
    ppo=PPO()
    episode_num=150
    x_s = np.arange(0, episode_num)
    y_s = np.arange(0, episode_num)
    z_s = np.arange(0, episode_num)
    for i_epoch in range(episode_num):
        state = env.reset()
        eps_reward = 0
        env.render()
        for t in count():
            action,action_prob=ppo.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            eps_reward+=reward
            trans = Transition(state, action, action_prob, reward, next_state)
            # print(ppo.buffer.current_mem_count)
            ppo.store_experience(trans)
            state = next_state
            if done:
                z_s[i_epoch]=t
                y_s[i_epoch]=eps_reward
                print("episode",i_epoch,"length:",t)
                print("episode reward",eps_reward)
                if len(ppo.buffer)>= batch_size:
                    ppo.learning()
                # agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break
    plt.title("reward")
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    plt.plot(x_s, y_s)
    plt.show()

    plt.title("episode_length")
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    plt.plot(x_s, z_s)
    plt.show()
