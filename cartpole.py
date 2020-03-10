import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
from torch.distributions import Categorical
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")



env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

episode_num=30000
N_F = env.observation_space.shape[0]
N_A = env.action_space.n
L_A=0.001
L_C=0.01
discount_factor=0.99

np.random.seed(2)

class Policy_Net(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Policy_Net,self).__init__()
        self.fc1=nn.Linear(input_dim,4)
        self.fc2=nn.Linear(4,5)
        self.fc3=nn.Linear(5,output_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        output=F.softmax(self.fc3(x),dim=0)
        return output
class Value_Net(nn.Module):
    def __init__(self,input_dim):
        super(Value_Net,self).__init__()
        self.fc1=nn.Linear(input_dim,4)
        self.fc2=nn.Linear(4,5)
        self.fc3=nn.Linear(5,1)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        output=self.fc3(x)
        return output

actor=Policy_Net(N_F,N_A)
critic=Value_Net(N_F)
x_s=np.arange(0,episode_num)
y_s=np.arange(0,episode_num)
z_s=np.arange(0,episode_num)
#print(x_s.shape,y_s.shape)
for i in range(episode_num):
    state = env.reset()
    state = torch.from_numpy(np.asarray(state).astype(np.float64)).type('torch.FloatTensor')
    eps_reward=0
    eps_len=0
    while True:
        with torch.no_grad():
            action_prob=actor(state)
        m = Categorical(action_prob)
    #    action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        action = m.sample().numpy()
        #print(action)
        next_state, reward, done, _ = env.step(action)
        x, x_dot, theta, theta_dot = next_state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward= r1 + r2
        next_state = torch.from_numpy(np.asarray(state)).type('torch.FloatTensor')
        value_next = critic(next_state)
        td_target = reward + discount_factor * value_next
        td_error = td_target - critic(state)
        eps_reward+=reward


        picked_action_probs = action_prob.gather(0, torch.from_numpy(action))
        loss_actor=-torch.log(picked_action_probs)*td_error
        value_now=critic(state)
       # loss_value=nn.MSELoss(td_target,value_now)
        loss_value=0.5*(td_target-value_now)**2



        #loss_value.register_backward_hook()
        optimizer1 = optim.Adam(actor.parameters(), lr=0.01)
        optimizer2 = optim.Adam(actor.parameters(), lr=0.01)

        Net = actor, critic
        losses = loss_actor, loss_value
        opts = optimizer1, optimizer2
        for net, loss, opt in zip(Net, losses, opts):
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
        if done:
            break
        state = next_state
        eps_len+=1
    y_s[i]=eps_reward
    z_s[i]=eps_reward
    print("episode reward:\n",eps_reward)
    print("episode length:\n", eps_len)

#print(x_s.shape,y_s.shape)
# fig1 = plt.figure('Figure1',figsize = (6,4))
# fig1.plot(x_s,y_s)
# fig2 = plt.figure('Figure2',figsize = (6,4))
# fig2.plot(x_s,z_s)
plt.title("reward")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x_s,y_s)
plt.show()

plt.title("episode_length")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x_s,z_s)
plt.show()






