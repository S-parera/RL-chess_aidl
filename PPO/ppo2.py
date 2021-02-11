import gym
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions.categorical import Categorical
from torch.optim import Adam

import matplotlib.pyplot as plt 
import numpy as np 

from itertools import count

import os, sys

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,128)
        self.relu = nn.ReLU(inplace=True)
        self.action = nn.Linear(128,2)
        self.value_head = nn.Linear(128,1)

    def forward(self,x):
        x = self.relu(self.fc1(x))
        action = self.action(x)
        value_head = self.value_head(x)

        return action, value_head

class MemoryBuffer():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.rtg = []
        self.advantages = []

    def reset_buffer(self):
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.rtg = []
        self.advantages = []



def get_action(state):
        state = torch.from_numpy(state).float()
        
        probs, value = policy(state.cuda())

        m = Categorical(logits=probs)

        action = m.sample()

        log_probs = m.log_prob(action)

        return action.item(), value, log_probs

def collect_trajectories(env):
    state = env.reset()

    trajectory_states = []
    trajectory_actions = []
    trajectory_values = []
    trajectory_log_probs = []
    trajectory_rewards = []

    for t in range(500):
        # get action
        action, value, log_prob = get_action(state)

        trajectory_actions.append(action)
        trajectory_values.append(value)
        trajectory_log_probs.append(log_prob)
        trajectory_states.append(state)

        # Step
        state, reward, done, _ = env.step(action)



        trajectory_rewards.append(reward)

        if (done):
            break
    # Calculate reward as advantage
    # if(not done):
    #     _, G, _ = get_action(state)
    #     trajectory_rewards.append(G)
    # else:
    #     G = 0

    buffer.log_probs.append(torch.tensor(trajectory_log_probs))
    buffer.actions.append(torch.tensor(trajectory_actions))
    buffer.rewards.append(trajectory_rewards)
    buffer.values.append(trajectory_values)
    buffer.states.append(torch.tensor(trajectory_states))

def rewards_to_go(rewards_buffer):
    for trajectory in rewards_buffer:
        trajectory_rtg = []
        G = 0
        for reward in reversed(trajectory):
            G = G * gamma + reward
            trajectory_rtg.insert(0, G)

        buffer.rtg.append(torch.tensor(trajectory_rtg))

def calc_advantages(value_buffer, rtg_buffer):
    for value_traj, G_traj in zip(value_buffer, rtg_buffer):
        trajectory_adv = []

        for value, G in zip(value_traj, G_traj):
            advantage_t = G - value
            trajectory_adv.append(advantage_t)

        buffer.advantages.append(torch.tensor(trajectory_adv))


def train():
    states = torch.cat(buffer.states).cuda()
    actions = torch.cat(buffer.actions).cuda()
    log_probs = torch.cat(buffer.log_probs)
    advantages = torch.cat(buffer.advantages).cuda()
    Gs = torch.cat(buffer.rtg).cuda()

    old_log_probs = log_probs.cuda()
    
    for i in range(10):

        probs, values = policy(states.float().cuda())

        m = Categorical(logits=probs)
        log_probs = m.log_prob(actions)

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantages
        ActorLoss = -torch.min(surr1, surr2).mean()

        CriticLoss = (Gs - values.squeeze()).mean()

        loss = ActorLoss + CriticLoss

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
         

def render_env():
    state, done = env.reset(), False
    while not done:
        env.render()
        action, value, log_prob = get_action(state)
        # Step
        state, reward, done, _ = env.step(action)

    env.close()


# env = gym.make("LunarLander-v2")
env = gym.make("CartPole-v1")

buffer = MemoryBuffer()

gamma = 0.99

# 1 Initial policy
policy = Policy().cuda()

optimizer = torch.optim.Adam(params=policy.parameters(), lr = 1e-4)

running_reward = 0
average_rewards = []

# 2 for k = 0, 1, 2 , ... do
for k in range(2000):
    # Reset buffer
    buffer.reset_buffer()
    # 3 Collect set of trajectories
    for i in range(10): # Collect 10 trajectories
        collect_trajectories(env)

    # 4 Compute rewards to go
    rewards_to_go(buffer.rewards)

    # 5 Compute advantages
    calc_advantages(buffer.values, buffer.rtg)

    # 6 PPO loss
    train()
    
    average_reward = 0
    for e in buffer.rewards:
        average_reward += sum(e)/10

    
    running_reward = 0.1 * average_reward + (1 - 0.1) * running_reward

    print(f"Episode: {k} Average reward {average_reward:.2f} Running reward: {running_reward:.2f}")

    # check if we have "solved" the cart pole problem
    if running_reward > env.spec.reward_threshold:
        print(f"Solved! Running reward is now {running_reward} and the last episode runs to {average_reward} time steps!")
        break

    average_rewards.append(average_reward)

    if k%100 == 0:
        render_env()






plt.plot(average_rewards)
#plt.savefig(path)
plt.show()
#plt.clf()
    
render_env()