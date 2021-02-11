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
        action = F.softmax(self.action(x), dim=-1)
        value_head = self.value_head(x)

        return action, value_head

class Bookkeeping():
    def __init__(self):
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.accumulated_rewards = []
        self.undiscounted_rewards = []
        self.advantages = []
        
        self.actor_loss = []
        self.critic_loss = []
    
    def reset_bookkeeping(self):
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.accumulated_rewards = []
        self.advantages = []

        self.actor_loss = []
        self.critic_loss = []

class PPO:
    def __init__(self, lr=1e-3, gamma=0.99, name="A2c"):
        # Initialize hyperparameters
        self.init_hyperparameters(lr,gamma)
        
        # Init bookkeeping
        self.bookkeeping = Bookkeeping()

        # Set save paths
        self.name = name
        self.create_folders()

        # Init policy
        self.policy = Policy()

        # Init optimizer
        self.optimizer = Adam(self.policy.parameters(), self.lr)

        # Init enviroment
        self.env = gym.make('CartPole-v1')

        # Init running reward
        self.running_reward = 10

    def init_hyperparameters(self,lr,gamma):
        self.max_episodes = 1000
        self.max_time_steps = 600
        self.gamma = gamma
        self.lr = lr

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        
        probs, value = self.policy(state)

        m = Categorical(probs)

        action = m.sample()

        log_probs = m.log_prob(action)

        return action.item(), value, log_probs

    def plot(self, to_plot, show=False):
            
            plt.plot(to_plot)
            plt.savefig(self.plot_path)

            if(show):
                plt.show()
            plt.clf()

    def create_folders(self):
        # Create folders for bookkeeping

        #Check if Models exists
        if(not os.path.exists("./Models")):
            # Create folder
            os.mkdir('./Models')

        self.model_path = os.path.join('./Models\\' + self.name + '.pth')

        if(not os.path.exists("./Plots")):
            # Create folder
            os.mkdir('./Plots')

        self.plot_path = os.path.join('./Plots\\' + self.name + '.png')

    def train(self):
        
        # Loop through episodes
        for i in range(self.max_episodes):

            # Reset enviroment
            state = self.env.reset()

            # Reset bookkeeping
            self.bookkeeping.reset_bookkeeping()

            # Loop through timesteps
            for e in range(self.max_time_steps):

                # Run action through policy
                action, value, log_probs = self.get_action(state)

                # Logprobs bookkeeping
                self.bookkeeping.log_probs.append(log_probs)


                # Calculate next step
                state, reward, done, _ = self.env.step(action)

                # Save reward
                self.bookkeeping.rewards.append(reward)

                #save value
                self.bookkeeping.values.append(value)

                if(done):
                    break

            # Calculate reward as advantage
            if(not done):
                _, G, _ = self.get_action(state)
            else:
                G = 0

            # Calculate undiscounted rewards
            self.bookkeeping.undiscounted_rewards.append(sum(self.bookkeeping.rewards))
            
            # Calculate discounted rewards and advantages
            for reward in reversed(self.bookkeeping.rewards):
                G = reward + self.gamma * G

                # Save values in good order
                self.bookkeeping.accumulated_rewards.insert(0,G)

            # Normalize G
            self.bookkeeping.accumulated_rewards = torch.tensor(self.bookkeeping.accumulated_rewards)
            self.bookkeeping.accumulated_rewards = (self.bookkeeping.accumulated_rewards - self.bookkeeping.accumulated_rewards.mean()) / (self.bookkeeping.accumulated_rewards.std() + 1e-8)


            for _ in range(5):
                previous_log_probs = self.bookkeeping.log_probs[0]
                for log_probs, value, G in zip(self.bookkeeping.log_probs, self.bookkeeping.values,self.bookkeeping.accumulated_rewards):

                    # Calculate advantage
                    advantage = G - value

                    # Calculate ratio
                    ratio = (log_probs - previous_log_probs).exp()

                    # Clipped PPO
                    clipped = torch.clamp(ratio, 1-0.2, 1+0.2) * advantage

                    # Accumulate actor_loss
                    self.bookkeeping.actor_loss.append(-min(ratio*advantage, clipped))

                    # Accumulate critic_loss
                    self.bookkeeping.critic_loss.append(F.mse_loss(value,torch.tensor([G])))

                    previous_log_probs = log_probs

                # Sum losses
                CriticLoss = torch.stack(self.bookkeeping.critic_loss).sum()
                ActorLoss = torch.stack(self.bookkeeping.actor_loss).sum()
                loss = CriticLoss + ActorLoss

                # Reset gradients
                self.optimizer.zero_grad()

                # Update policy
                loss.backward()
                self.optimizer.step()

            # Calculate running reward complementary filter
            self.running_reward = 0.05 * self.bookkeeping.undiscounted_rewards[-1] + (1 - 0.05) * self.running_reward

            print(f"Episode {i+1}. Steps: {e+1}, Running reward: {self.running_reward:.2f}.  Actor Loss: {ActorLoss:.2f}, Critic Loss: {CriticLoss:.2f}")

            # check if we have "solved" the cart pole problem
            if self.running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                        "the last episode runs to {} time steps!".format(self.running_reward, e+1))
                break

        # Plot rewards
        self.plot(self.bookkeeping.undiscounted_rewards, show=False)

        # Save model
        torch.save(self.policy.state_dict(), self.model_path)


model = PPO(5e-4, 0.99, "prova")
model.train()
                