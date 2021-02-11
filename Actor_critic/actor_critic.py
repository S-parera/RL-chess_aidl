"""
Create a network with two outputs, the action distribution prob
and the value_head.

loof forever (for each episode)
    loop while S is not terminate (for each time step)
        Input state to policy
        select action
            Categorical distribution with softmax (may crash)
            Logits thing
            Save log_probs
        Take action and observe S' and reward
            Save reward in R
        Calculate advantage
            A = G + gamma*value_head_next_step - value_head_actual step
            G = accumulated episode reward with discount
            G=0
            for reward in reverse(Rewards):
                G = reward + gamma * G
                save accumulated rewards

        
        loss function for value_head
            MSELoss for example
            F.smooth_l1_loss(value_head, G)
        loss function for J
            -log_prob * advantage * (gamma^num_episode)


Should keep track of Actions, log_probs and rewards
of each timestep of the episode to use latter.
"""

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions.categorical import Categorical
from torch.optim import Adam


import numpy as np 

from itertools import count

from plots import plot

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

class Bookeeping():
    def __init__(self):
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.accumulated_rewards = []
        self.running_rewards = []

    def reset_lob_probs(self):
        self.log_probs = []

    def reset_rewards(self):
        self.rewards = []

    def reset_values(self):
        self.values = []

    def reset_ac_rewards(self):
        self.accumulated_rewards = []


def get_action(state,policy, bookeeping):
    state = torch.from_numpy(state).float()

    probs, value = policy(state)

    m = Categorical(probs)

    action = m.sample()

    log_probs = m.log_prob(action)
    # Store log_probs
    bookeeping.log_probs.append(log_probs)

    return action.item(), value


def main():

    bookeeping = Bookeeping()

    policy = Policy()
    optimizer = Adam(policy.parameters(), lr=1e-2)

    env = gym.make("CartPole-v1")
    # print("Action space: ", env.action_space.n)
    # print("Observation space: ", env.observation_space.shape[0])


    # PARAMETERS
    max_episodes = 2000
    max_time_steps = 2000
    gamma = 0.98


    running_reward = 10

    for i in range(max_episodes):
        #Reset enviroment
        state = env.reset()

        bookeeping.reset_rewards()
        bookeeping.reset_lob_probs()
        bookeeping.reset_values()
        bookeeping.reset_ac_rewards()


        for e in range(max_time_steps):
            
            #env.render()

            # Run action through policy
            action, value = get_action(state,policy, bookeeping)

            state, reward, done, _ = env.step(action)

            # Save reward
            bookeeping.rewards.append(reward)

            # Save values
            bookeeping.values.append(value)

            if(done):
                break

        # CALCULATE ADVANTAGE
        # Approximate next step reward value
        if not done:
            _, value_head_next_step = get_action(state)
            bookeeping.rewards.append(value_head_next_step.item())
            bookeeping.values.append(value_head_next_step)


        # Calculate Reward until time step
        G=0
        for reward in reversed(bookeeping.rewards):
            G = reward + gamma * G
            #save accumulated rewards
            bookeeping.accumulated_rewards.insert(0, G)

        critic_loss = []
        actor_loss = []

        bookeeping.accumulated_rewards = torch.tensor(bookeeping.accumulated_rewards)
        bookeeping.accumulated_rewards = (bookeeping.accumulated_rewards - bookeeping.accumulated_rewards.mean()) / (bookeeping.accumulated_rewards.std() + 1e-8)

        for log_prob, value, ac_reward in zip(bookeeping.log_probs, bookeeping.values, bookeeping.accumulated_rewards):
            Advantage = ac_reward - value

            critic_loss.append(F.smooth_l1_loss(value, torch.tensor(ac_reward)))

            actor_loss.append(-log_prob * Advantage)

        CriticLoss = torch.stack(critic_loss).sum()
        ActorLoss = torch.stack(actor_loss).sum()

        loss = CriticLoss + ActorLoss

        optimizer.zero_grad()

        
        
        loss.backward()
        optimizer.step()



        running_reward = 0.05 * (e+1) + (1 - 0.05) * running_reward

        bookeeping.running_rewards.append(e+1)

        print(f"Episode {i+1}. Steps: {e+1}, Running reward: {running_reward:.2f}")

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, e+1))
            break


    


    model_path = os.path.join(sys.path[0], "Models\model_dict.pth")
    plot_path = os.path.join(sys.path[0], "Plots\plot.png")

    torch.save(policy.state_dict(), model_path)


    plot(bookeeping.running_rewards, plot_path)
    render_env(env, policy, bookeeping)


def render_env(env, policy, bookeeping):
    state, done, total_rew = env.reset(), False, 0
    while not done:
        env.render()
        action, value = get_action(state, policy, bookeeping)
        state, reward, done, _ = env.step(action)


if __name__ == '__main__':
    main()

