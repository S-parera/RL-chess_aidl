from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from torch.utils.data import Dataset

import gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")

torch.manual_seed(0)


class Episode:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.values = []
        self.log_probabilities = []


class History(Dataset):
    def __init__(self):
        self.episodes = []
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.log_probabilities = []

    def free_memory(self):
        del self.episodes[:]
        del self.observations[:]
        del self.actions[:]
        del self.advantages[:]
        del self.rewards[:]
        del self.rewards_to_go[:]
        del self.log_probabilities[:]

    def build_dataset(self):
        for episode in self.episodes:
            self.observations += episode.observations
            self.actions += episode.actions
            self.advantages += episode.advantages
            self.rewards += episode.rewards
            self.rewards_to_go += episode.rewards_to_go
            self.log_probabilities += episode.log_probabilities

       

        self.advantages = normalize_list(self.advantages)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.advantages[idx],
            self.log_probabilities[idx],
            self.rewards_to_go[idx],
        )

class PolicyNetwork(torch.nn.Module):
    def __init__(self, n=4, in_dim=128):
        super(PolicyNetwork, self).__init__()

        self.mlp = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, n)
        )

    def forward(self, x):

        y = self.mlp(x)

        return y



class ValueNetwork(torch.nn.Module):
    def __init__(self, in_dim=128):
        super(ValueNetwork, self).__init__()

        self.mlp = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
        )

    def forward(self, x):
        
        y = self.mlp(x)

        return y.squeeze(1) 



def normalize_list(array):
    array = np.array(array)
    array = (array - np.mean(array)) / (np.std(array) + 1e-5)
    return array.tolist()

def get_action(state):

    if not state is torch.Tensor:
        state = torch.from_numpy(state).float().to(device)

    if len(state.size()) == 1:
        state = state.unsqueeze(0) # Create batch dimension

    logits = policy_model(state)

    m = Categorical(logits=logits)

    action = m.sample()

    log_probability = m.log_prob(action)

    value = value_model(state)

    return action.item(), log_probability.item(), value.item()

# def cumulative_sum(array, discount=1.0):
#     curr = 0
#     cumulative_array = []

#     for a in array[::-1]:
#         curr = a + discount * curr
#         cumulative_array.append(curr)

#     return cumulative_array[::-1]

def cumulative_sum(vector, discount):
    out = np.zeros_like(vector)
    n = vector.shape[0]
    for i in reversed(range(n)):
        out[i] =  vector[i] + discount * (out[i+1] if i+1 < n else 0)
    return out.tolist()

def end_episode(last_value):
    # Calculate trajectory rewards to go
    # Calculate trajectory GAE

    # REWARDS TO GO


    rewards = np.array(episode.rewards + [last_value])
    values = np.array(episode.values + [last_value])

    episode.rewards_to_go = cumulative_sum(rewards, gamma)[:-1]

    # GAE
    
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]

    episode.advantages = cumulative_sum(deltas, gamma * gae_lambda)



# def end_episode(last_value):
#     rewards = np.array(episode.rewards + [last_value])
#     values = np.array(episode.values + [last_value])

#     deltas = rewards[:-1] + gamma * values[1:] - values[:-1]

#     episode.advantages = cumulative_sum(deltas.tolist(), discount=gamma * gae_lambda)

#     episode.rewards_to_go = cumulative_sum(rewards.tolist(), discount=gamma)[:-1]





def train_network(data_loader):
    policy_epoch_losses = []
    value_epoch_losses = []

    c1 = 0.01

    for i in range(n_epoch):

        policy_losses = []
        value_losses = []

        for observations, actions, advantages, log_probabilities, rewards_to_go in data_loader:
            observations = observations.float().to(device)
            actions = actions.long().to(device)
            advantages = advantages.float().to(device)
            old_log_probabilities = log_probabilities.float().to(device)
            rewards_to_go = rewards_to_go.float().to(device)

            

            logits = policy_model(observations)

            m = Categorical(logits=logits)

            entropy = m.entropy()

            new_log_probabilities = m.log_prob(actions)

            values = value_model(observations)


            probability_ratios = torch.exp(new_log_probabilities - old_log_probabilities)
            clipped_probabiliy_ratios = torch.clamp(
                probability_ratios, 1 - clip, 1 + clip
            )

            surrogate_1 = probability_ratios * advantages
            surrogate_2 = clipped_probabiliy_ratios * advantages

            Actor_loss = -torch.min(surrogate_1, surrogate_2).mean() - c1 * entropy.mean()

            Critic_loss = F.mse_loss(values, rewards_to_go)
            # Critic_loss = ((values - rewards_to_go)**2)

            policy_optimizer.zero_grad()

            value_optimizer.zero_grad()

            # Actor_loss.backward()
           
            # Critic_loss.backward()
            loss = Actor_loss + Critic_loss
            loss.backward()

            value_optimizer.step()

            policy_optimizer.step()


            policy_losses.append(Actor_loss.item())

            
            value_losses.append(Critic_loss.item())

        policy_epoch_losses.append(np.mean(policy_losses))
        value_epoch_losses.append(np.mean(value_losses))

    return policy_epoch_losses, value_epoch_losses



# env_name = "CartPole-v1"
env_name = "LunarLander-v2"

learning_rate = 1e-3
state_scale = 1.0
reward_scale = 1.0
clip = 0.2

env = gym.make(env_name)
observation = env.reset()

n_actions = env.action_space.n
feature_dim = observation.size

value_model = ValueNetwork(in_dim=feature_dim).to(device)
value_optimizer = optim.Adam(value_model.parameters(), lr=learning_rate)

policy_model = PolicyNetwork(in_dim=feature_dim, n=n_actions).to(device)
policy_optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)

n_epoch = 4

max_episodes = 10
max_timesteps = 200

batch_size = 32

max_iterations = 200

gamma = 0.99
gae_lambda = 0.95

history = History()

epoch_ite = 0
episode_ite = 0

running_reward = -500

timestr = time.strftime("%d%m%Y-%H%M%S-")

log_dir = "./runs/" + timestr + env_name + "-BS" + str(batch_size) + "-E" + \
        str(max_episodes) + "-MT" + str(max_timesteps) + "-NE" + str(n_epoch) + \
        "-LR" + str(learning_rate) + "-G" + str(gamma) + "-L" + str(gae_lambda)

writer = SummaryWriter(log_dir=log_dir)

for ite in tqdm(range(max_iterations), ascii=True):

    # if ite % 50 == 0:
    #     torch.save(
    #         policy_model.state_dict(),
    #         Path(log_dir) / (env_name + f"_{str(ite)}_policy.pth"),
    #     )
    #     torch.save(
    #         value_model.state_dict(),
    #         Path(log_dir) / (env_name + f"_{str(ite)}_value.pth"),
    #     )

    observation = env.reset()
    ep_reward = 0
    for episode_i in range(max_episodes):
        
        episode = Episode()

        for timestep in range(max_timesteps):
            # Loop through time_steps

            action, log_probability, value = get_action(observation / state_scale)

            new_observation, reward, done, info = env.step(action)

            ep_reward +=reward

            episode.observations.append(observation / state_scale)
            episode.actions.append(action)
            episode.rewards.append(reward / reward_scale)
            episode.values.append(value)
            episode.log_probabilities.append(log_probability)

 

            observation = new_observation

            if done:
                end_episode(last_value=0)
                episode_ite += 1

                writer.add_scalar(
                    "Average Episode Reward",
                    ep_reward,
                    episode_ite,
                )
                writer.add_scalar(
                    "Average Probabilities",
                    np.exp(np.mean(episode.log_probabilities)),
                    episode_ite,
                )

                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

                #Reset episode rewards and enviroment because one episode finished
                ep_reward = 0
                observation = env.reset()
                break

            if timestep == max_timesteps - 1:
                # Episode didn't finish so we have to append value to RTGs and advantages
                _, _, value = get_action(observation / state_scale)
                end_episode(last_value=value)

        


        # At this point we have collected a trajectory of T time_steps
        # This is not a full episode.

        history.episodes.append(episode)
    
    # Here we have collected N trajectories.
    history.build_dataset()

    data_loader = DataLoader(history, batch_size=batch_size, shuffle=True)


    policy_loss, value_loss = train_network(data_loader)


    for p_l, v_l in zip(policy_loss, value_loss):
        epoch_ite += 1
        writer.add_scalar("Policy Loss", p_l, epoch_ite)
        writer.add_scalar("Value Loss", v_l, epoch_ite)

    history.free_memory()

    # print("\n", running_reward)

    writer.add_scalar("Running Reward", running_reward, epoch_ite)


    if (running_reward > env.spec.reward_threshold):
        print("\nSolved!")
        break