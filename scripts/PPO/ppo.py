import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import gym
import numpy as np
from tqdm import tqdm
from pathlib import Path

import time

from memory import Episode, History
from network import PolicyNetwork, ValueNetwork
from collect_trajectories import collect


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")


def train_network(data_loader, policy_model, value_model, policy_optimizer, value_optimizer ,n_epoch, clip, train_ite, writer):

    policy_model.train()
    value_model.train()

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
            clipped_probabiliy_ratios = torch.clamp(probability_ratios, 1 - clip, 1 + clip)
            surrogate_1 = probability_ratios * advantages
            surrogate_2 = clipped_probabiliy_ratios * advantages

            Actor_loss = -torch.min(surrogate_1, surrogate_2).mean() - c1 * entropy.mean()

            Critic_loss = F.mse_loss(values, rewards_to_go)

            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()

            loss = Actor_loss + Critic_loss
            loss.backward()

            value_optimizer.step()
            policy_optimizer.step()


            policy_losses.append(Actor_loss.item())
            value_losses.append(Critic_loss.item())

        policy_epoch_losses.append(np.mean(policy_losses))
        value_epoch_losses.append(np.mean(value_losses))

    train_ite +=1
    
    for name, weight in policy_model.named_parameters():
        writer.add_histogram(name,weight, train_ite)
        writer.add_histogram(f'{name}.grad',weight.grad, train_ite)

    return policy_epoch_losses, value_epoch_losses, train_ite

def main():
    
    # ENVIROMENT
    # env_name = "CartPole-v1"
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    n_actions = env.action_space.n
    feature_dim = env.observation_space.shape[0]

    # PARAMETERS
    learning_rate = 1e-3
    state_scale = 1.0
    reward_scale = 1.0
    clip = 0.2
    n_epoch = 4
    max_episodes = 10
    max_timesteps = 200
    batch_size = 32
    max_iterations = 200
    gamma = 0.99
    gae_lambda = 0.95

    # NETWORK
    value_model = ValueNetwork(in_dim=feature_dim).to(device)
    value_optimizer = optim.Adam(value_model.parameters(), lr=learning_rate)

    policy_model = PolicyNetwork(in_dim=feature_dim, n=n_actions).to(device)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
    
    # INIT
    history = History()
    observation = env.reset()

    epoch_ite = 0
    episode_ite = 0
    train_ite = 0
    running_reward = -500

    # TENSORBOARD 

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

        
        episode_ite, running_reward = collect(episode_ite, running_reward, env, max_episodes, max_timesteps, state_scale,
                reward_scale, writer, history, policy_model, value_model, gamma, gae_lambda, device)
        
        # Here we have collected N trajectories.
        history.build_dataset()

        data_loader = DataLoader(history, batch_size=batch_size, shuffle=True, drop_last=True)


        policy_loss, value_loss, train_ite = train_network(data_loader, policy_model, value_model, policy_optimizer, value_optimizer ,n_epoch, clip, train_ite, writer)


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

if __name__ == '__main__':
    main()