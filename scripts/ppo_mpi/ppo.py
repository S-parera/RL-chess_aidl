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
import threading
import queue

from memory import Episode, History
from network import PolicyNetwork, ValueNetwork
from collect_trajectories import collect


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")


def train_network(data_loader, policy_model, value_model, policy_optimizer, value_optimizer ,n_epoch, clip, train_ite, writer, entropy_coefficient):

    policy_model.train()
    value_model.train()

    policy_epoch_losses = []
    value_epoch_losses = []

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

            Actor_loss = -torch.min(surrogate_1, surrogate_2).mean() - entropy_coefficient * entropy.mean()

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
    env_name = "CartPole-v1"
    # env_name = "LunarLander-v2"
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
    max_iterations = 500
    gamma = 0.99
    gae_lambda = 0.95
    entropy_coefficient = 0.01
    env_threshold = env.spec.reward_threshold

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

    # LOAD MODEL
    # Create folder models
    if not Path("./models").exists():
        print("Creating Models folder")
        Path("./models").mkdir()

    model_path = Path("./models/" + env_name + ".tar")
    if model_path.exists():
        print("Loading model!")
        #Load model
        checkpoint = torch.load(model_path)
        policy_model.load_state_dict(checkpoint['policy_model'])
        policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        value_model.load_state_dict(checkpoint['value_model'])
        value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        running_reward = checkpoint['running_reward']

    EnvQueue = queue.SimpleQueue()

    for _ in range(max_episodes):
        env = gym.make(env_name)
        observation = env.reset()
        EnvQueue.put((env, observation, 0))
    

    for ite in tqdm(range(max_iterations), ascii=True):

        if ite % 5 == 0:
            torch.save({
                'policy_model': policy_model.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict(),
                'value_model': value_model.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
                'running_reward': running_reward}, model_path)


        
        q = queue.SimpleQueue()

        
        env_list = []
        while not EnvQueue.empty():
            env_list.append(EnvQueue.get())

        threads = []
        for env in env_list:
            t = threading.Thread(target=collect, args=[q, env_name, env, EnvQueue,
                                max_timesteps, state_scale,reward_scale,
                                policy_model, value_model, gamma,
                                gae_lambda, device])
            t.start()
            threads.append(t)

        for t in threads:
            t.join()



        avg_episode_reward = []
        # Write all episodes from queue to history buffer
        while not q.empty():
            episode, done = q.get()
            history.episodes.append(episode)
            avg_episode_reward.append((episode.reward, done))
        
        for ep_reward, done in avg_episode_reward:
            if done:
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

                writer.add_scalar("Average Episode Reward", ep_reward, episode_ite)
                episode_ite += 1

        # avg_ep_reward = sum(avg_episode_reward) / len(avg_episode_reward)

        # Here we have collected N trajectories and prepare dataset
        history.build_dataset()

        data_loader = DataLoader(history, batch_size=batch_size, shuffle=True, drop_last=True)


        policy_loss, value_loss, train_ite = train_network(data_loader, policy_model, value_model,
                                                        policy_optimizer, value_optimizer ,n_epoch, clip,
                                                        train_ite, writer, entropy_coefficient)


        for p_l, v_l in zip(policy_loss, value_loss):
            epoch_ite += 1
            writer.add_scalar("Policy Loss", p_l, epoch_ite)
            writer.add_scalar("Value Loss", v_l, epoch_ite)

        history.free_memory()

        # print("\n", running_reward)

        writer.add_scalar("Running Reward", running_reward, epoch_ite)


        if (running_reward > env_threshold):
            print("\nSolved!")
            break

if __name__ == '__main__':
    main()