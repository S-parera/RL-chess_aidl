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

from chess_env import ChessEnv


device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_network(data_loader, policy_model, value_model, policy_optimizer, value_optimizer ,n_epoch, clip, train_ite, writer, entropy_coefficient):

    policy_model.train()
    value_model.train()

    policy_epoch_losses = []
    value_epoch_losses = []

    for i in range(n_epoch):

        policy_losses = []
        value_losses = []

        for observations, actions, advantages, log_probabilities, rewards_to_go, masks in data_loader:
            observations = observations.float().to(device)
            actions = actions.long().to(device)
            advantages = advantages.float().to(device)
            old_log_probabilities = log_probabilities.float().to(device)
            rewards_to_go = rewards_to_go.float().to(device)
            masks = masks.to(device)

            logits = policy_model(observations)

            for i in range(masks.shape[0]):
                logits[i][masks[i] == 0] = -float("Inf")

            m = Categorical(logits=logits)

            entropy = m.entropy()

            new_log_probabilities = m.log_prob(actions)

            values = value_model(observations).squeeze(1)

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
    env_name = "Chess"
    env = ChessEnv()


    # PARAMETERS
    learning_rate = 5e-4
    state_scale = 1.0
    reward_scale = 1.0
    clip = 0.2
    n_epoch = 4
    max_episodes = 4
    max_timesteps = 50
    batch_size = 16
    max_iterations = 200
    gamma = 0.99
    gae_lambda = 0.95
    entropy_coefficient = 0.01

    # NETWORK
    value_model = ValueNetwork().to(device)
    value_optimizer = optim.Adam(value_model.parameters(), lr=learning_rate)

    policy_model = PolicyNetwork().to(device)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)

   
    # INIT
    history = History()

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


    # Create SavedEnvs queue
    SavedEnv = queue.SimpleQueue()
    for _ in range(max_episodes):
        env = ChessEnv()
        SavedEnv.put((env, env.reset(), 0))
    
    # START ITERATING   
    for ite in tqdm(range(max_iterations), ascii=True):

        if ite % 5 == 0:
            torch.save({
                'policy_model': policy_model.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict(),
                'value_model': value_model.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
                'running_reward': running_reward}, model_path)

        print("\nSimulating")
        start_simulation = time.perf_counter()
        
        q = queue.SimpleQueue()

        env_list = []
        while not SavedEnv.empty():
            env_list.append(SavedEnv.get())

        threads = []
        for saved_env in env_list:
            t = threading.Thread(target=collect, args=[q, env_name, saved_env,
                                SavedEnv, max_timesteps, state_scale, reward_scale,
                                policy_model, value_model, gamma,
                                gae_lambda, device])
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        
        

        # for i in range(max_episodes):
        #     collect(q, env_name,
        #                         max_timesteps, state_scale,reward_scale,
        #                         policy_model, value_model, gamma,
        #                         gae_lambda, device)



        avg_episode_reward = []
        # Write all episodes from queue to history buffer
        while not q.empty():
            episode, done = q.get()
            history.episodes.append(episode)
            avg_episode_reward.append((episode.reward, done))
        
        end_simulation = time.perf_counter()
        print(f"Simulation time: {end_simulation-start_simulation} ")
        
        for ep_reward, done in avg_episode_reward:
            if done:
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

                writer.add_scalar("Average Episode Reward", ep_reward, episode_ite)
                episode_ite += 1

        # avg_ep_reward = sum(avg_episode_reward) / len(avg_episode_reward)

        # Here we have collected N trajectories and prepare dataset
        history.build_dataset()

        data_loader = DataLoader(history, batch_size=batch_size, shuffle=True, drop_last=True)



        print("Training")
        policy_loss, value_loss, train_ite = train_network(data_loader, policy_model, value_model,
                                                        policy_optimizer, value_optimizer ,n_epoch, clip,
                                                        train_ite, writer, entropy_coefficient)

        end_training = time.perf_counter()
        print(f"Training time: {end_training-end_simulation}")


        for p_l, v_l in zip(policy_loss, value_loss):
            epoch_ite += 1
            writer.add_scalar("Policy Loss", p_l, epoch_ite)
            writer.add_scalar("Value Loss", v_l, epoch_ite)

        history.free_memory()

        # print("\n", running_reward)

        writer.add_scalar("Running Reward", running_reward, epoch_ite)


        if (running_reward >0):
            print("\nSolved!")
            break

if __name__ == '__main__':
    main()