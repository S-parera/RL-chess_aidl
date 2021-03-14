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
from network import PolicyNetwork, ValueNetwork, ChessNN
from collect_trajectories import collect

from chess_env import ChessEnv


device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_network(data_loader, model, optimizer, n_epoch, clip, train_ite, writer, entropy_coefficient):

    model.train()

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

            logits, values = model(observations)

            values = values.squeeze(1)

            for i in range(masks.shape[0]):
                logits[i][masks[i] == 0] = -float("Inf")

            m = Categorical(logits=logits)

            entropy = m.entropy()

            new_log_probabilities = m.log_prob(actions)

            probability_ratios = torch.exp(new_log_probabilities - old_log_probabilities)
            clipped_probabiliy_ratios = torch.clamp(probability_ratios, 1 - clip, 1 + clip)
            surrogate_1 = probability_ratios * advantages
            surrogate_2 = clipped_probabiliy_ratios * advantages

            Actor_loss = -torch.min(surrogate_1, surrogate_2).mean() - entropy_coefficient * entropy.mean()

            Critic_loss = F.mse_loss(values, rewards_to_go)

            optimizer.zero_grad()

            loss = Actor_loss + Critic_loss
            loss.backward()

            optimizer.step()


            policy_losses.append(Actor_loss.item())
            value_losses.append(Critic_loss.item())

        policy_epoch_losses.append(np.mean(policy_losses))
        value_epoch_losses.append(np.mean(value_losses))

    train_ite +=1
    
    for name, weight in model.named_parameters():
        writer.add_histogram(name,weight, train_ite)
        writer.add_histogram(f'{name}.grad',weight.grad, train_ite)

    return policy_epoch_losses, value_epoch_losses, train_ite

def main(env_name, lr, state_scale, reward_scale, clip, train_epoch, max_episodes,
            max_timesteps, batch_size, max_iterations, gamma, gae_lambda, entropy_coefficient):
    
    # ENVIROMENT
    env_name = env_name
    env = ChessEnv()


    # PARAMETERS
    learning_rate = lr
    state_scale = state_scale
    reward_scale = reward_scale
    clip = clip
    n_epoch = train_epoch
    max_episodes = max_episodes
    max_timesteps = max_timesteps
    batch_size = batch_size
    max_iterations = max_iterations
    gamma = gamma
    gae_lambda = gae_lambda
    entropy_coefficient = entropy_coefficient

    # NETWORK
    # value_model = ValueNetwork().to(device)
    # value_optimizer = optim.Adam(value_model.parameters(), lr=learning_rate)

    # policy_model = PolicyNetwork().to(device)
    # policy_optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)

    model = ChessNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

   
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
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
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
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
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
                                model, gamma,
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
        print(f"Simulation time: {end_simulation-start_simulation:.2f} ")
        
        for ep_reward, done in avg_episode_reward:
            if done:
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

                writer.add_scalar("Episode Reward", ep_reward, episode_ite)
                writer.add_scalar("Running Reward", running_reward, episode_ite)
                episode_ite += 1

        # avg_ep_reward = sum(avg_episode_reward) / len(avg_episode_reward)

        # Here we have collected N trajectories and prepare dataset
        history.build_dataset()

        data_loader = DataLoader(history, batch_size=batch_size, shuffle=True, drop_last=True)



        print("Training")
        policy_loss, value_loss, train_ite = train_network(data_loader, model, optimizer,
                                                        n_epoch, clip, train_ite, writer,
                                                        entropy_coefficient)

        end_training = time.perf_counter()
        print(f"Training time: {end_training-end_simulation:.2f}")


        for p_l, v_l in zip(policy_loss, value_loss):
            epoch_ite += 1
            writer.add_scalar("Policy Loss", p_l, epoch_ite)
            writer.add_scalar("Value Loss", v_l, epoch_ite)

        history.free_memory()

        # print("\n", running_reward)

        


        if (running_reward >0):
            print("\nSolved!")
            break

if __name__ == '__main__':
    main(env_name = "Res34-Stock15-Chess",
            lr = 1e-3,
            state_scale = 1.0,
            reward_scale= 1.0,
            clip = 0.2,
            train_epoch = 4,
            max_episodes = 10,
            max_timesteps = 100,
            batch_size = 32,
            max_iterations = 1000,
            gamma = 0.95,
            gae_lambda = 0.99,
            entropy_coefficient = 0.01)


