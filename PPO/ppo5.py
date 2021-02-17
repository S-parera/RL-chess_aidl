import torch
import torch.nn as nn 
from torch.distributions.categorical import Categorical
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


import numpy as np
import gym 
import matplotlib.pyplot as plt 

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter



class EpisodeBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.values = []
        self.rewards =[]
        
        self.log_probs = []
        self.rtgs = []

        self.advantages = []


class HistoryBuffer(Dataset):
    def __init__(self):
        self.episodes = []
        self.states = []
        self.actions = []
        self.values = []
        self.rewards =[]
        
        self.log_probs = []
        self.rtgs = []

        self.advantages = []

    def flatten_data(self):
        for episode in self.episodes:
            self.states += episode.states[:]
            self.actions += episode.actions[:]
            self.values += episode.values[:]
            self.rewards += episode.rewards[:]
            self.log_probs += episode.log_probs[:]
            self.rtgs += list(episode.rtgs)[:]
            self.advantages += list(episode.advantages)[:]

    def reset_history(self):
        self.episodes = []
        self.states = []
        self.actions = []
        self.values = []
        self.rewards =[]
        
        self.log_probs = []
        self.rtgs = []

        self.advantages = []


    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.advantages[idx],
            self.log_probs[idx],
            self.rtgs[idx],
        )


class ActorNN(nn.Module):
    def __init__(self, action_space, observation_space):
        super().__init__()

        self.mlp = nn.Sequential(
                nn.Linear(observation_space, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_space)
        )

    def forward(self, state):

        logits = self.mlp(state)
        
        return logits

class CriticNN(nn.Module):
    def __init__(self, observation_space):
        super().__init__()

        self.mlp = nn.Sequential(
                nn.Linear(observation_space, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
        )

    def forward(self, state):

        value = self.mlp(state)
        
        return value

def get_action(state):
        state = torch.from_numpy(state).float().to(device)
        
        logits = actor(state)

        m = Categorical(logits=logits)

        action = m.sample()

        log_probs = m.log_prob(action)

        value = critic(state)

        return action.item(), value.item(), log_probs.item()


def cumulative_sum(vector, discount):
    out = np.zeros_like(vector)
    n = vector.shape[0]
    for i in reversed(range(n)):
        out[i] =  vector[i] + discount * (out[i+1] if i+1 < n else 0)
    return out


def finish_trajectory(G):
    # Calculate trajectory rewards to go
    # Calculate trajectory GAE

    # REWARDS TO GO


    rewards = np.array(episode.rewards + [G])

    episode.rtgs = cumulative_sum(rewards, gamma)[:-1]

    # GAE
    values = np.array(episode.values + [G])

    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    episode.advantages = cumulative_sum(deltas, gamma * gae_lambda)

def normalize(vector):
    vector = np.array(vector)

    return (vector - vector.mean()) / (vector.std() + 1e-5)

def create_batches():

    n = len(history.rewards)
    batch_starts = np.arange(0, n, batch_size)
    index = np.arange(n, dtype=np.int64)
    np.random.shuffle(index)
    batches = [index[i:i+batch_size] for i in batch_starts]

    advantages = normalize(history.advantages)

    return torch.tensor(history.actions).to(device), torch.tensor(history.states).float().to(device), \
            torch.from_numpy(advantages).to(device), torch.tensor(history.rtgs).to(device), torch.tensor(history.log_probs).to(device), batches
    

def train(data_loader):
    for i in range(train_iters):
        

        for states, actions, advantages, log_probs, rtgs in data_loader:
            states = states.float().to(device)
            actions = actions.float().to(device)
            advantages = advantages.float().to(device)
            log_probs = log_probs.float().to(device)
            rtgs = rtgs.float().to(device)

            old_log_probs = log_probs
        
            # Compute Policy_loss
            logits = actor(states)

            m = Categorical(logits=logits)

            entropy = m.entropy()

            log_probs = m.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)

            actor_optimizer.zero_grad()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantages

            ActorLoss = -torch.min(surr1, surr2)

            values = critic(states)
            
            CriticLoss = F.mse_loss(values.squeeze(1), rtgs)

            loss = ActorLoss.mean() + 0.5*CriticLoss.mean() - 1e-2 * entropy.mean()

            writer.add_scalar("Policy Loss", ActorLoss.mean(), total_time_steps)
            writer.add_scalar("Value Loss", CriticLoss.mean(), total_time_steps)

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            # ActorLoss.backward()
            # CriticLoss.backward()

            loss.backward()

            actor_optimizer.step()
            critic_optimizer.step()

#create env

env_name = "LunarLander-v2"

env = gym.make(env_name)

# Try cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Actor and critic
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

actor = ActorNN(action_space, observation_space).to(device)
critic = CriticNN(observation_space).to(device)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)



history = HistoryBuffer()

state = env.reset()

# Params
max_time_steps = 200
gamma = 0.99
gae_lambda = 0.95

train_iters = 4
batch_size = 32

max_episodes = 10
max_epochs = 500

total_time_steps = 0

average_rewards = []
running_reward = 0
episode_reward = 0


writer = SummaryWriter()

for epoch in tqdm(range(max_epochs)):
    print(f"\nEpoch: {epoch}/{max_epochs}\n")

    for e in range(max_episodes): #Ara en serie, despres en paral·lel

        episode = EpisodeBuffer()
        

        for t in range(max_time_steps):
            # get action, log_probs and value
            action, value, log_prob = get_action(state)

            # Advance state
            next_state, reward, done, _ = env.step(action)

            #save action, log_prob, value, state, reward
            episode.actions.append(action)
            episode.values.append(value)
            episode.states.append(state)
            episode.rewards.append(reward)
            episode.log_probs.append(log_prob)

            state = next_state

            episode_reward += reward

            total_time_steps += 1

            if(done):
                state = env.reset()
                G = 0
                running_reward = 0.1 * episode_reward + (1 - 0.1) * running_reward
                # print(f"Episode: {e} Average reward {episode_reward:.2f} Running reward: {running_reward:.2f} Total time steps simulated: {total_time_steps}")
                average_rewards.append(episode_reward)

                writer.add_scalar("Average Episode Reward", episode_reward, total_time_steps)

                episode_reward = 0     
                break
            else:
                _, G, _ = get_action(state)


        # 4 Finish trajectory
        finish_trajectory(G)

        
        history.episodes.append(episode)
        
    # flatten data

    history.flatten_data()

    # Train

    data_loader = DataLoader(history, batch_size=batch_size, shuffle=True)

    train(data_loader)

    # Clean history
    history.reset_history()

    print(running_reward)

    # check if we have "solved" the cart pole problem
    if running_reward > env.spec.reward_threshold:
        print(f"Solved! Running reward is now {running_reward} and the last episode runs to {episode_reward} time steps!")
        break


plt.plot(average_rewards)
#plt.savefig(path)
plt.show()
#plt.clf()