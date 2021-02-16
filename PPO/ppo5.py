import torch
import torch.nn as nn 
from torch.distributions.categorical import Categorical
import torch.optim as optim

import numpy as np
import gym 
import matplotlib.pyplot as plt 

from torch.utils.tensorboard import SummaryWriter

class MemoryBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.values = []
        self.rewards =[]
        
        self.log_probs = []
        self.rtgs = []

        self.advantages = []

        self.episode_reward = 0


    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.values = []
        self.rewards =[]
        
        self.log_probs = []
        self.rtgs = []

        self.advantages = []

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


    rewards = np.array(buffer.rewards + [G])

    buffer.rtgs = cumulative_sum(rewards, gamma)[:-1]

    # GAE
    values = np.array(buffer.values + [G])

    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    buffer.advantages = cumulative_sum(deltas, gamma * gae_lambda)

def normalize(vector):
    return (vector - vector.mean()) / (vector.std() + 1e-8)

def create_batches():

    n = len(buffer.rewards)
    batch_starts = np.arange(0, n, batch_size)
    index = np.arange(n, dtype=np.int64)
    np.random.shuffle(index)
    batches = [index[i:i+batch_size] for i in batch_starts]

    advantages = normalize(buffer.advantages)

    return torch.tensor(buffer.actions).to(device), torch.tensor(buffer.states).float().to(device), \
            torch.from_numpy(advantages).to(device), torch.from_numpy(buffer.rtgs).to(device), torch.tensor(buffer.log_probs).to(device), batches
    

def train():
    for i in range(train_iters):
        actions, states, advantages, rtgs, old_log_probs, batches = create_batches()

        for batch in batches:
            # Compute Policy_loss
            logits = actor(states[batch])

            m = Categorical(logits=logits)

            entropy = m.entropy()

            log_probs = m.log_prob(actions[batch])
            ratio = (log_probs - old_log_probs[batch]).exp()

            actor_optimizer.zero_grad()
            surr1 = ratio * advantages[batch]
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantages[batch]

            ActorLoss = -torch.min(surr1, surr2).mean()

            values = critic(states[batch])
            
            CriticLoss = ((values - rtgs[batch])**2).mean()

            loss = ActorLoss + CriticLoss - 1e-4 * entropy.mean()

            writer.add_scalar("Policy Loss", ActorLoss, total_time_steps)
            writer.add_scalar("Value Loss", CriticLoss, total_time_steps)

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            # ActorLoss.backward()
            # CriticLoss.backward()

            loss.backward()

            actor_optimizer.step()
            critic_optimizer.step()

# Params
max_time_steps = 100
gamma = 0.99
gae_lambda = 0.95

train_iters = 4
batch_size = 5

#create env

env_name = "CartPole-v1"

env = gym.make(env_name)

# Try cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Actor and critic
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

actor = ActorNN(action_space, observation_space).to(device)
critic = CriticNN(observation_space).to(device)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)


buffer = MemoryBuffer()


state = env.reset()

#Reset memory buffer

episode_reward = 0
max_episodes = 500

total_time_steps = 0

average_rewards = []
running_reward = 0

max_iters = 100

writer = SummaryWriter(log_dir="./logs", filename_suffix=env_name, comment=env_name)

for e in range(max_episodes):

    buffer.reset_buffer()
    

    for t in range(max_time_steps):
        # get action, log_probs and value
        action, value, log_prob = get_action(state)

        # Advance state
        next_state, reward, done, _ = env.step(action)

        #save action, log_prob, value, state, reward
        buffer.actions.append(action)
        buffer.values.append(value)
        buffer.states.append(state)
        buffer.rewards.append(reward)
        buffer.log_probs.append(log_prob)

        state = next_state

        buffer.episode_reward += reward

        total_time_steps += 1

        if(done):
            state = env.reset()
            G = 0
            running_reward = 0.1 * buffer.episode_reward + (1 - 0.1) * running_reward
            print(f"Episode: {e} Average reward {buffer.episode_reward:.2f} Running reward: {running_reward:.2f} Total time steps simulated: {total_time_steps}")
            average_rewards.append(buffer.episode_reward)

            writer.add_scalar("Average Episode Reward", buffer.episode_reward, e)

            buffer.episode_reward = 0     
            break
        else:
            _, G, _ = get_action(state)


    # 4 Finish trajectory
    finish_trajectory(G)

    

    # Train

    train()

    # check if we have "solved" the cart pole problem
    if running_reward > env.spec.reward_threshold:
        print(f"Solved! Running reward is now {running_reward} and the last episode runs to {ep_reward} time steps!")
        break


plt.plot(average_rewards)
#plt.savefig(path)
plt.show()
#plt.clf()