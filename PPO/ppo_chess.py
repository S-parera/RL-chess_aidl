import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import matplotlib.pyplot as plt
import gym

from python_chess_addons import *
import chess

class MemoryBuffer:
    def __init__(self, num_trajectories, time_steps, observation_space, action_space):
        self.states = []
        self.actions = []
        self.values = np.zeros((num_trajectories, num_time_steps), dtype=np.float32)
        self.rewards = np.zeros((num_trajectories, num_time_steps), dtype=np.float32)
        
        self.log_probs = np.zeros((num_trajectories, num_time_steps), dtype=np.float32)
        self.rtg = np.zeros((num_trajectories, num_time_steps), dtype=np.float32)

        self.advantages = np.zeros((num_trajectories, num_time_steps), dtype=np.float32)

        self.num_trajectories = num_trajectories
        self.time_steps = time_steps
        self.observation_space = observation_space
        self.action_space = action_space

        self.mask = []


    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.values = np.zeros((num_trajectories, num_time_steps), dtype=np.float32)
        self.rewards = np.zeros((num_trajectories, num_time_steps), dtype=np.float32)
        
        self.log_probs = np.zeros((num_trajectories, num_time_steps), dtype=np.float32)
        self.rtg = np.zeros((num_trajectories, num_time_steps), dtype=np.float32)

        self.advantages = np.zeros((num_trajectories, num_time_steps), dtype=np.float32)

        self.mask = []

class ActorNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1=nn.Conv2d(in_channels=21, out_channels=21*4, kernel_size=3,padding=1)
    self.relu1=nn.ReLU(inplace=True)
    self.maxpool1=nn.MaxPool2d(2)
    self.conv2=nn.Conv2d(in_channels=21*4, out_channels=21*4*4, kernel_size=3,padding=1)
    self.relu2=nn.ReLU(inplace=True)
    self.maxpool2 = nn.MaxPool2d(2)
    self.conv3=nn.Conv2d(in_channels=21*4*4, out_channels=21*4*4*4, kernel_size=3,padding=1)
    self.relu3=nn.ReLU(inplace=True)
    self.maxpool3=nn.MaxPool2d(2)
    self.mlp = nn.Sequential(
        nn.Linear(1344,2688),
        nn.ReLU(inplace=True),
        nn.Linear(2688,4272)
    )
    # Need to add more layers to make the receptive field big enough to capture the entire board. Consider including kernels of size 5 or 3, 3 or 4 layers. 
    # Better not to use padding as it might consider movements outside of the board.
    self.saved_log_probs = []
    self.rewards = []
  def forward(self, x):
    y = self.conv1(x)
    y = self.relu1(y)
    y = self.maxpool1(y)
    y = self.conv2(y)
    y = self.relu2(y)
    y = self.maxpool2(y)
    y = self.conv3(y)
    y = self.relu3(y)
    y = self.maxpool3(y)
    y = y.view(x.size(0), -1) 
    y = self.mlp(y)
    
#    y = F.softmax(y)
    return y

class CriticNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1=nn.Conv2d(in_channels=21, out_channels=21*4, kernel_size=3,padding=1)
    self.relu1=nn.ReLU(inplace=True)
    self.maxpool1=nn.MaxPool2d(2)
    self.conv2=nn.Conv2d(in_channels=21*4, out_channels=21*4*4, kernel_size=3,padding=1)
    self.relu2=nn.ReLU(inplace=True)
    self.maxpool2 = nn.MaxPool2d(2)
    self.conv3=nn.Conv2d(in_channels=21*4*4, out_channels=21*4*4*4, kernel_size=3,padding=1)
    self.relu3=nn.ReLU(inplace=True)
    self.maxpool3=nn.MaxPool2d(2)
    self.mlp = nn.Sequential(
        nn.Linear(1344,512),
        nn.ReLU(inplace=True),
        nn.Linear(512,1)
    )
    # Need to add more layers to make the receptive field big enough to capture the entire board. Consider including kernels of size 5 or 3, 3 or 4 layers. 
    # Better not to use padding as it might consider movements outside of the board.
    self.saved_log_probs = []
    self.rewards = []
  def forward(self, x):
    y = self.conv1(x)
    y = self.relu1(y)
    y = self.maxpool1(y)
    y = self.conv2(y)
    y = self.relu2(y)
    y = self.maxpool2(y)
    y = self.conv3(y)
    y = self.relu3(y)
    y = self.maxpool3(y)
    y = y.view(x.size(0), -1) 
    y = self.mlp(y)
#    y = F.softmax(y)
    return y



def get_action(state,board):
        #state = torch.from_numpy(state).float()

        mask=get_legal_moves_mask(list(board.legal_moves))

        buffer.mask.append(torch.tensor(mask))

        logits = actor(state.unsqueeze(dim=0))

        logits = logits[:,mask]

        m = Categorical(logits=logits)

        action = m.sample()

        move_mask = create_move_mask()
        a = list(board.legal_moves)[action]
        b = move_mask[str(a)]


        log_probs = m.log_prob(action)

        value = critic(state.unsqueeze(dim=0))

        move = list(board.legal_moves)[action]

        return move, value, log_probs


def cumulative_sum(vector, discount):
    out = np.zeros_like(vector)
    n = vector.shape[0]
    for i in reversed(range(n)):
        out[i] =  vector[i] + discount * (out[i+1] if i+1 < n else 0)
    return out


def finish_trajectory(G, trajectory, t):
    # Calculate trajectory rewards to go
    # Calculate trajectory GAE

    # REWARDS TO GO

    trajectory_rewards = buffer.rewards[trajectory]
    trajectory_rewards = np.append(trajectory_rewards, G.detach().numpy())

    buffer.rtg[trajectory] = cumulative_sum(trajectory_rewards, gamma)[:-1]

    # GAE
    trajectory_values = buffer.values[trajectory]
    trajectory_values = np.append(trajectory_values, G.detach().numpy())

    deltas = trajectory_rewards[:-1] + gamma * trajectory_values[1:] - trajectory_values[:-1]
    buffer.advantages[trajectory] = cumulative_sum(deltas, gamma * gae_lambda)

def normalize(vector):
    return (vector - vector.mean()) / (vector.std() + 1e-8)
    

def create_batches():
    advantages = np.empty((0, 1), dtype=np.float32)
    actions = np.empty((0, 1), dtype=np.float32)
    states = np.empty((0, observation_space), dtype=np.float32)
    rtgs = np.empty((0, 1), dtype=np.float32)
    log_probs = np.empty((0, 1), dtype=np.float32)


    for trajectory in range(num_trajectories):
        trajectory_slice = (np.trim_zeros(buffer.advantages[trajectory])).shape[0]
        advantages = np.append(advantages, np.trim_zeros(buffer.advantages[trajectory]))
        rtgs = np.append(rtgs, buffer.rtg[trajectory][0:trajectory_slice])
        log_probs = np.append(log_probs, buffer.log_probs[trajectory][0:trajectory_slice])



    # print("adv len: ", advantages.shape[0])
    # print("act len: ", actions.shape[0])
    # print("state len: ", states.shape[0])
    # print("rtg len: ", rtgs.shape[0])
    # print("log len: ", log_probs.shape[0])

    # advantages = normalize(advantages)

    n = advantages.shape[0]
    batch_starts = np.arange(0, n, batch_size)
    index = np.arange(n, dtype=np.int64)
    np.random.shuffle(index)
    batches = [index[i:i+batch_size] for i in batch_starts]

    return buffer.actions, torch.stack(buffer.states), torch.from_numpy(advantages), \
            torch.from_numpy(rtgs), torch.from_numpy(log_probs), torch.stack(buffer.mask), batches
    

def train():
    for i in range(train_iters):
        actions, states, advantages, rtgs, old_log_probs, mask, batches = create_batches()

        for batch in batches:
            # Compute Policy_loss
            print(states[[0,2]])
            logits = actor(states[batch], mask[batch])
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

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            # ActorLoss.backward()
            # CriticLoss.backward()

            loss.backward()

            actor_optimizer.step()
            critic_optimizer.step()

def evaluate(render):
    state, done = env2.reset(), False
    total_rw = 0
    while not done:
        if render:
            env2.render()
        action, value, log_prob = get_action(state)
        # Step
        state, reward, done, _ = env2.step(action)
        total_rw += reward

    env2.close()

    return total_rw
    

num_episodes = 1500
num_trajectories = 1
num_time_steps = 50
batch_size = 5
train_iters = 4



total_time_steps = 0

# env = gym.make("CartPole-v1")
# env2 = gym.make("CartPole-v1")
# env = gym.make("LunarLander-v2")
# env2 = gym.make("LunarLander-v2")

env = gym.make("Acrobot-v1")
env2 = gym.make("Acrobot-v1")


observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

actor = ActorNN()
critic = CriticNN()

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

buffer = MemoryBuffer(num_trajectories, num_time_steps, observation_space, action_space)

gamma = 0.99
gae_lambda = 0.95

average_rewards = []
running_reward = -500

traj = 0

episode = 1
ep_reward = 0
EP_reward = 0

#state = env.reset()
board = chess.Board(get_FEN(1))
state = BoardEncode(board)


while num_episodes > episode:
    
    buffer.reset_buffer()

    # state = env.reset()
    done = False

    for t in range(num_time_steps): #time_steps
        # get action
        action, value, log_prob = get_action(state, board)

        # Step
        board.push(action)
        print(board)
        next_state = BoardEncode(board)
        # next_state, reward, done, _ = env.step(action)

        if(board.is_game_over() == True):
            done = True

        reward = 0

        #collect data
        buffer.actions.append(action)
        buffer.values[traj][t] = value
        buffer.log_probs[traj][t] = log_prob
        buffer.states.append(state)
        buffer.rewards[traj][t] = reward

        state = next_state

        EP_reward += reward

        if (done):

            #state = env.reset()
            board = chess.Board(get_FEN(episode))
            state = BoardEncode(board)

            if(board.is_checkmate() == True):
                reward = 1
                buffer.rewards[traj][t] += reward

            episode +=1
            ep_reward = EP_reward
            EP_reward = 0
            running_reward = 0.1 * ep_reward + (1 - 0.1) * running_reward
            print(f"Episode: {episode} Average reward {ep_reward:.2f} Running reward: {running_reward:.2f} Total time steps simulated: {total_time_steps}")
            average_rewards.append(ep_reward)
            break


    # Calculate reward as advantage
    if(not done):
        _, G, _ = get_action(state, board)
    else:
        G = torch.tensor([0])

    # 4 Finish trajectory
    finish_trajectory(G, traj, t)

    train()


    total_time_steps += t
   
    
    #running_reward = 0.1 * ep_reward + (1 - 0.1) * running_reward

    

    # check if we have "solved" the cart pole problem
    if running_reward > env.spec.reward_threshold:
        print(f"Solved! Running reward is now {running_reward} and the last episode runs to {ep_reward} time steps!")
        break

    

    

plt.plot(average_rewards)
#plt.savefig(path)
plt.show()
#plt.clf()
    
a = evaluate(render=True)