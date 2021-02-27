import torch
import gym
import numpy as np 
from torch.distributions.categorical import Categorical

from memory import Episode
from chess_env import ChessEnv


def collect(q, env_name, max_timesteps, state_scale,
            reward_scale, policy_model, value_model, gamma, lambda_gae, device):

    # Create and enviroment for every thread
    env = ChessEnv()

    observation = env.reset()
    ep_reward = 0
            
    episode = Episode()

    for timestep in range(max_timesteps):
        # Loop through time_steps

        action, log_probability, value, mask = get_action(observation / state_scale, policy_model, value_model, device, env)

        new_observation, reward, done = env.step(action)

        ep_reward += reward

        episode.observations.append(observation / state_scale)
        episode.actions.append(action)
        episode.rewards.append(reward / reward_scale)
        episode.values.append(value)
        episode.log_probabilities.append(log_probability)
        episode.masks.append(mask)

        observation = new_observation

        if done:
            end_episode(episode, 0, gamma, lambda_gae)
            break

        if timestep == max_timesteps - 1:
            # Episode didn't finish so we have to append value to RTGs and advantages
            _, _, value, _ = get_action(observation / state_scale, policy_model, value_model, device, env)
            end_episode(episode, value, gamma, lambda_gae)

    # Calc episode reward
    episode.episode_reward()
    # Return episode

    q.put(episode)

def get_action(state, policy_model, value_model, device, env):

    policy_model.eval()
    value_model.eval()

    if not state is torch.Tensor:
        state = torch.from_numpy(state).float().to(device)

    if state.shape[0] != 1:
        state = state.unsqueeze(0) # Create batch dimension

    logits = policy_model(state)

    legal_actions = torch.tensor(env.legal_actions()).to(device)
    mask = torch.zeros(4272).to(device)
    mask.index_fill_(0,legal_actions, 1)
    logits[0][mask == 0] = -float("Inf")

    m = Categorical(logits=logits)

    action = m.sample()

    log_probability = m.log_prob(action)

    value = value_model(state)

    return action.item(), log_probability.item(), value.item(), mask

def cumulative_sum(vector, discount):
    out = np.zeros_like(vector)
    n = vector.shape[0]
    for i in reversed(range(n)):
        out[i] =  vector[i] + discount * (out[i+1] if i+1 < n else 0)
    return out.tolist()

def end_episode(episode, last_value, gamma, gae_lambda):
    # REWARDS TO GO
    rewards = np.array(episode.rewards + [last_value])
    values = np.array(episode.values + [last_value])

    episode.rewards_to_go = cumulative_sum(rewards, gamma)[:-1]

    # GAE    
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]

    episode.advantages = cumulative_sum(deltas, gamma * gae_lambda)