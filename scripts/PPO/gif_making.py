from matplotlib import animation
import matplotlib.pyplot as plt
from collect_trajectories import get_action
from network import PolicyNetwork, ValueNetwork
import torch
from pathlib import Path
import gym

device = torch.device("cpu")

trained = False

# env_name = "CartPole-v1"
# env_name = "LunarLander-v2"
# env_name = "Acrobot-v1"
env_name = "MountainCar-v0"


env = gym.make(env_name)
n_actions = env.action_space.n
feature_dim = env.observation_space.shape[0]

policy_model = PolicyNetwork(in_dim=feature_dim, n=n_actions).to(device)
value_model = ValueNetwork(in_dim=feature_dim).to(device)

if trained:
    gif_name = env_name + "-trained.gif"

    if not Path("./models").exists():
        print("Creating Models folder")
        Path("./models").mkdir()

    model_path = Path("./models/" + env_name + ".tar")
    if model_path.exists():
        print("Loading model!")
        #Load model
        checkpoint = torch.load(model_path)
        policy_model.load_state_dict(checkpoint['policy_model'])
        value_model.load_state_dict(checkpoint['value_model'])
else:
    gif_name = env_name + "-untrained.gif"

def save_frames_as_gif(frames, filename, path='./gifs/'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=40)
    anim.save(path + filename, writer=animation.PillowWriter(fps=24))


def make_gif(env_name, policy_model, value_model, device):
#Make gym env
    env = gym.make(env_name)
    reward = 0
    #Run the env
    observation = env.reset()
    frames = []
    for t in range(1000):
        #Render to frames buffer
        frames.append(env.render(mode="rgb_array"))

        action, log_probability, value = get_action(observation, policy_model, value_model, device)

        observation, _reward, done, _ = env.step(action)
        reward += _reward
        if done:
            print(reward)
            break
    env.close()
    
    save_frames_as_gif(frames, gif_name)

make_gif(env_name, policy_model, value_model, device)