#from a2c import A2C
from ppo import PPO



gamma = [0.99]
lr = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3,1e-2, 5e-2, 1e-1, 5e-1, 1]



for e in gamma:
    for i in lr:
        name = f"ppo-g{e}-lr{i}"
        print("Running " + name)
        model = PPO(i,e,name)
        model.train()
        