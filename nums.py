import numpy as np

num_time_steps = 5
num_trajectories = 2



actions = np.zeros((num_trajectories, num_time_steps), dtype=np.float32)

actions[0][1]= 3
actions[0][3]= 3
actions[0][4]= 3

actions[1][0]= 4
actions[1][3]= 4
actions[1][4]= 4

print(actions.shape)

ones = np.ones((2,3))


print(ones.sum())
