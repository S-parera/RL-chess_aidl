from chess_env import ChessEnv
import torch

env = ChessEnv()

state = env.reset(initial_state=True)

legal_actions = env.legal_actions()

mask = torch.zeros(10)

legal_actions = torch.tensor([2,5,7,9])

mask.index_fill_(0,legal_actions, 1)

print(mask)

illegal_actions = torch.tensor(range(10)).float()
print(illegal_actions)

illegal_actions[mask == 0] = -float("Inf")

print(illegal_actions)



# a = torch.tensor([1,2,3,4]).float()
# print(a.shape)
# mask = torch.tensor([0,3])

# a.index_fill_(0,mask, -float("Inf"))

# print(a)


env.close()