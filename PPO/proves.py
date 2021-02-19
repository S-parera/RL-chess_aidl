from chess_env import ChessEnv
import torch

env = ChessEnv()

state = env.reset(initial_state=True)

legal_actions = env.legal_actions()

print(len(legal_actions))

print(legal_actions)

mask = torch.tensor(legal_actions)

print(mask)


# a = torch.tensor([1,2,3,4]).float()
# print(a.shape)
# mask = torch.tensor([0,3])

# a.index_fill_(0,mask, -float("Inf"))

# print(a)


env.close()