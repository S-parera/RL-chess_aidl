import torch.nn as nn 

class PolicyNetwork(nn.Module):
    def __init__(self, n=4, in_dim=128):
        super(PolicyNetwork, self).__init__()

        self.mlp = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, n)
        )

    def forward(self, x):

        y = self.mlp(x)

        return y



class ValueNetwork(nn.Module):
    def __init__(self, in_dim=128):
        super(ValueNetwork, self).__init__()

        self.mlp = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
        )

    def forward(self, x):
        
        y = self.mlp(x)

        return y.squeeze(1) 
