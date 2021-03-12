import torch.nn as nn 
import torch.nn.functional as F 
import torch

import torchvision.models as models

from torchsummary import summary

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(in_channels=33, out_channels = 64, kernel_size=3, padding=1, bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=2048)
        self.fc1 = nn.Linear(in_features=2048, out_features=4272)
    

    def forward(self, x):
        x = self.model(x)
        y = self.fc1(x)

        return y

class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(in_channels=33, out_channels = 64, kernel_size=3, padding=1)
        self.model.fc = nn.Linear(in_features=512, out_features=1)
        self.tanh = nn.Tanh()
    

    def forward(self, x):
        x = self.model(x)
        y = self.tanh(x)

        return y

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(in_channels=33, out_channels = 64, kernel_size=3, padding=1)
        self.model.fc = nn.Linear(in_features = 512,out_features = 2048)

        self.policy_head_fc = nn.Linear(in_features=2048,out_features=4272)
        self.value_head_fc = nn.Linear(in_features = 2048, out_features = 512)
        self.value_head_fc1 = nn.Linear(in_features = 512, out_features = 1)
        self.value_head_tanh = nn.Tanh()
        self.value_head_relu = nn.ReLU()

    def forward(self, x):

        y = self.model(x)

        policy = self.policy_head_fc(y)

        value = self.value_head_fc(y)
        value = self.value_head_relu(value)
        value = self.value_head_fc1(value)
        value = self.value_head_tanh(value)

        return policy, value



# print(models.resnet18(pretrained=False))
# policy = PolicyNetwork().cuda()
# critic = ValueNetwork().cuda()
# chess = ChessNN().cuda()
# summary(policy.cuda(), (33,8,8))
# summary(critic, (33,8,8))
# summary(chess, (33, 8, 8))