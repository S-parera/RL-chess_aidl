import torch.nn as nn 
import torch.nn.functional as F 
import torch

import torchvision.models as models

from torchsummary import summary

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34(pretrained=False)
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
        self.model = models.resnet34(pretrained=False)
        self.model.conv1 = nn.Conv2d(in_channels=33, out_channels = 64, kernel_size=3, padding=1)
        self.model.fc = nn.Linear(in_features=512, out_features=1)
    

    def forward(self, x):
        y = self.model(x)

        return y


# policy = PolicyNetwork().cuda()
# critic = ValueNetwork().cuda()

# print(models.resnet34(pretrained=False))

# summary(policy.cuda(), (33,8,8))
# summary(critic, (33,8,8))