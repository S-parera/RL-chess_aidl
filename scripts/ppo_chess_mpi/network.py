import torch.nn as nn 
import torch.nn.functional as F 
import torch

import torchvision.models as models

from torchsummary import summary

def PolicyNetwork():

    policy_model = models.resnet18(pretrained=False)
    policy_model.conv1 = nn.Conv2d(in_channels=33, out_channels = 64, kernel_size=3, padding=1)
    policy_model.fc = nn.Linear(in_features=512, out_features=4272)
    
    return policy_model

def ValueNetwork():

    value_model = models.resnet18(pretrained=False)
    value_model.conv1 = nn.Conv2d(in_channels=33, out_channels = 64, kernel_size=3, padding=1)
    value_model.fc = nn.Linear(in_features=512, out_features=1)

    return value_model

# policy = PolicyNetwork().cuda()
# critic = ValueNetwork().cuda()

# summary(policy.cuda(), (33,8,8))
# summary(critic, (33,8,8))