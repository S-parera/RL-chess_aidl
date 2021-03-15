# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 18:23:54 2021

@author: JOSEPMARIA
"""

import torch.nn as nn 
import torch.nn.functional as F 
import torch

# from torchsummary import summary

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class OutBlockactor(nn.Module):
    def __init__(self):
        super(OutBlockactor, self).__init__()     
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(8*8*128, 4272)
    
    def forward(self,s):
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 8*8*128)
        p = self.fc(p)
        return p


class OutBlockcritic(nn.Module):
    def __init__(self):
        super(OutBlockcritic, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8, 64)
        self.fc2 = nn.Linear(64, 1)
   
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 8*8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        return v


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=21, out_channels=256, kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1=nn.ReLU(inplace=True)
        for block in range(13):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblockactor = OutBlockactor()


    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        for block in range(13):
            y = getattr(self, "res_%i" % block)(y)
        y = self.outblockactor(y)
        return y

class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=21, out_channels=256, kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1=nn.ReLU(inplace=True)
        for block in range(13):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblockcritic = OutBlockcritic()
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        for block in range(13):
            y = getattr(self, "res_%i" % block)(y)
        y = self.outblockcritic(y)
        return y.squeeze(1)

# policy = PolicyNetwork().cuda()
# critic = ValueNetwork().cuda()

# summary(policy, (21,8,8))
# summary(critic, (21,8,8))

