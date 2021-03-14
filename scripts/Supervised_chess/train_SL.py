"""
Pretrain policy_network as SL
Load Fens and moves dataset
"""

from network import PolicyNetwork
import pandas as pd
import numpy as np
from io import StringIO
from chess_env import ChessEnv
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time

train_ite = 0
test_ite = 0

class TrainDataset(Dataset):
    def __init__(self):
        # Read csv
    
        loaded = np.load("./train_dataset.npz")
        self.x = loaded['fens']
        self.y = loaded['moves']

        # Convert to tensor
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TestDataset(Dataset):
    def __init__(self):
        # Read csv
    
        loaded = np.load("./test_dataset.npz")
        self.x = loaded['fens']
        self.y = loaded['moves']

        # Convert to tensor
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ValDataset(Dataset):
    def __init__(self):
        # Read csv
    
        loaded = np.load("./val_dataset.npz")
        self.x = loaded['fens']
        self.y = loaded['moves']

        # Convert to tensor
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]





def correct_predictions(predicted_batch, label_batch):
  pred = predicted_batch.argmax(dim=1, keepdim=True) # get the index of the max log-probability
  acum = pred.eq(label_batch.view_as(pred)).sum().item()
  return acum

def train_epoch(train_loader, network, optimizer, criterion):
  # Activate the train=True flag inside the model
  global train_ite
  network.train()
  device = "cuda"
  avg_loss = None
  avg_weight = 0.1
  for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      logits = network(data)
      loss = criterion(logits, target)
      loss.backward()
      if avg_loss:
        avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
      else:
        avg_loss = loss.item()
      optimizer.step()

      writer.add_scalar("Train loss", loss, train_ite)
      train_ite +=1
      
      if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader), loss.item()))
  return avg_loss

def test_epoch(test_loader, network):
    network.eval()
    device = 'cuda'
    test_loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = network(data)
            loss = criterion(logits, target, reduction='sum').item()
            test_loss += loss # sum up batch loss
            # compute number of correct predictions in the batch
            acc_batch = correct_predictions(logits, target)
            acc += acc_batch

    # Average acc across all correct predictions batches now
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * acc / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset), test_acc,
        ))
    return test_loss, test_acc

tr_losses = []
te_losses = []
te_accs = []

batch_size = 128
num_epochs = 25
log_interval = 100

model = PolicyNetwork()
train_dataset = TrainDataset()
test_dataset = TestDataset()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=True, drop_last=True)


model.to('cuda')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = F.cross_entropy

timestr = time.strftime("%d%m%Y-%H%M%S-")

log_dir = "./runs/" + timestr + 'SLResnet34' 


writer = SummaryWriter(log_dir=log_dir)

# LOAD MODEL
# Create folder models
if not Path("./models").exists():
    print("Creating Models folder")
    Path("./models").mkdir()

model_path = Path("./models/" + 'SLResnet' + ".tar")
if model_path.exists():
    print("Loading model!")
    #Load model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['policy_model'])
    optimizer.load_state_dict(checkpoint['policy_optimizer'])

for epoch in tqdm(range(1, num_epochs + 1)):
    tr_loss = train_epoch(train_loader, model, optimizer, criterion)
    tr_losses.append(tr_loss)
    te_loss, te_acc = test_epoch(test_loader, model)
    te_losses.append(te_loss)
    te_accs.append(te_acc)
    
    writer.add_scalar("Test loss", te_loss, epoch)
    writer.add_scalar("Test accuracy", te_acc, epoch)

    torch.save({
        'policy_model': model.state_dict(),
        'policy_optimizer': optimizer.state_dict()}, model_path)

# plt.figure(figsize=(10, 8))
# plt.subplot(2,1,1)
# plt.xlabel('Epoch')
# plt.ylabel('NLLLoss')
# plt.plot(tr_losses, label='train')
# plt.show()
# plt.plot(te_losses, label='test')
# plt.show()
# plt.legend()
# plt.subplot(2,1,2)
# plt.xlabel('Epoch')
# plt.ylabel('Test Accuracy [%]')
# plt.plot(te_accs)
# plt.show()


