#!/usr/bin/env python3
import torch
import torch.nn as nn

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Sequential(
      nn.Conv2d(3, 6, kernel_size=5, padding=0, stride=1),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )
    self.l2 = nn.Sequential(
      nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )
    self.fc1 = nn.Linear(16*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.l1(x)
    out = self.l2(out)
    out = torch.flatten(out, 1)
    out = self.relu(self.fc1(out))
    out = self.relu(self.fc2(out))
    out = self.fc3(out)
    out = torch.softmax(out)
    return out
