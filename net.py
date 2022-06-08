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
    #print('1', x.shape)
    out = self.l1(x)

    #print('2', out.shape)
    out = self.l2(out)

    #print('3', out.shape)
    out = torch.flatten(out, 1)


    #out = out.view(32*32*3, 1)

    #print('4', out.shape)
    out = self.relu(self.fc1(out))

    #print('5', out.shape)
    out = self.relu(self.fc2(out))

    #print('6', out.shape)
    out = self.fc3(out)

    '''
    print('7', out.shape)
    out = torch.softmax(out)

    print('8', out.shape)
    '''
    return out
