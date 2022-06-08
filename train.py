#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
from net import Net
import util
from torch.utils.data import DataLoader
import torchvision

import matplotlib.pyplot as plt

def train():
  #model = Net()
  model = models.resnet18()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  batch_size = 128
  epochs = 100

  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
  ])

  trainset = datasets.CIFAR10(root='./data', train=True,
                              download=False, transform=transform_train)
  trainloader = DataLoader(trainset, batch_size=batch_size,
                           shuffle=True, num_workers=2)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f'-> Device: {device}')
  model.to(device)

  classes = ('airplane','automobile','bird','cat','deer',
             'dog','frog','horse','ship','truck')

  for epoch in range(epochs):
    print(f'[*] Epoch [ {epoch}/{epochs} ]')
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data

      # zero-out the parameter gradients
      optimizer.zero_grad()

      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # stats
      running_loss += loss.item()
      if i % 2000 == 1999:
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0
  print('** DONE **')
  torch.save(model.state_dict(), './cifar_net.pth')

def evaluate():
  #model = Net()
  model = models.resnet18()
  batch_size = 100
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
  )
  
  test_data = datasets.CIFAR10(root='./data', train=False,
                               download=False, transform=transform)
  testloader = DataLoader(test_data, batch_size=batch_size,
                          shuffle=True, num_workers=2)
  classes = ['airplane','automobile','bird','cat','deer',
             'dog','frog','horse','ship','truck']

  model.load_state_dict(torch.load('./trained_models/cifar_net.pth'))
  total, correct = 0, 0
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

if __name__ == '__main__':
  #train()
  evaluate()

