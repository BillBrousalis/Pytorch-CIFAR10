#!/usr/bin/env python3
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np
#from net import Net
import util

import matplotlib.pyplot as plt

def TrainEval():
  def __init__(self):
    self.model = models.resnet18()
    print(f'DEVICE :: [ {device} ]')
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.classes = ('airplane','automobile','bird','cat','deer',
                    'dog','frog','horse','ship','truck')
    self.fname = "resnet18_sgd_lr0,01.pth"
    self.train()
    self.evaluate()

  def train(self):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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

    self.model.to(self.device)
    # graph
    x, y = [], []

    for epoch in tqdm(epochs):
      print(f'[*] Epoch [ {epoch}/{epochs} ]')
      running_loss = 0.0
      avg_epoch_loss = 0.0
      for i, data in enumerate(trainloader):
        print(f"i={i}")
        inputs, labels = data

        # zero-out the parameter gradients
        optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # stats
        running_loss += loss.item()
        avg_epoch_loss += loss.item
        if i % 2000 == 1999:
          print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
          running_loss = 0.0

      avg_epoch_loss /= len(trainloader)
      print("AVG_LOSS = {avg_epoch_loss}")

      # graph
      x.append(epoch+1)
      y.append(avg_epoch_loss)

    print('** DONE **')
    torch.save(self.model.state_dict(), os.path.join("./trained_models/", self.fname)

    # graph
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel="EPOCH #", ylabel="LOSS", title="Loss Graph")
    ax.grid()
    fig.savefig(os.path.join("./trained_models/", f"{self.fname}.png")
    plt.show()

  def evaluate(self, self.fname):
    batch_size = 100
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    test_data = datasets.CIFAR10(root='./data', train=False,
                                 download=False, transform=transform)
    testloader = DataLoader(test_data, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    
    if self.device == "cpu":
      self.model.load_state_dict(torch.load(os.path.join("./trained_models/", self.fname), map_location=torch.device('cpu')))
    else:
      self.model.load_state_dict(torch.load(os.path.join("./trained_models/", self.fname)))
    total, correct = 0, 0
    with torch.no_grad():
      for data in testloader:
        images, labels = data
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {len(test_data)} test images: {100 * correct / total} %')

if __name__ == '__main__':
  train()
  evaluate()

