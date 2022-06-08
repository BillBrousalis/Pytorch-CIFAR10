#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def imgshow(batch_idx, idx):
  dat = unpickle(f'cifar-10-batches-py/data_batch_{batch_idx}')
  meta = unpickle('cifar-10-batches-py/batches.meta')
  img = dat[b'data'][idx]
  RGB = {}
  for i, key in enumerate(list('RGB')):
    RGB[key] = img[1024*i:1024*(i+1)].reshape(32, 32)
  img = np.dstack([mat for mat in RGB.values()])
  print(f"Showing: {meta[b'label_names'][dat[b'labels'][idx]].decode()}")
  plt.imshow(img)
  plt.show()

def load_imgs(batch_idx):
  dat = unpickle(f'cifar-10-batches-py/data_batch_{batch_idx}')
  data = []
  for i in range(getsize(batch_idx)):
    img = dat[b'data'][i]
    imgnorm = []
    for i in range(len(img)):
      imgnorm.append(np.float32(img[i]) / 255.0)
    imgnorm = np.array(imgnorm)
    RGB = {}
    for i, key in enumerate(list('RGB')):
      RGB[key] = imgnorm[1024*i:1024*(i+1)].reshape(32, 32)
    imgnorm = np.dstack([mat for mat in RGB.values()])
    data.append(imgnorm)
  return data

def load_labels(batch_idx):
  dat = unpickle(f'cifar-10-batches-py/data_batch_{batch_idx}')
  labels = []
  for label in dat[b'labels']:
    labels.append(label)
  return labels

def getsize(batch_idx):
  dat = unpickle(f'cifar-10-batches-py/data_batch_{batch_idx}')
  batch = dat[b'data']
  return batch.shape[0]
  
if __name__ == '__main__':
  #imgshow(1, 12)
  load_labels(1)
