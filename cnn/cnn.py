import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms # mnist dataset
from torchvision.utils import make_grid # for visualization

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Convert MNIST Image files into a 4D tensor (# images, height, width, colour channels)
transform = transforms.ToTensor()

# Load the training and test datasets
# set transform as transform to convert images to tensors
train_data = datasets.MNIST(root='./Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./Data', train=False, download=True, transform=transform)
print(f'train data: {train_data}')
print(f'test data: {test_data}')


