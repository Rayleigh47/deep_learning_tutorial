import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms # mnist dataset
from torchvision.utils import make_grid # for visualization
import tqdm
from tqdm import trange # for progress bar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Convert MNIST Image files into a 4D tensor (# images, height, width, colour channels)
transform = transforms.ToTensor()

# Load the training and test datasets
# set transform as transform to convert images to tensors
train_data = datasets.MNIST(root='./Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./Data', train=False, download=True, transform=transform)
print(f'train data: {train_data}')
print(f'test data: {test_data}')

# Create small batch size for images
train_loader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False, num_workers=4) # shuffle is False for test data

# Manually do CNN
# conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1)
# conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)

# # Load the first image
# for index, (X_train, y_train) in enumerate(train_data):
#     print(f'X_train: {X_train.shape}')
#     # print(f'y_train: {y_train.shape}')
#     break

# x = X_train.view(1,1,28,28)
# x = F.relu(conv1(x)) # conv1 -> relu
# print(x.shape) # torch.Size([1, 6, 26, 26]) -> 1 image, 6 out_channels, 26x26 image size, not using padding of the 28x28 image
# x = F.max_pool2d(x, 2, 2) # 2x2 kernel, stride 2
# print(x.shape) # torch.Size([1, 6, 13, 13]) -> 1 image, 6 out_channels, 13x13 image size
# x = F.relu(conv2(x)) # conv2 -> relu
# print(x.shape) # torch.Size([1, 16, 11, 11]) -> 1 image, 16 out_channels, 11x11 image size, not using padding of the 13x13 image
# x = F.max_pool2d(x, 2, 2) # 2x2 kernel, stride 2
# print(x.shape) # torch.Size([1, 16, 5, 5]) -> 1 image, 16 out_channels, 5x5 image size, 11/2 =5.5, so 5x5 image size (can't invent pixels)


# Define a convolutional neural network
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1) # 1 input channel, 6 output channels, 3x3 kernel, stride 1
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1) # 6 input channel, 16 output channels, 3x3 kernel, stride 1
        self.fc1 = nn.Linear(5*5*16, 120) # 5x5 image dimension, 16 channels, 120 output, refer to part above for the calculation
        self.fc2 = nn.Linear(120, 84) # 120 input, 84 output
        self.fc3 = nn.Linear(84, 10) # 84 input, 10 output (0-9)
    
    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2) # 2x2 kernel, stride 2
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2) # 2x2 kernel, stride 2
        X = X.view(-1, 5*5*16) # flatten the tensor, negative one so that we can vary the batch size
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1) # log softmax for multi-class classification
    
# Instantiate the model
model = ConvolutionalNetwork()
print(model)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model = model.to(device)

# set the seed
seed = 42
torch.manual_seed(seed)

# optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # Smaller learning rate, takes longer to converge

# time to train the model
import time
start_time = time.time()
epochs = 5

# Track variables
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# Start training
for i in trange(epochs, desc="Epochs"):
    # Training
    trn_correct = 0
    tst_correct = 0
    for batch, (X_train, y_train) in enumerate(train_loader):
        # load into GPU
        X_train, y_train = X_train.to(device), y_train.to(device)
        # increment batch
        batch += 1
        # Apply the model
        y_pred = model(X_train) # Not flattened yet, conv2d
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1] # add up number of correct predictions
        batch_correct = (predicted == y_train).sum()
        trn_correct += batch_correct # keep track of correct predictions

        # Update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 600 == 0:
            print(f"Epoch: {i} Batch: {batch} Loss: {loss.item()}")
    
    # Update the training loss and accuracy
    train_losses.append(loss)
    train_correct.append(trn_correct)

# Test the model
with torch.no_grad():
    for batch, (X_test, y_test) in enumerate(test_loader):
        # load into GPU
        X_test, y_test = X_test.to(device), y_test.to(device)
        # Apply the model
        y_val = model(X_test) # Not flattened yet, conv2d
        predicted = torch.max(y_val.data, 1)[1] # add up number of correct predictions
        tst_correct += (predicted == y_test).sum()

    # Update the test loss and accuracy
    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_correct)

# calculate time taken
end_time = time.time()
duration = end_time - start_time
print(f"Time taken: {duration/60:.2f} minutes")

# export trained model
torch.save(model.state_dict(), 'cnn_model.pt')



