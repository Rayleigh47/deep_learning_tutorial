import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms # mnist dataset
from torchvision.utils import make_grid # for visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Convert MNIST Image files into a 4D tensor (# images, height, width, colour channels)
transform = transforms.ToTensor()

# Load the test data
test_data = datasets.MNIST(root='./Data', train=False, download=True, transform=transform)
print(f'test data: {test_data}')

# Create small batch size for images
test_loader = DataLoader(test_data, batch_size=10, shuffle=False, num_workers=4) # shuffle is False

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

# Load the model
model = ConvolutionalNetwork()

# Import trained model
model.load_state_dict(torch.load('cnn_model.pt', weights_only=True))
model.eval() # set model to evaluation mode

# set the seed
seed = 42
torch.manual_seed(seed)

# optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # Smaller learning rate, takes longer to converge

# Test the model
test_losses = []
test_correct = []

with torch.no_grad():
    tst_correct = 0
    total_samples = 0
    for batch, (X_test, y_test) in enumerate(test_loader):
        # Apply the model
        y_val = model(X_test) # Not flattened yet, conv2d
        predicted = torch.max(y_val.data, 1)[1] # add up number of correct predictions
        tst_correct += (predicted == y_test).sum()
        total_samples += y_test.size(0)  # Increment total samples

        # Update the test loss and accuracy
        loss = criterion(y_val, y_test)
        test_losses.append(loss.cpu().item())

# Calculate the test accuracy
accuracy = 100 * tst_correct / total_samples
print(f'Test Accuracy: {accuracy.item()}%')


# Graph the results
plt.plot(test_losses, label="Test Loss")
plt.title("Loss at the end of each batch")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
# plt.show()

# Graph the accuracy at the end of the test
plt.plot([accuracy], label="Test Accuracy", marker='o')  # Just one point for accuracy
plt.title("Accuracy at the end of the test")
plt.xlabel("Test")
plt.ylabel("Accuracy (%)")
plt.legend()
# plt.show()

# Grab a test image
test_data = test_data[4143][0].reshape(28, 28) # 28x28 image
plt.imshow(test_data, cmap='gray')
plt.show()

# Pass image through model
with torch.no_grad():
    new_image = test_data.view(1, 1, 28, 28) # 1 image, 1 channel, 28x28 image
    y_val = model(new_image)
    print(f'Prediction: {y_val.argmax().item()}')
