# import mnist dataset and create a mlp neural network using PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Configure device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

# Use seed for reproducibility
torch.manual_seed(42)

# Import mnist dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
print(f"length of train_dataset: {len(train_dataset)}")
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
print(f"length of test_dataset: {len(test_dataset)}")
# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
print(f"length of each step of train_loader: {len(train_loader)}")
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
print(f"length of each step of test_loader: {len(test_loader)}")

# Create a neural network
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__() ## Inherit from nn.Module
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28) # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
# Create a model
model = MLP()
# move model to gpu if available
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# count time taken to train the model
import time
start = time.time()

# Train the model
losses = []
print("Training the model")
epochs = 3
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader): # 60000 / 64 = 937.5
        images, labels = images.to(device), labels.to(device) # move images and labels to device
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, len(train_loader), loss.item()))

# count time taken to train the model
end = time.time()
elapsed = end - start
print(f"Time taken to train the model: {elapsed} seconds")

# plot
import matplotlib.pyplot as plt
# Plot the loss
plt.plot((range(epoch*len(train_loader) + i + 1)), losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Test the model
print("Testing the model")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'correct: {correct} / total: {total}')
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))