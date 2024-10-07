import torch
import torch.nn as nn # neural network
import torch.nn.functional as F # activation functions

# Create model class that inherits from nn.Module
class Model(nn.Module):
    # Input layer (4 features of iris dataset)
    # Hidden layer 1 (8 neurons)
    # Hidden layer 2 (8 neurons)
    # Output layer (3 classes)
    def __init__(self, in_features=4, h1=8, h2=8, out_features=3):
        super().__init__() # Inherits the methods from nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    def forward(self, x):
        x = F.relu(self.fc1(x)) # activation function for hidden layer 1
        x = F.relu(self.fc2(x)) # activation function for hidden layer 2
        x = self.out(x) # return output layer
        return x

seed = 41
torch.manual_seed(seed) # set seed for reproducibility
model = Model() # create model instance

# Load iris dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)
# print(my_df.head())

# change output from string to integer (last column)
my_df['species'] = my_df['species'].map({'setosa': 0,
                                             'versicolor': 1,
                                             'virginica': 2
                                             })
# print(my_df.head())

# Train Test Split
X = my_df.drop('species', axis=1) # drop the 'species' axis == 1 for column
y = my_df['species'] # only the 'species' column
# print(X.head())
# print(y.head())

from sklearn.model_selection import train_test_split
# 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

X_train = torch.FloatTensor(X_train.values) # convert to tensor (floats)
X_test = torch.FloatTensor(X_test.values)

y_train = torch.LongTensor(y_train.values) # convert to tensor (long)
y_test = torch.LongTensor(y_test.values)

# Set criterion
criterion = nn.CrossEntropyLoss() # loss function for multi-class classification

# Set optimizer and learning rate (lower learning rate if error doesn't decrease)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # optimizer
# print(model.parameters)

# Train the model
epochs = 120
losses = []
for i in range(epochs):
    y_pred = model.forward(X_train) # forward pass
    loss = criterion(y_pred, y_train) # calculate loss

    losses.append(loss.detach().numpy()) # append loss to list
    if i % (epochs/10) == 0:
        print(f'Epoch {i} and loss is: {loss}')

    # Backpropagation
    optimizer.zero_grad() # zeros gradients
    loss.backward() # backpropagation
    optimizer.step() # update weights

# Plot the loss
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
# plt.show()

# export model
torch.save(model.state_dict(), 'my_iris_model.pt')

# Calculate loss
with torch.no_grad():
    y_val = model.forward(X_test)
    loss = criterion(y_val, y_test)
print(f'Loss: {loss:.8f}')





