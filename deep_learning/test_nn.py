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

# Load iris dataset
from sklearn.model_selection import train_test_split
# 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

X_train = torch.FloatTensor(X_train.values) # convert to tensor (floats)
X_test = torch.FloatTensor(X_test.values)

y_train = torch.LongTensor(y_train.values) # convert to tensor (long)
y_test = torch.LongTensor(y_test.values)

# import trained model
model.load_state_dict(torch.load('iris/my_iris_model.pt', weights_only=True))
model.eval() # set model to evaluation mode

# Set criterion
criterion = nn.CrossEntropyLoss() # loss function for multi-class classification

# Evaluate model on Test Data set
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        print(f"Prediction: {y_val.argmax().item()}, True: {y_test[i]}")
        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f"correct predictions: {correct} out of {X_test.shape[0]}")

# try new iris data
new_iris = torch.tensor([5.6, 3.7, 2.2, 0.5])
with torch.no_grad(): # turn off gradient calculation
    print(model(new_iris))


