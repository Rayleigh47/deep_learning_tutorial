import torch
import numpy as np

"""
Tensors are similar to numpy ndarrays
Works better on GPUs
Default data type of float32
More suitable for deep learning than numpy arrays
"""

# Create a python list
my_list = [1, 2, 3, 4, 5]
print(f"Python list: {my_list}")

# convert python list to numpy array
np_array = np.array(my_list)
print(f"Numpy array: {np_array}")

np1 = np.random.rand(3,4)
print(f"Numpy random array:\n {np1}")

# convert numpy array to tensor
tensor_2d = torch.from_numpy(np1)
print(f"2D Tensor:\n {tensor_2d}")

# Create a 3D tensor
tensor_3d = torch.rand(3, 4, 5)
print(f"3D Tensor:\n {tensor_3d}")

# shape of tensor
print(f"Shape of tensor: {tensor_3d.shape}")

# tensor operations
my_torch = torch.arange(15)
print(f"Tensor: {my_torch}")

# Reshape tensor
my_torch = my_torch.reshape(5, 3) # size has to be the same
print(f"Reshaped tensor 1: {my_torch}")

# Reshape if we don't know the number of items
my_torch = my_torch.reshape(3, -1)
print(f"Reshaped tensor 2: {my_torch}")

my_torch = my_torch.reshape(-1, 5)
print(f"Reshaped tensor 3: {my_torch}")

# view is contiguous in memory, data is stored in an unbroken sequence
# generally faster than reshape because it doesn't copy the data, it just changes the view
# reshape can handle non-contiguous data, it will internally make a copy of the tensor and
# reuturn a reshaped version of the tensor
my_torch = torch.arange(10).view(2, 5)
print(f"Reshaped tensor 4: {my_torch}")

# with reshape and view they will update accordingly when original tensor is updated
my_torch5 = torch.arange(10)
print(f"Tensor 5: {my_torch5}")
my_torch6 = my_torch5.reshape(2,5) # have to be careful when using reshape of another tensor
print(f"Reshaped tensor 6: {my_torch6}")

my_torch5[1] = 4141
print(f"Tensor 5: {my_torch5}")
print(f"Reshaped tensor 6: {my_torch6}")

# Grab a specific item
my_torch = torch.arange(10)
print(f"Tensor 7: {my_torch}")
print(f"Item 7: {my_torch[7]}")

# Grab a slice
my_torch = my_torch.reshape(5, 2) # 5 rows, 2 columns
print(f"Tensor 8: {my_torch}")
print(f"Slice column 1 as row: {my_torch[:,1]}")

# return column
print(f"Slice column 1 as col: {my_torch[:,1:]}")


