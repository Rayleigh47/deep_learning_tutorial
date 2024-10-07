import torch
import numpy as np
# tensor math
# add subtract multiply divide remainder exponents

tensor_a = torch.tensor([1, 2, 3, 4])
tensor_b = torch.tensor([5, 6, 7, 8])

# add
tensor_add = tensor_a + tensor_b
print(f"Addition: {tensor_add}")

tensor_add = torch.add(tensor_a, tensor_b)
print(f"Addition: {tensor_add}")

# subtract
tensor_sub = tensor_a - tensor_b
print(f"Subtraction: {tensor_sub}")

tensor_sub = torch.sub(tensor_a, tensor_b)
print(f"Subtraction: {tensor_sub}")

# multiply
tensor_mul = tensor_a * tensor_b
print(f"Multiplication: {tensor_mul}")

tensor_mul = torch.mul(tensor_a, tensor_b)
print(f"Multiplication: {tensor_mul}")

# divide
tensor_div = tensor_a / tensor_b
print(f"Division: {tensor_div}")

tensor_div = torch.div(tensor_a, tensor_b)
print(f"Division: {tensor_div}")

# remainder
tensor_rem = tensor_b % tensor_a
print(f"Remainder: {tensor_rem}")

tensor_rem = torch.remainder(tensor_b, tensor_a)
print(f"Remainder: {tensor_rem}")

# exponents
tensor_exp = tensor_a ** tensor_b
print(f"Exponents: {tensor_exp}")

tensor_exp = torch.pow(tensor_a, tensor_b)
print(f"Exponents: {tensor_exp}")

# reassign tensor_a
tensor_a.add(tensor_b) # does not update tensor_a
print(f"Addition: {tensor_a}")
tensor_a.add_(tensor_b) # underscore to update the tensor in place
print(f"Addition: {tensor_a}")
