#%%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% check PyTorch version
torch.__version__


#%% vector plotting
def plotVec(vectors):
    ax = plt.axes()

    # for loop to draw the vectors
    for vec in vectors:
        ax.arrow(0, 0, *vec["vector"], head_width = 0.05, color = vec["color"], head_length = 0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])

        plt.ylim(-2, 2)
        plt.xlim(-2, 2)

#%% Tensor shape and type
ints_to_tensor = torch.tensor([0, 1, 2, 3, 4])
print(f"dtype is: {ints_to_tensor.dtype}")
print(f"type is: {ints_to_tensor.type()}")

# floats to tensor
list_floats = [0.0, 1.0, 2.0, 3.0, 4.0]
floats_int_tensor = torch.tensor(list_floats, dtype = torch.int64)
print(f"dtype: {floats_int_tensor.dtype}")
print(f"type: {floats_int_tensor.type()}")

# convert tensor type
new_float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])
new_float_tensor.type()
print(f"type: {new_float_tensor.type()}")

# another method to convert the integer list to float tensor
old_int_tensor = torch.tensor([0, 1, 2, 3, 4])
new_float_tensor = old_int_tensor.type(torch.FloatTensor) # convert int tensor to float tensor
print(f"type: {new_float_tensor.type()}")

# see the size and dimension of the tensor
print(f"size: {new_float_tensor.size()}")
print(f"dimension: {new_float_tensor.ndimension()}")

# Tensor Addition
u = torch.tensor([1, 0])
v = torch.tensor([0, 1])

w = u + v
print(w)

plotVec([
    {"vector": u.numpy(), "name": 'u', "color": 'r'},
    {"vector": v.numpy(), "name": 'v', "color": 'b'},
    {"vector": w.numpy(), "name": 'w', "color": 'g'}
])