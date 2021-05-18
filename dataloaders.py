import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader
import data

# Data parameters
seed         = 4
f_rockstar   = 'Rockstar_z=0.0.txt'
batch_size    = 1

#Create datasets
train_Dataset, valid_Dataset, test_Dataset = data.create_datasets(seed, f_rockstar)

#Create Dataloaders
train_loader = DataLoader(dataset=train_Dataset, 
                          batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_Dataset, 
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_Dataset,  
                          batch_size=batch_size, shuffle=True)

# Load train data into containers and save as npy files
n_halos = int(3674*0.8)
train_x = np.zeros((n_halos, 9), dtype=np.float32)
train_y = np.zeros((n_halos, 1), dtype=np.float32)
row = -1

# Load each variable into container
for input, output in train_loader:
    row +=1
    # add input to first row
    print("Input: " + str(input))
    print("Output: " + str(output))
    train_x[row,:9] = input
    train_y[row] = output
    
# Load validation data into containers and save as npy files
valid_x = np.zeros((n_halos, 9), dtype=np.float32)
valid_y = np.zeros((n_halos, 1), dtype=np.float32)
row = -1

# Load each variable into container
for input, output in valid_loader:
    row +=1
    # add input to first row
    valid_x[row,:9] = input
    valid_y[row] = output

# Load test data into containers and save as npy file
test_x = np.zeros((n_halos, 9), dtype=np.float32)
test_y = np.zeros((n_halos, 1), dtype=np.float32)
row = -1

# Load each variable into container
for input, output in test_loader:
    row +=1
    # add input to first row
    test_x[row,:9] = input
    test_y[row] = output
    
    
# Save each dataset
np.save("norm_train_x.npy", train_x)
np.save("norm_train_y.npy", train_y)

np.save("norm_valid_x.npy", valid_x)
np.save("norm_valid_y.npy", valid_y)

np.save("norm_test_x.npy", test_x)
np.save("norm_test_y.npy", test_y)
