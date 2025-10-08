import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

torch.random.manual_seed(1234) # for reproducibility

# Global settings
dtype = torch.float32
device = "cpu"
N_batch = 32
N_epochs = 50000

lr_list = [0.001, 0.0001, 0.00001, 0.000001]
dropout_list = [0.0]
num_nodes_per_layer_list = [20, 40, 80, 160, 320]
num_hidden_layers_list = [2, 4, 6]

num_models = len(lr_list) * len(dropout_list) * len(num_nodes_per_layer_list) * len(num_hidden_layers_list)
print(f"Total number of models to train: {num_models}")


# Load dataset
blob = torch.load("DATA/loc_functional_data.pt", map_location="cpu")

x = blob["x"]
rho_train, y_train = blob["rho_train"], blob["targets_train"]
rho_val,   y_val   = blob["rho_val"],   blob["targets_val"]
rho_test,  y_test  = blob["rho_test"],  blob["targets_test"]




# Wrap as TensorDataset
ds_train = TensorDataset(rho_train, y_train)
ds_val   = TensorDataset(rho_val,   y_val)
ds_test  = TensorDataset(rho_test,  y_test)

# Build DataLoaders
train_loader = DataLoader(ds_train, batch_size=64, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(ds_val,   batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(ds_test,  batch_size=128, shuffle=False, num_workers=0, pin_memory=False)


