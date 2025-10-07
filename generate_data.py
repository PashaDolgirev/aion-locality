import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime

torch.random.manual_seed(1234) # for reproducibility
dtype = torch.float32
device = "cpu"

M_cutoff = 50 # maximum harmonic
N_grid   = 512
m = torch.arange(1, M_cutoff+1, dtype=dtype, device=device)             # (M,)
x = torch.linspace(0, 1, N_grid, dtype=dtype, device=device)            # (N,)
#design matrix needed to sample densities
DesignMatrix = torch.cos(torch.pi * torch.outer(m, x))                  # (M, N)
std_harm = 2.0 / (1.0 + m)**2

def sample_density(*, rho_b=0.0):
    """
    Sample rho_j = rho_avg + sum_{m=1}^M a_m cos(m pi x_j), with x_j in [0,1]
    note that the derivatives at the boundaries are zero
    Sampling is done such that the generated density is non-negative everywhere

    Returns:
      rho : (N,) density profile
    """    
    a = torch.normal(torch.zeros_like(std_harm), std_harm)
    rho = a @ DesignMatrix

    rho = rho - rho.min() + rho_b  # make non-negative
    return rho

def sample_density_batch(B: int, rho_b=0.0):
    """
    Sample a batch of B density profiles.
    Returns rho: (B, N_grid)
    """
    a = torch.normal(torch.zeros(B, std_harm.numel()), std_harm.expand(B, -1))
    rho = a @ DesignMatrix  # (B, N_grid)

    rho_min = rho.min(dim=1, keepdim=True).values
    rho = rho - rho_min + rho_b
    return rho

def E_tot_v1(rho: torch.Tensor) -> torch.Tensor:
    """
    Compute E_tot -- custom functional
    rho: (N,) or (B, N)
    Returns: scalar (if input 1D) or (B,) (if input 2D)
    """
    return (rho + rho**2).sum(dim=-1) / N_grid

N_train = 1500
N_test = 250
N_val = 250

rho_train = sample_density_batch(N_train)  # (N_train, N_grid)
rho_test  = sample_density_batch(N_test)   # (N_test, N_grid)
rho_val   = sample_density_batch(N_val)    # (N_val, N_grid)

targets_train = E_tot_v1(rho_train)            # (N_train,)
targets_test  = E_tot_v1(rho_test)             # (N_test,) 
targets_val   = E_tot_v1(rho_val)              # (N_val,)

save_dir = Path("DATA")
save_dir.mkdir(parents=True, exist_ok=True)

dataset_blob = {
    "x": x,  # grid (N_grid,)
    "rho_train": rho_train, "targets_train": targets_train,
    "rho_val":   rho_val,   "targets_val":   targets_val,
    "rho_test":  rho_test,  "targets_test":  targets_test,
}

torch.save(dataset_blob, save_dir / "dataset.pt")
print(f"Saved to {save_dir/'loc_functional_data.pt'} (~{(save_dir/'dataset.pt').stat().st_size/1e6:.2f} MB)")

