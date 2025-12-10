import torch
import torch.nn as nn
import torch.nn.functional as F

from .dct_utils import (
    kernel_eigenvals_dct,
    kernel_from_eigenvals_dct,
    rho_to_cosine_coeffs,
    cosine_coeffs_to_rho,
)


class LearnableRSKernelConv2d(nn.Module):
    """
    Learnable 2D convolution kernel K_{r_x, r_y} with range R in each dimension
    Produces phi = [K * rho] (linear convolution with padding)
    RS, real space kernel
    """
    def __init__(self, R=5, even_kernel=True, pad_mode="reflect"):
        super().__init__()
        self.R = R
        self.pad_mode = pad_mode
        self.even_kernel = even_kernel
        
        if even_kernel:
            # K_{r,r'} for r and r' in [0,R] are learned, full kernel is symmetric
            self.kernel_upper_quadrant = nn.Parameter(torch.randn(R+1, R+1) * 0.01)
        else:
            # fully unconstrained kernel of size (2R+1, 2R+1)
            self.kernel = nn.Parameter(torch.randn(2*R+1, 2*R+1) * 0.01)

    def build_kernel(self):
        """
        Returns kernel of shape (1,1,2R+1,2R+1) as required by conv2d
        """
        if self.even_kernel:
            pos_x = self.kernel_upper_quadrant[1:, :]               # (R, R+1)
            center = self.kernel_upper_quadrant[0:1, :]             # (1, R+1)
            neg_x = pos_x.flip(0)                                   # symmetric (R, R+1)
    
            upper_half = torch.cat([neg_x, center, pos_x], dim=0)   # (2R+1, R+1)
            center = upper_half[:, 0:1]                             # (2R+1, 1)
            pos_y = upper_half[:, 1:]                               # (2R+1, R)
            neg_y = pos_y.flip(1)                                   # symmetric (2R+1, R)
            full = torch.cat([neg_y, center, pos_y], dim=1)         # (2R+1, 2R+1)
        else:
            full = self.kernel

        return full.view(1,1,2*self.R+1,2*self.R+1)  # (out=1, in=1, kernel_size_x, kernel_size_y)
    
    def forward(self, rho):
        """
        rho: (B, N_x, N_y)
        Returns: phi: (B, N_x, N_y)
        """
        B, N_x, N_y = rho.shape
        kernel = self.build_kernel().to(dtype=rho.dtype, device=rho.device)
        R = self.R
        
        if self.pad_mode == "zero":
            x = F.pad(rho.unsqueeze(1), (R, R, R, R), mode='constant', value=0.0) 
        elif self.pad_mode == "reflect":
            x = F.pad(rho.unsqueeze(1), (R, R, R, R), mode='reflect')
        else:
            raise ValueError("pad_mode must be zero or reflect")
        
        phi = F.conv2d(x, kernel, padding=0).squeeze(1)
        return phi
