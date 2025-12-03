import torch
import torch.nn as nn
import torch.nn.functional as F

from .dct_utils import (
    kernel_eigenvals_dct,
    rho_to_cosine_coeffs,
    cosine_coeffs_to_rho,
)

class LearnableKernelConv1d(nn.Module):
    """
    Learnable 1D convolution kernel K_r with range R
    Produces phi = [K * rho] (linear convolution with padding)
    """
    def __init__(self, R=5, even_kernel=True, pad_mode="zero"):
        super().__init__()
        self.R = R
        self.pad_mode = pad_mode
        self.even_kernel = even_kernel
        
        if even_kernel:
            # learn half + center: w[0] (center), w[1..R] (positive r)
            self.kernel_half = nn.Parameter(torch.randn(R+1) * 0.01)
        else:
            # fully unconstrained kernel of size 2R+1
            self.kernel = nn.Parameter(torch.randn(2*R+1) * 0.01)

    def build_kernel(self):
        """
        Returns kernel of shape (1,1,2R+1) as required by conv1d
        """
        if self.even_kernel:
            center = self.kernel_half[0:1]          # (1,)
            pos = self.kernel_half[1:]             # (R,)
            neg = pos.flip(0)               # symmetric
            full = torch.cat([neg, center, pos], dim=0)  # (2R+1,)
        else:
            full = self.kernel
        return full.view(1,1,-1)  # (out=1, in=1, kernel_size)

    def forward(self, rho):
        """
        rho: (B, N_grid)
        Returns: phi: (B, N_grid)
        """
        B, N = rho.shape
        kernel = self.build_kernel()
        kernel = kernel.to(dtype=rho.dtype, device=rho.device)
        R = self.R
        
        if self.pad_mode == "zero":
            x = F.pad(rho.unsqueeze(1), (R, R), mode='constant', value=0.0)
        elif self.pad_mode == "reflect":
            x = F.pad(rho.unsqueeze(1), (R, R), mode='reflect')
        else:
            raise ValueError("pad_mode must be zero or reflect")
        
        phi = F.conv1d(x, kernel, padding=0).squeeze(1)
        return phi
    
class GaussianMixtureKernelConv1d(nn.Module):
    """
    Learnable 1D kernel K_r represented as a sum of Gaussians:
        K(r) = sum_{n=1}^M A_n * exp(-r^2 / sigma_n^2)

    Produces phi = [K * rho] via conv1d.
    """
    def __init__(self, R: int, n_components: int, pad_mode: str = "zero"):
        super().__init__()
        self.R = R
        self.n_components = n_components
        self.pad_mode = pad_mode

        # r-grid as a buffer: [-R, ..., R] - still is a cutoff in real space
        r_vals = torch.arange(-R, R + 1)
        self.register_buffer("r_vals", r_vals)  # (2R+1,)

        # ---- amplitudes A_n (sorted at init) ----
        # sample, then sort descending so A_1 >= A_2 >= ... >= A_M
        amps = 0.01 * torch.randn(n_components)
        amps, _ = torch.sort(amps, descending=True)
        self.amplitudes = nn.Parameter(amps)

        # Log-sigmas so sigma_n = softplus(log_sigma_n) > 0
        init_sigmas = torch.arange(1, n_components+1, dtype=torch.float32)**2
        log_sigmas = torch.log(torch.expm1(init_sigmas))  # inverse softplus, so softplus(raw)=init_sigma
        self.log_sigmas = nn.Parameter(log_sigmas)

    def build_kernel(self):
        """
        Returns kernel of shape (1, 1, 2R+1) as required by conv1d.
        """
        # (2R+1,) -> (1, L)
        r = self.r_vals.view(1, -1)              # (1, 2R+1)
        r2 = r * r                               # r^2

        sigmas = F.softplus(self.log_sigmas) + 1e-8   # (M,)
        sigma2 = sigmas.view(-1, 1) ** 2              # (M,1)

        # contributions from each Gaussian: (M, L)
        # exp(-r^2 / sigma_n^2)
        gauss = torch.exp(-r2 / sigma2)          # (M, 2R+1)

        # weighted sum over components
        kernel_1d = (self.amplitudes.view(-1, 1) * gauss).sum(dim=0)  # (2R+1,)

        # conv1d expects (out_channels, in_channels, kernel_size)
        return kernel_1d.view(1, 1, -1)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: (B, N_grid)
        Returns: phi = (K * rho): (B, N_grid)
        """
        B, N = rho.shape
        kernel = self.build_kernel().to(dtype=rho.dtype, device=rho.device) 
        R = self.R

        if self.pad_mode == "zero":
            x = F.pad(rho.unsqueeze(1), (R, R), mode="constant", value=0.0)
        elif self.pad_mode == "reflect":
            x = F.pad(rho.unsqueeze(1), (R, R), mode="reflect")
        else:
            raise ValueError("pad_mode must be 'zero' or 'reflect'")

        phi = F.conv1d(x, kernel, padding=0).squeeze(1)   # (B, N)
        return phi
    
class LearnableRSNonLocalKernelDCT(nn.Module):
    """
    Learnable 1D nonlocal kernel K_r with full range N_grid
    Produces phi = [K * rho] via DCT-I
    von Neumann BCs (even reflection)
    RS = real space
    """
    def __init__(self, N, zero_r_flag=True): # N = grid size
        super().__init__()
        self.zero_r_flag = zero_r_flag
        self.N = N
        if zero_r_flag:
            self.rs_kernel = nn.Parameter(torch.randn(N) * 0.01)  # full kernel K_r, r=0..N_grid-1
            self.rs_kernel.data[0] = 0.0    # enforce K_0 = 0
        else:
            self.rs_kernel = nn.Parameter(torch.randn(N) * 0.01) # full kernel K_r, r=0..N_grid-1

    def forward(self, rho):
        """
        rho: (B, N_grid)
        Returns: phi: (B, N_grid)
        """
        kernel = self.rs_kernel.to(dtype=rho.dtype, device=rho.device) # (N_grid,)
        if self.zero_r_flag:
            kernel = kernel.clone()
            kernel[..., 0] = 0.0

        lam_K = kernel_eigenvals_dct(kernel).to(device=rho.device, dtype=rho.dtype) # (N_grid,)
        a = rho_to_cosine_coeffs(rho)
        phi = cosine_coeffs_to_rho(lam_K.unsqueeze(0) * a)

        return phi

class ExpMixtureRSNonLocalKernelDCT(nn.Module):
    """
    Learnable 1D kernel K_r represented as a sum of exponentials:
        K(r) = sum_{n=1}^M A_n * exp(-r / sigma_n)
    """

    def __init__(self, N, zero_r_flag=False, n_components=3): # N = grid size
        super().__init__()
        self.zero_r_flag = zero_r_flag
        self.n_components = n_components
        self.N = N

        r_vals = torch.arange(0, N)
        self.register_buffer("r_vals", r_vals)  # (N,)

        # ---- amplitudes A_n (sorted at init) ----
        # sample, then sort descending so A_1 >= A_2 >= ... >= A_M
        amps = 0.01 * torch.randn(n_components)
        amps, _ = torch.sort(amps, descending=True)
        self.amplitudes = nn.Parameter(amps)

        # Log-sigmas so sigma_n = softplus(log_sigma_n) > 0
        init_sigmas = 10.0 + 20.0 * torch.arange(n_components)
        log_sigmas = torch.log(torch.expm1(init_sigmas))  # inverse softplus, so softplus(raw)=init_sigma
        self.log_sigmas = nn.Parameter(log_sigmas)

    def build_kernel(self):
        r = self.r_vals.view(1, -1)              # (1, N)
        sigmas = F.softplus(self.log_sigmas) + 1e-8   # (M,)
        exp_mixt = torch.exp(-r / sigmas.view(-1, 1))          # (M, N)
        return (self.amplitudes.view(-1, 1) * exp_mixt).sum(dim=0)  # (N,)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: (B, N_grid)
        Returns: phi = (K * rho): (B, N_grid)
        """
        kernel = self.build_kernel().to(dtype=rho.dtype, device=rho.device)
        if self.zero_r_flag:
            kernel = kernel.clone()
            kernel[..., 0] = 0.0

        lam_K = kernel_eigenvals_dct(kernel).to(device=rho.device, dtype=rho.dtype) # (N_grid,)
        a = rho_to_cosine_coeffs(rho)
        return cosine_coeffs_to_rho(lam_K.unsqueeze(0) * a)
    
class LearnableMSNonLocalKernelDCT(nn.Module):
    """
    Learnable nonlocal kernel parameterized directly in momentum space

    We learn λ_m for m = 1..range_ms.
    λ_0 is set to 0 (no uniform component), and λ_m = 0 for m > range_ms.
    """
    def __init__(self, range_ms: int = 50):
        super().__init__()
        self.range_ms = range_ms
        # learnable λ_m for m=1..range_ms
        self.ms_kernel = nn.Parameter(torch.randn(range_ms) * 0.01)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: (B, N_grid)
        Returns: phi = (K * rho): (B, N_grid)
        """
        B, N_grid = rho.shape
        device, dtype = rho.device, rho.dtype

        lam_head0 = torch.zeros(1, device=device, dtype=dtype)  
        lam_active = self.ms_kernel.to(device=device, dtype=dtype)  
        lam_tail = torch.zeros(N_grid - 1 - self.range_ms, device=device, dtype=dtype)
        lam_K = torch.cat([lam_head0, lam_active, lam_tail], dim=0)  # (N_grid,)

        a = rho_to_cosine_coeffs(rho)                      # (B, N_grid)
        return cosine_coeffs_to_rho(lam_K.unsqueeze(0) * a) # (B, N_grid)