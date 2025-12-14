import torch
import torch.nn as nn
import torch.nn.functional as F

from .dct_utils import (
    kernel_eigenvals_dct,
    kernel_from_eigenvals_dct,
    rho_to_cosine_coeffs,
    cosine_coeffs_to_rho,
)

from .energies_utils import (
    Lam_K_Coulomb,
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


class LearnableRSNonLocalKernelDCT(nn.Module):
    """
    Learnable 2D nonlocal kernel K_{r_x, r_y} with range R
    Produces phi = [K * rho] via DCT routines
    von Neumann BCs (even reflection)
    RS = real space
    """
    def __init__(self, N_x, N_y, zero_r_flag=True, R=1024):
        super().__init__()
        self.zero_r_flag = zero_r_flag
        self.N_x = N_x
        self.N_y = N_y
        self.R = R
        if zero_r_flag:
            self.rs_kernel = nn.Parameter(torch.randn(N_x, N_y) * 0.01)  # full kernel K_r, r=0..N_grid-1
            self.rs_kernel[0, 0] = 0.0    # enforce K_{0,0} = 0
        else:
            self.rs_kernel = nn.Parameter(torch.randn(N_x, N_y) * 0.01) # full kernel K_r, r=0..N_grid-1
        
        # create a mask to enforce the range R
        rx = torch.arange(N_x).view(-1, 1)   # (N_x, 1)
        ry = torch.arange(N_y).view(1, -1)   # (1, N_y)
        rs_mask = (rx**2 + ry**2) <= R**2    # (N_x, N_y)
        self.register_buffer("rs_mask", rs_mask.float())

        with torch.no_grad():
            self.rs_kernel *= self.rs_mask  

    def forward(self, rho):
        """
        rho: (B, N_x, N_y)
        Returns: phi: (B, N_x, N_y)
        """
        kernel = (self.rs_kernel * self.rs_mask).to(device=rho.device, dtype=rho.dtype)
        if self.zero_r_flag:
            kernel = kernel.clone()
            kernel[0, 0] = 0.0

        lam_K = kernel_eigenvals_dct(kernel).to(device=rho.device, dtype=rho.dtype)  # (N_x, N_y)
        a = rho_to_cosine_coeffs(rho)
        return cosine_coeffs_to_rho(lam_K.unsqueeze(0) * a)
    

class LearnableMSNonLocalKernelDCT(nn.Module):
    """
    Learnable nonlocal kernel parameterized directly in momentum space

    We learn λ_{mn} for m,n = 1..range_ms.
    λ_0 is set to 0 (no uniform component), and λ_{m,n} = 0 for m^2 + n^2 > range_ms^2
    """
    def __init__(self, N_x, N_y, range_ms: float = 500.0):
        super().__init__()
        self.N_x = N_x
        self.N_y = N_y
        self.range_ms = range_ms
        self.ms_kernel = nn.Parameter(torch.randn(N_x, N_y) * 0.01)

        mx = torch.arange(N_x).view(-1, 1)   # (N_x, 1)
        my = torch.arange(N_y).view(1, -1)   # (1, N_y)
        ms_mask = (mx**2 + my**2) <= range_ms**2    # (N_x, N_y)
        ms_mask[0, 0] = 0    # enforce λ_{0,0} = 0
        self.register_buffer("ms_mask", ms_mask.float())

        with torch.no_grad():
            self.ms_kernel *= self.ms_mask

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: (B, N_x, N_y)
        Returns: phi = (K * rho): (B, N_x, N_y)
        """
        lam_K = (self.ms_kernel * self.ms_mask).to(device=rho.device, dtype=rho.dtype)  # (N_x, N_y)    
        a = rho_to_cosine_coeffs(rho)                      # (B, N_x, N_y)
        return cosine_coeffs_to_rho(lam_K.unsqueeze(0) * a) # (B, N_x, N_y)
    

class ExpMixtureRSNonLocalKernelDCT(nn.Module):
    """
    Learnable 2D kernel K_r represented as a sum of exponentials:
        K(r) = sum_{n=1}^M A_n * exp(-r / sigma_n)
    """

    def __init__(self, N_x, N_y, zero_r_flag=False, n_components=3): 
        super().__init__()
        self.zero_r_flag = zero_r_flag
        self.n_components = n_components
        self.N_x = N_x
        self.N_y = N_y

        rx = torch.arange(N_x).view(-1, 1)   # (N_x, 1)
        ry = torch.arange(N_y).view(1, -1)   # (1, N_y)
        r_vals = torch.sqrt(rx**2 + ry**2)    # (N_x, N_y)
        self.register_buffer("r_vals", r_vals)  # (N_x, N_y)

        # ---- amplitudes A_n (sorted at init) ----
        # sample, then sort descending so A_1 >= A_2 >= ... >= A_M
        amps = 0.01 * torch.randn(n_components)
        amps, _ = torch.sort(amps, descending=True)
        self.amplitudes = nn.Parameter(amps)

        # Log-sigmas so sigma_n = softplus(log_sigma_n) > 0
        init_sigmas = 30.0 + 20.0 * torch.arange(n_components)
        log_sigmas = torch.log(torch.expm1(init_sigmas))  # inverse softplus, so softplus(raw)=init_sigma
        self.log_sigmas = nn.Parameter(log_sigmas)

    def build_kernel(self):
        r = self.r_vals.unsqueeze(0)            # (1, N_x, N_y)
        sigmas = F.softplus(self.log_sigmas) + 1e-8   # (M,)
        exp_mixt = torch.exp(-r / sigmas.view(-1, 1, 1))          # (M, N_x, N_y)
        return (self.amplitudes.view(-1, 1, 1) * exp_mixt).sum(dim=0)  # (N_x, N_y)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: (B, N_x, N_y)
        Returns: phi = (K * rho): (B, N_x, N_y)
        """
        kernel = self.build_kernel().to(dtype=rho.dtype, device=rho.device)
        if self.zero_r_flag:
            kernel = kernel.clone()
            kernel[0, 0] = 0.0

        lam_K = kernel_eigenvals_dct(kernel).to(device=rho.device, dtype=rho.dtype) # (N_x, N_y)
        a = rho_to_cosine_coeffs(rho)
        return cosine_coeffs_to_rho(lam_K.unsqueeze(0) * a)


class ScreenedCoulombNonLocalKernelDCT(nn.Module):
    """
    Nonlocal kernel K_r corresponding to screened Coulomb potential
    in 2D, via DCT routines
    """
    def __init__(self, N_x, N_y):
        super().__init__()
        self.N_x = N_x
        self.N_y = N_y

        m_vals = torch.arange(0, N_x)
        n_vals = torch.arange(0, N_y)
        q_x = torch.pi * m_vals.view(-1, 1) / (N_x - 1)
        q_y = torch.pi * n_vals.view(1, -1) / (N_y - 1)
        q_vals = torch.sqrt(q_x**2 + q_y**2)  # (N_x, N_y)

        self.register_buffer("q_vals", q_vals)
        self.amp = nn.Parameter(torch.randn() * 0.01)
        self.raw_qs = nn.Parameter(torch.tensor(1.0))

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: (B, N_x, N_y)
        Returns: phi = (K * rho): (B, N_x, N_y)
        """
        qs = F.softplus(self.raw_qs)
        lam_K = (self.amp * Lam_K_Coulomb(self.q_vals, qs=qs)).to(device=rho.device, dtype=rho.dtype)  # (N_x, N_y)
        a = rho_to_cosine_coeffs(rho)
        return cosine_coeffs_to_rho(lam_K.unsqueeze(0) * a)