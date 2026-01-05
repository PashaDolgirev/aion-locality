# feat_utils.py
import torch
from typing import Callable

EnergyFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor
]
LocEnergyFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, bool],
    torch.Tensor
]


from .dct_utils import (
    rho_to_cosine_coeffs,
    cosine_coeffs_to_rho,
)


from .energies_utils import (
    Lam_K_Coulomb,
)

def sample_density(std_harm: torch.Tensor,
                   DM_x: torch.Tensor,
                   DerDM_x: torch.Tensor,
                   DM_y: torch.Tensor,
                   DerDM_y: torch.Tensor):
    """
    Sample rho_{ij} = sum_{m=0}^{Nx - 1}sum_{n=0}^{Ny - 1} a_{mn} cos(m pi x_i) cos(n pi y_j), with x_i, y_j in [0,1].
    This sampling of amplitudes a_m implies that the derivatives at 
    the boundaries are zero.

    Args:
        std_harm:     (Mx, My) tensor of standard deviations for a_mn
        DM_x:         (Mx, Nx) tensor with cosines evaluated on the x grid
        DerDM_x:      (Mx, Nx) tensor with derivatives d/dx of the cosines on the x grid
        DM_y:         (My, Ny) tensor with cosines evaluated on the y grid
        DerDM_y:      (My, Ny) tensor with derivatives d/dy of the cosines on the y grid

    Returns:
        rho      : (Nx, Ny) density profile
        d_rho_x  : (Nx, Ny) derivative of rho wrt x
        d_rho_y  : (Nx, Ny) derivative of rho wrt y
        a        : (Mx, My) sampled amplitudes
    """
    # a_mn ~ N(0, std_harm_mn^2)
    a = torch.normal(mean=torch.zeros_like(std_harm),
                        std=std_harm)  # (Mx, My) 

    rho = DM_x.T @ a @ DM_y        # (Nx, Ny)
    d_rho_x = DerDM_x.T @ a @ DM_y  # (Nx, Ny)
    d_rho_y = DM_x.T @ a @ DerDM_y  # (Nx, Ny)
    
    return rho, d_rho_x, d_rho_y, a


def sample_density_batch(B: int,
                        std_harm: torch.Tensor,
                        DM_x: torch.Tensor,
                        DerDM_x: torch.Tensor,
                        DM_y: torch.Tensor,
                        DerDM_y: torch.Tensor):
    """
    Sample a batch of B density profiles.

    Args:
        B           : batch size
        std_harm:     (Mx, My) tensor of standard deviations for a_mn
        DM_x:         (Mx, Nx) tensor with cosines evaluated on the x grid
        DerDM_x:      (Mx, Nx) tensor with derivatives d/dx of the cosines on the x grid
        DM_y:         (My, Ny) tensor with cosines evaluated on the y grid
        DerDM_y:      (My, Ny) tensor with derivatives d/dy of the cosines on the y grid

     Returns:
        rho_batch      : (B, Nx, Ny) density profile
        d_rho_x_batch  : (B, Nx, Ny) derivative of rho wrt x
        d_rho_y_batch  : (B, Nx, Ny) derivative of rho wrt y
        a_batch        : (B, Mx, My) sampled amplitudes
    """
    # (B, Mx, My), each a_mn ~ N(0, std_harm_mn^2)
    a_batch = torch.normal(
        mean=torch.zeros(B, *std_harm.shape, device=std_harm.device, dtype=std_harm.dtype),
        std=std_harm.expand(B, -1, -1)
    )  # (B, Mx, My)
    rho_batch = DM_x.T @ a_batch @ DM_y        # (B, Nx, Ny)
    d_rho_x_batch = DerDM_x.T @ a_batch @ DM_y  # (B, Nx, Ny)
    d_rho_y_batch = DM_x.T @ a_batch @ DerDM_y  # (B, Nx, Ny)
    
    return rho_batch, d_rho_x_batch, d_rho_y_batch, a_batch


def compute_normalization_stats(features):
    """
    Compute mean and std for features with shape (N_data, N_x, N_y, N_feat)
    Averages over both data and spatial dimensions
    
    Args:
        features: torch.Tensor of shape (N_data, N_x, N_y, N_feat)
    
    Returns:
        mean: torch.Tensor of shape (1, 1, 1, N_feat)
        std: torch.Tensor of shape (1, 1, 1, N_feat)
    """

    mean_feat = features.mean(dim=(0, 1, 2), keepdim=True)  # Shape: (1, 1, 1, N_feat)
    std_feat = features.std(dim=(0, 1, 2), keepdim=True) # Shape: (1, 1, 1, N_feat)
    
    return mean_feat, std_feat


def normalize_features(features, mean_feat, std_feat):
    """
    Normalize features using provided or computed statistics
    
    Args:
        features: torch.Tensor of shape (B, N_x, N_y, N_feat)
        mean: torch.Tensor of shape (1, 1, 1, N_feat)
        std: torch.Tensor of shape (1, 1, 1, N_feat)

    Returns:
        normalized_features: torch.Tensor of same shape as input
        mean: mean used for normalization
        std: std used for normalization
    """
    normalized_features = (features - mean_feat) / std_feat

    return normalized_features


def generate_loc_features_rs(rho: torch.Tensor, N_pow=2) -> torch.Tensor:
    """
    Generate local features from density rho
    rs, real space   
    Args:
        rho: torch.Tensor of shape (B, N_x, N_y)
        N_feat: int, number of features to generate

    Returns:
        features: torch.Tensor of shape (B, N_x, N_y, N_feat)
        each feature is of the form rho^k, k=1,...,N_pow
    """
    features = [rho.unsqueeze(-1) ** k for k in range(1, N_pow + 1)]
    return torch.cat(features, dim=-1)


def generate_loc_features_ms(d_rho_x: torch.Tensor, d_rho_y: torch.Tensor, N_pow=2) -> torch.Tensor:
    """
    Generate local features from density derivative d_rho_x, d_rho_y
    ms, momentum space
    Args:
        d_rho_x: torch.Tensor of shape (B, N_x, N_y)
        d_rho_y: torch.Tensor of shape (B, N_x, N_y)
        N_feat: int, number of features to generate

    Returns:
        features: torch.Tensor of shape (B, N_x, N_y, N_pow * N_pow)
        each feature is of the form d_rho_x^k_x d_rho_y^k_y, k_x,k_y=1,...,N_pow
    """
    features = []
    for k_x in range(1, N_pow + 1):
        for k_y in range(1, N_pow + 1):
            features.append((d_rho_x.unsqueeze(-1) ** k_x) * (d_rho_y.unsqueeze(-1) ** k_y))
    return torch.cat(features, dim=-1)

def extend_features_neighbors_2d(features: torch.Tensor, R: float = 1.0) -> torch.Tensor:
    """
    Extend features by including neighboring grid points within radius R
    Args:
        features: torch.Tensor of shape (B, N_x, N_y, N_feat)
        R: float, radius of neighborhood in grid points
    """
    B, N_x, N_y, N_feat = features.shape
    pad_size = int(R)
    padded_features = torch.nn.functional.pad(features.permute(0, 3, 1, 2), (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    extended_features_list = []

    for dx in range(-pad_size, pad_size + 1):
        for dy in range(-pad_size, pad_size + 1):
            if (dx == 0 and dy == 0) or (dx**2 + dy**2 > R**2):
                continue
            shifted_features = padded_features[:, :, pad_size + dx:pad_size + dx + N_x, pad_size + dy:pad_size + dy + N_y]
            extended_features_list.append(shifted_features.permute(0, 2, 3, 1))

    extended_features = torch.cat(extended_features_list, dim=-1)
    return torch.cat([features, extended_features], dim=-1)


def generate_data_2d(
        N: int,                     # number of samples
        N_batch: int,               # batch size
        E_tot: EnergyFunction,      # total energy function
        std_harm: torch.Tensor,
        DM_x: torch.Tensor, 
        DerDM_x: torch.Tensor, 
        DM_y: torch.Tensor, 
        DerDM_y: torch.Tensor
        ):
    """
    Generate dataset of density profiles and corresponding total energies
    Done in mini-batches of size N_batch to save memory
    """
    rho_list = []
    d_rho_x_list = []
    d_rho_y_list = []
    a_list = []
    E_list = []

    num_iter = (N + N_batch - 1) // N_batch
    with torch.no_grad():
        for i in range(num_iter):
            current_batch_size = min(N_batch, N - i * N_batch)
            rho_batch, d_rho_x_batch, d_rho_y_batch, a_batch = sample_density_batch(
                current_batch_size, std_harm=std_harm, DM_x=DM_x, DerDM_x=DerDM_x, DM_y=DM_y, DerDM_y=DerDM_y)
            E_batch = E_tot(rho_batch, d_rho_x_batch, d_rho_y_batch)  # (B,)

            rho_list.append(rho_batch)
            d_rho_x_list.append(d_rho_x_batch)
            d_rho_y_list.append(d_rho_y_batch)
            a_list.append(a_batch)
            E_list.append(E_batch)

        rho_all = torch.cat(rho_list, dim=0)
        d_rho_x_all = torch.cat(d_rho_x_list, dim=0)
        d_rho_y_all = torch.cat(d_rho_y_list, dim=0)
        a_all = torch.cat(a_list, dim=0)
        E_all = torch.cat(E_list, dim=0)

    return rho_all, d_rho_x_all, d_rho_y_all, a_all, E_all


def generate_SC_data_2d(
        N: int,                         # number of samples
        N_batch: int,                   # batch size
        E_HF: LocEnergyFunction,        # unscreened total energy function
        E_SC: LocEnergyFunction,        # screened total energy function
        std_harm: torch.Tensor,
        DM_x: torch.Tensor, 
        DerDM_x: torch.Tensor, 
        DM_y: torch.Tensor, 
        DerDM_y: torch.Tensor
        ):
    """
    Generate dataset of density profiles and corresponding total energies
    Done in mini-batches of size N_batch to save memory
    """
    rho_list = []
    d_rho_x_list = []
    d_rho_y_list = []
    a_list = []
    E_loc_HF_list = []
    E_SC_list = []

    num_iter = (N + N_batch - 1) // N_batch
    with torch.no_grad():
        for i in range(num_iter):
            current_batch_size = min(N_batch, N - i * N_batch)
            rho_batch, d_rho_x_batch, d_rho_y_batch, a_batch = sample_density_batch(
                current_batch_size, std_harm=std_harm, DM_x=DM_x, DerDM_x=DerDM_x, DM_y=DM_y, DerDM_y=DerDM_y)
            
            E_loc_HF_batch = E_HF(rho_batch, d_rho_x_batch, d_rho_y_batch, eng_dens_flag=True)  # (B, N_x, N_y, 1)
            E_SC_batch = E_SC(rho_batch, d_rho_x_batch, d_rho_y_batch, eng_dens_flag=False)  # (B,)

            rho_list.append(rho_batch)
            d_rho_x_list.append(d_rho_x_batch)
            d_rho_y_list.append(d_rho_y_batch)
            a_list.append(a_batch)
            E_loc_HF_list.append(E_loc_HF_batch)
            E_SC_list.append(E_SC_batch)

        rho_all = torch.cat(rho_list, dim=0)
        d_rho_x_all = torch.cat(d_rho_x_list, dim=0)
        d_rho_y_all = torch.cat(d_rho_y_list, dim=0)
        a_all = torch.cat(a_list, dim=0)
        E_HF_all = torch.cat(E_loc_HF_list, dim=0)
        E_SC_all = torch.cat(E_SC_list, dim=0)

    return rho_all, d_rho_x_all, d_rho_y_all, a_all, E_HF_all, E_SC_all


def E_kin_custom(
        rho: torch.Tensor, 
        d_rho_x: torch.Tensor, 
        d_rho_y: torch.Tensor, 
        alpha: float, 
        beta: float, 
        qs: float,
        eng_dens_flag: bool = False
        ) -> torch.Tensor:
    """
    Kinetic energy functional:
        E_kin = 1 / (2 N_x  N_y) sum_{ij} kappa_{ij} (d_rho_x_{ij}^2 + d_rho_y_{ij}^2),
        where kappa_{ij} = 1 + alpha * rho_{r + ex} * rho_{r + ey} * rho_{r - ex} * rho_{r - ey} + beta * phi_{ij},
        phi_ij is the mediator field, with screening length set by qs

    Args:
        rho:      (N_x, N_y) or (B, N_x, N_y)
        d_rho_x:  (N_x, N_y) or (B, N_x, N_y) - derivative of rho w.r.t. x
        d_rho_y:  (N_x, N_y) or (B, N_x, N_y) - derivative of rho w.r.t. y
    """

    if rho.dim() == 2:
        rho = rho.unsqueeze(0)
        d_rho_x = d_rho_x.unsqueeze(0)
        d_rho_y = d_rho_y.unsqueeze(0)
    
    B, N_x, N_y = rho.shape
    device, dtype = rho.device, rho.dtype

    
    R_feat = 1.0 # radius for neighbor feature extension
    rho_neighbours = extend_features_neighbors_2d(rho.unsqueeze(-1), R=R_feat) # (B, N_x, N_y, N_nb), N_nb = number of neighbors within R_feat

    # rho_{r + ex} * rho_{r + ey} * rho_{r - ex} * rho_{r - ey}
    alpha_term = rho_neighbours[:,:,:,1] * \
                    rho_neighbours[:,:,:,2] * \
                    rho_neighbours[:,:,:,3] * \
                    rho_neighbours[:,:,:,4] # (B, N_x, N_y) 
    
    # compute mediator field phi using DCT routines
    m_vals = torch.arange(0, N_x, device=device, dtype=dtype)
    n_vals = torch.arange(0, N_y, device=device, dtype=dtype)
    q_x = torch.pi * m_vals.view(-1, 1) / (N_x - 1)
    q_y = torch.pi * n_vals.view(1, -1) / (N_y - 1)
    q_vals = torch.sqrt(q_x**2 + q_y**2)  # (N_x, N_y)

    lam_K = Lam_K_Coulomb(q_vals, qs=qs).to(device=device, dtype=dtype)  # (N_x, N_y)

    a = rho_to_cosine_coeffs(rho)                     # (B, N_x, N_y)
    phi = cosine_coeffs_to_rho(lam_K.unsqueeze(0) * a)  # (B, N_x, N_y)

    kappa = 1.0 + alpha * alpha_term + beta * phi # (B, N_x, N_y)

    E_kin_loc = 0.5 * kappa * (d_rho_x ** 2 + d_rho_y ** 2)  # (B, N_x, N_y)
    if eng_dens_flag:
        return E_kin_loc  # (B, N_x, N_y)

    return E_kin_loc.sum(dim=(1,2)) / (N_x * N_y) # (B,)