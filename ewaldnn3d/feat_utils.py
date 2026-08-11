# feat_utils.py
import torch
from typing import Callable

EnergyFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor
]
LocEnergyFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool],
    torch.Tensor
]


from .fft_utils import (
    conv_fft,
    q_grid_fft,
)


from .energies_utils import (
    Lam_K_Coulomb,
)

def sample_density(std_harm: torch.Tensor,
                   DM_x: torch.Tensor,
                   DerDM_x: torch.Tensor,
                   DM_y: torch.Tensor,
                   DerDM_y: torch.Tensor,
                   DM_z: torch.Tensor,
                   DerDM_z: torch.Tensor):
    """
    Sample rho_{ijk} = sum_{mnl} a_{mnl} cos(m pi x_i) cos(n pi y_j) cos(l pi z_k),
    with x_i, y_j, z_k in [0,1].
    This sampling of amplitudes a_{mnl} implies that the derivatives at
    the boundaries are zero.

    Args:
        std_harm:     (Mx, My, Mz) tensor of standard deviations for a_{mnl}
        DM_x:         (Mx, Nx) tensor with cosines evaluated on the x grid
        DerDM_x:      (Mx, Nx) tensor with derivatives d/dx of the cosines on the x grid
        DM_y:         (My, Ny) tensor with cosines evaluated on the y grid
        DerDM_y:      (My, Ny) tensor with derivatives d/dy of the cosines on the y grid
        DM_z:         (Mz, Nz) tensor with cosines evaluated on the z grid
        DerDM_z:      (Mz, Nz) tensor with derivatives d/dz of the cosines on the z grid

    Returns:
        rho      : (Nx, Ny, Nz) density profile
        d_rho_x  : (Nx, Ny, Nz) derivative of rho wrt x
        d_rho_y  : (Nx, Ny, Nz) derivative of rho wrt y
        d_rho_z  : (Nx, Ny, Nz) derivative of rho wrt z
        a        : (Mx, My, Mz) sampled amplitudes
    """
    # a_mnl ~ N(0, std_harm_mnl^2)
    a = torch.normal(mean=torch.zeros_like(std_harm),
                        std=std_harm)  # (Mx, My, Mz)

    rho     = torch.einsum('mnl,mi,nj,lk->ijk', a, DM_x, DM_y, DM_z)
    d_rho_x = torch.einsum('mnl,mi,nj,lk->ijk', a, DerDM_x, DM_y, DM_z)
    d_rho_y = torch.einsum('mnl,mi,nj,lk->ijk', a, DM_x, DerDM_y, DM_z)
    d_rho_z = torch.einsum('mnl,mi,nj,lk->ijk', a, DM_x, DM_y, DerDM_z)

    return rho, d_rho_x, d_rho_y, d_rho_z, a


def sample_density_batch(B: int,
                        std_harm: torch.Tensor,
                        DM_x: torch.Tensor,
                        DerDM_x: torch.Tensor,
                        DM_y: torch.Tensor,
                        DerDM_y: torch.Tensor,
                        DM_z: torch.Tensor,
                        DerDM_z: torch.Tensor):
    """
    Sample a batch of B density profiles.

    Args:
        B           : batch size
        std_harm, DM_*, DerDM_*: see sample_density

     Returns:
        rho_batch      : (B, Nx, Ny, Nz) density profile
        d_rho_x_batch  : (B, Nx, Ny, Nz) derivative of rho wrt x
        d_rho_y_batch  : (B, Nx, Ny, Nz) derivative of rho wrt y
        d_rho_z_batch  : (B, Nx, Ny, Nz) derivative of rho wrt z
        a_batch        : (B, Mx, My, Mz) sampled amplitudes
    """
    # (B, Mx, My, Mz), each a_mnl ~ N(0, std_harm_mnl^2)
    a_batch = torch.normal(
        mean=torch.zeros(B, *std_harm.shape, device=std_harm.device, dtype=std_harm.dtype),
        std=std_harm.expand(B, -1, -1, -1)
    )  # (B, Mx, My, Mz)
    rho_batch     = torch.einsum('bmnl,mi,nj,lk->bijk', a_batch, DM_x, DM_y, DM_z)
    d_rho_x_batch = torch.einsum('bmnl,mi,nj,lk->bijk', a_batch, DerDM_x, DM_y, DM_z)
    d_rho_y_batch = torch.einsum('bmnl,mi,nj,lk->bijk', a_batch, DM_x, DerDM_y, DM_z)
    d_rho_z_batch = torch.einsum('bmnl,mi,nj,lk->bijk', a_batch, DM_x, DM_y, DerDM_z)

    return rho_batch, d_rho_x_batch, d_rho_y_batch, d_rho_z_batch, a_batch


def sample_density_gaussians_batch(B: int,
                                   N_x: int, N_y: int, N_z: int,
                                   n_gauss: int = 4,
                                   amp_range: tuple = (0.5, 2.0),
                                   sigma_range: tuple = (0.05, 0.15),
                                   margin: float = 0.2,
                                   device=None, dtype=None):
    """
    Sample molecule-like densities: sums of isotropic Gaussians ("atoms") with
    random centers, widths, and amplitudes. Densities are localized away from
    the boundary (open BCs), coordinates x, y, z in [0,1].

    Args:
        B           : batch size
        n_gauss     : number of Gaussians per sample
        amp_range   : (min, max) amplitudes
        sigma_range : (min, max) widths, in units of the box
        margin      : centers are sampled in [margin, 1 - margin] per dimension

    Returns:
        rho_batch      : (B, Nx, Ny, Nz)
        d_rho_x_batch  : (B, Nx, Ny, Nz) derivative of rho wrt x
        d_rho_y_batch  : (B, Nx, Ny, Nz) derivative of rho wrt y
        d_rho_z_batch  : (B, Nx, Ny, Nz) derivative of rho wrt z
    """
    dtype = dtype or torch.get_default_dtype()
    x = torch.linspace(0, 1, N_x, device=device, dtype=dtype).view(1, 1, -1, 1, 1)
    y = torch.linspace(0, 1, N_y, device=device, dtype=dtype).view(1, 1, 1, -1, 1)
    z = torch.linspace(0, 1, N_z, device=device, dtype=dtype).view(1, 1, 1, 1, -1)

    def uni(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, device=device, dtype=dtype)

    centers = uni(margin, 1.0 - margin, B, n_gauss, 3).view(B, n_gauss, 3, 1, 1, 1)  # (B, G, 3, 1, 1, 1)
    amps    = uni(*amp_range, B, n_gauss).view(B, n_gauss, 1, 1, 1)                  # (B, G, 1, 1, 1)
    sigmas  = uni(*sigma_range, B, n_gauss).view(B, n_gauss, 1, 1, 1)                # (B, G, 1, 1, 1)

    dx = x - centers[:, :, 0]
    dy = y - centers[:, :, 1]
    dz = z - centers[:, :, 2]
    gauss = amps * torch.exp(-(dx**2 + dy**2 + dz**2) / (2.0 * sigmas**2))  # (B, G, Nx, Ny, Nz)

    rho_batch     = gauss.sum(dim=1)
    d_rho_x_batch = (-dx / sigmas**2 * gauss).sum(dim=1)
    d_rho_y_batch = (-dy / sigmas**2 * gauss).sum(dim=1)
    d_rho_z_batch = (-dz / sigmas**2 * gauss).sum(dim=1)

    return rho_batch, d_rho_x_batch, d_rho_y_batch, d_rho_z_batch


def compute_normalization_stats(features):
    """
    Compute mean and std for features with shape (N_data, N_x, N_y, N_z, N_feat)
    Averages over both data and spatial dimensions

    Args:
        features: torch.Tensor of shape (N_data, N_x, N_y, N_z, N_feat)

    Returns:
        mean: torch.Tensor of shape (1, 1, 1, 1, N_feat)
        std: torch.Tensor of shape (1, 1, 1, 1, N_feat)
    """

    mean_feat = features.mean(dim=(0, 1, 2, 3), keepdim=True)  # Shape: (1, 1, 1, 1, N_feat)
    std_feat = features.std(dim=(0, 1, 2, 3), keepdim=True) # Shape: (1, 1, 1, 1, N_feat)

    return mean_feat, std_feat


def normalize_features(features, mean_feat, std_feat):
    """
    Normalize features using provided or computed statistics

    Args:
        features: torch.Tensor of shape (B, N_x, N_y, N_z, N_feat)
        mean: torch.Tensor of shape (1, 1, 1, 1, N_feat)
        std: torch.Tensor of shape (1, 1, 1, 1, N_feat)

    Returns:
        normalized_features: torch.Tensor of same shape as input
    """
    normalized_features = (features - mean_feat) / std_feat

    return normalized_features


def generate_loc_features_rs(rho: torch.Tensor, N_pow=2) -> torch.Tensor:
    """
    Generate local features from density rho
    rs, real space
    Args:
        rho: torch.Tensor of shape (B, N_x, N_y, N_z)
        N_pow: int, number of features to generate

    Returns:
        features: torch.Tensor of shape (B, N_x, N_y, N_z, N_pow)
        each feature is of the form rho^k, k=1,...,N_pow
    """
    features = [rho.unsqueeze(-1) ** k for k in range(1, N_pow + 1)]
    return torch.cat(features, dim=-1)


def generate_loc_features_ms(d_rho_x: torch.Tensor, d_rho_y: torch.Tensor, d_rho_z: torch.Tensor, N_pow=2) -> torch.Tensor:
    """
    Generate local features from density derivatives d_rho_x, d_rho_y, d_rho_z
    ms, momentum space
    Args:
        d_rho_x, d_rho_y, d_rho_z: torch.Tensor of shape (B, N_x, N_y, N_z)
        N_pow: int, powers per direction

    Returns:
        features: torch.Tensor of shape (B, N_x, N_y, N_z, N_pow^3)
        each feature is of the form d_rho_x^k_x d_rho_y^k_y d_rho_z^k_z, k_x,k_y,k_z=1,...,N_pow
    """
    features = []
    for k_x in range(1, N_pow + 1):
        for k_y in range(1, N_pow + 1):
            for k_z in range(1, N_pow + 1):
                features.append((d_rho_x.unsqueeze(-1) ** k_x) * (d_rho_y.unsqueeze(-1) ** k_y) * (d_rho_z.unsqueeze(-1) ** k_z))
    return torch.cat(features, dim=-1)


def extend_features_neighbors_3d(features: torch.Tensor, R: float = 1.0, pad_mode: str = "zero") -> torch.Tensor:
    """
    Extend features by including neighboring grid points within radius R
    Args:
        features: torch.Tensor of shape (B, N_x, N_y, N_z, N_feat)
        R: float, radius of neighborhood in grid points
        pad_mode: "zero" (open BCs) or "reflect"
    """
    B, N_x, N_y, N_z, N_feat = features.shape
    pad_size = int(R)
    pad = (pad_size,) * 6
    x = features.permute(0, 4, 1, 2, 3)  # (B, N_feat, N_x, N_y, N_z)
    if pad_mode == "zero":
        padded_features = torch.nn.functional.pad(x, pad, mode='constant', value=0.0)
    elif pad_mode == "reflect":
        padded_features = torch.nn.functional.pad(x, pad, mode='reflect')
    else:
        raise ValueError("pad_mode must be zero or reflect")
    extended_features_list = []

    for dx in range(-pad_size, pad_size + 1):
        for dy in range(-pad_size, pad_size + 1):
            for dz in range(-pad_size, pad_size + 1):
                if (dx == 0 and dy == 0 and dz == 0) or (dx**2 + dy**2 + dz**2 > R**2):
                    continue
                shifted_features = padded_features[:, :,
                                                   pad_size + dx:pad_size + dx + N_x,
                                                   pad_size + dy:pad_size + dy + N_y,
                                                   pad_size + dz:pad_size + dz + N_z]
                extended_features_list.append(shifted_features.permute(0, 2, 3, 4, 1))

    extended_features = torch.cat(extended_features_list, dim=-1)
    return torch.cat([features, extended_features], dim=-1)


def generate_data_3d(
        N: int,                     # number of samples
        N_batch: int,               # batch size
        E_tot: EnergyFunction,      # total energy function
        std_harm: torch.Tensor,
        DM_x: torch.Tensor,
        DerDM_x: torch.Tensor,
        DM_y: torch.Tensor,
        DerDM_y: torch.Tensor,
        DM_z: torch.Tensor,
        DerDM_z: torch.Tensor
        ):
    """
    Generate dataset of density profiles and corresponding total energies
    Done in mini-batches of size N_batch to save memory
    """
    rho_list = []
    d_rho_x_list = []
    d_rho_y_list = []
    d_rho_z_list = []
    a_list = []
    E_list = []

    num_iter = (N + N_batch - 1) // N_batch
    with torch.no_grad():
        for i in range(num_iter):
            current_batch_size = min(N_batch, N - i * N_batch)
            rho_batch, d_rho_x_batch, d_rho_y_batch, d_rho_z_batch, a_batch = sample_density_batch(
                current_batch_size, std_harm=std_harm,
                DM_x=DM_x, DerDM_x=DerDM_x, DM_y=DM_y, DerDM_y=DerDM_y, DM_z=DM_z, DerDM_z=DerDM_z)
            E_batch = E_tot(rho_batch, d_rho_x_batch, d_rho_y_batch, d_rho_z_batch)  # (B,)

            rho_list.append(rho_batch)
            d_rho_x_list.append(d_rho_x_batch)
            d_rho_y_list.append(d_rho_y_batch)
            d_rho_z_list.append(d_rho_z_batch)
            a_list.append(a_batch)
            E_list.append(E_batch)

        rho_all = torch.cat(rho_list, dim=0)
        d_rho_x_all = torch.cat(d_rho_x_list, dim=0)
        d_rho_y_all = torch.cat(d_rho_y_list, dim=0)
        d_rho_z_all = torch.cat(d_rho_z_list, dim=0)
        a_all = torch.cat(a_list, dim=0)
        E_all = torch.cat(E_list, dim=0)

    return rho_all, d_rho_x_all, d_rho_y_all, d_rho_z_all, a_all, E_all


def generate_SC_data_3d(
        N: int,                         # number of samples
        N_batch: int,                   # batch size
        E_HF: LocEnergyFunction,        # unscreened total energy function
        E_SC: LocEnergyFunction,        # screened total energy function
        std_harm: torch.Tensor,
        DM_x: torch.Tensor,
        DerDM_x: torch.Tensor,
        DM_y: torch.Tensor,
        DerDM_y: torch.Tensor,
        DM_z: torch.Tensor,
        DerDM_z: torch.Tensor
        ):
    """
    Generate dataset of density profiles and corresponding total energies
    Done in mini-batches of size N_batch to save memory
    """
    rho_list = []
    d_rho_x_list = []
    d_rho_y_list = []
    d_rho_z_list = []
    a_list = []
    E_loc_HF_list = []
    E_SC_list = []

    num_iter = (N + N_batch - 1) // N_batch
    with torch.no_grad():
        for i in range(num_iter):
            current_batch_size = min(N_batch, N - i * N_batch)
            rho_batch, d_rho_x_batch, d_rho_y_batch, d_rho_z_batch, a_batch = sample_density_batch(
                current_batch_size, std_harm=std_harm,
                DM_x=DM_x, DerDM_x=DerDM_x, DM_y=DM_y, DerDM_y=DerDM_y, DM_z=DM_z, DerDM_z=DerDM_z)

            E_loc_HF_batch = E_HF(rho_batch, d_rho_x_batch, d_rho_y_batch, d_rho_z_batch, eng_dens_flag=True)  # (B, N_x, N_y, N_z, 1)
            E_SC_batch = E_SC(rho_batch, d_rho_x_batch, d_rho_y_batch, d_rho_z_batch, eng_dens_flag=False)  # (B,)

            rho_list.append(rho_batch)
            d_rho_x_list.append(d_rho_x_batch)
            d_rho_y_list.append(d_rho_y_batch)
            d_rho_z_list.append(d_rho_z_batch)
            a_list.append(a_batch)
            E_loc_HF_list.append(E_loc_HF_batch)
            E_SC_list.append(E_SC_batch)

        rho_all = torch.cat(rho_list, dim=0)
        d_rho_x_all = torch.cat(d_rho_x_list, dim=0)
        d_rho_y_all = torch.cat(d_rho_y_list, dim=0)
        d_rho_z_all = torch.cat(d_rho_z_list, dim=0)
        a_all = torch.cat(a_list, dim=0)
        E_HF_all = torch.cat(E_loc_HF_list, dim=0)
        E_SC_all = torch.cat(E_SC_list, dim=0)

    return rho_all, d_rho_x_all, d_rho_y_all, d_rho_z_all, a_all, E_HF_all, E_SC_all


def E_kin_custom(
        rho: torch.Tensor,
        d_rho_x: torch.Tensor,
        d_rho_y: torch.Tensor,
        d_rho_z: torch.Tensor,
        alpha: float,
        beta: float,
        qs: float,
        eng_dens_flag: bool = False
        ) -> torch.Tensor:
    """
    Kinetic energy functional:
        E_kin = 1 / (2 N_x N_y N_z) sum_r kappa_r (d_rho_x_r^2 + d_rho_y_r^2 + d_rho_z_r^2),
        where kappa_r = 1 + alpha * prod_{e in {±ex, ±ey, ±ez}} rho_{r + e} + beta * phi_r,
        phi_r is the mediator field, with screening length set by qs

    Args:
        rho:      (N_x, N_y, N_z) or (B, N_x, N_y, N_z)
        d_rho_x:  same shape - derivative of rho w.r.t. x
        d_rho_y:  same shape - derivative of rho w.r.t. y
        d_rho_z:  same shape - derivative of rho w.r.t. z
    """

    if rho.dim() == 3:
        rho = rho.unsqueeze(0)
        d_rho_x = d_rho_x.unsqueeze(0)
        d_rho_y = d_rho_y.unsqueeze(0)
        d_rho_z = d_rho_z.unsqueeze(0)

    B, N_x, N_y, N_z = rho.shape
    device, dtype = rho.device, rho.dtype


    R_feat = 1.0 # radius for neighbor feature extension
    rho_neighbours = extend_features_neighbors_3d(rho.unsqueeze(-1), R=R_feat) # (B, N_x, N_y, N_z, 1 + N_nb), N_nb = 6 nearest neighbors

    # prod_{e in {±ex, ±ey, ±ez}} rho_{r + e}
    alpha_term = rho_neighbours[..., 1:].prod(dim=-1) # (B, N_x, N_y, N_z)

    # compute mediator field phi using FFT routines
    q_vals = q_grid_fft(N_x, N_y, N_z, device=device, dtype=dtype)  # (2N_x, 2N_y, N_z+1)
    lam_K = Lam_K_Coulomb(q_vals, qs=qs).to(device=device, dtype=dtype)

    phi = conv_fft(rho, lam_K)  # (B, N_x, N_y, N_z)

    kappa = 1.0 + alpha * alpha_term + beta * phi # (B, N_x, N_y, N_z)

    E_kin_loc = 0.5 * kappa * (d_rho_x ** 2 + d_rho_y ** 2 + d_rho_z ** 2)  # (B, N_x, N_y, N_z)
    if eng_dens_flag:
        return E_kin_loc  # (B, N_x, N_y, N_z)

    return E_kin_loc.sum(dim=(1, 2, 3)) / (N_x * N_y * N_z) # (B,)
