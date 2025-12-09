# feat_utils.py
import torch

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