# feat_utils.py
import torch

def sample_density(std_harm: torch.Tensor,
                   DesignMatrix: torch.Tensor,
                   DerDM: torch.Tensor):
    """
    Sample rho_j = sum_{m=1}^M a_m cos(m pi x_j), with x_j in [0,1].
    This sampling of amplitudes a_m implies that the derivatives at 
    the boundaries are zero.

    Args:
        std_harm:     (M,) tensor of standard deviations for a_m
        DesignMatrix: (M, N_grid) tensor with cosines evaluated on the grid
        DerDM:        (M, N_grid) tensor with derivatives d/dx of the cosines

    Returns:
        rho    : (N_grid,) density profile
        d_rho  : (N_grid,) derivative of rho wrt x
        a      : (M,) sampled amplitudes
    """
    device = std_harm.device
    dtype  = std_harm.dtype
    M      = std_harm.numel()

    # a_m ~ N(0, std_harm_m^2)
    a = torch.normal(mean=torch.zeros(M, device=device, dtype=dtype),
                     std=std_harm)

    rho   = a @ DesignMatrix   # (N_grid,)
    d_rho = a @ DerDM          # (N_grid,)

    return rho, d_rho, a


def sample_density_batch(B: int,
                         std_harm: torch.Tensor,
                         DesignMatrix: torch.Tensor,
                         DerDM: torch.Tensor):
    """
    Sample a batch of B density profiles.

    Args:
        B           : batch size
        std_harm    : (M,) tensor of std devs for a_m
        DesignMatrix: (M, N_grid)
        DerDM       : (M, N_grid)

    Returns:
        rho_batch   : (B, N_grid)
        d_rho_batch : (B, N_grid)
        a_batch     : (B, M)
    """
    device = std_harm.device
    dtype  = std_harm.dtype
    M      = std_harm.numel()

    # (B, M), each row a_m ~ N(0, std_harm_m^2)
    a_batch = torch.normal(
        mean=torch.zeros(B, M, device=device, dtype=dtype),
        std=std_harm.expand(B, -1)
    )  # (B, M)

    rho_batch   = a_batch @ DesignMatrix  # (B, N_grid)
    d_rho_batch = a_batch @ DerDM         # (B, N_grid)

    return rho_batch, d_rho_batch, a_batch


def compute_normalization_stats(features):
    """
    Compute mean and std for features with shape (N_data, N_grid, N_feat)
    Averages over both data and spatial dimensions
    
    Args:
        features: torch.Tensor of shape (N_data, N_grid, N_feat)
    
    Returns:
        mean: torch.Tensor of shape (1, 1, N_feat)
        std: torch.Tensor of shape (1, 1, N_feat)
    """

    mean_feat = features.mean(dim=(0, 1), keepdim=True)  # Shape: (1, 1, N_feat)
    std_feat = features.std(dim=(0, 1), keepdim=True) # Shape: (1, 1, N_feat)
    
    return mean_feat, std_feat

def normalize_features(features, mean_feat, std_feat):
    """
    Normalize features using provided or computed statistics
    
    Args:
        features: torch.Tensor of shape (B, N_grid, N_feat)
        mean: torch.Tensor of shape (1, 1, N_feat)
        std: torch.Tensor of shape (1, 1, N_feat)

    Returns:
        normalized_features: torch.Tensor of same shape as input
        mean: mean used for normalization
        std: std used for normalization
    """
    normalized_features = (features - mean_feat) / std_feat

    return normalized_features

def generate_loc_features_rs(rho: torch.Tensor, N_feat=2) -> torch.Tensor:
    """
    Generate local features from density rho
    rs, real space   
    Args:
        rho: torch.Tensor of shape (B, N_grid)
        N_feat: int, number of features to generate

    Returns:
        features: torch.Tensor of shape (B, N_grid, N_feat)
        each feature is of the form rho^k, k=1,...,N_feat
    """
    features = [rho.unsqueeze(-1) ** k for k in range(1, N_feat + 1)]
    return torch.cat(features, dim=-1)


def generate_loc_features_ms(d_rho: torch.Tensor, N_feat=2) -> torch.Tensor:
    """
    Generate local features from density derivative d_rho
    ms, momentum space
    Args:
        d_rho: torch.Tensor of shape (B, N_grid)
        N_feat: int, number of features to generate

    Returns:
        features: torch.Tensor of shape (B, N_grid, N_feat)
        each feature is of the form d_rho^k, k=1,...,N_feat
    """
    features = [d_rho.unsqueeze(-1) ** k for k in range(1, N_feat + 1)]
    return torch.cat(features, dim=-1)