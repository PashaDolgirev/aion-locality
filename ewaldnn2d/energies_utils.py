# interaction_kernels.py

import torch
import torch.nn.functional as F

from .dct_utils import (
    kernel_eigenvals_dct,
    kernel_from_eigenvals_dct,
    rho_to_cosine_coeffs,
    cosine_coeffs_to_rho,
)

# ---- analytic density–density interaction kernels ----
# r: tensor (can be negative)
# q: tensor (can be negative)

def K_gaussian(r: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Gaussian kernel: K(r) = exp(-r^2 / sigma^2)
    """
    r = r.to(dtype=torch.get_default_dtype())
    return torch.exp(-(r ** 2) / (sigma ** 2))


def K_exp(r: torch.Tensor, xi: float = 2.0) -> torch.Tensor:
    """
    Exponential kernel: K(r) = exp(-|r| / xi)
    """
    r = r.to(dtype=torch.get_default_dtype())
    return torch.exp(-torch.abs(r) / xi)


def K_yukawa(r: torch.Tensor, lam: float = 10.0) -> torch.Tensor:
    """
    Yukawa-like kernel:
        K(r) = exp(-|r| / lam) / |r|
    with K(0) = 0.
    """
    r_abs = torch.abs(r).to(dtype=torch.get_default_dtype())
    out = torch.exp(-r_abs / lam) / r_abs.clamp(min=1.0)
    return out * (r_abs > 0)  # zero at r == 0


def K_power(r: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Power-law kernel:
        K(r) = 1 / |r|^alpha
    with K(0) = 0.
    """
    r_abs = torch.abs(r).to(dtype=torch.get_default_dtype())
    out = 1.0 / (r_abs.clamp(min=1.0) ** alpha)
    return out * (r_abs > 0)  # zero at r == 0


def Lam_K_Coulomb(q: torch.Tensor, qs: float = 0.0, Lambda_UV: float = 1000.0) -> torch.Tensor:
    # log-like Coulomb in 2D momentum space
    denom = q**2 + qs**2
    denom = torch.where(denom == 0, denom + 1e-12, denom)
    lam_K = 2.0 * torch.pi / denom

    lam_K[0, 0] = 0.0 # remove uniform mode
    lam_K = lam_K * (q < Lambda_UV) # UV cutoff

    # go to real space, kill self-interaction, go back
    K = kernel_from_eigenvals_dct(lam_K)
    K[0, 0] = 0.0
    lam_K = kernel_eigenvals_dct(K)

    return lam_K


# ---- real-space energy via convolution ----

def E_int_conv(
    rho: torch.Tensor,
    kernel: str,
    pad_mode: str = "reflect",
    **kwargs,
) -> torch.Tensor:
    """
    Interaction energy using real-space convolution.

    E_int = (1 / (2 N_x N_y)) sum_{i1,j1; i2,j2} K_{i1-i2, j1-j2} rho_{i1,j1} rho_{i2,j2}
          = (1 / (2 N_x N_y)) sum_{i,j} rho_{i,j} [K * rho]_{i,j}.

    Args:
        rho:    (N_x, N_y) or (B, N_x, N_y) tensor
        kernel: "gaussian", "exp", "yukawa", "power"
        pad_mode: "zero" or "reflect"
        kwargs: parameters for the kernel function (sigma, xi, lam, alpha, etc.)

    Returns:
        scalar if input was 2D, otherwise (B,)
    """
    # select kernel function
    if kernel == "gaussian":
        K_fun = K_gaussian
    elif kernel == "exp":
        K_fun = K_exp
    elif kernel == "yukawa":
        K_fun = K_yukawa
    elif kernel == "power":
        K_fun = K_power
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # ensure batch dim: (B, N_x, N_y)
    if rho.dim() == 2:
        rho = rho.unsqueeze(0)
    B, N_x, N_y = rho.shape
    device, dtype = rho.device, rho.dtype

    # displacement grid r_x, r_y ∈ {-(N_x-1)..(N_x-1)} × {-(N_y-1)..(N_y-1)}
    x_vals = torch.arange(-(N_x - 1), N_x, device=device, dtype=dtype)     # (2N_x-1,)
    y_vals = torch.arange(-(N_y - 1), N_y, device=device, dtype=dtype)     # (2N_y-1,)
    r_vals = torch.sqrt(x_vals.view(-1, 1) ** 2 + y_vals.view(1, -1) ** 2)  # (2N_x-1, 2N_y-1)

    # full 2D kernel on displacement grid
    k_full = K_fun(r_vals, **kwargs).to(device=device, dtype=dtype)        # (2N_x-1, 2N_y-1)

    # conv2d expects (out_channels, in_channels, kH, kW)
    weight = k_full.view(1, 1, 2 * N_x - 1, 2 * N_y - 1)

    if pad_mode == "zero":
        # zero padding with pad=(padH, padW)
        u = F.conv2d(rho.unsqueeze(1), weight, padding=(N_x - 1, N_y - 1)).squeeze(1)  # (B, N_x, N_y)
    elif pad_mode == "reflect":
        # even-ish reflection padding; F.pad uses (left, right, top, bottom)
        rho_pad = F.pad(
            rho.unsqueeze(1),
            (N_y - 1, N_y - 1, N_x - 1, N_x - 1),
            mode="reflect",
        )  # (B,1, N_x+2(N_x-1), N_y+2(N_y-1))
        u = F.conv2d(rho_pad, weight).squeeze(1)  # (B, N_x, N_y)
    else:
        raise ValueError(f"Unknown padding: {pad_mode}")

    # E = (1/2N_xN_y) Σ_{i,j} rho_{i,j} u_{i,j} per batch
    E = 0.5 * (rho * u).sum(dim=(-2, -1)) / (N_x * N_y)  # (B,)
    return E.squeeze(0) if E.numel() == 1 else E


# ---- DCT-based energy ----

def E_int_rs_dct(
    rho: torch.Tensor,
    kernel: str,
    **kwargs,
) -> torch.Tensor:
    """
    Interaction energy using DCT-I eigenvalues of the convolution operator.

    Same definition:
        E_int = (1 / (2 N_x N_y)) sum_{i1,j1; i2,j2} K_{i1-i2, j1-j2} rho_{i1,j1} rho_{i2,j2}
          = (1 / (2 N_x N_y)) sum_{i,j} rho_{i,j} [K * rho]_{i,j}.

    Args:
        rho:    (N_x, N_y) or (B, N_x, N_y)
        kernel: "gaussian", "exp", "yukawa", "power"
        kwargs: parameters for K_fun

    Returns:
        scalar if input was 1D, otherwise (B,)
    """
    if kernel == "gaussian":
        K_fun = K_gaussian
    elif kernel == "exp":
        K_fun = K_exp
    elif kernel == "yukawa":
        K_fun = K_yukawa
    elif kernel == "power":
        K_fun = K_power
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    if rho.dim() == 2:
        rho = rho.unsqueeze(0)
    B, N_x, N_y = rho.shape
    device, dtype = rho.device, rho.dtype

    x_vals = torch.arange(0, N_x, device=device, dtype=dtype)     # (N_x,)
    y_vals = torch.arange(0, N_y, device=device, dtype=dtype)     # (N_y,)
    r_vals = torch.sqrt(x_vals.view(-1, 1) ** 2 + y_vals.view(1, -1) ** 2)  # (N_x, N_y)

    K_vals = K_fun(r_vals, **kwargs).to(device=device, dtype=dtype)  # (N_x, N_y)

    lam_K = kernel_eigenvals_dct(K_vals).to(device=device, dtype=dtype)  # (N_x, N_y)
    a = rho_to_cosine_coeffs(rho)                    # (B, N_x, N_y)
    u = cosine_coeffs_to_rho(lam_K.unsqueeze(0) * a) # (B, N_x, N_y)

    E = 0.5 * (rho * u).sum(dim=(-2, -1)) / (N_x * N_y)  # (B,)
    return E.squeeze(0) if E.numel() == 1 else E


def E_int_ms_dct(rho, kernel: str, **kwargs):
    """
    Interaction energy using DCT-I eigenvalues of the convolution operator.

    Same definition:
        E_int = (1 / (2 N_x N_y)) sum_{i1,j1; i2,j2} K_{i1-i2, j1-j2} rho_{i1,j1} rho_{i2,j2}
          = (1 / (2 N_x N_y)) sum_{i,j} rho_{i,j} [K * rho]_{i,j}.

    Args:
        rho:    (N_x, N_y) or (B, N_x, N_y)
        kernel: "screened_coulomb" - parametrized in momentum space
        kwargs: parameters for K_fun

    Returns:
        scalar if input was 1D, otherwise (B,)
    """
    if kernel == "screened_coulomb":
        Lam_fun = Lam_K_Coulomb
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    if rho.dim() == 2:
        rho = rho.unsqueeze(0)
    B, N_x, N_y = rho.shape
    device, dtype = rho.device, rho.dtype

    m_vals = torch.arange(0, N_x, device=device, dtype=dtype)
    n_vals = torch.arange(0, N_y, device=device, dtype=dtype)
    q_x = torch.pi * m_vals.view(-1, 1) / (N_x - 1)
    q_y = torch.pi * n_vals.view(1, -1) / (N_y - 1)
    q_vals = torch.sqrt(q_x**2 + q_y**2)  # (N_x, N_y)

    lam_K = Lam_fun(q_vals, **kwargs).to(device=device, dtype=dtype)  # (N_x, N_y)

    a = rho_to_cosine_coeffs(rho)                     # (B, N_x, N_y)
    u = cosine_coeffs_to_rho(lam_K.unsqueeze(0) * a)  # (B, N_x, N_y)

    E = 0.5 * (rho * u).sum(dim=(-2, -1)) / (N_x * N_y)
    return E.squeeze(0) if E.numel() == 1 else E