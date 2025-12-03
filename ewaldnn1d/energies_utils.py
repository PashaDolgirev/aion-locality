# interaction_kernels.py

import torch
import torch.nn.functional as F

from .dct_utils import (
    kernel_eigenvals_dct,
    rho_to_cosine_coeffs,
    cosine_coeffs_to_rho,
)


# ---- analytic density–density interaction kernels K(r) ----
# r: tensor (can be negative)

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


# ---- real-space energy via convolution ----

def E_int_conv(
    rho: torch.Tensor,
    kernel: str,
    pad_mode: str = "zero",
    **kwargs,
) -> torch.Tensor:
    """
    Interaction energy using real-space convolution.

    E_int = (1 / 2N) sum_{i,j} K_{i-j} rho_i rho_j
          = (1 / 2N) sum_i rho_i [K * rho]_i

    Args:
        rho:    (N,) or (B, N)
        kernel: "gaussian", "exp", "yukawa", "power"
        pad_mode: "zero" or "reflect"
        kwargs: parameters for the kernel function (sigma, xi, lam, alpha, etc.)

    Returns:
        scalar if input was 1D, otherwise (B,)
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

    # ensure batch dim
    if rho.dim() == 1:
        rho = rho.unsqueeze(0)
    B, N = rho.shape
    device, dtype = rho.device, rho.dtype

    # r = -(N-1)..(N-1), kernel length = 2N-1
    r_vals = torch.arange(-(N - 1), N, device=device, dtype=dtype)  # (2N-1,)
    k_full = K_fun(r_vals, **kwargs).to(device=device, dtype=dtype)  # (2N-1,)

    weight = k_full.view(1, 1, -1)  # (1, 1, 2N-1)

    if pad_mode == "zero":
        u = F.conv1d(rho.unsqueeze(1), weight, padding=N - 1).squeeze(1)  # (B, N)
    elif pad_mode == "reflect":
        # even reflection padding
        rho_pad = F.pad(rho.unsqueeze(1), (N - 1, N - 1), mode="reflect")  # (B,1,N+2(N-1))
        u = F.conv1d(rho_pad, weight).squeeze(1)
    else:
        raise ValueError(f"Unknown padding: {pad_mode}")

    E = 0.5 * (rho * u).sum(dim=-1) / N  # (B,)
    return E.squeeze(0) if E.numel() == 1 else E


# ---- DCT-based energy ----

def E_int_dct(
    rho: torch.Tensor,
    kernel: str,
    **kwargs,
) -> torch.Tensor:
    """
    Interaction energy using DCT-I eigenvalues of the convolution operator.

    Same definition:
        E_int = (1 / 2N) sum_i rho_i [K * rho]_i

    Args:
        rho:    (N,) or (B, N)
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

    if rho.dim() == 1:
        rho = rho.unsqueeze(0)
    B, N = rho.shape
    device, dtype = rho.device, rho.dtype

    r_vals = torch.arange(0, N, device=device, dtype=dtype)  # (N,)
    K_vals = K_fun(r_vals, **kwargs).to(device=device, dtype=dtype)  # (N,)

    lam_K = kernel_eigenvals_dct(K_vals).to(device=device, dtype=dtype)  # (N,)
    a = rho_to_cosine_coeffs(rho)                    # (B, N)
    u = cosine_coeffs_to_rho(lam_K.unsqueeze(0) * a) # (B, N)

    E = 0.5 * (rho * u).sum(dim=-1) / N  # (B,)
    return E.squeeze(0) if E.numel() == 1 else E


# ---- DCT-based energy with explicit boundary corrections ----

def E_int_dct_v2(
    rho: torch.Tensor,
    kernel: str,
    **kwargs,
) -> torch.Tensor:
    """
    Interaction energy using DCT-I eigenvalues with explicit boundary corrections.
    
    Args:
        rho:    (N,) or (B, N)
        kernel: "gaussian", "exp", "yukawa", "power"
        kwargs: kernel parameters

    Returns:
        scalar if input was 1D, otherwise (B,)
    """
    # select kernel
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

    if rho.dim() == 1:
        rho = rho.unsqueeze(0)
    B, N = rho.shape
    device, dtype = rho.device, rho.dtype

    r_vals = torch.arange(0, N, device=device, dtype=dtype)   # (N,)
    K_vals = K_fun(r_vals, **kwargs).to(device=device, dtype=dtype)  # (N,)
    lam_K = kernel_eigenvals_dct(K_vals).to(device=device, dtype=dtype)  # (N,)
    a = rho_to_cosine_coeffs(rho)                             # (B, N)

    # spectral coefficients: λ_m a_m
    spec = lam_K.unsqueeze(0) * a                             # (B, N)

    # ---------- diagonal (bulk) term ----------
    # norms n_m = <φ_m|φ_m> in your weighted scalar product
    norms = torch.full(
        (N,),
        (N - 1.0) / 2.0,
        device=device,
        dtype=dtype,
    )
    norms[0] = N - 1.0
    norms[-1] = N - 1.0

    # E_diag = (1/(2N)) * sum_m λ_m a_m^2 n_m
    E_diag = 0.5 * (spec * a * norms.unsqueeze(0)).sum(dim=-1) / N   # (B,)

    # ---------- boundary (mixing) term ----------
    sum_spec = spec.sum(dim=-1)                                      # (B,)
    sign = (-1.0) ** torch.arange(N, device=device, dtype=dtype)     # (N,)
    sum_spec_signed = (spec * sign.unsqueeze(0)).sum(dim=-1)         # (B,)

    # E_bnd = (1/(4N)) * [ ρ_0 * sum_m λ_m a_m + ρ_{N-1} * sum_m λ_m a_m (-1)^m ]
    E_bnd = 0.25 * (rho[:, 0] * sum_spec + rho[:, -1] * sum_spec_signed) / N  # (B,)

    E = E_diag + E_bnd
    return E.squeeze(0) if E.numel() == 1 else E