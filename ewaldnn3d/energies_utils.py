# energies_utils.py

import torch
import torch.nn.functional as F

from .fft_utils import (
    kernel_eigenvals_fft,
    kernel_from_eigenvals_fft,
    conv_fft,
    displacement_grid,
    q_grid_fft,
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
    Yukawa kernel (screened Coulomb in 3D):
        K(r) = exp(-|r| / lam) / |r|
    with K(0) = 0 (no self-interaction).
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
    """
    Screened Coulomb in 3D momentum space: lam_K(q) = 4 pi / (q^2 + qs^2),
    with zero mode and self-interaction removed, and a UV cutoff.

    q: (2N_x, 2N_y, N_z+1) radial momenta on the padded rfftn grid (see q_grid_fft)
    """
    denom = q**2 + qs**2
    denom = torch.where(denom == 0, denom + 1e-12, denom)
    lam_K = 4.0 * torch.pi / denom

    lam_K[0, 0, 0] = 0.0 # remove uniform mode
    lam_K = lam_K * (q < Lambda_UV) # UV cutoff

    # go to real space, kill self-interaction, go back
    K = kernel_from_eigenvals_fft(lam_K)
    K[0, 0, 0] = 0.0
    lam_K = kernel_eigenvals_fft(K)

    return lam_K


# ---- real-space energy via convolution ----

def E_int_conv(
    rho: torch.Tensor,
    kernel: str,
    pad_mode: str = "zero",
    **kwargs,
) -> torch.Tensor:
    """
    Interaction energy using real-space convolution (brute-force conv3d;
    intended for small grids and testing).

    E_int = (1 / (2 N_x N_y N_z)) sum_{r1, r2} K_{r1-r2} rho_{r1} rho_{r2}
          = (1 / (2 N_x N_y N_z)) sum_r rho_r [K * rho]_r.

    Args:
        rho:    (N_x, N_y, N_z) or (B, N_x, N_y, N_z) tensor
        kernel: "gaussian", "exp", "yukawa", "power"
        pad_mode: "zero" (open BCs) or "reflect"
        kwargs: parameters for the kernel function (sigma, xi, lam, alpha, etc.)

    Returns:
        scalar if input was 3D, otherwise (B,)
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

    # ensure batch dim: (B, N_x, N_y, N_z)
    batched = rho.dim() == 4
    if not batched:
        rho = rho.unsqueeze(0)
    B, N_x, N_y, N_z = rho.shape
    device, dtype = rho.device, rho.dtype

    # displacement grid r ∈ {-(N-1)..(N-1)} per dimension
    x_vals = torch.arange(-(N_x - 1), N_x, device=device, dtype=dtype).view(-1, 1, 1)
    y_vals = torch.arange(-(N_y - 1), N_y, device=device, dtype=dtype).view(1, -1, 1)
    z_vals = torch.arange(-(N_z - 1), N_z, device=device, dtype=dtype).view(1, 1, -1)
    r_vals = torch.sqrt(x_vals**2 + y_vals**2 + z_vals**2)  # (2N_x-1, 2N_y-1, 2N_z-1)

    # full 3D kernel on displacement grid
    k_full = K_fun(r_vals, **kwargs).to(device=device, dtype=dtype)

    # conv3d expects (out_channels, in_channels, kD, kH, kW)
    weight = k_full.view(1, 1, 2 * N_x - 1, 2 * N_y - 1, 2 * N_z - 1)

    if pad_mode == "zero":
        u = F.conv3d(rho.unsqueeze(1), weight, padding=(N_x - 1, N_y - 1, N_z - 1)).squeeze(1)
    elif pad_mode == "reflect":
        rho_pad = F.pad(
            rho.unsqueeze(1),
            (N_z - 1, N_z - 1, N_y - 1, N_y - 1, N_x - 1, N_x - 1),
            mode="reflect",
        )
        u = F.conv3d(rho_pad, weight).squeeze(1)
    else:
        raise ValueError(f"Unknown padding: {pad_mode}")

    # E = (1 / 2 N_x N_y N_z) Σ_r rho_r u_r per batch
    E = 0.5 * (rho * u).sum(dim=(-3, -2, -1)) / (N_x * N_y * N_z)  # (B,)
    return E if batched else E.squeeze(0)


# ---- FFT-based energy (open BCs via zero padding) ----

def E_int_rs_fft(
    rho: torch.Tensor,
    kernel: str,
    **kwargs,
) -> torch.Tensor:
    """
    Interaction energy for a real-space kernel, computed via zero-padded FFT.

    Same definition:
        E_int = (1 / (2 N_x N_y N_z)) sum_r rho_r [K * rho]_r.

    Args:
        rho:    (N_x, N_y, N_z) or (B, N_x, N_y, N_z)
        kernel: "gaussian", "exp", "yukawa", "power"
        kwargs: parameters for K_fun

    Returns:
        scalar if input was 3D, otherwise (B,)
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

    batched = rho.dim() == 4
    if not batched:
        rho = rho.unsqueeze(0)
    B, N_x, N_y, N_z = rho.shape
    device, dtype = rho.device, rho.dtype

    r_vals = displacement_grid(N_x, N_y, N_z, device=device, dtype=dtype)  # (N_x, N_y, N_z)
    K_vals = K_fun(r_vals, **kwargs).to(device=device, dtype=dtype)

    lam_K = kernel_eigenvals_fft(K_vals)  # (2N_x, 2N_y, N_z+1)
    u = conv_fft(rho, lam_K)              # (B, N_x, N_y, N_z)

    E = 0.5 * (rho * u).sum(dim=(-3, -2, -1)) / (N_x * N_y * N_z)  # (B,)
    return E if batched else E.squeeze(0)


def E_int_ms_fft(rho, kernel: str, eng_dens_flag: bool = False, **kwargs) -> torch.Tensor:
    """
    Interaction energy for a momentum-space kernel, computed via zero-padded FFT.

    Same definition:
        E_int = (1 / (2 N_x N_y N_z)) sum_r rho_r [K * rho]_r.

    Args:
        rho:    (N_x, N_y, N_z) or (B, N_x, N_y, N_z)
        kernel: "screened_coulomb" - parametrized in momentum space
        kwargs: parameters for Lam_fun

    Returns:
        scalar if input was 3D, otherwise (B,);
        if eng_dens_flag, local energy density (B, N_x, N_y, N_z, 1)
    """
    if kernel == "screened_coulomb":
        Lam_fun = Lam_K_Coulomb
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    batched = rho.dim() == 4
    if not batched:
        rho = rho.unsqueeze(0)
    B, N_x, N_y, N_z = rho.shape
    device, dtype = rho.device, rho.dtype

    q_vals = q_grid_fft(N_x, N_y, N_z, device=device, dtype=dtype)  # (2N_x, 2N_y, N_z+1)
    lam_K = Lam_fun(q_vals, **kwargs).to(device=device, dtype=dtype)

    u = conv_fft(rho, lam_K)  # (B, N_x, N_y, N_z)

    E_loc = 0.5 * rho * u # (B, N_x, N_y, N_z)

    if eng_dens_flag:
        return E_loc.unsqueeze(-1)  # (B, N_x, N_y, N_z, 1)

    E = E_loc.sum(dim=(-3, -2, -1)) / (N_x * N_y * N_z) # (B,)
    return E if batched else E.squeeze(0)
