# fft_utils.py
#
# FFT utilities for 3D convolutions with OPEN boundary conditions (zero padding).
# The density lives on a (N_x, N_y, N_z) grid; convolutions are computed as exact
# linear (non-circular) convolutions by zero padding to (2N_x, 2N_y, 2N_z) and
# using the convolution theorem (Hockney's method for isolated systems).
#
# Conventions:
# - Kernels are even in each coordinate: K_{r_x,r_y,r_z} = K_{|r_x|,|r_y|,|r_z|},
#   and are stored on the "octant" grid r_x = 0..N_x-1, r_y = 0..N_y-1, r_z = 0..N_z-1.
# - Eigenvalues lam_K live on the rfftn grid of the padded box: (2N_x, 2N_y, N_z+1).
#   They are real because the kernel is even.
# - Memory: the padded grid has 8x the volume; for N_x = N_y = N_z = 128 one padded
#   float64 field is ~134 MB. Consider GPU / float32 and adjust N_batch.

import torch


def kernel_wraparound_embed(K: torch.Tensor) -> torch.Tensor:
    """
    Embed an octant kernel into the padded box in wrap-around (fft) order.

    K: (N_x, N_y, N_z) kernel values at r >= 0 (even kernel assumed).

    Returns:
        K_wrap: (2N_x, 2N_y, 2N_z) with K_wrap[i] = K[min(i, P - i)] per dim (P = 2N),
                clamped to N-1 (the displacement N is never accessed for outputs
                restricted to the first N points, so the clamp is harmless).
    """
    K_wrap = K
    for dim, N in enumerate(K.shape):
        P = 2 * N
        i = torch.arange(P, device=K.device)
        idx = torch.minimum(i, P - i).clamp(max=N - 1)
        K_wrap = K_wrap.index_select(dim, idx)
    return K_wrap


def kernel_eigenvals_fft(K: torch.Tensor) -> torch.Tensor:
    """
    Compute convolution eigenvalues of an even octant kernel via rfftn on the
    padded (2N_x, 2N_y, 2N_z) box.

    K: (N_x, N_y, N_z) kernel values at r >= 0.

    Returns:
        lam_K: (2N_x, 2N_y, N_z+1) real tensor (spectrum of an even kernel is real).
    """
    K_wrap = kernel_wraparound_embed(K)
    return torch.fft.rfftn(K_wrap, dim=(-3, -2, -1)).real


def kernel_from_eigenvals_fft(lam_K: torch.Tensor) -> torch.Tensor:
    """
    Invert kernel_eigenvals_fft: reconstruct the octant kernel K_{r>=0}.

    lam_K: (2N_x, 2N_y, N_z+1) real tensor
    Returns:
        K: (N_x, N_y, N_z) kernel values at r_x, r_y, r_z = 0..N-1
    """
    P_x, P_y = lam_K.shape[-3], lam_K.shape[-2]
    P_z = 2 * (lam_K.shape[-1] - 1)
    if not lam_K.is_complex():
        lam_K = lam_K.to(torch.complex128 if lam_K.dtype == torch.float64 else torch.complex64)
    K_wrap = torch.fft.irfftn(lam_K, s=(P_x, P_y, P_z), dim=(-3, -2, -1))
    return K_wrap[..., :P_x // 2, :P_y // 2, :P_z // 2]


def conv_fft(rho: torch.Tensor, lam_K: torch.Tensor) -> torch.Tensor:
    """
    Exact linear convolution phi = (K * rho) with open BCs via zero padding.

    rho:   (B, N_x, N_y, N_z) or (N_x, N_y, N_z)
    lam_K: (2N_x, 2N_y, N_z+1) real eigenvalues on the padded rfftn grid

    Returns:
        phi: same shape as rho
    """
    squeeze = rho.dim() == 3
    if squeeze:
        rho = rho.unsqueeze(0)
    B, N_x, N_y, N_z = rho.shape

    rho_pad = torch.nn.functional.pad(rho, (0, N_z, 0, N_y, 0, N_x))  # (B, 2N_x, 2N_y, 2N_z)
    phi_pad = torch.fft.irfftn(
        torch.fft.rfftn(rho_pad, dim=(-3, -2, -1)) * lam_K.unsqueeze(0),
        s=(2 * N_x, 2 * N_y, 2 * N_z), dim=(-3, -2, -1),
    )
    phi = phi_pad[..., :N_x, :N_y, :N_z]
    return phi.squeeze(0) if squeeze else phi


def displacement_grid(N_x: int, N_y: int, N_z: int, device=None, dtype=None) -> torch.Tensor:
    """
    Radial displacements |r| on the octant grid.

    Returns:
        r_vals: (N_x, N_y, N_z), r = sqrt(r_x^2 + r_y^2 + r_z^2), r_i = 0..N_i-1
    """
    dtype = dtype or torch.get_default_dtype()
    rx = torch.arange(N_x, device=device, dtype=dtype).view(-1, 1, 1)
    ry = torch.arange(N_y, device=device, dtype=dtype).view(1, -1, 1)
    rz = torch.arange(N_z, device=device, dtype=dtype).view(1, 1, -1)
    return torch.sqrt(rx**2 + ry**2 + rz**2)


def q_grid_fft(N_x: int, N_y: int, N_z: int, device=None, dtype=None) -> torch.Tensor:
    """
    Radial momenta |q| on the rfftn grid of the padded (2N_x, 2N_y, 2N_z) box,
    in lattice units: q_i = 2 pi k_i / (2 N_i) with signed frequency index k_i.

    Returns:
        q_vals: (2N_x, 2N_y, N_z+1)
    """
    dtype = dtype or torch.get_default_dtype()
    q_x = 2.0 * torch.pi * torch.fft.fftfreq(2 * N_x, device=device).to(dtype)   # (2N_x,)
    q_y = 2.0 * torch.pi * torch.fft.fftfreq(2 * N_y, device=device).to(dtype)   # (2N_y,)
    q_z = 2.0 * torch.pi * torch.fft.rfftfreq(2 * N_z, device=device).to(dtype)  # (N_z+1,)
    return torch.sqrt(q_x.view(-1, 1, 1) ** 2 + q_y.view(1, -1, 1) ** 2 + q_z.view(1, 1, -1) ** 2)
