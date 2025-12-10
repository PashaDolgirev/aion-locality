import torch
import torch_dct as dct


def rho_to_cosine_coeffs(rho_batch: torch.Tensor) -> torch.Tensor:
    """
    rho_batch: (B, N_x, N_y) real tensor
        rho[b, i, j] sampled on i = 0,...,N_x-1 and j = 0,...,N_y-1.
    Returns:
        a_batch: (B, N_x, N_y) cosine coefficients a_mn in
            rho_ij = sum_m sum_n a_mn cos(pi m i / (N_x-1)) cos(pi n j / (N_y-1)).
    """

    B, N_x, N_y = rho_batch.shape
    device = rho_batch.device
    dtype = rho_batch.dtype

    # 2D DCT-I over (i,j)
    X = dct.dct1(rho_batch)                   # DCT-I along j (last dim): (B, N_x, N_y)
    X = X.transpose(-1, -2).contiguous()      # (B, N_y, N_x)
    X = dct.dct1(X)                           # DCT-I along i (now last dim)
    X = X.transpose(-1, -2).contiguous()      # (B, N_x, N_y), DCT-I in both dims

    # 1D scaling factors for each dimension
    sx = torch.full((N_x,), 1.0 / (N_x - 1), device=device, dtype=dtype)
    sy = torch.full((N_y,), 1.0 / (N_y - 1), device=device, dtype=dtype)
    # endpoints get an extra 1/2 factor
    sx[0]      *= 0.5
    sx[-1]     *= 0.5
    sy[0]      *= 0.5
    sy[-1]     *= 0.5

    # broadcast over batch and other dimension
    # a[m,n] = X[m,n] * sx[m] * sy[n]
    a = X.clone()
    a = a * sx.view(1, N_x, 1) * sy.view(1, 1, N_y)

    return a


def cosine_coeffs_to_rho(a_batch: torch.Tensor) -> torch.Tensor:
    """
    Inverse of rho_to_cosine_coeffs, using inverse DCT-I.

    a_batch: (B, N_x, N_y) of a_mn
    Returns:
        rho_batch: (B, N_x, N_y)
    """

    B, N_x, N_y = a_batch.shape
    device = a_batch.device
    dtype = a_batch.dtype

    # 1D scaling factors for each dimension
    sx = torch.full((N_x,), 1.0 / (N_x - 1), device=device, dtype=dtype)
    sy = torch.full((N_y,), 1.0 / (N_y - 1), device=device, dtype=dtype)
    # endpoints get an extra 1/2 factor
    sx[0]      *= 0.5
    sx[-1]     *= 0.5
    sy[0]      *= 0.5
    sy[-1]     *= 0.5

    # Map back to standard DCT-I coefficients X_m
    X = a_batch.clone()
    X = X / (sx.view(1, N_x, 1) * sy.view(1, 1, N_y))

    # 2D Inverse DCT-I over (i,j)
    rho_rec = dct.idct1(X)
    rho_rec = rho_rec.transpose(-1, -2).contiguous()
    rho_rec = dct.idct1(rho_rec)
    rho_rec = rho_rec.transpose(-1, -2).contiguous()

    return rho_rec


def kernel_eigenvals_transform(K: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D convolution eigenvalues for kernel K using DCT-I, along the last dimension.
    Adopted from ewaldnn1d/dct_utils.py
    
    K: (..., N) real tensor, kernel values at r = 0,...,N-1
       (assumed even in that dimension: K_r = K_{-r}).

    Returns:
        lam_K: (..., N) real tensor, eigenvalues λ_m along the last dim
               for the convolution operator with Neumann BC.
    """

    N = K.shape[-1]
    X = dct.dct1(K)  # DCT-I along last dimension

    # (-1)^m with proper broadcasting
    m = torch.arange(N, device=K.device, dtype=K.dtype)
    phase = (-1.0) ** m
    phase = phase.view(*([1] * (K.dim() - 1)), N)  # (..., N)

    K_last = K[..., -1].unsqueeze(-1)  # (..., 1)

    # λ_m = X_m + (-1)^m * K_{N-1}
    lam_K = X + K_last * phase         # (..., N)

    return lam_K


def kernel_from_eigenvals_transform(lam_K: torch.Tensor) -> torch.Tensor:
    """
    Invert kernel_eigenvals_transform along the last dimension.
    Adopted from ewaldnn1d/dct_utils.py

    Given convolution eigenvalues lam_K (λ_m) along the last axis,
    reconstruct the 1D kernel K(r), r = 0,...,N-1, along that axis.

    Args:
        lam_K: (..., N) real tensor

    Returns:
        K: (..., N) real tensor, kernel values along the last dimension
    """

    lam_K = lam_K.contiguous() 
    device, dtype = lam_K.device, lam_K.dtype
    N = lam_K.shape[-1]

    # v_m = (-1)^m, shape (N,)
    m = torch.arange(N, device=device, dtype=dtype)
    v = (-1.0) ** m                      # (N,)

    # A = idct1(lam), c = idct1(v)
    # idct1 acts along the last dimension
    A = dct.idct1(lam_K)                 # (..., N)
    c = dct.idct1(v)                     # (N,)

    # K_{N-1} from the last component:
    # A_{N-1} = K_{N-1} + K_{N-1} c_{N-1} = K_{N-1} (1 + c_{N-1})
    denom = 1.0 + c[-1]                  # scalar
    K_last = A[..., -1] / denom          # (...,)

    # Broadcast c to (..., N)
    c_broadcast = c.view(*([1] * (lam_K.dim() - 1)), N)  # (..., N)
    K_last_exp = K_last.unsqueeze(-1)                    # (..., 1)

    # K = A - K_{N-1} * c
    K = A - K_last_exp * c_broadcast                    # (..., N)

    return K


def kernel_eigenvals_dct(K: torch.Tensor) -> torch.Tensor:
    """
    Compute the convolution eigenvalues for kernel K using DCT-I.

    K: (N_x, N_y) real tensor, kernel values at r_x = 0, 1, ..., N_x-1 and r_y = 0, 1, ..., N_y-1
       (assumed rotationally symmetric: K_{r_x, r_y} = K_{-r_x, -r_y} = K_{-r_x, r_y} = K_{r_x, -r_y})

    Returns:
        lam_K: (N_x, N_y) real tensor, eigenvalues λ_mn of the convolution operator
               with that kernel under even-reflection (Neumann BC).
    """

    lam_K = kernel_eigenvals_transform(K)
    lam_K = lam_K.transpose(-1, -2).contiguous()
    lam_K = kernel_eigenvals_transform(lam_K)
    lam_K = lam_K.transpose(-1, -2).contiguous()

    return lam_K


def kernel_from_eigenvals_dct(lam_K: torch.Tensor) -> torch.Tensor:
    """
    Invert kernel_eigenvals_dct:
    given eigenvalues lam_K (λ_mn), reconstruct kernel K_{r_x, r_y}.

    lam_K: (N_x, N_y)
    Returns:
        K: (N_x, N_y) real tensor, kernel values at r_x = 0, 1, ..., N_x-1 and r_y = 0, 1, ..., N_y-1
    """

    K = kernel_from_eigenvals_transform(lam_K)
    K = K.transpose(-1, -2).contiguous()
    K = kernel_from_eigenvals_transform(K)
    K = K.transpose(-1, -2).contiguous() 

    return K