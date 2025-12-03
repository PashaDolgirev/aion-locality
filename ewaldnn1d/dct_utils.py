import torch
import torch_dct as dct


def rho_to_cosine_coeffs(rho_batch: torch.Tensor) -> torch.Tensor:
    """
    rho_batch: (B, N) real tensor
        rho[j] sampled on j = 0,...,N-1.
    Returns:
        a_batch: (B, N) cosine coefficients a_m in
            rho_j = sum_m a_m cos(pi m j / (N-1)).
    """
    # DCT-I along the last dimension
    # torch_dct.dct1 works along the last axis by default
    X = dct.dct1(rho_batch)          # shape (B, N)
    B, N = X.shape

    a = X.clone()
    a[:, 0]      = 0.5 * X[:, 0]      / (N - 1)        # m = 0
    a[:, 1:N-1]  = X[:, 1:N-1] / (N - 1)   # 1 <= m <= N-2
    a[:, N-1]    = 0.5 * X[:, N-1]    / (N - 1)        # m = N-1

    return a


def cosine_coeffs_to_rho(a_batch: torch.Tensor) -> torch.Tensor:
    """
    Inverse of rho_to_cosine_coeffs, using inverse DCT-I.

    a_batch: (B, N) of a_m
    Returns:
        rho_batch: (B, N)
    """
    B, N = a_batch.shape

    # Map back to standard DCT-I coefficients X_m
    X = a_batch.clone()
    X[:, 0]      = a_batch[:, 0]      * (N - 1) * 2.0
    X[:, 1:N-1]  = a_batch[:, 1:N-1]  * (N - 1)
    X[:, N-1]    = a_batch[:, N-1]    * (N - 1) * 2.0

    # Inverse DCT-I along last dimension
    rho_rec = dct.idct1(X)

    return rho_rec


def kernel_eigenvals_dct(K: torch.Tensor) -> torch.Tensor:
    """
    Compute the convolution eigenvalues for kernel K using DCT-I.

    K: (N,) real tensor, kernel values at r = 0, 1, ..., N-1
       (assumed even: K_r = K_{-r})

    Returns:
        lam_K: (N,) real tensor, eigenvalues λ_m of the convolution operator
               with that kernel under even-reflection (Neumann BC).
    """
    K = K.squeeze()
    N = K.shape[-1]
    X = dct.dct1(K)                     # (N,)

    # λ_m = X_m + (-1)^m * K_{N-1}
    lam_K = X + K[-1] * ((-1.0) ** torch.arange(N, device=K.device, dtype=K.dtype))     # (N,)
    return lam_K


def kernel_from_eigenvals_dct(lam_K: torch.Tensor) -> torch.Tensor:
    """
    Invert kernel_eigenvals_dct:
    given eigenvalues lam_K (λ_m), reconstruct kernel K (K_r, r=0..N-1).

    lam_K: (N,)
    Returns:
        K: (N,) real tensor, kernel values at r = 0, 1, ..., N-1
    """
    
    device, dtype = lam_K.device, lam_K.dtype
    N = lam_K.shape[-1]

    # v_m = (-1)^m
    v = (-1.0) ** torch.arange(N, device=device, dtype=dtype)                # (N,)
    A = dct.idct1(lam_K)           # (N,)
    c = dct.idct1(v)               # (N,)

    # K_{N-1} = A_{N-1} / (1 + c_{N-1})   (do this per batch)
    denom = 1.0 + c[-1]
    K_nm1 = A[-1] / denom       

    # K = A - K_{N-1} * c
    K = A - K_nm1 * c  
    return K