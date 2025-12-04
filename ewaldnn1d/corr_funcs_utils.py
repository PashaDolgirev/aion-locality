import torch
import math

# --- Analytical routines --- 
def C_mm(m_vals: torch.Tensor, r_vals: torch.Tensor, N: int) -> torch.Tensor:
    """
    Compute C_mm(r) - diagonal term - for a list of m-values and r-values.
    Handles the singular mode m = N-1 carefully.
    
    Returns: (M, R) tensor
    """
    dtype = m_vals.dtype
    device = m_vals.device

    m = m_vals.view(-1, 1)              # (M,1)
    r = torch.abs(r_vals).view(1, -1)   # (1,R)

    pi = torch.tensor(math.pi, dtype=dtype, device=device)
    theta = pi * m / (N - 1)            # (M,1)
    # First term: (1/2) cos(theta*r)
    term1 = 0.5 * torch.cos(theta * r)

    # denominator sin(theta)
    den = torch.sin(theta)              # (M,1)
    # numerator sin(theta * (N-r))
    num = torch.sin(theta * (N - r))    # (M,R)
    # (-1)^m 
    sign = (m_vals.to(torch.long) % 2)*(-2) + 1  
    # maps even -> 1, odd -> -1
    sign = sign.view(-1,1).to(dtype)
    # general closed-form (will have 0/0 for m=N-1)
    term2 = 0.5 * sign * (num / den) / (N - r)

    # m = N-1  (exact expression = (-1)^r)
    mask_sing = (m_vals == (N-1))
    if mask_sing.any():
        idx = torch.nonzero(mask_sing, as_tuple=False).view(-1)
        C_sing = torch.cos(pi * r)          # (-1)^r
        term2[idx, :] = C_sing - term1[idx, :]   # enforce full C = term1+term2 ⇒ C=cos(pi*r)

    C = term1 + term2
    return C


def C_m1m2_sym(m_vals: torch.Tensor, r_vals: torch.Tensor, N: int) -> torch.Tensor:
    """
    Compute the symmetrized correlator C_{m1,m2}(r) for all m1,m2 and r.

    Formula:
        C_{m1, m2}(r) =
            [cos(pi (m1 - m2)/2) / (2 (N - |r|))] *
            [ sin(pi (m1 - m2)(N - |r|)/(2 (N-1))) *
              cos(pi (m1 + m2)|r|/(2 (N-1))) ] /
            sin(pi (m1 - m2)/(2 (N-1)))
        +
            [cos(pi (m1 + m2)/2) / (2 (N - |r|))] *
            [ sin(pi (m1 + m2)(N - |r|)/(2 (N-1))) *
              cos(pi (m1 - m2)|r|/(2 (N-1))) ] /
            sin(pi (m1 + m2)/(2 (N-1)))

    Might require more care for singular cases where the denominators vanish.

    Returns:
        C: (M, M, R) tensor where C[m1_idx, m2_idx, r_idx] = C_{m1,m2}(r)
    """
    dtype = m_vals.dtype
    device = m_vals.device
    M = m_vals.numel()

    # Shapes: (M,1,1) and (1,M,1)
    m1 = m_vals.view(-1, 1, 1)    # (M,1,1)
    m2 = m_vals.view(1, -1, 1)    # (1,M,1)

    # |r| as (1,1,R)
    r_abs = torch.abs(r_vals).view(1, 1, -1).to(device)  # (1,1,R)

    pi = torch.tensor(math.pi, dtype=dtype, device=device)
    # Convenient combos
    d = m1 - m2  # (M,M,1)
    s = m1 + m2  # (M,M,1)
    # N - |r| as (1,1,R)
    L = N - r_abs
    # Common denominators: alpha_d = pi d / [2 (N-1)], alpha_s = pi s / [2 (N-1)]
    alpha_d = pi * d / (2 * (N - 1))  # (M,M,1)
    alpha_s = pi * s / (2 * (N - 1))  # (M,M,1)

    # Denominators
    den_d = torch.sin(alpha_d)       # (M,M,1)
    den_s = torch.sin(alpha_s)       # (M,M,1)

    # Small epsilon to avoid division by exact 0
    eps = 1e-12
    den_d_safe = den_d.clone()
    den_s_safe = den_s.clone()
    den_d_safe[den_d_safe.abs() < eps] = eps
    den_s_safe[den_s_safe.abs() < eps] = eps

    # Numerators for the two terms
    num_d = torch.sin(pi * d * L / (2 * (N - 1))) * \
            torch.cos(pi * s * r_abs / (2 * (N - 1)))  # (M,M,R)

    num_s = torch.sin(pi * s * L / (2 * (N - 1))) * \
            torch.cos(pi * d * r_abs / (2 * (N - 1)))  # (M,M,R)

    # Prefactors cos(pi (m1±m2)/2) / [2 (N - |r|)]
    pref_d = torch.cos(pi * d / 2.0) / (2.0 * L)   # (M,M,R)
    pref_s = torch.cos(pi * s / 2.0) / (2.0 * L)   # (M,M,R)

    termA = pref_d * (num_d / den_d_safe)          # (M,M,R)
    termB = pref_s * (num_s / den_s_safe)          # (M,M,R)

    C = termA + termB  # (M,M,R), general off-diagonal expression

    # --- Diagonal correction: m1 == m2 ---
    # For diagonal we use the exact C_mm(r), including the m = N-1 singular mode
    C_diag = C_mm(m_vals, r_vals, N)   # (M,R)

    idx = torch.arange(M, device=device)
    C[idx, idx, :] = C_diag            # overwrite diagonal slices

    return C


def C_m1m2_sym_def(m_vals: torch.Tensor, r_vals: torch.Tensor, N: int) -> torch.Tensor:
    """
    Compute the symmetrized correlator C_{m1,m2}(r) from the *definition*:

        x_j = (j)/(N-1), j = 0,...,N-1
        phi_m(j) = cos(pi m x_j)

        tilde C_{m1,m2}(r) = 1/(N-|r|) sum_{i=0}^{N-|r|-1}
                             phi_{m1}(i) phi_{m2}(i+|r|)
        C_{m1,m2}(r) = 0.5[ tilde C_{m1,m2}(r) + tilde C_{m2,m1}(r) ]

    Args:
        m_vals: (M,) tensor of modes m >= 1
        r_vals: (R,) tensor of integer r (can be positive or negative)
        N:      number of grid points

    Returns:
        C: (M, M, R) tensor with C[m1_idx, m2_idx, r_idx] = C_{m1,m2}(r)
    """
    dtype = m_vals.dtype
    device = m_vals.device

    m_vals = m_vals.to(dtype=dtype, device=device)
    r_vals = r_vals.to(device=device)

    M = m_vals.numel()
    Rlen = r_vals.numel()

    # grid x_j = j/(N-1), j = 0,...,N-1 (this matches x_j = (j-1)/(N-1) with a shift)
    j = torch.arange(0, N, device=device, dtype=dtype)       # (N,)
    x = j / (N - 1)                                          # (N,)

    # phi[m, j] = cos(pi m x_j)
    pi = torch.tensor(math.pi, dtype=dtype, device=device)
    phi = torch.cos(pi * m_vals.view(M, 1) * x.view(1, N))   # (M, N)

    # Output tensor
    C = torch.empty((M, M, Rlen), dtype=dtype, device=device)

    for idx, r in enumerate(r_vals):
        k = abs(int(r.item()))
        L = N - k  # number of terms in sum

        # segments phi(:, i) and phi(:, i+k)
        A = phi[:, :L]          # (M, L)
        B = phi[:, k:k+L]       # (M, L)

        # tilde_C[m1, m2](r) = 1/L sum_i A[m1,i] * B[m2,i]
        # => (M,L) @ (M,L)^T with appropriate transpose:
        # we want A * B for (m1,m2) so:
        tilde_C = (A @ B.t()) / L    # (M, M)

        # symmetrize: C = 0.5(tilde_C + tilde_C^T)
        C_sym = 0.5 * (tilde_C + tilde_C.T)  # (M, M)

        C[:, :, idx] = C_sym

    return C


def second_moment_analytical(m_vals: torch.Tensor, R: int, N: int, std_harm: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix of second moments M(r, r') of the density–density
    correlation function for r, r' in [-R, ..., R].

    Args:
        m_vals   : (M,) harmonic indices
        R        : maximum |r|
        N        : number of grid points
        std_harm : (M,) std devs of harmonic amplitudes (gamma_m)

    Returns:
        M : (2R+1, 2R+1) tensor, with M[r_idx, r'_idx]
            corresponding to ⟨C(r) C(r')⟩.
    """

    gamma2 = std_harm**2      # (M,)
    r_vals = torch.arange(-R, R+1, device=m_vals.device)   # (2R+1,)
    
    C = (gamma2.unsqueeze(1) * C_mm(m_vals, r_vals, N)).sum(dim=0) # (2R+1,)
    M1 = C.view(-1, 1) * C.view(1, -1)   # (2R+1, 2R+1)

    C_m1m2 = C_m1m2_sym_def(m_vals, r_vals, N) #C_m1m2_sym(m, r_vals, N_grid) # (M, M, 2R+1)
    w = gamma2.view(m_vals.numel(), 1, 1, 1) * gamma2.view(1, m_vals.numel(), 1, 1)   # (M, M, 1, 1)
    C_r  = C_m1m2.unsqueeze(-1)   # (M, M, 2R+1, 1)
    C_rp = C_m1m2.unsqueeze(-2)   # (M, M, 1, 2R+1)

    prod = w * C_r * C_rp
    M2 = 2.0 * prod.sum(dim=(0, 1))          # (2R+1, 2R+1)

    return M1 + M2



# --- Numerical routines ---
def dens_dens_corr_func(rho_batch: torch.Tensor, R: int) -> torch.Tensor:
    """
    Compute numerically the correlation function C(r) for each sample in the batch

    Args:
        rho_batch: tensor of shape (B, N)
        R: maximum displacement (integer)

    Returns:
        C: tensor of shape (B, 2R+1)
           C[b, k] = c_b(r) for r = -R + k, k = 0,...,2R
    """
    B, N = rho_batch.shape
    Cs = []

    for r in range(-R, R + 1):
        if r == 0:
            prod = rho_batch * rho_batch / N                       # (B, N)
        elif r > 0:
            # i = 0..N-1-r, i+r = r..N-1
            prod = rho_batch[:, :N - r] * rho_batch[:, r:] / (N - r)    # (B, N-r)
        else:  # r < 0
            k = -r
            # i = k..N-1, i+r = i-k = 0..N-1-k
            prod = rho_batch[:, k:] * rho_batch[:, :N - k] / (N - k)    # (B, N-k)

        C_r = prod.sum(dim=1)                               # (B,)
        Cs.append(C_r)
        
    return torch.stack(Cs, dim=1)


def dens_dens_corr_func_sym(rho_batch: torch.Tensor, R: int) -> torch.Tensor:
    B, N = rho_batch.shape
    Cs = []

    for r in range(0, R + 1):
        if r == 0:
            prod = 0.5 * rho_batch * rho_batch / N                      # (B, N)
        else:
            prod = 0.5 * (rho_batch[:, :N - r] * rho_batch[:, r:] + rho_batch[:, r:] * rho_batch[:, :N - r]) / (N - r)

        C_r = prod.sum(dim=1)                              # (B,)
        Cs.append(C_r)
        
    return torch.stack(Cs, dim=1)


def make_symmetric(K_vec: torch.Tensor) -> torch.Tensor:
    """
    K_vec: tensor of shape (R+1,) or (R+1, 1) with entries K(0..R)
    Returns: tensor of shape (2R+1,) with K(-R..R), assuming K(-r) = K(r).
    """
    # Make sure it's a 1D tensor
    if K_vec.ndim > 1:
        K_vec = K_vec.squeeze()

    R = K_vec.shape[0] - 1

    # K(1..R) reversed → negative side
    K_left  = torch.flip(K_vec[1:], dims=[0])
    # K(0..R) → non-negative side
    K_right = K_vec

    K_full = torch.cat([K_left, K_right], dim=0)  # (2R+1,)
    return K_full
