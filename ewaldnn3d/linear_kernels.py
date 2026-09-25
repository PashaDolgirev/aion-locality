import torch
import torch.nn as nn
import torch.nn.functional as F

from .fft_utils import (
    kernel_eigenvals_fft,
    conv_fft,
    displacement_grid,
    q_grid_fft,
)

from .energies_utils import (
    Lam_K_Coulomb,
    Lam_dK_Coulomb,
)


class LearnableRSKernelConv3d(nn.Module):
    """
    Learnable 3D convolution kernel K_{r_x, r_y, r_z} with range R in each dimension
    Produces phi = [K * rho] (linear convolution with padding)
    RS, real space kernel
    """
    def __init__(self, R=5, even_kernel=True, pad_mode="zero"):
        super().__init__()
        self.R = R
        self.pad_mode = pad_mode
        self.even_kernel = even_kernel

        if even_kernel:
            # K on the octant r_x, r_y, r_z in [0,R] is learned, full kernel is symmetric
            self.kernel_octant = nn.Parameter(torch.randn(R+1, R+1, R+1) * 0.01)
        else:
            # fully unconstrained kernel of size (2R+1, 2R+1, 2R+1)
            self.kernel = nn.Parameter(torch.randn(2*R+1, 2*R+1, 2*R+1) * 0.01)

    def build_kernel(self):
        """
        Returns kernel of shape (1,1,2R+1,2R+1,2R+1) as required by conv3d
        """
        if self.even_kernel:
            full = self.kernel_octant
            for dim in range(3):
                # mirror positive part: [K_R..K_1, K_0, K_1..K_R] along dim
                pos = full.narrow(dim, 1, self.R)               # r = 1..R
                full = torch.cat([pos.flip(dim), full], dim=dim)  # (2R+1) along dim
        else:
            full = self.kernel

        return full.view(1, 1, 2*self.R+1, 2*self.R+1, 2*self.R+1)

    def forward(self, rho):
        """
        rho: (B, N_x, N_y, N_z)
        Returns: phi: (B, N_x, N_y, N_z)
        """
        B, N_x, N_y, N_z = rho.shape
        kernel = self.build_kernel().to(dtype=rho.dtype, device=rho.device)
        R = self.R

        if self.pad_mode == "zero":
            x = F.pad(rho.unsqueeze(1), (R,) * 6, mode='constant', value=0.0)
        elif self.pad_mode == "reflect":
            x = F.pad(rho.unsqueeze(1), (R,) * 6, mode='reflect')
        else:
            raise ValueError("pad_mode must be zero or reflect")

        phi = F.conv3d(x, kernel, padding=0).squeeze(1)
        return phi


class LearnableRSNonLocalKernelFFT(nn.Module):
    """
    Learnable 3D nonlocal kernel K_{r_x, r_y, r_z} with range R
    Produces phi = [K * rho] via zero-padded FFT routines
    open BCs (zero padding)
    RS = real space
    """
    def __init__(self, N_x, N_y, N_z, zero_r_flag=True, R=1024):
        super().__init__()
        self.zero_r_flag = zero_r_flag
        self.N_x = N_x
        self.N_y = N_y
        self.N_z = N_z
        self.R = R
        self.rs_kernel = nn.Parameter(torch.randn(N_x, N_y, N_z) * 0.01)  # octant kernel K_r, r=0..N-1
        if zero_r_flag:
            with torch.no_grad():
                self.rs_kernel[0, 0, 0] = 0.0    # enforce K_{0,0,0} = 0

        # create a mask to enforce the range R
        rx = torch.arange(N_x).view(-1, 1, 1)
        ry = torch.arange(N_y).view(1, -1, 1)
        rz = torch.arange(N_z).view(1, 1, -1)
        rs_mask = (rx**2 + ry**2 + rz**2) <= R**2    # (N_x, N_y, N_z)
        self.register_buffer("rs_mask", rs_mask.float())

        with torch.no_grad():
            self.rs_kernel *= self.rs_mask

    def forward(self, rho):
        """
        rho: (B, N_x, N_y, N_z)
        Returns: phi: (B, N_x, N_y, N_z)
        """
        kernel = (self.rs_kernel * self.rs_mask).to(device=rho.device, dtype=rho.dtype)
        if self.zero_r_flag:
            kernel = kernel.clone()
            kernel[0, 0, 0] = 0.0

        lam_K = kernel_eigenvals_fft(kernel).to(device=rho.device, dtype=rho.dtype)  # (2N_x, 2N_y, N_z+1)
        return conv_fft(rho, lam_K)


class LearnableMSNonLocalKernelFFT(nn.Module):
    """
    Learnable nonlocal kernel parameterized directly in momentum space

    We learn real eigenvalues λ(q) on the rfftn grid of the padded box.
    λ(0) is set to 0 (no uniform component), and λ(q) = 0 for |q| > q_range
    (q in lattice units, see q_grid_fft).
    """
    def __init__(self, N_x, N_y, N_z, q_range: float = 100.0):
        super().__init__()
        self.N_x = N_x
        self.N_y = N_y
        self.N_z = N_z
        self.q_range = q_range
        self.ms_kernel = nn.Parameter(torch.randn(2 * N_x, 2 * N_y, N_z + 1) * 0.01)

        q_vals = q_grid_fft(N_x, N_y, N_z)
        ms_mask = q_vals <= q_range
        ms_mask[0, 0, 0] = False    # enforce λ(0) = 0
        self.register_buffer("ms_mask", ms_mask.float())

        with torch.no_grad():
            self.ms_kernel *= self.ms_mask

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: (B, N_x, N_y, N_z)
        Returns: phi = (K * rho): (B, N_x, N_y, N_z)
        """
        lam_K = (self.ms_kernel * self.ms_mask).to(device=rho.device, dtype=rho.dtype)
        return conv_fft(rho, lam_K)


class ExpMixtureRSNonLocalKernelFFT(nn.Module):
    """
    Learnable 3D kernel K_r represented as a sum of exponentials:
        K(r) = sum_{n=1}^M A_n * exp(-r / sigma_n)
    """

    def __init__(self, N_x, N_y, N_z, zero_r_flag=False, n_components=3):
        super().__init__()
        self.zero_r_flag = zero_r_flag
        self.n_components = n_components
        self.N_x = N_x
        self.N_y = N_y
        self.N_z = N_z

        r_vals = displacement_grid(N_x, N_y, N_z)    # (N_x, N_y, N_z)
        self.register_buffer("r_vals", r_vals)

        # ---- amplitudes A_n (sorted at init) ----
        # sample, then sort descending so A_1 >= A_2 >= ... >= A_M
        amps = 0.01 * torch.randn(n_components)
        amps, _ = torch.sort(amps, descending=True)
        self.amplitudes = nn.Parameter(amps)

        # Log-sigmas so sigma_n = softplus(log_sigma_n) > 0
        init_sigmas = 30.0 + 20.0 * torch.arange(n_components)
        log_sigmas = torch.log(torch.expm1(init_sigmas))  # inverse softplus, so softplus(raw)=init_sigma
        self.log_sigmas = nn.Parameter(log_sigmas)

    def build_kernel(self):
        r = self.r_vals.unsqueeze(0)            # (1, N_x, N_y, N_z)
        sigmas = F.softplus(self.log_sigmas) + 1e-8   # (M,)
        exp_mixt = torch.exp(-r / sigmas.view(-1, 1, 1, 1))          # (M, N_x, N_y, N_z)
        return (self.amplitudes.view(-1, 1, 1, 1) * exp_mixt).sum(dim=0)  # (N_x, N_y, N_z)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: (B, N_x, N_y, N_z)
        Returns: phi = (K * rho): (B, N_x, N_y, N_z)
        """
        kernel = self.build_kernel().to(dtype=rho.dtype, device=rho.device)
        if self.zero_r_flag:
            kernel = kernel.clone()
            kernel[0, 0, 0] = 0.0

        lam_K = kernel_eigenvals_fft(kernel).to(device=rho.device, dtype=rho.dtype)
        return conv_fft(rho, lam_K)


class ScreenedCoulombRSNonLocalKernelFFT(nn.Module):
    """
    Screened Coulomb (Yukawa) kernel parameterized in REAL space:
        K(r) = amp * exp(-qs * r) / r,  K(0) = 0 (no self-interaction),
    with learnable amplitude and screening momentum qs, applied via
    zero-padded FFT routines.

    Unlike the momentum-space variant (ScreenedCoulombNonLocalKernelFFT),
    the real-space form is exactly 1/r at qs -> 0 - no Brillouin-zone
    ringing and no uniform-mode constant (see test_coulomb3d.py) - matching
    the bare-Coulomb (Hartree) convention for molecules.
    """
    def __init__(self, N_x, N_y, N_z):
        super().__init__()
        self.N_x = N_x
        self.N_y = N_y
        self.N_z = N_z

        r_vals = displacement_grid(N_x, N_y, N_z)  # (N_x, N_y, N_z)
        self.register_buffer("r_vals", r_vals)
        self.amp = nn.Parameter(torch.randn(1) * 0.01)
        self.raw_qs = nn.Parameter(torch.tensor(1.0))

    def build_kernel_unit(self):
        """Unit-amplitude kernel exp(-qs r) / r, K(0) = 0 (amp excluded)."""
        qs = F.softplus(self.raw_qs)
        return torch.exp(-qs * self.r_vals) / self.r_vals.clamp(min=1.0) * (self.r_vals > 0)

    def build_kernel(self):
        return self.amp * self.build_kernel_unit()

    def phi_and_feature(self, rho: torch.Tensor):
        """
        Returns (phi, phi_feat):
            phi      = amp * (K_unit * rho), the physical mediator field;
            phi_feat = (K_unit * rho) / max|lam_unit|, amp-independent.
        For this positive kernel max|lam_unit| = sum_r K_unit, so phi_feat is
        the kernel-weighted average of the density over the screening cloud.
        The normalizer depends on model parameters only, never on sample
        statistics, so the feature map stays strictly local.
        """
        lam_unit = kernel_eigenvals_fft(self.build_kernel_unit()).to(device=rho.device, dtype=rho.dtype)
        phi_unit = conv_fft(rho, lam_unit)
        return self.amp * phi_unit, phi_unit / lam_unit.abs().max()

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: (B, N_x, N_y, N_z)
        Returns: phi = (K * rho): (B, N_x, N_y, N_z)
        """
        phi, _ = self.phi_and_feature(rho)
        return phi


class ScreenedCoulombNonLocalKernelFFT(nn.Module):
    """
    Nonlocal kernel corresponding to screened Coulomb potential in 3D,
    parameterized in MOMENTUM space: lam_K(q) = amp * 4 pi / (q^2 + qs^2)
    with uniform mode and self-interaction removed (see Lam_K_Coulomb),
    via zero-padded FFT routines.
    """
    def __init__(self, N_x, N_y, N_z):
        super().__init__()
        self.N_x = N_x
        self.N_y = N_y
        self.N_z = N_z

        q_vals = q_grid_fft(N_x, N_y, N_z)  # (2N_x, 2N_y, N_z+1)
        self.register_buffer("q_vals", q_vals)
        self.amp = nn.Parameter(torch.randn(1) * 0.01)
        self.raw_qs = nn.Parameter(torch.tensor(1.0))

    def phi_and_feature(self, rho: torch.Tensor):
        """
        Same contract as ScreenedCoulombRSNonLocalKernelFFT.phi_and_feature:
        (phi, phi_feat) with phi_feat amp-independent, normalized by the
        largest eigenvalue magnitude of the unit kernel (strictly local).
        """
        qs = F.softplus(self.raw_qs)
        lam_unit = Lam_K_Coulomb(self.q_vals, qs=qs).to(device=rho.device, dtype=rho.dtype)
        phi_unit = conv_fft(rho, lam_unit)
        return self.amp * phi_unit, phi_unit / lam_unit.abs().max()

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: (B, N_x, N_y, N_z)
        Returns: phi = (K * rho): (B, N_x, N_y, N_z)
        """
        phi, _ = self.phi_and_feature(rho)
        return phi


class DeltaCoulombRSNonLocalKernelFFT(nn.Module):
    """
    Screened-minus-bare Coulomb (delta) kernel parameterized in REAL space:
        dK(r) = (exp(-qs * r) - 1) / r,  dK(0) = 0 (no self-interaction),
    with learnable screening momentum qs, applied via zero-padded FFT routines.

    This is the difference between the Yukawa kernel exp(-qs r)/r and the bare
    Coulomb 1/r, built analytically (expm1) so that the Hartree subtraction
    happens at the kernel level - the mediator field dphi = (dK * rho) yields
    the Hartree-subtracted pairwise energy directly, never as a difference of
    two separately computed large energies. Note dK has a long-range -1/r tail.

    There is no amplitude parameter: in the EwaldNN models the amplitude is
    carried by the local reweighting factor a(x_r).
    """
    def __init__(self, N_x, N_y, N_z):
        super().__init__()
        self.N_x = N_x
        self.N_y = N_y
        self.N_z = N_z

        r_vals = displacement_grid(N_x, N_y, N_z)  # (N_x, N_y, N_z)
        self.register_buffer("r_vals", r_vals)
        # init qs = 1 (order-unity screening in lattice units, healthy gradients);
        # the mediator term starts off via the amplitude a(x_r) ~ 0, not via qs -
        # a near-zero qs sits in a degenerate a*qs valley and traps the optimizer
        self.raw_qs = nn.Parameter(torch.log(torch.expm1(torch.tensor(1.0))))  # inverse softplus

    def build_kernel(self):
        """Delta kernel (exp(-qs r) - 1) / r, dK(0) = 0."""
        qs = F.softplus(self.raw_qs)
        return torch.expm1(-qs * self.r_vals) / self.r_vals.clamp(min=1.0) * (self.r_vals > 0)

    def build_kernel_screened(self):
        """Screened companion exp(-qs r) / r, K(0) = 0 (short-range, for the feature)."""
        qs = F.softplus(self.raw_qs)
        return torch.exp(-qs * self.r_vals) / self.r_vals.clamp(min=1.0) * (self.r_vals > 0)

    def phi_and_feature(self, rho: torch.Tensor):
        """
        Returns (dphi, phi_feat):
            dphi     = (dK * rho), the delta mediator field entering the energy;
            phi_feat = (K_screened * rho) / max|lam_screened|, the SCREENED
                       kernel-weighted density average over the screening cloud.
        The feature deliberately uses the short-range screened kernel, not dK:
        the -1/r tail of dK would make the feature map long-range. The
        normalizer depends on model parameters only, never on sample
        statistics, so the feature map stays strictly local.
        """
        lam_dK = kernel_eigenvals_fft(self.build_kernel()).to(device=rho.device, dtype=rho.dtype)
        lam_s = kernel_eigenvals_fft(self.build_kernel_screened()).to(device=rho.device, dtype=rho.dtype)
        return conv_fft(rho, lam_dK), conv_fft(rho, lam_s) / lam_s.abs().max()

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: (B, N_x, N_y, N_z)
        Returns: dphi = (dK * rho): (B, N_x, N_y, N_z)
        """
        dphi, _ = self.phi_and_feature(rho)
        return dphi


class DeltaCoulombNonLocalKernelFFT(nn.Module):
    """
    Screened-minus-bare Coulomb (delta) kernel parameterized in MOMENTUM space:
        lam_dK(q) = -4 pi qs^2 / (q^2 (q^2 + qs^2))  (see Lam_dK_Coulomb),
    with learnable screening momentum qs, via zero-padded FFT routines.
    Same contract and rationale as DeltaCoulombRSNonLocalKernelFFT.
    """
    def __init__(self, N_x, N_y, N_z):
        super().__init__()
        self.N_x = N_x
        self.N_y = N_y
        self.N_z = N_z

        q_vals = q_grid_fft(N_x, N_y, N_z)  # (2N_x, 2N_y, N_z+1)
        self.register_buffer("q_vals", q_vals)
        # init qs = 1 (see DeltaCoulombRSNonLocalKernelFFT)
        self.raw_qs = nn.Parameter(torch.log(torch.expm1(torch.tensor(1.0))))  # inverse softplus

    def phi_and_feature(self, rho: torch.Tensor):
        """
        Same contract as DeltaCoulombRSNonLocalKernelFFT.phi_and_feature:
        (dphi, phi_feat) with the feature built from the screened kernel
        4 pi / (q^2 + qs^2), normalized by its largest eigenvalue magnitude.
        """
        qs = F.softplus(self.raw_qs)
        lam_dK = Lam_dK_Coulomb(self.q_vals, qs=qs).to(device=rho.device, dtype=rho.dtype)
        lam_s = Lam_K_Coulomb(self.q_vals, qs=qs).to(device=rho.device, dtype=rho.dtype)
        return conv_fft(rho, lam_dK), conv_fft(rho, lam_s) / lam_s.abs().max()

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: (B, N_x, N_y, N_z)
        Returns: dphi = (dK * rho): (B, N_x, N_y, N_z)
        """
        dphi, _ = self.phi_and_feature(rho)
        return dphi
