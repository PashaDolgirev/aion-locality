import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_kernels import (
    LearnableRSKernelConv3d,
    LearnableRSNonLocalKernelFFT,
    ExpMixtureRSNonLocalKernelFFT,
    LearnableMSNonLocalKernelFFT,
    ScreenedCoulombNonLocalKernelFFT,
)

class RSKernelOnlyEnergyNN(nn.Module):
    """
    E_tot = (1 / 2 N_x N_y N_z) * sum_{r1, r2} rho_{r1} rho_{r2} K_{r1-r2}
          = (1 / 2 N_x N_y N_z) * sum_r rho_r [K * rho]_r

    The kernel K is assumed local and learnable via 3D convolution.
    """

    def __init__(
        self,
        R: int = 5,
        pad_mode: str = "zero",
        mean_feat: torch.Tensor = None,
        std_feat: torch.Tensor = None,
        E_mean: torch.Tensor = None,
        E_std: torch.Tensor = None,
    ):
        """
        mean_feat, std_feat: tensors broadcastable to features shape (1, N_x, N_y, N_z, N_feat)
        E_mean, E_std: scalars or shape (1,)
        """
        super().__init__()
        self.R = R
        self.kernel_conv = LearnableRSKernelConv3d(R, even_kernel=True, pad_mode=pad_mode)

        # register normalization stats as buffers so they move with .to(device)
        if mean_feat is not None:
            self.register_buffer("mean_feat", mean_feat)
        else:
            self.mean_feat = None

        if std_feat is not None:
            self.register_buffer("std_feat", std_feat)
        else:
            self.std_feat = None

        if E_mean is not None:
            self.register_buffer("E_mean", E_mean)
        else:
            self.E_mean = None

        if E_std is not None:
            self.register_buffer("E_std", E_std)
        else:
            self.E_std = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, N_x, N_y, N_z, N_feat) - only the first feature (density) is used
        Returns: total_energy_norm: (B,)
        """
        rho_norm = features[..., 0]  # (B, N_x, N_y, N_z)

        if self.mean_feat is None or self.std_feat is None:
            raise RuntimeError("Normalization stats (mean_feat, std_feat) are not set.")

        rho = rho_norm * self.std_feat[..., 0] + self.mean_feat[..., 0]
        B, N_x, N_y, N_z = rho.shape

        phi = self.kernel_conv(rho)  # (B, N_x, N_y, N_z)
        local_energies = 0.5 * rho * phi
        total_energy = local_energies.sum(dim=(1, 2, 3)) / (N_x * N_y * N_z)  # (B,)

        if self.E_mean is None or self.E_std is None:
            raise RuntimeError("Energy normalization stats (E_mean, E_std) are not set.")

        total_energy_norm = (total_energy - self.E_mean) / self.E_std
        return total_energy_norm


class FFTKernelEnergyNN(nn.Module):
    """
    E_tot = (1 / 2 N_x N_y N_z) * sum_{r1, r2} rho_{r1} rho_{r2} K_{r1-r2}
          = (1 / 2 N_x N_y N_z) * sum_r rho_r [K * rho]_r,
    where K is represented via:
        (i)  its eigenvalues λ(q) on the padded FFT grid (blind momentum-space learning),
        (ii) a real-space kernel K_r but applied via zero-padded FFT (blind real space learning),
        (iii) a parametric mixture of real space exponentials,
        (iv) screened Coulomb, parametrized in momentum space.

    learning_mode options: "fft_rs_blind", "fft_ms_blind", "fft_exp_rs_mixture", "fft_coulomb"
    """

    def __init__(
        self,
        N_x: int,
        N_y: int,
        N_z: int,
        learning_mode: str = "fft_rs_blind",
        zero_r_flag: bool = False,
        n_components: int = 3,
        q_range: float = 100.0,
        range_rs: float = 500.0,
        mean_feat: torch.Tensor = None,
        std_feat: torch.Tensor = None,
        E_mean: torch.Tensor = None,
        E_std: torch.Tensor = None,
    ):
        super().__init__()
        self.learning_mode = learning_mode

        if learning_mode == "fft_rs_blind":
            self.nonlocal_kernel = LearnableRSNonLocalKernelFFT(
                N_x=N_x, N_y=N_y, N_z=N_z, zero_r_flag=zero_r_flag, R=range_rs
                )
            self.range_rs = range_rs
        elif learning_mode == "fft_exp_rs_mixture":
            self.nonlocal_kernel = ExpMixtureRSNonLocalKernelFFT(
                N_x=N_x, N_y=N_y, N_z=N_z, zero_r_flag=zero_r_flag, n_components=n_components
                )
        elif learning_mode == "fft_ms_blind":
            self.nonlocal_kernel = LearnableMSNonLocalKernelFFT(
                N_x=N_x, N_y=N_y, N_z=N_z, q_range=q_range
                )
            self.q_range = q_range
        elif learning_mode == "fft_coulomb":
            self.nonlocal_kernel = ScreenedCoulombNonLocalKernelFFT(
                N_x=N_x, N_y=N_y, N_z=N_z
                )
        else:
            raise ValueError(f"Unknown learning_mode: {learning_mode}")

        if mean_feat is not None:
            self.register_buffer("mean_feat", mean_feat)
        else:
            self.mean_feat = None

        if std_feat is not None:
            self.register_buffer("std_feat", std_feat)
        else:
            self.std_feat = None

        if E_mean is not None:
            self.register_buffer("E_mean", E_mean)
        else:
            self.E_mean = None

        if E_std is not None:
            self.register_buffer("E_std", E_std)
        else:
            self.E_std = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        rho_norm = features[..., 0]  # (B, N_x, N_y, N_z)

        if self.mean_feat is None or self.std_feat is None:
            raise RuntimeError("Normalization stats (mean_feat, std_feat) are not set.")

        rho = rho_norm * self.std_feat[..., 0] + self.mean_feat[..., 0]
        B, N_x, N_y, N_z = rho.shape

        phi = self.nonlocal_kernel(rho)  # (B, N_x, N_y, N_z)

        local_energies = 0.5 * rho * phi
        total_energy = local_energies.sum(dim=(1, 2, 3)) / (N_x * N_y * N_z)

        if self.E_mean is None or self.E_std is None:
            raise RuntimeError("Energy normalization stats (E_mean, E_std) are not set.")

        total_energy_norm = (total_energy - self.E_mean) / self.E_std
        return total_energy_norm
