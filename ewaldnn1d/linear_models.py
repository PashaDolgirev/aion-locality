import torch
import torch.nn as nn
from .linear_kernels import (
    LearnableKernelConv1d,
    GaussianMixtureKernelConv1d,
    LearnableRSNonLocalKernelDCT,
    ExpMixtureRSNonLocalKernelDCT,
    LearnableMSNonLocalKernelDCT,
)

    
class KernelOnlyEnergyNN(nn.Module):
    """
    E_tot = (1 / 2N) * sum_{i,j} rho_i rho_j K_{i-j}
          = (1 / 2N) * sum_i rho_i [K * rho]_i

    The kernel K is assumed local and learnable via convolution.
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
        mean_feat, std_feat: tensors broadcastable to features shape (1, N_grid, N_feat)
        E_mean, E_std: scalars or shape (1,)
        """
        super().__init__()
        self.R = R
        self.kernel_conv = LearnableKernelConv1d(R, even_kernel=True, pad_mode=pad_mode)

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
        features: (B, N_grid, N_feat) - only the first feature (density) is used
        Returns: total_energy_norm: (B,)
        """
        rho_norm = features[..., 0]  # (B, N_grid)

        if self.mean_feat is None or self.std_feat is None:
            raise RuntimeError("Normalization stats (mean_feat, std_feat) are not set.")

        # broadcast over batch automatically:
        # self.std_feat[..., 0] has shape (1, N_grid) or (1, N_grid, 1), both broadcastable
        rho = rho_norm * self.std_feat[..., 0] + self.mean_feat[..., 0]
        B, N_grid = rho.shape

        phi = self.kernel_conv(rho)  # (B, N_grid)

        local_energies = 0.5 * rho * phi
        total_energy = local_energies.sum(dim=1) / N_grid  # (B,)

        if self.E_mean is None or self.E_std is None:
            raise RuntimeError("Energy normalization stats (E_mean, E_std) are not set.")

        total_energy_norm = (total_energy - self.E_mean) / self.E_std
        return total_energy_norm


class GaussianMixtureEnergyNN(nn.Module):
    """
    E_tot = (1 / 2N) * sum_{i,j} rho_i rho_j K_{i-j}
          = (1 / 2N) * sum_i rho_i [K * rho]_i

    K is parameterized as a sum of Gaussians in r.
    """

    def __init__(
        self,
        R: int = 20,
        n_components: int = 3,
        pad_mode: str = "zero",
        mean_feat: torch.Tensor = None,
        std_feat: torch.Tensor = None,
        E_mean: torch.Tensor = None,
        E_std: torch.Tensor = None,
    ):
        super().__init__()
        self.R = R
        self.kernel_conv = GaussianMixtureKernelConv1d(
            R=R,
            n_components=n_components,
            pad_mode=pad_mode,
        )

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
        rho_norm = features[..., 0]  # (B, N_grid)

        if self.mean_feat is None or self.std_feat is None:
            raise RuntimeError("Normalization stats (mean_feat, std_feat) are not set.")

        rho = rho_norm * self.std_feat[..., 0] + self.mean_feat[..., 0]
        B, N_grid = rho.shape

        phi = self.kernel_conv(rho)  # (B, N_grid)

        local_energies = 0.5 * rho * phi
        total_energy = local_energies.sum(dim=1) / N_grid

        if self.E_mean is None or self.E_std is None:
            raise RuntimeError("Energy normalization stats (E_mean, E_std) are not set.")

        total_energy_norm = (total_energy - self.E_mean) / self.E_std
        return total_energy_norm


class DCTKernelEnergyNN(nn.Module):
    """
    E_tot = (1 / 2N) sum_i rho_i [K * rho]_i
    where K is represented via:
        (i)  its DCT-I eigenvalues λ_m (momentum-space param),
        (ii) a real-space kernel K_r but applied via DCT-I,
        (iii) a parametric mixture in real/DCT space.
    """

    def __init__(
        self,
        N: int,
        learning_mode: str = "dct_rs_blind",
        zero_r_flag: bool = False,
        n_components: int = 3,
        range_ms: int = 50,
        mean_feat: torch.Tensor = None,
        std_feat: torch.Tensor = None,
        E_mean: torch.Tensor = None,
        E_std: torch.Tensor = None,
    ):
        super().__init__()
        self.learning_mode = learning_mode

        if learning_mode == "dct_rs_blind":
            self.nonlocal_kernel = LearnableRSNonLocalKernelDCT(
                N=N, zero_r_flag=zero_r_flag
            )
        elif learning_mode == "dct_exp_rs_mixture":
            self.nonlocal_kernel = ExpMixtureRSNonLocalKernelDCT(
                N=N, zero_r_flag=zero_r_flag, n_components=n_components
            )
        elif learning_mode == "dct_ms_blind":
            self.nonlocal_kernel = LearnableMSNonLocalKernelDCT(range_ms=range_ms)
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
        rho_norm = features[..., 0]  # (B, N_grid)

        if self.mean_feat is None or self.std_feat is None:
            raise RuntimeError("Normalization stats (mean_feat, std_feat) are not set.")

        rho = rho_norm * self.std_feat[..., 0] + self.mean_feat[..., 0]
        B, N_grid = rho.shape

        phi = self.nonlocal_kernel(rho)  # (B, N_grid)

        local_energies = 0.5 * rho * phi
        total_energy = local_energies.sum(dim=1) / N_grid

        if self.E_mean is None or self.E_std is None:
            raise RuntimeError("Energy normalization stats (E_mean, E_std) are not set.")

        total_energy_norm = (total_energy - self.E_mean) / self.E_std
        return total_energy_norm


class HybridKernelEnergyNN(nn.Module):
    """
    Hybrid local + nonlocal kernel energy model
    E_tot = E_local + E_nonlocal
    """

    def __init__(
        self,
        N: int,
        R: int = 5,
        n_exp_components: int = 1,
        mean_feat: torch.Tensor = None,
        std_feat: torch.Tensor = None,
        E_mean: torch.Tensor = None,
        E_std: torch.Tensor = None,
    ):
        super().__init__()
        self.R = R
        self.n_exp_components = n_exp_components

        self.local_kernel = LearnableKernelConv1d(
            R=R, even_kernel=True, pad_mode="reflect"
        )
        self.nonlocal_kernel = ExpMixtureRSNonLocalKernelDCT(
            N=N, zero_r_flag=False, n_components=n_exp_components
        )

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
        rho_norm = features[..., 0]  # (B, N_grid)

        if self.mean_feat is None or self.std_feat is None:
            raise RuntimeError("Normalization stats (mean_feat, std_feat) are not set.")

        rho = rho_norm * self.std_feat[..., 0] + self.mean_feat[..., 0]
        B, N_grid = rho.shape

        phi_local = self.local_kernel(rho)      # (B, N_grid)
        phi_nonlocal = self.nonlocal_kernel(rho)  # (B, N_grid)

        E_local = 0.5 * (rho * phi_local).sum(dim=1) / N_grid
        E_nonlocal = 0.5 * (rho * phi_nonlocal).sum(dim=1) / N_grid
        total_energy = E_local + E_nonlocal

        if self.E_mean is None or self.E_std is None:
            raise RuntimeError("Energy normalization stats (E_mean, E_std) are not set.")

        total_energy_norm = (total_energy - self.E_mean) / self.E_std
        return total_energy_norm