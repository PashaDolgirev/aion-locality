import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_kernels import (
    LearnableRSKernelConv2d,
    LearnableRSNonLocalKernelDCT,
    ExpMixtureRSNonLocalKernelDCT,
    LearnableMSNonLocalKernelDCT,
    ScreenedCoulombNonLocalKernelDCT,
)

class RSKernelOnlyEnergyNN(nn.Module):
    """
    E_tot = (1 / 2N_x N_y) * sum_{i1,j1} sum_{i2,j2} rho_{i1,j1} rho_{i2,j2} K_{i1-i2,j1-j2}
          = (1 / 2N_x N_y) * sum_{i,j} rho_{i,j} [K * rho]_{i,j}

    The kernel K is assumed local and learnable via 2D convolution.
    """

    def __init__(
        self,
        R: int = 5,
        pad_mode: str = "reflect",
        mean_feat: torch.Tensor = None,
        std_feat: torch.Tensor = None,
        E_mean: torch.Tensor = None,
        E_std: torch.Tensor = None,
    ):
        """
        mean_feat, std_feat: tensors broadcastable to features shape (1, N_x, N_y, N_feat)
        E_mean, E_std: scalars or shape (1,)
        """
        super().__init__()
        self.R = R
        self.kernel_conv = LearnableRSKernelConv2d(R, even_kernel=True, pad_mode=pad_mode)

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
        features: (B, N_x, N_y, N_feat) - only the first feature (density) is used
        Returns: total_energy_norm: (B,)
        """
        rho_norm = features[..., 0]  # (B, N_x, N_y)

        if self.mean_feat is None or self.std_feat is None:
            raise RuntimeError("Normalization stats (mean_feat, std_feat) are not set.")

        # broadcast over batch automatically:
        # self.std_feat[..., 0] has shape (1, N_x, N_y) or (1, N_x, N_y, 1), both broadcastable
        rho = rho_norm * self.std_feat[..., 0] + self.mean_feat[..., 0]
        B, N_x, N_y = rho.shape

        phi = self.kernel_conv(rho)  # (B, N_x, N_y)
        local_energies = 0.5 * rho * phi
        total_energy = local_energies.sum(dim=(1, 2)) / (N_x * N_y)  # (B,)

        if self.E_mean is None or self.E_std is None:
            raise RuntimeError("Energy normalization stats (E_mean, E_std) are not set.")

        total_energy_norm = (total_energy - self.E_mean) / self.E_std
        return total_energy_norm
    

class DCTKernelEnergyNN(nn.Module):
    """
    E_tot = (1 / 2N_x N_y) * sum_{i1,j1} sum_{i2,j2} rho_{i1,j1} rho_{i2,j2} K_{i1-i2,j1-j2}
          = (1 / 2N_x N_y) * sum_{i,j} rho_{i,j} [K * rho]_{i,j},
    where K is represented via:
        (i)  its eigenvalues λ_mn (blind learning in DCT space),
        (ii) a real-space kernel K_r but applied via DCT routines (blind real space learning),
        (iii) a parametric mixture of real space exponentials,
        (iv) screened Coulomb, parametrized in DCT space.

    learning_mode options: "dct_rs_blind", "dct_ms_blind", "dct_exp_rs_mixture", "dct_coulomb"
    """

    def __init__(
        self,
        N_x: int,
        N_y: int,
        learning_mode: str = "dct_rs_blind",
        zero_r_flag: bool = False,
        n_components: int = 3,
        range_ms: float = 500.0,
        range_rs: float = 500.0,
        mean_feat: torch.Tensor = None,
        std_feat: torch.Tensor = None,
        E_mean: torch.Tensor = None,
        E_std: torch.Tensor = None,
    ):
        super().__init__()
        self.learning_mode = learning_mode

        if learning_mode == "dct_rs_blind":
            self.nonlocal_kernel = LearnableRSNonLocalKernelDCT(
                N_x=N_x, N_y=N_y, zero_r_flag=zero_r_flag, R=range_rs
                )
            self.range_rs = range_rs
        elif learning_mode == "dct_exp_rs_mixture":
            self.nonlocal_kernel = ExpMixtureRSNonLocalKernelDCT(
                N_x=N_x, N_y=N_y, zero_r_flag=zero_r_flag, n_components=n_components
                )   
        elif learning_mode == "dct_ms_blind":
            self.nonlocal_kernel = LearnableMSNonLocalKernelDCT(
                N_x=N_x, N_y=N_y, range_ms=range_ms
                )
            self.range_ms = range_ms
        elif learning_mode == "dct_coulomb":
            self.nonlocal_kernel = ScreenedCoulombNonLocalKernelDCT(
                N_x=N_x, N_y=N_y
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
        rho_norm = features[..., 0]  # (B, N_x, N_y)

        if self.mean_feat is None or self.std_feat is None:
            raise RuntimeError("Normalization stats (mean_feat, std_feat) are not set.")

        rho = rho_norm * self.std_feat[..., 0] + self.mean_feat[..., 0]
        B, N_x, N_y = rho.shape

        phi = self.nonlocal_kernel(rho)  # (B, N_x, N_y)

        local_energies = 0.5 * rho * phi
        total_energy = local_energies.sum(dim=(1, 2)) / (N_x * N_y)

        if self.E_mean is None or self.E_std is None:
            raise RuntimeError("Energy normalization stats (E_mean, E_std) are not set.")

        total_energy_norm = (total_energy - self.E_mean) / self.E_std
        return total_energy_norm