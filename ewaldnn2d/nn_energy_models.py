import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_kernels import ScreenedCoulombNonLocalKernelDCT


class LocalNN2d(nn.Module):
    """
    f_a = NN(x_ij) is a small feedforward neural network, a = 1,...,N_energy_terms,
    shape of output f_a(i,j) is (B, N_x, N_y, N_energy_terms);
    x_ij are local features on 2D grid, shape (B, N_x, N_y, N_feat).
    """
    def __init__(
        self,
        N_feat: int,
        n_hidden: int = 3,
        n_neurons: int = 16,
        N_energy_terms: int = 1,
    ):
        super().__init__()

        layers = []
        input_dim = N_feat

        for _ in range(n_hidden):
            layers.append(nn.Linear(input_dim, n_neurons))
            layers.append(nn.LayerNorm(n_neurons))
            layers.append(nn.GELU())
            input_dim = n_neurons

        layers.append(nn.Linear(input_dim, N_energy_terms))
        self.loc_network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, N_x, N_y, N_feat = features.shape
        x = features.reshape(B * N_x * N_y, N_feat)

        z = self.loc_network(x)
        f_a = 1.0 + torch.tanh(z)  # enforce [0, 2]

        return f_a.reshape(B, N_x, N_y, -1) # (B, N_x, N_y, N_energy_terms)
    
    
class LERN2d(nn.Module):
    """
    LERN = Local Energy Reiweighting Network
    E_tot = (1 / N_x N_y) * sum_{i,j} sum_a f_a(x_{i,j}) * E_a(i,j).

    E_a(i,j) are given energy contributions (e.g. kinetic energy, Hartree-Fock terms, etc.), 
    a = 1,...,N_energy_terms,
    E_a is of shape (B, N_x, N_y, N_energy_terms),
    passed along as last N_energy_terms features in the input features tensor.

    f_a(x_{i,j}) are local reweighting factors, which depend on local features x_{i,j} only;
    f_a(x_{i,j}) = NN_a(x_{i,j}) where NN_a is a small feedforward neural network.

    x_{i,j} are local features at grid point (i,j), its shape is (B, N_x, N_y, N_feat).
    """

    def __init__(
        self,
        N_x: int,
        N_y: int,
        N_energy_terms: int,
        N_feat: int,
        n_hidden: int = 3,
        n_neurons: int = 16,
        mean_feat: torch.Tensor = None,
        std_feat: torch.Tensor = None,
        E_mean: torch.Tensor = None,
        E_std: torch.Tensor = None,
    ):

        super().__init__()
        self.N_x = N_x
        self.N_y = N_y
        self.N_feat = N_feat
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        self.N_energy_terms = N_energy_terms

        self.local_nn = LocalNN2d(
            N_feat=N_feat,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            N_energy_terms=N_energy_terms,
        )

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
        features_orig = features[...,:self.N_feat] # (B, N_x, N_y, N_feat), normalized features
        energy_terms = features[...,self.N_feat:] # (B, N_x, N_y, N_energy_terms), physical (unnormalized) energy terms

        factors = self.local_nn(features_orig)  # (B, N_x, N_y, N_energy_terms)

        if self.E_mean is None or self.E_std is None:
            raise RuntimeError("Energy normalization stats (E_mean, E_std) are not set.")

        E_tot = (factors * energy_terms).mean(dim=(1,2))  # (B, N_energy_terms)
        E_tot = E_tot.sum(dim=-1)  # (B,) total physical energy per batch element
        E_tot_norm = (E_tot - self.E_mean) / self.E_std  # (B,) normalized total energy
        return E_tot_norm


class EwaldNN2d(nn.Module):
    """
    EwaldNN:
        E_tot = (1 / N_x N_y) * sum_{i,j} [ sum_a f_a(x_{i,j}, phi_{i,j}) * E_a(i,j)
                + (1/2) phi_{i,j} rho_{i,j} ],

    where phi = (K * rho) is a mediator field built from the density through a
    learnable translationally invariant kernel (screened Coulomb with learnable
    amplitude and screening momentum qs, via DCT routines). phi enters both as
    an adaptive feature appended to x_{i,j} and as an explicit pairwise energy
    term. The feature copy is the kernel-weight-normalized, amp-independent
    density average (see the mediator's phi_and_feature), scaled by the density
    std. All normalizers are model parameters or fixed dataset constants -
    never per-sample statistics, which would introduce a spurious nonlocal
    interaction. The energy term uses the raw physical field.

    Input follows the LERN convention: features packs normalized features in
    features[..., :N_feat] (feature 0 = normalized density) and unnormalized
    energy terms E_a(i,j) in features[..., N_feat:].
    """

    def __init__(
        self,
        N_x: int,
        N_y: int,
        N_energy_terms: int,
        N_feat: int,
        n_hidden: int = 3,
        n_neurons: int = 16,
        mediator: nn.Module = None,
        mean_feat: torch.Tensor = None,
        std_feat: torch.Tensor = None,
        E_mean: torch.Tensor = None,
        E_std: torch.Tensor = None,
    ):

        super().__init__()
        self.N_x = N_x
        self.N_y = N_y
        self.N_feat = N_feat
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        self.N_energy_terms = N_energy_terms

        # mediator: any module mapping rho (B, N_x, N_y) -> phi (B, N_x, N_y)
        if mediator is not None:
            self.mediator = mediator
        else:
            self.mediator = ScreenedCoulombNonLocalKernelDCT(N_x=N_x, N_y=N_y)

        # +1 input feature: the mediator field phi
        self.local_nn = LocalNN2d(
            N_feat=N_feat + 1,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            N_energy_terms=N_energy_terms,
        )

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
        features_orig = features[...,:self.N_feat] # (B, N_x, N_y, N_feat), normalized features
        energy_terms = features[...,self.N_feat:] # (B, N_x, N_y, N_energy_terms), physical (unnormalized) energy terms

        if self.mean_feat is None or self.std_feat is None:
            raise RuntimeError("Normalization stats (mean_feat, std_feat) are not set.")

        # feature 0 is the normalized density; undo the normalization
        rho = features_orig[..., 0] * self.std_feat[..., 0] + self.mean_feat[..., 0]  # (B, N_x, N_y)

        if hasattr(self.mediator, "phi_and_feature"):
            phi, phi_feat = self.mediator.phi_and_feature(rho)
        else:
            phi = self.mediator(rho)  # custom mediator without a feature convention
            phi_feat = phi
        phi_feat = phi_feat / self.std_feat[..., 0]  # scale like the density feature

        features_ext = torch.cat([features_orig, phi_feat.unsqueeze(-1)], dim=-1)
        factors = self.local_nn(features_ext)  # (B, N_x, N_y, N_energy_terms)

        if self.E_mean is None or self.E_std is None:
            raise RuntimeError("Energy normalization stats (E_mean, E_std) are not set.")

        E_loc = (factors * energy_terms).sum(dim=-1) + 0.5 * phi * rho  # (B, N_x, N_y)
        E_tot = E_loc.mean(dim=(1,2))  # (B,) total physical energy per batch element
        E_tot_norm = (E_tot - self.E_mean) / self.E_std  # (B,) normalized total energy
        return E_tot_norm