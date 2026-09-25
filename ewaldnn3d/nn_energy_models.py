import torch
import torch.nn as nn
import torch.nn.functional as F

from .fft_utils import conv_fft, q_grid_fft, displacement_grid, kernel_eigenvals_fft
from .energies_utils import Lam_K_Coulomb, Lam_dK_Coulomb
from .linear_kernels import DeltaCoulombNonLocalKernelFFT, DeltaCoulombRSNonLocalKernelFFT


def _yukawa_eigenvals(N_x, N_y, N_z, qs):
    """Eigenvalues of the real-space Yukawa kernel exp(-qs r)/r, K(0) = 0."""
    r_vals = displacement_grid(N_x, N_y, N_z)
    K = torch.exp(-qs * r_vals) / r_vals.clamp(min=1.0) * (r_vals > 0)
    return kernel_eigenvals_fft(K)


def _yukawa_delta_eigenvals(N_x, N_y, N_z, qs):
    """Eigenvalues of the real-space delta kernel (exp(-qs r) - 1)/r, dK(0) = 0
    (Yukawa minus bare Coulomb, built analytically via expm1)."""
    r_vals = displacement_grid(N_x, N_y, N_z)
    dK = torch.expm1(-qs * r_vals) / r_vals.clamp(min=1.0) * (r_vals > 0)
    return kernel_eigenvals_fft(dK)


class LocalNN3d(nn.Module):
    """
    f_a = NN(x_r) is a small feedforward neural network, a = 1,...,N_energy_terms,
    shape of output f_a(r) is (B, N_x, N_y, N_z, N_energy_terms);
    x_r are local features on 3D grid, shape (B, N_x, N_y, N_z, N_feat).

    If raw_output, returns the raw network outputs z instead of 1 + tanh(z)
    (used by the EwaldNN models to apply their own heads).
    """
    def __init__(
        self,
        N_feat: int,
        n_hidden: int = 3,
        n_neurons: int = 16,
        N_energy_terms: int = 1,
        raw_output: bool = False,
    ):
        super().__init__()
        self.raw_output = raw_output

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
        B, N_x, N_y, N_z, N_feat = features.shape
        x = features.reshape(B * N_x * N_y * N_z, N_feat)

        z = self.loc_network(x)
        if not self.raw_output:
            z = 1.0 + torch.tanh(z)  # enforce [0, 2]

        return z.reshape(B, N_x, N_y, N_z, -1) # (B, N_x, N_y, N_z, N_out)


class _NormalizedEnergyModel(nn.Module):
    """
    Shared normalization-buffer plumbing for the 3D energy models.
    """
    def __init__(self, mean_feat=None, std_feat=None, E_mean=None, E_std=None):
        super().__init__()
        # register normalization stats as buffers so they move with .to(device)
        for name, val in [("mean_feat", mean_feat), ("std_feat", std_feat),
                          ("E_mean", E_mean), ("E_std", E_std)]:
            if val is not None:
                self.register_buffer(name, val)
            else:
                setattr(self, name, None)

    def _check_E_stats(self):
        if self.E_mean is None or self.E_std is None:
            raise RuntimeError("Energy normalization stats (E_mean, E_std) are not set.")

    def _rho_from_features(self, features_norm: torch.Tensor) -> torch.Tensor:
        # feature 0 is the normalized density; undo the normalization
        if self.mean_feat is None or self.std_feat is None:
            raise RuntimeError("Normalization stats (mean_feat, std_feat) are not set.")
        return features_norm[..., 0] * self.std_feat[..., 0] + self.mean_feat[..., 0]


class LERN3d(_NormalizedEnergyModel):
    """
    LERN = Local Energy Reweighting Network
    E_tot = (1 / N_x N_y N_z) * sum_r sum_a f_a(x_r) * E_a(r).

    E_a(r) are given energy contributions (e.g. kinetic energy, Hartree-Fock terms, etc.),
    a = 1,...,N_energy_terms,
    E_a is of shape (B, N_x, N_y, N_z, N_energy_terms),
    passed along as last N_energy_terms features in the input features tensor.

    f_a(x_r) are local reweighting factors, which depend on local features x_r only;
    f_a(x_r) = NN_a(x_r) where NN_a is a small feedforward neural network.

    x_r are local features at grid point r, its shape is (B, N_x, N_y, N_z, N_feat).
    """

    def __init__(
        self,
        N_x: int,
        N_y: int,
        N_z: int,
        N_energy_terms: int,
        N_feat: int,
        n_hidden: int = 3,
        n_neurons: int = 16,
        mean_feat: torch.Tensor = None,
        std_feat: torch.Tensor = None,
        E_mean: torch.Tensor = None,
        E_std: torch.Tensor = None,
    ):

        super().__init__(mean_feat, std_feat, E_mean, E_std)
        self.N_x = N_x
        self.N_y = N_y
        self.N_z = N_z
        self.N_feat = N_feat
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        self.N_energy_terms = N_energy_terms

        self.local_nn = LocalNN3d(
            N_feat=N_feat,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            N_energy_terms=N_energy_terms,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features_orig = features[..., :self.N_feat] # (B, N_x, N_y, N_z, N_feat), normalized features
        energy_terms = features[..., self.N_feat:] # (B, N_x, N_y, N_z, N_energy_terms), physical (unnormalized) energy terms

        factors = self.local_nn(features_orig)  # (B, N_x, N_y, N_z, N_energy_terms)

        self._check_E_stats()

        E_tot = (factors * energy_terms).mean(dim=(1, 2, 3))  # (B, N_energy_terms)
        E_tot = E_tot.sum(dim=-1)  # (B,) total physical energy per batch element
        E_tot_norm = (E_tot - self.E_mean) / self.E_std  # (B,) normalized total energy
        return E_tot_norm


class EwaldNN3d(_NormalizedEnergyModel):
    """
    EwaldNN (simple variant), targeting the exchange-correlation energy:
        E_xc = (1 / N_x N_y N_z) * sum_r [ sum_a f_a(x_r) * E_a(r)
               + (1/2) a(x_r) dphi_r rho_r ],

    where dphi = (dK * rho) is a mediator field built from the density through
    the DELTA kernel dK = K_screened - K_Coulomb with learnable screening
    momentum qs (rs: (exp(-qs r) - 1)/r; ms: -4 pi qs^2 / (q^2 (q^2 + qs^2))).
    The Hartree subtraction thus happens at the kernel level: the pairwise term
    (1/2) dphi rho is the Hartree-subtracted interaction energy directly, never
    a difference of two separately computed large energies. At qs -> 0 the
    kernel vanishes, so the mediator term is a pure exchange-correlation object.

    a(x_r) = softplus(.) > 0 is a local reweighting factor for the mediator
    term, output by the same trunk as the f_a; it carries the interaction
    amplitude - the delta kernel itself has none.

    The feature appended to x_r is the SCREENED kernel-weight-normalized
    density average (see the mediator's phi_and_feature), scaled by the
    density std - short-ranged, unlike dK which has a -1/r tail. All
    normalizers are model parameters or fixed dataset constants - never
    per-sample statistics, which would introduce a spurious nonlocal
    interaction. The energy term uses the raw delta field.

    mediator_repr selects the kernel representation (ignored if a custom
    mediator module is passed):
        "rs" (default) - real-space delta kernel (Hartree convention, molecules);
        "ms"           - momentum-space delta kernel with uniform mode and
                         self-interaction removed.

    Input follows the LERN convention: features packs normalized features in
    features[..., :N_feat] (feature 0 = normalized density) and unnormalized
    energy terms E_a(r) in features[..., N_feat:].
    """

    def __init__(
        self,
        N_x: int,
        N_y: int,
        N_z: int,
        N_energy_terms: int,
        N_feat: int,
        n_hidden: int = 3,
        n_neurons: int = 16,
        mediator: nn.Module = None,
        mediator_repr: str = "rs",
        mean_feat: torch.Tensor = None,
        std_feat: torch.Tensor = None,
        E_mean: torch.Tensor = None,
        E_std: torch.Tensor = None,
    ):
        super().__init__(mean_feat, std_feat, E_mean, E_std)
        self.N_x = N_x
        self.N_y = N_y
        self.N_z = N_z
        self.N_feat = N_feat
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        self.N_energy_terms = N_energy_terms
        self.mediator_repr = mediator_repr

        # mediator: any module mapping rho (B, N_x, N_y, N_z) -> dphi (B, N_x, N_y, N_z)
        if mediator is not None:
            self.mediator = mediator
        elif mediator_repr == "rs":
            self.mediator = DeltaCoulombRSNonLocalKernelFFT(N_x, N_y, N_z)
        elif mediator_repr == "ms":
            self.mediator = DeltaCoulombNonLocalKernelFFT(N_x, N_y, N_z)
        else:
            raise ValueError(f"Unknown mediator_repr: {mediator_repr}")

        # +1 input feature: the (screened) mediator feature;
        # +1 output head: the mediator reweighting a(x_r)
        self.local_nn = LocalNN3d(
            N_feat=N_feat + 1,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            N_energy_terms=N_energy_terms + 1,
            raw_output=True,
        )
        # small amplitude at initialization: zero the a-head weights, bias -2
        # (a = softplus(-2) ~ 0.13 everywhere at the start)
        with torch.no_grad():
            out_layer = self.local_nn.loc_network[-1]
            out_layer.weight[N_energy_terms].zero_()
            out_layer.bias[N_energy_terms] = -2.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features_orig = features[..., :self.N_feat] # (B, N_x, N_y, N_z, N_feat), normalized features
        energy_terms = features[..., self.N_feat:] # (B, N_x, N_y, N_z, N_energy_terms), physical (unnormalized) energy terms

        rho = self._rho_from_features(features_orig)  # (B, N_x, N_y, N_z)
        if hasattr(self.mediator, "phi_and_feature"):
            dphi, phi_feat = self.mediator.phi_and_feature(rho)
        else:
            dphi = self.mediator(rho)  # custom mediator without a feature convention
            phi_feat = dphi
        phi_feat = phi_feat / self.std_feat[..., 0]  # scale like the density feature

        features_ext = torch.cat([features_orig, phi_feat.unsqueeze(-1)], dim=-1)
        z = self.local_nn(features_ext)  # (B, N_x, N_y, N_z, N_energy_terms + 1), raw

        factors = 1.0 + torch.tanh(z[..., :self.N_energy_terms])  # reweighting f_a in [0, 2]
        a = F.softplus(z[..., self.N_energy_terms])               # mediator reweighting a(x_r) > 0

        self._check_E_stats()

        E_loc = (factors * energy_terms).sum(dim=-1) + 0.5 * a * dphi * rho  # (B, N_x, N_y, N_z)
        E_tot = E_loc.mean(dim=(1, 2, 3))  # (B,) total physical energy per batch element

        E_tot_norm = (E_tot - self.E_mean) / self.E_std
        return E_tot_norm


class EwaldNNExtended3d(_NormalizedEnergyModel):
    """
    EwaldNN (extended variant), targeting the exchange-correlation energy:
        E_xc = (1 / N_x N_y N_z) * sum_r [ sum_a f_a(x_r) * E_a(r)
               + (1/2) a(x_r) sum_m s_m(x_r) dphi_{q_m}(r) rho_r ],

    with a fixed set of screening momenta {q_1, ..., q_M}. As in EwaldNN3d,
    the Hartree subtraction happens at the kernel level: the energy uses the
    DELTA fields dphi_{q_m} = (K_{q_m} - K_Coulomb) * rho. Since the softmax
    weights sum to 1, sum_m s_m dphi_{q_m} realizes the position-dependent
    kernel sum_m s_m K_{q_m} - K_Coulomb, i.e. a spatially varying screening
    with the bare Coulomb subtracted exactly once.

    The network outputs the softmax selection s_m(x_r) together with the local
    mediator reweighting a(x_r) = softplus(.) > 0. The features appended to
    x_r are the SCREENED fields phi_{q_m} = K_{q_m} * rho (short-ranged; the
    delta kernels carry a -1/r tail and never enter the feature vector).
    Both stacks are precomputed eigenvalue-wise at init: 2M global convolutions
    per forward pass.

    mediator_repr selects the kernel representation of the K_{q_m}:
        "rs" (default) - real-space Yukawa exp(-q_m r)/r and delta (exp(-q_m r) - 1)/r;
        "ms"           - momentum-space, see Lam_K_Coulomb / Lam_dK_Coulomb.

    A single LocalNN3d trunk outputs [f-logits (N_energy_terms), a-logit (1),
    s-logits (M)]; f_a = 1 + tanh(.), a = softplus(.), s = softmax(.).
    As in EwaldNN3d, the a-head row of the output layer is initialized with
    zero weights and bias -2 (a ~ 0.13), so the mediator starts small.

    Input follows the LERN convention (see EwaldNN3d).
    """

    def __init__(
        self,
        N_x: int,
        N_y: int,
        N_z: int,
        N_energy_terms: int,
        N_feat: int,
        qs_list: list = (0.1, 0.3, 1.0, 3.0),
        mediator_repr: str = "rs",
        n_hidden: int = 3,
        n_neurons: int = 16,
        mean_feat: torch.Tensor = None,
        std_feat: torch.Tensor = None,
        E_mean: torch.Tensor = None,
        E_std: torch.Tensor = None,
    ):
        super().__init__(mean_feat, std_feat, E_mean, E_std)
        self.N_x = N_x
        self.N_y = N_y
        self.N_z = N_z
        self.N_feat = N_feat
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        self.N_energy_terms = N_energy_terms
        self.qs_list = list(qs_list)
        self.mediator_repr = mediator_repr
        M = len(self.qs_list)
        self.M = M

        # precompute, for each fixed q_m, the screened-Coulomb eigenvalues
        # (features) and the delta-kernel eigenvalues (energy): (M, 2N_x, 2N_y, N_z+1)
        if mediator_repr == "rs":
            lam_K = torch.stack([_yukawa_eigenvals(N_x, N_y, N_z, qs) for qs in self.qs_list], dim=0)
            lam_dK = torch.stack([_yukawa_delta_eigenvals(N_x, N_y, N_z, qs) for qs in self.qs_list], dim=0)
        elif mediator_repr == "ms":
            q_vals = q_grid_fft(N_x, N_y, N_z)
            lam_K = torch.stack([Lam_K_Coulomb(q_vals, qs=qs) for qs in self.qs_list], dim=0)
            lam_dK = torch.stack([Lam_dK_Coulomb(q_vals, qs=qs) for qs in self.qs_list], dim=0)
        else:
            raise ValueError(f"Unknown mediator_repr: {mediator_repr}")
        self.register_buffer("lam_K", lam_K)
        self.register_buffer("lam_dK", lam_dK)
        self.register_buffer("lam_norm", lam_K.abs().amax(dim=(1, 2, 3)))  # (M,) feature normalizers

        # M extra input features: the screened fields phi_{q_m}
        self.local_nn = LocalNN3d(
            N_feat=N_feat + M,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            N_energy_terms=N_energy_terms + 1 + M,
            raw_output=True,
        )
        # small amplitude at initialization: zero the a-head weights, bias -2
        # (a = softplus(-2) ~ 0.13 everywhere at the start)
        with torch.no_grad():
            out_layer = self.local_nn.loc_network[-1]
            out_layer.weight[N_energy_terms].zero_()
            out_layer.bias[N_energy_terms] = -2.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features_orig = features[..., :self.N_feat] # (B, N_x, N_y, N_z, N_feat), normalized features
        energy_terms = features[..., self.N_feat:] # (B, N_x, N_y, N_z, N_energy_terms), physical (unnormalized) energy terms

        rho = self._rho_from_features(features_orig)  # (B, N_x, N_y, N_z)

        # screened phi_{q_m} = K_{q_m} * rho (features) and delta
        # dphi_{q_m} = (K_{q_m} - K_Coulomb) * rho (energy)
        phi_stack = torch.stack(
            [conv_fft(rho, self.lam_K[m].to(dtype=rho.dtype)) for m in range(self.M)],
            dim=-1)  # (B, N_x, N_y, N_z, M)
        dphi_stack = torch.stack(
            [conv_fft(rho, self.lam_dK[m].to(dtype=rho.dtype)) for m in range(self.M)],
            dim=-1)  # (B, N_x, N_y, N_z, M)

        phi_feat = (phi_stack / self.lam_norm) / self.std_feat[..., 0]  # per-channel, fixed normalizers
        features_ext = torch.cat([features_orig, phi_feat], dim=-1)
        z = self.local_nn(features_ext)  # (B, N_x, N_y, N_z, N_energy_terms + 1 + M), raw

        factors = 1.0 + torch.tanh(z[..., :self.N_energy_terms])   # reweighting f_a in [0, 2]
        a = F.softplus(z[..., self.N_energy_terms])                # mediator reweighting a(x_r) > 0
        s = torch.softmax(z[..., self.N_energy_terms + 1:], dim=-1)  # softmax selection s_m(x_r)

        dphi_eff = (s * dphi_stack).sum(dim=-1)  # (B, N_x, N_y, N_z)

        self._check_E_stats()

        E_loc = (factors * energy_terms).sum(dim=-1) + 0.5 * a * dphi_eff * rho  # (B, N_x, N_y, N_z)
        E_tot = E_loc.mean(dim=(1, 2, 3))  # (B,) total physical energy per batch element

        E_tot_norm = (E_tot - self.E_mean) / self.E_std
        return E_tot_norm
