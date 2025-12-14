import torch
import torch.nn as nn
import torch.nn.functional as F

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