# lattice_cdw.py
#
# Half-filled square-lattice Coulomb gas with nearest-neighbour repulsion (checkerboard charge-density wave).
# N = L^2/2 unit charges on an L x L square lattice (spacing a = 1, periodic), at most one per site, uniform
# neutralising background, interacting via 1/r (all images) plus V1 for every nearest-neighbour pair.
# Units e = a = k_B = 1; temperature via Gamma = e^2/(a k_B T) = 1/T.
#
# Energy (notation of the 1D notes, N_s = L^2 sites, rho_i in {0,1}):
#     E[rho] = (1 / 2 N_s) sum_{i,j} K_{i-j} rho_i rho_j + N c_0 = (1 / 2 N_s^2) sum_k lam_k |rho_hat_k|^2 + N c_0,
#     K_r = N_s [ psi(r) + V1 delta_{|r| = 1} ],   K_0 = 0,   lam_k = DFT(K),
# where psi(r) is the periodic Coulomb pair potential of point charges (Ewald sum, derived in
# Notes/notes_CDW.tex) and c_0 collects the self and background constants; E is exactly the point-charge Ewald
# energy plus V1 x (number of nearest-neighbour pairs).

import math
import torch


def ewald_pair_potential(L, alpha=None, tol=1e-16):
    """
    psi(r) on the L x L torus (a = 1) for point charges with a neutralising background, and the constant c_0 with
        E = 1/2 sum_{i != j} psi(r_ij) + N c_0,      c_0 = 1/2 lim_{r -> 0} [psi(r) - 1/r].
    Ewald splitting 1/r = erfc(alpha r)/r + erf(alpha r)/r:
        psi(r) = sum_n erfc(alpha |r + L n|)/|r + L n| + (2 pi/A) sum_{G != 0} erfc(G/2alpha)/G cos(G.r) - 2 sqrt(pi)/(alpha A),
        c_0    = 1/2 sum_{n != 0} erfc(alpha L|n|)/(L|n|) + (pi/A) sum_{G != 0} erfc(G/2alpha)/G - alpha/sqrt(pi) - sqrt(pi)/(alpha A).
    Both sums are truncated where erfc < tol (relative to the leading term), so the result is independent of alpha to
    round-off for any alpha > 0; alpha = 9/L is the default (one real-space image suffices).
    Returns psi (L, L) with psi[0, 0] = 0, and c_0.
    """
    A = float(L * L); alpha = 9.0 / L if alpha is None else float(alpha)
    x_cut = 1.0
    while math.erfc(x_cut) > tol: x_cut += 0.1                                          # erfc(x_cut) <= tol
    # reciprocal sum over G = 2 pi m / L with G/2alpha <= x_cut
    n = math.ceil(x_cut * alpha * L / math.pi)
    mx, my = torch.meshgrid(torch.arange(-n, n + 1), torch.arange(-n, n + 1), indexing="ij")
    mask = ~((mx == 0) & (my == 0))
    G = 2 * math.pi * torch.stack([mx[mask], my[mask]], 1).double() / L
    g = G.norm(dim=1); w = torch.special.erfc(g / (2 * alpha)) / g
    # real-space sum over images |r + L n| with alpha |r + L n| <= x_cut  (r in the minimum-image cell, so |n| <= n_r)
    n_r = math.ceil(x_cut / (alpha * L) + 0.5)
    nx, ny = torch.meshgrid(torch.arange(-n_r, n_r + 1), torch.arange(-n_r, n_r + 1), indexing="ij")
    shifts = L * torch.stack([nx.reshape(-1), ny.reshape(-1)], 1).double()                 # (n_img, 2)
    r = torch.stack(torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij"), -1).double().reshape(-1, 2)
    r = r - L * torch.round(r / L)                                                          # minimum image
    d = (r.unsqueeze(1) + shifts.unsqueeze(0)).norm(dim=2)                                   # (L^2, n_img)
    real = torch.where(d > 0, torch.special.erfc(alpha * d) / d.clamp(min=1e-300), torch.zeros_like(d)).sum(1)
    recip = (2 * math.pi / A) * (torch.cos(r @ G.T) * w).sum(1)
    psi = (real + recip - 2 * math.sqrt(math.pi) / (alpha * A)).view(L, L)
    psi[0, 0] = 0.0
    d0 = shifts.norm(dim=1); self_images = float((torch.special.erfc(alpha * d0[d0 > 0]) / d0[d0 > 0]).sum())
    c0 = 0.5 * self_images + (math.pi / A) * float(w.sum()) - alpha / math.sqrt(math.pi) - math.sqrt(math.pi) / (alpha * A)
    return psi, c0


class LatticeCDW:
    """Metropolis lattice gas at half filling, C chains in lock-step; moves: nearest-neighbour hop (prob 1 - p_long) or
    hop to a uniformly random site (prob p_long), accepted with min(1, exp(-dE/T)); dE by table lookup."""

    def __init__(self, L, V1, Gamma, n_chains=1, seed=0, p_long=0.1, init="checkerboard", device="cpu"):
        assert L % 2 == 0
        self.gen = torch.Generator(device=device).manual_seed(seed); self.device = device
        self.L, self.V1, self.Gamma = L, V1, Gamma
        self.N_s, self.N, self.A = L * L, L * L // 2, float(L * L)
        self.kBT = 1.0 / Gamma; self.beta = Gamma
        psi, self.c0 = ewald_pair_potential(L)
        self.psi_coulomb = psi.clone()
        nn = torch.zeros(L, L, dtype=torch.float64); nn[1, 0] = nn[L - 1, 0] = nn[0, 1] = nn[0, L - 1] = 1.0
        self.psi = psi + V1 * nn                                             # full pair table, psi[0,0] = 0
        self.K = self.N_s * self.psi                                          
        self.lam = torch.fft.fft2(self.K.to(torch.complex128)).real           # lam_k = DFT(K), real by symmetry
        self.psi = self.psi.to(device); self.C, self.p_long = n_chains, p_long
        ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
        if init == "checkerboard":
            occ = ((ii + jj) % 2 == 0)
            pos = torch.stack([ii[occ], jj[occ]], 1).unsqueeze(0).expand(n_chains, -1, -1).clone()
        elif init == "random":
            pos = torch.stack([torch.randperm(self.N_s, generator=self.gen)[: self.N] for _ in range(n_chains)])
            pos = torch.stack([pos // L, pos % L], -1)
        else:
            raise ValueError(init)
        self.pos = pos.to(device)
        self.occ = torch.zeros(n_chains, L, L, dtype=torch.bool, device=device)
        self.occ[torch.arange(n_chains).view(-1, 1), self.pos[..., 0], self.pos[..., 1]] = True
        assert int(self.occ.sum()) == n_chains * self.N
        self.energy = self.total_energy()
        self.n_acc = torch.zeros(n_chains, device=device); self.n_att = 0
        self.nn_vec = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=device)

    def _pair_psi(self, x, pos):
        d = x.unsqueeze(2) - pos.unsqueeze(1)
        return torch.take(self.psi, (d[..., 0] % self.L) * self.L + d[..., 1] % self.L)

    def total_energy(self):
        d = self.pos.unsqueeze(2) - self.pos.unsqueeze(1)
        return 0.5 * self.psi[d[..., 0] % self.L, d[..., 1] % self.L].sum(dim=(1, 2)) + self.N * self.c0

    def density(self, pos=None):
        pos = self.pos if pos is None else pos; B = pos.shape[0]
        rho = torch.zeros(B, self.N_s, dtype=torch.float64, device=pos.device)
        rho.scatter_add_(1, pos[..., 0] * self.L + pos[..., 1], torch.ones(B, pos.shape[1], dtype=torch.float64, device=pos.device))
        return rho.view(B, self.L, self.L)

    def n_neighbour_pairs(self, pos=None):
        """Number of occupied nearest-neighbour pairs per configuration (zero for the checkerboard)."""
        rho = self.density(pos)
        return (rho * (rho.roll(1, 1) + rho.roll(1, 2))).sum(dim=(1, 2))

    def sweep(self, n_sweeps=1):
        C, N, L = self.C, self.N, self.L
        ar = torch.arange(C, device=self.device)
        for _ in range(n_sweeps * N):
            i = torch.randint(N, (C,), generator=self.gen, device=self.device)
            old = self.pos[ar, i]
            local = old + self.nn_vec[torch.randint(4, (C,), generator=self.gen, device=self.device)]
            glob = torch.randint(L, (C, 2), generator=self.gen, device=self.device)
            new = torch.where((torch.rand(C, generator=self.gen, device=self.device) < self.p_long).view(-1, 1), glob, local) % L
            free = ~self.occ[ar, new[:, 0], new[:, 1]]
            e = self._pair_psi(torch.stack([new, old], 1), self.pos); e[ar, 0, i] = 0.0
            dE = (e[:, 0] - e[:, 1]).sum(1)
            acc = free & ((dE <= 0) | (torch.rand(C, generator=self.gen, device=self.device) < torch.exp(-self.beta * dE.clamp(min=0))))
            a = ar[acc]
            self.occ[a, old[acc, 0], old[acc, 1]] = False; self.occ[a, new[acc, 0], new[acc, 1]] = True
            self.pos[a, i[acc]] = new[acc]
            self.energy = self.energy + torch.where(acc, dE, torch.zeros_like(dE))
            self.n_acc += acc.double(); self.n_att += 1

    def run(self, n_equil, n_prod, save_every, verbose=True):
        self.sweep(n_equil); self.energy = self.total_energy(); self.n_acc.zero_(); self.n_att = 0
        configs, energies = [], []
        for _ in range(n_prod // save_every):
            self.sweep(save_every); configs.append(self.pos.clone()); energies.append(self.energy.clone())
        E = torch.stack(energies); drift = float((E[-1] - self.total_energy()).abs().max())
        if verbose:
            print(f"  L={self.L} V1={self.V1:g} Gamma={self.Gamma:g} (T={self.kBT:.3g}) C={self.C}: <E/N>={float(E.mean()) / self.N:.5f}, "
                  f"acc={float(self.n_acc.mean()) / max(self.n_att, 1):.3f}, drift={drift:.1e}", flush=True)
        return torch.stack(configs), E


def integrated_autocorr_time(x):
    x = x.double() - x.double().mean(); T = len(x)
    f = torch.fft.rfft(x, n=2 * T); ac = torch.fft.irfft(f * f.conj(), n=2 * T)[:T] / torch.arange(T, 0, -1); ac = ac / ac[0]
    tau = 0.5
    for t in range(1, T // 2):
        if ac[t] < 0: break
        tau += float(ac[t])
    return tau
