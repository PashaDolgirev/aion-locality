# lattice_wc.py
#
# Wigner crystal: the two-dimensional one-component plasma on a square grid: N unit point charges on the sites of an N_x x N_y grid of
# spacing s (at most one charge per site), interacting via 1/r with all periodic images and a uniform neutralising
# background.  Units e = k_B = 1, number density n = N/A = 1, Wigner-Seitz radius a = (pi n)^{-1/2},
# coupling Gamma = 1/(k_B T a).
#
# Energy (same conventions as checkerboard/lattice_cdw.py, N_s = N_x N_y sites, rho_r in {0,1}):
#     E = 1/2 sum_{i != j} psi(r_i - r_j) + N c_0
#       = (1 / 2 N_s) sum_{r,r'} K_{r-r'} rho_r rho_r' + N c_0 = (1 / 2 N_s^2) sum_q lam_q |rho_q|^2 + N c_0,
#     K_r = N_s psi(r),  K_0 = 0 (psi(0) := 0),  lam_q = DFT(K),
# where psi is the exact periodic Coulomb pair potential of point charges (Ewald sum, all images, independent of the
# splitting parameter) and c_0 = 1/2 lim_{r->0} [psi(r) - 1/r] collects the self and background constants.
#
# The box is the commensurate rectangle of a triangular lattice with nearest-neighbour distance
# a_lat = (2/sqrt3)^{1/2}: L_x = n_x a_lat, L_y = n_y sqrt3 a_lat, N = 2 n_x n_y, with
# N_x = m n_x sites along x (s = a_lat/m) and N_y = round(m n_y sqrt3) along y.

import math
import torch

WS_RADIUS = 1.0 / math.sqrt(math.pi)            # a for n = 1
A_LAT = math.sqrt(2.0 / math.sqrt(3.0))         # triangular-lattice nearest-neighbour distance for n = 1


def kBT_from_Gamma(Gamma):
    return 1.0 / (Gamma * WS_RADIUS)


# ---------------------------------------------------------------- Coulomb kernel on the grid

def ewald_pair_potential(N_x, N_y, s, alpha=None, tol=1e-16):
    """
    psi(r) on the N_x x N_y grid (spacing s, box L = (N_x s, N_y s)) for unit point charges with a neutralising
    background, and the constant c_0 with  E = 1/2 sum_{i != j} psi(r_ij) + N c_0,  c_0 = 1/2 lim_{r->0} [psi(r) - 1/r].
    Ewald splitting 1/r = erfc(alpha r)/r + erf(alpha r)/r (A = L_x L_y, n o L = (n_x L_x, n_y L_y)):
        psi(r) = sum_n erfc(alpha |r + n o L|)/|r + n o L| + (2 pi/A) sum_{G != 0} erfc(G/2alpha)/G cos(G.r) - 2 sqrt(pi)/(alpha A),
        c_0    = 1/2 sum_{n != 0} erfc(alpha |n o L|)/|n o L| + (pi/A) sum_{G != 0} erfc(G/2alpha)/G - alpha/sqrt(pi) - sqrt(pi)/(alpha A).
    Both sums are truncated where erfc < tol, so the result is independent of alpha to round-off
    (debugging/test_wc.py scans alpha).  The reciprocal sum restricted to the grid momenta is one inverse FFT, which is
    exact as long as G_max = 2 alpha x_cut < pi/s (asserted; the default alpha = 9/min(L), reduced on coarse test grids, satisfies it).
    Returns psi (N_x, N_y) with psi[0, 0] = 0, and c_0.
    """
    L = torch.tensor([N_x * s, N_y * s], dtype=torch.float64); A = float(L.prod()); N_s = N_x * N_y
    x_cut = 1.0
    while math.erfc(x_cut) > tol: x_cut += 0.1                                          # erfc(x_cut) <= tol
    if alpha is None: alpha = min(9.0 / float(L.min()), 0.45 * math.pi / (s * x_cut))   # alpha L = 9 unless the grid is coarse
    assert 2 * alpha * x_cut < math.pi / s, "alpha too large for the grid: the reciprocal sum would alias"
    # reciprocal sum over the grid momenta G = 2 pi (m_x/L_x, m_y/L_y) (rfft2 half grid)
    k_x = torch.fft.fftfreq(N_x, dtype=torch.float64) * N_x; k_y = torch.fft.rfftfreq(N_y, dtype=torch.float64) * N_y
    g = torch.sqrt((2 * math.pi * k_x / L[0]).view(-1, 1) ** 2 + (2 * math.pi * k_y / L[1]).view(1, -1) ** 2)
    w = torch.where(g > 0, torch.special.erfc(g / (2 * alpha)) / g.clamp(min=1e-300), torch.zeros_like(g))
    recip = torch.fft.irfft2(w.to(torch.complex128), s=(N_x, N_y)) * N_s * (2 * math.pi / A)   # (2 pi/A) sum_G w(G) cos(G.r)
    mult = torch.full((k_y.numel(),), 2.0); mult[0] = 1.0                                  # half-grid multiplicities
    if N_y % 2 == 0: mult[-1] = 1.0
    sum_w = float((w * mult.view(1, -1)).sum())
    # real-space sum over all images with alpha |r + n o L| <= x_cut (r in the minimum-image cell)
    n_r = [math.ceil(x_cut / (alpha * float(Li)) + 0.5) for Li in L]
    nx, ny = torch.meshgrid(torch.arange(-n_r[0], n_r[0] + 1), torch.arange(-n_r[1], n_r[1] + 1), indexing="ij")
    shifts = torch.stack([nx.reshape(-1) * L[0], ny.reshape(-1) * L[1]], 1)                  # (n_img, 2)
    ix, iy = torch.arange(N_x), torch.arange(N_y)
    rx = (ix - N_x * torch.round(ix / N_x)) * s; ry = (iy - N_y * torch.round(iy / N_y)) * s
    real = torch.zeros(N_x, N_y, dtype=torch.float64)
    for c in range(0, N_x, 64):                                                               # chunked over rows
        r = torch.stack(torch.meshgrid(rx[c:c + 64], ry, indexing="ij"), -1)
        d = (r.unsqueeze(2) + shifts.view(1, 1, -1, 2)).norm(dim=-1)
        real[c:c + 64] = torch.where(d > 0, torch.special.erfc(alpha * d) / d.clamp(min=1e-300), torch.zeros_like(d)).sum(-1)
    psi = real + recip - 2 * math.sqrt(math.pi) / (alpha * A)
    psi[0, 0] = 0.0
    d0 = shifts.norm(dim=1); self_images = float((torch.special.erfc(alpha * d0[d0 > 0]) / d0[d0 > 0]).sum())
    c0 = 0.5 * self_images + (math.pi / A) * sum_w - alpha / math.sqrt(math.pi) - math.sqrt(math.pi) / (alpha * A)
    return psi, c0


# ---------------------------------------------------------------- lattice gas Monte Carlo

class LatticeWC:
    """
    Metropolis lattice gas, C independent chains in lock-step.
    pos: (C, N, 2) long site indices; occ: (C, N_x, N_y) bool.
    A trial moves one particle to an empty site: with probability p_long a uniformly random site,
    otherwise a site within the window [-w, w]^2 (w tuned during equilibration, then frozen).
        dE = sum_{j != i} [ psi(r_new - r_j) - psi(r_old - r_j) ]      (table lookups)
    """

    def __init__(self, n_x, n_y, m, Gamma, n_chains=1, seed=0, p_long=0.1, init="lattice", device="cpu"):
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.device = device
        self.n_x, self.n_y, self.m = n_x, n_y, m
        self.s = A_LAT / m
        self.N_x, self.N_y = m * n_x, int(round(m * n_y * math.sqrt(3.0)))
        self.N = 2 * n_x * n_y
        self.L = torch.tensor([self.N_x * self.s, self.N_y * self.s])
        self.A = float(self.L.prod())
        self.Gamma, self.kBT = Gamma, kBT_from_Gamma(Gamma); self.beta = 1.0 / self.kBT
        self.psi, self.c0 = ewald_pair_potential(self.N_x, self.N_y, self.s)        # exact periodic Coulomb pair table, psi[0,0] = 0
        self.psi = self.psi.to(device)
        self.C, self.p_long = n_chains, p_long

        if init == "lattice":
            i = torch.arange(n_x).view(-1, 1).expand(-1, 2 * n_y); j = torch.arange(2 * n_y).view(1, -1).expand(n_x, -1)
            x = (i + 0.5 * (j % 2)) * m; y = j * (self.N_y / (2 * n_y))            # ideal lattice in site units
            base = torch.stack([x.reshape(-1).round().long() % self.N_x, y.reshape(-1).round().long() % self.N_y], 1)
            pos = base.unsqueeze(0).expand(n_chains, -1, -1).clone()
        elif init == "random":
            pos = torch.stack([torch.randperm(self.N_x * self.N_y, generator=self.gen)[: self.N] for _ in range(n_chains)])
            pos = torch.stack([pos // self.N_y, pos % self.N_y], -1)
        else:
            raise ValueError(init)
        self.pos = pos.to(device)
        self.occ = torch.zeros(n_chains, self.N_x, self.N_y, dtype=torch.bool, device=device)
        ar = torch.arange(n_chains, device=device).view(-1, 1)
        self.occ[ar, self.pos[..., 0], self.pos[..., 1]] = True
        assert int(self.occ.sum()) == n_chains * self.N, "initial configuration has collisions"
        self.energy = self.total_energy()
        self.w = max(1, m // 4)
        self.n_acc = torch.zeros(n_chains, device=device); self.n_att = 0

    # ---- energies ----
    def _pair_psi(self, x, pos):
        """psi between sites x (C, K, 2) and all particles pos (C, N, 2) -> (C, K, N)."""
        d = x.unsqueeze(2) - pos.unsqueeze(1)
        return torch.take(self.psi, (d[..., 0] % self.N_x) * self.N_y + d[..., 1] % self.N_y)

    def total_energy(self):
        d = self.pos.unsqueeze(2) - self.pos.unsqueeze(1)                       # (C, N, N, 2)
        return 0.5 * self.psi[d[..., 0] % self.N_x, d[..., 1] % self.N_y].sum(dim=(1, 2)) + self.N * self.c0

    def density(self, pos=None):
        """Occupation grid rho (B, N_x, N_y) float from site indices (B, N, 2)."""
        pos = self.pos if pos is None else pos
        B = pos.shape[0]
        rho = torch.zeros(B, self.N_x * self.N_y, dtype=torch.float64, device=pos.device)
        rho.scatter_add_(1, pos[..., 0] * self.N_y + pos[..., 1], torch.ones(B, pos.shape[1], dtype=torch.float64, device=pos.device))
        return rho.view(B, self.N_x, self.N_y)

    # ---- Metropolis ----
    def sweep(self, n_sweeps=1):
        C, N = self.C, self.N
        ar = torch.arange(C, device=self.device)
        for _ in range(n_sweeps * N):
            i = torch.randint(N, (C,), generator=self.gen, device=self.device)
            old = self.pos[ar, i]
            local = torch.randint(-self.w, self.w + 1, (C, 2), generator=self.gen, device=self.device)
            glob = torch.stack([torch.randint(self.N_x, (C,), generator=self.gen, device=self.device),
                                torch.randint(self.N_y, (C,), generator=self.gen, device=self.device)], 1)
            use_long = torch.rand(C, generator=self.gen, device=self.device) < self.p_long
            new = torch.where(use_long.view(-1, 1), glob, old + local)
            new = torch.stack([new[:, 0] % self.N_x, new[:, 1] % self.N_y], 1)
            free = ~self.occ[ar, new[:, 0], new[:, 1]]                            # occupied (incl. own site) -> reject
            e = self._pair_psi(torch.stack([new, old], 1), self.pos)                # (C, 2, N): new and old site
            e[ar, 0, i] = 0.0                                                      # exclude j = i (psi(0) = 0 for old)
            dE = (e[:, 0] - e[:, 1]).sum(1)
            u = torch.rand(C, generator=self.gen, device=self.device)
            acc = free & ((dE <= 0) | (u < torch.exp(-self.beta * dE.clamp(min=0))))
            a = ar[acc]
            self.occ[a, old[acc, 0], old[acc, 1]] = False
            self.occ[a, new[acc, 0], new[acc, 1]] = True
            self.pos[a, i[acc]] = new[acc]
            self.energy = self.energy + torch.where(acc, dE, torch.zeros_like(dE))
            self.n_acc += acc.double(); self.n_att += 1

    def tune_window(self, target=(0.3, 0.55)):
        rate = float(self.n_acc.mean()) / max(self.n_att, 1)
        if rate < target[0]: self.w = max(1, int(self.w * 0.8))
        elif rate > target[1]: self.w = min(self.N_x // 4, int(self.w * 1.25) + 1)
        self.n_acc.zero_(); self.n_att = 0
        return rate

    def run(self, n_equil, n_prod, save_every, tune_every=50, verbose=True):
        rate = float("nan")
        for t in range(0, n_equil, tune_every):
            self.sweep(min(tune_every, n_equil - t)); rate = self.tune_window()
        self.energy = self.total_energy()
        if verbose:
            print(f"  Gamma={self.Gamma:g} N={self.N} grid {self.N_x}x{self.N_y} s={self.s:.4f} C={self.C} "
                  f"kBT={self.kBT:.4g} w={self.w} acc={rate:.2f} E/N={float(self.energy.mean())/self.N:.5f}", flush=True)
        configs, energies = [], []
        for _ in range(n_prod // save_every):
            self.sweep(save_every); configs.append(self.pos.clone()); energies.append(self.energy.clone())
        E_inc = torch.stack(energies); self.energy = self.total_energy()
        drift = float((E_inc[-1] - self.energy).abs().max())
        if verbose:
            print(f"  production: {len(configs)} snapshots/chain, <E/N>={float(E_inc.mean())/self.N:.5f}, "
                  f"incremental-vs-recomputed drift={drift:.1e}", flush=True)
        return torch.stack(configs), E_inc


# ---------------------------------------------------------------- statistics

def integrated_autocorr_time(x):
    """tau_int = 1/2 + sum_{t>=1} rho(t), summed to the first negative autocorrelation."""
    x = x.double() - x.double().mean(); T = len(x)
    f = torch.fft.rfft(x, n=2 * T); ac = torch.fft.irfft(f * f.conj(), n=2 * T)[:T] / torch.arange(T, 0, -1); ac = ac / ac[0]
    tau = 0.5
    for t in range(1, T // 2):
        if ac[t] < 0: break
        tau += float(ac[t])
    return tau


def neighbour_displacement(pos, sim, k=6, batch=8):
    """rms spread of the k nearest-neighbour distances (Lindemann-type measure of thermal displacement), pos (B, N, 2) site indices."""
    nn = []
    for b in range(0, pos.shape[0], batch):
        r = pos[b:b + batch].double() * sim.s
        d = r[:, :, None, :] - r[:, None, :, :]; d = d - sim.L * torch.round(d / sim.L)
        dist = d.norm(dim=-1); dist[:, range(sim.N), range(sim.N)] = float("inf")
        nn.append(dist.topk(k, dim=-1, largest=False).values)
    nn = torch.cat(nn)
    return float((nn - nn.mean()).pow(2).mean().sqrt()), float(nn.mean())
