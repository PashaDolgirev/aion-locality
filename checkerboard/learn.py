# learn.py -- Ewald spectral layer on the L x L torus, one parameter per D4-irreducible momentum, solved in closed
# form, with the second-moment diagnostics of the 1D notes.
#
#     E_b = sum_s X_{b,s} lam_s + c,     X_{b,s} = (1 / 2 N_s^2) sum_{k in orbit s} |rho_hat_{k,b}|^2,
# where orbit s = {(±k_x, ±k_y), (±k_y, ±k_x)} is the set of momenta related by the square-lattice symmetries
# (the kernel of a square torus has lam_k constant on each orbit).  With fixed N, sum_k |rho_hat_k|^2 = N N_s for
# every sample, so lam is determined up to an additive constant (the K_0 convention), and |rho_hat_0|^2 = N^2 is
# constant too, so lam_0 is unobservable (a uniform shift of K_r).  Gauge: lam_0 = 0 and sum_{k != 0} lam_k = 0,
# i.e. K_0 = 0 -- the convention of the energy formula itself (psi(0) := 0); the regression drops the k = 0 column and
# one further column (the last orbit) to remove the null direction, and align() then moves the result to the gauge.

import math
import torch


class OrbitBasis:
    def __init__(self, L):
        self.L, self.N_s = L, L * L
        k = torch.arange(L); kx, ky = torch.meshgrid(k, k, indexing="ij")
        kx = torch.minimum(kx, L - kx); ky = torch.minimum(ky, L - ky)        # fold to 0..L/2
        a, b = torch.maximum(kx, ky), torch.minimum(kx, ky)                 # 0 <= b <= a <= L/2
        key = a * (L // 2 + 1) + b
        uniq, inv = torch.unique(key.reshape(-1), return_inverse=True)
        self.orbit = inv.view(L, L)                                          # orbit index of every k
        self.n_par = len(uniq)
        self.k_rep = torch.stack([uniq // (L // 2 + 1), uniq % (L // 2 + 1)], 1)   # representative (a, b) per orbit
        self.mult = torch.bincount(inv, minlength=self.n_par).double()
        # reorder: k = 0 first (constant column, dropped from the regression), (pi, 0) last (column dropped to remove the null direction)
        i0 = int(((self.k_rep[:, 0] == 0) & (self.k_rep[:, 1] == 0)).nonzero()); iX = int(((self.k_rep[:, 0] == L // 2) & (self.k_rep[:, 1] == 0)).nonzero())
        order = [i0] + [i for i in range(self.n_par) if i not in (i0, iX)] + [iX]
        perm = torch.empty(self.n_par, dtype=torch.long); perm[torch.tensor(order)] = torch.arange(self.n_par)
        self.orbit = perm[self.orbit]; self.k_rep = self.k_rep[torch.tensor(order)]; self.mult = self.mult[torch.tensor(order)]
        self.q = 2 * math.pi * self.k_rep.double() / L                       # representative wavevector per orbit

    def design(self, pos, sim, batch=64):
        """X (B, n_par) from site indices pos (B, N, 2)."""
        out = []
        for b in range(0, pos.shape[0], batch):
            P = (torch.fft.fft2(sim.density(pos[b:b + batch])).abs() ** 2 / (2 * self.N_s ** 2)).reshape(-1, self.N_s)
            X = torch.zeros(P.shape[0], self.n_par, dtype=P.dtype); X.index_add_(1, self.orbit.reshape(-1), P)
            out.append(X)
        return torch.cat(out)

    def project(self, lam_full):
        """Orbit values of a D4-symmetric lam_k (L, L) (mean over the orbit)."""
        s = torch.zeros(self.n_par, dtype=torch.float64).index_add_(0, self.orbit.reshape(-1), lam_full.reshape(-1).double())
        return s / self.mult

    def lam_full(self, theta):
        return theta[self.orbit]

    def kernel(self, theta):
        """Real-space K_r (L, L) from orbit values in the gauge of align(): lam_0 = 0 gives sum_r K_r = 0 and
        sum_k lam_k = 0 gives K_0 = 0 (with fixed N the energies are invariant under lam_k -> lam_k + c and
        lam_0 -> lam_0 + c', i.e. K_r is determined up to a constant plus a delta at r = 0)."""
        return torch.fft.ifft2(self.lam_full(theta).to(torch.complex128)).real

    def mean_structure_factor(self, pos, sim, batch=64):
        acc = torch.zeros(self.L, self.L)
        for b in range(0, pos.shape[0], batch):
            acc += (torch.fft.fft2(sim.density(pos[b:b + batch])).abs() ** 2).sum(0)
        S = acc / (pos.shape[0] * sim.N); S[0, 0] = 0.0
        return torch.fft.fftshift(S)


def fit_kernel(X, E):
    """Least squares E = X theta + c (columns 1..n-2; column 0 is k = 0 and the last is the gauge).  Returns (theta, c)."""
    Xr = X[:, 1:-1]; Xm, Em = Xr.mean(0), E.mean()
    sc = (Xr - Xm).norm(dim=0).clamp(min=1e-300)
    th = torch.linalg.lstsq((Xr - Xm) / sc, (E - Em).unsqueeze(1), driver="gelsd").solution.squeeze() / sc
    theta = torch.cat([th.new_zeros(1), th, th.new_zeros(1)])
    return theta, float(Em - Xm @ th)


def align(lam, mult):
    """Gauge: lam_0 = 0 (unobservable) and sum_{k != 0} lam_k = 0 over the full grid (orbit multiplicities mult),
    i.e. K_0 = 0.  Applied to learned and true kernels alike before they are compared."""
    out = lam - float((lam[1:] * mult[1:]).sum() / mult[1:].sum()); out[0] = 0.0
    return out


def second_moment(X):
    """Eigenvalues (descending, normalised) of the correlation matrix of the regression columns."""
    Xc = X[:, 1:-1] - X[:, 1:-1].mean(0)
    if float(Xc.abs().max()) == 0.0: return torch.zeros(Xc.shape[1])        # identical configurations: no information
    s = torch.linalg.svdvals(Xc / Xc.norm(dim=0).clamp(min=1e-300))
    return (s / s[0]) ** 2


def effective_rank(ev, eps):
    """Smooth count of non-singular directions: df(eps) = sum_i e_i / (e_i + eps) (ridge degrees of freedom), with e_i the
    eigenvalues of second_moment normalised to the largest; eps ~ (relative label noise)^2 makes it the number of kernel
    parameters that noise can resolve.  Its eps -> 0 limit is the threshold count."""
    ev = ev[torch.isfinite(ev)]
    return float((ev / (ev + eps)).sum())


def jackknife(values):
    """Leave-one-out jackknife: values (K,) of a statistic computed on the K leave-one-group-out subsets -> (mean, std)."""
    v = torch.as_tensor(values, dtype=torch.float64); K = len(v)
    return float(v.mean()), float(((K - 1) / K * ((v - v.mean()) ** 2).sum()).sqrt())


def sensitivity(X, theta, E_std, n_noise=256, seed=0):
    """Relative error of each orbit value per unit relative label noise (data property, no noise added)."""
    Xc = X[:, 1:-1] - X[:, 1:-1].mean(0); sc = Xc.norm(dim=0).clamp(min=1e-300)
    eps = torch.randn(len(X), n_noise, generator=torch.Generator().manual_seed(seed))
    dth = torch.linalg.lstsq(Xc / sc, eps, driver="gelsd").solution / sc.unsqueeze(1)
    out = E_std * dth.std(1) / theta[1:-1].abs().clamp(min=1e-300)
    return torch.cat([out.new_zeros(1), out, out.new_zeros(1)])
