# learn.py
#
#
# Conventions (unnormalised DFT, N_s = N_x N_y):  rho_q = sum_r rho_r e^{-i q.r},  q = 2 pi (m_x/L_x, m_y/L_y)
#     E[rho] = (1 / 2 N_s) sum_{r,r'} K_{r-r'} rho_r rho_r' + const = (1 / 2 N_s^2) sum_q lam_q |rho_q|^2 + const.
# The grid-sampled periodic potential is the aliased continuum kernel V(q):
#     lam_q = (N_s^2 / A) sum_{m in Z^2} V(|q + Q_m|) + C,     Q_m = 2 pi m / s,
# so we parametrise V, not lam.  Inductive bias: V is isotropic with the Coulomb singularity, V(q) = mu(q)/q, mu free:
# inside the grid zone (m = 0) mu is one unknown per exact radius for small q and linearly interpolated between knots
# above; the aliased terms (m != 0) sample V only beyond the zone, where V = mu_inf / q with one free amplitude, giving
# the fixed lattice function T(q) = sum_{m != 0} [1/|q + Q_m| - 1/Q_m] (obtained from the exact Ewald kernel as
# T = lam^exact A/(2 pi N_s^2) - 1/q).  The energy is then a linear regression in theta = (mu at the knots, mu_inf):
#     E_b = sum_s X_{b,s} theta_s + const,   X_{b,s} = (1 / 2 N_s^2) sum_q W_{q s} |rho_{q,b}|^2 / q,
#                                            X_{b,T} = (1 / 2 A) sum_q T(q) |rho_{q,b}|^2.
# The exact point-charge kernel is mu = mu_inf = 2 pi and lies exactly in the model class.  Two directions are
# unobservable: lam_0 (|rho_0|^2 = N^2) -- q = 0 carries no weight -- and a uniform shift
# of lam (sum_q |rho_q|^2 = N_s N for 0/1 occupations; theta_s -> theta_s + c q_s), fixed by K_0 = 0, i.e.
# sum_{q != 0} lam_q = 0, imposed on learned and exact kernels alike (align).

import math
import torch


class RingBasis:
    """
    Isotropic basis lam(|q|) = mu(|q|)/|q|: one parameter per exact radius for |q| < q_exact (where radii are
    sparse), then piecewise-linear interpolation of mu between knots of growing spacing.
    lam_k = sum_s W_{k s} theta_s with at most two nonzero entries per k, stored as (i0, w0, w1):
    lam_k = w0_k theta_{i0_k} + w1_k theta_{i0_k + 1};  design X = P W.
    """
    def __init__(self, N_x, N_y, L, lam_exact=None, q_exact=3.0, spacings=((20.0, 0.1), (60.0, 0.5), (1e9, 4.0))):
        """lam_exact: the exact lattice kernel on the rfft2 half grid (DFT of N_s psi, K_0 = 0); it supplies the
        alias function T(q).  Without it the basis is the m = 0 (purely isotropic) model."""
        self.N_x, self.N_y = N_x, N_y
        self.N_s = N_x * N_y
        L_x, L_y = float(L[0]), float(L[1])
        k_x = torch.fft.fftfreq(self.N_x) * self.N_x; k_y = torch.fft.rfftfreq(self.N_y) * self.N_y
        self.q = torch.sqrt((2 * math.pi * k_x / L_x).view(-1, 1) ** 2 + (2 * math.pi * k_y / L_y).view(1, -1) ** 2)  # (N_x, N_y//2+1)
        w = torch.full((self.N_y // 2 + 1,), 2.0); w[0] = 1.0
        if self.N_y % 2 == 0: w[-1] = 1.0
        self.weight = w.view(1, -1).expand(self.N_x, -1).clone()       # rfft2 half-grid multiplicities
        qf = self.q.reshape(-1); self.n_k = len(qf)
        # exact radii below q_exact
        radii = torch.unique(torch.round(qf[(qf > 0) & (qf < q_exact)], decimals=9))
        knots = [float(r) for r in radii]
        # interpolation knots above
        q = q_exact; q_max = float(qf.max())
        for q_hi, dq in spacings:
            while q < min(q_hi, q_max + dq):
                knots.append(q); q += dq
            if q_hi > q_max: break
        knots.append(q)                                                  # one knot beyond q_max
        q_knots = torch.tensor(knots); n_par = len(knots)
        idx_lo = (torch.searchsorted(q_knots, qf, right=True) - 1).clamp(0, n_par - 2)
        q_lo, q_hi = q_knots[idx_lo], q_knots[idx_lo + 1]
        t = ((qf - q_lo) / (q_hi - q_lo)).clamp(0, 1)
        t = torch.where(qf < q_exact, torch.round(t), t)               # exact region: snap to the matching radius
        valid = qf > 0
        inv_q = valid / torch.where(valid, qf, torch.ones_like(qf))      # lam = mu / q; q = 0 carries no weight
        w0, w1 = (1 - t) * inv_q, t * inv_q
        # drop knots that no momentum touches and renumber
        support = torch.zeros(n_par).index_add_(0, idx_lo, w0 * self.weight.reshape(-1)).index_add_(0, idx_lo + 1, w1 * self.weight.reshape(-1))
        keep = support > 0; new_index = torch.cumsum(keep.long(), 0) - 1
        w0 = torch.where(keep[idx_lo], w0, torch.zeros_like(w0)); w1 = torch.where(keep[idx_lo + 1], w1, torch.zeros_like(w1))
        self.i0 = new_index[idx_lo].clamp(min=0); self.i1 = new_index[idx_lo + 1].clamp(min=0)
        self.w0, self.w1 = w0, w1
        self.q_knots = q_knots[keep]; self.n_par = int(keep.sum())
        self.n_exact = int((self.q_knots < q_exact).sum())
        self.count = support[keep]                                       # k-weight per parameter
        self.A = L_x * L_y
        if lam_exact is not None:                                        # alias column: lam^T = lam_exact - (N_s^2/A) 2pi/q
            coul = torch.where(self.q > 0, (self.N_s ** 2 / self.A) * 2 * math.pi / self.q.clamp(min=1e-300), torch.zeros_like(self.q))
            self.lam_T = (lam_exact - coul) * self.A / (2 * math.pi * self.N_s ** 2); self.lam_T[0, 0] = 0.0   # per unit theta_T = (N_s^2/A) mu_inf
            self.n_par += 1; self.q_knots = torch.cat([self.q_knots, torch.tensor([float("inf")])])
        else:
            self.lam_T = None

    def _apply_W(self, P):
        """P (B, n_k) -> P W (B, n_par)."""
        X = torch.zeros(P.shape[0], self.n_par - (self.lam_T is not None), dtype=P.dtype)
        X.index_add_(1, self.i0, P * self.w0); X.index_add_(1, self.i1, P * self.w1)
        return X

    def design(self, pos, sim, batch=8):
        """X (B, n_par) from site indices pos (B, N, 2): X = P W, P_{b,k} = weight_k |rho_hat_k|^2 / (2 N_s^2), streamed."""
        out = []
        for b in range(0, pos.shape[0], batch):
            P2 = torch.fft.rfft2(sim.density(pos[b:b + batch])).abs() ** 2 * self.weight / (2 * self.N_s ** 2)
            X = self._apply_W(P2.reshape(-1, self.n_k))
            if self.lam_T is not None: X = torch.cat([X, (P2 * self.lam_T).sum(dim=(1, 2)).unsqueeze(1)], 1)
            out.append(X)
        return torch.cat(out)

    def mean_structure_factor(self, pos, sim, batch=8):
        """<S(k)> = <|rho_hat_k|^2>/N on the full (N_x, N_y) grid, fftshifted, S(0) := 0."""
        acc = torch.zeros(self.N_x, self.N_y)
        for b in range(0, pos.shape[0], batch):
            acc += (torch.fft.fft2(sim.density(pos[b:b + batch])).abs() ** 2).sum(0)
        S = acc / (pos.shape[0] * sim.N); S[0, 0] = 0.0
        return torch.fft.fftshift(S)

    def lam_from_theta(self, theta):
        """lam on the rfft2 half grid from theta = (mu at the knots[, mu_inf])."""
        th = theta[:-1] if self.lam_T is not None else theta
        lam = (self.w0 * th[self.i0] + self.w1 * th[self.i1]).view(self.N_x, self.N_y // 2 + 1)
        return lam + theta[-1] * self.lam_T if self.lam_T is not None else lam

    def theta_exact(self):
        """theta of the exact point-charge kernel: mu = 2 pi at every knot and mu_inf = 2 pi, in the units of the
        regression, theta = (N_s^2/A) mu (so that lam_q = theta_s / q inside the zone)."""
        return torch.full((self.n_par,), 2 * math.pi * self.N_s ** 2 / self.A)

    def V_knots(self, theta):
        """Continuum kernel V(q) = mu(q)/q at the knots (physical units), from theta = (N_s^2/A) mu."""
        th = theta[:-1] if self.lam_T is not None else theta
        return th * self.A / self.N_s ** 2 / self.q_knots[: len(th)]

    def full_sum(self, lam_half):
        """sum over the full grid of a half-grid quantity (rfft2 multiplicities)."""
        return float((lam_half * self.weight).sum())

    def align(self, lam_half):
        """Gauge: lam_0 = 0 and sum_{q != 0} lam_q = 0 (K_0 = 0)."""
        out = lam_half - (self.full_sum(lam_half) - float(lam_half[0, 0])) / (self.N_s - 1); out[0, 0] = 0.0
        return out

    def ring_average(self, lam_half, edges):
        """Mean of a half-grid quantity over |q| in the bins given by edges (for plotting lam(|q|))."""
        q = self.q.reshape(-1); v = (lam_half * self.weight).reshape(-1); w = self.weight.reshape(-1)
        idx = torch.bucketize(q, edges) - 1; ok = (idx >= 0) & (idx < len(edges) - 1) & (q > 0)
        num = torch.zeros(len(edges) - 1).index_add_(0, idx[ok], v[ok]); den = torch.zeros(len(edges) - 1).index_add_(0, idx[ok], w[ok])
        return num / den.clamp(min=1e-300), den > 0

    def lam_knots(self, theta):
        """m = 0 part of lam(|q|) at the knots, mu(q_s)/q_s (the alias amplitude is theta[-1])."""
        th = theta[:-1] if self.lam_T is not None else theta
        return th / self.q_knots[: len(th)]

    def kernel_x(self, theta):
        """Real-space kernel K(r, 0) along the x axis, r = 1..N_x/2 sites, from basis values."""
        K = torch.fft.irfft2(self.lam_from_theta(theta).to(torch.complex128), s=(self.N_x, self.N_y))
        return K[1: self.N_x // 2 + 1, 0]


# With a fixed number of particles sum_q |rho_q|^2 = N N_s is the same for every sample, so theta -> theta + c q_knots
# (a uniform shift of lam, the K_0 convention) is invisible: that direction is a constant column of the design and
# drops out of the centred least-squares problem (min-norm solution).  Comparisons of kernels are made after align().

def align(lam_half, basis):
    """Gauge: lam_0 = 0 and sum_{q != 0} lam_q = 0 (K_0 = 0); see RingBasis.align."""
    return basis.align(lam_half)


def fit_kernel(X, E):
    """Least squares E = X theta + c in closed form (centred, column-scaled, min-norm in the gauge direction)."""
    Xm, Em = X.mean(0), E.mean()
    sc = (X - Xm).norm(dim=0).clamp(min=1e-300)
    th = torch.linalg.lstsq((X - Xm) / sc, (E - Em).unsqueeze(1), driver="gelsd").solution.squeeze() / sc
    return th, float(Em - Xm @ th)


def second_moment(X):
    """
    Second moment of the knot-resolved structure factor across samples, Cov(X_s, X_s'), as the correlation
    matrix of the design columns (scale-free).  Returns its eigenvalues (descending, normalised to the largest).
    """
    Xc = X - X.mean(0)
    s = torch.linalg.svdvals(Xc / Xc.norm(dim=0).clamp(min=1e-300))
    return (s / s[0]) ** 2                                               # one eigenvalue is ~0: the gauge direction
