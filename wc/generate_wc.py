# generate_wc.py -- Monte Carlo data for the Wigner-crystal experiment: the lattice one-component plasma
# (wc/lattice_wc.py, exact periodic Coulomb kernel, no regularisation) across the crystallisation transition.
#
#   python wc/generate_wc.py                 all couplings (resumable: one file per Gamma, existing files are kept)
#   python wc/generate_wc.py 73 274          selected values of Gamma = e^2/(a k_B T) (liquid / crystal side)
#
# Output: wc/data/G<Gamma>.pt with keys Gamma, n_x, n_y, m, chains, w (final move window), acc (acceptance rate),
# pos (n_saved, chains, N, 2) int16 site indices, E (n_saved, chains) exact energies, tau (integrated autocorrelation
# time of E in sweeps), delta (rms nearest-neighbour distance spread).  Analysis: analyze_wc.ipynb.  Data are not committed.
#
# Production settings, identical at every coupling and to the checkerboard experiment: 32 chains started from the
# perfect triangular lattice, 1000 equilibration + 4000 production sweeps, one configuration every 10 sweeps
# (400 per chain, 12 800 per Gamma).  The analysis discards the first quarter of every chain, leaving 9 600
# configurations per Gamma, the last 4 chains of which are the validation set.

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from wc.lattice_wc import LatticeWC, integrated_autocorr_time, neighbour_displacement, A_LAT

torch.set_default_dtype(torch.float64)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

N_X, N_Y, M = 19, 11, 40                  # box: 19 x 11 rectangular cells of the triangular crystal (N = 418); grid spacing s = a_lat/40
GAMMAS = [46, 73, 87, 110, 128, 146, 183, 229, 274, 326, 390, 457, 520, 640, 760, 915, 1150, 1600, 2300]
# 19 couplings, log-even in T/T_c = (Gamma_c ~ 115)/Gamma from 2.5 down to 0.05: a coarse liquid side,
# the transition bracketed by 110/128, and a dense ordered-side tail (the lever arm for the
# diffuse-weight ∝ T phonon law).  2300 is the floor for this grid: beyond it the rms displacement
# drops below one grid site (0.84 sites there) and the discreteness opens an artificial gap that would
# fake CDW-like activated behaviour; colder points would need a finer grid (larger m).
CHAINS, N_EQUIL, N_PROD, SAVE_EVERY = 32, 1000, 4000, 10


def generate(gammas):
    os.makedirs(DATA, exist_ok=True)
    for G in gammas:
        f = os.path.join(DATA, f"G{G:g}.pt")
        if os.path.exists(f): print(f, "exists"); continue
        t0 = time.time()
        sim = LatticeWC(N_X, N_Y, M, float(G), n_chains=CHAINS, seed=1000 + int(G), init="lattice")
        pos, E = sim.run(N_EQUIL, N_PROD, SAVE_EVERY)
        out = dict(Gamma=float(G), n_x=N_X, n_y=N_Y, m=M, chains=CHAINS, w=sim.w, acc=float(sim.n_acc.mean()) / max(sim.n_att, 1),
                   pos=pos.to(torch.int16), E=E, n_equil=N_EQUIL, n_prod=N_PROD, save_every=SAVE_EVERY)
        torch.save(out, f)                                                 # save first, diagnostics after
        tau = max(integrated_autocorr_time(E[:, c]) for c in range(CHAINS)) * SAVE_EVERY
        delta, d_nn = neighbour_displacement(pos[-1], sim)
        out.update(tau=tau, delta=delta, d_nn=d_nn); torch.save(out, f)
        print(f"  tau_E ~ {tau:.0f} sweeps;  rms NN-distance spread delta = {delta / A_LAT:.3f} a_lat (mean NN distance {d_nn:.4f});  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    generate([float(g) for g in sys.argv[1:]] or GAMMAS)
