# generate_cdw.py -- Monte Carlo data for the checkerboard (CDW) experiment: half-filled square-lattice Coulomb gas
# with nearest-neighbour repulsion V1 = 1, L = 32 (N = 512), across the ordering transition.
#
#   python checkerboard/generate_cdw.py             all temperatures (resumable: one file per Gamma, existing files are kept)
#   python checkerboard/generate_cdw.py 2.0 2.5     selected values of Gamma = 1/T
#
# Output: checkerboard/data/G<Gamma>.pt with keys Gamma, L, V1, chains, pos (n_saved, chains, N, 2) int16 site indices,
# E (n_saved, chains) exact energies.  Analysis: analyze_cdw.ipynb.  Data are not committed.
#
# Production settings, identical at every temperature: 32 chains, 1000 equilibration + 4000 production sweeps, one
# configuration every 10 sweeps (400 per chain, 12 800 per T).  The analysis discards the first quarter of every chain,
# leaving 9 600 configurations per T (the last 4 chains of which are the validation set), well above the 151 unknowns.

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from checkerboard.lattice_cdw import LatticeCDW, integrated_autocorr_time

torch.set_default_dtype(torch.float64)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

L, V1, SAVE_EVERY = 32, 1.0, 10
GAMMAS = [0.5, 0.8, 0.95, 1.0, 1.2, 1.4, 1.45, 1.5, 1.55, 1.6, 2.0, 2.5, 2.8, 3.0, 3.2, 3.5, 3.56, 5.0, 10.0]   
CHAINS, N_EQUIL, N_PROD = 32, 1000, 4000


def generate(gammas):
    os.makedirs(DATA, exist_ok=True)
    for G in gammas:
        f = os.path.join(DATA, f"G{G:g}.pt")
        if os.path.exists(f): print(f, "exists"); continue
        t0 = time.time()
        sim = LatticeCDW(L, V1, float(G), n_chains=CHAINS, seed=100 + round(100 * G), init="checkerboard")   # round(100 G): distinct for every Gamma in GAMMAS 
        pos, E = sim.run(N_EQUIL, N_PROD, SAVE_EVERY)
        torch.save(dict(Gamma=float(G), L=L, V1=V1, pos=pos.to(torch.int16), E=E, chains=CHAINS), f)
        tau = max(integrated_autocorr_time(E[:, c]) for c in range(CHAINS)) * SAVE_EVERY
        m = torch.fft.fft2(sim.density(pos[-1]))[:, L // 2, L // 2].abs().mean() / sim.N
        print(f"  tau_E ~ {tau:.0f} sweeps; |rho_Q|/N = {float(m):.3f}; NN pairs/N = {float(sim.n_neighbour_pairs(pos[-1]).mean()) / sim.N:.4f}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    generate([float(g) for g in sys.argv[1:]] or GAMMAS)
