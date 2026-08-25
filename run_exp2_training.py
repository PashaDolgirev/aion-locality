# run_exp2_training.py
#
# Sequential training driver for 2D Experiment 2 (smooth regime).
# Usage:  python run_exp2_training.py {LERN|DM21|EwaldNN} [--smoke]
#
# Runs 12 rounds per model: beta in {-0.002, 0.0} x qs in {0.0, 0.1, 0.5} x
# R_feat in {1.0, 1.5}; each round is the 4 x 4 x 3 (n_hidden x n_neurons x seeds)
# hyperparameter grid of the LearnEngFunc2d_*.ipynb notebooks, with identical
# datasets, seeds, run names, and checkpoint locations - the notebooks can be
# used for analysis afterwards. Rounds whose history csv already exists are
# skipped, so the script resumes cleanly after an interruption.
#
# Uses at most half the machine's cores (torch.set_num_threads).

import sys, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ewaldnn2d import *

MODELS = {
    "LERN":    dict(prefix="LERN2d_EngFunc_",     regime="LERN2d",    cls=LERN2d,    n_terms=2, hf_feature=False),
    "DM21":    dict(prefix="LERN2d_mod_EngFunc_", regime="LERN2d",    cls=LERN2d,    n_terms=2, hf_feature=True),
    "EwaldNN": dict(prefix="EwaldNN2d_EngFunc_",  regime="EwaldNN2d", cls=EwaldNN2d, n_terms=1, hf_feature=False),
}


def main():
    model_name = sys.argv[1]
    smoke = "--smoke" in sys.argv
    spec = MODELS[model_name]

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # leave half the cores free

    # ---- settings identical to LearnEngFunc2d_*.ipynb ----
    dtype = torch.float64
    device = "cpu"
    N_batch = 100
    N_epochs = 10000
    lr = 1e-1
    min_delta = 1e-5
    patience = 100
    N_pow = 1
    N_train, N_test, N_val = 1500, 250, 250
    N_x = N_y = 32
    data_regime = "smooth"
    kernel_regime = "screened_coulomb"
    alpha = 0.005
    amp = 1.0
    ckpt_dir = "LearningEngFunc2d_checkpoints"
    n_hidden_list = [1, 2, 3, 4]
    n_neurons_list = [8, 16, 32, 64]
    n_seeds = 3
    beta_list = [-0.002, 0.0]
    qs_list = [0.0, 0.1, 0.5]
    R_feat_list = [1.0, 1.5]

    if smoke:  # tiny end-to-end check in an isolated directory
        smoke_dir = os.environ.get("EXP2_SMOKE_DIR", "/tmp/exp2_smoke")
        os.makedirs(smoke_dir, exist_ok=True)
        os.chdir(smoke_dir)
        N_train, N_test, N_val = 40, 10, 10
        N_epochs = 2
        n_hidden_list, n_neurons_list, n_seeds = [1], [8], 1
        beta_list, qs_list, R_feat_list = [-0.002], [0.1], [1.0, 1.5]

    # grid and basis settings
    m_x = torch.arange(0, N_x, dtype=dtype, device=device)
    m_y = torch.arange(0, N_y, dtype=dtype, device=device)
    abs_val = torch.sqrt(m_x[:, None]**2 + m_y[None, :]**2)
    x = torch.linspace(0, 1, N_x, dtype=dtype, device=device)
    y = torch.linspace(0, 1, N_y, dtype=dtype, device=device)
    DM_x = torch.cos(torch.pi * torch.outer(m_x, x))
    DM_y = torch.cos(torch.pi * torch.outer(m_y, y))
    DerDM_x = -torch.pi * m_x[:, None] * torch.sin(torch.pi * torch.outer(m_x, x))
    DerDM_y = -torch.pi * m_y[:, None] * torch.sin(torch.pi * torch.outer(m_y, y))

    M_cutoff = 10
    std_harm = 2.0 / (1.0 + 0.2 * abs_val)**2 * (abs_val <= M_cutoff).double()
    std_harm[0, 0] = 0.0

    def get_dataset(beta, qs):
        """Generate (if missing) or load the shared (beta, qs) dataset, as in the notebooks."""
        fname = f"DATA2d/EngFunc_dataset_{data_regime}_{kernel_regime}_{alpha}_{beta}_{qs}_{amp}_{N_x}_{N_y}.pt"

        def E_kin_loc(rho, d_rho_x, d_rho_y, eng_dens_flag=False):
            return E_kin_custom(rho, d_rho_x, d_rho_y, alpha=0.0, beta=0.0, qs=qs, eng_dens_flag=eng_dens_flag)

        def E_HF_loc(rho, d_rho_x, d_rho_y, eng_dens_flag=False):
            return amp * E_int_ms_dct(rho, kernel=kernel_regime, eng_dens_flag=eng_dens_flag, qs=0.0)

        def E_tot(rho, d_rho_x, d_rho_y):
            return E_kin_custom(rho, d_rho_x, d_rho_y, alpha=alpha, beta=beta, qs=qs) \
                 + amp * E_int_ms_dct(rho, kernel=kernel_regime, qs=qs)

        if not os.path.isfile(fname):
            print(f"[data] generating {fname}", flush=True)
            N_batch_int = 10
            torch.manual_seed(1234)
            splits = {}
            for split, N in [("train", N_train), ("test", N_test), ("val", N_val)]:
                rho, d_rho_x, d_rho_y, a, E_loc_kin, E_loc_HF, E_tot_s = generate_EngFunc_data_2d(
                    N, N_batch_int, E_kin_loc, E_HF_loc, E_tot,
                    std_harm=std_harm, DM_x=DM_x, DerDM_x=DerDM_x, DM_y=DM_y, DerDM_y=DerDM_y)
                splits.update({
                    f"rho_{split}": rho, f"d_rho_x_{split}": d_rho_x, f"d_rho_y_{split}": d_rho_y,
                    f"a_{split}": a, f"E_loc_kin_{split}": E_loc_kin, f"E_loc_HF_{split}": E_loc_HF,
                    f"E_tot_{split}": E_tot_s,
                })
            os.makedirs("DATA2d", exist_ok=True)
            torch.save({**splits, "data_regime": data_regime, "kernel_regime": kernel_regime,
                        "alpha": alpha, "beta": beta, "qs": qs, "amp": amp}, fname)
            return splits
        print(f"[data] loading {fname}", flush=True)
        return torch.load(fname)

    def make_loaders(data, R_feat):
        """Features + loaders for one round; returns (loaders, N_feat, normalization stats)."""
        feats = {}
        for split in ("train", "test", "val"):
            f = extend_features_neighbors_2d(
                generate_loc_features_rs(data[f"rho_{split}"], N_pow=N_pow), R=R_feat)
            if spec["hf_feature"]:
                f = torch.cat([f, data[f"E_loc_HF_{split}"]], dim=-1)
            feats[split] = f
        N_feat = feats["train"].shape[-1]

        mean_feat, std_feat = compute_normalization_stats(feats["train"])
        E_mean = data["E_tot_train"].mean()
        E_std = data["E_tot_train"].std()

        loaders = {}
        for split in ("train", "test", "val"):
            f = normalize_features(feats[split], mean_feat, std_feat)
            terms = [data[f"E_loc_kin_{split}"]]
            if spec["n_terms"] == 2:
                terms.append(data[f"E_loc_HF_{split}"])
            f = torch.cat([f] + terms, dim=-1)
            targets = (data[f"E_tot_{split}"] - E_mean) / E_std
            loaders[split] = DataLoader(TensorDataset(f, targets),
                                        batch_size=N_batch, shuffle=(split == "train"), drop_last=False)
        return loaders, N_feat, mean_feat, std_feat, E_mean, E_std

    N_energy_terms = spec["n_terms"]
    rounds = [(beta, qs, R_feat) for beta in beta_list for qs in qs_list for R_feat in R_feat_list]
    t0 = time.time()
    print(f"===== {model_name}: {len(rounds)} rounds x "
          f"{len(n_hidden_list) * len(n_neurons_list) * n_seeds} trainings "
          f"({torch.get_num_threads()} torch threads) =====", flush=True)

    for i_round, (beta, qs, R_feat) in enumerate(rounds, start=1):
        data = get_dataset(beta, qs)
        loaders, N_feat, mean_feat, std_feat, E_mean, E_std = make_loaders(data, R_feat)
        print(f"\n===== {model_name} ROUND {i_round}/{len(rounds)}: "
              f"beta={beta}, qs={qs}, R_feat={R_feat} (N_feat={N_feat}) "
              f"[{(time.time() - t0) / 3600:.2f} h elapsed] =====", flush=True)

        best_val, best_cfg = float("inf"), None
        for n_hidden in n_hidden_list:
            for n_neurons in n_neurons_list:
                for seed in range(n_seeds):
                    run_name = spec["prefix"] + data_regime + '_' + kernel_regime + \
                        f"_{alpha}_{beta}_{qs}_{amp}_{N_x}_{N_y}_{N_feat}_{N_energy_terms}_{n_hidden}_{n_neurons}_{seed}"
                    if os.path.isfile(os.path.join(ckpt_dir, f"{run_name}_history.csv")):
                        print(f"[skip] {run_name} (already trained)", flush=True)
                        continue

                    torch.manual_seed(1234 + seed)
                    model = spec["cls"](
                        N_x=N_x, N_y=N_y,
                        N_energy_terms=N_energy_terms, N_feat=N_feat,
                        n_hidden=n_hidden, n_neurons=n_neurons,
                        mean_feat=mean_feat, std_feat=std_feat,
                        E_mean=E_mean, E_std=E_std,
                    ).to(device=device, dtype=dtype)

                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=50, cooldown=2, min_lr=1e-6)

                    t_run = time.time()
                    hist, best_epoch = train_with_early_stopping(
                        model=model, train_loader=loaders["train"], val_loader=loaders["val"],
                        criterion=nn.MSELoss(), optimizer=optimizer, scheduler=scheduler,
                        max_epochs=N_epochs, patience=patience, min_delta=min_delta,
                        ckpt_dir=ckpt_dir, run_name=run_name, learning_regime=spec["regime"],
                        N_x=N_x, N_y=N_y, device=device,
                    )
                    val = min(hist["val_loss"])
                    if val < best_val:
                        best_val, best_cfg = val, (n_hidden, n_neurons, seed)
                    print(f"[done] {run_name}: best val {val:.3e} @ epoch {best_epoch} "
                          f"({time.time() - t_run:.0f} s)", flush=True)

        print(f"[round best] {model_name} beta={beta} qs={qs} R_feat={R_feat}: "
              f"val {best_val:.3e} (n_hidden, n_neurons, seed) = {best_cfg}", flush=True)

    print(f"\n===== {model_name} COMPLETE in {(time.time() - t0) / 3600:.2f} h =====", flush=True)


if __name__ == "__main__":
    main()
