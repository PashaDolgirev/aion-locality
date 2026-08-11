# Project: aion-locality

Physics-motivated neural networks ("EwaldNN") that learn the locality structure of energy functionals
`E[ρ]` from datasets of density profiles and their energies. The repo accompanies the paper
*"Learning locality and emergence in many-body physical systems"* — see `Notes/EwaldNN.pdf`
for the full mathematical setup.

**Current status**: 1D (`ewaldnn1d/`) and 2D (`ewaldnn2d/`) are complete. The active goal is to
**work out the 3D case** (build `ewaldnn3d/` and supporting notebooks).

## Mathematical setup (read first)

The inductive bias is that the energy functional is a spatial average of a *local* density
`E[ρ] = (1/V) Σ_r E_loc(x_r)` where `x_r` is a feature vector at site `r`. For nonlocal
interactions, locality is restored via a **mediator field** `φ = K * ρ` learned in momentum space
(diagonalized by DCT-I under von Neumann / reflection BCs).

Three model classes appear (Sec. III of `Notes/EwaldNN.pdf`):

- **LERN** — `E = δV Σ_r Σ_a E_a(r) f_a(x_r)`. Feature vector `x_r` is *static* (precomputed
  from local density and gradients). `E_a(r)` are known precomputed energy contributions; the
  network learns local reweighting factors `f_a`.
- **DM21** — same as LERN but feature vector may include nonlocal precomputed pieces (e.g., HF
  energy density). Still static.
- **EwaldNN** — feature vector is *adaptive*: it includes `φ_r = (K * ρ)_r` whose kernel is
  learned. Energy includes `(1/2) δV Σ_r φ_qs(r) ρ_r f_φ(x_r)` where `qs` (screening momentum) is
  a learned scalar.
- **EwaldNN extended** — same as EwaldNN, but `qs(x_r)` is a spatially varying MLP output.

Two canonical data regimes (drives learnability):
- **smooth**: only low-momentum harmonics excited → second-moment correlation matrix is
  low-rank, kernel is predictive but not unique.
- **rough**: all harmonics excited → full kernel reconstruction is well-conditioned.

## Directory layout

```
ewaldnn1d/                  reference 1D implementation
ewaldnn2d/                  current 2D implementation (mirror of 1D structure)
ewaldnn3d/                  TO BUILD — mirror the 2D structure
Notes/EwaldNN.pdf           SI write-up; Sec. II is 2D, Sec. III is 3D (sparse, finish here)
Notes/DM21.pdf              reference for DM21-style DFT functionals
DATA2d/                     cached .pt datasets, e.g. LERN_dataset_<regime>_<kernel>_<qs>_<amp>_<Nx>_<Ny>.pt
LearningSC2d_checkpoints/   saved 2D models (run_name → .pt + history.csv)
Learn*.ipynb                top-level training / analysis notebooks
Figure_*/                   per-figure data dumps for the paper
```

The `ewaldnn{1d,2d}` packages have an identical six-file structure:

| file | role |
|---|---|
| `feat_utils.py` | density sampling (Gaussian-weighted cosine series), feature generation (`_rs`/`_ms`), neighbor extension, normalization stats |
| `dct_utils.py` | DCT-I forward/inverse for `ρ ↔ a_mn`; kernel `K_r ↔ λ_m` eigenvalue transforms (Neumann BC) |
| `energies_utils.py` | analytic reference kernels (`K_gaussian`, `K_exp`, `K_yukawa`, `K_power`, in 2D also `Lam_K_Coulomb`) and reference `E_int_*` |
| `linear_kernels.py` | learnable `nn.Module` kernel parametrizations (RSCL, RS-DCT, MS-DCT, exp mixture, screened Coulomb) |
| `linear_energy_models.py` (2D) / `linear_models.py` (1D) | linear energy heads `E = ½ ρ·(K*ρ)/V` wrapping each kernel |
| `nn_energy_models.py` (2D only) | `LocalNN2d` (small MLP per grid point) and `LERN2d` |
| `training_utils.py` | `train_with_early_stopping`, `evaluate`, `load_checkpoint`, `_run_epoch` |

The 1D package additionally has `corr_funcs_utils.py` with analytical first/second moments
of the density-density correlation function (Sec. I C of the notes).

## Conventions and gotchas

- **Default dtype**: `torch.float64` everywhere. `device = "cpu"` in the notebooks.
- **Density sampling**: `ρ_j = Σ_m a_m cos(πm x_j)` (1D) or
  `ρ_ij = Σ_mn a_mn cos(πm x_i) cos(πn y_j)` (2D), with `x ∈ [0,1]`. This enforces von Neumann
  BCs (zero derivatives at boundary) and is consistent with DCT-I.
- **Always set `a[0,...,0] = 0`** (no uniform background) when constructing `std_harm`.
- **Boundary conditions**: reflection (`pad_mode="reflect"`) padding everywhere — this is what
  makes the DCT-I diagonalize the convolution. Do not silently switch to zero padding.
- **Kernel symmetry**: `K_r = K_{-r}` (1D); 2D extends to `K_{rx,ry} = K_{±rx,±ry}`. Only the
  first quadrant (`r ≥ 0`) is independent; `LearnableRSKernelConv2d` builds the full kernel from
  the upper quadrant via flips.
- **DCT-I eigenvalue formula**: `λ_m = DCT[K_r] + (-1)^m K_{N-1}` (note S34). The
  `kernel_eigenvals_*` and `kernel_from_eigenvals_*` pair invert each other; never replace one
  without the other.
- **Energy normalization**: `E_int = (1 / 2 V) Σ ρ (K * ρ)` where `V = N_x N_y` (2D) or `δV = dx dy dz`
  in 3D continuum form (eq. S38). Models stash `mean_feat`, `std_feat`, `E_mean`, `E_std` as
  *buffers* (not parameters) so they move with `.to(device)`.
- **Feature concatenation pattern (LERN)**: in `LERN2d.forward`, the input tensor packs
  *normalized features* in `features[..., :N_feat]` and *unnormalized energy contributions*
  `E_a(r)` in `features[..., N_feat:]`. Notebooks build this by appending `E_loc_HF_*` twice:
  once before normalization, once after (the second copy is the unnormalized energy term).
  Preserve this convention when extending.
- **Output activation**: `LocalNN2d` uses `1 + tanh(z)` so reweighting factors lie in `[0, 2]`.
- **`Lam_K_Coulomb` (2D)**: returns `2π / (q² + qs²)`, zeros the `(0,0)` mode, and explicitly
  round-trips through real space to kill the `K(r=0)` self-interaction. Replicate this
  round-trip in 3D — *do not* skip the self-interaction zeroing.
- **`zero_r_flag`** on RS kernels enforces `K(0)=0` for self-interaction-free interactions.

## Training pattern

Used in every notebook:

```python
optimizer = optim.Adam(model.parameters(), lr=1e-1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=50, cooldown=2, min_lr=1e-6)
hist, best_epoch = train_with_early_stopping(
    model, train_loader, val_loader, criterion=nn.MSELoss(),
    optimizer=optimizer, scheduler=scheduler,
    max_epochs=10000, patience=100, min_delta=1e-5,
    ckpt_dir=..., run_name=..., learning_regime=..., N_x=N_x, N_y=N_y, device=device)
```

`learning_regime` is a string keyed in `train_with_early_stopping` that selects how to pickle
the config dict. **Adding a new regime requires editing this `if/elif` chain.**

Checkpoints (`{run_name}_best.pt`) store `model_state_dict`, `config`, `normalization`, `epoch`,
`val_loss`. `load_checkpoint` reconstructs the model from these — always pair new model
classes with matching config keys.

Typical sweep: 3 seeds × `n_hidden ∈ {1,2,3,4}` × `n_neurons ∈ {8,16,32,64}`; pick best by val
loss.

## What to do for 3D (the active task)

Per Sec. III of `Notes/EwaldNN.pdf` ("Functional learning in 3D"), implement an `ewaldnn3d/`
package by mirroring `ewaldnn2d/` and lifting every spatial axis count from 2 to 3:

1. **`dct_utils.py`** — apply `dct.dct1` along each of the three spatial axes (3 transposes).
   `kernel_eigenvals_dct` and its inverse become 3D analogs of the existing 2D versions.
2. **`feat_utils.py`** — `sample_density` takes `(DM_x, DerDM_x, DM_y, DerDM_y, DM_z, DerDM_z)`;
   produces `rho, d_rho_x, d_rho_y, d_rho_z, a` of shapes `(B, N_x, N_y, N_z)` and
   `(B, M_x, M_y, M_z)`. `extend_features_neighbors_3d` should use the same R-ball logic
   but with 3D shifts — note this scales as `O(R³)`.
3. **`energies_utils.py`** — 3D screened Coulomb `Lam_K_Coulomb_3d(q, qs) = 4π / (q² + qs²)`
   with `q² = qx² + qy² + qz²` (vs 2π/(q²+qs²) in 2D). Same `(0,0,0)`-mode and `K(r=0)`
   zeroing pattern.
4. **`linear_kernels.py`** — `LearnableRSKernelConv3d` (3D conv with reflect pad),
   `LearnableRSNonLocalKernelDCT3d`, `LearnableMSNonLocalKernelDCT3d`,
   `ScreenedCoulombNonLocalKernelDCT3d`. Use `F.conv3d` and a 3D mask `rx²+ry²+rz² ≤ R²`.
5. **`nn_energy_models.py`** — `LERN3d` with the same `features[..., :N_feat]` + energy-terms
   layout, mean over `dim=(1,2,3)`.
6. **`linear_energy_models.py`** — 3D heads divide by `N_x N_y N_z`.
7. **`training_utils.py`** — `train_with_early_stopping` needs `N_z` added to the signature
   and to every config dict in the `learning_regime` chain. Add a new regime per 3D model
   class to keep checkpoints loadable.
8. **EwaldNN-extended** — when implementing the spatially-varying screening (eq. S41), the MLP
   for `qs(x_r)` has to remain nonnegative (use `softplus`, like the existing
   `ScreenedCoulombNonLocalKernelDCT.raw_qs`). The kernel field `φ(r; qs(x_r))` is no longer
   translation-invariant, so it cannot be computed by a single DCT — expect to evaluate it
   per-site or via a parametrized basis. Plan this carefully; the 2D code does *not* yet
   have this case and is not a template.

### Compute considerations
- Memory cost scales with `N³`. The 2D notebooks use `N_x = N_y = 32` (1024 sites). With
  `N_x=N_y=N_z=32` you have 32k sites — feasible but adjust `N_batch` and consider GPU.
- Generate datasets in mini-batches (`generate_data_2d` already does
  `(N + N_batch - 1) // N_batch`) — keep this pattern in 3D.
- Cache datasets as `DATA3d/LERN_dataset_<regime>_<kernel>_<qs>_<amp>_<Nx>_<Ny>_<Nz>.pt`.

### Sanity checks for the 3D port
1. `kernel_from_eigenvals_dct(kernel_eigenvals_dct(K)) == K` for random `K` (numerical
   tolerance).
2. `cosine_coeffs_to_rho(rho_to_cosine_coeffs(ρ)) == ρ`.
3. `E_int_conv` (real-space brute force) and `E_int_*_dct` (spectral) agree on the same
   density and kernel — this caught subtle BC bugs in the 2D code.
4. For the screened Coulomb 3D head, verify the analytic `Lam_K_Coulomb_3d` matches a
   direct real-space sum of `exp(-qs r)/r` after the round-trip self-interaction removal.

## Working style preferences (from this repo's notebooks)
- Reproducibility: `torch.manual_seed(1234 + seed)` at the top; seeds also encoded into
  `run_name` so checkpoints don't collide.
- Notebooks save figure-input arrays to `Figure_<n>/` (commented `np.savetxt(...)` lines).
  Don't delete those — they're how the paper figures are reproduced.
- Don't introduce new dependencies casually. Current stack: `torch`, `torch_dct`, `numpy`,
  `pandas`, `matplotlib`.
