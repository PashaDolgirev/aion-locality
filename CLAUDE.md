# Project: aion-locality

Physics-motivated neural networks ("EwaldNN") that learn the locality structure of energy functionals
`E[ρ]` from datasets of density profiles and their energies. The repo accompanies the paper
*"Learning locality and emergence in many-body physical systems"* — see `Notes/notes_1D.pdf` and `Notes/notes_3D.pdf`
for the full mathematical setup.

**Current status**: 1D (`ewaldnn1d/`) and 2D (`ewaldnn2d/`) are complete. The active goal is to
**work out the 3D case** (build `ewaldnn3d/` and supporting notebooks).

## Mathematical setup (read carefully first `Notes/notes_3D.pdf`)
Several model classes appear:
- **LERN** Feature vector `x_r` is *static* (precomputed
  from local density and gradients). `E_a(r)` are known precomputed energy contributions; the
  network learns local reweighting factors `f_a`.
- **DM21** Same as LERN but feature vector may include nonlocal precomputed pieces. Still static. Does not require new architecture.
- **EwaldNN** — feature vector is *adaptive*: it includes `φ_r = (K * ρ)_r` whose kernel is
  learned. Energy includes `(1/2) Σ_r φ_qs(r) ρ_r` where `qs` (screening momentum) is
  a learned scalar.
- **EwaldNN extended** — `qs(x_r)` is a spatially varying field, implemented via soft-max.



## Conventions and gotchas
- **Boundary conditions**: The primary goal is to apply the 3D codebase to molecules -> use open boundary conditions which can be implemented via zero padding.
- **Feature concatenation pattern (LERN)**: in `LERN2d.forward`, the input tensor packs
  *normalized features* in `features[..., :N_feat]` and *unnormalized energy contributions*
  `E_a(r)` in `features[..., N_feat:]`. Notebooks build this by appending `E_loc_HF_*` twice:
  once before normalization, once after (the second copy is the unnormalized energy term).
  Preserve this convention when extending.
- **Grid**: Assume homogeneous grid with `N_x`, `N_y`, `N_z` being powers of two for FFT. Typically `N_x = N_y = N_y = 128`, which might limit memory
  consider GPU and adjust `N_batch`.


## Working style preferences (from this repo's notebooks)
- Don't introduce new dependencies casually. Current stack: `torch`, `torch_dct`, `numpy`,
  `pandas`, `matplotlib`.
- The code style should be consistent throughtout the codebase. Conciseness is preferred. Clarity is critical. 
- Write test cases which might be inefficient to test implementation.
