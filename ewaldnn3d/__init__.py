# --- Features ---
from .feat_utils import (
    sample_density,
    sample_density_batch,
    sample_density_gaussians_batch,
    compute_normalization_stats,
    normalize_features,
    generate_loc_features_rs,
    generate_loc_features_ms,
    generate_data_3d,
    generate_SC_data_3d,
    extend_features_neighbors_3d,
    E_kin_custom,
)


# --- FFT utilities (open BCs via zero padding) ---
from .fft_utils import (
    kernel_wraparound_embed,
    kernel_eigenvals_fft,
    kernel_from_eigenvals_fft,
    conv_fft,
    displacement_grid,
    q_grid_fft,
)


# --- Analytic kernels and energy routines ---
from .energies_utils import (
    K_gaussian,
    K_exp,
    K_yukawa,
    K_power,
    Lam_K_Coulomb,
    E_int_conv,
    E_int_rs_fft,
    E_int_ms_fft,
)


# --- Linear convolutional kernels ---
from .linear_kernels import (
    LearnableRSKernelConv3d,
    LearnableRSNonLocalKernelFFT,
    LearnableMSNonLocalKernelFFT,
    ExpMixtureRSNonLocalKernelFFT,
    ScreenedCoulombRSNonLocalKernelFFT,
    ScreenedCoulombNonLocalKernelFFT,
)


# --- Linear energy models ---
from .linear_energy_models import (
    RSKernelOnlyEnergyNN,
    FFTKernelEnergyNN,
)


# --- Neural network energy models ---
from .nn_energy_models import (
    LocalNN3d,
    LERN3d,
    EwaldNN3d,
    EwaldNNExtended3d,
)


# --- Training utilities ---
from .training_utils import (
    evaluate,
    load_checkpoint,
    _run_epoch,
    train_with_early_stopping,
)


__all__ = [
    # Features
    "sample_density",
    "sample_density_batch",
    "sample_density_gaussians_batch",
    "compute_normalization_stats",
    "normalize_features",
    "generate_loc_features_rs",
    "generate_loc_features_ms",
    "generate_data_3d",
    "generate_SC_data_3d",
    "extend_features_neighbors_3d",
    "E_kin_custom",

    # FFT utils
    "kernel_wraparound_embed",
    "kernel_eigenvals_fft",
    "kernel_from_eigenvals_fft",
    "conv_fft",
    "displacement_grid",
    "q_grid_fft",

    # analytic kernels + energies
    "K_gaussian",
    "K_exp",
    "K_yukawa",
    "K_power",
    "Lam_K_Coulomb",
    "E_int_conv",
    "E_int_rs_fft",
    "E_int_ms_fft",

    # linear kernels
    "LearnableRSKernelConv3d",
    "LearnableRSNonLocalKernelFFT",
    "LearnableMSNonLocalKernelFFT",
    "ExpMixtureRSNonLocalKernelFFT",
    "ScreenedCoulombRSNonLocalKernelFFT",
    "ScreenedCoulombNonLocalKernelFFT",

    # linear energy models
    "RSKernelOnlyEnergyNN",
    "FFTKernelEnergyNN",

    # training utils
    "evaluate",
    "load_checkpoint",
    "train_with_early_stopping",
    "_run_epoch",

    # nn energy models
    "LocalNN3d",
    "LERN3d",
    "EwaldNN3d",
    "EwaldNNExtended3d",
]
