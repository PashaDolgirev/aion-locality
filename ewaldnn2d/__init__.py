# --- Features ---
from .feat_utils import (
    sample_density,
    sample_density_batch,
    compute_normalization_stats,
    normalize_features,
    generate_loc_features_rs,
    generate_loc_features_ms,
    generate_data_2d,
    generate_SC_data_2d,
    extend_features_neighbors_2d,
    E_kin_custom,
)


# --- DCT utilities ---
from .dct_utils import (
    rho_to_cosine_coeffs,
    cosine_coeffs_to_rho,
    kernel_eigenvals_transform,
    kernel_from_eigenvals_transform,
    kernel_eigenvals_dct,
    kernel_from_eigenvals_dct,
)


# --- Analytic kernels and energy routines ---
from .energies_utils import (
    K_gaussian,
    K_exp,
    K_yukawa,
    K_power,
    Lam_K_Coulomb,
    E_int_conv,
    E_int_rs_dct,
    E_int_ms_dct,
)


# --- Linear convolutional kernels ---
from .linear_kernels import (
    LearnableRSKernelConv2d,
    LearnableRSNonLocalKernelDCT,
    LearnableMSNonLocalKernelDCT,
    ExpMixtureRSNonLocalKernelDCT,
    ScreenedCoulombNonLocalKernelDCT,
)


# --- Linear energy models ---
from .linear_energy_models import (
    RSKernelOnlyEnergyNN,
    DCTKernelEnergyNN,
)


# --- Neural network energy models ---
from .nn_energy_models import (
    LocalNN2d,
    LERN2d,
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
    "compute_normalization_stats",
    "normalize_features",
    "generate_loc_features_rs",
    "generate_loc_features_ms",
    "generate_data_2d",
    "generate_SC_data_2d",
    "extend_features_neighbors_2d",
    "E_kin_custom",

    # DCT utils
    "rho_to_cosine_coeffs",
    "cosine_coeffs_to_rho",
    "kernel_eigenvals_transform",
    "kernel_from_eigenvals_transform",
    "kernel_eigenvals_dct",
    "kernel_from_eigenvals_dct",

    # analytic kernels + energies
    "K_gaussian",
    "K_exp",
    "K_yukawa",
    "K_power",
    "Lam_K_Coulomb",
    "E_int_conv",
    "E_int_rs_dct",
    "E_int_ms_dct",

    # linear kernels
    "LearnableRSKernelConv2d",
    "LearnableRSNonLocalKernelDCT",
    "LearnableMSNonLocalKernelDCT",
    "ExpMixtureRSNonLocalKernelDCT",
    "ScreenedCoulombNonLocalKernelDCT",

    # linear energy models
    "RSKernelOnlyEnergyNN",
    "DCTKernelEnergyNN",

    # training utils
    "evaluate",
    "load_checkpoint",
    "train_with_early_stopping",
    "_run_epoch",

    # nn energy models
    "LocalNN2d",
    "LERN2d",
]