# ewaldnn1d/__init__.py

# --- Features ---
from .feat_utils import (
    sample_density,
    sample_density_batch,
    normalize_features,
    compute_normalization_stats,
    generate_loc_features_rs,
    generate_loc_features_ms,
)

# --- Learnable kernels (PyTorch modules) ---
from .linear_kernels import (
    LearnableKernelConv1d,
    GaussianMixtureKernelConv1d,
    LearnableRSNonLocalKernelDCT,
    ExpMixtureRSNonLocalKernelDCT,
    LearnableMSNonLocalKernelDCT,
)

# --- DCT utilities ---
from .dct_utils import (
    kernel_eigenvals_dct,
    kernel_from_eigenvals_dct,
    rho_to_cosine_coeffs,
    cosine_coeffs_to_rho,
)

# --- Analytic kernels and energy routines ---
from .energies_utils import (
    K_gaussian,
    K_exp,
    K_yukawa,
    K_power,
    E_int_conv,
    E_int_dct,
    E_int_dct_v2,
)

# --- Energy neural networks ---
from .linear_models import (
    KernelOnlyEnergyNN,
    GaussianMixtureEnergyNN,
    DCTKernelEnergyNN,
    HybridKernelEnergyNN,
)

# --- Training utilities ---
from .training_utils import (
    evaluate,
    load_checkpoint,
    _run_epoch,
    train_with_early_stopping,
)

__all__ = [
    # features
    "sample_density",
    "sample_density_batch",
    "normalize_features",
    "compute_normalization_stats",
    "generate_loc_features_rs",
    "generate_loc_features_ms",

    # kernels
    "LearnableKernelConv1d",
    "GaussianMixtureKernelConv1d",
    "LearnableRSNonLocalKernelDCT",
    "ExpMixtureRSNonLocalKernelDCT",
    "LearnableMSNonLocalKernelDCT",

    # models
    "KernelOnlyEnergyNN",
    "GaussianMixtureEnergyNN",
    "DCTKernelEnergyNN",
    "HybridKernelEnergyNN",

    # dct utils
    "rho_to_cosine_coeffs",
    "cosine_coeffs_to_rho",
    "kernel_eigenvals_dct",
    "kernel_from_eigenvals_dct",

    # analytic kernels + energies
    "K_gaussian",
    "K_exp",
    "K_yukawa",
    "K_power",
    "E_int_conv",
    "E_int_dct",
    "E_int_dct_v2",

    # training utils
    "evaluate",
    "load_checkpoint",
    "train_with_early_stopping",
    "_run_epoch",
]