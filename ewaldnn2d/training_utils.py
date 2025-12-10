import os
import math
import copy
import torch
import torch.nn as nn
import csv


def evaluate(model, loader, criterion, device="cpu"):
    """
    Simple evaluation loop.

    Args:
        model: nn.Module
        loader: DataLoader
        criterion: loss function
        device: torch device

    Returns:
        avg_loss: float
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)


def load_checkpoint(path, model_class, device="cpu"):
    """
    Loads a saved model checkpoint.

    Args:
        path: path to .pt file
        model_class: class object (e.g. RSKernelOnlyEnergyNN)
        device: "cpu" or "cuda"

    Returns:
        model          : reconstructed model on `device`
        normalization  : dict with mean/std/E stats (or None)
        epoch          : best epoch saved
        val_loss       : best validation loss
    """
    ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]
    normalization = ckpt.get("normalization", None)
    state_dict = ckpt["model_state_dict"]

    # --- reconstruct model with normalization if present ---
    if normalization is not None:
        mean_feat = normalization["mean_feat"].to(device)
        std_feat  = normalization["std_feat"].to(device)
        E_mean    = normalization["E_mean"].to(device)
        E_std     = normalization["E_std"].to(device)

        model = model_class(
            **config,
            mean_feat=mean_feat,
            std_feat=std_feat,
            E_mean=E_mean,
            E_std=E_std,
        ).to(device)
    else:
        model = model_class(**config).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    epoch = ckpt.get("epoch", None)
    val_loss = ckpt.get("val_loss", None)
    return model, normalization, epoch, val_loss


def _run_epoch(model,
               loader,
               criterion,
               optimizer,
               device="cpu",
               train: bool = True):
    """
    One training or validation epoch.

    Args:
        model: nn.Module
        loader: DataLoader
        criterion: loss function
        optimizer: torch optimizer (used only if train=True)
        device: torch device
        train: bool, whether to train or just eval

    Returns:
        avg_loss: float
    """
    if train:
        model.train()
    else:
        model.eval()

    running = 0.0
    n_batches = 0

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if train:
                optimizer.zero_grad()

            y_pred = model(xb)
            loss = criterion(y_pred, yb)

            if train:
                loss.backward()
                optimizer.step()

            running += loss.item()
            n_batches += 1

    return running / max(1, n_batches)


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    max_epochs: int = 10000,
    patience: int = 10,
    min_delta: float = 1e-5,
    ckpt_dir: str = "checkpoints",
    run_name: str | None = None,
    learning_regime: str = "rs_window",
    N_x: int | None = None,
    N_y: int | None = None,
    device: str = "cpu",
):
    """
    Generic training loop with early stopping + checkpoint saving.

    learning_regime (linear models):
        "rs_window"         -> RSKernelOnlyEnergyNN
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    if N_x is None or N_y is None:
        raise ValueError("Both N_x and N_y must be provided to train_with_early_stopping.")

    best_val = math.inf
    best_state = None
    best_epoch = -1
    since_improved = 0

    hist = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss   = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        improved = (best_val - val_loss) > min_delta
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            since_improved = 0

            # -------- build config dict for reconstructing the model ----------
            if learning_regime == "rs_window":
                config = {
                    "R": model.R,
                    "pad_mode": model.kernel_conv.pad_mode,
                }
            else:
                raise ValueError(f"Unknown learning_regime: {learning_regime}")

            # -------- normalization: read from model buffers if present ----------
            normalization = None
            if (
                hasattr(model, "mean_feat") and model.mean_feat is not None
                and hasattr(model, "std_feat") and model.std_feat is not None
                and hasattr(model, "E_mean") and model.E_mean is not None
                and hasattr(model, "E_std") and model.E_std is not None
            ):
                normalization = {
                    "mean_feat": model.mean_feat.detach().cpu(),
                    "std_feat":  model.std_feat.detach().cpu(),
                    "E_mean":    model.E_mean.detach().cpu(),
                    "E_std":     model.E_std.detach().cpu(),
                    "N_x":    int(N_x),
                    "N_y":    int(N_y),
                }

            ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": best_epoch,
                    "val_loss": best_val,
                    "config": config,
                    "normalization": normalization,
                },
                ckpt_path,
            )

        else:
            since_improved += 1

        if (epoch % 10) == 0 or epoch == 1:
            print(
                f"[{epoch:04d}] train={train_loss:.6f} | val={val_loss:.6f} "
                f"| best_val={best_val:.6f} (epoch {best_epoch})"
            )

        if since_improved >= patience:
            print(f"Early stopping at epoch {epoch} (best @ {best_epoch}).")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # write CSV if run_name is given
    if run_name is not None:
        csv_path = os.path.join(ckpt_dir, f"{run_name}_history.csv")
        try:
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss", "val_loss"])
                for i, (tr, va) in enumerate(zip(hist["train_loss"], hist["val_loss"]), start=1):
                    w.writerow([i, tr, va])
        except Exception as e:
            print(f"[warn] could not write CSV: {e}")

    return hist, best_epoch