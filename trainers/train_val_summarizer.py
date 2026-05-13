"""Training and evaluation loop for the LaplaceAE summarizer model."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

from configs import config
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from configs.dataset_registry import resolve_run_experiment
from Model.summarizer import LaplaceAE
LoaderTuple = Tuple[DataLoader, DataLoader, DataLoader]


def _loss_weights(config_obj) -> Tuple[float, float, float, float, float]:
    return (
        float(getattr(config_obj, "SUM_LOSS_W_X", 1.0)),
        float(getattr(config_obj, "SUM_LOSS_W_V", 0.1)),
        float(getattr(config_obj, "SUM_LOSS_W_T", 0.1)),
        float(getattr(config_obj, "SUM_LOSS_W_DT", 0.0)),
        float(getattr(config_obj, "SUM_LOSS_W_OBS", 0.0)),
    )


def set_seed(seed: int = 42) -> None:
    """Seed all relevant RNGs for reproducible runs."""

    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def save_ckpt(path: Path, model: nn.Module, stats: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "stats": stats}, path)


def _ensure_loaders(
    train_loader: Optional[DataLoader],
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader],
    sizes: Optional[Sequence[int]],
    config=config,
) -> Tuple[LoaderTuple, Optional[Tuple[int, int, int]]]:
    if any(loader is None for loader in (train_loader, val_loader, test_loader)):
        run_experiment = resolve_run_experiment(config.DATA_DIR)
        train_loader, val_loader, test_loader, sizes = run_experiment(
            data_dir=config.DATA_DIR,
            date_batching=config.date_batching,
            dates_per_batch=int(config.DATES_PER_BATCH),
            K=config.WINDOW,
            H=config.PRED,
            coverage=config.COVERAGE,
            ratios=(config.train_ratio, config.val_ratio, config.test_ratio),
        )
    elif sizes is None:
        try:
            sizes = tuple(len(dl.dataset) for dl in (train_loader, val_loader, test_loader))
        except Exception:
            sizes = None

    if train_loader is None or val_loader is None or test_loader is None:
        raise RuntimeError("Failed to obtain train/val/test dataloaders.")

    return (train_loader, val_loader, test_loader), sizes


def _summarize_dataset(
    train_loader: DataLoader,
    sizes: Optional[Sequence[int]],
    *,
    verbose: bool = True,
) -> Tuple[int, int]:
    if verbose:
        if sizes is not None:
            print(f"sizes: {tuple(sizes)}")
        else:
            print("sizes: (unknown)")

    try:
        (xb, yb, meta) = next(iter(train_loader))
    except StopIteration as exc:  # pragma: no cover - defensive
        raise RuntimeError("Training dataloader produced no batches.") from exc

    V, T = xb
    _, num_entities, _, feat_dim = V.shape
    if verbose:
        print("V:", tuple(V.shape), "T:", tuple(T.shape), "y:", tuple(yb.shape))
    return num_entities, feat_dim


def _permute_to_seq_first(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def _nan_to_num(x: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(x).all():
        return x
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _entity_finite_mask(x: torch.Tensor) -> torch.Tensor:
    """Return per-entity mask marking sequences with all finite values."""

    finite = torch.isfinite(x)
    for _ in range(x.dim() - 2):
        finite = finite.all(dim=-1)
    return finite


def _apply_entity_mask(series: torch.Tensor, mask_bn: torch.Tensor) -> torch.Tensor:
    if mask_bn.dtype != torch.bool:
        mask_bn = mask_bn.to(dtype=torch.bool)
    if mask_bn.shape[0] != series.shape[0] or mask_bn.shape[1] != series.shape[2]:
        raise ValueError(
            f"Mask shape {tuple(mask_bn.shape)} incompatible with series shape {tuple(series.shape)}"
        )
    mask = mask_bn[:, None, :, None].to(device=series.device, dtype=series.dtype)
    return series * mask


def _batch_elements(mask: torch.Tensor, steps: int) -> float:
    return mask.float().sum().item() * float(steps)


def _prepare_batch(
        batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]],
        device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Prepare a batch for summarizer pretraining.

    Returns:
        V: values, shape [B,K,N,F]
        T: finite-difference proxy, shape [B,K,N,F]
        mask: entity mask [B,N]
        elems: scalar normalization weight for reporting
        dt: optional timestamps / deltas (normalized layout)
        obs_mask: optional observation mask (normalized layout)
    """
    (V, T), _, meta = batch

    # 1. Calculate finite masks on the original (Batch, Entity, Time, Feat) data.
    #    This ensures the output shape is (Batch, Entity) and catches NaNs before removal.
    v_finite = _entity_finite_mask(V).to(device)
    t_finite = _entity_finite_mask(T).to(device)

    mask = meta["entity_mask"].to(device=device, dtype=torch.bool)
    mask = mask & v_finite & t_finite

    # 2. Permute and sanitize (NaN -> 0.0)
    V = _nan_to_num(_permute_to_seq_first(V)).to(device)
    T = _nan_to_num(_permute_to_seq_first(T)).to(device)

    # 2b. Permute and sanitize timestamps and observation masks (paper-consistent temporal conditioning)
    dt = meta.get("delta_t")
    if dt is not None:
        dt = torch.as_tensor(dt, dtype=torch.float32, device=device)
        # common layout: [B, N, K] -> [B, K, N]
        if dt.dim() == 3 and dt.size(1) == mask.size(1):
            dt = dt.permute(0, 2, 1).contiguous()
        elif dt.dim() == 2:
            # [B, K] -> [B, K, N]
            dt = dt.unsqueeze(-1).expand(-1, -1, mask.size(1))
        dt = torch.nan_to_num(dt, nan=0.0, posinf=0.0, neginf=0.0)
        dt = dt * mask[:, None, :].to(dtype=dt.dtype)

    obs_mask = meta.get("x_obs_mask")
    if obs_mask is not None:
        obs_mask = torch.as_tensor(obs_mask, device=device, dtype=torch.bool)
        # expected [B, N, K, F] -> [B, K, N, F]
        if obs_mask.dim() == 4 and obs_mask.size(1) == mask.size(1):
            obs_mask = _permute_to_seq_first(obs_mask)
        obs_mask = obs_mask & mask[:, None, :, None]

    # 3. Apply the combined mask
    V = _apply_entity_mask(V, mask)
    T = _apply_entity_mask(T, mask)
    elems = _batch_elements(mask, V.size(1))

    return V, T, mask, elems, dt, obs_mask


def _run_epoch(
    loader: Iterable,
    model: LaplaceAE,
    device: torch.device,
    *,
    loss_weights: Tuple[float, float, float, float, float],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    grad_clip: float = 0.0,
    amp: bool = False,
) -> float:
    is_train = optimizer is not None
    total_loss = 0.0
    total_elems = 0.0

    for batch in loader:
        V, T, mask, elems, dt, obs_mask = _prepare_batch(batch, device)
        if elems == 0.0:
            continue

        if is_train:
            if scaler is None:
                raise ValueError("GradScaler must be provided when training.")
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            _, aux = model(V, pad_mask=mask, ctx_diff=T, dt=dt, obs_mask=obs_mask)
            loss = model.recon_loss(aux, mask, weights=loss_weights)

        if not torch.isfinite(loss):
            print("[warn] non-finite summarizer loss detected; aborting epoch")
            return float("nan")

        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * elems
        total_elems += elems

    if total_elems == 0.0:
        return 0.0 if is_train else float("inf")
    return total_loss / total_elems


def _build_model(
    train_loader: DataLoader,
    sizes: Optional[Sequence[int]],
    device: torch.device,
    *,
    config=config,
    verbose: bool = True,
) -> LaplaceAE:
    num_entities, feat_dim = _summarize_dataset(train_loader, sizes, verbose=verbose)
    model = LaplaceAE(
        num_entities=num_entities,
        feat_dim=feat_dim,
        window_size=config.WINDOW,
        mix_dim=int(getattr(config, "SUM_MIX_DIM", 64)),
        tv_hidden=config.SUM_TV_HIDDEN,
        out_len=config.SUM_CONTEXT_LEN,
        context_dim=config.SUM_CONTEXT_DIM,
        dropout=config.SUM_DROPOUT,
        time2vec_dim=int(getattr(config, "SUM_TIME2VEC_DIM", 9)),
        irreg_pooling=str(getattr(config, "SUM_IRREG_POOLING", "none")),
        irreg_hidden=int(getattr(config, "SUM_IRREG_HIDDEN", 32)),
        irreg_residual_scale=float(getattr(config, "SUM_IRREG_RES_SCALE", 0.1)),
        t_token_mode=str(getattr(config, "SUM_T_TOKEN_MODE", "none")),
        t_token_scale=float(getattr(config, "SUM_T_TOKEN_SCALE", 0.1)),
        pos_encoding=str(getattr(config, "SUM_POS_ENCODING", "continuous_rope")),
        rope_base=float(getattr(config, "SUM_ROPE_BASE", 10000.0)),
    ).to(device)
    return model


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    *,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    sizes: Optional[Sequence[int]] = None,
    config=config,
) -> Dict[str, object]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return {
            "checkpoint": str(checkpoint),
            "status": "fail",
            "messages": [f"missing checkpoint: {checkpoint}"],
        }

    (train_loader, val_loader, test_loader), sizes = _ensure_loaders(
        train_loader, val_loader, test_loader, sizes, config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config.SUM_AMP and device.type == "cuda")
    loss_weights = _loss_weights(config)
    messages = []

    try:
        model = _build_model(train_loader, sizes, device, config=config, verbose=False)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        best_val = float(state.get("stats", {}).get("val_loss", float("nan")))
    except Exception as exc:
        return {
            "checkpoint": str(checkpoint),
            "status": "fail",
            "messages": [f"failed to load checkpoint: {exc}"],
        }

    model.eval()
    with torch.no_grad():
        val_loss = _run_epoch(val_loader, model, device, loss_weights=loss_weights, amp=amp)
        test_loss = _run_epoch(test_loader, model, device, loss_weights=loss_weights, amp=amp)

    status = "pass"
    for name, value in (("best_val", best_val), ("val_loss", val_loss), ("test_loss", test_loss)):
        if not torch.isfinite(torch.tensor(value)):
            status = "fail"
            messages.append(f"{name} is non-finite")

    if status == "pass" and torch.isfinite(torch.tensor(best_val)) and torch.isfinite(torch.tensor(test_loss)):
        if test_loss > max(best_val * 1.50, best_val + 1e-6):
            status = "warn"
            messages.append(f"test loss drifted above best val loss: best_val={best_val:.6f}, test={test_loss:.6f}")

    return {
        "checkpoint": str(checkpoint),
        "status": status,
        "messages": messages,
        "sizes": tuple(sizes) if sizes is not None else None,
        "best_val": best_val,
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
    }


def run(
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    sizes: Optional[Sequence[int]] = None,
    config=config,
) -> Dict[str, object]:
    (train_loader, val_loader, test_loader), sizes = _ensure_loaders(
        train_loader, val_loader, test_loader, sizes, config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config.SUM_AMP and device.type == "cuda")
    grad_clip = getattr(config, "SUM_GRAD_CLIP", getattr(config, "GRAD_CLIP", 0.0))
    print(f"Using device: {device}")

    set_seed(config.SEED)

    model = _build_model(train_loader, sizes, device, config=config, verbose=True)
    print(f"Model params: {count_params(model) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.SUM_LR, weight_decay=config.SUM_WEIGHT_DECAY)
    scaler = GradScaler(enabled=amp)
    loss_weights = _loss_weights(config)
    print(
        "Summarizer loss weights: "
        f"x={loss_weights[0]:.3f} "
        f"v={loss_weights[1]:.3f} "
        f"t={loss_weights[2]:.3f} "
        f"dt={loss_weights[3]:.3f} "
        f"obs={loss_weights[4]:.3f}"
    )

    epochs = config.SUM_EPOCHS
    patience = config.SUM_PATIENCE
    min_delta = config.SUM_MIN_DELTA

    ckpt_path = Path(
        getattr(config, "SUM_CKPT", "")
        or (Path(config.SUM_DIR) / f"{config.PRED}-{config.VAE_LATENT_CHANNELS}-summarizer.pt")
    )

    best_val = math.inf
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = _run_epoch(
            train_loader,
            model,
            device,
            loss_weights=loss_weights,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=grad_clip,
            amp=amp,
        )

        model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(val_loader, model, device, loss_weights=loss_weights, amp=amp)

        elapsed = time.time() - epoch_start
        improved = val_loss < (best_val - min_delta)
        if improved:
            best_val = val_loss
            best_epoch = epoch
            patience_ctr = 0
            save_ckpt(ckpt_path, model, {"epoch": epoch, "val_loss": val_loss})
        else:
            patience_ctr += 1

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | train {train_loss:.6f} | val {val_loss:.6f} | "
            f"best {best_val:.6f} @ {best_epoch:03d} | patience {patience_ctr}/{patience} | {elapsed:.1f}s"
        )

        if patience_ctr >= patience:
            print(f"\nEarly stopping at epoch {epoch}: validation loss plateaued.")
            break

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        best_val = state.get("stats", {}).get("val_loss", best_val)

    model.eval()
    with torch.no_grad():
        test_loss = _run_epoch(test_loader, model, device, loss_weights=loss_weights, amp=amp)

    print(f"Best val loss: {best_val:.6f} | Test loss: {test_loss:.6f}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "sizes": sizes,
        "best_val": best_val,
        "val_loss": best_val,
        "test_loss": test_loss,
        "checkpoint": str(ckpt_path),
    }


if __name__ == "__main__":  # pragma: no cover
    run()
