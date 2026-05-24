"""Training and evaluation loop for the LaplaceAE summarizer model."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Iterable, MutableMapping, Optional, Sequence, Tuple

from llapdiffusion.configs import config
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from llapdiffusion.configs.dataset_registry import resolve_run_experiment
from llapdiffusion.logging_utils import is_debug, is_verbose, progress_iter
from llapdiffusion.models.summarizer import LaplaceAE
from llapdiffusion.target_artifacts import loader_target_request_from_config
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


def _grads_are_finite(params) -> bool:
    for param in params:
        grad = param.grad
        if grad is not None and not torch.isfinite(grad).all():
            return False
    return True


def _record_epoch_stat(epoch_stats: Optional[MutableMapping[str, int]], key: str, value: int = 1) -> None:
    if epoch_stats is not None:
        epoch_stats[key] = int(epoch_stats.get(key, 0)) + int(value)


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
        target_col, target_cols = loader_target_request_from_config(config)
        batch_size = int(getattr(config, "BATCH_SIZE", getattr(config, "DATES_PER_BATCH", 1)))
        train_loader, val_loader, test_loader, sizes = run_experiment(
            data_dir=config.DATA_DIR,
            date_batching=config.date_batching,
            dates_per_batch=batch_size,
            K=config.WINDOW,
            H=config.PRED,
            coverage=config.COVERAGE,
            batch_size=batch_size,
            ratios=(config.train_ratio, config.val_ratio, config.test_ratio),
            split_policy=getattr(config, "split_policy", "global_purged_horizon"),
            exact_timestamp_batches=bool(getattr(config, "exact_timestamp_batches", True)),
            target_col=target_col,
            target_cols=target_cols,
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

    # 1. Move context to sequence-first layout and validate values marked observed.
    V = _permute_to_seq_first(V).to(device)
    T = _permute_to_seq_first(T).to(device)
    mask = meta["entity_mask"].to(device=device, dtype=torch.bool)

    obs_mask = None
    raw_obs_mask = meta.get("x_obs_mask")
    if raw_obs_mask is not None:
        obs_mask = LaplaceAE._canon_obs_mask(
            raw_obs_mask,
            x=V,
            B=V.size(0),
            K=V.size(1),
            N=V.size(2),
            D=V.size(3),
            device=device,
        )
        entity_obs = mask[:, None, :, None].expand_as(V)
        observed = obs_mask & entity_obs
        if (observed & ~torch.isfinite(V)).any() or (observed & ~torch.isfinite(T)).any():
            raise ValueError("x_obs_mask marks non-finite context values as observed")
        mask = mask & obs_mask.any(dim=(1, 3))
    else:
        v_finite = torch.isfinite(V).all(dim=(1, 3))
        t_finite = torch.isfinite(T).all(dim=(1, 3))
        mask = mask & v_finite & t_finite

    # 2. Fill values that are masked out before passing tensors to the model.
    V = _nan_to_num(V)
    T = _nan_to_num(T)

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
        if dt.dim() != 3:
            raise ValueError(f"delta_t must have shape [B,K,N] after canonicalization, got {tuple(dt.shape)}")
        observed_dt = mask[:, None, :].expand_as(dt)
        if (observed_dt & ~torch.isfinite(dt)).any():
            raise ValueError("delta_t contains non-finite values for observed entities")
        dt = torch.nan_to_num(dt, nan=0.0, posinf=0.0, neginf=0.0)
        dt = dt * mask[:, None, :].to(dtype=dt.dtype)

    if obs_mask is not None:
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
    max_nonfinite_grad_steps: int = 0,
    epoch_stats: Optional[MutableMapping[str, int]] = None,
    progress_enabled: bool = False,
    progress_label: Optional[str] = None,
) -> float:
    is_train = optimizer is not None
    total_loss = 0.0
    total_elems = 0.0
    nonfinite_grad_steps = 0
    max_nonfinite_grad_steps = max(0, int(max_nonfinite_grad_steps))

    batches = progress_iter(
        loader,
        desc=progress_label or "summ epoch",
        enabled=progress_enabled,
        unit="batch",
    )
    for batch in batches:
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
            raise FloatingPointError("non-finite summarizer loss detected")

        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            params = list(model.parameters())
            if not _grads_are_finite(params):
                nonfinite_grad_steps += 1
                _record_epoch_stat(epoch_stats, "skipped_nonfinite_grad_steps")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if amp and nonfinite_grad_steps <= max_nonfinite_grad_steps:
                    continue
                raise FloatingPointError(
                    "non-finite summarizer gradients detected "
                    f"after {nonfinite_grad_steps} skipped optimizer step(s)"
                )
            if grad_clip and grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(params, grad_clip)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    raise FloatingPointError("non-finite summarizer gradient norm detected")
            scaler.step(optimizer)
            scaler.update()
            _record_epoch_stat(epoch_stats, "optimizer_steps")

        total_loss += loss.item() * elems
        total_elems += elems

    if total_elems == 0.0:
        if is_train and nonfinite_grad_steps > 0:
            raise FloatingPointError(
                "all summarizer optimizer steps were skipped because gradients were non-finite"
            )
        split = "training" if is_train else "evaluation"
        raise RuntimeError(f"Summarizer {split} epoch processed no valid elements")
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
        pos_encoding=str(getattr(config, "SUM_POS_ENCODING", "learned_abs")),
        rope_base=float(getattr(config, "SUM_ROPE_BASE", 10000.0)),
        channel_balanced_x_loss=bool(getattr(config, "SUM_CHANNEL_BALANCED_X_LOSS", False)),
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
    verbose = is_verbose(config)
    debug = is_debug(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config.SUM_AMP and device.type == "cuda")
    grad_clip = getattr(config, "SUM_GRAD_CLIP", getattr(config, "GRAD_CLIP", 0.0))
    max_nonfinite_grad_steps = max(0, int(getattr(config, "SUM_MAX_NONFINITE_GRAD_STEPS", 0) or 0))
    if verbose:
        print(f"Using device: {device}")

    set_seed(config.SEED)

    model = _build_model(train_loader, sizes, device, config=config, verbose=verbose)
    if verbose:
        print(f"Model params: {count_params(model) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.SUM_LR, weight_decay=config.SUM_WEIGHT_DECAY)
    scaler = GradScaler(enabled=amp)
    loss_weights = _loss_weights(config)
    if debug:
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
    skipped_nonfinite_grad_steps = 0
    epoch_stats_history = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_stats: Dict[str, int] = {}
        train_progress_kwargs = (
            {
                "progress_enabled": True,
                "progress_label": f"summ train e{epoch:03d}/{epochs:03d}",
            }
            if verbose
            else {}
        )
        train_loss = _run_epoch(
            train_loader,
            model,
            device,
            loss_weights=loss_weights,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=grad_clip,
            amp=amp,
            max_nonfinite_grad_steps=max(0, max_nonfinite_grad_steps - skipped_nonfinite_grad_steps),
            epoch_stats=epoch_stats,
            **train_progress_kwargs,
        )
        skipped_nonfinite_grad_steps += int(epoch_stats.get("skipped_nonfinite_grad_steps", 0))
        epoch_stats_history.append({"epoch": epoch, **epoch_stats})

        model.eval()
        with torch.no_grad():
            val_progress_kwargs = (
                {
                    "progress_enabled": True,
                    "progress_label": f"summ val e{epoch:03d}/{epochs:03d}",
                }
                if verbose
                else {}
            )
            val_loss = _run_epoch(
                val_loader,
                model,
                device,
                loss_weights=loss_weights,
                amp=amp,
                **val_progress_kwargs,
            )

        elapsed = time.time() - epoch_start
        improved = val_loss < (best_val - min_delta)
        if improved:
            best_val = val_loss
            best_epoch = epoch
            patience_ctr = 0
            save_ckpt(
                ckpt_path,
                model,
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "skipped_nonfinite_grad_steps": skipped_nonfinite_grad_steps,
                    "sum_max_nonfinite_grad_steps": max_nonfinite_grad_steps,
                },
            )
        else:
            patience_ctr += 1

        if verbose:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | train {train_loss:.6f} | val {val_loss:.6f} | "
                f"best {best_val:.6f} @ {best_epoch:03d} | patience {patience_ctr}/{patience} | {elapsed:.1f}s"
            )
            if epoch_stats.get("skipped_nonfinite_grad_steps"):
                print(
                    "Skipped "
                    f"{epoch_stats['skipped_nonfinite_grad_steps']} summarizer optimizer step(s) "
                    "with non-finite AMP gradients."
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
        test_progress_kwargs = (
            {"progress_enabled": True, "progress_label": "summ test"}
            if verbose
            else {}
        )
        test_loss = _run_epoch(
            test_loader,
            model,
            device,
            loss_weights=loss_weights,
            amp=amp,
            **test_progress_kwargs,
        )

    if verbose:
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
        "skipped_nonfinite_grad_steps": skipped_nonfinite_grad_steps,
        "sum_max_nonfinite_grad_steps": max_nonfinite_grad_steps,
        "epoch_stats": epoch_stats_history,
    }
