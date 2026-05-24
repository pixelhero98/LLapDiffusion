"""Training routine for the latent VAE used in the pipeline.

This version trains a *set-attention* VAE:
- For each time step t, treat N entities as a set of tokens.
- Token features are [value * obs_mask, obs_mask] (2 channels).
- Loss is masked reconstruction (only where obs_mask==1) + β * KL.
- Optional denoising corruption: random additional masking + small Gaussian noise.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from llapdiffusion.configs import config
from llapdiffusion.configs.dataset_registry import resolve_run_experiment
from llapdiffusion.logging_utils import is_debug, is_verbose, progress_iter
from llapdiffusion.latent_space.latent_vae import LatentVAE
from llapdiffusion.latent_space.latent_vae_utils import normalize_and_check
from llapdiffusion.models.llapdiff_utils import (
    infer_target_dim_from_loader,
    target_obs_mask_to_bhnc,
    target_time_observed,
    targets_to_bhnc,
    vae_io_dims_for_target_dim,
)
from llapdiffusion.target_artifacts import (
    loader_target_request_from_config,
    target_metadata_from_config,
    unwrap_checkpoint_model,
    validate_checkpoint_target_metadata,
)


LoaderTuple = Tuple[DataLoader, DataLoader, DataLoader]


def _nan_to_num(tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    """Replace NaN/Inf values to keep losses finite."""

    if torch.isfinite(tensor).all():
        return tensor
    return torch.nan_to_num(tensor, nan=value, posinf=value, neginf=value)


def _grads_are_finite(params) -> bool:
    for param in params:
        grad = param.grad
        if grad is not None and not torch.isfinite(grad).all():
            return False
    return True


def _ensure_loaders(
    train_dl: Optional[DataLoader],
    val_dl: Optional[DataLoader],
    test_dl: Optional[DataLoader],
    sizes: Optional[Sequence[int]],
    config=config,
) -> Tuple[LoaderTuple, Optional[Tuple[int, int, int]]]:
    """Return dataloaders, creating them if necessary, and infer dataset sizes."""

    if any(loader is None for loader in (train_dl, val_dl, test_dl)):
        run_experiment = resolve_run_experiment(config.DATA_DIR)
        target_col, target_cols = loader_target_request_from_config(config)
        batch_size = int(getattr(config, "BATCH_SIZE", getattr(config, "DATES_PER_BATCH", 1)))
        train_dl, val_dl, test_dl, sizes = run_experiment(
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
            sizes = tuple(len(dl.dataset) for dl in (train_dl, val_dl, test_dl))
        except Exception:
            sizes = None

    if train_dl is None or val_dl is None or test_dl is None:
        raise RuntimeError("Failed to obtain train/val/test dataloaders.")

    return (train_dl, val_dl, test_dl), sizes


def _log_dataset_summary(train_loader: DataLoader, sizes: Optional[Sequence[int]]) -> None:
    if sizes is not None:
        print(f"sizes: {tuple(sizes)}")
    else:
        print("sizes: (unknown)")

    try:
        (xb, yb, meta) = next(iter(train_loader))
    except StopIteration as exc:  # pragma: no cover - defensive
        raise RuntimeError("Training dataloader produced no batches.") from exc

    V, T = xb
    mask = meta["entity_mask"]
    print("V:", tuple(V.shape), "T:", tuple(T.shape), "y:", tuple(yb.shape))
    mask_float = mask.float()
    min_cov = float(mask_float.mean(1).min().item())
    frac_padded = float((~mask).float().mean().item())
    print(f"min coverage: {min_cov:.4f}")
    print(f"frac padded: {frac_padded:.4f}")


def _prepare_latent_batch(
    y: torch.Tensor,
    entity_mask: torch.Tensor,
    *,
    y_obs_mask: Optional[torch.Tensor] = None,
    p_drop: float,
    noise_std: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build mask-aware tokens for set-VAE training.

    Args:
      y:           [B, N, T] or [B, N, T, C] target trajectories (may include NaNs).
      entity_mask: [B, N] bool, True for real entities, False for padded.
      y_obs_mask:  optional target observation mask from the dataset.
      p_drop:      additional random masking rate on observed entries.
      noise_std:   gaussian noise std added to observed values.

    Returns:
      x_tok:      [B, T, N, 2*C] token = [values*keep_mask, keep_mask]
      y_clean:    [B, N, T, C] clean values (NaNs replaced with 0)
      obs:        [B, N, T, C] bool, original observation mask (finite & entity_mask)
      entity_pad: [B, N] bool, True for padded/non-existent entities
    """

    if entity_mask.dtype != torch.bool:
        entity_mask = entity_mask.to(dtype=torch.bool)
    entity_mask = entity_mask.to(device=y.device)

    y_bhnc = targets_to_bhnc(y, entity_mask, device=y.device)
    if y_bhnc is None:
        raise ValueError(f"target shape {tuple(y.shape)} is incompatible with entity mask {tuple(entity_mask.shape)}")
    obs_bhnc = target_obs_mask_to_bhnc(y_obs_mask, y_bhnc, entity_mask, device=y.device)
    y_bntc = y_bhnc.permute(0, 2, 1, 3).contiguous()
    obs = obs_bhnc.permute(0, 2, 1, 3).contiguous()
    if obs.sum().item() == 0:
        return None

    y_clean = _nan_to_num(y_bntc)

    if p_drop > 0:
        keep = (torch.rand_like(y_clean) > p_drop) & obs
    else:
        keep = obs

    if noise_std > 0:
        y_noisy = y_clean + noise_std * torch.randn_like(y_clean)
    else:
        y_noisy = y_clean

    x_tok = torch.cat([y_noisy * keep.float(), keep.float()], dim=-1)  # [B,N,T,2*C]
    x_tok = x_tok.permute(0, 2, 1, 3).contiguous()  # -> [B,T,N,2*C]

    entity_pad = ~entity_mask
    return x_tok, y_clean, obs, entity_pad


def _meta_y_obs_mask(meta: Dict[str, object], y: torch.Tensor) -> Optional[torch.Tensor]:
    mask = meta.get("y_obs_mask")
    if mask is None:
        return None
    return torch.as_tensor(mask, device=y.device)


def _target_mask_health(loader: Iterable, *, max_batches: int = 8) -> Dict[str, object]:
    missing = 0
    all_false = 0
    total = 0
    observed = 0
    for batch_idx, (_, yb, meta) in enumerate(loader):
        total += 1
        mask = meta.get("y_obs_mask")
        if mask is None:
            missing += 1
        else:
            mask_t = torch.as_tensor(mask, dtype=torch.bool)
            count = int(mask_t.sum().item())
            observed += count
            if count == 0:
                all_false += 1
        if batch_idx + 1 >= int(max_batches):
            break
    return {
        "checked_batches": total,
        "missing_batches": missing,
        "all_false_batches": all_false,
        "observed_entries": observed,
    }


def _masked_mse(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    obs: torch.Tensor,
    *,
    balance_mode: str = "none",
) -> Tuple[torch.Tensor, int]:
    """Compute mean squared error over observed entries only."""
    if y_hat.dim() + 1 == y_true.dim():
        y_hat = y_hat.unsqueeze(-1)
    if y_true.dim() + 1 == y_hat.dim():
        y_true = y_true.unsqueeze(-1)
    if obs.dim() + 1 == y_true.dim():
        obs = obs.unsqueeze(-1)
    if obs.shape != y_true.shape:
        try:
            obs = obs.expand_as(y_true)
        except RuntimeError as exc:
            raise ValueError(
                f"observation mask shape {tuple(obs.shape)} does not match target shape {tuple(y_true.shape)}"
            ) from exc
    obs_f = obs.float()
    denom = int(obs_f.sum().item())
    if denom <= 0:
        return y_hat.new_tensor(0.0), 0

    mode = str(balance_mode or "none").strip().lower()
    if mode in {"coverage", "coverage_balanced"}:
        reduce_dims = tuple(range(2, obs_f.dim()))
        total_slots = 1
        for dim in reduce_dims:
            total_slots *= max(1, obs_f.size(dim))
        coverage = obs_f.mean(dim=reduce_dims).clamp_min(1.0 / total_slots)
        valid_series = obs.any(dim=reduce_dims)
        inv_cov = torch.where(valid_series, 1.0 / coverage, torch.zeros_like(coverage))
        mean_weight = inv_cov[valid_series].mean().clamp_min(1e-6) if valid_series.any() else inv_cov.new_tensor(1.0)
        weight = (inv_cov / mean_weight).clamp(max=10.0)
        while weight.dim() < obs_f.dim():
            weight = weight.unsqueeze(-1)
        obs_f = weight * obs_f

    sq = (y_hat - y_true).pow(2) * obs_f
    return sq.sum() / obs_f.sum().clamp(min=1.0), denom

def _epoch_pass(
    loader: Iterable,
    model: LatentVAE,
    device: torch.device,
    beta: float,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    grad_clip: float = 1.0,
    amp_enabled: bool = False,
    p_drop: float = 0.0,
    noise_std: float = 0.0,
    cons_lambda: float = 0.0,
    recon_balance_mode: str = "none",
    progress_enabled: bool = False,
    progress_label: Optional[str] = None,
) -> Dict[str, float]:
    """Run one epoch step (train or eval) and accumulate statistics."""

    is_train = optimizer is not None
    totals = {
        "recon_sum": 0.0,
        "recon_elems": 0,
        "kl_sum": 0.0,
        "kl_count": 0,
        "cons_sum": 0.0,
        "cons_count": 0,
    }

    grad_ctx_factory = nullcontext if is_train else torch.no_grad

    batches = progress_iter(
        loader,
        desc=progress_label or "vae epoch",
        enabled=progress_enabled,
        unit="batch",
    )
    for (_, yb, meta) in batches:
        y = yb.to(device)
        entity_mask = meta["entity_mask"].to(device)
        y_obs_mask = _meta_y_obs_mask(meta, y)

        prepared = _prepare_latent_batch(
            y,
            entity_mask,
            y_obs_mask=y_obs_mask,
            p_drop=p_drop,
            noise_std=noise_std,
        )
        if prepared is None:
            continue
        x_tok, y_clean, obs, entity_pad = prepared

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with grad_ctx_factory():
            with autocast(enabled=amp_enabled):
                y_hat_bt, mu, logvar = model(x_tok, entity_pad=entity_pad)  # [B,T,N,C]
                y_hat = y_hat_bt.permute(0, 2, 1, 3).contiguous()  # -> [B,N,T,C]

                recon_loss, recon_count = _masked_mse(
                    y_hat,
                    y_clean,
                    obs,
                    balance_mode=recon_balance_mode,
                )

                kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B,T,C]
                kl_bt = kl_elem.sum(dim=-1)  # [B,T]
                obs_any_bt = target_time_observed(obs.permute(0, 2, 1, 3).contiguous())  # [B,T]
                kl_loss = kl_bt[obs_any_bt].mean() if obs_any_bt.any() else kl_bt.new_tensor(0.0)

                cons_loss = y_hat.new_tensor(0.0)
                if cons_lambda and cons_lambda > 0:
                    prepared2 = _prepare_latent_batch(
                        y,
                        entity_mask,
                        y_obs_mask=y_obs_mask,
                        p_drop=p_drop,
                        noise_std=noise_std,
                    )
                    if prepared2 is not None:
                        x_tok2, _, _, entity_pad2 = prepared2
                        _, mu2, _ = model(x_tok2, entity_pad=entity_pad2)
                        cons_bt = (mu - mu2.detach()).pow(2).mean(dim=-1)
                        cons_loss = cons_bt[obs_any_bt].mean() if obs_any_bt.any() else cons_bt.new_tensor(0.0)

                loss = recon_loss + beta * kl_loss + cons_lambda * cons_loss

        if not torch.isfinite(loss):
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite VAE loss detected")

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                params = list(model.parameters())
                if not _grads_are_finite(params):
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    raise FloatingPointError("non-finite VAE gradients detected")
                if grad_clip and grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    if not torch.isfinite(torch.as_tensor(grad_norm)):
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        raise FloatingPointError("non-finite VAE gradient norm detected")
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                params = list(model.parameters())
                if not _grads_are_finite(params):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("non-finite VAE gradients detected")
                if grad_clip and grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    if not torch.isfinite(torch.as_tensor(grad_norm)):
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError("non-finite VAE gradient norm detected")
                optimizer.step()

        if recon_count > 0:
            totals["recon_sum"] += float(recon_loss.item() * recon_count)
            totals["recon_elems"] += int(recon_count)

        if obs_any_bt.any():
            totals["kl_sum"] += float(kl_bt[obs_any_bt].sum().item())
            totals["kl_count"] += int(obs_any_bt.sum().item())

        if cons_lambda and cons_lambda > 0:
            totals["cons_sum"] += float(cons_loss.item())
            totals["cons_count"] += 1

    return totals

def _aggregate_metrics(totals: Dict[str, float]) -> Tuple[float, float, float]:
    if totals["recon_elems"] <= 0:
        raise RuntimeError("VAE epoch processed no reconstruction targets")
    if totals["kl_count"] <= 0:
        raise RuntimeError("VAE epoch processed no latent KL targets")
    recon = totals["recon_sum"] / totals["recon_elems"]
    kl = totals["kl_sum"] / totals["kl_count"]
    cons = totals["cons_sum"] / totals["cons_count"] if totals["cons_count"] > 0 else 0.0
    return recon, kl, cons


def _vae_entity_suffix(config=config) -> str:
    return "_entity" if bool(getattr(config, "VAE_ENTITY_CONDITION", False)) else ""


def _vae_checkpoint_path(kind: str, config=config) -> Path:
    target_suffix = str(getattr(config, "TARGET_ARTIFACT_SUFFIX", "") or "")
    return Path(config.VAE_DIR) / f"pred-{config.PRED}_ch-{config.VAE_LATENT_CHANNELS}{_vae_entity_suffix(config)}{target_suffix}_{kind}.pt"


def _vae_checkpoint_payload(model: LatentVAE, config=config, **extra) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "model": model.state_dict(),
        "target_metadata": target_metadata_from_config(config),
        "target_dim": int(getattr(config, "TARGET_DIM", 1)),
        "target_cols": list(getattr(config, "TARGET_COLS", []) or []),
        "target_indices": list(getattr(config, "TARGET_INDICES", []) or []),
        "target_source": str(getattr(config, "TARGET_SOURCE", "")),
        "vae_input_dim": int(getattr(config, "VAE_INPUT_DIM", 2)),
        "vae_output_dim": int(getattr(config, "VAE_OUTPUT_DIM", 1)),
    }
    payload.update(extra)
    return payload


def _infer_num_entities(loader: Iterable) -> int:
    try:
        _, yb, _ = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("Cannot infer VAE num_entities from an empty dataloader.") from exc
    if yb.dim() < 2:
        raise ValueError(f"Cannot infer num_entities from target shape={tuple(yb.shape)}")
    return int(yb.shape[1])


def _ensure_vae_num_entities(loader: Iterable, config=config) -> None:
    if bool(getattr(config, "VAE_ENTITY_CONDITION", False)) and getattr(config, "VAE_NUM_ENTITIES", None) is None:
        setattr(config, "VAE_NUM_ENTITIES", _infer_num_entities(loader))


def _ensure_vae_target_dims(loader: Iterable, config=config) -> int:
    target_dim = int(getattr(config, "TARGET_DIM", 0) or 0)
    if target_dim <= 0:
        target_dim = infer_target_dim_from_loader(loader)
        setattr(config, "TARGET_DIM", target_dim)
    input_dim, output_dim = vae_io_dims_for_target_dim(config, target_dim)
    setattr(config, "VAE_INPUT_DIM", input_dim)
    setattr(config, "VAE_OUTPUT_DIM", output_dim)
    return target_dim


def _build_model(device: torch.device, config=config) -> LatentVAE:
    return LatentVAE(
        seq_len=config.PRED,
        latent_dim=config.VAE_LATENT_DIM,
        latent_channel=config.VAE_LATENT_CHANNELS,
        enc_layers=config.VAE_LAYERS,
        enc_heads=config.VAE_HEADS,
        enc_ff=config.VAE_FF,
        dec_layers=config.VAE_LAYERS,
        dec_heads=config.VAE_HEADS,
        dec_ff=config.VAE_FF,
        input_dim=int(getattr(config, "VAE_INPUT_DIM", 2)),
        output_dim=int(getattr(config, "VAE_OUTPUT_DIM", 1)),
        dropout=float(getattr(config, "VAE_DROPOUT", 0.1)),
        num_entities=getattr(config, "VAE_NUM_ENTITIES", None),
        entity_conditioned=bool(getattr(config, "VAE_ENTITY_CONDITION", False)),
    ).to(device)


def _load_checkpoint_model(
    checkpoint_path: Path | str,
    device: torch.device,
    config=config,
) -> LatentVAE:
    model = _build_model(device, config=config)
    payload = torch.load(Path(checkpoint_path), map_location=device)
    validate_checkpoint_target_metadata(payload, config, context="VAE")
    model.load_state_dict(unwrap_checkpoint_model(payload))
    model.eval()
    return model


def _metrics_dict_from_totals(totals: Dict[str, float], beta: float) -> Dict[str, float]:
    recon, kl, cons = _aggregate_metrics(totals)
    return {
        "recon": float(recon),
        "kl": float(kl),
        "cons": float(cons),
        "beta_elbo": float(recon + beta * kl),
    }


def evaluate_metrics(
    loader: Iterable,
    model: LatentVAE,
    device: torch.device,
    beta: float,
    *,
    amp_enabled: bool = False,
    config=config,
) -> Dict[str, float]:
    totals = _epoch_pass(
        loader,
        model,
        device,
        beta,
        amp_enabled=amp_enabled,
        p_drop=0.0,
        noise_std=0.0,
        cons_lambda=0.0,
        recon_balance_mode=str(getattr(config, "VAE_RECON_BALANCE", "none")),
    )
    return _metrics_dict_from_totals(totals, beta)


def collect_latent_means(
    loader: Iterable,
    model: LatentVAE,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    all_mu = []
    with torch.no_grad():
        for batch_idx, (_, yb, meta) in enumerate(loader):
            y = yb.to(device)
            entity_mask = meta["entity_mask"].to(device)
            prepared = _prepare_latent_batch(
                y,
                entity_mask,
                y_obs_mask=_meta_y_obs_mask(meta, y),
                p_drop=0.0,
                noise_std=0.0,
            )
            if prepared is None:
                continue
            x_tok, _, obs, entity_pad = prepared
            _, mu, _ = model(x_tok, entity_pad=entity_pad)
            if not torch.isfinite(mu).all():
                raise FloatingPointError("VAE encoder produced non-finite latent means")
            obs_any = target_time_observed(obs.permute(0, 2, 1, 3).contiguous())
            if obs_any.any():
                all_mu.append(mu[obs_any].detach().cpu())
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    if not all_mu:
        return torch.empty((0, int(config.PRED), int(config.VAE_LATENT_CHANNELS)), dtype=torch.float32)
    return torch.cat(all_mu, dim=0)


def _compute_per_dim_stats(all_mu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if all_mu.numel() == 0:
        raise ValueError("Cannot compute latent statistics from an empty tensor.")
    if all_mu.dim() == 2:
        mu_per_dim = all_mu.mean(dim=0)
        std_per_dim = all_mu.std(dim=0, unbiased=False).clamp(min=1e-6)
    else:
        mu_per_dim = all_mu.mean(dim=(0, 1))
        std_per_dim = all_mu.std(dim=(0, 1), unbiased=False).clamp(min=1e-6)
    return mu_per_dim, std_per_dim


def summarize_normalized_latents(
    all_mu: torch.Tensor,
    *,
    ref_mean: Optional[torch.Tensor] = None,
    ref_std: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    if all_mu.numel() == 0:
        return {
            "num_sequences": 0,
            "global_mean": float("nan"),
            "global_std": float("nan"),
            "nan_count": 0,
            "inf_count": 0,
            "abs_p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    if ref_mean is None or ref_std is None:
        ref_mean, ref_std = _compute_per_dim_stats(all_mu)
    view_shape = (1, -1) if all_mu.dim() == 2 else (1, 1, -1)
    mu_b = ref_mean.view(*view_shape)
    std_b = ref_std.view(*view_shape)
    norm = (all_mu - mu_b) / std_b
    flat = norm.reshape(-1)
    finite = torch.isfinite(flat)
    finite_vals = flat[finite]
    if finite_vals.numel() == 0:
        global_mean = float("nan")
        global_std = float("nan")
        abs_p95 = float("nan")
        min_val = float("nan")
        max_val = float("nan")
    else:
        global_mean = float(finite_vals.mean().item())
        global_std = float(finite_vals.std().item())
        abs_p95 = float(torch.quantile(finite_vals.abs(), 0.95).item())
        min_val = float(finite_vals.min().item())
        max_val = float(finite_vals.max().item())
    return {
        "num_sequences": int(all_mu.size(0)),
        "global_mean": global_mean,
        "global_std": global_std,
        "nan_count": int(torch.isnan(norm).sum().item()),
        "inf_count": int(torch.isinf(norm).sum().item()),
        "abs_p95": abs_p95,
        "min": min_val,
        "max": max_val,
    }


def audit_checkpoint(
    checkpoint_path: Path | str,
    *,
    train_dl: Optional[DataLoader] = None,
    val_dl: Optional[DataLoader] = None,
    test_dl: Optional[DataLoader] = None,
    sizes: Optional[Sequence[int]] = None,
    config=config,
    max_latent_batches: Optional[int] = 64,
) -> Dict[str, object]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return {
            "checkpoint": str(checkpoint),
            "status": "fail",
            "messages": [f"missing checkpoint: {checkpoint}"],
        }

    (train_dl, val_dl, test_dl), sizes = _ensure_loaders(train_dl, val_dl, test_dl, sizes, config)
    _ensure_vae_num_entities(train_dl, config=config)
    _ensure_vae_target_dims(train_dl, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = _vae_amp_enabled(device, config=config)
    beta = float(getattr(config, "VAE_BETA", 0.0))
    messages = []

    try:
        model = _load_checkpoint_model(checkpoint, device, config=config)
    except Exception as exc:
        return {
            "checkpoint": str(checkpoint),
            "status": "fail",
            "messages": [f"failed to load checkpoint: {exc}"],
        }

    val_metrics = evaluate_metrics(val_dl, model, device, beta, amp_enabled=amp_enabled, config=config)
    test_metrics = evaluate_metrics(test_dl, model, device, beta, amp_enabled=amp_enabled, config=config)
    train_latents = collect_latent_means(train_dl, model, device, max_batches=max_latent_batches)
    val_latents = collect_latent_means(val_dl, model, device, max_batches=max_latent_batches)
    target_mask_health = {
        "train": _target_mask_health(train_dl),
        "val": _target_mask_health(val_dl),
        "test": _target_mask_health(test_dl),
    }

    status = "pass"
    for split_name, health in target_mask_health.items():
        if health["missing_batches"]:
            if status == "pass":
                status = "warn"
            messages.append(f"{split_name} y_obs_mask missing in {health['missing_batches']} checked batches")
        if health["all_false_batches"]:
            status = "fail"
            messages.append(f"{split_name} y_obs_mask all-false in {health['all_false_batches']} checked batches")
    for split_name, metrics in (("val", val_metrics), ("test", test_metrics)):
        for metric_name, metric_value in metrics.items():
            if not torch.isfinite(torch.tensor(metric_value)):
                status = "fail"
                messages.append(f"{split_name} {metric_name} is non-finite")

    if train_latents.numel() == 0 or val_latents.numel() == 0:
        status = "fail"
        messages.append("empty train/val latent collection")
        train_norm = summarize_normalized_latents(train_latents)
        val_norm = summarize_normalized_latents(val_latents)
    else:
        ref_mean, ref_std = _compute_per_dim_stats(train_latents)
        train_norm = summarize_normalized_latents(train_latents, ref_mean=ref_mean, ref_std=ref_std)
        val_norm = summarize_normalized_latents(val_latents, ref_mean=ref_mean, ref_std=ref_std)
        for split_name, stats in (("train", train_norm), ("val", val_norm)):
            if stats["nan_count"] or stats["inf_count"]:
                status = "fail"
                messages.append(f"{split_name} latent normalization produced NaN/Inf")
            elif (
                abs(float(stats["global_mean"])) > 0.10
                or float(stats["global_std"]) < 0.85
                or float(stats["global_std"]) > 1.15
            ):
                if status == "pass":
                    status = "warn"
                messages.append(
                    f"{split_name} latent normalization drift: mean={stats['global_mean']:.4f}, std={stats['global_std']:.4f}"
                )

    return {
        "checkpoint": str(checkpoint),
        "status": status,
        "messages": messages,
        "sizes": tuple(sizes) if sizes is not None else None,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "target_mask_health": target_mask_health,
        "latent_norm": {
            "train": train_norm,
            "val": val_norm,
        },
    }


def _beta_for_epoch(
    target_beta: float,
    epoch: int,
    *,
    warmup_epochs: int,
    anneal_epochs: int,
) -> float:
    target_beta = float(target_beta)
    if target_beta <= 0.0:
        return 0.0
    if epoch <= int(warmup_epochs):
        return 0.0
    if int(anneal_epochs) <= 0:
        return target_beta

    ramp_step = max(0, int(epoch) - int(warmup_epochs))
    ramp = min(1.0, ramp_step / float(int(anneal_epochs)))
    return float(target_beta * ramp)


def _vae_amp_enabled(device: torch.device, config=config) -> bool:
    """Return whether the VAE stage should use CUDA AMP."""
    return device.type == "cuda" and bool(getattr(config, "VAE_AMP", False))


def run(
    train_dl: Optional[DataLoader] = None,
    val_dl: Optional[DataLoader] = None,
    test_dl: Optional[DataLoader] = None,
    sizes: Optional[Sequence[int]] = None,
    plot_only: bool = False,
    config=config,
) -> Dict[str, object]:
    (train_dl, val_dl, test_dl), sizes = _ensure_loaders(train_dl, val_dl, test_dl, sizes, config)
    _ensure_vae_num_entities(train_dl, config=config)
    target_dim = _ensure_vae_target_dims(train_dl, config=config)
    verbose = is_verbose(config)
    debug = is_debug(config)
    if verbose:
        _log_dataset_summary(train_dl, sizes)
        print(f"VAE target_dim: {target_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = _vae_amp_enabled(device, config=config)
    if verbose:
        print(f"Using device: {device}")
    grad_clip = getattr(config, "GRAD_CLIP", 1.0)

    # Denoising / robustness knobs (defaults are safe if not present in config.py)
    p_drop = float(getattr(config, "VAE_INPUT_DROPOUT", 0.20))
    noise_std = float(getattr(config, "VAE_NOISE_STD", 0.01))
    cons_lambda = float(getattr(config, "VAE_CONSIST_LAMBDA", 0.0))
    recon_balance_mode = str(getattr(config, "VAE_RECON_BALANCE", "none"))

    model = _build_model(device, config=config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.VAE_LEARNING_RATE,
        weight_decay=config.VAE_WEIGHT_DECAY,
    )
    scaler = GradScaler(enabled=amp_enabled)

    vae_beta = float(getattr(config, "VAE_BETA", 0.0))
    warmup_epochs = int(getattr(config, "VAE_WARMUP_EPOCHS", 0))
    anneal_epochs = int(getattr(config, "VAE_KL_ANNEAL_EPOCHS", 0))
    min_epochs = int(getattr(config, "VAE_MIN_EPOCHS", 0))

    model_dir = Path(config.VAE_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_val_elbo = float("inf")
    best_val_recon = float("inf")
    best_elbo_path: Optional[Path] = None
    best_recon_path: Optional[Path] = None
    patience_counter = 0
    max_patience = int(getattr(config, "VAE_MAX_PATIENCE", 0))

    if not plot_only:
        if verbose:
            print("Starting VAE training.")
        for epoch in range(1, config.EPOCHS + 1):
            beta = _beta_for_epoch(
                vae_beta,
                epoch,
                warmup_epochs=warmup_epochs,
                anneal_epochs=anneal_epochs,
            )

            train_totals = _epoch_pass(
                train_dl,
                model,
                device,
                beta,
                optimizer=optimizer,
                scaler=scaler,
                grad_clip=grad_clip,
                amp_enabled=amp_enabled,
                p_drop=p_drop,
                noise_std=noise_std,
                cons_lambda=cons_lambda,
                recon_balance_mode=recon_balance_mode,
                progress_enabled=verbose,
                progress_label=f"vae train e{epoch:03d}/{config.EPOCHS:03d}",
            )
            val_totals = _epoch_pass(
                val_dl,
                model,
                device,
                beta,
                amp_enabled=amp_enabled,
                p_drop=0.0,
                noise_std=0.0,
                cons_lambda=0.0,
                recon_balance_mode=recon_balance_mode,
                progress_enabled=verbose,
                progress_label=f"vae val e{epoch:03d}/{config.EPOCHS:03d}",
            )

            train_recon, train_kl, train_cons = _aggregate_metrics(train_totals)
            val_recon, val_kl, _ = _aggregate_metrics(val_totals)

            train_elbo_beta = train_recon + beta * train_kl
            val_elbo_beta = val_recon + beta * val_kl

            cons_str = f", Cons {train_cons:.6f}" if cons_lambda > 0 else ""
            if verbose:
                print(
                    f"Epoch {epoch:03d}/{config.EPOCHS:03d} - β={beta:.6g} | "
                    f"Train β·ELBO {train_elbo_beta:.6f} [Recon {train_recon:.6f}, KL/sample {train_kl:.6f}{cons_str}] | "
                    f"Val β·ELBO {val_elbo_beta:.6f} [Recon {val_recon:.6f}, KL/sample {val_kl:.6f}]"
                )

            improved_elbo = val_elbo_beta < 0.99 * best_val_elbo
            improved_recon = val_recon < 0.99 * best_val_recon
            checkpoint_ready = epoch >= min_epochs

            if checkpoint_ready and improved_elbo:
                best_val_elbo = val_elbo_beta
                best_elbo_path = _vae_checkpoint_path("elbo", config=config)
                torch.save(
                    _vae_checkpoint_payload(
                        model,
                        config=config,
                        epoch=epoch,
                        checkpoint_kind="elbo",
                        val_elbo_beta=float(val_elbo_beta),
                        val_recon=float(val_recon),
                    ),
                    best_elbo_path,
                )
                if debug:
                    print("  -> Saved best beta-ELBO checkpoint")

            if checkpoint_ready and improved_recon:
                best_val_recon = val_recon
                best_recon_path = _vae_checkpoint_path("recon", config=config)
                torch.save(
                    _vae_checkpoint_payload(
                        model,
                        config=config,
                        epoch=epoch,
                        checkpoint_kind="recon",
                        val_elbo_beta=float(val_elbo_beta),
                        val_recon=float(val_recon),
                    ),
                    best_recon_path,
                )
                if debug:
                    print("  -> Saved best reconstruction checkpoint")

            patience_counter = 0 if improved_elbo else (patience_counter + 1)
            if epoch >= min_epochs and patience_counter >= max_patience:
                if verbose:
                    print(
                        f"\nEarly stopping at epoch {epoch}: β·ELBO hasn't improved in {max_patience} epochs "
                        f"after the minimum {min_epochs} epochs."
                    )
                break
    else:
        best_elbo_path = _vae_checkpoint_path("elbo", config=config)
        best_recon_path = _vae_checkpoint_path("recon", config=config)
        if best_elbo_path.exists() or best_recon_path.exists():
            print("plot_only=True: skipping VAE training and loading existing checkpoint.")
        else:
            print("plot_only=True requested but no checkpoint was found; proceeding with current model state.")

    checkpoint_to_load: Optional[Path] = None
    for candidate in (best_elbo_path, best_recon_path):
        if candidate is not None and candidate.exists():
            checkpoint_to_load = candidate
            break

    if checkpoint_to_load is not None:
        if verbose:
            print(f"Loading checkpoint: {checkpoint_to_load}")
        vae = _load_checkpoint_model(checkpoint_to_load, device, config=config)
    else:
        if verbose:
            print("No improved checkpoints saved; using the final training state.")
        vae = model

    vae.eval()

    for param in vae.encoder.parameters():
        param.requires_grad = False

    all_mu = []
    with torch.no_grad():
        for (_, yb, meta) in train_dl:
            y = yb.to(device)
            entity_mask = meta["entity_mask"].to(device)
            prepared = _prepare_latent_batch(
                y,
                entity_mask,
                y_obs_mask=_meta_y_obs_mask(meta, y),
                p_drop=0.0,
                noise_std=0.0,
            )
            if prepared is None:
                continue
            x_tok, _, _, entity_pad = prepared
            _, mu, _ = vae(x_tok, entity_pad=entity_pad)
            if not torch.isfinite(mu).all():
                raise FloatingPointError("VAE encoder produced non-finite latent means")
            all_mu.append(mu.cpu())

    if all_mu:
        latents = torch.cat(all_mu, dim=0)
        plot_latents = bool(getattr(config, "VAE_PLOT_LATENTS", False))
        normalize_and_check(latents, plot=plot_latents, verbose=verbose)
    else:
        raise RuntimeError("No latent means collected from the training dataloader")

    final_val_metrics = evaluate_metrics(val_dl, vae, device, vae_beta, amp_enabled=amp_enabled, config=config)
    final_test_metrics = evaluate_metrics(test_dl, vae, device, vae_beta, amp_enabled=amp_enabled, config=config)

    return {
        "train_loader": train_dl,
        "val_loader": val_dl,
        "test_loader": test_dl,
        "sizes": sizes,
        "best_val_elbo": best_val_elbo,
        "best_val_recon": best_val_recon,
        "best_elbo_path": str(best_elbo_path) if best_elbo_path else None,
        "best_recon_path": str(best_recon_path) if best_recon_path else None,
        "loaded_checkpoint": str(checkpoint_to_load) if checkpoint_to_load else None,
        "final_val_metrics": final_val_metrics,
        "final_test_metrics": final_test_metrics,
        "schedule": {
            "vae_beta": vae_beta,
            "vae_warmup_epochs": warmup_epochs,
            "vae_kl_anneal_epochs": anneal_epochs,
            "vae_min_epochs": min_epochs,
            "vae_max_patience": max_patience,
            "vae_input_dropout": p_drop,
            "vae_noise_std": noise_std,
            "vae_consist_lambda": cons_lambda,
            "vae_entity_condition": bool(getattr(config, "VAE_ENTITY_CONDITION", False)),
            "vae_num_entities": getattr(config, "VAE_NUM_ENTITIES", None),
            "vae_target_dim": int(getattr(config, "TARGET_DIM", 1)),
            "vae_input_dim": int(getattr(config, "VAE_INPUT_DIM", 2)),
            "vae_output_dim": int(getattr(config, "VAE_OUTPUT_DIM", 1)),
        },
        "model": vae,
    }
