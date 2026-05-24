"""Disk-backed precomputation for frozen LLapDiff diffusion inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy.lib.format import open_memmap

from llapdiffusion.models.llapdiff_utils import (
    build_context,
    pack_targets_tokens,
    sample_t_uniform,
    simple_norm,
    target_time_observed,
)
from llapdiffusion.logging_utils import progress_iter


_CHUNK_ROWS = 128


def _bool_config(config_obj: object, name: str, default: bool) -> bool:
    return bool(getattr(config_obj, name, default))


def _dtype_from_name(name: str, *, default: np.dtype) -> np.dtype:
    text = str(name or "").strip().lower()
    if text in {"fp16", "float16", "half"}:
        return np.dtype(np.float16)
    if text in {"fp32", "float32", "single"}:
        return np.dtype(np.float32)
    return np.dtype(default)


def _tensor_digest(value, *, floating_round: Optional[int] = None) -> Optional[str]:
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    arr = tensor.numpy()
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32, copy=False)
        if floating_round is not None:
            arr = np.round(arr, decimals=int(floating_round)).astype(np.float32, copy=False)
    elif np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64, copy=False)
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.uint8, copy=False)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _tensor_int_list(value) -> Optional[List[int]]:
    if value is None:
        return None
    arr = torch.as_tensor(value).detach().cpu().reshape(-1).to(dtype=torch.int64)
    return [int(x) for x in arr.tolist()]


def batch_fingerprint(meta: Dict[str, torch.Tensor]) -> Dict[str, object]:
    entity_mask = meta.get("entity_mask")
    if entity_mask is None:
        raise KeyError("cache fingerprint requires meta['entity_mask']")
    rows = int(torch.as_tensor(entity_mask).shape[0])
    return {
        "rows": rows,
        "context_end_time_keys": _tensor_int_list(meta.get("context_end_time_keys")),
        "entity_mask": _tensor_digest(entity_mask),
        "delta_t_y": _tensor_digest(meta.get("delta_t_y"), floating_round=6),
        "cache_asset_ids": _tensor_digest(meta.get("cache_asset_ids")),
        "cache_window_starts": _tensor_digest(meta.get("cache_window_starts")),
    }


def fingerprint_digest(meta: Dict[str, torch.Tensor]) -> str:
    payload = json.dumps(batch_fingerprint(meta), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def _digest_strings(values: Sequence[str]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _file_fingerprint(path_like) -> Dict[str, object]:
    if path_like is None:
        return {"path": None, "exists": False}
    path = Path(str(path_like)).expanduser()
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "hash": h.hexdigest(),
    }


@dataclass
class _SplitPlan:
    name: str
    batch_rows: List[int]
    batch_digests: List[str]

    @property
    def total_rows(self) -> int:
        return int(sum(self.batch_rows))

    @property
    def digest(self) -> str:
        return _digest_strings(self.batch_digests)


@dataclass
class PreparedDiffusionBatch:
    mu_norm: Optional[torch.Tensor]
    obs_any: Optional[torch.Tensor]
    summary_raw: Optional[torch.Tensor]


class DiffusionSplitCache:
    """Sequential split cache aligned with a deterministic date-batched loader."""

    def __init__(
        self,
        *,
        name: str,
        root: Path,
        batch_rows: Sequence[int],
        batch_digests: Sequence[str],
        latent_shape: Tuple[int, int],
        summary_shape: Optional[Tuple[int, int]],
        latent_dtype: str,
        summary_dtype: Optional[str],
    ) -> None:
        self.name = str(name)
        self.root = Path(root)
        self.batch_rows = [int(x) for x in batch_rows]
        self.batch_digests = [str(x) for x in batch_digests]
        self.latent_shape = tuple(int(x) for x in latent_shape)
        self.summary_shape = None if summary_shape is None else tuple(int(x) for x in summary_shape)
        self.cursor = 0

        self._latents = np.load(self.root / f"{self.name}_latents.npy", mmap_mode="r")
        self._obs = np.load(self.root / f"{self.name}_obs_any.npy", mmap_mode="r")
        self._summary = None
        if self.summary_shape is not None:
            summary_path = self.root / f"{self.name}_summary_raw.npy"
            if summary_path.exists():
                self._summary = np.load(summary_path, mmap_mode="r")
        self.latent_dtype = str(latent_dtype)
        self.summary_dtype = summary_dtype

    def reset(self) -> None:
        self.cursor = 0

    def _claim(self, meta: Dict[str, torch.Tensor]) -> slice:
        if self.cursor >= len(self.batch_rows):
            raise RuntimeError(f"{self.name} diffusion cache exhausted before dataloader ended")
        got = fingerprint_digest(meta)
        expected = self.batch_digests[self.cursor]
        if got != expected:
            raise RuntimeError(
                f"{self.name} diffusion cache order mismatch at batch {self.cursor}: "
                f"expected {expected}, got {got}"
            )
        start = int(sum(self.batch_rows[: self.cursor]))
        stop = start + int(self.batch_rows[self.cursor])
        self.cursor += 1
        return slice(start, stop)

    def next_batch(
        self,
        meta: Dict[str, torch.Tensor],
        *,
        device: torch.device,
        mu_mean: Optional[torch.Tensor] = None,
        mu_std: Optional[torch.Tensor] = None,
        load_latents: bool = True,
        load_summary: bool = True,
    ) -> PreparedDiffusionBatch:
        row_slice = self._claim(meta)
        mu_norm = None
        obs_any = None
        summary_raw = None

        if load_latents:
            if mu_mean is None or mu_std is None:
                raise ValueError("mu_mean and mu_std are required when loading cached latents")
            mu_raw = torch.from_numpy(np.array(self._latents[row_slice], copy=True)).to(
                device=device,
                dtype=torch.float32,
            )
            obs_any = torch.from_numpy(np.array(self._obs[row_slice], copy=True)).to(
                device=device,
                dtype=torch.bool,
            )
            mu_norm = simple_norm(mu_raw, mu_mean, mu_std, clip_val=None)
            mu_norm = mu_norm * obs_any.unsqueeze(-1).to(dtype=mu_norm.dtype)
            if not torch.isfinite(mu_norm).all():
                raise FloatingPointError("cached normalized VAE latent means are non-finite")
            mu_norm = mu_norm.detach()
            obs_any = obs_any.detach()

        if load_summary:
            if self._summary is None:
                raise RuntimeError(f"{self.name} summary cache is not available")
            summary_raw = torch.from_numpy(np.array(self._summary[row_slice], copy=True)).to(
                device=device,
                dtype=torch.float32,
            )
            if not torch.isfinite(summary_raw).all():
                raise FloatingPointError("cached raw conditioning summary is non-finite")
            summary_raw = summary_raw.detach()

        return PreparedDiffusionBatch(mu_norm=mu_norm, obs_any=obs_any, summary_raw=summary_raw)

    def compute_latent_stats(self, device: torch.device, *, mode: str = "global") -> Tuple[torch.Tensor, torch.Tensor]:
        mode_name = str(mode).strip().lower()
        if mode_name not in {"global", "per_horizon", "horizon"}:
            raise ValueError(f"Unknown latent stats mode '{mode}'. Use 'global' or 'per_horizon'.")

        mu_sum = mu_sumsq = mu_count = None
        total = int(self._latents.shape[0])
        for start in range(0, total, _CHUNK_ROWS):
            stop = min(total, start + _CHUNK_ROWS)
            mu = torch.from_numpy(np.array(self._latents[start:stop], copy=True)).float()
            obs_any = torch.from_numpy(np.array(self._obs[start:stop], copy=True)).bool()
            if not obs_any.any():
                continue

            if mode_name == "global":
                mu_obs = mu[obs_any]
                if mu_obs.numel() == 0:
                    continue
                batch_sum = mu_obs.sum(dim=0).to(dtype=torch.float64)
                batch_sumsq = mu_obs.square().sum(dim=0).to(dtype=torch.float64)
                batch_count = torch.tensor(float(mu_obs.shape[0]), dtype=torch.float64)
            else:
                obs_f = obs_any.unsqueeze(-1).to(dtype=mu.dtype)
                batch_sum = (mu * obs_f).sum(dim=0).to(dtype=torch.float64)
                batch_sumsq = (mu.square() * obs_f).sum(dim=0).to(dtype=torch.float64)
                batch_count = obs_f.sum(dim=0).to(dtype=torch.float64)

            if mu_sum is None:
                mu_sum = batch_sum
                mu_sumsq = batch_sumsq
                mu_count = batch_count
            else:
                mu_sum += batch_sum
                mu_sumsq += batch_sumsq
                mu_count += batch_count

        if mu_sum is None or mu_count is None:
            raise RuntimeError("No valid latent samples found in diffusion cache.")
        denom = mu_count.clamp_min(1.0)
        mu_mean = (mu_sum / denom).to(device=device, dtype=torch.float32)
        mu_var = (mu_sumsq / denom).to(device=device, dtype=torch.float32) - mu_mean.square()
        mu_std = mu_var.clamp_min(0.0).sqrt().clamp_min(1e-6)
        return mu_mean, mu_std

    def calculate_target_variance(
        self,
        *,
        predict_type: str,
        scheduler,
        device: torch.device,
        mu_mean: torch.Tensor,
        mu_std: torch.Tensor,
    ) -> float:
        ptype = str(predict_type).strip().lower()
        if ptype == "eps":
            return 1.0
        if ptype not in {"v", "x0"}:
            raise ValueError(f"Unknown predict_type '{predict_type}'.")
        if ptype == "v" and scheduler is None:
            raise ValueError("scheduler is required for v-target variance.")

        total_sum = 0.0
        total_sumsq = 0.0
        total_count = 0
        total = int(self._latents.shape[0])
        for start in range(0, total, _CHUNK_ROWS):
            stop = min(total, start + _CHUNK_ROWS)
            mu_raw = torch.from_numpy(np.array(self._latents[start:stop], copy=True)).to(
                device=device,
                dtype=torch.float32,
            )
            obs_any = torch.from_numpy(np.array(self._obs[start:stop], copy=True)).to(
                device=device,
                dtype=torch.bool,
            )
            if not obs_any.any():
                continue
            x0 = simple_norm(mu_raw, mu_mean, mu_std, clip_val=None)[obs_any]
            if x0.numel() == 0:
                continue
            if ptype == "v":
                t = sample_t_uniform(scheduler, x0.size(0), device)
                eps_true = torch.randn_like(x0)
                x_t, _ = scheduler.q_sample(x0, t, eps_true)
                vals = scheduler.v_from_eps(x_t, t, eps_true)
            else:
                vals = x0
            vals = vals.detach().float()
            total_sum += float(vals.sum().item())
            total_sumsq += float(vals.square().sum().item())
            total_count += int(vals.numel())

        if total_count <= 0:
            raise RuntimeError("Cannot estimate target variance from an empty diffusion cache")
        mean = total_sum / float(total_count)
        return max(0.0, total_sumsq / float(total_count) - mean * mean)


class DiffusionInputCache:
    def __init__(
        self,
        *,
        root: Path,
        train: DiffusionSplitCache,
        val: Optional[DiffusionSplitCache],
        test: Optional[DiffusionSplitCache],
        mu_mean: torch.Tensor,
        mu_std: torch.Tensor,
        summary_enabled: bool,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.val = val
        self.test = test
        self.mu_mean = mu_mean
        self.mu_std = mu_std
        self.summary_enabled = bool(summary_enabled)


def cache_allowed(config_obj: object, *, summary_ft_mode: Optional[str] = None) -> Tuple[bool, str]:
    if not _bool_config(config_obj, "DIFF_PRECOMPUTE_INPUTS", True):
        return False, "DIFF_PRECOMPUTE_INPUTS is disabled"
    if not _bool_config(config_obj, "date_batching", False):
        return False, "date_batching is disabled"
    if not _bool_config(config_obj, "exact_timestamp_batches", False):
        return False, "exact_timestamp_batches is disabled"
    mode = str(summary_ft_mode if summary_ft_mode is not None else getattr(config_obj, "SUM_FT_MODE", "none"))
    if mode.strip().lower() != "none":
        return False, "summarizer fine-tuning is enabled"
    return True, "enabled"


def _metadata_plan(name: str, dataloader) -> _SplitPlan:
    batch_rows: List[int] = []
    batch_digests: List[str] = []
    for _, _, meta in dataloader:
        if meta.get("context_end_time_keys") is None:
            raise RuntimeError(
                "diffusion input cache requires context_end_time_keys; "
                "enable exact timestamp date batching"
            )
        rows = int(torch.as_tensor(meta["entity_mask"]).shape[0])
        batch_rows.append(rows)
        batch_digests.append(fingerprint_digest(meta))
    if not batch_rows:
        raise RuntimeError(f"{name} dataloader produced no batches for diffusion cache")
    return _SplitPlan(name=name, batch_rows=batch_rows, batch_digests=batch_digests)


def _manifest_core(config_obj: object, *, summary_enabled: bool) -> Dict[str, object]:
    target_cols = getattr(config_obj, "TARGET_COLS", None)
    if target_cols is not None and not isinstance(target_cols, (str, bytes)):
        target_cols = list(target_cols)
    return {
        "version": 1,
        "dataset_key": str(getattr(config_obj, "DATASET_KEY", "")),
        "data_dir": str(getattr(config_obj, "DATA_DIR", "")),
        "pred": int(getattr(config_obj, "PRED", 0)),
        "window": int(getattr(config_obj, "WINDOW", 0)),
        "batch_size": int(getattr(config_obj, "BATCH_SIZE", 0)),
        "dates_per_batch": int(getattr(config_obj, "DATES_PER_BATCH", 0)),
        "latent_channels": int(getattr(config_obj, "VAE_LATENT_CHANNELS", 0)),
        "latent_norm_mode": str(getattr(config_obj, "LATENT_NORM_MODE", "global")),
        "sum_context_len": int(getattr(config_obj, "SUM_CONTEXT_LEN", 0)),
        "sum_context_dim": int(getattr(config_obj, "SUM_CONTEXT_DIM", 0)),
        "target_col": getattr(config_obj, "TARGET_COL", None),
        "target_cols": target_cols,
        "target_source": getattr(config_obj, "TARGET_SOURCE", None),
        "split_policy": str(getattr(config_obj, "split_policy", "")),
        "split_scope": str(getattr(config_obj, "split_scope", "")),
        "coverage": float(getattr(config_obj, "COVERAGE", 0.0)),
        "summary_enabled": bool(summary_enabled),
        "latent_dtype": str(_dtype_from_name(getattr(config_obj, "DIFF_PRECOMPUTE_LATENT_DTYPE", "float32"), default=np.float32)),
        "summary_dtype": str(_dtype_from_name(getattr(config_obj, "DIFF_PRECOMPUTE_SUMMARY_DTYPE", "float16"), default=np.float16)),
        "vae_ckpt": _file_fingerprint(getattr(config_obj, "VAE_CKPT", None)),
        "sum_ckpt": _file_fingerprint(getattr(config_obj, "SUM_CKPT", None)),
    }


def _cache_root(config_obj: object) -> Path:
    configured = getattr(config_obj, "DIFF_PRECOMPUTE_DIR", None)
    if configured:
        return Path(str(configured)).expanduser()
    artifact_root = Path(str(getattr(config_obj, "ARTIFACT_ROOT", "./ldt"))).expanduser()
    dataset = str(getattr(config_obj, "DATASET_KEY", "dataset") or "dataset")
    pred = int(getattr(config_obj, "PRED", 0))
    batch = int(getattr(config_obj, "BATCH_SIZE", 0))
    return artifact_root / "diffusion_cache" / dataset / f"pred-{pred}_batch-{batch}"


def _split_from_manifest(root: Path, name: str, split_manifest: Dict[str, object]) -> DiffusionSplitCache:
    return DiffusionSplitCache(
        name=name,
        root=root,
        batch_rows=split_manifest["batch_rows"],
        batch_digests=split_manifest["batch_digests"],
        latent_shape=tuple(split_manifest["latent_shape"]),
        summary_shape=(
            None
            if split_manifest.get("summary_shape") is None
            else tuple(split_manifest["summary_shape"])
        ),
        latent_dtype=str(split_manifest["latent_dtype"]),
        summary_dtype=split_manifest.get("summary_dtype"),
    )


def _load_existing_cache(
    root: Path,
    manifest_core: Dict[str, object],
    plans: Dict[str, _SplitPlan],
    *,
    device: torch.device,
) -> Optional[DiffusionInputCache]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("core") != manifest_core:
        return None
    splits = manifest.get("splits", {})
    for name, plan in plans.items():
        split = splits.get(name)
        if split is None:
            return None
        if split.get("batch_digest") != plan.digest:
            return None
        if split.get("total_rows") != plan.total_rows:
            return None
        if not (root / f"{name}_latents.npy").exists():
            return None
        if not (root / f"{name}_obs_any.npy").exists():
            return None
        if manifest_core.get("summary_enabled") and not (root / f"{name}_summary_raw.npy").exists():
            return None

    train = _split_from_manifest(root, "train", splits["train"])
    val = _split_from_manifest(root, "val", splits["val"]) if "val" in splits else None
    test = _split_from_manifest(root, "test", splits["test"]) if "test" in splits else None
    mu_mean = torch.tensor(manifest["latent_stats"]["mu_mean"], device=device, dtype=torch.float32)
    mu_std = torch.tensor(manifest["latent_stats"]["mu_std"], device=device, dtype=torch.float32)
    return DiffusionInputCache(
        root=root,
        train=train,
        val=val,
        test=test,
        mu_mean=mu_mean,
        mu_std=mu_std,
        summary_enabled=bool(manifest_core.get("summary_enabled")),
    )


def _write_split(
    *,
    name: str,
    dataloader,
    plan: _SplitPlan,
    root: Path,
    vae,
    summarizer,
    device: torch.device,
    latent_dtype: np.dtype,
    summary_dtype: np.dtype,
    summary_enabled: bool,
    stats_mode: str,
    collect_stats: bool,
    progress_enabled: bool = False,
) -> Tuple[Dict[str, object], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    latents_mm = obs_mm = summary_mm = None
    latent_shape = None
    summary_shape = None
    row_offset = 0
    batch_idx = 0
    mu_sum = mu_sumsq = mu_count = None

    vae.eval()
    summarizer.eval()
    with torch.no_grad():
        batches = progress_iter(
            dataloader,
            desc=f"diffusion-cache {name}",
            enabled=progress_enabled,
            unit="batch",
        )
        for xb, yb, meta in batches:
            rows = int(torch.as_tensor(meta["entity_mask"]).shape[0])
            if batch_idx >= len(plan.batch_rows):
                raise RuntimeError(f"{name} cache writer saw more batches than planned")
            if rows != plan.batch_rows[batch_idx]:
                raise RuntimeError(f"{name} cache writer row count changed at batch {batch_idx}")

            mask_bn = meta["entity_mask"].to(device=device, dtype=torch.bool)
            yb = yb.to(device)
            y_obs_mask = meta.get("y_obs_mask")
            if y_obs_mask is not None:
                y_obs_mask = torch.as_tensor(y_obs_mask, device=device, dtype=torch.bool)
            x_tok, entity_pad, obs = pack_targets_tokens(
                yb,
                mask_bn,
                device=device,
                y_obs_mask=y_obs_mask,
            )
            if x_tok is None or obs is None:
                raise RuntimeError(f"{name} cache cannot encode an empty target batch")
            _, mu, _ = vae(x_tok, entity_pad)
            if not torch.isfinite(mu).all():
                raise FloatingPointError("VAE encoder produced non-finite latent means during cache build")
            obs_any = target_time_observed(obs)

            V, T = xb
            summary_raw = None
            if summary_enabled:
                summary_raw = build_context(
                    summarizer,
                    V.to(device),
                    T.to(device),
                    mask_bn,
                    device,
                    dt=meta.get("delta_t"),
                    x_obs_mask=meta.get("x_obs_mask"),
                    norm=False,
                    requires_grad=False,
                )
                if not torch.isfinite(summary_raw).all():
                    raise FloatingPointError("summarizer produced non-finite raw context during cache build")

            if latents_mm is None:
                latent_shape = (int(mu.shape[1]), int(mu.shape[2]))
                latents_mm = open_memmap(
                    root / f"{name}_latents.npy",
                    mode="w+",
                    dtype=latent_dtype,
                    shape=(plan.total_rows, *latent_shape),
                )
                obs_mm = open_memmap(
                    root / f"{name}_obs_any.npy",
                    mode="w+",
                    dtype=np.bool_,
                    shape=(plan.total_rows, int(obs_any.shape[1])),
                )
                if summary_enabled:
                    if summary_raw is None:
                        raise RuntimeError("summary cache requested but summary_raw is missing")
                    summary_shape = (int(summary_raw.shape[1]), int(summary_raw.shape[2]))
                    summary_mm = open_memmap(
                        root / f"{name}_summary_raw.npy",
                        mode="w+",
                        dtype=summary_dtype,
                        shape=(plan.total_rows, *summary_shape),
                    )

            latents_mm[row_offset: row_offset + rows] = mu.detach().cpu().numpy().astype(latent_dtype, copy=False)
            obs_mm[row_offset: row_offset + rows] = obs_any.detach().cpu().numpy().astype(np.bool_, copy=False)
            if summary_enabled and summary_mm is not None and summary_raw is not None:
                summary_mm[row_offset: row_offset + rows] = (
                    summary_raw.detach().cpu().numpy().astype(summary_dtype, copy=False)
                )

            if collect_stats:
                mode_name = str(stats_mode).strip().lower()
                mu_f = mu.detach().float()
                if obs_any.any():
                    if mode_name == "global":
                        mu_obs = mu_f[obs_any]
                        batch_sum = mu_obs.sum(dim=0).cpu().to(dtype=torch.float64)
                        batch_sumsq = mu_obs.square().sum(dim=0).cpu().to(dtype=torch.float64)
                        batch_count = torch.tensor(float(mu_obs.shape[0]), dtype=torch.float64)
                    else:
                        obs_f = obs_any.unsqueeze(-1).to(dtype=mu_f.dtype)
                        batch_sum = (mu_f * obs_f).sum(dim=0).cpu().to(dtype=torch.float64)
                        batch_sumsq = (mu_f.square() * obs_f).sum(dim=0).cpu().to(dtype=torch.float64)
                        batch_count = obs_f.sum(dim=0).cpu().to(dtype=torch.float64)
                    if mu_sum is None:
                        mu_sum = batch_sum
                        mu_sumsq = batch_sumsq
                        mu_count = batch_count
                    else:
                        mu_sum += batch_sum
                        mu_sumsq += batch_sumsq
                        mu_count += batch_count

            row_offset += rows
            batch_idx += 1

    if row_offset != plan.total_rows:
        raise RuntimeError(f"{name} cache writer expected {plan.total_rows} rows, wrote {row_offset}")
    if latents_mm is None or obs_mm is None or latent_shape is None:
        raise RuntimeError(f"{name} cache writer produced no arrays")
    latents_mm.flush()
    obs_mm.flush()
    if summary_mm is not None:
        summary_mm.flush()

    stats = None
    if collect_stats:
        if mu_sum is None or mu_count is None:
            raise RuntimeError("No valid latent samples found while building train diffusion cache")
        denom = mu_count.clamp_min(1.0)
        mu_mean = (mu_sum / denom).to(device=device, dtype=torch.float32)
        mu_var = (mu_sumsq / denom).to(device=device, dtype=torch.float32) - mu_mean.square()
        mu_std = mu_var.clamp_min(0.0).sqrt().clamp_min(1e-6)
        stats = (mu_mean, mu_std)

    split_manifest = {
        "total_rows": plan.total_rows,
        "batch_rows": plan.batch_rows,
        "batch_digests": plan.batch_digests,
        "batch_digest": plan.digest,
        "latent_shape": list(latent_shape),
        "summary_shape": None if summary_shape is None else list(summary_shape),
        "latent_dtype": str(latent_dtype),
        "summary_dtype": None if not summary_enabled else str(summary_dtype),
    }
    return split_manifest, stats


def build_or_load_diffusion_input_cache(
    *,
    train_dl,
    val_dl,
    test_dl,
    vae,
    summarizer,
    device: torch.device,
    config_obj: object,
    summary_ft_mode: Optional[str] = None,
    verbose: bool = False,
) -> Optional[DiffusionInputCache]:
    allowed, reason = cache_allowed(config_obj, summary_ft_mode=summary_ft_mode)
    if not allowed:
        if verbose:
            print(f"[diffusion cache] disabled: {reason}")
        return None

    summary_enabled = str(summary_ft_mode or getattr(config_obj, "SUM_FT_MODE", "none")).strip().lower() == "none"
    root = _cache_root(config_obj)
    latent_dtype = _dtype_from_name(
        getattr(config_obj, "DIFF_PRECOMPUTE_LATENT_DTYPE", "float32"),
        default=np.float32,
    )
    summary_dtype = _dtype_from_name(
        getattr(config_obj, "DIFF_PRECOMPUTE_SUMMARY_DTYPE", "float16"),
        default=np.float16,
    )

    if verbose:
        print(f"[diffusion cache] planning cache at {root}")
    plans = {
        "train": _metadata_plan("train", train_dl),
        "val": _metadata_plan("val", val_dl),
        "test": _metadata_plan("test", test_dl),
    }
    core = _manifest_core(config_obj, summary_enabled=summary_enabled)
    existing = _load_existing_cache(root, core, plans, device=device)
    if existing is not None:
        if verbose:
            print(f"[diffusion cache] reusing precomputed inputs from {root}")
        return existing

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    if verbose:
        total_rows = sum(plan.total_rows for plan in plans.values())
        print(
            "[diffusion cache] building "
            f"rows={total_rows} latent_dtype={latent_dtype} summary_dtype={summary_dtype}"
        )

    stats_mode = str(getattr(config_obj, "LATENT_NORM_MODE", "global"))
    splits: Dict[str, Dict[str, object]] = {}
    train_stats = None
    for name, loader in (("train", train_dl), ("val", val_dl), ("test", test_dl)):
        split_manifest, stats = _write_split(
            name=name,
            dataloader=loader,
            plan=plans[name],
            root=root,
            vae=vae,
            summarizer=summarizer,
            device=device,
            latent_dtype=latent_dtype,
            summary_dtype=summary_dtype,
            summary_enabled=summary_enabled,
            stats_mode=stats_mode,
            collect_stats=(name == "train"),
            progress_enabled=verbose,
        )
        splits[name] = split_manifest
        if stats is not None:
            train_stats = stats
        if verbose:
            print(f"[diffusion cache] wrote {name}: rows={split_manifest['total_rows']}")

    if train_stats is None:
        raise RuntimeError("Failed to compute latent stats while building diffusion cache")
    mu_mean, mu_std = train_stats
    manifest = {
        "core": core,
        "splits": splits,
        "latent_stats": {
            "mode": stats_mode,
            "mu_mean": mu_mean.detach().cpu().tolist(),
            "mu_std": mu_std.detach().cpu().tolist(),
        },
    }
    with (root / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    train = _split_from_manifest(root, "train", splits["train"])
    val = _split_from_manifest(root, "val", splits["val"])
    test = _split_from_manifest(root, "test", splits["test"])
    return DiffusionInputCache(
        root=root,
        train=train,
        val=val,
        test=test,
        mu_mean=mu_mean,
        mu_std=mu_std,
        summary_enabled=summary_enabled,
    )
