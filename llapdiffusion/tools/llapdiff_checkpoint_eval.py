"""Evaluate LLapDiff checkpoints on forecast and target imputation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Sequence

import torch

from llapdiffusion.benchmark_protocol import llapdiff_protocol_metadata, split_protocol_metadata
from llapdiffusion.trainers import train_val_llapdiff as tv
from llapdiffusion.configs.dataset_archives import configure_dataset_archive
from llapdiffusion.configs.config_utils import clone_config, make_jsonable, normalize_predict_type
from llapdiffusion.configs.dataset_defaults import apply_dataset_preset, dataset_keys, default_horizons
from llapdiffusion.configs.dataset_registry import resolve_run_experiment
from llapdiffusion.datasets.target_selection import resolve_target_selection
from llapdiffusion.logging_utils import apply_verbosity, is_verbose, progress_task

from llapdiffusion.latent_space.latent_vae import LatentVAE
from llapdiffusion.models.summarizer import LaplaceAE
from llapdiffusion.models.llapdiff_utils import (
    decode_latents_with_vae,
    encode_mu_norm,
    infer_target_dim_from_loader,
    pack_targets_tokens,
    set_torch,
    target_time_observed,
    targets_to_bhnc,
    vae_io_dims_for_target_dim,
)
from llapdiffusion.target_artifacts import (
    checkpoint_target_metadata,
    sync_target_artifact_config,
    unwrap_checkpoint_model,
    validate_checkpoint_target_metadata,
)


COVERAGE_HELP = "fraction of observed context entries to hide; 0 disables induced missingness"
PREDICT_TYPE_HELP = (
    "Diffusion prediction parameterization for legacy checkpoints that do not record it. "
    "Modern checkpoints infer this from checkpoint metadata."
)


def build_eval_config(
    dataset_key: str,
    pred: int,
    *,
    target_col: str | None = None,
    target_cols: Sequence[str] | None = None,
    coverage: float = 0.0,
) -> SimpleNamespace:
    cfg = clone_config()
    apply_dataset_preset(cfg, dataset_key, pred=pred)
    if target_col and target_cols:
        raise ValueError("Use either target_col or target_cols, not both.")
    cfg.TARGET_COL = target_col
    cfg.TARGET_COLS = list(target_cols) if target_cols else None
    cfg.COVERAGE = _validate_coverage(coverage)
    return cfg


def _enforce_valid_keep_mask(obs_any: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    keep = keep & obs_any
    for b in range(obs_any.size(0)):
        idx = torch.where(obs_any[b])[0]
        if idx.numel() < 2:
            keep[b].zero_()
            continue
        if keep[b].sum().item() == 0:
            keep[b, idx[0]] = True
        hidden = obs_any[b] & (~keep[b])
        if hidden.sum().item() == 0:
            keep[b, idx[-1]] = False
            if keep[b].sum().item() == 0:
                keep[b, idx[0]] = True
    return keep & obs_any


def _make_regular_keep(obs_any: torch.Tensor, stride: int = 4) -> torch.Tensor:
    keep = torch.zeros_like(obs_any, dtype=torch.bool)
    keep[:, ::stride] = True
    return _enforce_valid_keep_mask(obs_any, keep)


def _make_random_keep(obs_any: torch.Tensor, frac: float, *, generator: torch.Generator) -> torch.Tensor:
    keep = (torch.rand(obs_any.shape, generator=generator, device=obs_any.device) < frac) & obs_any
    return _enforce_valid_keep_mask(obs_any, keep)


def _validate_random_mask_ratio(value: float) -> float:
    ratio = float(value)
    if not 0.0 < ratio < 1.0:
        raise ValueError("imputation random mask ratio must be in the open interval (0, 1)")
    return ratio


def _validate_coverage(value: object) -> float:
    coverage = float(value)
    if not 0.0 <= coverage < 1.0:
        raise ValueError("--coverage must satisfy 0 <= coverage < 1.")
    return coverage


def _validate_sample_count(value: int, *, name: str) -> int:
    count = int(value)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count


def _resolve_sample_counts(
    cfg: SimpleNamespace,
    *,
    num_samples: Optional[int],
    forecast_num_samples: Optional[int],
    imputation_num_samples: Optional[int],
) -> tuple[int, int]:
    default_samples = _validate_sample_count(
        getattr(cfg, "NUM_EVAL_SAMPLES", 25),
        name="cfg.NUM_EVAL_SAMPLES",
    )
    shared_samples = (
        default_samples
        if num_samples is None
        else _validate_sample_count(num_samples, name="num_samples")
    )
    forecast_samples = (
        shared_samples
        if forecast_num_samples is None
        else _validate_sample_count(forecast_num_samples, name="forecast_num_samples")
    )
    imputation_samples = (
        shared_samples
        if imputation_num_samples is None
        else _validate_sample_count(imputation_num_samples, name="imputation_num_samples")
    )
    return forecast_samples, imputation_samples


def _resolve_max_eval_batches(max_eval_batches: Optional[int]) -> Optional[int]:
    if max_eval_batches is None:
        return None
    batch_cap = int(max_eval_batches)
    if batch_cap < 0:
        raise ValueError("max_eval_batches must be non-negative")
    return None if batch_cap == 0 else batch_cap


class _LimitedBatches:
    def __init__(self, dataloader, max_batches: int):
        self._dataloader = dataloader
        self._max_batches = int(max_batches)

    def __iter__(self):
        for batch_idx, batch in enumerate(self._dataloader):
            if batch_idx >= self._max_batches:
                break
            yield batch

    def __len__(self) -> int:
        try:
            return min(len(self._dataloader), self._max_batches)
        except TypeError:
            return self._max_batches


def _limit_batches(dataloader, max_batches: Optional[int]):
    if max_batches is None:
        return dataloader
    return _LimitedBatches(dataloader, max_batches)


def _config_with_num_eval_samples(cfg: SimpleNamespace, num_samples: int) -> SimpleNamespace:
    cfg_copy = SimpleNamespace(**vars(cfg))
    cfg_copy.NUM_EVAL_SAMPLES = int(num_samples)
    return cfg_copy


def _with_imputation_metric_target(metrics: Dict[str, float]) -> Dict[str, object]:
    tagged = dict(metrics)
    tagged["metric_target_type"] = "target_horizon_imputation"
    return tagged


def _add_predict_type_candidate(candidates: list[tuple[str, str]], value: object, source: str) -> None:
    if value is None:
        return
    if isinstance(value, str) and value.strip() == "":
        return
    candidates.append((normalize_predict_type(value), source))


def _checkpoint_predict_type_metadata(payload: object) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    candidates: list[tuple[str, str]] = []
    model_config = payload.get("model_config")
    if isinstance(model_config, dict):
        llapdiff_config = model_config.get("llapdiff")
        if isinstance(llapdiff_config, dict):
            _add_predict_type_candidate(
                candidates,
                llapdiff_config.get("predict_type"),
                "checkpoint.model_config.llapdiff.predict_type",
            )
        _add_predict_type_candidate(
            candidates,
            model_config.get("predict_type"),
            "checkpoint.model_config.predict_type",
        )
    _add_predict_type_candidate(candidates, payload.get("predict_type"), "checkpoint.predict_type")

    if not candidates:
        return None

    values = {value for value, _ in candidates}
    if len(values) > 1:
        detail = ", ".join(f"{source}={value}" for value, source in candidates)
        raise ValueError(f"Checkpoint has conflicting predict_type metadata: {detail}")
    return candidates[0][0]


def _resolve_checkpoint_predict_type(
    payload: object,
    *,
    explicit_predict_type: Optional[str],
) -> tuple[str, str]:
    metadata_predict_type = _checkpoint_predict_type_metadata(payload)
    explicit = (
        None
        if explicit_predict_type is None
        or (isinstance(explicit_predict_type, str) and explicit_predict_type.strip() == "")
        else normalize_predict_type(explicit_predict_type)
    )
    if metadata_predict_type is not None:
        if explicit is not None and explicit != metadata_predict_type:
            raise ValueError(
                "Explicit --predict-type does not match checkpoint metadata: "
                f"{explicit} != {metadata_predict_type}."
            )
        return metadata_predict_type, "checkpoint_metadata"
    if explicit is None:
        raise ValueError(
            "Checkpoint does not record predict_type metadata; pass --predict-type explicitly "
            "when evaluating a legacy checkpoint."
        )
    return explicit, "cli"


def _apply_checkpoint_predict_type(
    cfg: SimpleNamespace,
    payload: object,
    *,
    explicit_predict_type: Optional[str],
) -> tuple[str, str]:
    resolved_predict_type, predict_type_source = _resolve_checkpoint_predict_type(
        payload,
        explicit_predict_type=explicit_predict_type,
    )
    setattr(cfg, "PREDICT_TYPE", resolved_predict_type)
    setattr(cfg, "PREDICT_TYPE_SOURCE", predict_type_source)
    return resolved_predict_type, predict_type_source


def _has_target_request(cfg: SimpleNamespace) -> bool:
    requested = getattr(cfg, "TARGET_COL", None)
    requested_cols = getattr(cfg, "TARGET_COLS", None)
    return bool(requested) or bool(requested_cols)


def _apply_checkpoint_target_metadata_if_unrequested(cfg: SimpleNamespace, payload: object) -> bool:
    if _has_target_request(cfg):
        return False
    metadata = checkpoint_target_metadata(payload)
    if metadata is None:
        return False
    sync_target_artifact_config(cfg, metadata, update_output_dirs=False)
    return True


def _target_policy(cfg: SimpleNamespace) -> Dict[str, object]:
    requested = getattr(cfg, "TARGET_COL", None)
    requested_cols = getattr(cfg, "TARGET_COLS", None)
    meta_path = Path(str(getattr(cfg, "DATA_DIR", ""))) / "cache_ratio_index" / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        selected = resolve_target_selection(
            meta,
            None if requested_cols else requested,
            requested_target_cols=requested_cols,
        )
        return {
            "target_col": selected.target_col,
            "target_cols": list(selected.target_cols),
            "target_indices": list(selected.target_indices),
            "target_dim": selected.target_dim,
            "target_source": selected.target_source,
            "requested_target_col": selected.requested_target_col,
            "requested_target_cols": list(selected.requested_target_cols or []),
            "calendar_feature_cols": list(selected.calendar_feature_cols),
        }
    except Exception as exc:
        if requested not in (None, "") or requested_cols not in (None, "", []):
            raise ValueError("Could not resolve requested target columns.") from exc
        return {
            "target_col": requested,
            "target_cols": (
                list(requested_cols)
                if requested_cols
                else ([requested] if requested else [])
            ),
            "target_indices": [],
            "target_dim": 1,
            "target_source": "unresolved",
            "requested_target_col": requested,
            "requested_target_cols": list(requested_cols) if requested_cols else [],
            "calendar_feature_cols": [],
        }


def _load_stack(
    cfg: SimpleNamespace,
    ckpt_path: Path,
    device: torch.device,
    train_dl,
    *,
    checkpoint_payload: object | None = None,
    predict_type: Optional[str] = None,
    verbose: bool = False,
):
    _, num_entities, window_size, feat_dim = tv._summarize_dataset(train_dl, None, verbose=verbose)
    target_dim = int(getattr(cfg, "TARGET_DIM", 0) or 0)
    if target_dim <= 0:
        target_dim = infer_target_dim_from_loader(train_dl)
        setattr(cfg, "TARGET_DIM", target_dim)
    vae_input_dim, vae_output_dim = vae_io_dims_for_target_dim(cfg, target_dim)
    setattr(cfg, "VAE_INPUT_DIM", vae_input_dim)
    setattr(cfg, "VAE_OUTPUT_DIM", vae_output_dim)
    payload = (
        torch.load(ckpt_path, map_location=device)
        if checkpoint_payload is None
        else checkpoint_payload
    )
    if not getattr(cfg, "PREDICT_TYPE_SOURCE", None):
        _apply_checkpoint_predict_type(cfg, payload, explicit_predict_type=predict_type)
    else:
        setattr(cfg, "PREDICT_TYPE", normalize_predict_type(getattr(cfg, "PREDICT_TYPE")))

    vae = LatentVAE(
        seq_len=cfg.PRED,
        latent_dim=cfg.VAE_LATENT_DIM,
        latent_channel=cfg.VAE_LATENT_CHANNELS,
        enc_layers=cfg.VAE_LAYERS,
        enc_heads=cfg.VAE_HEADS,
        enc_ff=cfg.VAE_FF,
        dec_layers=cfg.VAE_LAYERS,
        dec_heads=cfg.VAE_HEADS,
        dec_ff=cfg.VAE_FF,
        input_dim=vae_input_dim,
        output_dim=vae_output_dim,
        num_entities=num_entities,
        entity_conditioned=bool(getattr(cfg, "VAE_ENTITY_CONDITION", False)),
    ).to(device)
    vae_payload = torch.load(cfg.VAE_CKPT, map_location=device)
    validate_checkpoint_target_metadata(vae_payload, cfg, context="VAE")
    tv._load_module_state(vae, unwrap_checkpoint_model(vae_payload), strict=True)
    vae.eval()

    summarizer = LaplaceAE(
        num_entities=num_entities,
        feat_dim=feat_dim,
        window_size=window_size,
        mix_dim=int(getattr(cfg, "SUM_MIX_DIM", 64)),
        tv_hidden=cfg.SUM_TV_HIDDEN,
        out_len=cfg.SUM_CONTEXT_LEN,
        context_dim=cfg.SUM_CONTEXT_DIM,
        n_heads=cfg.NUM_HEADS,
        dropout=cfg.SUM_DROPOUT,
        time2vec_dim=int(getattr(cfg, "SUM_TIME2VEC_DIM", 9)),
        irreg_pooling=str(getattr(cfg, "SUM_IRREG_POOLING", "none")),
        irreg_hidden=int(getattr(cfg, "SUM_IRREG_HIDDEN", 32)),
        irreg_residual_scale=float(getattr(cfg, "SUM_IRREG_RES_SCALE", 0.1)),
        t_token_mode=str(getattr(cfg, "SUM_T_TOKEN_MODE", "none")),
        t_token_scale=float(getattr(cfg, "SUM_T_TOKEN_SCALE", 0.1)),
        pos_encoding=str(getattr(cfg, "SUM_POS_ENCODING", "learned_abs")),
        rope_base=float(getattr(cfg, "SUM_ROPE_BASE", 10000.0)),
        channel_balanced_x_loss=bool(getattr(cfg, "SUM_CHANNEL_BALANCED_X_LOSS", False)),
    ).to(device)
    sum_state = torch.load(cfg.SUM_CKPT, map_location=device)
    tv._load_module_state(
        summarizer,
        sum_state["model"] if isinstance(sum_state, dict) and "model" in sum_state else sum_state,
        strict=True,
    )
    summarizer.eval()

    validate_checkpoint_target_metadata(payload, cfg, context="LLapDiff")
    diff_model = tv.build_llapdiff_model(cfg, device, checkpoint_payload=payload)
    tv._load_module_state(diff_model, payload["model"], strict=True)
    diff_model.eval()
    mu_mean = payload["mu_mean"].to(device)
    mu_std = payload["mu_std"].to(device)
    return diff_model, vae, summarizer, mu_mean, mu_std


@torch.inference_mode()
def _evaluate_impute_case(
    test_dl,
    *,
    diff_model,
    vae,
    summarizer,
    device: torch.device,
    mu_mean: torch.Tensor,
    mu_std: torch.Tensor,
    keep_fn: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int = 25,
    steps: int = 64,
    guidance_strength=(1.0, 2.0),
    guidance_power: float = 1.0,
    eta: float = 0.0,
    dynamic_thresh_p: float = 0.0,
    dynamic_thresh_max: float = 1.0,
    rho: float = 7.5,
    generator_seed: Optional[int] = None,
    progress_label: Optional[str] = None,
    progress_enabled: bool = False,
) -> Dict[str, float]:
    abs_sum = sq_sum = elts = 0.0
    crps_sum = crps_elts = 0.0
    obs_abs_sum = obs_elts = 0.0
    observed_token_sum = hidden_token_sum = candidate_token_sum = 0.0
    generator = None
    if generator_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(generator_seed))

    try:
        progress_total = len(test_dl) * int(num_samples) * max(1, int(steps))
    except TypeError:
        progress_total = None

    processed_batches = 0
    valid_batches = 0
    with progress_task(
        desc=progress_label or "checkpoint-eval imputation",
        enabled=progress_enabled,
        total=progress_total,
        unit="step",
    ) as progress:
        for xb, yb, meta in test_dl:
            processed_batches += 1
            (V, T), yb, mask_bn = tv._sanitize_batch(xb, yb, meta, device)
            if not mask_bn.any():
                continue

            cond_summary, cond_summary_raw = tv._build_cond_summary_pair(
                summarizer,
                diff_model,
                V,
                T,
                mask_bn,
                device,
                dt=meta.get("delta_t"),
                x_obs_mask=meta.get("x_obs_mask"),
            )
            if not tv._is_finite_tensor(cond_summary):
                raise FloatingPointError("non-finite cond_summary detected during checkpoint evaluation")
            if cond_summary_raw is not None and not tv._is_finite_tensor(cond_summary_raw):
                raise FloatingPointError("non-finite raw conditioning summary detected during checkpoint evaluation")
            dt_b = tv._flatten_dt(
                meta,
                mask_bn,
                device,
                key="delta_t_y",
            )
            x_tok, entity_pad, obs = pack_targets_tokens(
                yb,
                mask_bn,
                device,
                y_obs_mask=meta.get("y_obs_mask"),
            )
            if x_tok is None or not obs.any():
                continue

            obs_any = target_time_observed(obs)
            keep_mask = keep_fn(obs_any)
            valid_seq = keep_mask.any(dim=1) & (obs_any & (~keep_mask)).any(dim=1)
            if not valid_seq.any():
                continue
            valid_batches += 1

            cond_summary = cond_summary[valid_seq]
            cond_summary_raw = cond_summary_raw[valid_seq]
            yb = yb[valid_seq]
            mask_bn = mask_bn[valid_seq]
            x_tok = x_tok[valid_seq]
            entity_pad = entity_pad[valid_seq]
            obs = obs[valid_seq]
            obs_any = obs_any[valid_seq]
            keep_mask = keep_mask[valid_seq]
            dt_model = tv._match_dt_to_horizon(
                dt_b[valid_seq] if dt_b is not None else None,
                x_tok.size(1),
            )

            mu_norm = encode_mu_norm(
                vae,
                x_tok,
                entity_pad=entity_pad,
                mu_mean=mu_mean,
                mu_std=mu_std,
            )
            mu_norm = mu_norm * obs_any.unsqueeze(-1).to(dtype=mu_norm.dtype)
            y_obs = mu_norm * keep_mask.unsqueeze(-1).to(dtype=mu_norm.dtype)

            all_samples = []
            for _ in range(num_samples):
                x0_norm = diff_model.generate(
                    shape=tuple(mu_norm.shape),
                    steps=steps,
                    guidance_strength=guidance_strength,
                    guidance_power=guidance_power,
                    eta=eta,
                    cond_summary=cond_summary,
                    cond_summary_raw=cond_summary_raw,
                    y_obs=y_obs,
                    obs_mask=keep_mask,
                    dt=dt_model,
                    cfg_rescale=True,
                    self_cond=False,
                    dynamic_thresh_p=dynamic_thresh_p,
                    dynamic_thresh_max=dynamic_thresh_max,
                    rho=rho,
                    generator=generator,
                )
                all_samples.append(
                    decode_latents_with_vae(
                        vae,
                        x0_norm,
                        entity_pad=entity_pad,
                        mu_mean=mu_mean,
                        mu_std=mu_std,
                    )
                )
                progress.update(max(1, int(steps)))

            all_samples = torch.stack(all_samples, dim=0)
            point_forecast = all_samples.mean(dim=0)
            y_true = torch.nan_to_num(
                targets_to_bhnc(yb, mask_bn, device=device),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            hidden_valid = (obs & (~keep_mask[:, :, None, None])).to(dtype=y_true.dtype)
            observed_valid = (obs & keep_mask[:, :, None, None]).to(dtype=y_true.dtype)

            res_hidden = (point_forecast - y_true) * hidden_valid
            abs_sum += res_hidden.abs().sum().item()
            sq_sum += (res_hidden**2).sum().item()
            elts += hidden_valid.sum().item()

            res_obs = (point_forecast - y_true) * observed_valid
            obs_abs_sum += res_obs.abs().sum().item()
            obs_elts += observed_valid.sum().item()

            term1 = (all_samples - y_true.unsqueeze(0)).abs().mean(dim=0)
            sample_count = all_samples.shape[0]
            if sample_count <= 1:
                term2 = torch.zeros_like(term1)
            else:
                diffs = []
                for i in range(sample_count):
                    for j in range(i + 1, sample_count):
                        diffs.append((all_samples[i] - all_samples[j]).abs())
                term2 = torch.stack(diffs, dim=0).mean(dim=0)
            crps_elem = term1 - 0.5 * term2
            crps_sum += (crps_elem * hidden_valid).sum().item()
            crps_elts += hidden_valid.sum().item()

            observed_token_sum += keep_mask.sum().item()
            hidden_token_sum += (obs_any & (~keep_mask)).sum().item()
            candidate_token_sum += obs_any.sum().item()

    if candidate_token_sum <= 0:
        raise RuntimeError("Imputation evaluation found no candidate observed tokens")
    if hidden_token_sum <= 0 or elts <= 0 or crps_elts <= 0:
        raise RuntimeError("Imputation evaluation found no hidden target tokens")
    if observed_token_sum <= 0 or obs_elts <= 0:
        raise RuntimeError("Imputation evaluation found no retained observed tokens")

    return {
        "hidden_mae": abs_sum / elts,
        "hidden_mse": sq_sum / elts,
        "hidden_crps": crps_sum / crps_elts,
        "observed_mae": obs_abs_sum / obs_elts,
        "observed_token_frac": observed_token_sum / candidate_token_sum,
        "hidden_token_frac": hidden_token_sum / candidate_token_sum,
    }


def evaluate_checkpoint(
    cfg: SimpleNamespace,
    ckpt_path,
    label: str,
    out_path: Optional[str] = None,
    *,
    generator_seed: Optional[int] = None,
    random_mask_ratio: Optional[float] = None,
    num_samples: Optional[int] = None,
    forecast_num_samples: Optional[int] = None,
    imputation_num_samples: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    predict_type: Optional[str] = None,
    verbose: Optional[bool] = None,
) -> Dict[str, object]:
    ckpt_path = Path(ckpt_path)
    checkpoint_payload = torch.load(ckpt_path, map_location="cpu")
    _apply_checkpoint_predict_type(cfg, checkpoint_payload, explicit_predict_type=predict_type)
    setattr(
        cfg,
        "CHECKPOINT_TARGET_METADATA_APPLIED",
        _apply_checkpoint_target_metadata_if_unrequested(cfg, checkpoint_payload),
    )

    verbose = is_verbose(cfg) if verbose is None else bool(verbose)
    if random_mask_ratio is None:
        random_mask_ratio = float(getattr(cfg, "IMPUTATION_RANDOM_MASK_RATIO", 0.30))
    random_mask_ratio = _validate_random_mask_ratio(random_mask_ratio)
    random_keep_frac = 1.0 - random_mask_ratio
    forecast_samples, imputation_samples = _resolve_sample_counts(
        cfg,
        num_samples=num_samples,
        forecast_num_samples=forecast_num_samples,
        imputation_num_samples=imputation_num_samples,
    )
    batch_cap = _resolve_max_eval_batches(max_eval_batches)
    device = set_torch(seed=int(getattr(cfg, "SEED", 42)), deterministic=bool(getattr(cfg, "DETERMINISTIC", False)))
    run_experiment = resolve_run_experiment(cfg.DATA_DIR)
    batch_size = int(getattr(cfg, "BATCH_SIZE", getattr(cfg, "DATES_PER_BATCH", 1)))
    train_dl, val_dl, test_dl, sizes = run_experiment(
        data_dir=cfg.DATA_DIR,
        date_batching=cfg.date_batching,
        dates_per_batch=batch_size,
        K=cfg.WINDOW,
        H=cfg.PRED,
        coverage=cfg.COVERAGE,
        batch_size=batch_size,
        ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio),
        split_policy=getattr(cfg, "split_policy", "global_purged_horizon"),
        exact_timestamp_batches=bool(getattr(cfg, "exact_timestamp_batches", True)),
        target_col=None if getattr(cfg, "TARGET_COLS", None) else getattr(cfg, "TARGET_COL", None),
        target_cols=getattr(cfg, "TARGET_COLS", None),
    )
    if verbose and sizes is not None:
        print("eval sizes:", tuple(sizes))
    sync_target_artifact_config(cfg, _target_policy(cfg), update_output_dirs=False)
    diff_model, vae, summarizer, mu_mean, mu_std = _load_stack(
        cfg,
        ckpt_path,
        device,
        train_dl,
        checkpoint_payload=checkpoint_payload,
        predict_type=predict_type,
        verbose=verbose,
    )
    test_sampling = tv._sampling_kwargs(cfg, prefix="TEST")
    forecast_cfg = _config_with_num_eval_samples(cfg, forecast_samples)

    forecast = tv.evaluate_regression(
        diff_model,
        vae,
        summarizer,
        _limit_batches(test_dl, batch_cap),
        device,
        mu_mean,
        mu_std,
        forecast_cfg,
        ema=None,
        self_cond=bool(getattr(cfg, "SELF_COND", False)),
        generator_seed=generator_seed,
        verbose=verbose,
        progress_enabled=verbose,
        progress_label="checkpoint-eval forecast_test",
        **test_sampling,
    )
    regular = _with_imputation_metric_target(_evaluate_impute_case(
        _limit_batches(test_dl, batch_cap),
        diff_model=diff_model,
        vae=vae,
        summarizer=summarizer,
        device=device,
        mu_mean=mu_mean,
        mu_std=mu_std,
        keep_fn=lambda obs_any: _make_regular_keep(obs_any, stride=4),
        num_samples=imputation_samples,
        steps=int(test_sampling["steps"]),
        guidance_strength=test_sampling["guidance_strength"],
        guidance_power=float(test_sampling["guidance_power"]),
        eta=float(test_sampling["eta"]),
        dynamic_thresh_p=float(test_sampling["dynamic_thresh_p"]),
        dynamic_thresh_max=float(test_sampling["dynamic_thresh_max"]),
        rho=float(test_sampling["rho"]),
        generator_seed=generator_seed,
        progress_enabled=verbose,
        progress_label="checkpoint-eval regular_keep25",
    ))
    random_keep_generator = torch.Generator(device=device)
    random_keep_generator.manual_seed(1234)
    random_mask = _with_imputation_metric_target(_evaluate_impute_case(
        _limit_batches(test_dl, batch_cap),
        diff_model=diff_model,
        vae=vae,
        summarizer=summarizer,
        device=device,
        mu_mean=mu_mean,
        mu_std=mu_std,
        keep_fn=lambda obs_any: _make_random_keep(obs_any, frac=random_keep_frac, generator=random_keep_generator),
        num_samples=imputation_samples,
        steps=int(test_sampling["steps"]),
        guidance_strength=test_sampling["guidance_strength"],
        guidance_power=float(test_sampling["guidance_power"]),
        eta=float(test_sampling["eta"]),
        dynamic_thresh_p=float(test_sampling["dynamic_thresh_p"]),
        dynamic_thresh_max=float(test_sampling["dynamic_thresh_max"]),
        rho=float(test_sampling["rho"]),
        generator_seed=None if generator_seed is None else int(generator_seed) + 100003,
        progress_enabled=verbose,
        progress_label="checkpoint-eval random_mask",
    ))

    result = {
        "label": label,
        "checkpoint": str(ckpt_path),
        "predict_type": getattr(cfg, "PREDICT_TYPE", None),
        "predict_type_source": getattr(cfg, "PREDICT_TYPE_SOURCE", None),
        "checkpoint_target_metadata_applied": bool(
            getattr(cfg, "CHECKPOINT_TARGET_METADATA_APPLIED", False)
        ),
        "benchmark_protocol": llapdiff_protocol_metadata(),
        "data_policy": {
            **_target_policy(cfg),
            "target_dim": int(getattr(cfg, "TARGET_DIM", 1)),
            "vae_input_dim": int(getattr(cfg, "VAE_INPUT_DIM", 2)),
            "vae_output_dim": int(getattr(cfg, "VAE_OUTPUT_DIM", 1)),
            "split_policy": getattr(cfg, "split_policy", "global_purged_horizon"),
            "split_scope": getattr(cfg, "split_scope", "global_target_time"),
            "batching_policy": (
                "exact_context_end_timestamp"
                if bool(getattr(cfg, "exact_timestamp_batches", True))
                else "calendar_day"
            ),
            **split_protocol_metadata(
                getattr(cfg, "DATASET_KEY", ""),
                split_policy=getattr(cfg, "split_policy", "global_purged_horizon"),
                split_scope=getattr(cfg, "split_scope", "global_target_time"),
            ),
        },
        "forecast_test": forecast,
        "regular_keep25": regular,
        "random_mask_ratio": random_mask_ratio,
        "random_mask": random_mask,
        "balanced_summary": {
            "avg_hidden_crps": 0.5
            * (float(regular["hidden_crps"]) + float(random_mask["hidden_crps"])),
            "passes_forecast_guardrail": None,
        },
    }
    if abs(random_mask_ratio - 0.30) < 1e-12:
        result["random_mask30"] = dict(random_mask)
    if out_path is not None:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(make_jsonable(result), indent=2))
        if verbose:
            print(out_file)
    return result


def annotate_forecast_guardrail(
    evaluation: Dict[str, object],
    baseline_forecast_crps: float,
    *,
    tolerance: float = 0.001,
) -> Dict[str, object]:
    forecast = evaluation.get("forecast_test")
    summary = evaluation.get("balanced_summary")
    if not isinstance(forecast, dict) or not isinstance(summary, dict):
        return evaluation
    forecast_crps = forecast.get("crps")
    if forecast_crps is None:
        return evaluation
    summary["passes_forecast_guardrail"] = bool(
        float(forecast_crps) <= float(baseline_forecast_crps) + float(tolerance)
    )
    return evaluation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an LLapDiff checkpoint on forecast and target imputation.")
    parser.add_argument("--dataset-key", choices=dataset_keys(), required=True, help="Dataset preset key.")
    parser.add_argument("--pred", type=int, default=None, help="Prediction horizon. Defaults to the longest preset horizon.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path to evaluate.")
    parser.add_argument("--label", type=str, default=None, help="Optional label for the evaluation payload.")
    parser.add_argument("--out-json", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Optional scalar target feature column. Defaults to the dataset cache target_col.",
    )
    parser.add_argument(
        "--target-cols",
        type=str,
        nargs="+",
        default=None,
        help="Optional target feature columns for multi-target evaluation.",
    )
    parser.add_argument("--coverage", type=float, default=0.0, help=COVERAGE_HELP)
    parser.add_argument("--print-json", action="store_true", help="Print the full evaluation JSON to stdout.")
    parser.add_argument("--verbose", action="store_true", help="Print extra evaluation progress details.")
    parser.add_argument("--debug", action="store_true", help="Print verbose diagnostics.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Shared forecast and imputation sample count. Specific sample flags take precedence.",
    )
    parser.add_argument(
        "--forecast-num-samples",
        type=int,
        default=None,
        help="Forecast sample count override.",
    )
    parser.add_argument(
        "--imputation-num-samples",
        type=int,
        default=None,
        help="Imputation sample count override.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Maximum evaluation batches per pass. Use 0 for no cap.",
    )
    parser.add_argument(
        "--imputation-random-mask-ratio",
        type=float,
        default=None,
        help="Fraction of observed target entries hidden in the random-mask imputation case.",
    )
    parser.add_argument(
        "--predict-type",
        type=str,
        default=None,
        help=PREDICT_TYPE_HELP,
    )
    parser.add_argument(
        "--dataset-zip",
        type=str,
        default=None,
        help="Optional zipped dataset cache. Required when the preset cache directory is absent.",
    )
    parser.add_argument(
        "--dataset-extract-dir",
        type=str,
        default=None,
        help="Optional directory for extracting --dataset-zip. Defaults to the user cache directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_dataset_archive(args.dataset_zip, args.dataset_extract_dir)
    pred = int(args.pred) if args.pred is not None else int(default_horizons(args.dataset_key)[-1])
    if args.target_col and args.target_cols:
        raise ValueError("Use either --target-col or --target-cols, not both.")
    cfg = build_eval_config(
        args.dataset_key,
        pred,
        target_col=args.target_col,
        target_cols=args.target_cols,
        coverage=args.coverage,
    )
    apply_verbosity(cfg, verbose=args.verbose, debug=args.debug)
    label = args.label or f"{args.dataset_key}_pred{pred}"
    result = evaluate_checkpoint(
        cfg,
        args.checkpoint,
        label=label,
        out_path=args.out_json,
        random_mask_ratio=args.imputation_random_mask_ratio,
        num_samples=args.num_samples,
        forecast_num_samples=args.forecast_num_samples,
        imputation_num_samples=args.imputation_num_samples,
        max_eval_batches=args.max_eval_batches,
        predict_type=args.predict_type,
        verbose=args.verbose or args.debug,
    )
    if args.print_json:
        print(json.dumps(make_jsonable(result), indent=2))
    else:
        forecast = result.get("forecast_test", {})
        balanced = result.get("balanced_summary", {})
        crps = forecast.get("crps") if isinstance(forecast, dict) else None
        hidden = balanced.get("avg_hidden_crps") if isinstance(balanced, dict) else None
        output = f"{label}: forecast_crps={crps} avg_hidden_crps={hidden}"
        if args.out_json:
            output += f" json={args.out_json}"
        print(output)


if __name__ == "__main__":
    main()
