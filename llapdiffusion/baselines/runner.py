from __future__ import annotations

import csv
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from llapdiffusion.benchmark_protocol import (
    DETERMINISTIC_BASELINE_SEEDS,
    PROBABILISTIC_BASELINE_NUM_SAMPLES,
    baseline_protocol_metadata,
)
from llapdiffusion.baselines.adapters import build_adapter
from llapdiffusion.baselines.data import (
    batch_to_device,
    context_target_mask,
    find_batch,
    load_dataset_loaders,
    target_mask,
)
from llapdiffusion.baselines.features import target_context
from llapdiffusion.baselines.metrics import masked_error_sums, masked_mse, sample_crps_sums
from llapdiffusion.baselines.registry import (
    BASELINES,
)
from llapdiffusion.baselines.sources import SourceManager
from llapdiffusion.configs.dataset_defaults import default_horizons


@dataclass(frozen=True)
class TrainConfig:
    source_root: Path | str | None
    output_dir: Path | str | None = None
    work_cache_dir: Path | str | None = None
    device: str = "auto"
    seed: int = 42
    num_samples: int = PROBABILISTIC_BASELINE_NUM_SAMPLES
    deterministic_seeds: tuple[int, ...] = DETERMINISTIC_BASELINE_SEEDS
    run_suffix: str | None = None
    imputation_random_mask_ratio: float = 0.30
    allow_cache_copy: bool = False
    epochs: int = 600
    patience: int = 20
    lr: float = 1e-3
    horizons: tuple[int, ...] | str | None = "all"
    input_policy: str = "target_only"
    target_col: str | None = None
    target_cols: tuple[str, ...] | None = None
    coverage: float = 0.0
    verbose: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is unavailable")
    return torch.device(requested)


def output_dir(path: Path | str | None) -> Path:
    out = Path(path or "baseline_results").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _validate_imputation_random_mask_ratio(value: float) -> float:
    ratio = float(value)
    if not 0.0 < ratio < 1.0:
        raise ValueError("imputation_random_mask_ratio must be in the open interval (0, 1)")
    return ratio


def _selected_horizons(dataset: str, config: TrainConfig) -> tuple[int, ...]:
    supported = tuple(int(h) for h in default_horizons(dataset))
    requested = config.horizons
    if requested is None:
        return supported
    if isinstance(requested, str):
        if requested != "all":
            raise ValueError("horizons must be 'all' or a sequence of supported integers")
        return supported
    selected = tuple(int(h) for h in requested)
    if not selected:
        raise ValueError("At least one horizon must be selected")
    invalid = [h for h in selected if h not in supported]
    if invalid:
        raise ValueError(f"{dataset}: unsupported horizons {invalid}; supported horizons are {supported}")
    return selected


def notes_payload() -> dict[str, dict[str, object]]:
    return {
        key: {
            "placement": spec.placement,
            "source": f"{spec.source_name}@{spec.source_sha}",
            "dependency_sources": {name: sha for name, sha in spec.dependency_sources},
            "official_reference": spec.official_reference,
            "metric_type": spec.metric_type,
            "time_handling": spec.time_handling,
            **baseline_protocol_metadata(key),
            "dependency_caveat": spec.dependency_caveat,
        }
        for key, spec in BASELINES.items()
    }


def export_notes(path: Path | str) -> Path:
    out = output_dir(path)
    notes_path = out / "baseline_pool_notes.json"
    notes_path.write_text(json.dumps(notes_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return notes_path


def _source_row(source: dict[str, object]) -> dict[str, object]:
    return {
        "source_name": source["source_name"],
        "source_sha": source["source_sha"],
        "source_clean": source["source_clean"],
        "official_reference": source["official_reference"],
        "dependency_caveat": source["dependency_caveat"],
        "dependency_sources": source["dependency_sources"],
    }


def _loss_for_training(model, key: str, batch, dataset_info: dict[str, object]) -> torch.Tensor:
    if key in {"timegrad", "mtan", "mr-diff", "csdi"}:
        return model.loss(batch, dataset_info)
    pred = model(batch, dataset_info)
    _, _, y_clean, valid = target_context(batch, dataset_info)
    return masked_mse(pred, y_clean, valid)


def _has_supervision(
    key: str,
    batch,
    dataset_info: dict[str, object],
) -> bool:
    if key == "csdi":
        has_context = bool(context_target_mask(batch[2], batch[0][0], dataset_info).any().detach().cpu().item())
        has_target = bool(target_mask(batch[2], batch[1]).any().detach().cpu().item())
        return has_context and has_target
    return bool(target_mask(batch[2], batch[1]).any().detach().cpu().item())


def _evaluate_batch(model, key: str, batch, dataset_info: dict[str, object]):
    y = batch[1]
    valid = target_mask(batch[2], y)
    output_shape = None
    sample_shape = None
    target_shape = list(y.shape)
    metric_valid = valid
    metric_target_type = "forecast_horizon"
    metric_crps = None
    metric_mae = None
    metric_sums: dict[str, torch.Tensor | None] | None = None

    if key in {"timegrad", "mr-diff"}:
        loss, samples = model.loss_and_samples(batch, dataset_info)
        sample_shape = list(samples.shape)
        metric_sums = sample_crps_sums(samples, torch.nan_to_num(y, nan=0.0), valid)
        crps = metric_sums["crps_sum"] / metric_sums["count"].clamp_min(1.0)
        mse = metric_sums["sq_sum"] / metric_sums["count"].clamp_min(1.0)
        metric_mae = metric_sums["abs_sum"] / metric_sums["count"].clamp_min(1.0)
        metric_crps, metric_mse = crps, mse
    elif key == "mtan":
        loss = model.loss(batch, dataset_info)
        _, _, samples = model.forward_dist(batch, dataset_info)
        sample_shape = list(samples.shape)
        metric_sums = sample_crps_sums(samples, torch.nan_to_num(y, nan=0.0), valid)
        crps = metric_sums["crps_sum"] / metric_sums["count"].clamp_min(1.0)
        mse = metric_sums["sq_sum"] / metric_sums["count"].clamp_min(1.0)
        metric_mae = metric_sums["abs_sum"] / metric_sums["count"].clamp_min(1.0)
        metric_crps, metric_mse = crps, mse
    elif key == "csdi":
        loss, samples, observed, target_mask_csdi = model.loss_and_samples(batch, dataset_info)
        sample_shape = list(samples.shape)
        target_shape = list(observed.shape)
        metric_valid = target_mask_csdi.to(dtype=torch.bool)
        metric_target_type = getattr(model, "metric_target_type", "target_horizon_imputation")
        metric_sums = sample_crps_sums(samples, observed, metric_valid)
        crps = metric_sums["crps_sum"] / metric_sums["count"].clamp_min(1.0)
        mse = metric_sums["sq_sum"] / metric_sums["count"].clamp_min(1.0)
        metric_mae = metric_sums["abs_sum"] / metric_sums["count"].clamp_min(1.0)
        metric_crps, metric_mse = crps, mse
    else:
        pred = model(batch, dataset_info)
        output_shape = list(pred.shape)
        _, _, y_clean, valid = target_context(batch, dataset_info)
        loss = masked_mse(pred, y_clean, valid)
        metric_sums = masked_error_sums(pred, y_clean, valid)
        metric_mse = (metric_sums["sq_sum"] / metric_sums["count"].clamp_min(1.0)).detach()
        metric_mae = (metric_sums["abs_sum"] / metric_sums["count"].clamp_min(1.0)).detach()
        metric_sums["crps_sum"] = None

    return {
        "loss_tensor": loss,
        "mse_tensor": metric_mse,
        "mae_tensor": metric_mae,
        "crps_tensor": metric_crps,
        "metric_sums_tensor": metric_sums,
        "output_shape": output_shape,
        "sample_shape": sample_shape,
        "target_shape": target_shape,
        "metric_target_type": metric_target_type,
        "valid_observations": int(metric_valid.sum().detach().cpu().item()),
    }


def _jsonable(row: dict[str, object]) -> dict[str, object]:
    return {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v) for k, v in row.items() if not k.endswith("_tensor")}


def _config_payload(config: TrainConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["source_root"] = "<external-baseline-source-root>"
    if payload.get("output_dir") is not None:
        payload["output_dir"] = "<result-output-dir>"
    if payload.get("work_cache_dir") is not None:
        payload["work_cache_dir"] = "<work-cache-dir>"
    return payload


def _seed_run_config(config: TrainConfig, seed: int) -> TrainConfig:
    seed_value = int(seed)
    return replace(config, seed=seed_value, run_suffix=f"seed{seed_value}")


def _mean_std(values: Sequence[object]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not finite:
        return None, None
    mean = float(statistics.fmean(finite))
    std = float(statistics.stdev(finite)) if len(finite) > 1 else 0.0
    return mean, std


def _aggregate_deterministic_seed_rows(seed_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if not seed_rows:
        raise ValueError("Cannot aggregate an empty deterministic seed result set.")

    first = seed_rows[0]
    seeds = [int(row["seed"]) for row in seed_rows]
    aggregate = {key: value for key, value in first.items() if key not in {"history", "checkpoint", "test"}}
    aggregate["seed"] = None
    aggregate["seeds"] = seeds
    aggregate["seed_count"] = len(seeds)
    aggregate["seed_aggregation"] = "mean"
    aggregate["num_samples"] = None
    aggregate["checkpoint"] = [row.get("checkpoint") for row in seed_rows]
    aggregate["runtime_seconds"] = sum(float(row.get("runtime_seconds") or 0.0) for row in seed_rows)

    test_rows = [row.get("test") for row in seed_rows if isinstance(row.get("test"), dict)]
    test = dict(test_rows[0]) if test_rows else {}
    for metric in ("loss", "mse", "mae", "crps"):
        mean, std = _mean_std([row.get(metric) for row in test_rows])
        test[metric] = mean
        test[f"{metric}_std"] = std
    for metric in ("best_epoch", "best_val_mse"):
        mean, std = _mean_std([row.get(metric) for row in seed_rows])
        aggregate[metric] = mean
        aggregate[f"{metric}_std"] = std
    aggregate["test"] = test

    train_config = dict(first.get("train_config") or {})
    train_config["seed"] = None
    train_config["deterministic_seeds"] = seeds
    train_config["run_suffix"] = None
    aggregate["train_config"] = train_config
    return aggregate


def _target_dim(dataset_info: dict[str, object]) -> int:
    return int(dataset_info.get("target_dim") or max(1, len(dataset_info.get("target_cols") or [])))


def _ensure_baseline_supports_targets(baseline: str, dataset_info: dict[str, object]) -> None:
    if _target_dim(dataset_info) <= 1:
        return
    if baseline not in {"dlinear", "patchtst"}:
        raise ValueError(
            f"{baseline} currently supports scalar targets only. "
            "Use dlinear or patchtst for multi-target baseline runs."
        )


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _flatten_csv_row(row: dict[str, object]) -> dict[str, object]:
    flat = dict(row)
    test = row.get("test")
    if isinstance(test, dict):
        for key, value in test.items():
            flat[f"test_{key}"] = value
    loader_batches = row.get("loader_batches")
    if isinstance(loader_batches, dict):
        for key, value in loader_batches.items():
            flat[f"{key}_loader_batches"] = value
    return flat


def write_rows(rows: Sequence[dict[str, object]], output: Path | str, *, prefix: str = "baseline_practical") -> None:
    out = output_dir(output)
    json_rows = [_jsonable(r) for r in rows]
    csv_rows = [_flatten_csv_row(r) for r in json_rows]
    (out / f"{prefix}.json").write_text(json.dumps({"rows": json_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = [
        "status",
        "baseline",
        "placement",
        "dataset",
        "metric_type",
        "metric_target_type",
        "comparison_type",
        "window",
        "horizon",
        "entity_selection_mode",
        "input_policy",
        "target_col",
        "target_cols",
        "target_indices",
        "target_dim",
        "target_source",
        "requested_target_col",
        "requested_target_cols",
        "calendar_feature_cols",
        "input_policy_effective",
        "input_scope",
        "missingness_scope",
        "modeling_scope",
        "split_policy",
        "split_scope",
        "split_note",
        "split_caveat",
        "batching_policy",
        "time_feature_protocol",
        "parameter_count",
        "completion_mode",
        "seed",
        "seeds",
        "seed_count",
        "seed_aggregation",
        "num_samples",
        "eval_replicate_protocol",
        "num_eval_samples",
        "deterministic_seed_count",
        "deterministic_seeds",
        "num_entities_used",
        "valid_observations",
        "loss",
        "mse",
        "mae",
        "crps",
        "best_epoch",
        "best_epoch_std",
        "best_val_mse",
        "best_val_mse_std",
        "test_batches",
        "test_valid_batches",
        "test_raw_batches_scanned",
        "test_valid_observations",
        "test_loss",
        "test_loss_std",
        "test_mse",
        "test_mse_std",
        "test_mae",
        "test_mae_std",
        "test_crps",
        "test_crps_std",
        "test_metric_aggregation",
        "test_loss_aggregation",
        "train_loader_batches",
        "val_loader_batches",
        "test_loader_batches",
        "output_shape",
        "sample_shape",
        "source_sha",
        "copied_cache",
        "runtime_seconds",
        "error",
    ]
    with (out / f"{prefix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    export_notes(out)


def _evaluate_loader(
    model,
    key: str,
    loader,
    dataset_info: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    loss_total = 0.0
    raw_batches = 0
    valid_batches = 0
    valid_observations = 0.0
    abs_sum = 0.0
    sq_sum = 0.0
    crps_sum = 0.0
    crps_count = 0.0
    metric_target_type = "target_horizon_imputation" if key == "csdi" else "forecast_horizon"
    model.eval()
    with torch.no_grad():
        for raw in loader:
            raw_batches += 1
            batch = batch_to_device(raw, device)
            if not _has_supervision(key, batch, dataset_info):
                continue
            row = _evaluate_batch(model, key, batch, dataset_info)
            metric_target_type = str(row.get("metric_target_type", metric_target_type))
            sums = row["metric_sums_tensor"]
            if not isinstance(sums, dict):
                continue
            count_tensor = sums["count"]
            count = float(count_tensor.detach().cpu().item()) if torch.is_tensor(count_tensor) else float(count_tensor)
            if count <= 0:
                continue
            loss_total += float(row["loss_tensor"].detach().cpu().item())
            abs_sum += float(sums["abs_sum"].detach().cpu().item())
            sq_sum += float(sums["sq_sum"].detach().cpu().item())
            crps_value = sums.get("crps_sum")
            if torch.is_tensor(crps_value):
                crps_sum += float(crps_value.detach().cpu().item())
                crps_count += count
            valid_observations += count
            valid_batches += 1
    if valid_batches == 0 or valid_observations <= 0:
        raise RuntimeError(f"{key}/{dataset_info['dataset']}: no valid evaluation batches")
    return {
        "loss": loss_total / valid_batches,
        "mse": sq_sum / valid_observations,
        "mae": abs_sum / valid_observations,
        "crps": crps_sum / crps_count if crps_count > 0 else None,
        "batches": valid_batches,
        "valid_batches": valid_batches,
        "raw_batches_scanned": raw_batches,
        "valid_observations": int(valid_observations),
        "metric_aggregation": "valid_observation_weighted",
        "loss_aggregation": "batch_mean",
        "metric_target_type": metric_target_type,
    }


def _loader_length(loader) -> int | None:
    try:
        return int(len(loader))
    except TypeError:
        return None


def run_practical_one(
    baseline: str,
    dataset: str,
    config: TrainConfig,
    run_root: Path | str,
    *,
    horizon: int | None = None,
) -> dict[str, object]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    if config.target_cols and len(config.target_cols) > 1 and baseline not in {"dlinear", "patchtst"}:
        raise ValueError(
            f"{baseline} currently supports scalar targets only. "
            "Use dlinear or patchtst for multi-target baseline runs."
        )
    source_manager = SourceManager(config.source_root)
    spec = BASELINES[baseline]
    source = source_manager.validate(spec)
    loaders, dataset_info = load_dataset_loaders(
        dataset,
        horizon=horizon,
        allow_cache_copy=config.allow_cache_copy,
        work_cache_dir=Path(config.work_cache_dir).expanduser().resolve() if config.work_cache_dir else None,
        target_col=config.target_col,
        target_cols=config.target_cols,
        coverage=config.coverage,
    )
    train_dl, val_dl, test_dl = loaders
    dataset_info["input_policy"] = str(config.input_policy)
    _ensure_baseline_supports_targets(baseline, dataset_info)
    sample_batch, _ = find_batch(
        train_dl,
        dataset_info,
        device,
    )
    model = build_adapter(
        baseline,
        dataset_info,
        sample_batch,
        source_manager,
        device,
        num_samples=config.num_samples,
        imputation_random_mask_ratio=_validate_imputation_random_mask_ratio(config.imputation_random_mask_ratio),
    ).to(device)
    parameter_count = _parameter_count(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    run_name = f"{baseline}_{dataset}_h{dataset_info['horizon']}"
    if config.run_suffix:
        run_name = f"{run_name}_{config.run_suffix}"
    run_dir = output_dir(run_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    history = []
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for raw in train_dl:
            batch = batch_to_device(raw, device)
            if not _has_supervision(baseline, batch, dataset_info):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_training(model, baseline, batch, dataset_info)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{baseline}/{dataset}: non-finite train loss")
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu().item())
            train_batches += 1
        if train_batches == 0:
            raise RuntimeError(f"{baseline}/{dataset}: no valid training batches")

        val = _evaluate_loader(model, baseline, val_dl, dataset_info, device)
        epoch_row = {"epoch": epoch, "train_loss": train_loss / train_batches, "train_batches": train_batches, "val": val}
        history.append(epoch_row)
        if val["mse"] < best_val:
            best_val = val["mse"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), best_path)
            if config.verbose:
                print(
                    f"{baseline}/{dataset}/h{dataset_info['horizon']} "
                    f"epoch={epoch} train_loss={epoch_row['train_loss']:.6f} val_mse={val['mse']:.6f} best=1",
                    flush=True,
                )
        else:
            stale += 1
            if config.verbose:
                print(
                    f"{baseline}/{dataset}/h{dataset_info['horizon']} "
                    f"epoch={epoch} train_loss={epoch_row['train_loss']:.6f} val_mse={val['mse']:.6f}",
                    flush=True,
                )
        if stale >= config.patience:
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test = _evaluate_loader(model, baseline, test_dl, dataset_info, device)
    num_entities_used = int(sample_batch[0][0].shape[1])
    loader_batches = {"train": _loader_length(train_dl), "val": _loader_length(val_dl), "test": _loader_length(test_dl)}
    protocol = baseline_protocol_metadata(baseline, requested_input_policy=config.input_policy)
    result = {
        "status": "ok",
        "baseline": baseline,
        "placement": spec.placement,
        "dataset": dataset,
        "metric_type": spec.metric_type,
        "metric_target_type": test.get("metric_target_type", "target_horizon_imputation" if baseline == "csdi" else "forecast_horizon"),
        **protocol,
        "window": dataset_info["window"],
        "horizon": dataset_info["horizon"],
        "dataset_lengths": dataset_info["lengths"],
        "copied_cache": dataset_info["copied_cache"],
        "entity_selection_mode": "full_panel",
        "num_entities_used": num_entities_used,
        "loader_batches": loader_batches,
        "train_config": _config_payload(config),
        "input_policy": str(config.input_policy),
        "target_col": dataset_info.get("target_col"),
        "target_cols": dataset_info.get("target_cols"),
        "target_indices": dataset_info.get("target_indices"),
        "target_dim": dataset_info.get("target_dim"),
        "target_source": dataset_info.get("target_source"),
        "requested_target_col": dataset_info.get("requested_target_col"),
        "requested_target_cols": dataset_info.get("requested_target_cols"),
        "calendar_feature_cols": dataset_info.get("calendar_feature_cols"),
        "split_policy": dataset_info.get("split_policy", "global_purged_horizon"),
        "split_scope": dataset_info.get("split_scope", "global_target_time"),
        "split_note": dataset_info.get("split_note", ""),
        "split_caveat": dataset_info.get("split_caveat", ""),
        "batching_policy": dataset_info.get("batching_policy", "exact_context_end_timestamp"),
        "parameter_count": parameter_count,
        "completion_mode": "full_train_loop",
        "seed": int(config.seed),
        "seeds": [int(config.seed)],
        "seed_count": 1,
        "seed_aggregation": "single_seed",
        "num_samples": int(config.num_samples) if bool(spec.probabilistic) else None,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "history": history,
        "test": test,
        "checkpoint": str(best_path),
        "runtime_seconds": time.time() - start,
        "device": str(device),
        **_source_row(source),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return result


def run_practical_matrix(baselines: Sequence[str], datasets: Sequence[str], config: TrainConfig, run_root: Path | str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    deterministic_seed_rows: list[dict[str, object]] = []
    for dataset in datasets:
        for horizon in _selected_horizons(dataset, config):
            for baseline in baselines:
                if BASELINES[baseline].probabilistic:
                    rows.append(run_practical_one(baseline, dataset, config, run_root, horizon=horizon))
                    continue
                seed_rows = [
                    run_practical_one(baseline, dataset, _seed_run_config(config, seed), run_root, horizon=horizon)
                    for seed in config.deterministic_seeds
                ]
                deterministic_seed_rows.extend(seed_rows)
                rows.append(_aggregate_deterministic_seed_rows(seed_rows))
    write_rows(rows, run_root, prefix="baseline_practical")
    if deterministic_seed_rows:
        write_rows(deterministic_seed_rows, run_root, prefix="baseline_practical_seed_rows")
    return rows
