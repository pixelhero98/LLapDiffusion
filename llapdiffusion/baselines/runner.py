from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

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
    DATASET_KEYS,
    EXTRAPOLATION_BASELINES,
)
from llapdiffusion.baselines.sources import SourceManager
from llapdiffusion.configs.dataset_defaults import default_horizons


@dataclass(frozen=True)
class TrainConfig:
    source_root: Path | str | None
    output_dir: Path | str | None = None
    work_cache_dir: Path | str | None = None
    device: str = "cuda"
    seed: int = 42
    num_samples: int = 25
    imputation_random_mask_ratio: float = 0.30
    allow_cache_copy: bool = False
    epochs: int = 600
    patience: int = 50
    lr: float = 1e-3
    horizons: tuple[int, ...] | str | None = "all"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
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
        "window",
        "horizon",
        "entity_selection_mode",
        "num_entities_used",
        "valid_observations",
        "loss",
        "mse",
        "mae",
        "crps",
        "best_epoch",
        "best_val_mse",
        "test_batches",
        "test_valid_batches",
        "test_raw_batches_scanned",
        "test_valid_observations",
        "test_loss",
        "test_mse",
        "test_mae",
        "test_crps",
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
    source_manager = SourceManager(config.source_root)
    spec = BASELINES[baseline]
    source = source_manager.validate(spec)
    loaders, dataset_info = load_dataset_loaders(
        dataset,
        horizon=horizon,
        allow_cache_copy=config.allow_cache_copy,
        work_cache_dir=Path(config.work_cache_dir).expanduser().resolve() if config.work_cache_dir else None,
    )
    train_dl, val_dl, test_dl = loaders
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    run_dir = output_dir(run_root) / f"{baseline}_{dataset}_h{dataset_info['horizon']}"
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
        print(
            f"{baseline}/{dataset}/h{dataset_info['horizon']} "
            f"epoch={epoch} train_loss={epoch_row['train_loss']:.6f} val_mse={val['mse']:.6f}",
            flush=True,
        )
        if val["mse"] < best_val:
            best_val = val["mse"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
        if stale >= config.patience:
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test = _evaluate_loader(model, baseline, test_dl, dataset_info, device)
    num_entities_used = int(sample_batch[0][0].shape[1])
    loader_batches = {"train": _loader_length(train_dl), "val": _loader_length(val_dl), "test": _loader_length(test_dl)}
    result = {
        "status": "ok",
        "baseline": baseline,
        "placement": spec.placement,
        "dataset": dataset,
        "metric_type": spec.metric_type,
        "metric_target_type": test.get("metric_target_type", "target_horizon_imputation" if baseline == "csdi" else "forecast_horizon"),
        "window": dataset_info["window"],
        "horizon": dataset_info["horizon"],
        "dataset_lengths": dataset_info["lengths"],
        "copied_cache": dataset_info["copied_cache"],
        "entity_selection_mode": "full_panel",
        "num_entities_used": num_entities_used,
        "loader_batches": loader_batches,
        "train_config": _config_payload(config),
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
    rows = [
        run_practical_one(baseline, dataset, config, run_root, horizon=horizon)
        for dataset in datasets
        for horizon in _selected_horizons(dataset, config)
        for baseline in baselines
    ]
    write_rows(rows, run_root, prefix="baseline_practical")
    return rows


def write_isambard_jobs(
    output: Path | str,
    *,
    source_root: str | None = None,
    datasets: Iterable[str] = DATASET_KEYS,
    baselines: Iterable[str] = EXTRAPOLATION_BASELINES,
    horizons: Iterable[int] | str | None = None,
    time_limit: str = "08:00:00",
) -> list[Path]:
    out = output_dir(output)
    scripts = []
    selected_baselines = tuple(baselines)
    if any(not BASELINES[baseline].first_party for baseline in selected_baselines) and not source_root:
        raise ValueError("External baseline jobs require --baseline-source-root.")
    for dataset in datasets:
        if horizons is None:
            selected_horizons = default_horizons(dataset)
        elif isinstance(horizons, str):
            if horizons != "all":
                raise ValueError("horizons must be 'all' or a sequence of supported integers")
            selected_horizons = default_horizons(dataset)
        else:
            selected_horizons = tuple(int(h) for h in horizons)
        if not selected_horizons:
            raise ValueError("At least one horizon must be selected")
        supported_horizons = default_horizons(dataset)
        supported = set(supported_horizons)
        invalid = [h for h in selected_horizons if h not in supported]
        if invalid:
            raise ValueError(f"{dataset}: unsupported horizons {invalid}; supported horizons are {supported_horizons}")

        for horizon in selected_horizons:
            for baseline in selected_baselines:
                source_arg = f"--baseline-source-root {source_root} " if source_root and not BASELINES[baseline].first_party else ""
                script = out / f"llapdiff_{baseline}_{dataset}_h{horizon}.sh"
                script.write_text(
                    "#!/bin/bash\n"
                    f"#SBATCH --job-name=llap-{baseline}-{dataset}-h{horizon}\n"
                    "#SBATCH --partition=hopper\n"
                    "#SBATCH --nodes=1\n"
                    "#SBATCH --gpus=1\n"
                    f"#SBATCH --time={time_limit}\n"
                    "#SBATCH --output=%x.%j.out\n\n"
                    "set -euo pipefail\n"
                    "nvidia-smi\n"
                    "llapdiff-baselines practical-extrapolation "
                    f"--baseline {baseline} --dataset {dataset} --horizons {horizon} "
                    f"{source_arg}"
                    "--output-dir ${SCRATCHDIR:-$PWD}/llapdiffusion_baselines/runs "
                    "--allow-cache-copy --work-cache-dir ${SCRATCHDIR:-$PWD}/llapdiffusion_baselines/cache_work\n",
                    encoding="utf-8",
                )
                scripts.append(script)
    return scripts
