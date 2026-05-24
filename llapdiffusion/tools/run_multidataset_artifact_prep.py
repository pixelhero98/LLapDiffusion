from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from llapdiffusion.trainers import train_val_latent, train_val_summarizer
from llapdiffusion.configs.config_utils import clone_config, make_jsonable
from llapdiffusion.configs.dataset_archives import configure_dataset_archive
from llapdiffusion.logging_utils import apply_verbosity
from llapdiffusion.configs.dataset_defaults import (
    apply_dataset_preset,
    dataset_keys,
    get_dataset_preset,
    validate_dataset_presets,
)
from llapdiffusion.configs.dataset_registry import resolve_run_experiment
from llapdiffusion.datasets.target_selection import resolve_target_selection


COVERAGE_HELP = "fraction of observed context entries to hide; 0 disables induced missingness"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VAE and summarizer artifacts using the public dataset presets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(dataset_keys()),
        default=tuple(dataset_keys()),
        help="Datasets to prepare. Defaults to all supported public datasets.",
    )
    parser.add_argument(
        "--recompute-vae",
        action="store_true",
        help="Force retraining VAE checkpoints even if they already exist.",
    )
    parser.add_argument(
        "--recompute-summarizer",
        action="store_true",
        help="Force retraining summarizer checkpoints even if they already exist.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="ldt/results/multidataset_artifact_prep_summary.json",
        help="Path to write the combined prep summary JSON.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Optional scalar target feature column. Defaults to each dataset cache target_col.",
    )
    parser.add_argument(
        "--target-cols",
        type=str,
        nargs="+",
        default=None,
        help="Optional target feature columns applied to every selected dataset.",
    )
    parser.add_argument(
        "--target-cols-map",
        type=str,
        default=None,
        help="JSON object or JSON file mapping dataset keys to target column lists.",
    )
    parser.add_argument("--coverage", type=float, default=0.0, help=COVERAGE_HELP)
    parser.add_argument("--print-json", action="store_true", help="Print the full summary JSON to stdout.")
    parser.add_argument("--verbose", action="store_true", help="Print trainer diagnostics.")
    parser.add_argument("--debug", action="store_true", help="Print verbose trainer diagnostics.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset specs and emit the planned work without starting any training jobs.",
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


def _validate_dataset_specs() -> Dict[str, object]:
    return validate_dataset_presets(dataset_keys())


def _coerce_target_cols(value: object) -> tuple[str, ...] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
    elif isinstance(value, Sequence):
        raw = [str(part).strip() for part in value]
    else:
        raw = [str(value).strip()]
    cols = tuple(part for part in raw if part)
    return cols or None


def _validate_coverage(value: object) -> float:
    coverage = float(value)
    if not 0.0 <= coverage < 1.0:
        raise ValueError("--coverage must satisfy 0 <= coverage < 1.")
    return coverage


def _effective_batch_size(cfg) -> int:
    fallback = getattr(cfg, "DATES_PER_BATCH", 1)
    batch_size = int(getattr(cfg, "BATCH_SIZE", fallback))
    if batch_size < 1:
        raise ValueError("BATCH_SIZE must be a positive integer.")
    return batch_size


def _load_target_cols_map(value: str | None) -> dict[str, tuple[str, ...]]:
    if not value:
        return {}
    candidate = Path(value)
    try:
        text = candidate.read_text(encoding="utf-8") if candidate.exists() else value
    except OSError:
        text = value
    parsed = json.loads(text)
    if not isinstance(parsed, Mapping):
        raise ValueError("--target-cols-map must be a JSON object.")
    known_dataset_keys = set(dataset_keys())
    out: dict[str, tuple[str, ...]] = {}
    for dataset_key, cols in parsed.items():
        dataset_key = str(dataset_key)
        if dataset_key not in known_dataset_keys:
            raise ValueError(
                f"--target-cols-map contains unknown dataset key {dataset_key!r}; "
                f"supported keys are {tuple(sorted(known_dataset_keys))}."
            )
        coerced = _coerce_target_cols(cols)
        if not coerced:
            raise ValueError(f"--target-cols-map entry for {dataset_key!r} is empty.")
        out[dataset_key] = coerced
    return out


def _target_cols_for_dataset(
    dataset_key: str,
    *,
    global_target_cols: Sequence[str] | None,
    target_cols_map: Mapping[str, Sequence[str]],
) -> tuple[str, ...] | None:
    if dataset_key in target_cols_map:
        return tuple(target_cols_map[dataset_key])
    return tuple(global_target_cols) if global_target_cols else None


def _artifact_config(
    dataset_key: str,
    pred: int,
    *,
    target_col: str | None = None,
    target_cols: Sequence[str] | None = None,
    coverage: float = 0.0,
):
    cfg = clone_config()
    apply_dataset_preset(cfg, dataset_key, pred=int(pred))
    if target_col and target_cols:
        raise ValueError("Use either target_col or target_cols, not both.")
    cfg.TARGET_COL = target_col
    cfg.TARGET_COLS = list(target_cols) if target_cols else None
    cfg.COVERAGE = _validate_coverage(coverage)
    cfg.DATES_PER_BATCH = _effective_batch_size(cfg)
    return cfg


def _build_loaders(cfg):
    run_experiment = resolve_run_experiment(cfg.DATA_DIR)
    batch_size = _effective_batch_size(cfg)
    return run_experiment(
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


def _target_policy(cfg) -> Dict[str, object]:
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


def _select_vae_checkpoint(stage_stats: Dict[str, object], cfg) -> str:
    for key in ("best_elbo_path", "loaded_checkpoint", "best_recon_path"):
        value = stage_stats.get(key)
        if value:
            return str(value)
    return str(cfg.VAE_CKPT)


def _select_summarizer_checkpoint(stage_stats: Dict[str, object], cfg) -> str:
    checkpoint = stage_stats.get("checkpoint")
    if checkpoint:
        return str(checkpoint)
    return str(cfg.SUM_CKPT)


def _compact_vae_stage(stage_stats: Dict[str, object], *, trained: bool) -> Dict[str, object]:
    if stage_stats.get("status") == "skipped":
        return {
            "action": "reused",
            "reason": stage_stats.get("reason"),
            "loaded_checkpoint": stage_stats.get("loaded_checkpoint") or stage_stats.get("best_elbo_path"),
            "best_elbo_path": stage_stats.get("best_elbo_path"),
            "best_recon_path": stage_stats.get("best_recon_path"),
        }
    return {
        "action": "trained" if trained else "reused",
        "best_val_elbo": stage_stats.get("best_val_elbo"),
        "best_val_recon": stage_stats.get("best_val_recon"),
        "best_elbo_path": stage_stats.get("best_elbo_path"),
        "best_recon_path": stage_stats.get("best_recon_path"),
        "loaded_checkpoint": stage_stats.get("loaded_checkpoint"),
        "final_val_metrics": stage_stats.get("final_val_metrics"),
        "final_test_metrics": stage_stats.get("final_test_metrics"),
    }


def _compact_summarizer_stage(stage_stats: Dict[str, object], *, trained: bool) -> Dict[str, object]:
    if stage_stats.get("status") == "skipped":
        return {
            "action": "reused",
            "reason": stage_stats.get("reason"),
            "checkpoint": stage_stats.get("checkpoint"),
        }
    return {
        "action": "trained" if trained else "reused",
        "best_val": stage_stats.get("best_val"),
        "val_loss": stage_stats.get("val_loss"),
        "test_loss": stage_stats.get("test_loss"),
        "checkpoint": stage_stats.get("checkpoint"),
        "skipped_nonfinite_grad_steps": stage_stats.get("skipped_nonfinite_grad_steps"),
        "sum_max_nonfinite_grad_steps": stage_stats.get("sum_max_nonfinite_grad_steps"),
    }


def _status_counts(records: Iterable[Dict[str, object]]) -> Dict[str, int]:
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for record in records:
        for stage_name in ("vae", "summarizer"):
            status = str(record[stage_name]["audit"].get("status", "fail"))
            counts[status] = counts.get(status, 0) + 1
    return counts


def _write_summary(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_jsonable(payload), indent=2))


def main() -> None:
    args = _parse_args()
    configure_dataset_archive(args.dataset_zip, args.dataset_extract_dir)
    if args.target_col and (args.target_cols or args.target_cols_map):
        raise ValueError("Use either --target-col or --target-cols/--target-cols-map, not both.")
    target_cols_map = _load_target_cols_map(args.target_cols_map)
    coverage = _validate_coverage(args.coverage)
    spec_validation = _validate_dataset_specs()

    records = []
    for dataset_key in args.datasets:
        preset = get_dataset_preset(dataset_key)
        for pred in preset.horizons:
            target_cols = _target_cols_for_dataset(
                dataset_key,
                global_target_cols=args.target_cols,
                target_cols_map=target_cols_map,
            )
            cfg = _artifact_config(
                dataset_key,
                int(pred),
                target_col=args.target_col,
                target_cols=target_cols,
                coverage=coverage,
            )
            records.append(
                {
                    "dataset": dataset_key,
                    "artifact_name": preset.artifact_name,
                    "data_dir": cfg.DATA_DIR,
                    "pred": int(pred),
                    **_target_policy(cfg),
                    "fixed_context": preset.context_length,
                    "vae_latent_channels": preset.vae_latent_channels,
                    "vae_checkpoint": cfg.VAE_CKPT,
                    "summarizer_checkpoint": cfg.SUM_CKPT,
                }
            )

    summary_path = Path(args.summary_json)
    if args.dry_run:
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "dry_run",
            "spec_validation": spec_validation,
            "planned_records": records,
        }
        _write_summary(summary_path, payload)
        if args.print_json:
            print(json.dumps(make_jsonable(payload), indent=2))
        else:
            print(f"dry_run: planned_records={len(records)} summary_json={summary_path}")
        return

    executed_records = []
    for dataset_key in args.datasets:
        preset = get_dataset_preset(dataset_key)
        for pred in preset.horizons:
            target_cols = _target_cols_for_dataset(
                dataset_key,
                global_target_cols=args.target_cols,
                target_cols_map=target_cols_map,
            )
            cfg = _artifact_config(
                dataset_key,
                int(pred),
                target_col=args.target_col,
                target_cols=target_cols,
                coverage=coverage,
            )
            apply_verbosity(cfg, verbose=args.verbose, debug=args.debug)
            if args.verbose or args.debug:
                print(f"\n=== dataset={dataset_key} pred={pred} context={preset.context_length} ===")
            train_dl, val_dl, test_dl, sizes = _build_loaders(cfg)

            vae_checkpoint = Path(cfg.VAE_CKPT)
            run_vae = args.recompute_vae or not vae_checkpoint.exists()
            if run_vae:
                vae_stats = train_val_latent.run(
                    train_dl=train_dl,
                    val_dl=val_dl,
                    test_dl=test_dl,
                    sizes=sizes,
                    config=cfg,
                )
            else:
                vae_stats = {
                    "status": "skipped",
                    "reason": "checkpoint_exists",
                    "loaded_checkpoint": str(vae_checkpoint),
                    "best_elbo_path": str(vae_checkpoint),
                    "best_recon_path": str(vae_checkpoint.with_name(vae_checkpoint.name.replace("_elbo.pt", "_recon.pt"))),
                }
            vae_ckpt = _select_vae_checkpoint(vae_stats, cfg)
            vae_audit = train_val_latent.audit_checkpoint(
                vae_ckpt,
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                sizes=sizes,
                config=cfg,
            )

            summ_checkpoint = Path(cfg.SUM_CKPT)
            run_summarizer = args.recompute_summarizer or not summ_checkpoint.exists()
            if run_summarizer:
                summ_stats = train_val_summarizer.run(
                    train_loader=train_dl,
                    val_loader=val_dl,
                    test_loader=test_dl,
                    sizes=sizes,
                    config=cfg,
                )
            else:
                summ_stats = {
                    "status": "skipped",
                    "reason": "checkpoint_exists",
                    "checkpoint": str(summ_checkpoint),
                }
            summ_ckpt = _select_summarizer_checkpoint(summ_stats, cfg)
            summ_audit = train_val_summarizer.evaluate_checkpoint(
                summ_ckpt,
                train_loader=train_dl,
                val_loader=val_dl,
                test_loader=test_dl,
                sizes=sizes,
                config=cfg,
            )

            executed_records.append(
                {
                    "dataset": dataset_key,
                    "artifact_name": preset.artifact_name,
                    "data_dir": cfg.DATA_DIR,
                    "pred": int(pred),
                    **_target_policy(cfg),
                    "fixed_context": preset.context_length,
                    "vae_latent_channels": preset.vae_latent_channels,
                    "vae": {
                        "stage": _compact_vae_stage(vae_stats, trained=run_vae),
                        "audit": vae_audit,
                    },
                    "summarizer": {
                        "stage": _compact_summarizer_stage(summ_stats, trained=run_summarizer),
                        "audit": summ_audit,
                    },
                }
            )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "spec_validation": spec_validation,
        "records": executed_records,
        "status_counts": _status_counts(executed_records),
    }
    _write_summary(summary_path, payload)
    if args.print_json:
        print(json.dumps(make_jsonable(payload), indent=2))
    else:
        print(f"completed: records={len(executed_records)} summary_json={summary_path}")


if __name__ == "__main__":
    main()
