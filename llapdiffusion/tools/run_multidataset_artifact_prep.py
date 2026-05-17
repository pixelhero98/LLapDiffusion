from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

from llapdiffusion.trainers import train_val_latent, train_val_summarizer
from llapdiffusion.configs.config_utils import clone_config, make_jsonable
from llapdiffusion.configs.dataset_archives import configure_dataset_archive
from llapdiffusion.configs.dataset_defaults import (
    apply_dataset_preset,
    dataset_keys,
    get_dataset_preset,
    validate_dataset_presets,
)
from llapdiffusion.configs.dataset_registry import resolve_run_experiment


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


def _artifact_config(dataset_key: str, pred: int):
    cfg = clone_config()
    apply_dataset_preset(cfg, dataset_key, pred=int(pred))
    # Artifact prep prioritizes stable loader semantics over the table batch-size row.
    cfg.BATCH_SIZE = 1
    cfg.DATES_PER_BATCH = 1
    return cfg


def _build_loaders(cfg):
    run_experiment = resolve_run_experiment(cfg.DATA_DIR)
    return run_experiment(
        data_dir=cfg.DATA_DIR,
        date_batching=cfg.date_batching,
        dates_per_batch=cfg.DATES_PER_BATCH,
        K=cfg.WINDOW,
        H=cfg.PRED,
        coverage=cfg.COVERAGE,
        ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio),
    )


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
    print(path)


def main() -> None:
    args = _parse_args()
    configure_dataset_archive(args.dataset_zip, args.dataset_extract_dir)
    spec_validation = _validate_dataset_specs()

    records = []
    for dataset_key in args.datasets:
        preset = get_dataset_preset(dataset_key)
        for pred in preset.horizons:
            cfg = _artifact_config(dataset_key, int(pred))
            records.append(
                {
                    "dataset": dataset_key,
                    "artifact_name": preset.artifact_name,
                    "data_dir": cfg.DATA_DIR,
                    "pred": int(pred),
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
        print(json.dumps(make_jsonable(payload), indent=2))
        return

    executed_records = []
    for dataset_key in args.datasets:
        preset = get_dataset_preset(dataset_key)
        for pred in preset.horizons:
            cfg = _artifact_config(dataset_key, int(pred))
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
    print(json.dumps(make_jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
