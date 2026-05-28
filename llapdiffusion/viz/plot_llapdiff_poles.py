"""Plot global and conditioned LLapDiff Laplace poles for a checkpoint."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from llapdiffusion.configs.config_utils import clone_config
from llapdiffusion.configs.dataset_archives import configure_dataset_archive
from llapdiffusion.configs.dataset_defaults import apply_dataset_preset, dataset_keys
from llapdiffusion.configs.dataset_registry import resolve_run_experiment
from llapdiffusion.models.llapdiff import LLapDiff
from llapdiffusion.models.llapdiff_utils import EMA, set_torch
from llapdiffusion.models.summarizer import LaplaceAE
from llapdiffusion.trainers import train_val_llapdiff as tv


CHECKPOINT_KINDS = ("raw", "ema", "best", "last", "auto")
SPLITS = ("train", "val", "test")
COVERAGE_HELP = "fraction of observed context entries to hide; 0 disables induced missingness"


def _infer_dataset_key_from_path(path: Path) -> Optional[str]:
    path_str = str(path).lower()
    matches = [key for key in dataset_keys() if key in path_str]
    if not matches:
        return None
    return sorted(matches, key=len, reverse=True)[0]


def _infer_pred_from_checkpoint(path: Path) -> Optional[int]:
    match = re.search(r"pred-(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _is_ema_checkpoint(path: Path) -> bool:
    return path.stem.endswith("_best_ema")


def _build_config(dataset_key: str, pred: int):
    cfg = clone_config()
    apply_dataset_preset(cfg, dataset_key, pred=pred)
    return cfg


def _validate_coverage(value: object) -> float:
    coverage = float(value)
    if not 0.0 <= coverage < 1.0:
        raise ValueError("--coverage must satisfy 0 <= coverage < 1.")
    return coverage


def _checkpoint_filename(pred: int, kind: str) -> str:
    if kind == "raw":
        return f"llapdiff_pred-{pred}_best_raw.pt"
    if kind == "ema":
        return f"llapdiff_pred-{pred}_best_ema.pt"
    if kind == "best":
        return f"llapdiff_pred-{pred}_best.pt"
    if kind == "last":
        return f"llapdiff_pred-{pred}_last.pt"
    raise ValueError(f"Unsupported checkpoint kind: {kind}")


def _candidate_checkpoint_roots(cfg) -> Tuple[Path, ...]:
    roots = []
    for value in (getattr(cfg, "OUT_DIR", ""), getattr(cfg, "CKPT_DIR", "")):
        if value:
            path = Path(str(value)).expanduser().resolve()
            if path not in roots:
                roots.append(path)
    return tuple(roots)


def _find_checkpoint_by_kind(cfg, pred: int, kind: str) -> Optional[Path]:
    filename = _checkpoint_filename(pred, kind)
    roots = _candidate_checkpoint_roots(cfg)

    for root in roots:
        exact = root / filename
        if exact.exists():
            return exact

    matches = []
    for root in roots:
        if root.exists():
            matches.extend(root.rglob(filename))

    unique = sorted({path.resolve() for path in matches}, key=lambda path: str(path))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        choices = "\n".join(str(path) for path in unique)
        raise ValueError(
            f"Multiple {kind} checkpoints found for pred={pred}. "
            f"Pass --checkpoint explicitly.\n{choices}"
        )
    return None


def _resolve_checkpoint(explicit: Optional[str], cfg, *, pred: int, checkpoint_kind: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    kinds: Iterable[str]
    if checkpoint_kind == "auto":
        kinds = ("raw", "ema", "best", "last")
    else:
        kinds = (checkpoint_kind,)

    for kind in kinds:
        checkpoint = _find_checkpoint_by_kind(cfg, pred, kind)
        if checkpoint is not None:
            return checkpoint

    roots = ", ".join(str(path) for path in _candidate_checkpoint_roots(cfg))
    raise FileNotFoundError(
        f"No {checkpoint_kind} checkpoint found for pred={pred} under: {roots}. "
        "Pass --checkpoint explicitly."
    )


def _state_dict_from_checkpoint(payload: object) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        return payload
    state_dict = payload.get("model") or payload.get("model_state") or payload.get("state_dict")
    if state_dict is None and any(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    if state_dict is None:
        raise ValueError("Checkpoint does not contain model weights.")
    return state_dict


def _read_model_config(payload: object) -> Dict[str, object]:
    if not isinstance(payload, dict):
        return {"rho_conditioning_mode": "legacy_effective"}
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        return {"rho_conditioning_mode": "legacy_effective"}
    llapdiff_config = model_config.get("llapdiff")
    if isinstance(llapdiff_config, dict):
        config = dict(llapdiff_config)
    else:
        config = {
            key: value
            for key, value in model_config.items()
            if key != "cond_adapter"
        }
    config.setdefault("rho_conditioning_mode", "legacy_effective")
    return config


def _infer_model_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, object]:
    inferred: Dict[str, object] = {}

    time_embed = state_dict.get("time_embed.weight")
    if isinstance(time_embed, torch.Tensor) and time_embed.dim() == 2:
        inferred["timesteps"] = int(time_embed.shape[0])
        inferred["hidden_dim"] = int(time_embed.shape[1])

    rho_raw = state_dict.get("model.analysis._rho_raw")
    if isinstance(rho_raw, torch.Tensor) and rho_raw.dim() == 1:
        inferred["laplace_k"] = int(rho_raw.shape[0])

    head_weight = state_dict.get("model.head_proj.weight")
    if isinstance(head_weight, torch.Tensor) and head_weight.dim() == 2:
        inferred["data_dim"] = int(head_weight.shape[0])

    block_ids = set()
    block_pattern = re.compile(r"model\.blocks\.(\d+)\.")
    for name in state_dict:
        match = block_pattern.match(name)
        if match:
            block_ids.add(int(match.group(1)))
    if block_ids:
        inferred["num_layers"] = max(block_ids) + 1

    if any(name.startswith("model.summary_pool.") for name in state_dict):
        inferred["summary_pool_mode"] = "attn"
    if any(name.startswith("model.analysis.q_cond_proj.") for name in state_dict):
        inferred["analysis_summary_qk"] = True

    adaln_weight = state_dict.get("model.blocks.0.self_blk.norm1.to_ss.1.weight")
    hidden_dim = inferred.get("hidden_dim")
    if isinstance(adaln_weight, torch.Tensor) and hidden_dim is not None:
        inferred["block_summary_adaln"] = int(adaln_weight.shape[1]) == 2 * int(hidden_dim)

    inferred["self_conditioning"] = "model.sc_gate" in state_dict
    return inferred


def _model_kwargs(cfg, payload: object, state_dict: Dict[str, torch.Tensor]) -> Dict[str, object]:
    kwargs = {
        "data_dim": int(getattr(cfg, "VAE_LATENT_CHANNELS")),
        "hidden_dim": int(getattr(cfg, "MODEL_WIDTH")),
        "num_layers": int(getattr(cfg, "NUM_LAYERS")),
        "num_heads": int(getattr(cfg, "NUM_HEADS")),
        "predict_type": str(getattr(cfg, "PREDICT_TYPE")),
        "laplace_k": int(getattr(cfg, "LAPLACE_K")),
        "timesteps": int(getattr(cfg, "TIMESTEPS")),
        "schedule": str(getattr(cfg, "SCHEDULE")),
        "dropout": float(getattr(cfg, "DROPOUT")),
        "attn_dropout": float(getattr(cfg, "ATTN_DROPOUT")),
        "self_conditioning": bool(getattr(cfg, "SELF_COND")),
        "summary_pool_mode": str(getattr(cfg, "COND_POOL_MODE", "mean")),
        "pole_pool_use_raw_summary": bool(getattr(cfg, "COND_POOL_USE_RAW", False)),
        "block_summary_adaln": bool(getattr(cfg, "BLOCK_SUMMARY_ADALN", False)),
        "analysis_summary_qk": bool(getattr(cfg, "ANALYSIS_SUMMARY_QK", False)),
        "analysis_qk_use_raw_summary": bool(getattr(cfg, "ANALYSIS_QK_USE_RAW", False)),
        "rho_conditioning_mode": str(getattr(cfg, "RHO_CONDITIONING_MODE", "raw")),
    }
    kwargs.update(_infer_model_config_from_state_dict(state_dict))
    kwargs.update(_read_model_config(payload))
    return kwargs


def _load_model_from_checkpoint(
    cfg,
    checkpoint: Path,
    *,
    device: torch.device,
    ema_decay: float,
    use_ema: bool,
) -> Tuple[LLapDiff, Dict[str, object]]:
    payload = torch.load(checkpoint, map_location=device)
    state_dict = _state_dict_from_checkpoint(payload)
    model_kwargs = _model_kwargs(cfg, payload, state_dict)
    model = LLapDiff(**model_kwargs).to(device)
    model.load_state_dict(state_dict, strict=True)

    ema_state = payload.get("ema") or payload.get("ema_state") if isinstance(payload, dict) else None
    if use_ema and ema_state is not None:
        ema = EMA(model, decay=float(payload.get("ema_decay", ema_decay)) if isinstance(payload, dict) else ema_decay)
        ema.load_state_dict(ema_state)
        ema.copy_to(model)

    model.eval()
    return model, model_kwargs


def _align_config_to_model(cfg, pred: int, model_kwargs: Dict[str, object]) -> None:
    data_dim = model_kwargs.get("data_dim")
    if data_dim is not None:
        cfg.VAE_LATENT_CHANNELS = int(data_dim)
        vae_suffix = "_entity" if bool(getattr(cfg, "VAE_ENTITY_CONDITION", False)) else ""
        cfg.VAE_CKPT = str(
            Path(cfg.VAE_DIR) / f"pred-{int(pred)}_ch-{int(data_dim)}{vae_suffix}_elbo.pt"
        )
        cfg.SUM_CKPT = str(
            Path(cfg.SUM_DIR) / f"{int(pred)}-{int(data_dim)}-summarizer.pt"
        )
    if model_kwargs.get("laplace_k") is not None:
        cfg.LAPLACE_K = int(model_kwargs["laplace_k"])
    if model_kwargs.get("timesteps") is not None:
        cfg.TIMESTEPS = int(model_kwargs["timesteps"])


def _build_loaders(cfg):
    run_experiment = resolve_run_experiment(cfg.DATA_DIR)
    batch_size = int(getattr(cfg, "BATCH_SIZE", getattr(cfg, "DATES_PER_BATCH", 1)))
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
    )


def _load_summarizer(cfg, train_dl, device: torch.device) -> LaplaceAE:
    ckpt_path = Path(str(cfg.SUM_CKPT)).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Summarizer checkpoint is required for real conditioning context: {ckpt_path}"
        )

    _, num_entities, window_size, feat_dim = tv._summarize_dataset(train_dl, None)
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
        irreg_pooling=str(getattr(cfg, "SUM_IRREG_POOLING", "none")),
        irreg_hidden=int(getattr(cfg, "SUM_IRREG_HIDDEN", 32)),
        irreg_residual_scale=float(getattr(cfg, "SUM_IRREG_RES_SCALE", 0.1)),
        t_token_mode=str(getattr(cfg, "SUM_T_TOKEN_MODE", "none")),
        t_token_scale=float(getattr(cfg, "SUM_T_TOKEN_SCALE", 0.1)),
        pos_encoding=str(getattr(cfg, "SUM_POS_ENCODING", "learned_abs")),
        rope_base=float(getattr(cfg, "SUM_ROPE_BASE", 10000.0)),
        channel_balanced_x_loss=bool(getattr(cfg, "SUM_CHANNEL_BALANCED_X_LOSS", False)),
    ).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    tv._load_module_state(summarizer, state_dict, strict=True)
    summarizer.eval()
    return summarizer


def _select_loader(split: str, loaders):
    train_dl, val_dl, test_dl, _ = loaders
    if split == "train":
        return train_dl
    if split == "val":
        return val_dl
    return test_dl


@torch.no_grad()
def _collect_conditioning(
    cfg,
    model: LLapDiff,
    summarizer: LaplaceAE,
    loader,
    device: torch.device,
    *,
    max_examples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cond_batches = []
    raw_batches = []
    collected = 0

    for xb, yb, meta in loader:
        (V, T), _, mask_bn = tv._sanitize_batch(xb, yb, meta, device)
        if not mask_bn.any():
            continue

        cond_summary, cond_summary_raw = tv._build_cond_summary_pair(
            summarizer,
            model,
            V,
            T,
            mask_bn,
            device,
            dt=meta.get("delta_t"),
            x_obs_mask=meta.get("x_obs_mask"),
        )
        take = min(max_examples - collected, cond_summary.size(0))
        if take <= 0:
            break
        cond_batches.append(cond_summary[:take].detach())
        raw_batches.append(cond_summary_raw[:take].detach())
        collected += take
        if collected >= max_examples:
            break

    if not cond_batches:
        raise RuntimeError("Could not build any valid conditioning context from the selected split.")

    return torch.cat(cond_batches, dim=0), torch.cat(raw_batches, dim=0)


@torch.no_grad()
def _extract_global_poles(model: LLapDiff) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    rho, omega = model.model.analysis.effective_poles(1, dtype, device, cond=None)
    return rho[0].detach().cpu(), omega[0].detach().cpu()


@torch.no_grad()
def _extract_conditioned_poles(
    model: LLapDiff,
    *,
    t_idx: int,
    cond_summary: torch.Tensor,
    cond_summary_raw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    batch = cond_summary.size(0)
    t = torch.full((batch,), int(t_idx), device=device, dtype=torch.long)
    t_vec = model._time_embed(t).to(cond_summary.dtype)
    cond_vec = model.model.make_pole_cond(
        t_vec,
        cond_summary=cond_summary,
        cond_summary_raw=cond_summary_raw,
    )
    rho, omega = model.model.analysis.effective_poles(batch, t_vec.dtype, device, cond=cond_vec)
    return rho.detach().cpu(), omega.detach().cpu()


def _pick_timesteps(total_steps: int, n: int) -> Sequence[int]:
    if n <= 1:
        return [total_steps // 2]
    return [int(x) for x in torch.linspace(1, total_steps - 1, n).tolist()]


def _scatter_conjugates(
    rho: torch.Tensor,
    omega: torch.Tensor,
    *,
    marker: str,
    color,
    label: Optional[str],
    alpha: float = 1.0,
) -> None:
    x = (-rho).reshape(-1).numpy()
    y = omega.reshape(-1).numpy()
    plt.scatter(x, y, marker=marker, color=color, label=label, alpha=alpha, s=18)
    plt.scatter(x, -y, marker=marker, color=color, alpha=alpha, s=18)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LLapDiff global and conditioned poles for a checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to visualize.")
    parser.add_argument(
        "--checkpoint-kind",
        choices=CHECKPOINT_KINDS,
        default="auto",
        help="Checkpoint type to resolve when --checkpoint is omitted. Default auto searches raw, ema, best, then last.",
    )
    parser.add_argument("--dataset-key", choices=dataset_keys(), default=None, help="Dataset preset key.")
    parser.add_argument("--pred", type=int, default=None, help="Prediction horizon. Defaults to the checkpoint tag.")
    parser.add_argument("--split", choices=SPLITS, default="test", help="Dataset split used to build conditioning context.")
    parser.add_argument("--num-context-examples", type=int, default=4, help="Number of real context examples to plot.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for the saved PDF.")
    parser.add_argument("--use-ema", action="store_true", default=False, help="Apply EMA weights when available. Implied by --checkpoint-kind ema.")
    parser.add_argument("--num-timesteps", type=int, default=5, help="Number of diffusion timesteps to plot.")
    parser.add_argument("--coverage", type=float, default=0.0, help=COVERAGE_HELP)
    parser.add_argument("--dataset-zip", type=str, default=None, help="Optional zipped dataset cache.")
    parser.add_argument("--dataset-extract-dir", type=str, default=None, help="Optional directory for extracting --dataset-zip.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if int(args.num_context_examples) <= 0:
        raise ValueError("--num-context-examples must be positive.")

    configure_dataset_archive(args.dataset_zip, args.dataset_extract_dir)

    dataset_key = args.dataset_key
    pred = args.pred
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None

    if checkpoint is not None and dataset_key is None:
        dataset_key = _infer_dataset_key_from_path(checkpoint)
    if checkpoint is not None and pred is None:
        pred = _infer_pred_from_checkpoint(checkpoint)

    if dataset_key is None:
        raise ValueError("Could not infer the dataset preset. Pass --dataset-key explicitly.")
    if pred is None:
        raise ValueError("Could not infer the prediction horizon. Pass --pred explicitly.")

    cfg = _build_config(dataset_key, int(pred))
    cfg.COVERAGE = _validate_coverage(args.coverage)
    checkpoint = _resolve_checkpoint(
        str(checkpoint) if checkpoint is not None else None,
        cfg,
        pred=int(pred),
        checkpoint_kind=str(args.checkpoint_kind),
    )

    device = set_torch(seed=int(getattr(cfg, "SEED", 42)), deterministic=bool(getattr(cfg, "DETERMINISTIC", False)))
    model, model_kwargs = _load_model_from_checkpoint(
        cfg,
        checkpoint,
        device=device,
        ema_decay=float(getattr(cfg, "EMA_DECAY", 0.999)),
        use_ema=bool(args.use_ema or args.checkpoint_kind == "ema" or _is_ema_checkpoint(checkpoint)),
    )
    _align_config_to_model(cfg, int(pred), model_kwargs)

    loaders = _build_loaders(cfg)
    summarizer = _load_summarizer(cfg, loaders[0], device)
    context_loader = _select_loader(str(args.split), loaders)
    cond_summary, cond_summary_raw = _collect_conditioning(
        cfg,
        model,
        summarizer,
        context_loader,
        device,
        max_examples=int(args.num_context_examples),
    )

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(cfg.POLE_PLOT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{dataset_key}_pred{pred}_{checkpoint.stem}_poles.pdf"

    timesteps = _pick_timesteps(int(model.scheduler.timesteps), max(1, int(args.num_timesteps)))
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(7, 5.5))

    global_rho, global_omega = _extract_global_poles(model)
    _scatter_conjugates(global_rho, global_omega, marker="o", color="black", label="global/base", alpha=0.85)

    for idx, t_idx in enumerate(timesteps):
        rho, omega = _extract_conditioned_poles(
            model,
            t_idx=t_idx,
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
        )
        _scatter_conjugates(
            rho,
            omega,
            marker="x",
            color=cmap(idx % 10),
            label=f"conditioned t={t_idx}",
            alpha=0.45,
        )

    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Re(s) = -rho")
    plt.ylabel("Im(s) = +/-omega")
    plt.title(f"LLapDiff poles ({dataset_key}, pred={pred}, split={args.split})")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved pole plot to: {save_path}")


if __name__ == "__main__":
    main()
