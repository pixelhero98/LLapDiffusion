"""Evaluate LLapDiff checkpoints on forecast and target imputation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Optional

import torch

from llapdiffusion.trainers import train_val_llapdiff as tv
from llapdiffusion.configs.dataset_archives import configure_dataset_archive
from llapdiffusion.configs.config_utils import clone_config, make_jsonable
from llapdiffusion.configs.dataset_defaults import apply_dataset_preset, dataset_keys, default_horizons
from llapdiffusion.configs.dataset_registry import resolve_run_experiment

from llapdiffusion.latent_space.latent_vae import LatentVAE
from llapdiffusion.models.summarizer import LaplaceAE
from llapdiffusion.models.llapdiff_utils import (
    decode_latents_with_vae,
    encode_mu_norm,
    pack_targets_tokens,
    set_torch,
)


def build_eval_config(dataset_key: str, pred: int) -> SimpleNamespace:
    cfg = clone_config()
    return apply_dataset_preset(cfg, dataset_key, pred=pred)


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


def _load_stack(cfg: SimpleNamespace, ckpt_path: Path, device: torch.device, train_dl):
    _, num_entities, window_size, feat_dim = tv._summarize_dataset(train_dl, None)

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
        input_dim=int(getattr(cfg, "VAE_INPUT_DIM", 2)),
        num_entities=num_entities,
        entity_conditioned=bool(getattr(cfg, "VAE_ENTITY_CONDITION", False)),
    ).to(device)
    tv._load_module_state(vae, torch.load(cfg.VAE_CKPT, map_location=device), strict=True)
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

    diff_model = tv.build_llapdiff_model(cfg, device)
    payload = torch.load(ckpt_path, map_location=device)
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
    num_samples: int = 8,
    steps: int = 64,
    guidance_strength=(1.0, 2.0),
    guidance_power: float = 1.0,
    eta: float = 0.0,
    dynamic_thresh_p: float = 0.0,
    dynamic_thresh_max: float = 1.0,
    rho: float = 7.5,
    generator_seed: Optional[int] = None,
) -> Dict[str, float]:
    abs_sum = sq_sum = elts = 0.0
    crps_sum = crps_elts = 0.0
    obs_abs_sum = obs_elts = 0.0
    observed_token_sum = hidden_token_sum = candidate_token_sum = 0.0
    generator = None
    if generator_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(generator_seed))

    for xb, yb, meta in test_dl:
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

        obs_any = obs.any(dim=2)
        keep_mask = keep_fn(obs_any)
        valid_seq = keep_mask.any(dim=1) & (obs_any & (~keep_mask)).any(dim=1)
        if not valid_seq.any():
            continue

        cond_summary = cond_summary[valid_seq]
        cond_summary_raw = cond_summary_raw[valid_seq]
        yb = yb[valid_seq]
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

        all_samples = torch.stack(all_samples, dim=0)
        point_forecast = all_samples.mean(dim=0)
        y_true = (
            torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
            .permute(0, 2, 1)
            .contiguous()
            .unsqueeze(-1)
        )

        hidden_valid = (obs & (~keep_mask.unsqueeze(-1))).unsqueeze(-1).to(dtype=y_true.dtype)
        observed_valid = (obs & keep_mask.unsqueeze(-1)).unsqueeze(-1).to(dtype=y_true.dtype)

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
) -> Dict[str, object]:
    ckpt_path = Path(ckpt_path)
    if random_mask_ratio is None:
        random_mask_ratio = float(getattr(cfg, "IMPUTATION_RANDOM_MASK_RATIO", 0.30))
    random_mask_ratio = _validate_random_mask_ratio(random_mask_ratio)
    random_keep_frac = 1.0 - random_mask_ratio
    device = set_torch(seed=int(getattr(cfg, "SEED", 42)), deterministic=bool(getattr(cfg, "DETERMINISTIC", False)))
    run_experiment = resolve_run_experiment(cfg.DATA_DIR)
    train_dl, val_dl, test_dl, sizes = run_experiment(
        data_dir=cfg.DATA_DIR,
        date_batching=cfg.date_batching,
        dates_per_batch=cfg.DATES_PER_BATCH,
        K=cfg.WINDOW,
        H=cfg.PRED,
        coverage=cfg.COVERAGE,
        ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio),
    )
    if sizes is not None:
        print("eval sizes:", tuple(sizes))
    diff_model, vae, summarizer, mu_mean, mu_std = _load_stack(cfg, ckpt_path, device, train_dl)
    test_sampling = tv._sampling_kwargs(cfg, prefix="TEST")

    forecast = tv.evaluate_regression(
        diff_model,
        vae,
        summarizer,
        test_dl,
        device,
        mu_mean,
        mu_std,
        cfg,
        ema=None,
        self_cond=bool(getattr(cfg, "SELF_COND", False)),
        generator_seed=generator_seed,
        **test_sampling,
    )
    regular = _evaluate_impute_case(
        test_dl,
        diff_model=diff_model,
        vae=vae,
        summarizer=summarizer,
        device=device,
        mu_mean=mu_mean,
        mu_std=mu_std,
        keep_fn=lambda obs_any: _make_regular_keep(obs_any, stride=4),
        num_samples=8,
        steps=int(test_sampling["steps"]),
        guidance_strength=test_sampling["guidance_strength"],
        guidance_power=float(test_sampling["guidance_power"]),
        eta=float(test_sampling["eta"]),
        dynamic_thresh_p=float(test_sampling["dynamic_thresh_p"]),
        dynamic_thresh_max=float(test_sampling["dynamic_thresh_max"]),
        rho=float(test_sampling["rho"]),
        generator_seed=generator_seed,
    )
    random_keep_generator = torch.Generator(device=device)
    random_keep_generator.manual_seed(1234)
    random_mask = _evaluate_impute_case(
        test_dl,
        diff_model=diff_model,
        vae=vae,
        summarizer=summarizer,
        device=device,
        mu_mean=mu_mean,
        mu_std=mu_std,
        keep_fn=lambda obs_any: _make_random_keep(obs_any, frac=random_keep_frac, generator=random_keep_generator),
        num_samples=8,
        steps=int(test_sampling["steps"]),
        guidance_strength=test_sampling["guidance_strength"],
        guidance_power=float(test_sampling["guidance_power"]),
        eta=float(test_sampling["eta"]),
        dynamic_thresh_p=float(test_sampling["dynamic_thresh_p"]),
        dynamic_thresh_max=float(test_sampling["dynamic_thresh_max"]),
        rho=float(test_sampling["rho"]),
        generator_seed=None if generator_seed is None else int(generator_seed) + 100003,
    )

    result = {
        "label": label,
        "checkpoint": str(ckpt_path),
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
        result["random_mask30"] = random_mask
    if out_path is not None:
        out_file = Path(out_path)
        out_file.write_text(json.dumps(make_jsonable(result), indent=2))
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
        "--imputation-random-mask-ratio",
        type=float,
        default=None,
        help="Fraction of observed target entries hidden in the random-mask imputation case.",
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
    cfg = build_eval_config(args.dataset_key, pred)
    label = args.label or f"{args.dataset_key}_pred{pred}"
    result = evaluate_checkpoint(
        cfg,
        args.checkpoint,
        label=label,
        out_path=args.out_json,
        random_mask_ratio=args.imputation_random_mask_ratio,
    )
    print(json.dumps(make_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
