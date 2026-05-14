# LLapDiff

Official implementation of **Latent Laplace Diffusion for Irregular Multivariate Time Series**.

> Accepted at ICML 2026.  
> Proceedings citation will be added once the official entry is available.

LLapDiff is a Laplace-domain latent diffusion model for irregular, partially observed panel and multivariate time series. It generates a low-dimensional latent trajectory over arbitrary query timestamps, avoiding sequential numerical integration over physical time while preserving continuous-time structure.

## Highlights

- **Irregular-time forecasting and imputation** through timestamp-aware latent trajectory generation.
- **Latent diffusion backbone** that denoises compact target trajectories instead of sparse observation-space sequences.
- **Stable Laplace-domain modal parameterization** with learned complex-conjugate poles for horizon-wide trajectory synthesis.
- **Gap-aware history conditioning** that uses observed values, local dynamics proxies, timestamps, time gaps, and masks.

### Stability prior from stochastic port-Hamiltonian dynamics

Let $x_t \in \mathbb{R}^{d_z}$ be an auxiliary latent Hamiltonian state, $H(x_t;\psi_t)$ a context-conditioned energy function, and $\psi_t$ the history-dependent context. The stability prior starts from the stochastic port-Hamiltonian SDE

$$dx_t = \big((J-R)\nabla_x H(x_t;\psi_t)+G(\psi_t)u_t\big)dt+\Sigma dW_t,\quad J^\top=-J,\quad R\succ0.$$

The port-collocated output is

$$\tilde{y}_t = G^\top(\psi_t)\nabla_x H(x_t;\psi_t).$$

Applying Itô's lemma gives the expected energy balance

$$\frac{d}{dt}\mathbb{E}[H] = \mathbb{E}[\partial_{\psi}H\,\dot{\psi}_t] - \mathbb{E}[\nabla_xH^\top R\nabla_xH] + \mathbb{E}[\tilde{y}_t^\top u_t] + \frac{1}{2}\mathbb{E}[\mathrm{tr}(\Sigma\Sigma^\top\nabla_x^2H)].$$

In the forecasting setting, the context is updated at observation times and held fixed between updates. Without future external input and noise, the dissipative term makes the expected energy non-increasing between updates. LLapDiff uses this as a stability bias for the clean latent trajectory $x_t \equiv z_0(t)$, while diffusion variables $z_\tau(t)$ are denoised toward that trajectory.


<p align="center">
  <img src="imgs/llapdiff_complex_pole_trajectory.png" alt="LLapDiff complex-pole latent trajectory" width="900">
</p>

Blue context samples and an orange query path condition a green continuous latent trajectory; the pole panel shows the stable complex-conjugate poles that define the damped oscillatory basis.

## Installation

Create a Python 3.11 environment and install the public dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If you need a specific CUDA build of PyTorch, install `torch` with the official PyTorch selector first, then install the remaining requirements.

The public stack uses standard scientific PyTorch tooling, including `torch`, `numpy`, `pandas`, `matplotlib`, `pyarrow`, `fastparquet`, `yfinance`, `requests`, and `tqdm`.

## Quick start

Train and validate LLapDiff for one dataset with the public preset:

```bash
cd /path/to/LLapDiff
python train_val_pipeline.py \
  --dataset-key crypto \
  --summary-json ldt/results/crypto_pipeline_summary.json
```

Run a single forecasting horizon while refreshing the VAE and summarizer artifacts:

```bash
cd /path/to/LLapDiff
python train_val_pipeline.py \
  --dataset-key us_equity \
  --preds 100 \
  --recompute-vae \
  --recompute-summarizer \
  --summary-json ldt/results/us_equity_pred100.json
```

Available public dataset keys are:

```text
bms_air, uci_air, physionet, noaa_us, noaa_uk, us_equity, crypto
```

Run the public preset pipeline for all seven datasets:

```bash
cd /path/to/LLapDiff
for dataset in bms_air uci_air physionet noaa_us noaa_uk us_equity crypto; do
  python train_val_pipeline.py \
    --dataset-key "$dataset" \
    --summary-json "ldt/results/${dataset}_pipeline_summary.json"
done
```

## Evaluation datasets

The cached evaluation datasets are included as `Dataset/LLapDiff-evaluation-datasets.zip`. The pipeline extracts the matching cache root automatically when a preset cache directory is absent. You can also provide an alternate archive:

```bash
python train_val_pipeline.py \
  --dataset-key crypto \
  --dataset-zip /path/to/LLapDiff-evaluation-datasets.zip
```

The included or alternate archive must contain seven cache roots:

```text
fin_dataset/crypto
fin_dataset/us_equity
noaa_uk
noaa_us
bms_air
uci_air
physionet
```

You can also configure the archive and extraction directory through environment variables:

```bash
export LLAPDIFF_DATASET_ZIP=/path/to/LLapDiff-evaluation-datasets.zip
export LLAPDIFF_DATASET_EXTRACT_DIR=./Dataset/extracted
```

The pipeline and evaluation CLIs also accept `--dataset-zip` and `--dataset-extract-dir`. If neither an existing preset cache directory nor an available archive is present, the run fails early with a clear dataset-cache error.

## Controlled synthetic shifts

The repository also includes two generated regime-shift tests:

- `synthetic_freq_shift`: the sinusoid frequency changes after a known change point.
- `synthetic_decay_shift`: the damping/decay rate changes after a known change point.

Boundary-crossing robustness uses the original regime-crossing slice. It trains with forecast-only windows, then evaluates test windows whose context ends shortly before the change point and whose forecast horizon crosses into the shifted regime:

```bash
cd /path/to/LLapDiff
python -m tools.run_synthetic_regime_shift \
  --protocol-name boundary_crossing \
  --tasks synthetic_freq_shift synthetic_decay_shift \
  --seeds 3407 3408 3409 \
  --freq-multipliers 1.5 2.0 2.5 3.0 \
  --decay-multipliers 1.25 1.5 2.0 2.5 \
  --conditioning-modes conditioned unconditioned \
  --output-root ldt/results/synthetic_boundary_crossing
```

Strict unseen-regime extrapolation keeps train/validation windows fully before the change point and exposes shifted targets only at test time. Defaults are `window=96`, `horizon=48`, `series_length=432`, and `change_point=373`; the runner validates split geometry before training and sets `TARGET_MASK_AUX_P=0.0` so no target-imputation objective is used:

```bash
cd /path/to/LLapDiff
python -m tools.run_synthetic_regime_shift \
  --protocol-name strict_unseen_regime \
  --tasks synthetic_freq_shift synthetic_decay_shift \
  --seeds 3407 3408 3409 \
  --freq-multipliers 1.5 2.0 2.5 3.0 \
  --decay-multipliers 1.25 1.5 2.0 2.5 \
  --conditioning-modes conditioned unconditioned \
  --output-root ldt/results/synthetic_strict_unseen
```

For a fast geometry check without training, add `--validate-split-only`. The runner writes `synthetic_regime_raw.csv`, `synthetic_regime_summary.csv`, and `conditioned_vs_unconditioned_summary.csv` when evaluation is run.

## Common utilities

Prepare VAE and summarizer artifacts for all public datasets:

```bash
cd /path/to/LLapDiff
python -m tools.run_multidataset_artifact_prep \
  --datasets bms_air uci_air physionet noaa_us noaa_uk us_equity crypto \
  --summary-json ldt/results/multidataset_artifact_prep_summary.json
```

Evaluate a trained LLapDiff checkpoint on raw scale:

```bash
cd /path/to/LLapDiff
python -m tools.llapdiff_checkpoint_eval \
  --dataset-key crypto \
  --pred 100 \
  --dataset-zip /path/to/LLapDiff-evaluation-datasets.zip \
  --checkpoint /path/to/LLapDiff/ldt/output/crypto/llapdiff_pred-100_best_raw.pt \
  --out-json ldt/results/crypto_eval.json
```

Plot learned poles from a checkpoint:

```bash
cd /path/to/LLapDiff
python -m Viz.plot_llapdiff_poles \
  --dataset-key crypto \
  --pred 100 \
  --checkpoint /path/to/LLapDiff/ldt/output/crypto/llapdiff_pred-100_best_raw.pt \
  --dataset-zip /path/to/LLapDiff-evaluation-datasets.zip \
  --output-dir ldt/results/pole_plot_smoke
```

The pole plot overlays global/base poles with conditioned effective poles built from real dataset context.

## Repository layout

```text
LLapDiff/
├── train_val_pipeline.py      # Main training and validation entrypoint
├── configs/                   # Generic config, dataset presets, and config helpers
├── trainers/                  # VAE, summarizer, and LLapDiff trainers
├── tools/                     # Artifact preparation and checkpoint evaluation CLIs
├── Dataset/                   # Dataset builders, cache helpers, and summaries
├── Latent_Space/              # Latent VAE modules and utilities
├── Model/                     # Summarizer, LLapDiff backbone, and diffusion utilities
├── Viz/                       # Visualization utilities, including pole plots
└── README.md                  # Public overview and usage guide
```

## Public presets

The repository exposes one canonical latent channel setting per dataset. The horizon-wise winners below document the development evidence behind those defaults, while runtime configuration remains dataset-level and simple.

| Dataset | Horizons | Horizon best | Default latent channel |
| --- | --- | --- | --- |
| `bms_air` | `24, 48, 96, 168` | `32, 24, 32, 20` | `24` |
| `uci_air` | `24, 48, 96, 168` | `16, 16, 12, 16` | `16` |
| `physionet` | `4, 8, 10, 12` | `12, 20, 16, 16` | `16` |
| `noaa_us` | `24, 48, 96, 168` | `24, 24, 32, 16` | `24` |
| `noaa_uk` | `24, 48, 96, 168` | `12, 16, 20, 20` | `16` |
| `us_equity` | `5, 20, 60, 100` | `8, 16, 8, 12` | `12` |
| `crypto` | `5, 20, 60, 100` | `12, 16, 16, 20` | `16` |

### VAE defaults

These are the public forward defaults in `configs/config.py`:

| Setting | Value |
| --- | --- |
| `VAE_WARMUP_EPOCHS` | `5` |
| `VAE_KL_ANNEAL_EPOCHS` | `25` |
| `VAE_MIN_EPOCHS` | `40` |
| `VAE_BETA` | `1e-3` |
| `VAE_MAX_PATIENCE` | `20` |
| `VAE_INPUT_DROPOUT` | `0.20` |
| `VAE_NOISE_STD` | `0.01` |
| `VAE_RECON_BALANCE` | `none` |

### Summarizer defaults

The summarizer uses the shared public baseline in `configs/config.py`, with dataset-specific presets applied by the registry:

| Dataset | Override |
| --- | --- |
| `bms_air` | `SUM_LR = 1e-4`, `SUM_AMP = False` |
| `physionet` | dataset-specific VAE and summarizer preset |
| all others | shared summarizer defaults |


### Time and imputation settings

The default summarizer position mode is `SUM_POS_ENCODING="learned_abs"` for checkpoint-compatible public baselines. Variants can set `SUM_POS_ENCODING="continuous_rope"` or `"learned_plus_continuous_rope"` to rotate attention queries and keys by context-window relative time. `SUM_ROPE_BASE=10000.0` controls the RoPE frequency base.

`IMPUTATION_TRAINING=True` means imputation-style target anchors are allowed by configuration, not active by default. Pure extrapolation-training remains clean by `TARGET_MASK_AUX_P=0.0`; set a positive value only when intentionally running dual-task or imputation-style training. Extrapolation and interpolation both support in either training setting by querying the same model.

Context and target `delta_t` metadata are interpreted as window-local offsets when they start at zero and are monotone. Increment-style deltas are converted to offsets before use.

## Practical tuning guidance

When moving beyond the public presets, the most reliable order is:

1. Validate the dataset cache and window geometry.
2. Check normalization and data scale.
3. Adjust the objective, training schedule, and optionally turn on VAE decoder fine-tuning.
4. Change the architecture last.

## Citation

Citation information will be added once the official proceedings entry is available.

## License

This repository is released under the MIT License. See `LICENSE`.
