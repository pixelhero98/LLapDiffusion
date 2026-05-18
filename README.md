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
cd /path/to/LLapDiffusion
llapdiff-train \
  --dataset-key crypto \
  --summary-json ldt/results/crypto_pipeline_summary.json
```

Run a single forecasting horizon while refreshing the VAE and summarizer artifacts:

```bash
cd /path/to/LLapDiffusion
llapdiff-train \
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
cd /path/to/LLapDiffusion
for dataset in bms_air uci_air physionet noaa_us noaa_uk us_equity crypto; do
  llapdiff-train \
    --dataset-key "$dataset" \
    --summary-json "ldt/results/${dataset}_pipeline_summary.json"
done
```

## Evaluation datasets

The cached evaluation datasets are bundled as package data at `llapdiffusion/datasets/LLapDiff-evaluation-datasets.zip` with SHA256 `afd74b04ba498e9fca521938b6090867a61a8de1b7c2dea01007794b051d89e0`. The pipeline extracts the matching cache root automatically when a preset cache directory is absent. You can also provide an alternate archive:

```bash
llapdiff-train \
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
export LLAPDIFF_DATASET_EXTRACT_DIR=~/.cache/llapdiffusion/datasets
```

The pipeline and evaluation CLIs also accept `--dataset-zip` and `--dataset-extract-dir`. By default, extracted caches are written under the user cache directory, not into the installed package or source checkout. If neither an existing preset cache directory nor an available archive is present, the run fails early with a clear dataset-cache error.

## Controlled synthetic shifts

The repository also includes two generated regime-shift tests:

- `synthetic_freq_shift`: the sinusoid frequency changes after a known change point.
- `synthetic_decay_shift`: the damping/decay rate changes after a known change point.

Boundary-crossing robustness uses the original regime-crossing slice. It trains with forecast-only windows, then evaluates test windows whose context ends shortly before the change point and whose forecast horizon crosses into the shifted regime:

```bash
cd /path/to/LLapDiffusion
llapdiff-synthetic-regime \
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
cd /path/to/LLapDiffusion
llapdiff-synthetic-regime \
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
cd /path/to/LLapDiffusion
llapdiff-artifact-prep \
  --datasets bms_air uci_air physionet noaa_us noaa_uk us_equity crypto \
  --summary-json ldt/results/multidataset_artifact_prep_summary.json
```

Evaluate a trained LLapDiff checkpoint on raw scale:

```bash
cd /path/to/LLapDiffusion
llapdiff-checkpoint-eval \
  --dataset-key crypto \
  --pred 100 \
  --dataset-zip /path/to/LLapDiff-evaluation-datasets.zip \
  --checkpoint /path/to/LLapDiffusion/ldt/output/crypto/llapdiff_pred-100_best_raw.pt \
  --out-json ldt/results/crypto_eval.json
```

Plot learned poles from a checkpoint:

```bash
cd /path/to/LLapDiffusion
llapdiff-plot-poles \
  --dataset-key crypto \
  --pred 100 \
  --checkpoint /path/to/LLapDiffusion/ldt/output/crypto/llapdiff_pred-100_best_raw.pt \
  --dataset-zip /path/to/LLapDiff-evaluation-datasets.zip \
  --output-dir ldt/results/pole_plot_smoke
```

The pole plot overlays global/base poles with conditioned effective poles built from real dataset context.

## External baselines

Baseline adapters are packaged under `llapdiffusion.baselines`, but the official upstream repositories remain external. Clone the pinned sources outside this repository and pass their parent directory explicitly:

```bash
mkdir -p /path/to/baseline-sources
cd /path/to/baseline-sources
git clone https://github.com/cure-lab/LTSF-Linear.git LTSF-Linear && git -C LTSF-Linear checkout 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6
git clone https://github.com/patrick-kidger/NeuralCDE.git NeuralCDE && git -C NeuralCDE checkout 7e529f58441d719d2ce85f56bdee3208a90d5132
git clone https://github.com/PatchTST/PatchTST.git PatchTST && git -C PatchTST checkout bb0bc6058ddc421c02e8afe77e7e8db99f913957
git clone https://github.com/kongqi404/timegrad.git timegrad && git -C timegrad checkout dec29a5679a65f5464a9da2dd27a3521000d8b75
git clone https://github.com/reml-lab/mTAN.git mTAN && git -C mTAN checkout 7a3d536ee742f1cacb4a6d3478ac78a228d995ff
git clone https://github.com/usail-hkust/t-PatchGNN.git t-PatchGNN && git -C t-PatchGNN checkout 00c94e7bbaf21c71b03ed84ff690ae59e37129e5
git clone https://github.com/microsoft/SeqML.git SeqML && git -C SeqML checkout 1ecaa5b28fd14fa30eabf5c7de9fe11444e315ce
git clone https://github.com/ermongroup/CSDI.git CSDI && git -C CSDI checkout 7f24a436f08d98853a6b43d4f7f04e5a65ecdf27
git clone https://github.com/microsoft/physiopro.git physiopro && git -C physiopro checkout 5486d1ccaff8f33d635753e3debd7465234b09f1
```

```bash
llapdiff-baselines smoke \
  --baseline all \
  --dataset all \
  --baseline-source-root /path/to/baseline-sources \
  --output-dir ldt/results/baseline_smoke \
  --allow-cache-copy \
  --work-cache-dir ldt/cache_work
```

The baseline pool includes extrapolation adapters for DLinear, NeuralCDE, PatchTST, TimeGrad, mTAN, t-PatchGNN, and ContiFormer, plus CSDI under imputation. Smoke results are one-batch source/import/forward/loss/backward checks, not benchmark scores. CSDI is reported as `context_imputation_holdout`; it is not a forecast-horizon extrapolation result.

Bounded practical runs use the same LLapDiffusion dataset presets and longest horizons:

```bash
llapdiff-baselines practical-extrapolation \
  --baseline dlinear \
  --dataset crypto \
  --baseline-source-root /path/to/baseline-sources \
  --output-dir ldt/results/baseline_runs

llapdiff-baselines csdi-imputation \
  --dataset all \
  --baseline-source-root /path/to/baseline-sources \
  --output-dir ldt/results/csdi_runs
```

## Repository layout

```text
LLapDiffusion/
|-- llapdiffusion/
|   |-- pipeline.py            # Main training and validation entrypoint
|   |-- baselines/             # External baseline adapters, metrics, and runners
|   |-- configs/               # Generic config, dataset presets, and config helpers
|   |-- trainers/              # VAE, summarizer, and LLapDiff trainers
|   |-- tools/                 # Artifact preparation and checkpoint evaluation CLIs
|   |-- datasets/              # Dataset builders, cache helpers, summaries, and bundled archive
|   |-- latent_space/          # Latent VAE modules and utilities
|   |-- models/                # Summarizer, LLapDiff backbone, and diffusion utilities
|   `-- viz/                   # Visualization utilities, including pole plots
`-- README.md                  # Public overview and usage guide
```

## Public presets

The repository exposes one canonical latent channel setting per dataset.

| Dataset | Horizons | Default latent channel |
| --- | --- | --- |
| `bms_air` | `24, 48, 96, 168` | `24` |
| `uci_air` | `24, 48, 96, 168` | `16` |
| `physionet` | `4, 8, 10, 12` | `16` |
| `noaa_us` | `24, 48, 96, 168` | `24` |
| `noaa_uk` | `24, 48, 96, 168` | `16` |
| `us_equity` | `5, 20, 60, 100` | `12` |
| `crypto` | `5, 20, 60, 100` | `16` |

### VAE defaults

These are the public forward defaults in `llapdiffusion/configs/config.py`:

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

The summarizer uses the shared public baseline in `llapdiffusion/configs/config.py`, with dataset-specific presets applied by the registry:

| Dataset | Override |
| --- | --- |
| `bms_air` | `SUM_LR = 1e-4`, `SUM_AMP = False` |
| `physionet` | dataset-specific VAE and summarizer preset |
| `crypto` | dataset-specific VAE and summarizer preset |
| all others | shared summarizer defaults |


### Time and imputation settings

The default summarizer position mode is `SUM_POS_ENCODING="learned_abs"` for public baselines. Variants can set `SUM_POS_ENCODING="continuous_rope"` or `"learned_plus_continuous_rope"` to rotate attention queries and keys by context-window relative time. `SUM_ROPE_BASE=10000.0` controls the RoPE frequency base.

`IMPUTATION_TRAINING=True` means imputation-style target anchors are allowed by configuration, not active by default. Pure extrapolation-training remains clean by `TARGET_MASK_AUX_P=0.0`; set a positive value only when intentionally running dual-task or imputation-style training. Extrapolation and interpolation both support in either training setting by querying the same model.

Context and target `delta_t` metadata are interpreted as nondecreasing window-local offsets and are shifted relative to the first timestamp before use.

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
