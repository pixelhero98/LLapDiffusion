# LLapDiff

Official implementation of **Latent Laplace Diffusion for Irregular Multivariate Time Series**.
> 19/05/2026 - fixed incorrect mixed-precision training that potentially affected baselines and llapdiff evaluation behavior.
> 
> 20/05/2026 - fixed incorrect data plumbing in baselines that potentially affected baseline evaluation behavior.
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

Applying Ito's lemma gives the expected energy balance

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

Baseline replication needs the optional baseline dependency group:

```bash
python -m pip install -e ".[baselines]"
```

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

The archive contains compact caches derived from public sources: UCI Air Quality, Beijing Multi-Site Air Quality, PhysioNet Challenge 2012, NOAA ISD, and Yahoo Finance market data through `yfinance`. These cached data are redistributed only for reproducible evaluation convenience; the repository code is MIT-licensed, while each dataset remains governed by its original source terms. Check the upstream dataset pages before redistributing derived caches or using them beyond replication.

### Timestamp convention

Context offsets `delta_t` are relative to the first context timestamp. Target/query offsets `delta_t_y` are relative to the last context timestamp, so a regular daily horizon is `[1, 2, ..., H]` and a gapped query grid with context ending at day 14 and future days `[15, 18, 19]` is `[1, 4, 5]`. Lower-level LLapDiff generation treats `generate(dt=...)` as an already-relative finite, nondecreasing query grid and does not re-zero it. Arbitrary future grids are supported at the model/query-grid level when the requested latent shape matches the checkpoint; public preset training still uses the registered dataset horizons.

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
  --imputation-random-mask-ratio 0.30 \
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
  --output-dir ldt/results/pole_plot
```

The pole plot overlays global/base poles with conditioned effective poles built from real dataset context.

## Baselines

Baseline adapters are packaged under `llapdiffusion.baselines`. DLinear, NeuralCDE, PatchTST, TimeGrad, mTAN, t-PatchGNN, ContiFormer, and CSDI use pinned external upstream repositories; clone those sources outside this repository and pass their parent directory explicitly. MR-Diff is implemented first-party in this repository from the ICLR 2024 paper, and it has no GitHub repo or official implementation:

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

### Practical all-horizon runs

The practical baseline runners are full-data comparison runs by default. They evaluate every public horizon, use the full entity panel, scan the full train/validation/test loaders, train with a 600-epoch budget and 50-epoch patience, report observation-weighted metrics, and use 25 samples for probabilistic CRPS. `--dataset all` covers all seven public datasets and `--baseline all` covers DLinear, NeuralCDE, PatchTST, TimeGrad, mTAN, MR-Diff, t-PatchGNN, and ContiFormer:

```bash
llapdiff-baselines practical-extrapolation \
  --baseline all \
  --dataset all \
  --baseline-source-root /path/to/baseline-sources \
  --output-dir ldt/results/baseline_runs \
  --allow-cache-copy \
  --work-cache-dir ldt/cache_work
```

CSDI is an imputation baseline, not a forecast-horizon extrapolator. It runs target-horizon imputation: the model conditions on the context plus retained target-horizon tokens, randomly hides the configured fraction of observed target-horizon tokens, and scores only those hidden target tokens:

```bash
llapdiff-baselines csdi-imputation \
  --dataset all \
  --baseline-source-root /path/to/baseline-sources \
  --imputation-random-mask-ratio 0.30 \
  --output-dir ldt/results/csdi_runs \
  --allow-cache-copy \
  --work-cache-dir ldt/cache_work
```

MR-Diff can be selected with `--baseline mr-diff` without `--baseline-source-root`. For resource scheduling, select one baseline, dataset, or horizon explicitly; the selected scope still uses the full entity panel and full loaders.

## Repository layout

```text
LLapDiffusion/
|-- llapdiffusion/
|   |-- pipeline.py            # Main training and validation entrypoint
|   |-- baselines/             # Baseline adapters, metrics, and runners
|   |-- configs/               # Generic config, dataset presets, and config helpers
|   |-- trainers/              # VAE, summarizer, and LLapDiff trainers
|   |-- tools/                 # Artifact preparation and checkpoint evaluation CLIs
|   |-- datasets/              # Dataset builders, cache helpers, summaries, and bundled archive
|   |-- latent_space/          # Latent VAE modules and utilities
|   |-- models/                # Summarizer, LLapDiff backbone, and diffusion utilities
|   `-- viz/                   # Visualization utilities, including pole plots
`-- README.md                  # Public overview and usage guide
```

## Practical notes

The dataset registry defines the public context lengths, horizons, latent dimensions, VAE settings, and summarizer settings used by the pipeline. The command-line examples above intentionally use dataset keys instead of duplicating those defaults in the README. Advanced users can inspect `llapdiffusion/configs/config.py` and `llapdiffusion/configs/dataset_defaults.py`.

### LLapDiff imputation modes

LLapDiff supports two practical imputation paths:

1. Same-model imputation query from a forecast-only checkpoint. The standard public pipeline trains with `TARGET_MASK_AUX_P=0.0`, so no target-imputation objective is mixed into training. After training, run `llapdiff-checkpoint-eval` on the same checkpoint to hide target-horizon observations and query the model for those missing values.
2. Dual-task target-mask training. Set a positive `TARGET_MASK_AUX_P` when intentionally mixing target-mask reconstruction batches into LLapDiff training, then evaluate the resulting checkpoint with the same `llapdiff-checkpoint-eval` imputation command. The target-mask controls are `TARGET_MASK_AUX_KEEP_MODE`, `TARGET_MASK_AUX_KEEP_PROB`, `TARGET_MASK_AUX_KEEP_STRIDE`, and `TARGET_MASK_AUX_START_EPOCH`.

Forecast-only training plus imputation evaluation:

```bash
llapdiff-train \
  --dataset-key crypto \
  --preds 100 \
  --summary-json ldt/results/crypto_forecast_only.json

llapdiff-checkpoint-eval \
  --dataset-key crypto \
  --pred 100 \
  --checkpoint /path/to/LLapDiffusion/ldt/output/crypto/llapdiff_pred-100_best_raw.pt \
  --imputation-random-mask-ratio 0.30 \
  --out-json ldt/results/crypto_forecast_only_imputation.json
```

Dual-task training plus imputation evaluation:

```bash
llapdiff-train \
  --dataset-key crypto \
  --preds 100 \
  --target-mask-aux-p 0.20 \
  --target-mask-aux-keep-mode mixed \
  --target-mask-aux-keep-prob 0.70 \
  --target-mask-aux-keep-stride 4 \
  --target-mask-aux-start-epoch 10 \
  --summary-json ldt/results/crypto_dual_task.json

llapdiff-checkpoint-eval \
  --dataset-key crypto \
  --pred 100 \
  --checkpoint /path/to/LLapDiffusion/ldt/output/crypto/llapdiff_pred-100_best_raw.pt \
  --imputation-random-mask-ratio 0.30 \
  --out-json ldt/results/crypto_dual_task_imputation.json
```

When using CLI entry points that expose random-mask imputation evaluation, `--imputation-random-mask-ratio` is the fraction of observed target-horizon entries to hide for the random-mask imputation case. For example, `--imputation-random-mask-ratio 0.30` hides 30% and keeps 70% of observed target-horizon entries. Use this as an evaluation mask setting, not as evidence that the checkpoint was trained with an imputation loss.

## Practical tuning guidance

When moving beyond the default examples, the most reliable order is:

1. Validate the dataset cache and window geometry.
2. Check normalization and data scale.
3. Adjust the objective, training schedule, and optionally turn on VAE decoder fine-tuning.
4. Change the architecture last.

## Citation

Citation information will be added once the official proceedings entry is available.

## License

This repository is released under the MIT License. See `LICENSE`.
