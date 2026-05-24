from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from llapdiffusion.configs.dataset_archives import find_dataset_archive, resolve_dataset_dir
from llapdiffusion.configs.dataset_registry import dataset_name_from_data_dir


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PACKAGE_ROOT / "datasets"



_BASE_MODEL_SETTINGS: Mapping[str, object] = {
    "VAE_MAX_PATIENCE": 20,
    "VAE_INPUT_DROPOUT": 0.20,
    "VAE_NOISE_STD": 0.01,
    "VAE_CONSIST_LAMBDA": 0.0,
    "VAE_RECON_BALANCE": "none",
    "SUM_LR": 5e-4,
    "SUM_AMP": True,
    "SUM_MAX_NONFINITE_GRAD_STEPS": 8,
    "SUM_PATIENCE": 10,
    "SUM_LOSS_W_DT": 0.0,
    "SUM_LOSS_W_OBS": 0.0,
    "SUM_CHANNEL_BALANCED_X_LOSS": False,
    "SUM_IRREG_POOLING": "none",
    "SUM_T_TOKEN_MODE": "none",
    "SUM_T_TOKEN_SCALE": 0.1,
    "SUM_POS_ENCODING": "learned_abs",
}


_IRREGULAR_PUBLIC_PRESET: Mapping[str, object] = {
    "VAE_INPUT_DROPOUT": 0.35,
    "VAE_NOISE_STD": 0.02,
    "VAE_CONSIST_LAMBDA": 0.05,
    "VAE_MAX_PATIENCE": 25,
    "VAE_RECON_BALANCE": "coverage",
    "SUM_LOSS_W_DT": 0.05,
    "SUM_LOSS_W_OBS": 0.05,
    "SUM_CHANNEL_BALANCED_X_LOSS": True,
    "SUM_IRREG_POOLING": "repair",
    "SUM_T_TOKEN_MODE": "both",
    "SUM_T_TOKEN_SCALE": 0.1,
    "SUM_POS_ENCODING": "continuous_rope",
    "SUM_PATIENCE": 15,
}


@dataclass(frozen=True)
class DatasetPreset:
    key: str
    artifact_name: str
    data_subdir: str
    horizons: tuple[int, ...]
    context_length: int
    table_batch_size: int
    vae_latent_channels: int
    minsnr_gamma: float
    epochs: int = 600
    sum_lr: float | None = None
    sum_amp: bool | None = None
    model_overrides: Mapping[str, object] | None = None
    split_policy: str = "global_purged_horizon"
    split_scope: str = "global_target_time"
    exact_timestamp_batches: bool = True

    @property
    def expected_data_dir(self) -> Path:
        return (DATASET_ROOT / self.data_subdir).resolve()

    @property
    def data_dir(self) -> Path:
        return resolve_dataset_dir(self.expected_data_dir, package_root=PACKAGE_ROOT)

    @property
    def summary_length(self) -> int:
        return self.context_length


DATASET_PRESETS: Mapping[str, DatasetPreset] = {
    "bms_air": DatasetPreset(
        key="bms_air",
        artifact_name="bms_air",
        data_subdir="bms_air",
        horizons=(24, 48, 96, 168),
        context_length=336,
        table_batch_size=10,
        vae_latent_channels=24,
        minsnr_gamma=5.0,
        sum_lr=1e-4,
        sum_amp=False,
    ),
    "uci_air": DatasetPreset(
        key="uci_air",
        artifact_name="uci_air",
        data_subdir="uci_air",
        horizons=(24, 48, 96, 168),
        context_length=336,
        table_batch_size=10,
        vae_latent_channels=16,
        minsnr_gamma=4.5,
        sum_lr=1e-4,
        sum_amp=False,
    ),
    "physionet": DatasetPreset(
        key="physionet",
        artifact_name="physionet",
        data_subdir="physionet",
        horizons=(4, 8, 10, 12),
        context_length=24,
        table_batch_size=5,
        vae_latent_channels=16,
        minsnr_gamma=5.0,
        model_overrides=_IRREGULAR_PUBLIC_PRESET,
        split_policy="contiguous",
        split_scope="physionet_patient_relative_time",
    ),
    "noaa_us": DatasetPreset(
        key="noaa_us",
        artifact_name="noaa_us",
        data_subdir="noaa_us",
        horizons=(24, 48, 96, 168),
        context_length=336,
        table_batch_size=15,
        vae_latent_channels=24,
        minsnr_gamma=4.5,
        sum_lr=1e-4,
        sum_amp=False,
    ),
    "noaa_uk": DatasetPreset(
        key="noaa_uk",
        artifact_name="noaa_uk",
        data_subdir="noaa_uk",
        horizons=(24, 48, 96, 168),
        context_length=336,
        table_batch_size=15,
        vae_latent_channels=16,
        minsnr_gamma=4.5,
        sum_lr=1e-4,
        sum_amp=False,
    ),
    "us_equity": DatasetPreset(
        key="us_equity",
        artifact_name="us_equity",
        data_subdir="fin_dataset/us_equity",
        horizons=(5, 20, 60, 100),
        context_length=200,
        table_batch_size=5,
        vae_latent_channels=12,
        minsnr_gamma=5.0,
    ),
    "crypto": DatasetPreset(
        key="crypto",
        artifact_name="crypto",
        data_subdir="fin_dataset/crypto",
        horizons=(5, 20, 60, 100),
        context_length=200,
        table_batch_size=5,
        vae_latent_channels=16,
        minsnr_gamma=5.0,
        model_overrides=_IRREGULAR_PUBLIC_PRESET,
    ),
}


_META_DATASET_TO_KEY = {
    "bms_air_quality": "bms_air",
    "bms_air_dataset": "bms_air",
    "uci_air_quality": "uci_air",
    "uci_air_dataset": "uci_air",
    "physionet_cinc": "physionet",
    "physionet_cinc_dataset": "physionet",
}


def dataset_keys(*, include_crypto: bool = True) -> tuple[str, ...]:
    keys = tuple(DATASET_PRESETS.keys())
    if include_crypto:
        return keys
    return tuple(key for key in keys if key != "crypto")


def non_crypto_dataset_keys() -> tuple[str, ...]:
    return dataset_keys(include_crypto=False)


def get_dataset_preset(dataset_key: str) -> DatasetPreset:
    key = str(dataset_key).strip().lower()
    if key not in DATASET_PRESETS:
        raise KeyError(f"Unknown dataset preset: {dataset_key}")
    return DATASET_PRESETS[key]


def default_horizons(dataset_key: str) -> tuple[int, ...]:
    return get_dataset_preset(dataset_key).horizons


def infer_dataset_key(data_dir: object) -> str:
    path = Path(str(data_dir)).resolve()
    parts = {part.lower() for part in path.parts}
    for key in DATASET_PRESETS:
        if key in parts:
            return key
    meta_dataset = dataset_name_from_data_dir(path)
    if meta_dataset in _META_DATASET_TO_KEY:
        return _META_DATASET_TO_KEY[meta_dataset]
    raise KeyError(f"Could not infer dataset preset from data_dir={path}")


def apply_dataset_preset(cfg: object, dataset_key: str, *, pred: int | None = None) -> object:
    preset = get_dataset_preset(dataset_key)
    if pred is None:
        pred_value = int(getattr(cfg, "PRED", preset.horizons[-1]))
    else:
        pred_value = int(pred)
    if pred_value not in preset.horizons:
        raise ValueError(f"{dataset_key}: pred={pred_value} not in supported horizons {preset.horizons}")

    artifact_root = Path(str(getattr(cfg, "ARTIFACT_ROOT", Path.cwd() / "ldt"))).resolve()
    setattr(cfg, "DATASET_KEY", preset.key)
    setattr(cfg, "MKT", preset.artifact_name)
    setattr(cfg, "DATA_DIR", str(preset.data_dir))
    setattr(cfg, "PIPELINE_PREDS", preset.horizons)
    setattr(cfg, "PRED", pred_value)
    setattr(cfg, "WINDOW", preset.context_length)
    setattr(cfg, "COVERAGE", 0.0)
    setattr(cfg, "date_batching", True)
    setattr(cfg, "split_policy", preset.split_policy)
    setattr(cfg, "split_scope", preset.split_scope)
    setattr(cfg, "exact_timestamp_batches", preset.exact_timestamp_batches)
    setattr(cfg, "BATCH_SIZE", preset.table_batch_size)
    setattr(cfg, "DATES_PER_BATCH", preset.table_batch_size)

    setattr(cfg, "VAE_LATENT_CHANNELS", preset.vae_latent_channels)
    vae_entity_condition = bool(getattr(cfg, "VAE_ENTITY_CONDITION", True))
    setattr(cfg, "VAE_ENTITY_CONDITION", vae_entity_condition)
    setattr(cfg, "VAE_NUM_ENTITIES", None)
    vae_suffix = "_entity" if vae_entity_condition else ""
    setattr(cfg, "VAE_DIR", str((artifact_root / "vae" / "saved_model" / preset.artifact_name).resolve()))
    setattr(
        cfg,
        "VAE_CKPT",
        str(Path(getattr(cfg, "VAE_DIR")) / f"pred-{pred_value}_ch-{preset.vae_latent_channels}{vae_suffix}_elbo.pt"),
    )

    setattr(cfg, "SUM_CONTEXT_LEN_FIXED", preset.context_length)
    setattr(cfg, "SUM_CONTEXT_LEN", preset.context_length)
    setattr(cfg, "SUM_CONTEXT_DIM", 256)
    setattr(cfg, "SUM_MIX_DIM", 64)
    setattr(cfg, "SUM_TV_HIDDEN", 32)
    setattr(cfg, "SUM_TIME2VEC_DIM", int(getattr(cfg, "SUM_TIME2VEC_DIM", 9)))
    for name, value in _BASE_MODEL_SETTINGS.items():
        setattr(cfg, name, value)
    if preset.sum_lr is not None:
        setattr(cfg, "SUM_LR", float(preset.sum_lr))
    if preset.sum_amp is not None:
        setattr(cfg, "SUM_AMP", bool(preset.sum_amp))
    setattr(cfg, "SUM_DIR", str((artifact_root / "summarizer" / "saved_model" / preset.artifact_name).resolve()))
    setattr(
        cfg,
        "SUM_CKPT",
        str(Path(getattr(cfg, "SUM_DIR")) / f"{pred_value}-{preset.vae_latent_channels}-summarizer.pt"),
    )

    setattr(cfg, "CKPT_DIR", str((artifact_root / "checkpoints" / preset.artifact_name).resolve()))
    setattr(cfg, "OUT_DIR", str((artifact_root / "output" / preset.artifact_name).resolve()))
    setattr(cfg, "POLE_PLOT_DIR", str((Path(getattr(cfg, "OUT_DIR")) / "pole_plots").resolve()))

    setattr(cfg, "EPOCHS", preset.epochs)
    setattr(cfg, "PREDICT_TYPE", "v")
    setattr(cfg, "LOSS_WEIGHT_SCHEME", "weighted_min_snr")
    setattr(cfg, "MINSNR_GAMMA", preset.minsnr_gamma)
    setattr(cfg, "BASE_LR", 1.5e-4)
    setattr(cfg, "PRIMARY_EVAL_METRIC", str(getattr(cfg, "PRIMARY_EVAL_METRIC", "crps")))
    setattr(cfg, "IMPUTATION_TRAINING", True)
    setattr(cfg, "TARGET_MASK_AUX_P", 0.0)
    setattr(cfg, "TARGET_MASK_AUX_KEEP_MODE", "prefix")
    setattr(cfg, "TARGET_MASK_AUX_KEEP_PROB", 0.5)
    setattr(cfg, "TARGET_MASK_AUX_KEEP_STRIDE", 4)
    setattr(cfg, "TARGET_MASK_AUX_START_EPOCH", 10)

    for name, value in (preset.model_overrides or {}).items():
        setattr(cfg, name, value)

    setattr(cfg, "TIMESTEPS", 1000)
    setattr(cfg, "SCHEDULE", "cosine")
    setattr(cfg, "MODEL_WIDTH", 256)
    setattr(cfg, "NUM_LAYERS", 5)
    setattr(cfg, "NUM_HEADS", 4)
    setattr(cfg, "LAPLACE_K", 256)

    return cfg


def validate_dataset_presets(keys: Iterable[str] | None = None) -> dict[str, object]:
    dataset_list: Sequence[str] = tuple(keys) if keys is not None else dataset_keys()
    rows = []
    for key in dataset_list:
        preset = get_dataset_preset(key)
        if preset.context_length != 2 * max(preset.horizons):
            raise ValueError(
                f"{preset.key}: context_length={preset.context_length} is not 2x longest horizon {max(preset.horizons)}"
            )
        if preset.epochs != 600:
            raise ValueError(f"{preset.key}: epochs={preset.epochs} expected 600")
        expected_data_dir = preset.expected_data_dir
        archive_path = find_dataset_archive(PACKAGE_ROOT)
        if expected_data_dir.exists() is False and archive_path is None:
            raise FileNotFoundError(f"{preset.key}: dataset path missing: {expected_data_dir}")
        rows.append(
            {
                "dataset": preset.key,
                "artifact_name": preset.artifact_name,
                "data_dir": str(expected_data_dir),
                "dataset_cache_exists": expected_data_dir.exists(),
                "dataset_archive": str(archive_path) if archive_path is not None else None,
                "horizons": list(preset.horizons),
                "context_length": preset.context_length,
                "table_batch_size": preset.table_batch_size,
                "runtime_dates_per_batch": preset.table_batch_size,
                "split_policy": preset.split_policy,
                "split_scope": preset.split_scope,
                "exact_timestamp_batches": preset.exact_timestamp_batches,
                "vae_latent_channels": preset.vae_latent_channels,
                "minsnr_gamma": preset.minsnr_gamma,
                "epochs": preset.epochs,
                "predict_type": "v",
            }
        )
    return {"status": "ok", "rows": rows}
