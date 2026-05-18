from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from llapdiffusion.configs.dataset_defaults import DATASET_PRESETS


DATASET_KEYS = tuple(DATASET_PRESETS.keys())
EXTRAPOLATION_BASELINES = (
    "dlinear",
    "neuralcde",
    "patchtst",
    "timegrad",
    "mtan",
    "mr-diff",
    "t_patchgnn",
    "contiformer",
)
IMPUTATION_BASELINES = ("csdi",)


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    placement: str
    metric_type: str
    source_name: str
    source_sha: str
    official_reference: str
    time_handling: str
    probabilistic: bool = False
    dependency_caveat: str = "none"
    dependency_sources: tuple[tuple[str, str], ...] = ()
    first_party: bool = False


BASELINES: Mapping[str, BaselineSpec] = {
    "dlinear": BaselineSpec(
        "dlinear",
        "extrapolation/dlinear",
        "point_mae_mse",
        "LTSF-Linear",
        "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6",
        "models/DLinear.py::Model",
        "timestamp/gap/mask channels appended as covariates",
    ),
    "neuralcde": BaselineSpec(
        "neuralcde",
        "extrapolation/neuralcde",
        "point_mae_mse",
        "NeuralCDE",
        "7e529f58441d719d2ce85f56bdee3208a90d5132",
        "experiments/models/metamodel.py::NeuralCDE + vector_fields.py::FinalTanh",
        "native CDE time path with time as first channel",
    ),
    "patchtst": BaselineSpec(
        "patchtst",
        "extrapolation/patchtst",
        "point_mae_mse",
        "PatchTST",
        "bb0bc6058ddc421c02e8afe77e7e8db99f913957",
        "PatchTST_supervised/models/PatchTST.py::Model",
        "timestamp/gap/mask channels appended as covariates",
    ),
    "timegrad": BaselineSpec(
        "timegrad",
        "extrapolation/timegrad",
        "probabilistic_crps_mse",
        "timegrad",
        "dec29a5679a65f5464a9da2dd27a3521000d8b75",
        "epsilon_theta.py::EpsilonTheta + module.py::GaussianDiffusion",
        "past timestamp/gap/mask channels plus future timestamp encodings condition the official diffusion block",
        probabilistic=True,
        dependency_caveat=(
            "full TimeGradTrainingNetwork is tied to GluonTS dataset objects; "
            "the runner uses official diffusion modules with LLapDiffusion context/future encoders"
        ),
    ),
    "mtan": BaselineSpec(
        "mtan",
        "extrapolation/mtan",
        "probabilistic_crps_mse",
        "mTAN",
        "7a3d536ee742f1cacb4a6d3478ac78a228d995ff",
        "src/models.py::enc_mtan_rnn + dec_mtan_rnn",
        "native multi-time attention with context and future query times",
        probabilistic=True,
        dependency_caveat=(
            "Gaussian likelihood is supplied by the runner: official mTAN encoder/decoder "
            "produce the mean path and a learned scalar std provides probabilistic CRPS"
        ),
    ),
    "mr-diff": BaselineSpec(
        "mr-diff",
        "extrapolation/mr-diff",
        "probabilistic_crps_mse",
        "LLapDiffusion",
        "first-party-paper-derived",
        "ICLR 2024 Multi-Resolution Diffusion Models for Time Series Forecasting",
        (
            "paper-derived multi-resolution diffusion with TimeGrad-style normalized timestamps, "
            "gap features, observation masks, and future timestamp encodings"
        ),
        probabilistic=True,
        dependency_caveat=(
            "MR-Diff has no official public implementation; this adapter is a first-party "
            "implementation from the ICLR 2024 paper."
        ),
        first_party=True,
    ),
    "t_patchgnn": BaselineSpec(
        "t_patchgnn",
        "extrapolation/t_patchgnn",
        "point_mae_mse",
        "t-PatchGNN",
        "00c94e7bbaf21c71b03ed84ff690ae59e37129e5",
        "tPatchGNN/model/tPatchGNN.py::tPatchGNN.forecasting",
        "native learnable time embedding and patch masks",
    ),
    "contiformer": BaselineSpec(
        "contiformer",
        "extrapolation/contiformer",
        "point_mae_mse",
        "SeqML",
        "1ecaa5b28fd14fa30eabf5c7de9fe11444e315ce",
        "physiopro.network.contiformer::ContiFormer, as required by SeqML/ContiFormer/spiral.py",
        "native ContiFormer time input with observed-token attention masking; index time is used when delta_t has no progression",
        dependency_caveat="SeqML ContiFormer delegates network code to microsoft/physiopro; cloned as an official dependency source",
        dependency_sources=(("physiopro", "5486d1ccaff8f33d635753e3debd7465234b09f1"),),
    ),
    "csdi": BaselineSpec(
        "csdi",
        "imputation/csdi",
        "probabilistic_crps_mse",
        "CSDI",
        "7f24a436f08d98853a6b43d4f7f04e5a65ecdf27",
        "main_model.py::CSDI_Physio + diff_models.py::diff_CSDI",
        "native CSDI time embedding over context and target-horizon tokens",
        probabilistic=True,
        dependency_caveat="CSDI is evaluated as target-horizon imputation on held-out observed target tokens",
    ),
}


def selected(values: tuple[str, ...] | list[str], requested: str) -> list[str]:
    return list(values) if requested == "all" else [requested]


def longest_horizon(dataset_key: str) -> int:
    return max(DATASET_PRESETS[dataset_key].horizons)


def context_length(dataset_key: str) -> int:
    return DATASET_PRESETS[dataset_key].context_length
