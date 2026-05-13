"""NeuralCDE-style illustration of LLapDiff complex-pole latent generation.

This script is intentionally synthetic and checkpoint-free. It builds a small
irregular context path, synthesizes a continuous latent trajectory from stable
complex-conjugate poles, projects that trajectory to 3D, and saves a clean
paper-style PDF/PNG figure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASENAME = "llapdiff_complex_pole_trajectory"
DEFAULT_FORMATS = ("pdf", "png")


@dataclass(frozen=True)
class IllustrationData:
    t_query: np.ndarray
    t_context: np.ndarray
    context_xy: np.ndarray
    query_xy: np.ndarray
    latent: np.ndarray
    latent_pca: np.ndarray
    rho: np.ndarray
    omega: np.ndarray


def stable_complex_poles() -> tuple[np.ndarray, np.ndarray]:
    """Return a small stable pole set s=-rho +/- i*omega."""

    rho = np.array([0.035, 0.055, 0.080, 0.120], dtype=np.float64)
    omega = np.array([0.75, 1.15, 1.65, 2.35], dtype=np.float64)
    return rho, omega


def pole_basis(t: np.ndarray, rho: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Build damped cosine/sine basis functions from complex poles."""

    t_col = np.asarray(t, dtype=np.float64)[:, None]
    rho_row = np.asarray(rho, dtype=np.float64)[None, :]
    omega_row = np.asarray(omega, dtype=np.float64)[None, :]
    envelope = np.exp(-t_col * rho_row)
    return np.concatenate(
        [envelope * np.cos(t_col * omega_row), envelope * np.sin(t_col * omega_row)],
        axis=1,
    )


def pca_project(x: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Project rows of x to a deterministic PCA coordinate system."""

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x.shape}")
    if not 1 <= int(n_components) <= min(x.shape):
        raise ValueError(f"n_components={n_components} is incompatible with shape {x.shape}")

    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: int(n_components)]

    # Fix sign ambiguity so repeated runs and platforms are easier to compare.
    for idx in range(components.shape[0]):
        pivot = np.argmax(np.abs(components[idx]))
        if components[idx, pivot] < 0:
            components[idx] *= -1.0
    return centered @ components.T


def _smooth_control_path(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    x = 0.55 * np.cos(0.58 * t) + 0.12 * np.sin(1.65 * t + 0.35)
    y = 0.45 * np.sin(0.72 * t + 0.25) + 0.08 * np.cos(1.25 * t)
    return np.column_stack([x, y])


def _irregular_context_times(rng: np.random.Generator, n_points: int, *, max_time: float) -> np.ndarray:
    if int(n_points) < 4:
        raise ValueError("--num-context-points must be at least 4")
    interior = np.sort(rng.uniform(0.08 * max_time, 0.92 * max_time, size=int(n_points) - 2))
    return np.concatenate([[0.0], interior, [max_time]])


def _scale_xy_to_reference_plane(xy: np.ndarray, reference_xy: np.ndarray) -> np.ndarray:
    xy_centered = xy - xy.mean(axis=0, keepdims=True)
    ref_center = reference_xy.mean(axis=0, keepdims=True)
    ref_scale = np.ptp(reference_xy, axis=0).max()
    xy_scale = max(float(np.ptp(xy_centered, axis=0).max()), 1e-6)
    return ref_center + xy_centered * (0.85 * ref_scale / xy_scale)


def generate_illustration_data(
    *,
    seed: int = 7,
    num_context_points: int = 9,
    num_query_points: int = 240,
    latent_dim: int = 8,
) -> IllustrationData:
    """Generate deterministic synthetic data for the LLapDiff illustration."""

    if int(num_query_points) < 32:
        raise ValueError("--num-query-points must be at least 32")
    if int(latent_dim) < 3:
        raise ValueError("latent_dim must be at least 3")

    rng = np.random.default_rng(int(seed))
    horizon = 12.0
    context_end = 4.8
    t_query = np.linspace(0.0, horizon, int(num_query_points), dtype=np.float64)
    t_context = _irregular_context_times(rng, int(num_context_points), max_time=context_end)

    rho, omega = stable_complex_poles()
    basis = pole_basis(t_query, rho, omega)
    weights = rng.normal(loc=0.0, scale=0.8, size=(basis.shape[1], int(latent_dim)))
    weights *= np.linspace(1.0, 0.55, basis.shape[1])[:, None]
    latent = basis @ weights
    latent += 0.08 * np.column_stack(
        [np.sin(0.22 * t_query + phase) for phase in np.linspace(0.0, 1.2, int(latent_dim))]
    )

    latent_pca = pca_project(latent, n_components=3)
    query_xy_raw = _smooth_control_path(t_query)
    context_xy_raw = _smooth_control_path(t_context) + rng.normal(scale=0.018, size=(len(t_context), 2))
    query_xy = _scale_xy_to_reference_plane(query_xy_raw, latent_pca[:, :2])
    context_xy = _scale_xy_to_reference_plane(context_xy_raw, latent_pca[:, :2])

    return IllustrationData(
        t_query=t_query,
        t_context=t_context,
        context_xy=context_xy,
        query_xy=query_xy,
        latent=latent,
        latent_pca=latent_pca,
        rho=rho,
        omega=omega,
    )


def _set_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _style_3d_axis(ax) -> None:
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, color="0.82", linewidth=0.7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("0.75")
    ax.set_xlabel("latent PC 1", labelpad=6)
    ax.set_ylabel("latent PC 2", labelpad=6)
    ax.set_zlabel("latent PC 3", labelpad=6)


def _plot_dependency_guides(ax, data: IllustrationData, base_z: float) -> None:
    guide_count = min(8, len(data.t_context))
    context_indices = np.linspace(0, len(data.t_context) - 1, guide_count, dtype=int)
    query_indices = np.searchsorted(data.t_query, data.t_context[context_indices]).clip(0, len(data.t_query) - 1)
    for c_idx, q_idx in zip(context_indices, query_indices):
        ax.plot(
            [data.context_xy[c_idx, 0], data.latent_pca[q_idx, 0]],
            [data.context_xy[c_idx, 1], data.latent_pca[q_idx, 1]],
            [base_z, data.latent_pca[q_idx, 2]],
            color="black",
            linestyle=(0, (4, 4)),
            linewidth=0.75,
            alpha=0.45,
        )


def render_figure(data: IllustrationData, *, figsize: tuple[float, float] = (10.4, 4.8)) -> plt.Figure:
    """Render the static NeuralCDE-style LLapDiff figure."""

    _set_matplotlib_style()
    blue = "#0072B2"
    cyan = "#56B4E9"
    orange = "#E69F00"
    green = "#009E73"

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.25, 1.0), wspace=0.28)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_time = fig.add_subplot(gs[0, 1])

    z_span = max(float(np.ptp(data.latent_pca[:, 2])), 1e-6)
    base_z = float(data.latent_pca[:, 2].min() - 0.24 * z_span)
    query_z = np.full(data.query_xy.shape[0], base_z)
    context_z = np.full(data.context_xy.shape[0], base_z)

    ax3d.plot(data.query_xy[:, 0], data.query_xy[:, 1], query_z, color=orange, linewidth=1.8, label="query path")
    ax3d.plot(data.context_xy[:, 0], data.context_xy[:, 1], context_z, color=blue, linewidth=1.5, label="data x")
    ax3d.scatter(
        data.context_xy[:, 0],
        data.context_xy[:, 1],
        context_z,
        s=42,
        color=cyan,
        edgecolor="black",
        linewidth=0.6,
        depthshade=False,
        label="irregular samples",
    )
    ax3d.plot(
        data.latent_pca[:, 0],
        data.latent_pca[:, 1],
        data.latent_pca[:, 2],
        color=green,
        linewidth=2.2,
        label=r"latent $z(t)$",
    )
    _plot_dependency_guides(ax3d, data, base_z)
    _style_3d_axis(ax3d)
    ax3d.set_title("LLapDiff: complex poles generate a continuous latent trajectory", pad=12)

    ax3d.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=8, frameon=False)
    ax3d.text(data.latent_pca[-1, 0], data.latent_pca[-1, 1], data.latent_pca[-1, 2], " $z(t)$", color=green)

    for idx, color in enumerate((green, "#CC79A7", "#999999")):
        ax_time.plot(data.t_query, data.latent_pca[:, idx], color=color, linewidth=1.4, label=f"PC {idx + 1}")
    ax_time.scatter(
        data.t_context,
        np.interp(data.t_context, data.t_query, data.latent_pca[:, 0]),
        color=cyan,
        edgecolor="black",
        linewidth=0.5,
        s=25,
        zorder=3,
        label="context times",
    )
    ax_time.set_xlabel("continuous time")
    ax_time.set_ylabel("latent coordinate")
    ax_time.set_title("Latent path over arbitrary query times")
    ax_time.legend(loc="upper right", fontsize=8, frameon=False)

    inset = ax_time.inset_axes([0.08, 0.08, 0.42, 0.36])
    re = -data.rho
    inset.scatter(re, data.omega, marker="o", color="black", s=18)
    inset.scatter(re, -data.omega, marker="o", color="black", s=18)
    for x, y in zip(re, data.omega):
        inset.plot([x, x], [-y, y], color="0.45", linewidth=0.6, linestyle=":")
    inset.axvline(0.0, color="black", linewidth=0.8)
    inset.set_title(r"poles $-\rho \pm i\omega$", fontsize=8)
    inset.set_xlabel("Re(s)", fontsize=7)
    inset.set_ylabel("Im(s)", fontsize=7)
    inset.tick_params(labelsize=7)

    fig.text(
        0.5,
        0.02,
        "Damped complex-conjugate poles define smooth basis functions that synthesize z(t) at any requested time.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.16, top=0.90, wspace=0.28)
    return fig


def save_figure(
    data: IllustrationData,
    *,
    output_dir: Path,
    basename: str = DEFAULT_BASENAME,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = 220,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = render_figure(data)
    saved_paths: list[Path] = []
    for fmt in formats:
        fmt_clean = str(fmt).lower().lstrip(".")
        if fmt_clean not in {"pdf", "png"}:
            raise ValueError(f"Unsupported format {fmt!r}; use pdf and/or png")
        path = output_dir / f"{basename}.{fmt_clean}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt_clean == "png":
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(path, **save_kwargs)
        saved_paths.append(path)
    plt.close(fig)
    return saved_paths


def _parse_formats(values: Iterable[str]) -> tuple[str, ...]:
    formats = tuple(str(value).lower().lstrip(".") for value in values)
    if not formats:
        raise ValueError("At least one output format is required")
    unsupported = sorted(set(formats) - {"pdf", "png"})
    if unsupported:
        raise ValueError(f"Unsupported output format(s): {', '.join(unsupported)}")
    return formats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a NeuralCDE-style LLapDiff complex-pole latent trajectory illustration."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"), help="Directory for output files.")
    parser.add_argument("--basename", type=str, default=DEFAULT_BASENAME, help="Output filename without extension.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the synthetic illustration.")
    parser.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS), help="Output formats: pdf png.")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution.")
    parser.add_argument("--num-context-points", type=int, default=9, help="Number of irregular context samples.")
    parser.add_argument("--num-query-points", type=int, default=240, help="Number of continuous query times.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data = generate_illustration_data(
        seed=int(args.seed),
        num_context_points=int(args.num_context_points),
        num_query_points=int(args.num_query_points),
    )
    paths = save_figure(
        data,
        output_dir=Path(args.output_dir),
        basename=str(args.basename),
        formats=_parse_formats(args.formats),
        dpi=int(args.dpi),
    )
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
