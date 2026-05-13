from pathlib import Path

import numpy as np

from Viz.plot_llapdiff_complex_pole_trajectory import (
    generate_illustration_data,
    save_figure,
    stable_complex_poles,
)


def test_synthetic_illustration_is_deterministic():
    first = generate_illustration_data(seed=11, num_context_points=8, num_query_points=64)
    second = generate_illustration_data(seed=11, num_context_points=8, num_query_points=64)

    assert np.allclose(first.t_query, second.t_query)
    assert np.allclose(first.t_context, second.t_context)
    assert np.allclose(first.context_xy, second.context_xy)
    assert np.allclose(first.latent, second.latent)
    assert np.allclose(first.latent_pca, second.latent_pca)


def test_synthetic_illustration_shapes_and_stable_poles():
    data = generate_illustration_data(seed=3, num_context_points=7, num_query_points=72)
    rho, omega = stable_complex_poles()

    assert data.latent.shape == (72, 8)
    assert data.latent_pca.shape == (72, 3)
    assert data.context_xy.shape == (7, 2)
    assert np.allclose(data.rho, rho)
    assert np.allclose(data.omega, omega)
    assert np.all(data.rho > 0.0)
    assert np.all(np.isfinite(data.omega))


def test_save_figure_writes_readable_pdf_and_png(tmp_path: Path):
    data = generate_illustration_data(seed=5, num_context_points=6, num_query_points=48)
    paths = save_figure(
        data,
        output_dir=tmp_path,
        basename="smoke",
        formats=("pdf", "png"),
        dpi=90,
    )

    by_suffix = {path.suffix: path for path in paths}
    assert {".pdf", ".png"} == set(by_suffix)
    assert by_suffix[".pdf"].stat().st_size > 1000
    assert by_suffix[".png"].stat().st_size > 1000
    assert by_suffix[".pdf"].read_bytes().startswith(b"%PDF")
    assert by_suffix[".png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
