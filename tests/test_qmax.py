import numpy as np

from pdf_fitting.models.gr_model import apply_qmax_termination


def test_qmax_disabled_returns_input():
    r = np.linspace(
        0.0,
        10.0,
        1001,
    )

    G = np.sin(r)

    result = apply_qmax_termination(
        G,
        r,
        qmax=0.0,
    )

    np.testing.assert_allclose(
        result,
        G,
    )


def test_qmax_output_is_finite():
    r = np.linspace(
        0.0,
        10.0,
        1001,
    )

    G = np.exp(
        -0.5
        * ((r - 3.0) / 0.1) ** 2
    )

    result = apply_qmax_termination(
        G,
        r,
        qmax=20.0,
        n_zeros=5,
    )

    assert result.shape == G.shape
    assert np.all(np.isfinite(result))