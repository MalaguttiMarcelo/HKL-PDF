"""Shell directions: exact reciprocal direction parallel to the pair vector."""

import numpy as np

from pdf_fitting.models.direction_utils import reciprocal_directions


def _lattice(a, b, c, gamma_deg=90.0):
    g = np.radians(gamma_deg)
    return np.array([[a, 0, 0], [b * np.cos(g), b * np.sin(g), 0], [0, 0, c]], dtype=float)


def test_cubic_long_vectors_are_exact():
    A = _lattice(2.87, 2.87, 2.87)
    frac = np.array([[-1.5, -0.5, 1.5], [-2.0, 1.0, 0.0], [-2.5, -0.5, 0.5], [0.5, 0.5, 0.5]])
    d = reciprocal_directions(frac @ A, A)
    assert d.tolist() == [[3, 1, -3], [2, -1, 0], [5, 1, -1], [1, 1, 1]]
    # Very long vector: smallest triple within the 0.1° tolerance of (103 101 1).
    far = reciprocal_directions(np.array([[51.5, 50.5, 0.5]]) @ A, A)[0].astype(float)
    exact = np.array([103.0, 101.0, 1.0])
    ang = np.degrees(np.arccos(far @ exact / (np.linalg.norm(far) * np.linalg.norm(exact))))
    assert ang <= 0.1


def test_hexagonal_short_vectors_not_zero_and_parallel():
    A = _lattice(4.99, 4.99, 17.06, 120.0)
    rng = np.random.default_rng(0)
    frac = rng.uniform(-0.4, 0.4, size=(500, 3))
    cart = frac @ A
    d = reciprocal_directions(cart, A).astype(float)
    assert not np.any(np.all(d == 0, axis=1))
    h = cart @ A.T
    cosang = np.abs(np.sum(h * d, 1)) / (np.linalg.norm(h, axis=1) * np.linalg.norm(d, axis=1))
    assert np.degrees(np.arccos(np.clip(cosang, -1, 1))).max() <= 0.1 + 1e-9
