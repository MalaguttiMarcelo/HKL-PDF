"""esd / correlation engine on a linear model with a known covariance."""

import numpy as np

from pdf_fitting.uncertainty import compute_uncertainties, nyquist_points


class _Model:
    pass


class _Engine:
    """Minimal FitEngine stand-in: G = a·f1 + b·f2, constraint c = 2·a."""

    def __init__(self, x, f1, f2, y):
        self.param_names = ["a", "b", "c"]
        self.initial_values = np.array([x[0], x[1], 2 * x[0]])
        self.fit_indices = np.array([0, 1])
        self.constraints = {"c": "2*a"}
        self.model = _Model()
        self.f1, self.f2, self.y = f1, f2, y

    def _build_bounds_for(self, idx):
        return np.full(len(idx), -np.inf), np.full(len(idx), np.inf)

    def _symmetrize_coordinates(self, p):
        return p

    def _evaluate_model(self, p, use_subgrid=False):
        return None, self.y, p["a"] * self.f1 + p["b"] * self.f2, None


def test_linear_covariance_and_dependent():
    rng = np.random.default_rng(3)
    r = np.linspace(1, 20, 1500)
    f1, f2 = np.sin(r), np.sin(r) + 0.3 * np.cos(2 * r)   # correlated columns
    sig = 0.05
    y = 1.3 * f1 - 0.7 * f2 + sig * rng.standard_normal(r.size)
    X = np.column_stack([f1, f2])
    xhat, *_ = np.linalg.lstsq(X, y, rcond=None)
    s2 = float(np.sum((y - X @ xhat) ** 2) / (r.size - 2))
    cov = s2 * np.linalg.inv(X.T @ X)
    res = compute_uncertainties(_Engine(xhat, f1, f2, y))
    assert np.allclose(res.esd, np.sqrt(np.diag(cov)), rtol=1e-4)
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert abs(res.correlation[0, 1] - rho) < 1e-4
    assert abs(res.dependent["c"][1] - 2 * res.esd[0]) < 1e-6
    assert res.rank == 2 and not res.singular
    # Correlated-points correction: N/N_eff on the covariance.
    n_eff = nyquist_points([(19.0, 20.0)])
    res2 = compute_uncertainties(_Engine(xhat, f1, f2, y), n_eff=n_eff)
    assert np.allclose(res2.esd, res.esd * np.sqrt(r.size / n_eff), rtol=1e-6)


def test_degenerate_parameters_are_reported():
    r = np.linspace(1, 20, 500)
    f = np.sin(r)
    y = 2.0 * f
    res = compute_uncertainties(_Engine(np.array([1.0, 1.0]), f, f, y + 1e-3 * np.cos(3 * r)))
    assert res.rank == 1 and res.singular
    assert {n for n, _c in res.singular[0]["params"]} == {"a", "b"}
