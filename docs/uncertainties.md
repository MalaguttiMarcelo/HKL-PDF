# Esds and Correlation Matrix

HKL-PDF can estimate the standard deviations (esds) of refined parameters
and their correlation matrix. The calculation is **off by default**.

## Turning it on

- **Refinement settings › esds and correlation matrix:** computed after
  every refinement.
- **Refined parameters › Compute esds now:** computed once for the current
  values, without refining.

The esds appear in the **Esd** column of every parameter grid and on the
**Refined parameters** page. That page also shows a heat map of the
correlation matrix and a table of the strongest correlations. The results
are saved in the project file.

## Method

At the final parameters $\hat{\mathbf{x}}$, the Jacobian $J$ of the
weighted residual

$$
r_k = \sqrt{w_k}\left(G_\mathrm{obs}(r_k) - G_\mathrm{calc}(r_k)\right)
$$

is calculated by central differences on the **full** $r$ grid. Constraints,
lattice symmetry, and tied coordinates are applied exactly as during the
fit. Then

$$
S = \sum_k r_k^2, \qquad
s^2 = \frac{S}{N - p},
$$

$$
C = s^2\left(J^\mathsf{T}J\right)^{-1}, \qquad
\mathrm{esd}_i = \sqrt{C_{ii}}, \qquad
\rho_{ij} = \frac{C_{ij}}{\mathrm{esd}_i\,\mathrm{esd}_j},
$$

where $N$ is the number of points and $p$ the number of refined
parameters. The inverse uses a singular-value decomposition. Directions with
a negligible singular value are reported as undetermined and left out of the
inverse.

Constrained parameters (constraint expressions, lattice relations such as
$b = a$, and tied coordinates) receive esds by linear propagation from the
parameters they depend on.

The fit may use a robust loss. The esds are always those of ordinary least
squares at the final point.

## Correlated PDF points

The points of $G(r)$ are not independent. A PDF obtained up to
$Q_\mathrm{max}$ carries about one independent value per $\pi/Q_\mathrm{max}$
in $r$. The number of independent points is approximately

$$
N_\mathrm{eff} = \sum_\mathrm{datasets}
\frac{(r_\mathrm{max} - r_\mathrm{min})\,Q_\mathrm{max}}{\pi}.
$$

With **Correct esds for correlated PDF points** turned on, the covariance
is multiplied by $N/N_\mathrm{eff}$ (esds by $\sqrt{N/N_\mathrm{eff}}$) when
$N_\mathrm{eff} < N$. This option is also off by default. It gives larger,
usually more realistic esds for finely sampled data.

## Interpretation

- Esds describe only the statistical precision within the chosen model.
  They do not include model errors.
- Correlations above about 0.9 in magnitude mean that the two parameters
  cannot be determined independently. Fix one of them, add a constraint, or
  extend the data range.
- A parameter that ends at a bound has no meaningful esd.

## Cost

The calculation needs two model evaluations per refined parameter. For a
large model it can take as long as several refinement iterations. It can be
stopped with **Stop**.
