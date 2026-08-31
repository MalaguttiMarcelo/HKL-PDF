# Conventional PDF Model

The conventional model represents a periodic crystal using grouped
crystallographic pair shells.

## Shell information

Each shell stores:

- a fractional pair vector;
- pair species;
- multiplicity;
- effective multiplicity;
- approximate Miller indices;
- reduced crystallographic direction;
- ordered and unordered pair identifiers.

Real-space distances are recalculated from fractional vectors when the lattice
changes. Shells therefore do not need to be regenerated during every lattice
trial.

## Partial normalization

For a shell,

$$
g_{\alpha\beta}^{(ij)}(r)=
\frac{m_{ij}}
{4\pi r_{ij}^2\rho_\beta}
N(r;r_{ij},\sigma_{ij}),
$$

where $N$ is a normalized Gaussian.

## Total PDF

$$
g(r)=
\sum_{\alpha,\beta}
w_{\alpha\beta}g_{\alpha\beta}(r),
$$

$$
G(r)=
s\,4\pi\rho_0r[g(r)-1].
$$

An analytical shape envelope may modify both the peak contribution and the
negative baseline.

## Supported corrections

- spherical size envelope;
- cylinder/disk common volume;
- Biso broadening;
- pair-specific lambda and delta correlations;
- isotropic microstrain;
- Wilkens microstrain;
- PAH microstrain;
- `qdamp`;
- finite `qmax`;
- optional SAXS belly.

## Recommended use

Use this model as the starting point for periodic structures. Use finite-shape
mode only when explicit particle boundaries are required.