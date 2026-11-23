# Anisotropic Displacement Parameters

Sites can use anisotropic displacement parameters (ADPs) instead of an
isotropic Biso. HKL-PDF parametrises them with the dimensionless
$\beta_{ij}$ tensor, as in the CIF `_atom_site_aniso_beta_*` convention.

## Parameters

Each anisotropic site has six components:

```text
beta11_<site>  beta22_<site>  beta33_<site>
beta12_<site>  beta13_<site>  beta23_<site>
```

The atomic displacement factor is

$$
T(\mathbf{h}) = \exp\left(-\mathbf{h}^\mathsf{T}\boldsymbol{\beta}\,\mathbf{h}\right)
= \exp\left(-\sum_{ij}\beta_{ij}h_i h_j\right).
$$

The relation to $U_{ij}$ is
$\beta_{ij} = 2\pi^2\,a_i^*a_j^*\,U_{ij}$.

## Site symmetry

The tensor of each site must be invariant under its site-symmetry
operations. HKL-PDF finds, for each component, whether it is:

- **free:** refined independently;
- **tied:** set from other components, for example $\beta_{22} = \beta_{11}$
  or $\beta_{12} = \beta_{11}/2$ in hexagonal sites;
- **fixed:** zero by symmetry.

Tied and fixed components are locked in the grid. During refinement only
the free components move, and the tensor is symmetrised after every step.
Every atom of an orbit receives the tensor rotated by its own symmetry
operation.

## Effect on the PDF peak width

For a pair of atoms $a$ and $b$ at separation $\mathbf{r}$, the mean-square
displacement along the pair direction $\hat{\mathbf{n}}$ is

$$
\langle u_n^2\rangle_a =
\frac{(R^\mathsf{T}\mathbf{v})^\mathsf{T}\,\boldsymbol{\beta}\,(R^\mathsf{T}\mathbf{v})}{2\pi^2},
\qquad
\mathbf{v} = \frac{G\,\mathbf{f}}{r},
$$

where $\mathbf{f}$ is the fractional separation, $G$ the direct metric
tensor, and $R$ the rotation of the atom's orbit position. The Gaussian
width of the pair peak is

$$
\sigma^2 = \left(\langle u_n^2\rangle_a + \langle u_n^2\rangle_b\right)(1-\lambda)
$$

followed by the usual correlated-motion, microstrain, and instrumental
terms. Isotropic sites keep using Biso.

The equivalent isotropic value is

$$
B_\mathrm{eq} = \frac{4}{3}\,\mathrm{tr}\!\left(\boldsymbol{\beta}\,G\right).
$$

It is shown beside the tensor on the Atoms page.

## Using ADPs

On **Atoms & occupancy**:

1. select a site;
2. click **Make anisotropic**. The tensor starts from the isotropic
   equivalent of the current Biso, $\beta = (B/4)\,G^{*}$, so the
   calculated PDF does not change;
3. flag the free components for refinement.

**Make isotropic** removes the $\beta$ parameters of the site. When Biso is
refined per site, the site Biso is set to $B_\mathrm{eq}$.

Anisotropic ADPs add up to six parameters per site and correlate strongly
with Biso, $\lambda$, and microstrain parameters. Refine them only when the
data range and quality support it, and check the
[correlation matrix](uncertainties.md).

## CIF import and export

- CIF files with `_atom_site_aniso_beta_*`, `_atom_site_aniso_U_*`, or
  `_atom_site_aniso_B_*` loops are imported as $\beta$ tensors.
- **Export CIF…** in the structure editor writes the $\beta$ loop.

## Display

**Thermal ellipsoids (50 %)** in the structure editor draws the 50 %
probability ellipsoids of the anisotropic sites.

## Limitations

- When ADPs are active, the peak widths are calculated on the CPU even if
  the CUDA PDF backend is selected.
- ADPs are not used by the stacking-fault models.
