# Free-Peak Fit and Width Analysis per Direction

The **Peak widths (free fit)** page of a phase has two steps:

1. a free-peak fit, in which every peak group gets its own width;
2. an analysis of those widths per crystallographic direction, similar to a
   Warren or Williamson–Hall analysis in real space.

The aim is to obtain **starting values** for the conventional refinement
and to show how the peak widths depend on direction. The results are
approximate. The conventional refinement remains the reference.

## 1 · Free-peak fit

Peak positions come from the structure of the phase (lattice and atomic
positions). Within the chosen $r$ range, every **peak group** gets a free
width $\sigma$ and a free amplitude multiplier. Everything else is the full
model: the shape function, `qdamp`, $Q_\mathrm{max}$ termination, the
baseline, the other phases of the dataset, and the peaks outside the range.

### Peak groups

Pair shells are first gathered into **orbits**: the same element pair, the
same distance, and symmetry-equivalent vectors. Orbits closer in distance
than the merge tolerance cannot be separated in a one-dimensional PDF, so
they form one **cluster**. A cluster is the fitted unit, with one $\sigma$
and one amplitude multiplier. A cluster with several orbits is marked as
*merged*; its $\sigma$ is an amplitude-weighted average of its members.

### Speed

The Jacobian is calculated analytically, cluster by cluster, as a sparse
matrix. One Jacobian costs about one model evaluation. Long ranges with
hundreds of clusters are therefore possible, but they still take time; for
example, about 2 minutes for Fe–Cr from 2 to 150 Å. **Stop** ends the fit
early.

The results are stored in the project and shown again when it is opened.

## 2 · Width analysis per direction

For each cluster the observed width is compared with

$$
\sigma_c^2 = \sum_m w_m\left[
\frac{B_{ab}}{4\pi^2}\,(1-\lambda_m)
+ \delta_\mathrm{broad}^2\,r_m^2
+ \sigma^2_{\mathrm{static},m}
\right],
$$

summed over the members $m$ of the cluster with weights $w_m$, where
$B_{ab} = (B_a + B_b)/2$.

The instrumental term $\delta_\mathrm{broad}$ is fixed and removed from what
is plotted. The plots show the sample widths
$\sigma^2_\mathrm{sample} = \sigma^2_\mathrm{obs} - \delta_\mathrm{broad}^2 r^2$.

### Thermal part

$\mathrm{MSRD} = (B/4\pi^2)(1-\lambda)$, following Scardi & Malagutti,
*Cryst. Growth Des.* **24** (2024) 4380. $\lambda$ is free per element pair
for the first shells; beyond them $\lambda = \delta_1/r + \delta_2/r^2$.
$B$, $\delta_1$, and $\delta_2$ can be shared by all directions, refined per
direction, or fixed.

### Static (microstrain) part

Along the direction $d$ of the pair vector, with the shape function
$f^{*}(r/R_e)\,r^2$ (Wilkens) or $r^2$ (Gaussian strain):

| Model | Static term |
|---|---|
| Isotropic | $\delta_g^2\,r^2$ |
| Per direction | $S_d \cdot \mathrm{shape}(r)$, one free $S_d$ per direction |
| Contrast-factor invariants | $S_d = \sum_i \tilde E_i\,x_i(hkl_d)$ |
| Phase contrast factors | $S_d = \frac{\rho b^2}{4\pi}\left[f_E C_\mathrm{edge}(d) + (1-f_E) C_\mathrm{screw}(d)\right]$ |

The invariant expansion of the powder-average contrast factor for the Laue
class of the phase follows Malagutti, D'Incau & Scardi, *J. Appl. Cryst.*
**59** (2026) 1720.

With the Wilkens shape, $S_d = \rho b^2 C_d / 4\pi$, exactly as in the PDF
model. The values per direction are therefore measured contrast factors up
to the common scale $\rho b^2/4\pi$. They are also fitted afterwards with
the invariant expansion and with the model contrast factors.

### Presets

| Preset | Meaning |
|---|---|
| Shared dynamics, independent strain per direction | One $B$; one $S_d$ per direction |
| Each direction independent | Own $B$ and $S_d$ per direction |
| Constrained by contrast-factor invariants | $S_d$ from the invariants $E_i$ |
| Constrained by the phase contrast factors | $\rho$ and $f_E$ free |
| Isotropic strain | One $\delta_g$ |

### Plots

- **Directions window…** opens a separate window with one plot per
  direction. Select the directions to show (Ctrl or Shift for several).
- The overview plots all directions together.
- The contrast plot compares the per-direction values with the fitted
  contrast factors.

The fit is weighted by the free-peak esds. The page reports the reduced
$\chi^2$, the parameter esds, and the correlations. The problem can be
ill-conditioned (Scardi, Leoni & Delhez, *J. Appl. Cryst.* **37** (2004)
381), so check the correlations before using the values.

### Using the results

- **Apply as starting values…** copies the selected results (B, λ,
  δ, ρ, …) to the phase parameters. Each value can be ticked or unticked.
- **Load microstructure model** fills the analysis from the current
  microstructure parameters of the phase.

## Limitations

- The PAH strain model is not yet available in the width analysis.
- Peak positions are fixed by the structure; lattice-only positions are not
  yet an option.
- Recovering the contrast-factor invariants $E_i$ from PDF widths alone is
  difficult. Treat them as optional.
