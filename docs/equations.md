# Model Equations

## Number density

For a unit cell containing $N$ atoms and volume $V$,

$
\rho_0=\frac{N}{V}.
$

For neighbour species $\beta$,

$
\rho_\beta=\frac{N_\beta}{V}.
$

## Partial pair function

For a shell at distance $r_{ij}$,

$
A_{ij}=
\frac{m_{ij}}
{4\pi r_{ij}^{2}\rho_\beta}.
$

The shell is represented by a normalized Gaussian:

$
g_{\alpha\beta}^{(ij)}(r)=
A_{ij}
\frac{1}{\sqrt{2\pi}\sigma_{ij}}
\exp\left[
-\frac{(r-r_{ij})^2}{2\sigma_{ij}^2}
\right].
$

## Pair weights

The current X-ray weighting approximation uses atomic numbers $Z_\alpha$:

$
\langle Z\rangle=\sum_\alpha c_\alpha Z_\alpha,
$

$
w_{\alpha\beta}=
\frac{c_\alpha c_\beta Z_\alpha Z_\beta}
{\langle Z\rangle^2}.
$

The total pair distribution is

$
g(r)=\sum_{\alpha,\beta}
w_{\alpha\beta}g_{\alpha\beta}(r).
$

## Conventional reduced PDF

$
G(r)=4\pi\rho_0r[g(r)-1].
$

Including the scale $s$,

$
G(r)=s\,4\pi\rho_0r[g(r)-1].
$

## Peak variance

The shell variance is

$
\sigma_{ij}^{2}
=
\frac{B_{ij}}{4\pi^2}c_{ij}
+(\delta_g r_{ij})^2
+(\delta_{\mathrm{broad}}r_{ij})^2
+\sigma_{\mathrm{strain},ij}^2,
$

where

$
B_{ij}=\frac{B_i+B_j}{2}.
$

The default correlation term is

$
c_{ij}=
\max\left(
1-\frac{\delta_1}{r_{ij}}
-\frac{\delta_2}{r_{ij}^{2}},
0
\right).
$

If a pair-shell lambda is present,

$
c_{ij}=\max(1-\lambda_{\alpha\beta,k},0).
$

## Spherical finite-size envelope

For diameter $D$,

$
\gamma(r)=
1-\frac{3r}{2D}
+\frac{r^3}{2D^3},
\qquad r<D,
$

and

$
\gamma(r)=0,\qquad r\ge D.
$

The conventional spherical PDF is

$
G(r)=
s\,4\pi\rho_0r\,\gamma(r)[g(r)-1].
$

## Instrumental damping

$
G_{\mathrm{damped}}(r)=
G(r)
\exp\left(
-\frac{1}{2}Q_{\mathrm{damp}}^2r^2
\right).
$

## Finite-$Q_{\max}$ termination

The finite sine-transform kernel is

$
K(r,r')=
\frac{1}{\pi}
\left[
\frac{\sin(Q_{\max}(r-r'))}{r-r'}
-
\frac{\sin(Q_{\max}(r+r'))}{r+r'}
\right].
$

When termination is enabled, HKL-PDF evaluates the model on an internal grid
beginning at $r=0$. This reduces artificial low-$r$ boundary effects.

## Cylinder common volume

For pair length $L$ and angle $\phi$ to the cylinder axis,

$
z=L|\cos\phi|,
\qquad
r_\perp=L|\sin\phi|.
$

Let

$
x=\frac{r_\perp}{D}.
$

The radial overlap is

$
\gamma_{\mathrm{radial}}=
\frac{2}{\pi}
\left[
\cos^{-1}(x)-x\sqrt{1-x^2}
\right].
$

For monodisperse thickness $T$,

$
\gamma_{\mathrm{axial}}=
\max\left(1-\frac{z}{T},0\right).
$

Thus,

$
\gamma_{\mathrm{cyl}}=
\gamma_{\mathrm{radial}}\gamma_{\mathrm{axial}}.
$

## Wilkens microstrain

$
\langle\varepsilon^2(L)\rangle=
\frac{\rho b^2}{4\pi}
C_{hkl}
f^\ast\left(\frac{L}{R_e}\right),
$

$
\sigma_{\mathrm{W}}^2(L)=
L^2\langle\varepsilon^2(L)\rangle.
$

For $x=L/R_e\le1$,

$
f^\ast(x)=
-\ln x+\frac{7}{4}-\ln2
+\frac{x^2}{6}
-\frac{32x^3}{225\pi}.
$

For $x>1$,

$
f^\ast(x)=
\frac{512}{90\pi x}
-
\frac{\frac{11}{24}+\frac14\ln(2x)}
{x^2}.
$

## Cubic contrast factor

$
q_{hkl}=
\frac{h^2k^2+k^2l^2+l^2h^2}
{(h^2+k^2+l^2)^2},
$

$
C_{\mathrm{edge}}=
C_{\mathrm{Edge}A}
+C_{\mathrm{Edge}B}q_{hkl},
$

$
C_{\mathrm{screw}}=
C_{\mathrm{Screw}A}
+C_{\mathrm{Screw}B}q_{hkl},
$

$
C_{hkl}=
f_EC_{\mathrm{edge}}
+(1-f_E)C_{\mathrm{screw}}.
$

## PAH variance

$
\sigma_{\mathrm{PAH}}^2(L)=
I_{hkl}
\left(
a_{\mathrm{PAH}}L+
b_{\mathrm{PAH}}L^2
\right).
$

## SAXS belly

When enabled,

$
G_{\mathrm{SAXS}}(r)=
-s\,s_{\mathrm{SAXS}}\,
4\pi\rho_0r\,
\gamma_{\mathrm{SAXS}}(r).
$

## Refinement residual

$
e_i=
G_{\mathrm{exp}}(r_i)
-
G_{\mathrm{calc}}(r_i).
$

With weights $w_i$, the optimizer receives

$
e_i^{\mathrm{weighted}}=
\sqrt{w_i}\,e_i.
$

The displayed profile residual is

$
R_{\mathrm{wp}}=
100
\sqrt{
\frac{
\sum_iw_i
[G_{\mathrm{exp}}(r_i)-G_{\mathrm{calc}}(r_i)]^2
}{
\sum_iw_iG_{\mathrm{exp}}(r_i)^2
}
}.
$