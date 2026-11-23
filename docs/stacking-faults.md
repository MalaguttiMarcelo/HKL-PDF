# Layer Stacking and Stacking Faults

HKL-PDF can model layered materials as a sequence of layers with defined or
random stacking vectors. The layers can fill a finite crystallite or act as
the bulk material.

Open the **crystallite & stacking builder** from
**Shape & stacking › Open crystallite & stacking builder…** in the project
window, or from the size-model section of the classic Builder.

## Stacking models

| Model | Input name | Description |
|---|---|---|
| Deterministic explicit sequence | `periodic_sequence` | A fixed, repeating sequence of layers (for example A B or A B C) with their own shifts |
| Correlated stochastic A/B stacking | `correlated_ab_disorder` | Two stacking states A and B; each new layer keeps or changes the state with given probabilities (similar to the TOPAS approach) |
| Generic multi-state Markov stacking | `generic_markov` | Any number of states and a full transition-probability matrix |

Aliases such as `markov_ab`, `stochastic_ab`, or `multistate_markov` are
accepted in input files.

## Layer definition

The source structure (CIF) provides one layer. The stacking direction is
normally $c$; the lateral shifts are given in the $a$–$b$ basis of the
layer.

- **Layer spacing and repeat translation:** the nominal spacing between
  layers and the repeat translation of the periodic sequence.
- **Sequence rows:** label, lateral shift $(u, v)$, and spacing change of
  each layer. Rows can be added, moved, and duplicated. **Generate A**,
  **Generate A/B**, and **Generate A/B/C** fill common sequences.
- **Vectors** can be incremental (from the previous layer) or absolute
  (from the sequence origin).

## Stochastic models

Each **state** defines a stacking vector: the lateral shifts $du$ and $dv$
and the spacing $h$ (Å). For the A/B model these are the refinable
parameters `stack_state_a_du`, `stack_state_a_dv`, `stack_state_a_h`, and
the same for B. The transition probabilities set how likely each state is
to follow another.

A finite crystallite contains an **ensemble** of random realisations. The
ensemble settings are:

| Setting | Meaning |
|---|---|
| Ensemble samples | Number of random realisations averaged |
| Ensemble seed | Random seed (fixed for reproducible results) |
| Start state | State of the first layer |
| Bulk layers | Layers used when the stacking is treated as bulk |

**Regenerate random stacking** changes only the seed of the displayed
realisation; it does not change the model.

## Crystallite shape

The stacking can fill a finite crystallite: a prism, cylinder or disk,
sphere, or ellipsoid, oriented by the stack axis and the in-plane base
directions. Dimensions are given in unit cells. **Pre-defined dimensions**
calculates one shape; **Grid search / scan dimensions** scans a range.

## Searches and refinement

- **Stacking search:** scans lateral shifts $u$ and $v$ on a regular grid
  or with random samples, with fixed or refined spacing.
- **Transition-probability grid:** scans the probability of alternating
  states (outer grid search).
- **Refine stochastic State A/B vectors:** refines the state vectors in the
  conventional least-squares fit. The refinable stacking parameters appear
  in **Shape & stacking › Stacking parameters**.

## Calculation

The stacking model is compiled once into a pair **topology**: every
distinct pair of layer positions, with the coefficients that combine the
state vectors. Each later evaluation only updates the vectors and
probabilities, so refinement steps are fast.

- The compiled topology is kept for the session and also saved, slimmed,
  in the user cache (`cache/stochastic_topology`, the 4 most recent setups). A new
  session with the same stacking setup loads it instead of recompiling.
- The deterministic and stochastic models can evaluate on an NVIDIA GPU.
  See [GPU acceleration](gpu.md).
- Wilkens and PAH microstrain, finite size, and the cylinder SAXS term are
  supported on both CPU and GPU.
- Anisotropic ADPs and occupancy refinement are not used by the stacking
  models.

## Visualisation and export

The builder shows the generated layers in 3D with layer labels, origins,
stacking vectors, the parent cell, and the crystallite envelope. It can
export:

- the realised layers and stacking vectors as CSV;
- the displayed prism realisation as a periodic CIF supercell;
- XYZ coordinates;
- a screenshot of the preview.

## Notes

- The first compilation of a large stochastic ensemble can take about
  20 seconds. Later evaluations take a fraction of a second.
- Stacking models are new and have been tested on a limited set of
  materials. Please report problems with a minimal example.
