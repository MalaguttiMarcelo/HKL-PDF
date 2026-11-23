# GPU Acceleration and Performance

HKL-PDF runs on the CPU by default, using Numba kernels in parallel threads.
Large models can also run on an **NVIDIA GPU** through CuPy (CUDA).

AMD GPUs are not supported yet.

## Installing GPU support

1. Install a recent NVIDIA driver.
2. Install the CuPy build that matches your CUDA version, for example:

   ```bash
   python -m pip install cupy-cuda12x
   ```

3. Check it:

   ```bash
   python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
   ```

CuPy is optional. Without it, HKL-PDF runs on the CPU.

## Selecting the GPU

In **Refinement settings › Backends**:

| Setting | Options | Meaning |
|---|---|---|
| Shell generation | CPU · CUDA · Automatic | Where pair shells are generated and grouped |
| PDF accumulation | CPU · CUDA · Automatic | Where peak widths, Gaussians, and the PDF are evaluated |
| GPU device | 0, 1, … | Which GPU to use |
| GPU memory fraction | 0.05–0.95 | Share of GPU memory HKL-PDF may use |
| Numba threads | 1 … | CPU threads for the CPU kernels |

The same keys can be set in `[refinement]` of an `.inp` file or in
`refinement` of a project file:

```ini
[refinement]
shell_backend = cuda
pdf_backend = cuda
gpu_device = 0
gpu_memory_fraction = 0.7
```

The selected backends are shown in the chips at the top of the project
window. If the GPU cannot be used, HKL-PDF prints a message and continues on
the CPU.

## What runs on the GPU

| Calculation | GPU |
|---|---|
| Conventional model: shell grouping and PDF accumulation | Yes |
| Finite crystallite shapes | Yes |
| Deterministic layer stacking | Yes |
| Stochastic (A/B and Markov) layer stacking | Yes |
| Wilkens and PAH microstrain in stacking models | Yes |
| Cylinder SAXS term in stacking models | Yes |
| Peak widths with anisotropic ADPs | No (CPU) |
| $Q_\mathrm{max}$ termination | CPU (it takes only a few milliseconds) |

The crystallite-shape grid search uses the same backends as a normal
calculation.

## Measured speed

The measurements were made on an NVIDIA Quadro P4000 (8 GB) and a
multi-core workstation CPU. They give an indication only.

| Case | CPU | GPU |
|---|---|---|
| Stochastic COF stacking, one evaluation | 0.27 s | 0.047 s |
| Fe–Cr conventional model, 150 Å, one evaluation | 0.54 s | — |
| Calcite conventional model, 150 Å, one evaluation | 1.2 s | — |

## Caches

HKL-PDF stores reusable results in the user cache, normally
`%LOCALAPPDATA%\HKL-PDF` on Windows:

| Folder | Contents |
|---|---|
| `shell_cache` | Packed pair-shell tables of conventional models |
| `finite_shape_cache` | Finite-crystallite pair catalogues and shells |
| `cache/stochastic_topology` | Compiled stacking topologies (4 most recent) |
| `edited_structures` | CIFs derived from structure-editor edits |

Caches are versioned. When the program changes their format, old files are
ignored and rebuilt. They can be deleted safely at any time.

## Start-up

The project window loads heavy libraries (VTK, CuPy, the stacking models)
in the background shortly after it opens. The first **Calculate** or the
first **Structure (3D)** view is therefore faster. Importing CuPy can take
several seconds on network drives.

## Tips

- Use a coarse refinement $r$ step (`refinement_r_step_final`) and a fine
  final calculation.
- Calculate pair contributions, Warren plots, and local trends only when
  needed.
- For stacking models, keep the stacking setup unchanged between runs so the
  cached topology is reused.
- Set **Numba threads** to the number of physical cores if other programs
  need the CPU.
