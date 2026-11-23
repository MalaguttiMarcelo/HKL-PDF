from __future__ import annotations

import numpy as np
import pytest

from pymatgen.core import Lattice
from pymatgen.core import Structure

from pdf_fitting.models.compiled_stacked_geometry import (
    compile_stacked_geometry,
)

from pdf_fitting.models.compiled_stacked_pairs import (
    compile_stacked_pair_topology,
)

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)

from pdf_fitting.models.gpu.backend import (
    cuda_available,
)

from pdf_fitting.models.gpu.dynamic_stacked_geometry import (
    DynamicStackedCudaGeometryBackend,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    SequenceLayer,
    layer_basis,
)


pytestmark = pytest.mark.skipif(
    not cuda_available(),
    reason="CUDA/CuPy is not available.",
)


def _make_test_structure() -> Structure:
    lattice = Lattice.from_parameters(
        5.0,
        5.0,
        8.0,
        90.0,
        90.0,
        90.0,
    )

    return Structure(
        lattice,
        [
            "C",
            "C",
            "N",
        ],
        [
            (0.00, 0.00, 0.00),
            (0.25, 0.25, 0.00),
            (0.50, 0.50, 0.05),
        ],
        coords_are_cartesian=False,
        to_unit_cell=False,
    )


def _make_shape_spec() -> CrystalliteShapeSpec:
    return CrystalliteShapeSpec(
        shape_type="cylinder",
        search_mode="predefined",
        axis1_cells=3.0,
        axis2_cells=3.0,
        axis3_cells=4.0,
        diameter_cells=3.0,
        height_cells=4.0,
        axis_h=0,
        axis_k=0,
        axis_l=1,
        base1_h=1,
        base1_k=0,
        base1_l=0,
        base2_h=0,
        base2_k=1,
        base2_l=0,
    )


def _make_stack_spec() -> LayerStackSpec:
    return LayerStackSpec(
        enabled=True,
        definition_mode="repeating_motif",
        vector_mode="absolute",
        spacing_mode="independent",
        nominal_spacing=3.20,
        refine_nominal_spacing=True,
        nominal_spacing_min=2.50,
        nominal_spacing_max=4.50,
        repeat_period=1.0,
        repeat_shift_u=0.0,
        repeat_shift_v=0.0,
        sequence=[
            SequenceLayer(
                enabled=True,
                label="A",
                shift_u=0.0,
                shift_v=0.0,
                stack_coordinate=0.0,
                normal_delta=0.0,
            ),
        ],
    ).normalized()


def _make_reference_parameters() -> dict:
    return {
        "a": 5.00,
        "b": 5.00,
        "c": 8.00,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "stacking_distance": 3.20,
    }


def _make_trial_parameters() -> dict:
    return {
        "a": 5.07,
        "b": 4.96,
        "c": 7.91,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 89.40,
        "stacking_distance": 3.35,
    }


def test_dynamic_stacked_cuda_geometry_matches_cpu() -> None:
    structure = _make_test_structure()
    shape_spec = _make_shape_spec()
    stack_spec = _make_stack_spec()

    reference_parameters = _make_reference_parameters()
    trial_parameters = _make_trial_parameters()

    compiled_geometry = compile_stacked_geometry(
        structure=structure,
        shape_spec=shape_spec,
        stack_spec=stack_spec,
        reference_parameters=reference_parameters,
    )

    compiled_pairs = compile_stacked_pair_topology(
        geometry=compiled_geometry,
        pair_cutoff=8.0,
        topology_margin=1.0,
        reference_parameters=reference_parameters,
        shell_backend="cpu",
    )

    cpu_evaluated = compiled_pairs.evaluate(
        trial_parameters
    )

    lattice_matrix = np.asarray(
        cpu_evaluated.geometry.parent_lattice_matrix,
        dtype=float,
    )

    vector_a, vector_b, stack_normal = layer_basis(
        lattice_matrix
    )

    gpu_backend = DynamicStackedCudaGeometryBackend(
        shell_delta_fractional=(
            compiled_pairs.shell_delta_fractional
        ),
        shell_delta_a=(
            compiled_pairs.shell_delta_a
        ),
        shell_delta_b=(
            compiled_pairs.shell_delta_b
        ),
        shell_delta_s=(
            compiled_pairs.shell_delta_s
        ),
        shell_delta_normal=(
            compiled_pairs.shell_delta_normal
        ),
        shell_alpha_id=(
            compiled_pairs.shell_alpha_id
        ),
        shell_mult=(
            compiled_pairs.shell_mult
        ),
        shell_bulk_mult=(
            compiled_pairs.shell_bulk_mult
        ),
        finite_species_counts=(
            compiled_pairs.finite_species_counts
        ),
        bulk_species_counts=(
            compiled_pairs.bulk_species_counts
        ),
        pair_cutoff=float(
            compiled_pairs.pair_cutoff
        ),
        device_id=0,
    )

    gpu_evaluated = gpu_backend.evaluate(
        lattice_matrix=lattice_matrix,
        vector_a=vector_a,
        vector_b=vector_b,
        stack_normal=stack_normal,
        spacing=float(
            cpu_evaluated.geometry.spacing
        ),
        stacking_inverse_lattice_matrix=(
            cpu_evaluated.geometry
            .stacking_inverse_lattice_matrix
        ),
    )

    np.testing.assert_allclose(
        gpu_evaluated[
            "shell_cart"
        ],
        cpu_evaluated.shell_cart,
        rtol=2.0e-5,
        atol=2.0e-5,
    )

    np.testing.assert_allclose(
        gpu_evaluated[
            "shell_frac"
        ],
        cpu_evaluated.shell_frac,
        rtol=2.0e-5,
        atol=2.0e-5,
    )

    np.testing.assert_allclose(
        gpu_evaluated[
            "shell_distance"
        ],
        cpu_evaluated.shell_distance,
        rtol=2.0e-5,
        atol=2.0e-5,
    )

    np.testing.assert_array_equal(
        gpu_evaluated[
            "shell_active"
        ],
        cpu_evaluated.shell_active,
    )

    np.testing.assert_allclose(
        gpu_evaluated[
            "shell_mult_eff"
        ],
        cpu_evaluated.shell_mult_eff,
        rtol=2.0e-5,
        atol=2.0e-5,
    )

    np.testing.assert_allclose(
        gpu_evaluated[
            "shell_bulk_mult_eff"
        ],
        cpu_evaluated.shell_bulk_mult_eff,
        rtol=2.0e-5,
        atol=2.0e-5,
    )

    np.testing.assert_allclose(
        gpu_evaluated[
            "shell_gamma"
        ],
        cpu_evaluated.shell_gamma,
        rtol=2.0e-5,
        atol=2.0e-5,
    )