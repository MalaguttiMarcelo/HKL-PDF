from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from PySide6 import QtWidgets

from vispy import scene
from vispy.scene import visuals

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
    build_orientation_basis,
    shape_dimension_lengths,
)


_ATOM_COLORS = {
    "H": (1.00, 1.00, 1.00, 1.00),
    "C": (0.25, 0.25, 0.25, 1.00),
    "N": (0.15, 0.25, 0.95, 1.00),
    "O": (0.95, 0.10, 0.10, 1.00),
    "F": (0.45, 0.85, 0.25, 1.00),
    "P": (1.00, 0.45, 0.00, 1.00),
    "S": (0.95, 0.85, 0.10, 1.00),
    "Cl": (0.10, 0.85, 0.10, 1.00),
    "Si": (0.90, 0.70, 0.55, 1.00),
    "Fe": (0.85, 0.35, 0.18, 1.00),
    "Co": (0.90, 0.50, 0.60, 1.00),
    "Ni": (0.25, 0.75, 0.25, 1.00),
    "Cu": (0.75, 0.45, 0.20, 1.00),
    "Zn": (0.45, 0.50, 0.70, 1.00),
    "Ca": (0.30, 0.90, 0.10, 1.00),
}


def clean_element_symbol(value: str) -> str:
    text = str(value or "").strip()

    if not text:
        return "X"

    first = text[0].upper()

    if len(text) > 1 and text[1].islower():
        return first + text[1].lower()

    return first


def color_for_element(element: str) -> Tuple[float, float, float, float]:
    return _ATOM_COLORS.get(
        clean_element_symbol(element),
        (0.50, 0.50, 0.50, 1.00),
    )


class FiniteCrystallitePreview(QtWidgets.QWidget):
    """
    VisPy preview for a finite crystallite and explicit layer stack.

    Camera fitting is controlled through ``auto_fit`` or ``fit_scene()``.
    Normal atom/table edits therefore remain visually observable instead
    of being hidden by automatic recentering.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.canvas = scene.SceneCanvas(
            keys="interactive",
            bgcolor="#F4F4F4",
            show=False,
        )

        self.view = self.canvas.central_widget.add_view()

        self.view.camera = scene.cameras.TurntableCamera(
            fov=45.0,
            distance=75.0,
            elevation=25.0,
            azimuth=35.0,
        )

        self.atom_markers = visuals.Markers(
            parent=self.view.scene
        )

        self.layer_origin_markers = visuals.Markers(
            parent=self.view.scene
        )

        self.layer_label_visual = scene.visuals.Text(
            parent=self.view.scene,
            color="black",
            font_size=12,
            anchor_x="center",
            anchor_y="bottom",
        )

        self.layer_label_visual.visible = False

        self._last_layer_labels = []
        self._last_layer_label_positions = np.zeros(
            (0, 3),
            dtype=np.float32,
        )

        self._show_parent_unit_cell = True
        self._show_lattice_vectors = True

        self.transition_lines = scene.visuals.Line(
            parent=self.view.scene,
            width=5.0,
            connect="segments",
        )
        self.transition_lines.visible = False

        self.cell_edges = scene.visuals.Line(
            parent=self.view.scene,
            color=(0.08, 0.08, 0.08, 1.0),
            width=2.0,
            connect="segments",
        )
        self.cell_edges.visible = False

        self.lattice_axis_lines = scene.visuals.Line(
            parent=self.view.scene,
            width=3.0,
            connect="segments",
        )
        self.lattice_axis_lines.visible = False

        self.shape_axis_lines = scene.visuals.Line(
            parent=self.view.scene,
            width=4.0,
            connect="segments",
        )
        self.shape_axis_lines.visible = False

        self.shape_surface = scene.visuals.Mesh(
            parent=self.view.scene,
            shading="smooth",
        )

        self.shape_surface.set_gl_state(
            "translucent",
            depth_test=True,
            cull_face=False,
        )
        self.shape_surface.visible = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas.native)

        self._last_atom_positions = np.zeros(
            (0, 3),
            dtype=np.float32,
        )

        self._last_layer_origins = np.zeros(
            (0, 3),
            dtype=np.float32,
        )

        self._last_transition_points = np.zeros(
            (0, 3),
            dtype=np.float32,
        )

        self._last_shape_vertices = np.zeros(
            (0, 3),
            dtype=np.float32,
        )

    def set_parent_unit_cell_visible(
        self,
        visible: bool,
    ) -> None:
        self._show_parent_unit_cell = bool(
            visible
        )

        self.cell_edges.visible = bool(
            visible
        )

        self.canvas.update()

    def set_lattice_vectors_visible(
        self,
        visible: bool,
    ) -> None:
        self._show_lattice_vectors = bool(
            visible
        )

        self.lattice_axis_lines.visible = bool(
            visible
        )

        self.canvas.update()

    def set_view_preset(
        self,
        preset: str,
    ) -> None:
        """
        Set standard crystallographic camera orientations.

        Default:
            ac_section

        This is intended for layered structures:
            a horizontal
            c vertical
            looking approximately along b.
        """
        preset = str(
            preset
            or "ac_section"
        ).strip().lower()

        camera = self.view.camera

        if preset == "ac_section":
            camera.azimuth = -90.0
            camera.elevation = 0.0

        elif preset == "bc_section":
            camera.azimuth = 0.0
            camera.elevation = 0.0

        elif preset == "ab_plane":
            camera.azimuth = 0.0
            camera.elevation = 90.0

        elif preset == "stack_axis":
            camera.azimuth = -90.0
            camera.elevation = 0.0

        elif preset == "isometric":
            camera.azimuth = 35.0
            camera.elevation = 25.0

        self.canvas.update()

    def set_custom_view_direction(
        self,
        direction,
        *,
        horizontal_direction=None,
    ) -> None:
        """
        Approximate custom view orientation.

        direction:
            Cartesian viewing direction.

        horizontal_direction:
            Preferred Cartesian horizontal direction.

        The current implementation maps the direction onto TurntableCamera
        azimuth/elevation. It is suitable for inspection and does not alter
        the physical structure.
        """
        direction = np.asarray(
            direction,
            dtype=float,
        )

        norm = float(
            np.linalg.norm(
                direction
            )
        )

        if norm <= 1.0e-12:
            return

        direction = direction / norm

        elevation = np.degrees(
            np.arcsin(
                np.clip(
                    direction[2],
                    -1.0,
                    1.0,
                )
            )
        )

        azimuth = np.degrees(
            np.arctan2(
                direction[1],
                direction[0],
            )
        )

        self.view.camera.azimuth = float(
            azimuth
        )

        self.view.camera.elevation = float(
            elevation
        )

        self.canvas.update()

    def save_screenshot(
        self,
        filename: str,
    ) -> None:
        image = self.canvas.render(
            alpha=True
        )

        from imageio.v3 import imwrite

        imwrite(
            filename,
            image,
        )

    def set_orthographic_projection(
        self,
        enabled: bool,
    ) -> None:
        self.view.camera.fov = (
            0.0
            if bool(enabled)
            else 45.0
        )

        self.canvas.update()

    def set_scene(
        self,
        *,
        lattice_matrix: np.ndarray,
        atom_cart: np.ndarray,
        atom_elements: List[str],
        atom_colors: Optional[np.ndarray],
        layer_origins: np.ndarray,
        show_atoms: bool,
        show_layer_origins: bool,
        preview_center: np.ndarray,
        shape_spec: CrystalliteShapeSpec,
        show_shape_surface: bool,
        shape_opacity: float,
        show_shape_axes: bool,
        transition_starts: Optional[np.ndarray] = None,
        transition_ends: Optional[np.ndarray] = None,
        show_transition_vectors: bool = False,
        layer_labels=None,
        show_layer_labels: bool = False,
        layer_label_font_size: int = 12,
        maximum_layer_labels: int = 50,
        auto_fit: bool = False,
    ) -> None:
        lattice_matrix = np.asarray(
            lattice_matrix,
            dtype=np.float32,
        )

        atom_cart = np.asarray(
            atom_cart,
            dtype=np.float32,
        )

        layer_origins = np.asarray(
            layer_origins,
            dtype=np.float32,
        )

        preview_center = np.asarray(
            preview_center,
            dtype=np.float32,
        )

        self._last_atom_positions = atom_cart
        self._last_layer_origins = layer_origins

        self._draw_unit_cell(
            lattice_matrix
        )

        self._draw_lattice_vectors(
            lattice_matrix
        )

        self.cell_edges.visible = bool(
            self._show_parent_unit_cell
        )

        self.lattice_axis_lines.visible = bool(
            self._show_lattice_vectors
        )

        self._draw_atoms(
            atom_cart,
            atom_elements,
            colors=atom_colors,
            visible=show_atoms,
        )

        self._draw_layer_origins(
            layer_origins,
            visible=show_layer_origins,
        )

        self.set_layer_labels(
            layer_origins,
            layer_labels,
            visible=show_layer_labels,
            font_size=layer_label_font_size,
            maximum_labels=maximum_layer_labels,
        )

        self._draw_transition_vectors(
            transition_starts,
            transition_ends,
            visible=show_transition_vectors,
        )

        self._draw_shape_axes(
            lattice_matrix=lattice_matrix,
            center=preview_center,
            shape_spec=shape_spec,
            visible=show_shape_axes,
        )

        self._draw_shape_surface(
            lattice_matrix=lattice_matrix,
            center=preview_center,
            shape_spec=shape_spec,
            visible=show_shape_surface,
            opacity=shape_opacity,
        )

        if auto_fit:
            self.fit_scene()
        else:
            self.canvas.update()

    def clear_scene(self) -> None:
        self.atom_markers.set_data(
            np.zeros((0, 3), dtype=np.float32)
        )

        self.layer_origin_markers.set_data(
            np.zeros((0, 3), dtype=np.float32)
        )

        self.transition_lines.visible = False
        self.shape_surface.visible = False
        self.shape_axis_lines.visible = False

        self._last_atom_positions = np.zeros(
            (0, 3),
            dtype=np.float32,
        )

        self._last_layer_origins = np.zeros(
            (0, 3),
            dtype=np.float32,
        )

        self._last_transition_points = np.zeros(
            (0, 3),
            dtype=np.float32,
        )

        self._last_shape_vertices = np.zeros(
            (0, 3),
            dtype=np.float32,
        )

        self.canvas.update()

    def _draw_atoms(
        self,
        coordinates: np.ndarray,
        elements: List[str],
        *,
        colors: Optional[np.ndarray],
        visible: bool,
    ) -> None:
        if (
            not visible
            or coordinates.ndim != 2
            or coordinates.shape[1] != 3
            or coordinates.shape[0] == 0
        ):
            self.atom_markers.set_data(
                np.zeros((0, 3), dtype=np.float32)
            )
            return

        if colors is None:
            face_colors = np.asarray(
                [
                    color_for_element(element)
                    for element in elements
                ],
                dtype=np.float32,
            )
        else:
            face_colors = np.asarray(
                colors,
                dtype=np.float32,
            )

            valid = (
                face_colors.ndim == 2
                and face_colors.shape[0]
                == coordinates.shape[0]
                and face_colors.shape[1] in (3, 4)
            )

            if not valid:
                face_colors = np.asarray(
                    [
                        color_for_element(element)
                        for element in elements
                    ],
                    dtype=np.float32,
                )

            elif face_colors.shape[1] == 3:
                alpha = np.ones(
                    (face_colors.shape[0], 1),
                    dtype=np.float32,
                )

                face_colors = np.hstack(
                    [face_colors, alpha]
                )

        self.atom_markers.set_data(
            coordinates,
            face_color=face_colors,
            edge_color=(0.04, 0.04, 0.04, 1.0),
            size=10.0,
            symbol="disc",
            edge_width=0.8,
        )

    def _draw_layer_origins(
        self,
        origins: np.ndarray,
        *,
        visible: bool,
    ) -> None:
        if (
            not visible
            or origins.ndim != 2
            or origins.shape[1] != 3
            or origins.shape[0] == 0
        ):
            self.layer_origin_markers.set_data(
                np.zeros(
                    (0, 3),
                    dtype=np.float32,
                )
            )
            return

        palette = np.asarray(
            [
                (0.90, 0.15, 0.15, 0.95),
                (0.10, 0.45, 0.90, 0.95),
                (0.15, 0.70, 0.25, 0.95),
                (0.82, 0.35, 0.85, 0.95),
                (0.95, 0.55, 0.10, 0.95),
                (0.10, 0.72, 0.72, 0.95),
            ],
            dtype=np.float32,
        )

        colors = palette[
            np.arange(
                origins.shape[0]
            )
            % palette.shape[0]
        ]

        self.layer_origin_markers.set_data(
            origins,
            face_color=colors,
            edge_color=(0.02, 0.02, 0.02, 1.0),
            size=15.0,
            symbol="diamond",
            edge_width=1.5,
        )

    def set_layer_labels(
        self,
        positions: np.ndarray,
        labels,
        *,
        visible: bool,
        font_size: int = 12,
        maximum_labels: int = 50,
    ) -> None:
        positions = np.asarray(
            positions,
            dtype=np.float32,
        )

        labels = list(
            labels
            or []
        )

        if (
            not visible
            or positions.ndim != 2
            or positions.shape[1] != 3
            or positions.shape[0] == 0
            or not labels
        ):
            self.layer_label_visual.visible = False
            self.layer_label_visual.text = []
            self.layer_label_visual.pos = np.zeros(
                (0, 3),
                dtype=np.float32,
            )
            self.canvas.update()
            return

        maximum_labels = max(
            1,
            int(
                maximum_labels
            ),
        )

        number = min(
            positions.shape[0],
            len(labels),
            maximum_labels,
        )

        positions = positions[:number]
        labels = labels[:number]

        self._last_layer_labels = list(
            labels
        )

        self._last_layer_label_positions = positions.copy()

        self.layer_label_visual.text = [
            str(label)
            for label in labels
        ]

        self.layer_label_visual.pos = positions

        self.layer_label_visual.font_size = int(
            font_size
        )

        self.layer_label_visual.visible = True

        self.canvas.update()

    def _draw_transition_vectors(
        self,
        starts: Optional[np.ndarray],
        ends: Optional[np.ndarray],
        *,
        visible: bool,
    ) -> None:
        if (
            not visible
            or starts is None
            or ends is None
        ):
            self.transition_lines.visible = False

            self._last_transition_points = np.zeros(
                (0, 3),
                dtype=np.float32,
            )
            return

        starts = np.asarray(
            starts,
            dtype=np.float32,
        )

        ends = np.asarray(
            ends,
            dtype=np.float32,
        )

        valid = (
            starts.ndim == 2
            and ends.ndim == 2
            and starts.shape == ends.shape
            and starts.shape[1] == 3
            and starts.shape[0] > 0
        )

        if not valid:
            self.transition_lines.visible = False

            self._last_transition_points = np.zeros(
                (0, 3),
                dtype=np.float32,
            )
            return

        positions = np.empty(
            (starts.shape[0] * 2, 3),
            dtype=np.float32,
        )

        positions[0::2] = starts
        positions[1::2] = ends

        self.transition_lines.set_data(
            pos=positions,
            color=(1.0, 0.35, 0.02, 1.0),
            width=5.0,
            connect="segments",
        )

        self.transition_lines.visible = True
        self._last_transition_points = positions

    def _draw_unit_cell(
        self,
        matrix: np.ndarray,
    ) -> None:
        a = matrix[0]
        b = matrix[1]
        c = matrix[2]

        zero = np.zeros(
            3,
            dtype=np.float32,
        )

        corners = np.asarray(
            [
                zero,
                a,
                b,
                c,
                a + b,
                a + c,
                b + c,
                a + b + c,
            ],
            dtype=np.float32,
        )

        edges = (
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (1, 5),
            (2, 4),
            (2, 6),
            (3, 5),
            (3, 6),
            (4, 7),
            (5, 7),
            (6, 7),
        )

        positions = np.asarray(
            [
                corners[index]
                for edge in edges
                for index in edge
            ],
            dtype=np.float32,
        )

        self.cell_edges.set_data(
            pos=positions,
            color=(0.08, 0.08, 0.08, 1.0),
            width=2.0,
            connect="segments",
        )

        self.cell_edges.visible = True

    def _draw_lattice_vectors(
        self,
        matrix: np.ndarray,
    ) -> None:
        zero = np.zeros(
            3,
            dtype=np.float32,
        )

        positions = np.asarray(
            [
                zero,
                matrix[0],
                zero,
                matrix[1],
                zero,
                matrix[2],
            ],
            dtype=np.float32,
        )

        colors = np.asarray(
            [
                (0.90, 0.10, 0.10, 1.0),
                (0.90, 0.10, 0.10, 1.0),
                (0.10, 0.70, 0.15, 1.0),
                (0.10, 0.70, 0.15, 1.0),
                (0.10, 0.25, 0.95, 1.0),
                (0.10, 0.25, 0.95, 1.0),
            ],
            dtype=np.float32,
        )

        self.lattice_axis_lines.set_data(
            pos=positions,
            color=colors,
            width=3.0,
            connect="segments",
        )

        self.lattice_axis_lines.visible = True

    @staticmethod
    def _shape_geometry(
        lattice_matrix: np.ndarray,
        shape_spec: CrystalliteShapeSpec,
    ):
        e1, e2, e3 = build_orientation_basis(
            lattice_matrix,
            axis_hkl=(
                shape_spec.axis_h,
                shape_spec.axis_k,
                shape_spec.axis_l,
            ),
            base1_hkl=(
                shape_spec.base1_h,
                shape_spec.base1_k,
                shape_spec.base1_l,
            ),
            base2_hkl=(
                shape_spec.base2_h,
                shape_spec.base2_k,
                shape_spec.base2_l,
            ),
        )

        length_1, length_2, length_3 = (
            shape_dimension_lengths(
                lattice_matrix,
                axis_hkl=(
                    shape_spec.axis_h,
                    shape_spec.axis_k,
                    shape_spec.axis_l,
                ),
                base1_hkl=(
                    shape_spec.base1_h,
                    shape_spec.base1_k,
                    shape_spec.base1_l,
                ),
                base2_hkl=(
                    shape_spec.base2_h,
                    shape_spec.base2_k,
                    shape_spec.base2_l,
                ),
            )
        )

        axis1_cells = float(
            getattr(
                shape_spec,
                "axis1_cells",
                shape_spec.diameter_cells,
            )
        )

        axis2_cells = float(
            getattr(
                shape_spec,
                "axis2_cells",
                shape_spec.diameter_cells,
            )
        )

        axis3_cells = float(
            getattr(
                shape_spec,
                "axis3_cells",
                shape_spec.height_cells,
            )
        )

        shape_type = str(
            shape_spec.shape_type
        ).strip().lower()

        if shape_type in {"sphere", "spherical"}:
            axis2_cells = axis1_cells
            axis3_cells = axis1_cells

        elif shape_type in {
            "cylinder",
            "disk",
            "disk-like",
            "disk_like",
        }:
            axis2_cells = axis1_cells

        half_1 = max(
            0.5 * axis1_cells * length_1,
            1.0e-6,
        )

        half_2 = max(
            0.5 * axis2_cells * length_2,
            1.0e-6,
        )

        half_3 = max(
            0.5 * axis3_cells * length_3,
            1.0e-6,
        )

        return (
            shape_type,
            e1,
            e2,
            e3,
            half_1,
            half_2,
            half_3,
        )

    def _draw_shape_axes(
        self,
        *,
        lattice_matrix: np.ndarray,
        center: np.ndarray,
        shape_spec: CrystalliteShapeSpec,
        visible: bool,
    ) -> None:
        if not visible:
            self.shape_axis_lines.visible = False
            return

        (
            _,
            e1,
            e2,
            e3,
            half_1,
            half_2,
            half_3,
        ) = self._shape_geometry(
            lattice_matrix,
            shape_spec,
        )

        length = max(
            half_1,
            half_2,
            half_3,
            5.0,
        ) * 1.2

        positions = np.asarray(
            [
                center,
                center + length * e1,
                center,
                center + length * e2,
                center,
                center + length * e3,
            ],
            dtype=np.float32,
        )

        colors = np.asarray(
            [
                (0.10, 0.70, 0.15, 1.0),
                (0.10, 0.70, 0.15, 1.0),
                (0.65, 0.10, 0.85, 1.0),
                (0.65, 0.10, 0.85, 1.0),
                (0.95, 0.10, 0.10, 1.0),
                (0.95, 0.10, 0.10, 1.0),
            ],
            dtype=np.float32,
        )

        self.shape_axis_lines.set_data(
            pos=positions,
            color=colors,
            width=4.0,
            connect="segments",
        )

        self.shape_axis_lines.visible = True

    def _draw_shape_surface(
        self,
        *,
        lattice_matrix: np.ndarray,
        center: np.ndarray,
        shape_spec: CrystalliteShapeSpec,
        visible: bool,
        opacity: float,
    ) -> None:
        if not visible:
            self.shape_surface.visible = False

            self._last_shape_vertices = np.zeros(
                (0, 3),
                dtype=np.float32,
            )
            return

        (
            shape_type,
            e1,
            e2,
            e3,
            half_1,
            half_2,
            half_3,
        ) = self._shape_geometry(
            lattice_matrix,
            shape_spec,
        )

        if shape_type in {"sphere", "spherical"}:
            vertices, faces = self._ellipsoid_mesh(
                center=center,
                e1=e1,
                e2=e2,
                e3=e3,
                radius_1=half_1,
                radius_2=half_1,
                radius_3=half_1,
            )

        elif shape_type in {
            "ellipsoid",
            "ellipsoidal",
        }:
            vertices, faces = self._ellipsoid_mesh(
                center=center,
                e1=e1,
                e2=e2,
                e3=e3,
                radius_1=half_1,
                radius_2=half_2,
                radius_3=half_3,
            )

        elif shape_type in {
            "prism",
            "rectangular_prism",
        }:
            vertices, faces = self._prism_mesh(
                center=center,
                e1=e1,
                e2=e2,
                e3=e3,
                half_1=half_1,
                half_2=half_2,
                half_3=half_3,
            )

        else:
            vertices, faces = self._cylinder_mesh(
                center=center,
                e1=e1,
                e2=e2,
                e3=e3,
                radius_1=half_1,
                radius_2=half_2,
                half_height=half_3,
            )

        alpha = max(
            0.02,
            min(0.90, float(opacity)),
        )

        color = np.asarray(
            [0.15, 0.45, 0.95, alpha],
            dtype=np.float32,
        )

        face_colors = np.tile(
            color,
            (faces.shape[0], 1),
        )

        self.shape_surface.set_data(
            vertices=vertices,
            faces=faces,
            face_colors=face_colors,
        )

        self.shape_surface.set_gl_state(
            "translucent",
            depth_test=True,
            cull_face=False,
        )

        self.shape_surface.visible = True
        self._last_shape_vertices = vertices

    @staticmethod
    def _ellipsoid_mesh(
        *,
        center,
        e1,
        e2,
        e3,
        radius_1,
        radius_2,
        radius_3,
        n_theta=40,
        n_phi=20,
    ):
        theta = np.linspace(
            0.0,
            2.0 * np.pi,
            n_theta,
            endpoint=False,
        )

        phi = np.linspace(
            0.0,
            np.pi,
            n_phi,
        )

        vertices = []

        for phi_value in phi:
            for theta_value in theta:
                vertices.append(
                    center
                    + radius_1
                    * np.sin(phi_value)
                    * np.cos(theta_value)
                    * e1
                    + radius_2
                    * np.sin(phi_value)
                    * np.sin(theta_value)
                    * e2
                    + radius_3
                    * np.cos(phi_value)
                    * e3
                )

        faces = []

        for ip in range(n_phi - 1):
            for it in range(n_theta):
                next_it = (it + 1) % n_theta

                a = ip * n_theta + it
                b = ip * n_theta + next_it
                c = (ip + 1) * n_theta + it
                d = (ip + 1) * n_theta + next_it

                faces.append((a, c, b))
                faces.append((b, c, d))

        return (
            np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.uint32),
        )

    @staticmethod
    def _prism_mesh(
        *,
        center,
        e1,
        e2,
        e3,
        half_1,
        half_2,
        half_3,
    ):
        vertices = np.asarray(
            [
                center - half_1 * e1 - half_2 * e2 - half_3 * e3,
                center + half_1 * e1 - half_2 * e2 - half_3 * e3,
                center + half_1 * e1 + half_2 * e2 - half_3 * e3,
                center - half_1 * e1 + half_2 * e2 - half_3 * e3,
                center - half_1 * e1 - half_2 * e2 + half_3 * e3,
                center + half_1 * e1 - half_2 * e2 + half_3 * e3,
                center + half_1 * e1 + half_2 * e2 + half_3 * e3,
                center - half_1 * e1 + half_2 * e2 + half_3 * e3,
            ],
            dtype=np.float32,
        )

        faces = np.asarray(
            [
                (0, 2, 1),
                (0, 3, 2),
                (4, 5, 6),
                (4, 6, 7),
                (0, 1, 5),
                (0, 5, 4),
                (1, 2, 6),
                (1, 6, 5),
                (2, 3, 7),
                (2, 7, 6),
                (3, 0, 4),
                (3, 4, 7),
            ],
            dtype=np.uint32,
        )

        return vertices, faces

    @staticmethod
    def _cylinder_mesh(
        *,
        center,
        e1,
        e2,
        e3,
        radius_1,
        radius_2,
        half_height,
        n_segments=48,
    ):
        theta = np.linspace(
            0.0,
            2.0 * np.pi,
            n_segments,
            endpoint=False,
        )

        vertices = []

        for z_sign in (-1.0, 1.0):
            for theta_value in theta:
                vertices.append(
                    center
                    + z_sign * half_height * e3
                    + radius_1 * np.cos(theta_value) * e1
                    + radius_2 * np.sin(theta_value) * e2
                )

        vertices.append(
            center - half_height * e3
        )
        bottom_center = len(vertices) - 1

        vertices.append(
            center + half_height * e3
        )
        top_center = len(vertices) - 1

        faces = []

        for index in range(n_segments):
            next_index = (
                index + 1
            ) % n_segments

            bottom_a = index
            bottom_b = next_index
            top_a = n_segments + index
            top_b = n_segments + next_index

            faces.append(
                (bottom_a, top_a, bottom_b)
            )
            faces.append(
                (bottom_b, top_a, top_b)
            )

            faces.append(
                (
                    bottom_center,
                    bottom_b,
                    bottom_a,
                )
            )

            faces.append(
                (
                    top_center,
                    top_a,
                    top_b,
                )
            )

        return (
            np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.uint32),
        )

    def fit_scene(self) -> None:
        all_points = []

        for points in (
            self._last_atom_positions,
            self._last_layer_origins,
            self._last_transition_points,
            self._last_shape_vertices,
        ):
            if (
                points.ndim == 2
                and points.shape[0] > 0
                and points.shape[1] == 3
            ):
                all_points.append(points)

        if not all_points:
            return

        points = np.vstack(all_points)

        lower = np.min(
            points,
            axis=0,
        )

        upper = np.max(
            points,
            axis=0,
        )

        center = 0.5 * (
            lower + upper
        )

        extent = max(
            float(
                np.max(
                    upper - lower
                )
            ),
            5.0,
        )

        self.view.camera.center = center
        self.view.camera.distance = (
            2.2 * extent + 10.0
        )

        self.canvas.update()