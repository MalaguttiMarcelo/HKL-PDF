from __future__ import annotations

import os
import re
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from PySide6 import QtWidgets

from pymatgen.core.structure import Structure

from .fast_shape_viewer import FastShapeViewer
from .cif_utils import read_cif_asu_sites, clean_el_symbol


_ATOM_COLORS = {
    "H": "#FFFFFF",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Cl": "#1FF01F",
    "Br": "#A62929",
    "I": "#940094",
    "S": "#FFFF30",
    "P": "#FF8000",
    "B": "#FFB5B5",
    "Si": "#F0C8A0",
    "Li": "#CC80FF",
    "Na": "#AB5CF2",
    "K": "#8F40D4",
    "Mg": "#8AFF00",
    "Ca": "#3DFF00",
    "Ti": "#BFC2C7",
    "Fe": "#E06633",
    "Co": "#F090A0",
    "Ni": "#50D050",
    "Cu": "#C88033",
    "Zn": "#7D80B0",
    "Al": "#BFA6A6",
    "Ge": "#668F8F",
    "Zr": "#94E0E0",
    "Mo": "#54B5B5",
    "Ag": "#C0C0C0",
    "Sn": "#668080",
    "Pd": "#006985",
}


def _hex_to_rgb_u8(hex_color: str) -> Tuple[int, int, int]:
    s = str(hex_color).strip()

    if s.startswith("#"):
        s = s[1:]

    if len(s) != 6:
        return (128, 128, 128)

    try:
        return (
            int(s[0:2], 16),
            int(s[2:4], 16),
            int(s[4:6], 16),
        )
    except Exception:
        return (128, 128, 128)


def _rgb_for_element(el: str) -> Tuple[int, int, int]:
    el = clean_el_symbol(el)

    return _hex_to_rgb_u8(
        _ATOM_COLORS.get(
            el,
            "#808080",
        )
    )


class StructureViewer(QtWidgets.QWidget):
    """
    Fast OpenGL/VTK structure viewer.

    This replaces the old Matplotlib 3D structure viewer.

    Public API intentionally matches the old StructureViewer:
        clear()
        load_structure(path)
        set_biso_by_species(...)
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._structure: Optional[Structure] = None
        self._loaded_structure_path: str = ""
        self._last_points = np.zeros((0, 3), dtype=np.float32)
        self._last_rgb = np.zeros((0, 3), dtype=np.uint8)
        self._last_elements: List[str] = []

        self.viewer = FastShapeViewer(self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.viewer, 1)

    def clear(self) -> None:
        self._structure = None
        self._loaded_structure_path = ""
        self._last_points = np.zeros((0, 3), dtype=np.float32)
        self._last_rgb = np.zeros((0, 3), dtype=np.uint8)
        self._last_elements = []

        try:
            self.viewer.clear_scene()
        except Exception:
            pass

    def set_biso_by_species(self, biso_by_species: Dict[str, float]) -> None:
        """
        Kept for compatibility with MainWindow.

        The new fast viewer does not display the atom table, so there is
        nothing to update visually here.
        """
        return

    def _unit_cell_corners(self, structure: Structure) -> np.ndarray:
        M = np.asarray(structure.lattice.matrix, dtype=float)

        a = M[0]
        b = M[1]
        c = M[2]
        O = np.zeros(3, dtype=float)

        corners = np.array(
            [
                O,
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

        return corners

    def _load_structure_object(self, path: str) -> Optional[Structure]:
        path = str(path or "").strip()

        if not path:
            return None

        if not os.path.exists(path):
            return None

        try:
            if path.lower().endswith(".cif"):
                structure, _asu = read_cif_asu_sites(path)

                if structure is not None:
                    return structure

            return Structure.from_file(path)

        except Exception:
            try:
                return Structure.from_file(path)
            except Exception:
                return None

    def _structure_points_and_colors(
        self,
        structure: Structure,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        points = []
        colors = []
        elements = []

        for site in structure.sites:
            try:
                el = clean_el_symbol(str(site.specie))
            except Exception:
                el = "X"

            try:
                xyz = np.asarray(site.coords, dtype=float)
            except Exception:
                continue

            points.append(xyz)
            colors.append(_rgb_for_element(el))
            elements.append(el)

        if not points:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                [],
            )

        return (
            np.asarray(points, dtype=np.float32),
            np.asarray(colors, dtype=np.uint8),
            elements,
        )

    def load_structure(self, path: str) -> None:
        path = str(path or "").strip()
    
        try:
            path_abs = os.path.abspath(path) if path else ""
        except Exception:
            path_abs = path
    
        if (
            path_abs
            and path_abs == self._loaded_structure_path
            and self._structure is not None
        ):
            return
    
        if not path or not os.path.exists(path):
            self.clear()
            return
    
        structure = self._load_structure_object(path)
    
        if structure is None:
            self.clear()
            return
    
        self._structure = structure
        self._loaded_structure_path = path_abs
    
        points, rgb, elements = self._structure_points_and_colors(
            structure
        )
    
        self._last_points = points
        self._last_rgb = rgb
        self._last_elements = elements
    
        n_atoms = int(points.shape[0])
    
        try:
            self.viewer.clear_scene()
        except Exception:
            pass
        
        # Real sphere glyphs give proper three-dimensional shading.
        #
        # For extremely large structures, fall back to GPU point spheres
        # to avoid generating millions of sphere triangles.
        use_sphere_glyphs = n_atoms <= 25000
    
        if n_atoms < 20000:
            point_size = 12.0
        elif n_atoms < 100000:
            point_size = 7.0
        else:
            point_size = 4.0
    
        use_sphere_glyphs = n_atoms <= 25000

        self.viewer.set_atoms(
            points,
            rgb,
            visible=True,
            point_size=point_size,
            render_as_spheres=True,
            reset_camera=True,
            use_sphere_glyphs=use_sphere_glyphs,
            sphere_radius=0.42,
            sphere_theta_resolution=24,
            sphere_phi_resolution=18,
            show_silhouette=True,
        )
    
        # Draw the unit cell.
        try:
            corners = self._unit_cell_corners(structure)
    
            self.viewer.set_unit_cell(
                corners,
                visible=True,
                color="black",
                line_width=2.0,
            )
        except Exception:
            pass
        
        # Remove labels placed inside the structure.
        try:
            self.viewer.hide("labels")
            self.viewer.hide("element_labels")
        except Exception:
            pass
        
        # Build one legend entry for each unique element type.
        try:
            element_to_color = {}
    
            for element in sorted(set(elements)):
                element = clean_el_symbol(element)
    
                if not element:
                    continue
                
                element_to_color[element] = _ATOM_COLORS.get(
                    element,
                    "#808080",
                )
    
            self.viewer.set_element_legend(
                element_to_color,
                visible=True,
            )
    
        except Exception:
            try:
                self.viewer.set_element_legend(
                    {},
                    visible=False,
                )
            except Exception:
                pass
            
        try:
            self.viewer.render()
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            if hasattr(self.viewer, "close_viewer"):
                self.viewer.close_viewer()
        except Exception:
            pass

        super().closeEvent(event)