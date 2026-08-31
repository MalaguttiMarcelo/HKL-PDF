# pdf_fitting/gui/crystallite_shape_dialog.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from .fast_shape_viewer import FastShapeViewer

from pymatgen.core.structure import Structure

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
    lattice_matrix_from_params,
    build_orientation_basis,
    shape_mask,
    generate_translation_grid,
    iter_shape_scan,
)


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
}


def _clean_element_symbol(x: Any) -> str:
    import re

    s = str(x).strip()
    m = re.match(r"([A-Za-z]{1,2})", s)

    if not m:
        return s

    sym = m.group(1)
    return sym.upper() if len(sym) == 1 else sym[0].upper() + sym[1:].lower()


def _color_for_element(el: str) -> str:
    el = _clean_element_symbol(el)
    return _ATOM_COLORS.get(el, "#808080")

def _hex_to_rgb_u8(hex_color: str) -> tuple[int, int, int]:
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

def _rgb_for_element(el: str) -> tuple[int, int, int]:
    return _hex_to_rgb_u8(_color_for_element(el))



class CrystalliteShapeDialog(QtWidgets.QDialog):
    """
    Window to define and preview anisotropic crystallite shapes.

    Visualization defaults to real-space Cartesian coordinates.
    """

    shapeAccepted = QtCore.Signal(dict)

    def __init__(
        self,
        parent=None,
        *,
        initial_spec: Dict[str, Any] | None = None,
        lattice_params: Dict[str, float] | None = None,
        structure: Optional[Structure] = None,
    ):
        super().__init__(parent)

        self.setWindowTitle("Build crystallite shape")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )



        self._structure = structure

        if structure is not None:
            lat = structure.lattice
            self._lattice_params = {
                "a": float(lat.a),
                "b": float(lat.b),
                "c": float(lat.c),
                "alpha": float(lat.alpha),
                "beta": float(lat.beta),
                "gamma": float(lat.gamma),
            }
        else:
            self._lattice_params = lattice_params or {
                "a": 1.0,
                "b": 1.0,
                "c": 1.0,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0,
            }

        self._spec = CrystalliteShapeSpec.from_dict(initial_spec or {})
        self._scan_specs: List[CrystalliteShapeSpec] = []

        # Preview cloud cache.
        # This is only for visualization.
        self._cloud_cache_max_cells = 0
        self._cloud_frac = None
        self._cloud_cart = None
        self._cloud_elements = None
        self._cloud_labels = None

        # Last visible shape atoms, used for XYZ export.
        self._last_visible_cart = np.zeros((0, 3), dtype=np.float32)
        self._last_visible_elements: List[str] = []

        self._build_ui()
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview)
        self._load_spec_to_ui(self._spec)
        self._connect_signals()
        self._update_preview()



    def showEvent(self, event):
        super().showEvent(event)

        if not getattr(self, "_shown_fullscreen_once", False):
            self._shown_fullscreen_once = True
            QtCore.QTimer.singleShot(0, self.showMaximized)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QWidget()
        left.setMaximumWidth(390)
        left_layout = QtWidgets.QVBoxLayout(left)

        # -------------------------
        # Shape type
        # -------------------------
        gb_shape = QtWidgets.QGroupBox("Shape")
        form_shape = QtWidgets.QFormLayout(gb_shape)
        
        self.combo_search_mode = QtWidgets.QComboBox()
        self.combo_search_mode.addItem("Pre-defined crystallite dimensions", "predefined")
        self.combo_search_mode.addItem("Grid search / scan dimensions", "grid_search")
        self.combo_search_mode.setToolTip(
            "Pre-defined: fit one fixed crystallite shape.\n"
            "Grid search: scan diameter/height ranges and compare Rwp."
        )
        
        self.combo_shape = QtWidgets.QComboBox()
        self.combo_shape.addItems([
            "sphere",
            "ellipsoid",
            "cylinder",
            "disk",
            "prism",
        ])
        
        form_shape.addRow("Run mode:", self.combo_search_mode)
        form_shape.addRow("Shape type:", self.combo_shape)
        
        left_layout.addWidget(gb_shape)

        # -------------------------
        # Dimensions
        # -------------------------
        gb_dim = QtWidgets.QGroupBox("Dimensions / scan")
        form_dim = QtWidgets.QFormLayout(gb_dim)

        self.spin_d_current = QtWidgets.QDoubleSpinBox()
        self.spin_h_current = QtWidgets.QDoubleSpinBox()
        self.spin_d_min = QtWidgets.QDoubleSpinBox()
        self.spin_d_max = QtWidgets.QDoubleSpinBox()
        self.spin_h_min = QtWidgets.QDoubleSpinBox()
        self.spin_h_max = QtWidgets.QDoubleSpinBox()
        self.spin_step = QtWidgets.QDoubleSpinBox()

        for sp in (
            self.spin_d_current,
            self.spin_h_current,
            self.spin_d_min,
            self.spin_d_max,
            self.spin_h_min,
            self.spin_h_max,
            self.spin_step,
        ):
            sp.setDecimals(3)
            sp.setRange(0.001, 10000.0)
            sp.setSingleStep(1.0)

        self.chk_scan_diameter = QtWidgets.QCheckBox("Scan/refine diameter")
        self.chk_scan_diameter.setChecked(True)

        self.chk_diameter_infinite = QtWidgets.QCheckBox(
            "If diameter not scanned: treat diameter as infinite"
        )
        self.chk_diameter_infinite.setChecked(False)
        self.chk_diameter_infinite.setToolTip(
            "Future slab/infinite-diameter mode. "
            "The first CPU backend implementation supports finite diameters."
        )

        self.chk_scan_height = QtWidgets.QCheckBox("Scan/refine axis length")
        self.chk_scan_height.setChecked(True)

        form_dim.addRow("Current diameter / cells:", self.spin_d_current)
        form_dim.addRow("Current height / cells:", self.spin_h_current)

        form_dim.addRow("Diameter min / cells:", self.spin_d_min)
        form_dim.addRow("Diameter max / cells:", self.spin_d_max)
        form_dim.addRow("Height min / cells:", self.spin_h_min)
        form_dim.addRow("Height max / cells:", self.spin_h_max)
        form_dim.addRow("Step / cells:", self.spin_step)

        form_dim.addRow(self.chk_scan_diameter)
        form_dim.addRow(self.chk_diameter_infinite)
        form_dim.addRow(self.chk_scan_height)

        left_layout.addWidget(gb_dim)        

        # -------------------------
        # Orientation
        # -------------------------
        gb_orient = QtWidgets.QGroupBox("Axis orientation")
        form_orient = QtWidgets.QFormLayout(gb_orient)

        self.spin_axis_h = QtWidgets.QSpinBox()
        self.spin_axis_k = QtWidgets.QSpinBox()
        self.spin_axis_l = QtWidgets.QSpinBox()

        self.spin_base1_h = QtWidgets.QSpinBox()
        self.spin_base1_k = QtWidgets.QSpinBox()
        self.spin_base1_l = QtWidgets.QSpinBox()

        self.spin_base2_h = QtWidgets.QSpinBox()
        self.spin_base2_k = QtWidgets.QSpinBox()
        self.spin_base2_l = QtWidgets.QSpinBox()

        for sp in (
            self.spin_axis_h,
            self.spin_axis_k,
            self.spin_axis_l,
            self.spin_base1_h,
            self.spin_base1_k,
            self.spin_base1_l,
            self.spin_base2_h,
            self.spin_base2_k,
            self.spin_base2_l,
        ):
            sp.setRange(-20, 20)

        form_orient.addRow("Axis h:", self.spin_axis_h)
        form_orient.addRow("Axis k:", self.spin_axis_k)
        form_orient.addRow("Axis l:", self.spin_axis_l)

        form_orient.addRow("Base 1 h:", self.spin_base1_h)
        form_orient.addRow("Base 1 k:", self.spin_base1_k)
        form_orient.addRow("Base 1 l:", self.spin_base1_l)

        form_orient.addRow("Base 2 h:", self.spin_base2_h)
        form_orient.addRow("Base 2 k:", self.spin_base2_k)
        form_orient.addRow("Base 2 l:", self.spin_base2_l)

        left_layout.addWidget(gb_orient)

        # -------------------------
        # Display options
        # -------------------------
        gb_display = QtWidgets.QGroupBox("Display options")
        v_disp = QtWidgets.QVBoxLayout(gb_display)

        self.chk_real_space = QtWidgets.QCheckBox("Plot in real space")
        self.chk_real_space.setChecked(True)
        self.chk_real_space.setToolTip(
            "Recommended. In real space, non-orthogonal cells such as hexagonal cells "
            "are displayed correctly."
        )

        self.chk_show_atoms = QtWidgets.QCheckBox("Show atoms")
        self.chk_show_atoms.setChecked(True)

        self.chk_show_labels = QtWidgets.QCheckBox("Show atom IDs")
        self.chk_show_labels.setChecked(False)

        self.spin_max_labels = QtWidgets.QSpinBox()
        self.spin_max_labels.setRange(1, 10000)
        self.spin_max_labels.setValue(250)
        self.spin_max_labels.setToolTip(
            "Atom labels are only drawn if the number of visible atoms is below this limit."
        )

        self.chk_show_unit_cell = QtWidgets.QCheckBox("Show first unit cell")
        self.chk_show_unit_cell.setChecked(True)

        self.chk_show_axis = QtWidgets.QCheckBox("Show shape/cylinder axis")
        self.chk_show_axis.setChecked(True)

        self.chk_show_outline = QtWidgets.QCheckBox("Show shape outline")
        self.chk_show_outline.setChecked(True)

        self.chk_show_base_vectors = QtWidgets.QCheckBox("Show base vectors")
        self.chk_show_base_vectors.setChecked(True)

        self.chk_show_element_legend = QtWidgets.QCheckBox("Show element legend")
        self.chk_show_element_legend.setChecked(True)

        v_disp.addWidget(self.chk_real_space)
        v_disp.addWidget(self.chk_show_atoms)

        row_labels = QtWidgets.QHBoxLayout()
        row_labels.addWidget(self.chk_show_labels)
        row_labels.addWidget(QtWidgets.QLabel("max labels:"))
        row_labels.addWidget(self.spin_max_labels)
        v_disp.addLayout(row_labels)

        v_disp.addWidget(self.chk_show_unit_cell)
        v_disp.addWidget(self.chk_show_axis)
        v_disp.addWidget(self.chk_show_base_vectors)
        v_disp.addWidget(self.chk_show_outline)
        v_disp.addWidget(self.chk_show_element_legend)

        left_layout.addWidget(gb_display)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setWordWrap(True)
        left_layout.addWidget(self.lbl_info)

        
        self.btn_save_xyz = QtWidgets.QPushButton("Save current shape as .xyz")
        self.btn_apply = QtWidgets.QPushButton("Apply to input")
        self.btn_close = QtWidgets.QPushButton("Close")

        
        left_layout.addWidget(self.btn_save_xyz)
        left_layout.addStretch(1)
        

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_close)
        left_layout.addLayout(btn_row)

        root.addWidget(left)

        # -------------------------
        # Fast PyVista/VTK viewer
        # -------------------------
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.viewer = FastShapeViewer(self)
        right_layout.addWidget(self.viewer, 1)

        root.addWidget(right, 1)

    def _connect_signals(self) -> None:
        widgets = [
            self.combo_shape,
            self.spin_d_current,
            self.spin_h_current,
            self.spin_d_min,
            self.spin_d_max,
            self.spin_h_min,
            self.spin_h_max,
            self.spin_step,
            self.chk_scan_diameter,
            self.chk_diameter_infinite,
            self.chk_scan_height,
            self.spin_axis_h,
            self.spin_axis_k,
            self.spin_axis_l,
            self.spin_base1_h,
            self.spin_base1_k,
            self.spin_base1_l,
            self.spin_base2_h,
            self.spin_base2_k,
            self.spin_base2_l,
            self.chk_real_space,
            self.chk_show_atoms,
            self.chk_show_labels,
            self.spin_max_labels,
            self.chk_show_unit_cell,
            self.chk_show_axis,
            self.chk_show_outline,
            self.chk_show_base_vectors,
            self.chk_show_element_legend,
            self.combo_search_mode,

        ]

        for w in widgets:
            if isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(lambda _=None: self._schedule_preview_update())
            elif isinstance(w, QtWidgets.QAbstractButton):
                w.toggled.connect(lambda _=None: self._schedule_preview_update())
            else:
                w.valueChanged.connect(lambda _=None: self._schedule_preview_update())

        self.combo_search_mode.currentIndexChanged.connect(
            lambda _=None: self._update_search_mode_controls()
        )

        
        self.btn_save_xyz.clicked.connect(self._save_current_xyz)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_close.clicked.connect(self.close)
        

    def _schedule_preview_update(self) -> None:
        try:
            self._preview_timer.start(120)
        except Exception:
            self._update_preview()

    # ------------------------------------------------------------------
    # Spec handling
    # ------------------------------------------------------------------
    def _load_spec_to_ui(self, spec: CrystalliteShapeSpec) -> None:
        search_mode = str(getattr(spec, "search_mode", "grid_search")).strip().lower()

        idx_mode = self.combo_search_mode.findData(search_mode)

        if idx_mode >= 0:
            self.combo_search_mode.setCurrentIndex(idx_mode)
        else:
            idx_default = self.combo_search_mode.findData("grid_search")
            if idx_default >= 0:
                self.combo_search_mode.setCurrentIndex(idx_default)

        idx = self.combo_shape.findText(str(spec.shape_type))
        if idx >= 0:
            self.combo_shape.setCurrentIndex(idx)

        self.spin_d_current.setValue(float(spec.diameter_cells))
        self.spin_h_current.setValue(float(spec.height_cells))

        self.spin_d_min.setValue(float(spec.diameter_min_cells))
        self.spin_d_max.setValue(float(spec.diameter_max_cells))
        self.spin_h_min.setValue(float(spec.height_min_cells))
        self.spin_h_max.setValue(float(spec.height_max_cells))
        self.spin_step.setValue(float(spec.step_cells))

        self.spin_axis_h.setValue(int(spec.axis_h))
        self.spin_axis_k.setValue(int(spec.axis_k))
        self.spin_axis_l.setValue(int(spec.axis_l))

        self.spin_base1_h.setValue(int(spec.base1_h))
        self.spin_base1_k.setValue(int(spec.base1_k))
        self.spin_base1_l.setValue(int(spec.base1_l))

        self.spin_base2_h.setValue(int(spec.base2_h))
        self.spin_base2_k.setValue(int(spec.base2_k))
        self.spin_base2_l.setValue(int(spec.base2_l))

        self.chk_scan_diameter.setChecked(bool(getattr(spec, "scan_diameter", True)))
        self.chk_scan_height.setChecked(bool(getattr(spec, "scan_height", True)))
        self.chk_diameter_infinite.setChecked(bool(getattr(spec, "diameter_infinite", False)))

        self._update_search_mode_controls()


    def _spec_from_ui(self) -> CrystalliteShapeSpec:
        return CrystalliteShapeSpec(
            shape_type=str(self.combo_shape.currentText()).strip(),
            search_mode=str(self.combo_search_mode.currentData() or "grid_search"),

            diameter_cells=float(self.spin_d_current.value()),
            height_cells=float(self.spin_h_current.value()),

            diameter_min_cells=float(self.spin_d_min.value()),
            diameter_max_cells=float(self.spin_d_max.value()),
            height_min_cells=float(self.spin_h_min.value()),
            height_max_cells=float(self.spin_h_max.value()),
            step_cells=float(self.spin_step.value()),

            scan_diameter=bool(self.chk_scan_diameter.isChecked()),
            scan_height=bool(self.chk_scan_height.isChecked()),
            diameter_infinite=bool(self.chk_diameter_infinite.isChecked()),

            axis_h=int(self.spin_axis_h.value()),
            axis_k=int(self.spin_axis_k.value()),
            axis_l=int(self.spin_axis_l.value()),

            base1_h=int(self.spin_base1_h.value()),
            base1_k=int(self.spin_base1_k.value()),
            base1_l=int(self.spin_base1_l.value()),

            base2_h=int(self.spin_base2_h.value()),
            base2_k=int(self.spin_base2_k.value()),
            base2_l=int(self.spin_base2_l.value()),
        )
    
    # ------------------------------------------------------------------
    # Atom generation
    # ------------------------------------------------------------------
    def _lattice_matrix(self) -> np.ndarray:
        if self._structure is not None:
            return np.asarray(self._structure.lattice.matrix, dtype=float)

        return lattice_matrix_from_params(
            self._lattice_params.get("a", 1.0),
            self._lattice_params.get("b", 1.0),
            self._lattice_params.get("c", 1.0),
            self._lattice_params.get("alpha", 90.0),
            self._lattice_params.get("beta", 90.0),
            self._lattice_params.get("gamma", 90.0),
        )

    def _ensure_preview_cloud(self, max_cells: int) -> None:
        """
        Build a reusable atom cloud large enough for the requested shape.

        The shape mask is applied later. This avoids rebuilding the full supercell
        whenever only graphical settings change.
        """
        max_cells = int(max(1, max_cells))

        if (
            self._cloud_cart is not None
            and self._cloud_frac is not None
            and self._cloud_cache_max_cells >= max_cells
        ):
            return

        M = self._lattice_matrix()
        translations = generate_translation_grid(max_cells)

        if self._structure is None:
            frac_all = translations.astype(np.float32)
            cart_all = (frac_all @ M).astype(np.float32)

            elements = np.array(["X"] * frac_all.shape[0], dtype=object)
            labels = np.array([str(i) for i in range(frac_all.shape[0])], dtype=object)

        else:
            sites = list(self._structure.sites)

            site_frac = np.asarray(
                [np.asarray(site.frac_coords, dtype=float) for site in sites],
                dtype=np.float32,
            )

            site_elements = np.asarray(
                [_clean_element_symbol(str(site.specie)) for site in sites],
                dtype=object,
            )

            n_trans = translations.shape[0]
            n_sites = site_frac.shape[0]

            frac_all = (
                translations[:, None, :].astype(np.float32)
                + site_frac[None, :, :]
            ).reshape(-1, 3)

            site_index_all = np.tile(np.arange(n_sites, dtype=np.int32), n_trans)

            cart_all = (frac_all @ M).astype(np.float32)

            elements = site_elements[site_index_all]

            labels = np.asarray(
                [f"{int(i)}:{site_elements[int(i)]}" for i in site_index_all],
                dtype=object,
            )

        self._cloud_cache_max_cells = max_cells
        self._cloud_frac = frac_all
        self._cloud_cart = cart_all
        self._cloud_elements = elements
        self._cloud_labels = labels


    def _build_preview_atoms(self, spec: CrystalliteShapeSpec) -> Dict[str, Any]:
        """
        Apply the current shape mask to the cached atom cloud.
        """
        M = self._lattice_matrix()

        max_cells = int(
            np.ceil(max(float(spec.diameter_cells), float(spec.height_cells), 1.0))
        ) + 2

        self._ensure_preview_cloud(max_cells)

        if self._cloud_cart is None:
            return {
                "frac": np.zeros((0, 3), dtype=float),
                "cart": np.zeros((0, 3), dtype=float),
                "elements": [],
                "labels": [],
                "n_atoms": 0,
            }

        mask = shape_mask(self._cloud_cart, spec, M)

        frac = self._cloud_frac[mask]
        cart = self._cloud_cart[mask]
        elements = self._cloud_elements[mask].tolist()
        labels = self._cloud_labels[mask].tolist()

        return {
            "frac": frac,
            "cart": cart,
            "elements": elements,
            "labels": labels,
            "n_atoms": int(cart.shape[0]),
        }
    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _shape_lengths_angstrom(self, spec: CrystalliteShapeSpec, M: np.ndarray) -> Tuple[float, float]:
        a_len = np.linalg.norm(M[0])
        b_len = np.linalg.norm(M[1])
        c_len = np.linalg.norm(M[2])

        mean_ab = 0.5 * (a_len + b_len)

        D_ang = float(spec.diameter_cells) * mean_ab
        H_ang = float(spec.height_cells) * c_len

        return D_ang, H_ang




    # ------------------------------------------------------------------
    # Main preview
    # ------------------------------------------------------------------
    def _unit_cell_corners(self, M: np.ndarray, use_real: bool = True) -> np.ndarray:
        a = M[0]
        b = M[1]
        c = M[2]
        O = np.zeros(3)

        corners_cart = np.array(
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
            dtype=float,
        )

        if use_real:
            return corners_cart.astype(np.float32)

        return (corners_cart @ np.linalg.inv(M)).astype(np.float32)


    def _shape_axis_points(self, spec: CrystalliteShapeSpec, M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        e1, e2, e3 = build_orientation_basis(
            M,
            axis_hkl=(spec.axis_h, spec.axis_k, spec.axis_l),
            base1_hkl=(spec.base1_h, spec.base1_k, spec.base1_l),
            base2_hkl=(spec.base2_h, spec.base2_k, spec.base2_l),
        )

        D_ang, H_ang = self._shape_lengths_angstrom(spec, M)
        L = 0.65 * max(D_ang, H_ang, 1.0)

        p0 = -L * e3
        p1 = +L * e3

        return p0, p1, e3


    def _update_preview(self) -> None:
        spec = self._spec_from_ui()
        self._spec = spec

        # For PyVista, real-space should be the main/accurate visualization.
        use_real = bool(self.chk_real_space.isChecked())
        M = self._lattice_matrix()

        try:
            data = self._build_preview_atoms(spec)

            frac = np.asarray(data["frac"], dtype=np.float32)
            cart = np.asarray(data["cart"], dtype=np.float32)
            elements = list(data["elements"])
            labels = list(data["labels"])

            coords = cart if use_real else frac

            n_atoms = coords.shape[0]

            # Store real-space coordinates for XYZ export.
            # XYZ should always be Cartesian Å, not fractional.
            self._last_visible_cart = cart.copy()
            self._last_visible_elements = list(elements)

            # Atom colors
            rgb = np.asarray(
                [_rgb_for_element(el) for el in elements],
                dtype=np.uint8,
            ) if n_atoms > 0 else None

            # Use the same spherical GPU rendering as StructureViewer.
            render_as_spheres = True
            
            if n_atoms < 20000:
                point_size = 12.0
            elif n_atoms < 100000:
                point_size = 7.0
            else:
                point_size = 4.0

            self.viewer.set_atoms(
                coords,
                rgb,
                visible=bool(self.chk_show_atoms.isChecked()),
                point_size=point_size,
                render_as_spheres=render_as_spheres,
                reset_camera=not getattr(self.viewer, "_camera_initialized", False),
            )

            # Element legend in top-right corner
            if elements:
                element_to_color = {
                    el: _color_for_element(el)
                    for el in sorted(set(elements))
                }
            else:
                element_to_color = {}

            try:
                self.viewer.set_element_legend(
                    element_to_color,
                    visible=bool(self.chk_show_element_legend.isChecked()),
                )
            except Exception:
                import traceback
                traceback.print_exc()

            # Unit cell
            if self.chk_show_unit_cell.isChecked():
                corners = self._unit_cell_corners(M, use_real=use_real)
                self.viewer.set_unit_cell(
                    corners,
                    visible=True,
                    color="black",
                    line_width=2.0,
                )
            else:
                self.viewer.set_unit_cell(
                    np.zeros((8, 3), dtype=np.float32),
                    visible=False,
                )

            # Axis and base vectors
            e1, e2, e3 = build_orientation_basis(
                M,
                axis_hkl=(spec.axis_h, spec.axis_k, spec.axis_l),
                base1_hkl=(spec.base1_h, spec.base1_k, spec.base1_l),
                base2_hkl=(spec.base2_h, spec.base2_k, spec.base2_l),
            )

            D_ang, H_ang = self._shape_lengths_angstrom(spec, M)

            # Shape radius in the base plane is D_ang / 2.
            # Shape half-height along the axis is H_ang / 2.
            # Therefore lengths larger than 0.5*D or 0.5*H go outside the shape.
            #
            # For thin disks, H_ang can be very small, so we also enforce a minimum
            # visible axis length based on D_ang.
            axis_len = max(
                0.65 * H_ang,     # goes beyond top/bottom for cylinders
                0.35 * D_ang,     # keeps axis visible for thin disks
                1.0,
            )

            base_len = max(
                0.65 * D_ang,     # goes beyond cylinder/disk radius
                1.0,
            )

            origin = np.zeros(3, dtype=float)

            axis_end = axis_len * e3
            base1_end = base_len * e1
            base2_end = base_len * e2

            if not use_real:
                invM = np.linalg.inv(M)
                origin_plot = origin @ invM
                axis_end_plot = axis_end @ invM
                base1_end_plot = base1_end @ invM
                base2_end_plot = base2_end @ invM
            else:
                origin_plot = origin
                axis_end_plot = axis_end
                base1_end_plot = base1_end
                base2_end_plot = base2_end

            self.viewer.set_vector(
                "axis_arrow",
                origin_plot,
                axis_end_plot,
                visible=bool(self.chk_show_axis.isChecked()),
                color="red",
                label=f"axis [{spec.axis_h} {spec.axis_k} {spec.axis_l}]",
                line_width=5.0,
                label_size=12,
            )

            self.viewer.set_vector(
                "base1_arrow",
                origin_plot,
                base1_end_plot,
                visible=bool(self.chk_show_base_vectors.isChecked()),
                color="green",
                label=f"base1 [{spec.base1_h} {spec.base1_k} {spec.base1_l}]",
                line_width=5.0,
                label_size=12,
            )

            self.viewer.set_vector(
                "base2_arrow",
                origin_plot,
                base2_end_plot,
                visible=bool(self.chk_show_base_vectors.isChecked()),
                color="purple",
                label=f"base2 [{spec.base2_h} {spec.base2_k} {spec.base2_l}]",
                line_width=5.0,
                label_size=12,
            )

            # Shape outline
            shape_type = str(spec.shape_type).strip().lower()

            # For non-real-space mode, the transparent surface is less meaningful.
            # Keep outline most accurate in real-space.
            if use_real and self.chk_show_outline.isChecked():
                if shape_type in ("cylinder", "disk", "disk-like", "disk_like"):
                    self.viewer.set_cylinder_outline(
                        center=np.zeros(3),
                        axis=e3,
                        radius=0.5 * D_ang,
                        height=H_ang,
                        visible=True,
                        color="dodgerblue",
                        opacity=0.18,
                    )

                elif shape_type in ("sphere", "spherical"):
                    self.viewer.set_sphere_outline(
                        center=np.zeros(3),
                        radius=0.5 * D_ang,
                        visible=True,
                        color="dodgerblue",
                        opacity=0.16,
                    )

                elif shape_type in ("prism", "rectangular_prism"):
                    # Simple axis-aligned fallback. Later we can implement fully oriented prism mesh.
                    half = 0.5 * D_ang
                    half_h = 0.5 * H_ang
                    self.viewer.set_box_outline(
                        (-half, half, -half, half, -half_h, half_h),
                        visible=True,
                        color="dodgerblue",
                        opacity=0.12,
                    )

                else:
                    self.viewer.set_cylinder_outline(
                        center=np.zeros(3),
                        axis=e3,
                        radius=0.5 * D_ang,
                        height=H_ang,
                        visible=True,
                        color="dodgerblue",
                        opacity=0.18,
                    )
            else:
                self.viewer.hide("shape_outline")

            # Labels
            show_labels = (
                bool(self.chk_show_labels.isChecked())
                and n_atoms <= int(self.spin_max_labels.value())
            )

            self.viewer.set_labels(
                coords,
                labels,
                visible=show_labels,
                font_size=10,
            )

            coord_label = "real space / Å" if use_real else "fractional-cell coordinates"

            self.lbl_info.setText(
                f"Visible atoms/points: {n_atoms}\n"
                f"Coordinates: {coord_label}\n"
                f"Shape origin: unit-cell origin, Cartesian [0, 0, 0]\n"
                f"Rendering: {'spheres' if render_as_spheres else 'GPU points'}\n"
                f"Axis = [{spec.axis_h} {spec.axis_k} {spec.axis_l}]\n"
                f"Base1 = [{spec.base1_h} {spec.base1_k} {spec.base1_l}], "
                f"Base2 = [{spec.base2_h} {spec.base2_k} {spec.base2_l}]"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.lbl_info.setText(f"Preview failed: {e}")


    def _update_search_mode_controls(self) -> None:
        mode = str(self.combo_search_mode.currentData() or "grid_search").strip().lower()
        predefined = mode == "predefined"

        # Current values are always useful.
        self.spin_d_current.setEnabled(True)
        self.spin_h_current.setEnabled(True)

        # Scan controls only needed for grid search.
        for w in (
            self.spin_d_min,
            self.spin_d_max,
            self.spin_h_min,
            self.spin_h_max,
            self.spin_step,
            self.chk_scan_diameter,
            self.chk_scan_height,
            self.chk_diameter_infinite,
        ):
            try:
                w.setEnabled(not predefined)
            except Exception:
                pass

        if predefined:
            self.chk_scan_diameter.setChecked(False)
            self.chk_scan_height.setChecked(False)
            self.chk_diameter_infinite.setChecked(False)



    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------
    def _on_start_scan(self) -> None:
        spec = self._spec_from_ui()
        self._scan_specs = list(iter_shape_scan(spec))

        self.lbl_info.setText(
            f"Generated {len(self._scan_specs)} shape candidates.\n"
            f"Diameter: {spec.diameter_min_cells:g} to {spec.diameter_max_cells:g} cells\n"
            f"Height: {spec.height_min_cells:g} to {spec.height_max_cells:g} cells\n"
            f"Step: {spec.step_cells:g} cells"
        )

        if self._scan_specs:
            first = self._scan_specs[0]
            self.spin_d_current.setValue(first.diameter_cells)
            self.spin_h_current.setValue(first.height_cells)
            self._update_preview()


    def _save_current_xyz(self) -> None:
        """
        Save currently visible crystallite atoms as an XYZ file.

        Coordinates are always Cartesian Å.
        """
        try:
            cart = np.asarray(getattr(self, "_last_visible_cart", np.zeros((0, 3))), dtype=float)
            elements = list(getattr(self, "_last_visible_elements", []) or [])

            if cart.size == 0 or len(elements) == 0:
                QtWidgets.QMessageBox.information(
                    self,
                    "No atoms",
                    "No atoms are currently visible in the crystallite shape.",
                )
                return

            if cart.shape[0] != len(elements):
                QtWidgets.QMessageBox.warning(
                    self,
                    "XYZ save failed",
                    "Internal atom coordinate/element mismatch.",
                )
                return

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save crystallite shape as XYZ",
                "crystallite_shape.xyz",
                "XYZ files (*.xyz);;All files (*)",
            )

            if not path:
                return

            spec = self._spec_from_ui()

            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{cart.shape[0]}\n")
                f.write(
                    "Crystallite shape "
                    f"type={spec.shape_type}, "
                    f"D_cells={spec.diameter_cells}, "
                    f"H_cells={spec.height_cells}, "
                    f"axis=[{spec.axis_h} {spec.axis_k} {spec.axis_l}]\n"
                )

                for el, xyz in zip(elements, cart):
                    x, y, z = [float(v) for v in xyz]
                    f.write(f"{el:2s}  {x: .8f}  {y: .8f}  {z: .8f}\n")

            QtWidgets.QMessageBox.information(
                self,
                "XYZ saved",
                f"Saved {cart.shape[0]} atoms to:\n{path}",
            )

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "XYZ save failed",
                str(e),
            )

    def _on_apply(self) -> None:
        spec = self._spec_from_ui()
        d = spec.to_dict()

        d["mode"] = "finite_shape"

        search_mode = str(self.combo_search_mode.currentData() or "grid_search").strip().lower()
        d["search_mode"] = search_mode

        if search_mode == "predefined":
            # One fixed crystallite shape.
            d["scan_diameter"] = False
            d["scan_height"] = False
            d["diameter_infinite"] = False

            # Keep current fixed dimensions.
            d["diameter_cells"] = float(self.spin_d_current.value())
            d["height_cells"] = float(self.spin_h_current.value())

            d["n_scan_candidates"] = 1

        else:
            # Normal grid search.
            d["scan_diameter"] = bool(self.chk_scan_diameter.isChecked())
            d["scan_height"] = bool(self.chk_scan_height.isChecked())
            d["diameter_infinite"] = bool(self.chk_diameter_infinite.isChecked())

            try:
                d["n_scan_candidates"] = len(list(iter_shape_scan(spec)))
            except Exception:
                d["n_scan_candidates"] = 0

        d["preview_coordinates"] = (
            "real_space" if self.chk_real_space.isChecked() else "fractional"
        )

        self.shapeAccepted.emit(d)
        self.accept()

    def _cleanup_vtk(self) -> None:
        try:
            if hasattr(self, "_preview_timer") and self._preview_timer is not None:
                self._preview_timer.stop()
        except Exception:
            pass

        try:
            if hasattr(self, "viewer") and self.viewer is not None:
                if hasattr(self.viewer, "close_viewer"):
                    self.viewer.close_viewer()
        except Exception:
            pass


    def done(self, result: int) -> None:
        self._cleanup_vtk()
        super().done(result)

    def closeEvent(self, event):
        self._cleanup_vtk()
        super().closeEvent(event)