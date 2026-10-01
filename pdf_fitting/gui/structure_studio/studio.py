from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

from pymatgen.core import Structure

from .chemistry import (
    clean_element_symbol,
    covalent_radius,
    display_radius,
    element_color,
)

from .display_model import (
    base_site_index,
    build_display_model,
    visible_periodic_structure,
)


class StructureStudioWindow(QtWidgets.QMainWindow):
    """
    GPU structure viewer for the PDF-refinement application.

    Features
    --------
    - PyVista/VTK GPU rendering
    - element and site styles
    - periodic bond generation
    - VESTA-style two-color bonds
    - atom selection
    - coordination polyhedra
    - crystallographic camera views
    """

    closed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor
        except Exception as exc:
            raise RuntimeError(
                "The structure viewer requires PyVista, VTK and PyVistaQt.\n\n"
                "Install them with:\n"
                "pip install pyvista vtk pyvistaqt qtpy"
            ) from exc

        self.pv = pv
        self.QtInteractor = QtInteractor

        self.setWindowTitle(
            "GPU Crystal Structure Viewer"
        )

        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowType.Window
            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
            | QtCore.Qt.WindowType.WindowMaximizeButtonHint
            | QtCore.Qt.WindowType.WindowCloseButtonHint
        )

        self.resize(
            1500,
            920,
        )

        self.structure: Optional[Structure] = None
        self.display_structure: Optional[Structure] = None

        self.structure_path: str = ""
        self.input_path: str = ""
        self.configuration_path: str = ""

        self.display_atom_records: List[Dict[str, Any]] = []
        self.display_atom_lookup: Dict[Any, Dict[str, Any]] = {}

        self.polyhedra_requested = False
        self._loading_configuration = False
        self._pending_camera_state = None

        self.element_styles: Dict[str, Dict[str, Any]] = {}
        self.site_styles: Dict[int, Dict[str, Any]] = {}

        self.atom_records: List[Dict[str, Any]] = []
        self.bond_records: List[Dict[str, Any]] = []

        self.atom_actors: List[Any] = []
        self.bond_actors: List[Any] = []
        self.cell_actors: List[Any] = []
        self.label_actors: List[Any] = []
        self.polyhedron_actors: List[Any] = []
        self.highlight_actors: List[Any] = []

        self.selected_site_index: Optional[int] = None
        self.selected_atom_record: Optional[Dict[str, Any]] = None
        self._updating_style_tree = False
        self._camera_initialized = False

        self._build_ui()
        self._configure_renderer()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)

        self.splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal
        )

        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)

        root_layout.addWidget(
            self.splitter,
            1,
        )

        self.controls = QtWidgets.QTabWidget()
        self.controls.setMinimumWidth(380)

        self.splitter.addWidget(
            self.controls
        )

        self.plotter = self.QtInteractor(self)

        self.splitter.addWidget(
            self.plotter.interactor
        )

        self.splitter.setSizes(
            [
                460,
                1040,
            ]
        )

        self.tab_general = QtWidgets.QWidget()
        self.tab_atoms = QtWidgets.QWidget()
        self.tab_bonds = QtWidgets.QWidget()
        self.tab_polyhedra = QtWidgets.QWidget()

        self.controls.addTab(
            self.tab_general,
            "General",
        )

        self.controls.addTab(
            self.tab_atoms,
            "Atoms",
        )

        self.controls.addTab(
            self.tab_bonds,
            "Bonds",
        )

        self.controls.addTab(
            self.tab_polyhedra,
            "Polyhedra",
        )

        self._build_general_tab()
        self._build_atoms_tab()
        self._build_bonds_tab()
        self._build_polyhedra_tab()

        self.statusBar().showMessage(
            "No structure loaded."
        )

    def _scroll_page(self, page):
        outer_layout = QtWidgets.QVBoxLayout(page)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)

        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        return layout

    def _build_general_tab(self) -> None:
        layout = self._scroll_page(
            self.tab_general
        )

        display_group = QtWidgets.QGroupBox(
            "Display"
        )

        display_layout = QtWidgets.QFormLayout(
            display_group
        )

        self.chk_show_atoms = QtWidgets.QCheckBox(
            "Show atoms"
        )
        self.chk_show_atoms.setChecked(True)

        self.chk_show_cell = QtWidgets.QCheckBox(
            "Show unit cell"
        )
        self.chk_show_cell.setChecked(True)

        self.chk_show_axes = QtWidgets.QCheckBox(
            "Show lattice vectors a, b, c"
        )

        self.chk_show_axes.setToolTip(
            "Show the actual crystallographic direct-lattice vectors. "
            "These are not Cartesian x, y, z axes."
        )

        self.chk_show_axes.setChecked(True)

        self.chk_show_labels = QtWidgets.QCheckBox(
            "Show selected site labels"
        )
        self.chk_show_labels.setChecked(True)

        self.chk_orthographic = QtWidgets.QCheckBox(
            "Orthographic projection"
        )

        display_layout.addRow(
            self.chk_show_atoms
        )
        display_layout.addRow(
            self.chk_show_cell
        )
        display_layout.addRow(
            self.chk_show_axes
        )
        display_layout.addRow(
            self.chk_show_labels
        )
        display_layout.addRow(
            self.chk_orthographic
        )

        self.atom_scale_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )

        self.atom_scale_slider.setRange(
            20,
            300,
        )
        self.atom_scale_slider.setValue(
            100
        )

        self.atom_scale_label = QtWidgets.QLabel(
            "1.00"
        )

        atom_size_row = QtWidgets.QHBoxLayout()
        atom_size_row.addWidget(
            self.atom_scale_slider,
            1,
        )
        atom_size_row.addWidget(
            self.atom_scale_label
        )

        display_layout.addRow(
            "Global atom scale:",
            atom_size_row,
        )

        background_button = QtWidgets.QPushButton(
            "Choose background color..."
        )

        display_layout.addRow(
            background_button
        )

        layout.addWidget(
            display_group
        )

        supercell_group = QtWidgets.QGroupBox(
            "Displayed supercell"
        )

        supercell_layout = QtWidgets.QFormLayout(
            supercell_group
        )

        self.spin_repeat_a = QtWidgets.QSpinBox()
        self.spin_repeat_b = QtWidgets.QSpinBox()
        self.spin_repeat_c = QtWidgets.QSpinBox()

        for spinbox in (
            self.spin_repeat_a,
            self.spin_repeat_b,
            self.spin_repeat_c,
        ):
            spinbox.setRange(
                1,
                20,
            )

            spinbox.setValue(
                1
            )

        self.chk_boundary_images = QtWidgets.QCheckBox(
            "Show periodic boundary images"
        )

        self.chk_boundary_images.setChecked(
            False
        )

        self.chk_boundary_images.setToolTip(
            "Show periodic atoms required at the outer supercell boundaries. "
            "Periodic bonds crossing the boundary are also displayed."
        )

        repeat_row = QtWidgets.QHBoxLayout()

        repeat_row.addWidget(
            QtWidgets.QLabel("a")
        )
        repeat_row.addWidget(
            self.spin_repeat_a
        )

        repeat_row.addWidget(
            QtWidgets.QLabel("b")
        )
        repeat_row.addWidget(
            self.spin_repeat_b
        )

        repeat_row.addWidget(
            QtWidgets.QLabel("c")
        )
        repeat_row.addWidget(
            self.spin_repeat_c
        )

        supercell_layout.addRow(
            "Repeat counts:",
            repeat_row,
        )

        supercell_layout.addRow(
            self.chk_boundary_images
        )

        layout.addWidget(
            supercell_group
        )

        camera_group = QtWidgets.QGroupBox(
            "Camera"
        )

        camera_layout = QtWidgets.QGridLayout(
            camera_group
        )

        button_fit = QtWidgets.QPushButton(
            "Fit scene"
        )
        button_iso = QtWidgets.QPushButton(
            "Isometric"
        )
        button_x = QtWidgets.QPushButton(
            "View X"
        )
        button_y = QtWidgets.QPushButton(
            "View Y"
        )
        button_z = QtWidgets.QPushButton(
            "View Z"
        )

        camera_layout.addWidget(
            button_fit,
            0,
            0,
            1,
            3,
        )
        camera_layout.addWidget(
            button_x,
            1,
            0,
        )
        camera_layout.addWidget(
            button_y,
            1,
            1,
        )
        camera_layout.addWidget(
            button_z,
            1,
            2,
        )
        camera_layout.addWidget(
            button_iso,
            2,
            0,
            1,
            3,
        )

        layout.addWidget(
            camera_group
        )

        hkl_group = QtWidgets.QGroupBox(
            "View along (hkl) plane normal"
        )

        hkl_layout = QtWidgets.QGridLayout(
            hkl_group
        )

        self.spin_h = QtWidgets.QSpinBox()
        self.spin_k = QtWidgets.QSpinBox()
        self.spin_l = QtWidgets.QSpinBox()

        for spinbox in (
            self.spin_h,
            self.spin_k,
            self.spin_l,
        ):
            spinbox.setRange(
                -99,
                99,
            )

        self.spin_l.setValue(
            1
        )

        self.chk_reverse_hkl = QtWidgets.QCheckBox(
            "Reverse direction"
        )

        button_apply_hkl = QtWidgets.QPushButton(
            "Apply (hkl) view"
        )

        hkl_layout.addWidget(
            QtWidgets.QLabel("h"),
            0,
            0,
        )
        hkl_layout.addWidget(
            self.spin_h,
            0,
            1,
        )
        hkl_layout.addWidget(
            QtWidgets.QLabel("k"),
            0,
            2,
        )
        hkl_layout.addWidget(
            self.spin_k,
            0,
            3,
        )
        hkl_layout.addWidget(
            QtWidgets.QLabel("l"),
            0,
            4,
        )
        hkl_layout.addWidget(
            self.spin_l,
            0,
            5,
        )
        hkl_layout.addWidget(
            self.chk_reverse_hkl,
            1,
            0,
            1,
            6,
        )
        hkl_layout.addWidget(
            button_apply_hkl,
            2,
            0,
            1,
            6,
        )

        layout.addWidget(
            hkl_group
        )

        selection_group = QtWidgets.QGroupBox(
            "Selected atom"
        )

        selection_layout = QtWidgets.QVBoxLayout(
            selection_group
        )

        self.selection_label = QtWidgets.QLabel(
            "Click an atom to inspect it."
        )

        self.selection_label.setWordWrap(True)
        self.selection_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.selection_label.setStyleSheet(
            """
            QLabel {
                background-color: white;
                border: 1px solid #B0B0B0;
                border-radius: 3px;
                padding: 7px;
            }
            """
        )

        button_clear_selection = QtWidgets.QPushButton(
            "Clear selected atom"
        )

        selection_layout.addWidget(
            self.selection_label
        )
        selection_layout.addWidget(
            button_clear_selection
        )

        layout.addWidget(
            selection_group
        )

        output_group = QtWidgets.QGroupBox(
            "Configuration and export"
        )

        output_layout = QtWidgets.QVBoxLayout(
            output_group
        )

        self.button_save_configuration = QtWidgets.QPushButton(
            "Save viewer configuration"
        )

        self.button_load_configuration = QtWidgets.QPushButton(
            "Load viewer configuration..."
        )

        self.configuration_path_label = QtWidgets.QLabel(
            "Viewer configuration: not assigned"
        )

        self.configuration_path_label.setWordWrap(
            True
        )

        button_export_cif = QtWidgets.QPushButton(
            "Export displayed supercell as CIF..."
        )

        button_export_xyz = QtWidgets.QPushButton(
            "Export displayed atoms as XYZ..."
        )

        button_export_coordination = QtWidgets.QPushButton(
            "Export selected coordination environment..."
        )

        button_screenshot = QtWidgets.QPushButton(
            "Save screenshot..."
        )

        button_gpu_info = QtWidgets.QPushButton(
            "Renderer / GPU information"
        )

        output_layout.addWidget(
            self.button_save_configuration
        )

        output_layout.addWidget(
            self.button_load_configuration
        )

        output_layout.addWidget(
            self.configuration_path_label
        )

        output_layout.addSpacing(
            8
        )

        output_layout.addWidget(
            button_export_cif
        )

        output_layout.addWidget(
            button_export_xyz
        )

        output_layout.addWidget(
            button_export_coordination
        )

        output_layout.addWidget(
            button_screenshot
        )

        output_layout.addWidget(
            button_gpu_info
        )

        layout.addWidget(
            output_group
        )

        layout.addStretch(1)

        self.chk_show_atoms.toggled.connect(
            self._rebuild_scene
        )
        self.chk_show_cell.toggled.connect(
            self._rebuild_scene
        )
        self.chk_show_labels.toggled.connect(
            self._rebuild_scene
        )
        self.chk_show_axes.toggled.connect(
            self._rebuild_scene
        )
        self.chk_orthographic.toggled.connect(
            self._set_orthographic
        )

        self.atom_scale_slider.valueChanged.connect(
            self._atom_scale_changed
        )

        background_button.clicked.connect(
            self._choose_background
        )

        button_fit.clicked.connect(
            self._fit_scene
        )
        button_iso.clicked.connect(
            self._view_isometric
        )
        button_x.clicked.connect(
            self.plotter.view_yz
        )
        button_y.clicked.connect(
            self.plotter.view_xz
        )
        button_z.clicked.connect(
            self.plotter.view_xy
        )

        button_apply_hkl.clicked.connect(
            self._apply_hkl_view
        )

        button_clear_selection.clicked.connect(
            self.clear_selection
        )

        self.button_save_configuration.clicked.connect(
            self.save_viewer_configuration
        )

        self.button_load_configuration.clicked.connect(
            self.load_viewer_configuration_dialog
        )

        button_export_cif.clicked.connect(
            self.export_displayed_cif
        )

        button_export_xyz.clicked.connect(
            self.export_displayed_xyz
        )

        button_export_coordination.clicked.connect(
            self.export_selected_coordination_environment
        )

        button_screenshot.clicked.connect(
            self._save_screenshot
        )

        button_gpu_info.clicked.connect(
            self._show_gpu_information
        )

        self.spin_repeat_a.valueChanged.connect(
            self._on_supercell_changed
        )

        self.spin_repeat_b.valueChanged.connect(
            self._on_supercell_changed
        )

        self.spin_repeat_c.valueChanged.connect(
            self._on_supercell_changed
        )

        self.chk_boundary_images.toggled.connect(
            self._on_supercell_changed
        )

    def _build_atoms_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(
            self.tab_atoms
        )

        information = QtWidgets.QLabel(
            "Edit element or individual-site display styles. "
            "Site styles override element styles."
        )

        information.setWordWrap(True)

        layout.addWidget(
            information
        )

        self.style_tree = QtWidgets.QTreeWidget()
        self.style_tree.setColumnCount(6)

        self.style_tree.setHeaderLabels(
            [
                "Element / site",
                "Radius (Å)",
                "Color",
                "Visible",
                "Label",
                "Opacity",
            ]
        )

        self.style_tree.setAlternatingRowColors(True)
        self.style_tree.setRootIsDecorated(True)

        self.style_tree.setColumnWidth(
            0,
            150,
        )
        self.style_tree.setColumnWidth(
            1,
            85,
        )
        self.style_tree.setColumnWidth(
            2,
            65,
        )

        layout.addWidget(
            self.style_tree,
            1,
        )

        button_row = QtWidgets.QHBoxLayout()

        button_color = QtWidgets.QPushButton(
            "Change selected color..."
        )
        button_reset = QtWidgets.QPushButton(
            "Reset selected"
        )
        button_reset_all = QtWidgets.QPushButton(
            "Reset all styles"
        )

        button_row.addWidget(
            button_color
        )
        button_row.addWidget(
            button_reset
        )
        button_row.addWidget(
            button_reset_all
        )

        layout.addLayout(
            button_row
        )

        self.style_tree.itemChanged.connect(
            self._style_item_changed
        )

        self.style_tree.itemClicked.connect(
            self._style_item_clicked
        )

        button_color.clicked.connect(
            self._change_selected_style_color
        )

        button_reset.clicked.connect(
            self._reset_selected_style
        )

        button_reset_all.clicked.connect(
            self._reset_all_styles
        )

    def _build_bonds_tab(self) -> None:
        layout = QtWidgets.QFormLayout(
            self.tab_bonds
        )

        self.chk_show_bonds = QtWidgets.QCheckBox(
            "Show bonds"
        )
        self.chk_show_bonds.setChecked(True)

        self.chk_two_color_bonds = QtWidgets.QCheckBox(
            "Two-color VESTA-style bonds"
        )
        self.chk_two_color_bonds.setChecked(True)

        self.spin_bond_tolerance = QtWidgets.QDoubleSpinBox()
        self.spin_bond_tolerance.setRange(
            0.50,
            2.50,
        )
        self.spin_bond_tolerance.setDecimals(
            3
        )
        self.spin_bond_tolerance.setValue(
            1.15
        )

        self.spin_bond_minimum = QtWidgets.QDoubleSpinBox()
        self.spin_bond_minimum.setRange(
            0.0,
            10.0,
        )
        self.spin_bond_minimum.setDecimals(
            3
        )
        self.spin_bond_minimum.setValue(
            0.10
        )
        self.spin_bond_minimum.setSuffix(
            " Å"
        )

        self.spin_bond_maximum = QtWidgets.QDoubleSpinBox()
        self.spin_bond_maximum.setRange(
            0.0,
            20.0,
        )
        self.spin_bond_maximum.setDecimals(
            3
        )
        self.spin_bond_maximum.setSpecialValueText(
            "Automatic"
        )
        self.spin_bond_maximum.setValue(
            0.0
        )
        self.spin_bond_maximum.setSuffix(
            " Å"
        )

        self.spin_bond_radius = QtWidgets.QDoubleSpinBox()
        self.spin_bond_radius.setRange(
            0.01,
            1.0,
        )
        self.spin_bond_radius.setDecimals(
            3
        )
        self.spin_bond_radius.setValue(
            0.08
        )
        self.spin_bond_radius.setSuffix(
            " Å"
        )

        self.spin_bond_opacity = QtWidgets.QDoubleSpinBox()
        self.spin_bond_opacity.setRange(
            0.05,
            1.0,
        )
        self.spin_bond_opacity.setDecimals(
            2
        )
        self.spin_bond_opacity.setValue(
            1.0
        )

        button_apply = QtWidgets.QPushButton(
            "Apply / regenerate bonds"
        )

        self.bond_status_label = QtWidgets.QLabel(
            "No bonds generated."
        )
        self.bond_status_label.setWordWrap(True)

        layout.addRow(
            self.chk_show_bonds
        )
        layout.addRow(
            self.chk_two_color_bonds
        )
        layout.addRow(
            "Covalent-radius scale:",
            self.spin_bond_tolerance,
        )
        layout.addRow(
            "Minimum distance:",
            self.spin_bond_minimum,
        )
        layout.addRow(
            "Maximum distance:",
            self.spin_bond_maximum,
        )
        layout.addRow(
            "Bond radius:",
            self.spin_bond_radius,
        )
        layout.addRow(
            "Bond opacity:",
            self.spin_bond_opacity,
        )
        layout.addRow(
            button_apply
        )
        layout.addRow(
            self.bond_status_label
        )

        button_apply.clicked.connect(
            self._regenerate_bonds_and_scene
        )

        self.chk_show_bonds.toggled.connect(
            self._on_bonds_visibility_changed
        )

    def _build_polyhedra_tab(self) -> None:
        layout = QtWidgets.QFormLayout(
            self.tab_polyhedra
        )

        information = QtWidgets.QLabel(
            "Polyhedra are generated from the current periodic bond network."
        )
        information.setWordWrap(True)

        self.chk_show_polyhedra = QtWidgets.QCheckBox(
            "Show coordination polyhedra"
        )

        self.combo_poly_center = QtWidgets.QComboBox()
        self.combo_poly_neighbor = QtWidgets.QComboBox()

        self.combo_poly_neighbor.addItem(
            "All bonded elements",
            "*",
        )

        self.combo_poly_mode = QtWidgets.QComboBox()
        self.combo_poly_mode.addItem(
            "Selected atom only",
            "selected",
        )
        self.combo_poly_mode.addItem(
            "All central atoms",
            "all",
        )

        self.button_poly_face_color = QtWidgets.QPushButton(
            "Choose..."
        )
        self.button_poly_face_color.setProperty(
            "selected_color",
            "#4A90E2",
        )
        self.button_poly_face_color.setStyleSheet(
            "background-color: #4A90E2;"
        )

        self.button_poly_edge_color = QtWidgets.QPushButton(
            "Choose..."
        )
        self.button_poly_edge_color.setProperty(
            "selected_color",
            "#1F4F8A",
        )
        self.button_poly_edge_color.setStyleSheet(
            "background-color: #1F4F8A;"
        )

        self.spin_poly_opacity = QtWidgets.QDoubleSpinBox()
        self.spin_poly_opacity.setRange(
            0.05,
            1.0,
        )
        self.spin_poly_opacity.setDecimals(
            2
        )
        self.spin_poly_opacity.setValue(
            0.35
        )

        self.spin_poly_edge_width = QtWidgets.QDoubleSpinBox()
        self.spin_poly_edge_width.setRange(
            0.5,
            10.0,
        )
        self.spin_poly_edge_width.setValue(
            2.0
        )

        button_generate = QtWidgets.QPushButton(
            "Generate polyhedra"
        )
        button_clear = QtWidgets.QPushButton(
            "Clear polyhedra"
        )

        self.polyhedron_status_label = QtWidgets.QLabel(
            "No polyhedra generated."
        )
        self.polyhedron_status_label.setWordWrap(True)

        layout.addRow(
            information
        )
        layout.addRow(
            self.chk_show_polyhedra
        )
        layout.addRow(
            "Central element:",
            self.combo_poly_center,
        )
        layout.addRow(
            "Neighbor element:",
            self.combo_poly_neighbor,
        )
        layout.addRow(
            "Display mode:",
            self.combo_poly_mode,
        )
        layout.addRow(
            "Face color:",
            self.button_poly_face_color,
        )
        layout.addRow(
            "Edge color:",
            self.button_poly_edge_color,
        )
        layout.addRow(
            "Face opacity:",
            self.spin_poly_opacity,
        )
        layout.addRow(
            "Edge width:",
            self.spin_poly_edge_width,
        )
        layout.addRow(
            button_generate
        )
        layout.addRow(
            button_clear
        )
        layout.addRow(
            self.polyhedron_status_label
        )

        self.button_poly_face_color.clicked.connect(
            lambda:
                self._choose_button_color(
                    self.button_poly_face_color
                )
        )

        self.button_poly_edge_color.clicked.connect(
            lambda:
                self._choose_button_color(
                    self.button_poly_edge_color
                )
        )

        button_generate.clicked.connect(
            self._generate_polyhedra
        )

        button_clear.clicked.connect(
            self._clear_polyhedra
        )

        self.chk_show_polyhedra.toggled.connect(
            self._set_polyhedra_visible
        )

    # ------------------------------------------------------------------
    # Structure loading
    # ------------------------------------------------------------------
    def load_structure(
        self,
        path: str,
        *,
        lattice_params: Optional[Dict[str, float]] = None,
        input_path: Optional[str] = None,
    ) -> None:
        path = str(
            path
            or ""
        ).strip()

        if not path:
            raise ValueError(
                "The structure path is empty."
            )

        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                path
            )

        structure = None

        try:
            if path.lower().endswith(
                ".cif"
            ):
                from pdf_fitting.gui.cif_utils import read_cif_asu_sites

                structure, _sites = read_cif_asu_sites(
                    path
                )

        except Exception:
            structure = None

        if structure is None:
            structure = Structure.from_file(
                path
            )

        if lattice_params:
            structure = self._structure_with_lattice_params(
                structure,
                lattice_params,
            )

        self.structure = structure
        self.structure_path = os.path.abspath(
            path
        )

        self.set_input_path(
            input_path
        )

        self.element_styles = {}
        self.site_styles = {}

        self.selected_site_index = None
        self.selected_atom_record = None

        self.polyhedra_requested = False

        self._initialize_default_styles()
        self._populate_style_tree()
        self._populate_polyhedron_elements()

        loaded_configuration = False

        if (
            self.configuration_path
            and os.path.exists(
                self.configuration_path
            )
        ):
            loaded_configuration = self.load_viewer_configuration(
                self.configuration_path,
                silent=True,
            )

        if not loaded_configuration:
            self._rebuild_display_model()

            self._rebuild_scene(
                reset_camera=True
            )

        self.setWindowTitle(
            "GPU Crystal Structure Viewer — "
            f"{os.path.basename(path)}"
        )

        lattice = structure.lattice

        self.statusBar().showMessage(
            f"Loaded {len(structure.sites)} base-cell atoms. "
            f"a={lattice.a:.5g}, b={lattice.b:.5g}, c={lattice.c:.5g} Å; "
            f"α={lattice.alpha:.4g}, β={lattice.beta:.4g}, "
            f"γ={lattice.gamma:.4g}°."
        )

    def set_input_path(
        self,
        input_path: Optional[str],
    ) -> None:
        """
        Set the PDF input file associated with this viewer.

        The automatic sidecar file is:

            input_name.structure-view.json
        """
        input_path = str(
            input_path
            or ""
        ).strip()

        self.input_path = input_path

        if input_path:
            absolute_input_path = os.path.abspath(
                input_path
            )

            input_directory = os.path.dirname(
                absolute_input_path
            )

            input_stem = os.path.splitext(
                os.path.basename(
                    absolute_input_path
                )
            )[0]

            self.configuration_path = os.path.join(
                input_directory,
                f"{input_stem}.structure-view.json",
            )

        else:
            self.configuration_path = ""

        if self.configuration_path:
            self.configuration_path_label.setText(
                "Viewer configuration:\n"
                f"{self.configuration_path}"
            )

            self.button_save_configuration.setEnabled(
                True
            )

        else:
            self.configuration_path_label.setText(
                "Viewer configuration: the PDF input must be saved first."
            )

            self.button_save_configuration.setEnabled(
                False
            )

    def _supercell_repeats(
        self,
    ) -> Tuple[int, int, int]:
        return (
            max(
                1,
                int(
                    self.spin_repeat_a.value()
                ),
            ),
            max(
                1,
                int(
                    self.spin_repeat_b.value()
                ),
            ),
            max(
                1,
                int(
                    self.spin_repeat_c.value()
                ),
            ),
        )

    def _refresh_selected_atom_record(
        self,
    ) -> None:
        """
        Replace the selected record with its current display-model copy.

        Clear the selection if that atom no longer exists or is hidden.
        """
        if self.selected_atom_record is None:
            return

        selected_key = self.selected_atom_record.get(
            "key"
        )

        current_record = self.display_atom_lookup.get(
            selected_key
        )

        if (
            current_record is None
            or not bool(
                current_record.get(
                    "visible",
                    True,
                )
            )
        ):
            self.selected_atom_record = None
            self.selected_site_index = None
            return

        self.selected_atom_record = dict(
            current_record
        )

        self.selected_site_index = int(
            current_record.get(
                "base_site_index",
                0,
            )
        )


    def _rebuild_display_model(
        self,
    ) -> None:
        """
        Rebuild displayed supercell atoms and periodic bonds.
        """
        if self.structure is None:
            self.display_structure = None
            self.display_atom_records = []
            self.display_atom_lookup = {}
            self.bond_records = []
            return

        maximum_distance = float(
            self.spin_bond_maximum.value()
        )

        if maximum_distance <= 0.0:
            maximum_distance = None

        model = build_display_model(
            original_structure=self.structure,
            repeats=self._supercell_repeats(),
            include_boundary_images=bool(
                self.chk_boundary_images.isChecked()
            ),
            style_resolver=self._resolved_style,
            bonds_enabled=bool(
                self.chk_show_bonds.isChecked()
            ),
            tolerance_scale=float(
                self.spin_bond_tolerance.value()
            ),
            minimum_distance=float(
                self.spin_bond_minimum.value()
            ),
            maximum_distance=maximum_distance,
            bond_radius=float(
                self.spin_bond_radius.value()
            ),
            bond_opacity=float(
                self.spin_bond_opacity.value()
            ),
            two_color=bool(
                self.chk_two_color_bonds.isChecked()
            ),
        )

        self.display_structure = model[
            "display_structure"
        ]

        self.display_atom_records = model[
            "atoms"
        ]

        self.display_atom_lookup = model[
            "atom_lookup"
        ]

        self.bond_records = model[
            "bonds"
        ]

        self.bond_status_label.setText(
            f"Displayed atoms: {len(self.display_atom_records):,}; "
            f"periodic bonds: {len(self.bond_records):,}."
        )

        self._refresh_selected_atom_record()


    def _on_supercell_changed(
        self,
        value=None,
    ) -> None:
        if self._loading_configuration:
            return

        self.clear_selection()
        self._clear_polyhedra()

        self._rebuild_display_model()

        self._rebuild_scene(
            reset_camera=True
        )

    def _on_bonds_visibility_changed(
        self,
        checked: bool,
    ) -> None:
        if self._loading_configuration:
            return

        self._clear_polyhedra(
            rebuild=False
        )

        self._rebuild_display_model()
        self._rebuild_scene()


    def _regenerate_bonds_and_scene(
        self,
    ) -> None:
        self._clear_polyhedra()
        self._rebuild_display_model()
        self._rebuild_scene()

                       
    def _structure_with_lattice_params(
        self,
        structure: Structure,
        params: Dict[str, float],
    ) -> Structure:
        from pymatgen.core import Lattice

        old = structure.lattice

        lattice = Lattice.from_parameters(
            float(params.get("a", old.a)),
            float(params.get("b", old.b)),
            float(params.get("c", old.c)),
            float(params.get("alpha", old.alpha)),
            float(params.get("beta", old.beta)),
            float(params.get("gamma", old.gamma)),
        )

        return Structure(
            lattice,
            structure.species,
            structure.frac_coords,
            site_properties=structure.site_properties,
        )

    def _initialize_default_styles(self) -> None:
        if self.structure is None:
            return

        for index, site in enumerate(
            self.structure.sites
        ):
            symbol = clean_element_symbol(
                str(site.specie)
            )

            self.element_styles.setdefault(
                symbol,
                {
                    "color": element_color(symbol),
                    "radius": display_radius(symbol),
                    "visible": True,
                    "show_label": False,
                    "opacity": 1.0,
                },
            )

            self.site_styles[index] = {}

    # ------------------------------------------------------------------
    # Style handling
    # ------------------------------------------------------------------
    def _resolved_style(
        self,
        index: int,
    ) -> Dict[str, Any]:
        if self.structure is None:
            return {}

        site = self.structure[index]

        symbol = clean_element_symbol(
            str(site.specie)
        )

        style = dict(
            self.element_styles.get(
                symbol,
                {
                    "color": element_color(symbol),
                    "radius": display_radius(symbol),
                    "visible": True,
                    "show_label": False,
                    "opacity": 1.0,
                },
            )
        )

        style.update(
            self.site_styles.get(
                index,
                {},
            )
        )

        return style

    def _site_label(
        self,
        index: int,
    ) -> str:
        if self.structure is None:
            return str(index + 1)

        site = self.structure[index]

        label = getattr(
            site,
            "label",
            None,
        )

        if label:
            return str(label)

        symbol = clean_element_symbol(
            str(site.specie)
        )

        return f"{symbol}{index + 1}"

    def _populate_style_tree(self) -> None:
        self._updating_style_tree = True

        try:
            self.style_tree.clear()

            if self.structure is None:
                return

            element_to_indices: Dict[str, List[int]] = {}

            for index, site in enumerate(
                self.structure.sites
            ):
                symbol = clean_element_symbol(
                    str(site.specie)
                )

                element_to_indices.setdefault(
                    symbol,
                    [],
                ).append(
                    index
                )

            for symbol in sorted(
                element_to_indices.keys()
            ):
                style = self.element_styles[symbol]

                parent = self._make_style_item(
                    text=symbol,
                    metadata={
                        "level": "element",
                        "key": symbol,
                    },
                    style=style,
                )

                font = parent.font(
                    0
                )
                font.setBold(True)
                parent.setFont(
                    0,
                    font,
                )

                self.style_tree.addTopLevelItem(
                    parent
                )

                for index in element_to_indices[symbol]:
                    child = self._make_style_item(
                        text=self._site_label(index),
                        metadata={
                            "level": "site",
                            "key": index,
                        },
                        style=self._resolved_style(index),
                    )

                    parent.addChild(
                        child
                    )

                parent.setExpanded(
                    True
                )

        finally:
            self._updating_style_tree = False

    def _make_style_item(
        self,
        *,
        text: str,
        metadata: dict,
        style: dict,
    ):
        item = QtWidgets.QTreeWidgetItem()

        item.setText(
            0,
            text,
        )

        item.setData(
            0,
            QtCore.Qt.ItemDataRole.UserRole,
            metadata,
        )

        item.setText(
            1,
            f"{float(style.get('radius', 0.4)):.3f}",
        )

        color = str(
            style.get(
                "color",
                "#808080",
            )
        )

        item.setData(
            2,
            QtCore.Qt.ItemDataRole.UserRole,
            color,
        )

        item.setBackground(
            2,
            QtGui.QBrush(
                QtGui.QColor(
                    color
                )
            ),
        )

        item.setCheckState(
            3,
            (
                QtCore.Qt.CheckState.Checked
                if style.get("visible", True)
                else QtCore.Qt.CheckState.Unchecked
            ),
        )

        item.setCheckState(
            4,
            (
                QtCore.Qt.CheckState.Checked
                if style.get("show_label", False)
                else QtCore.Qt.CheckState.Unchecked
            ),
        )

        item.setText(
            5,
            f"{float(style.get('opacity', 1.0)):.2f}",
        )

        item.setFlags(
            item.flags()
            | QtCore.Qt.ItemFlag.ItemIsEditable
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )

        return item

    def _style_from_item(
        self,
        item,
    ) -> Dict[str, Any]:
        try:
            radius = float(
                item.text(1)
            )
        except Exception:
            radius = 0.4

        try:
            opacity = float(
                item.text(5)
            )
        except Exception:
            opacity = 1.0

        color = item.data(
            2,
            QtCore.Qt.ItemDataRole.UserRole,
        )

        return {
            "radius": max(
                0.01,
                radius,
            ),
            "color": str(
                color
                or "#808080"
            ),
            "visible": (
                item.checkState(3)
                == QtCore.Qt.CheckState.Checked
            ),
            "show_label": (
                item.checkState(4)
                == QtCore.Qt.CheckState.Checked
            ),
            "opacity": max(
                0.0,
                min(
                    1.0,
                    opacity,
                ),
            ),
        }

    def _style_item_changed(
        self,
        item,
        column,
    ) -> None:
        if self._updating_style_tree:
            return

        metadata = item.data(
            0,
            QtCore.Qt.ItemDataRole.UserRole,
        )

        if not isinstance(
            metadata,
            dict,
        ):
            return

        style = self._style_from_item(
            item
        )

        level = metadata.get(
            "level"
        )
        key = metadata.get(
            "key"
        )

        if level == "element":
            self.element_styles[str(key)] = dict(
                style
            )

        elif level == "site":
            self.site_styles[int(key)] = dict(
                style
            )

        self._rebuild_display_model()
        self._rebuild_scene()

    def _style_item_clicked(
        self,
        item,
        column,
    ) -> None:
        if column != 2:
            return

        self._choose_style_item_color(
            item
        )

    def _choose_style_item_color(
        self,
        item,
    ) -> None:
        old_color = item.data(
            2,
            QtCore.Qt.ItemDataRole.UserRole,
        )

        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(
                str(
                    old_color
                    or "#808080"
                )
            ),
            self,
            "Choose atom color",
        )

        if not color.isValid():
            return

        self._updating_style_tree = True

        try:
            item.setData(
                2,
                QtCore.Qt.ItemDataRole.UserRole,
                color.name(),
            )

            item.setBackground(
                2,
                QtGui.QBrush(
                    color
                ),
            )

        finally:
            self._updating_style_tree = False

        self._style_item_changed(
            item,
            2,
        )

    def _change_selected_style_color(self) -> None:
        item = self.style_tree.currentItem()

        if item is None:
            return

        self._choose_style_item_color(
            item
        )

    def _reset_selected_style(self) -> None:
        item = self.style_tree.currentItem()

        if item is None or self.structure is None:
            return

        metadata = item.data(
            0,
            QtCore.Qt.ItemDataRole.UserRole,
        )

        if not isinstance(
            metadata,
            dict,
        ):
            return

        if metadata.get("level") == "element":
            symbol = str(
                metadata["key"]
            )

            self.element_styles[symbol] = {
                "color": element_color(symbol),
                "radius": display_radius(symbol),
                "visible": True,
                "show_label": False,
                "opacity": 1.0,
            }

        else:
            index = int(
                metadata["key"]
            )

            self.site_styles[index] = {}

        self._populate_style_tree()
        self._rebuild_display_model()
        self._rebuild_scene()

    def _reset_all_styles(self) -> None:
        self.element_styles = {}
        self.site_styles = {}

        self._initialize_default_styles()
        self._populate_style_tree()
        self._rebuild_display_model()
        self._rebuild_scene()

    # ------------------------------------------------------------------
    # Bonds
    # ------------------------------------------------------------------

    def _regenerate_bonds_and_scene(self) -> None:
        self._rebuild_display_model()
        self._clear_polyhedra()
        self._rebuild_scene()

    # ------------------------------------------------------------------
    # Scene rendering
    # ------------------------------------------------------------------
    def _configure_renderer(self) -> None:
        self.plotter.set_background(
            "#F4F4F4"
        )

        try:
            self.plotter.enable_anti_aliasing(
                "ssaa"
            )
        except Exception:
            try:
                self.plotter.enable_anti_aliasing(
                    "fxaa"
                )
            except Exception:
                pass

        try:
            self.plotter.enable_depth_peeling(
                number_of_peels=8,
                occlusion_ratio=0.0,
            )
        except Exception:
            pass

        try:
            self.plotter.enable_ssao(
                radius=1.5,
                bias=0.01,
                kernel_size=64,
                blur=True,
            )
        except Exception:
            pass

        self._configure_lighting()

        try:
            self.plotter.hide_axes()
        except Exception:
            pass

    def _configure_lighting(self) -> None:
        try:
            self.plotter.remove_all_lights()
        except Exception:
            pass

        try:
            key_light = self.pv.Light(
                light_type="headlight",
                color="white",
                intensity=0.90,
            )
            self.plotter.add_light(
                key_light
            )
        except Exception:
            pass

        try:
            fill_light = self.pv.Light(
                light_type="camera light",
                color="#FFF1E6",
                intensity=0.30,
            )

            fill_light.set_direction_angle(
                35.0,
                -45.0,
            )

            self.plotter.add_light(
                fill_light
            )
        except Exception:
            pass

        try:
            rim_light = self.pv.Light(
                light_type="camera light",
                color="#DDE8FF",
                intensity=0.20,
            )

            rim_light.set_direction_angle(
                -35.0,
                135.0,
            )

            self.plotter.add_light(
                rim_light
            )
        except Exception:
            pass

    def _capture_camera(self):
        if not self._camera_initialized:
            return None

        try:
            return copy.deepcopy(
                self.plotter.camera_position
            )
        except Exception:
            return None

    def _rebuild_scene(
        self,
        checked=False,
        *,
        reset_camera: bool = False,
    ) -> None:
        if self.structure is None:
            return

        previous_camera = self._capture_camera()

        try:
            self.plotter.clear()
        except Exception:
            pass

        regenerate_polyhedra = bool(
            self.polyhedra_requested
        )

        self.atom_actors = []
        self.bond_actors = []
        self.cell_actors = []
        self.label_actors = []
        self.highlight_actors = []

        self._configure_lighting()

        if self.chk_show_cell.isChecked():
            self._add_unit_cell()

        if self.chk_show_axes.isChecked():
            self._add_lattice_vectors()

        if (
            self.chk_show_bonds.isChecked()
            and self.bond_records
        ):
            self._add_bonds()

        if self.chk_show_atoms.isChecked():
            self._add_atoms()

        if self.chk_show_labels.isChecked():
            self._add_labels()

        self._enable_picking()

        if reset_camera or previous_camera is None:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
            self._camera_initialized = True

        else:
            try:
                self.plotter.camera_position = previous_camera
            except Exception:
                self.plotter.reset_camera()

        if self.selected_atom_record is not None:
            self._highlight_atom_record(
                self.selected_atom_record
            )
        try:
            self.plotter.render()
        except Exception:
            pass

        if regenerate_polyhedra:
            QtCore.QTimer.singleShot(
                0,
                self._generate_polyhedra,
            )

    def _add_unit_cell(self) -> None:
        if self.display_structure is None:
            return

        matrix = np.asarray(
            self.display_structure.lattice.matrix,
            dtype=float,
        )

        a = matrix[0]
        b = matrix[1]
        c = matrix[2]

        corners = np.asarray(
            [
                [0.0, 0.0, 0.0],
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

        edges = [
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
        ]

        lines = []

        for start, end in edges:
            lines.extend(
                [
                    2,
                    start,
                    end,
                ]
            )

        mesh = self.pv.PolyData()
        mesh.points = corners
        mesh.lines = np.asarray(
            lines,
            dtype=np.int64,
        )

        actor = self.plotter.add_mesh(
            mesh,
            color="#202020",
            line_width=2.5,
            render_lines_as_tubes=True,
            lighting=True,
            pickable=False,
            render=False,
        )

        self.cell_actors.append(
            actor
        )

    def _add_lattice_vectors(
        self,
    ) -> None:
        """
        Draw the actual crystallographic direct-lattice vectors a, b and c.

        These replace the Cartesian x, y, z orientation widget.
        """
        if self.display_structure is None:
            return

        matrix = np.asarray(
            self.display_structure.lattice.matrix,
            dtype=float,
        )

        colors = {
            "a": "#E53935",
            "b": "#43A047",
            "c": "#1E5BFF",
        }

        label_points = []
        label_text = []

        for index, label in enumerate(
            (
                "a",
                "b",
                "c",
            )
        ):
            vector = np.asarray(
                matrix[index],
                dtype=float,
            )

            length = float(
                np.linalg.norm(
                    vector
                )
            )

            if length <= 1.0e-12:
                continue

            arrow = self.pv.Arrow(
                start=np.zeros(
                    3,
                    dtype=float,
                ),
                direction=vector,
                tip_length=0.16,
                tip_radius=0.06,
                shaft_radius=0.018,
                scale="auto",
            )

            actor = self.plotter.add_mesh(
                arrow,
                color=colors[label],
                smooth_shading=True,
                ambient=0.25,
                diffuse=0.75,
                specular=0.25,
                specular_power=20.0,
                pickable=False,
                name=f"lattice_vector_{label}",
                render=False,
            )

            self.cell_actors.append(
                actor
            )

            label_points.append(
                1.08 * vector
            )

            label_text.append(
                label
            )

        if label_points:
            actor = self.plotter.add_point_labels(
                np.asarray(
                    label_points,
                    dtype=float,
                ),
                label_text,
                font_size=18,
                text_color="#202020",
                point_size=0,
                shape=None,
                always_visible=True,
                name="lattice_vector_labels",
                render=False,
            )

            self.label_actors.append(
                actor
            )


    def _add_atoms(
        self,
    ) -> None:
        """
        Render the complete displayed supercell, including optional boundary images.
        """
        self.atom_records = []

        grouped: Dict[
            Tuple[Any, ...],
            List[Dict[str, Any]],
        ] = {}

        global_scale = (
            float(
                self.atom_scale_slider.value()
            )
            / 100.0
        )

        for source_record in self.display_atom_records:
            if not bool(
                source_record.get(
                    "visible",
                    True,
                )
            ):
                continue

            record = dict(
                source_record
            )

            record["radius"] = (
                float(
                    source_record.get(
                        "radius",
                        0.4,
                    )
                )
                * global_scale
            )

            self.atom_records.append(
                record
            )

            group_key = (
                str(
                    record.get(
                        "color",
                        "#808080",
                    )
                ),
                round(
                    float(
                        record["radius"]
                    ),
                    5,
                ),
                round(
                    float(
                        record.get(
                            "opacity",
                            1.0,
                        )
                    ),
                    4,
                ),
            )

            grouped.setdefault(
                group_key,
                [],
            ).append(
                record
            )

        for group_index, (
            group_key,
            records,
        ) in enumerate(
            grouped.items()
        ):
            color, radius, opacity = group_key

            points = np.asarray(
                [
                    record["position"]
                    for record in records
                ],
                dtype=float,
            )

            cloud = self.pv.PolyData(
                points
            )

            sphere = self.pv.Sphere(
                radius=float(
                    radius
                ),
                theta_resolution=24,
                phi_resolution=18,
            )

            glyphs = cloud.glyph(
                geom=sphere,
                scale=False,
                orient=False,
            )

            actor = self.plotter.add_mesh(
                glyphs,
                color=color,
                opacity=float(
                    opacity
                ),
                smooth_shading=True,
                ambient=0.10,
                diffuse=0.78,
                specular=0.70,
                specular_power=55.0,
                show_edges=False,
                pickable=True,
                name=f"atom_group_{group_index}",
                render=False,
            )

            self.atom_actors.append(
                actor
            )
            
    def _make_bond_tube(
        self,
        start,
        end,
    ):
        start = np.asarray(
            start,
            dtype=float,
        )

        end = np.asarray(
            end,
            dtype=float,
        )

        if np.linalg.norm(
            end - start
        ) <= 1.0e-12:
            return None

        line = self.pv.Line(
            start,
            end,
            resolution=1,
        )

        return line.tube(
            radius=float(
                self.spin_bond_radius.value()
            ),
            n_sides=20,
            capping=True,
        )

    def _add_bonds(
        self,
    ) -> None:
        grouped: Dict[
            Tuple[str, float, float],
            List[Any],
        ] = {}

        for bond in self.bond_records:
            radius = float(
                bond.get(
                    "radius",
                    self.spin_bond_radius.value(),
                )
            )

            opacity = float(
                bond.get(
                    "opacity",
                    self.spin_bond_opacity.value(),
                )
            )

            two_color = bool(
                bond.get(
                    "two_color",
                    self.chk_two_color_bonds.isChecked(),
                )
            )

            if two_color:
                segments = [
                    (
                        bond["start"],
                        bond["midpoint"],
                        bond["color_center"],
                    ),
                    (
                        bond["midpoint"],
                        bond["end"],
                        bond["color_neighbor"],
                    ),
                ]
            else:
                segments = [
                    (
                        bond["start"],
                        bond["end"],
                        "#808080",
                    )
                ]

            for start, end, color in segments:
                line = self.pv.Line(
                    np.asarray(
                        start,
                        dtype=float,
                    ),
                    np.asarray(
                        end,
                        dtype=float,
                    ),
                    resolution=1,
                )

                tube = line.tube(
                    radius=radius,
                    n_sides=20,
                    capping=True,
                )

                group_key = (
                    str(
                        color
                    ),
                    round(
                        radius,
                        6,
                    ),
                    round(
                        opacity,
                        4,
                    ),
                )

                grouped.setdefault(
                    group_key,
                    [],
                ).append(
                    tube
                )

        for index, (
            group_key,
            meshes,
        ) in enumerate(
            grouped.items()
        ):
            color, radius, opacity = group_key

            if len(meshes) == 1:
                merged = meshes[0]
            else:
                merged = self.pv.merge(
                    meshes,
                    merge_points=False,
                )

            actor = self.plotter.add_mesh(
                merged,
                color=color,
                opacity=float(
                    opacity
                ),
                smooth_shading=True,
                ambient=0.20,
                diffuse=0.75,
                specular=0.30,
                specular_power=24.0,
                pickable=False,
                name=f"bond_group_{index}",
                render=False,
            )

            self.bond_actors.append(
                actor
            )

    def _add_labels(
        self,
    ) -> None:
        points = []
        labels = []

        for record in self.display_atom_records:
            if not bool(
                record.get(
                    "visible",
                    True,
                )
            ):
                continue

            if not bool(
                record.get(
                    "show_label",
                    False,
                )
            ):
                continue

            points.append(
                np.asarray(
                    record["position"],
                    dtype=float,
                )
            )

            labels.append(
                str(
                    record.get(
                        "label",
                        "",
                    )
                )
            )

        if not points:
            return

        actor = self.plotter.add_point_labels(
            np.asarray(
                points,
                dtype=float,
            ),
            labels,
            font_size=13,
            point_size=0,
            shape=None,
            text_color="#202020",
            always_visible=True,
            name="site_labels",
            render=False,
        )

        self.label_actors.append(
            actor
        )

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------
    def _enable_picking(self) -> None:
        try:
            self.plotter.disable_picking()
        except Exception:
            pass

        try:
            self.plotter.enable_surface_point_picking(
                callback=self._picked_point,
                show_message=False,
                show_point=False,
                left_clicking=True,
                pickable_window=False,
            )
        except Exception:
            try:
                self.plotter.enable_point_picking(
                    callback=self._picked_point,
                    show_message=False,
                    show_point=False,
                    left_clicking=True,
                )
            except Exception:
                pass

    def _picked_point(
        self,
        point,
    ) -> None:
        if not self.atom_records:
            return

        try:
            point = np.asarray(
                point,
                dtype=float,
            ).reshape(3)
        except Exception:
            return

        positions = np.asarray(
            [
                record["position"]
                for record in self.atom_records
            ],
            dtype=float,
        )

        distances = np.linalg.norm(
            positions - point[None, :],
            axis=1,
        )

        row = int(
            np.argmin(
                distances
            )
        )

        record = self.atom_records[row]

        self.select_atom_record(
            record
        )

    def select_atom_record(
        self,
        record: Dict[str, Any],
    ) -> None:
        """
        Select one displayed supercell atom or periodic boundary image.
        """
        if self.structure is None:
            return

        self.selected_atom_record = dict(
            record
        )

        self.selected_site_index = int(
            record.get(
                "base_site_index",
                0,
            )
        )

        fractional = np.asarray(
            record.get(
                "fractional",
                [
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
            dtype=float,
        )

        cartesian = np.asarray(
            record.get(
                "position",
                [
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
            dtype=float,
        )

        atom_key = record.get(
            "key"
        )

        coordination = 0

        for bond in self.bond_records:
            if (
                bond.get(
                    "center_key"
                )
                == atom_key
                or bond.get(
                    "neighbor_key"
                )
                == atom_key
            ):
                coordination += 1

        self.selection_label.setText(
            "\n".join(
                [
                    f"Site: {record.get('label', '')}",
                    f"Element: {record.get('element', 'X')}",
                    (
                        "Fractional in displayed supercell: "
                        f"({fractional[0]:.6f}, "
                        f"{fractional[1]:.6f}, "
                        f"{fractional[2]:.6f})"
                    ),
                    (
                        "Cartesian (Å): "
                        f"({cartesian[0]:.6f}, "
                        f"{cartesian[1]:.6f}, "
                        f"{cartesian[2]:.6f})"
                    ),
                    (
                        "Periodic boundary image: "
                        f"{bool(record.get('boundary_image', False))}"
                    ),
                    f"Coordination number: {coordination}",
                ]
            )
        )

        self._highlight_atom_record(
            record
        )

        try:
            self.plotter.camera.focal_point = (
                cartesian.tolist()
            )

            self.plotter.render()

        except Exception:
            pass


    def select_site(
        self,
        site_index: int,
    ) -> None:
        if self.structure is None:
            return

        if (
            site_index < 0
            or site_index >= len(self.structure)
        ):
            return

        self.selected_site_index = int(
            site_index
        )

        site = self.structure[
            self.selected_site_index
        ]

        symbol = clean_element_symbol(
            str(site.specie)
        )

        fractional = np.asarray(
            site.frac_coords,
            dtype=float,
        )

        cartesian = np.asarray(
            site.coords,
            dtype=float,
        )

        coordination = sum(
            1
            for bond in self.bond_records
            if (
                bond["center"] == self.selected_site_index
                or bond["neighbor"] == self.selected_site_index
            )
        )

        self.selection_label.setText(
            "\n".join(
                [
                    f"Site: {self._site_label(self.selected_site_index)}",
                    f"Element: {symbol}",
                    (
                        "Fractional: "
                        f"({fractional[0]:.6f}, "
                        f"{fractional[1]:.6f}, "
                        f"{fractional[2]:.6f})"
                    ),
                    (
                        "Cartesian (Å): "
                        f"({cartesian[0]:.6f}, "
                        f"{cartesian[1]:.6f}, "
                        f"{cartesian[2]:.6f})"
                    ),
                    f"Coordination number: {coordination}",
                ]
            )
        )

        self._highlight_site(
            self.selected_site_index
        )

        try:
            self.plotter.camera.focal_point = (
                cartesian.tolist()
            )
            self.plotter.render()
        except Exception:
            pass

    def _clear_highlight(self) -> None:
        for actor in self.highlight_actors:
            try:
                self.plotter.remove_actor(
                    actor,
                    reset_camera=False,
                    render=False,
                )
            except Exception:
                pass

        self.highlight_actors = []

    def _highlight_atom_record(
        self,
        record: Dict[str, Any],
    ) -> None:
        self._clear_highlight()

        global_scale = (
            float(
                self.atom_scale_slider.value()
            )
            / 100.0
        )

        radius = (
            float(
                record.get(
                    "radius",
                    0.4,
                )
            )
            * global_scale
            * 1.25
        )

        sphere = self.pv.Sphere(
            radius=radius,
            center=np.asarray(
                record.get(
                    "position",
                    [
                        0.0,
                        0.0,
                        0.0,
                    ],
                ),
                dtype=float,
            ),
            theta_resolution=40,
            phi_resolution=32,
        )

        shell_actor = self.plotter.add_mesh(
            sphere,
            color="#FFD400",
            opacity=0.25,
            smooth_shading=True,
            pickable=False,
            render=False,
        )

        wire_actor = self.plotter.add_mesh(
            sphere,
            style="wireframe",
            color="#202020",
            opacity=0.8,
            line_width=1.5,
            pickable=False,
            render=False,
        )

        self.highlight_actors = [
            shell_actor,
            wire_actor,
        ]

        self.plotter.render()


    def _highlight_site(
        self,
        site_index: int,
    ) -> None:
        self._clear_highlight()

        if self.structure is None:
            return

        style = self._resolved_style(
            site_index
        )

        global_scale = (
            float(
                self.atom_scale_slider.value()
            )
            / 100.0
        )

        radius = (
            float(
                style.get(
                    "radius",
                    0.4,
                )
            )
            * global_scale
            * 1.25
        )

        sphere = self.pv.Sphere(
            radius=radius,
            center=np.asarray(
                self.structure[site_index].coords,
                dtype=float,
            ),
            theta_resolution=40,
            phi_resolution=32,
        )

        shell_actor = self.plotter.add_mesh(
            sphere,
            color="#FFD400",
            opacity=0.25,
            smooth_shading=True,
            pickable=False,
            render=False,
        )

        wire_actor = self.plotter.add_mesh(
            sphere,
            style="wireframe",
            color="#202020",
            opacity=0.8,
            line_width=1.5,
            pickable=False,
            render=False,
        )

        self.highlight_actors = [
            shell_actor,
            wire_actor,
        ]

        self.plotter.render()

    def clear_selection(
        self,
    ) -> None:
        self.selected_site_index = None
        self.selected_atom_record = None

        self._clear_highlight()

        self.selection_label.setText(
            "Click an atom to inspect it."
        )

        try:
            self.plotter.render()
        except Exception:
            pass

             
    # ------------------------------------------------------------------
    # Polyhedra
    # ------------------------------------------------------------------
    def _populate_polyhedron_elements(self) -> None:
        self.combo_poly_center.clear()
        self.combo_poly_neighbor.clear()

        self.combo_poly_neighbor.addItem(
            "All bonded elements",
            "*",
        )

        if self.structure is None:
            return

        elements = sorted(
            {
                clean_element_symbol(
                    str(site.specie)
                )
                for site in self.structure.sites
            }
        )

        self.combo_poly_center.addItems(
            elements
        )

        for symbol in elements:
            self.combo_poly_neighbor.addItem(
                symbol,
                symbol,
            )

    def _choose_button_color(
        self,
        button,
    ) -> None:
        old = button.property(
            "selected_color"
        )

        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(
                str(
                    old
                    or "#808080"
                )
            ),
            self,
            "Choose color",
        )

        if not color.isValid():
            return

        button.setProperty(
            "selected_color",
            color.name(),
        )

        button.setStyleSheet(
            f"background-color: {color.name()};"
        )

    def _neighbor_points_for_record(
        self,
        central_record: Dict[str, Any],
        neighbor_element: Optional[str],
    ) -> np.ndarray:
        """
        Return bonded periodic-neighbor positions for one displayed atom.
        """
        central_key = central_record.get(
            "key"
        )

        central_position = np.asarray(
            central_record.get(
                "position",
                [
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
            dtype=float,
        )

        points = []

        for bond in self.bond_records:
            if bond.get(
                "center_key"
            ) == central_key:
                if (
                    neighbor_element is not None
                    and bond.get(
                        "neighbor_element"
                    )
                    != neighbor_element
                ):
                    continue

                points.append(
                    np.asarray(
                        bond["end"],
                        dtype=float,
                    )
                )

            elif bond.get(
                "neighbor_key"
            ) == central_key:
                if (
                    neighbor_element is not None
                    and bond.get(
                        "center_element"
                    )
                    != neighbor_element
                ):
                    continue

                displacement = (
                    np.asarray(
                        bond["start"],
                        dtype=float,
                    )
                    - np.asarray(
                        bond["end"],
                        dtype=float,
                    )
                )

                points.append(
                    central_position
                    + displacement
                )

        if not points:
            return np.zeros(
                (
                    0,
                    3,
                ),
                dtype=float,
            )

        points = np.asarray(
            points,
            dtype=float,
        )

        rounded = np.round(
            points,
            decimals=7,
        )

        _unique, indices = np.unique(
            rounded,
            axis=0,
            return_index=True,
        )

        return points[
            np.sort(
                indices
            )
        ]

    
    def _polyhedron_mesh(
        self,
        points: np.ndarray,
    ):
        if points.shape[0] < 4:
            return None

        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(
                points,
                qhull_options="QJ",
            )

            faces = []

            for triangle in hull.simplices:
                faces.extend(
                    [
                        3,
                        int(triangle[0]),
                        int(triangle[1]),
                        int(triangle[2]),
                    ]
                )

            return self.pv.PolyData(
                points,
                np.asarray(
                    faces,
                    dtype=np.int64,
                ),
            )

        except Exception:
            return None

    def _generate_polyhedra(self) -> None:
        """
        Generate coordination polyhedra for either:

        - the currently selected displayed atom, or
        - all non-boundary-image atoms of the selected central element.
        """
        if self.structure is None:
            return

        if self.display_structure is None:
            return

        center_element = str(
            self.combo_poly_center.currentText()
        ).strip()

        if not center_element:
            self.polyhedron_status_label.setText(
                "No central element was selected."
            )
            return

        neighbor_data = self.combo_poly_neighbor.currentData()

        if neighbor_data in (
            None,
            "*",
        ):
            neighbor_element = None
        else:
            neighbor_element = str(
                neighbor_data
            ).strip()

        mode = str(
            self.combo_poly_mode.currentData()
            or "selected"
        ).strip().lower()

        center_records: List[Dict[str, Any]] = []

        if mode == "selected":
            if self.selected_atom_record is None:
                self.polyhedra_requested = False

                self.polyhedron_status_label.setText(
                    "Select a central atom in the 3D view first."
                )

                QtWidgets.QMessageBox.information(
                    self,
                    "No atom selected",
                    "Select a central atom in the 3D view first.",
                )
                return

            selected_element = str(
                self.selected_atom_record.get(
                    "element",
                    "X",
                )
            )

            if selected_element != center_element:
                self.polyhedra_requested = False

                self.polyhedron_status_label.setText(
                    f"Select a {center_element} atom first."
                )

                QtWidgets.QMessageBox.information(
                    self,
                    "Wrong central element",
                    f"Select a {center_element} atom first.",
                )
                return

            center_records = [
                self.selected_atom_record
            ]

        else:
            seen_display_indices = set()

            for record in self.display_atom_records:
                if not bool(
                    record.get(
                        "visible",
                        True,
                    )
                ):
                    continue

                if bool(
                    record.get(
                        "boundary_image",
                        False,
                    )
                ):
                    continue

                if str(
                    record.get(
                        "element",
                        "X",
                    )
                ) != center_element:
                    continue

                display_index = int(
                    record.get(
                        "display_site_index",
                        -1,
                    )
                )

                if display_index < 0:
                    continue

                if display_index in seen_display_indices:
                    continue

                seen_display_indices.add(
                    display_index
                )

                center_records.append(
                    record
                )

        if not center_records:
            self.polyhedra_requested = False

            self.polyhedron_status_label.setText(
                f"No visible {center_element} atoms were found."
            )
            return

        # Remove previous polyhedron actors without recursively rebuilding.
        self._clear_polyhedra(
            rebuild=False,
            preserve_request=True,
        )

        face_color = str(
            self.button_poly_face_color.property(
                "selected_color"
            )
            or "#4A90E2"
        )

        edge_color = str(
            self.button_poly_edge_color.property(
                "selected_color"
            )
            or "#1F4F8A"
        )

        opacity = float(
            self.spin_poly_opacity.value()
        )

        edge_width = float(
            self.spin_poly_edge_width.value()
        )

        generated = 0
        skipped = 0

        for center_record in center_records:
            neighbor_points = self._neighbor_points_for_record(
                center_record,
                neighbor_element,
            )

            mesh = self._polyhedron_mesh(
                neighbor_points
            )

            if mesh is None:
                skipped += 1
                continue

            actor = self.plotter.add_mesh(
                mesh,
                color=face_color,
                opacity=opacity,
                show_edges=True,
                edge_color=edge_color,
                line_width=edge_width,
                smooth_shading=False,
                pickable=False,
                name=f"coordination_polyhedron_{generated}",
                render=False,
            )

            self.polyhedron_actors.append(
                actor
            )

            generated += 1

        self.polyhedra_requested = generated > 0

        self.chk_show_polyhedra.blockSignals(
            True
        )

        try:
            self.chk_show_polyhedra.setChecked(
                generated > 0
            )
        finally:
            self.chk_show_polyhedra.blockSignals(
                False
            )

        self.polyhedron_status_label.setText(
            f"Generated {generated} polyhedra. "
            f"Skipped {skipped} centers with fewer than four "
            f"suitable neighbor points."
        )

        try:
            self.plotter.render()
        except Exception:
            pass

    def _add_polyhedra_from_current_settings(self) -> None:
        return

    def _clear_polyhedra(
        self,
        checked=False,
        *,
        rebuild: bool = False,
        preserve_request: bool = False,
    ) -> None:
        """
        Remove all currently rendered coordination polyhedra.

        Parameters
        ----------
        rebuild
            Rebuild the complete structure scene after removing polyhedra.

        preserve_request
            Keep polyhedra_requested unchanged. This is used internally when
            replacing old polyhedra with newly generated polyhedra.
        """
        if not preserve_request:
            self.polyhedra_requested = False

        for actor in list(
            self.polyhedron_actors
        ):
            try:
                self.plotter.remove_actor(
                    actor,
                    reset_camera=False,
                    render=False,
                )
            except Exception:
                pass

        self.polyhedron_actors = []

        if not preserve_request:
            self.chk_show_polyhedra.blockSignals(
                True
            )

            try:
                self.chk_show_polyhedra.setChecked(
                    False
                )
            finally:
                self.chk_show_polyhedra.blockSignals(
                    False
                )

            self.polyhedron_status_label.setText(
                "No polyhedra generated."
            )

        if rebuild:
            self._rebuild_scene()
        else:
            try:
                self.plotter.render()
            except Exception:
                pass


    def _set_polyhedra_visible(
        self,
        visible: bool,
    ) -> None:
        for actor in self.polyhedron_actors:
            try:
                actor.SetVisibility(
                    bool(visible)
                )
            except Exception:
                pass

        self.plotter.render()

    # ------------------------------------------------------------------
    # Camera and output
    # ------------------------------------------------------------------
    def _atom_scale_changed(
        self,
        value: int,
    ) -> None:
        self.atom_scale_label.setText(
            f"{float(value) / 100.0:.2f}"
        )

        self._rebuild_scene()

    def _set_axes_visible(
        self,
        visible: bool,
    ) -> None:
        """
        Retained for compatibility.

        The checkbox now controls actual a, b, c lattice-vector actors.
        """
        self._rebuild_scene()


    def _set_orthographic(
        self,
        enabled: bool,
    ) -> None:
        if enabled:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()

        self.plotter.render()

    def _choose_background(self) -> None:
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(
                "#F4F4F4"
            ),
            self,
            "Choose viewer background",
        )

        if not color.isValid():
            return

        self.plotter.set_background(
            color.name()
        )

        self.plotter.render()

    def _fit_scene(self) -> None:
        self.plotter.reset_camera()
        self.plotter.render()

    def _view_isometric(self) -> None:
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def _apply_hkl_view(self) -> None:
        if self.structure is None:
            return

        h = int(
            self.spin_h.value()
        )
        k = int(
            self.spin_k.value()
        )
        l = int(
            self.spin_l.value()
        )

        if h == 0 and k == 0 and l == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid hkl",
                "h, k and l cannot all be zero.",
            )
            return

        if self.display_structure is None:
            return

        matrix = np.asarray(
            self.display_structure.lattice.matrix,
            dtype=float,
        )

        reciprocal = np.linalg.inv(
            matrix
        ).T

        direction = (
            float(h) * reciprocal[:, 0]
            + float(k) * reciprocal[:, 1]
            + float(l) * reciprocal[:, 2]
        )

        norm = float(
            np.linalg.norm(
                direction
            )
        )

        if norm <= 1.0e-12:
            return

        direction /= norm

        if self.chk_reverse_hkl.isChecked():
            direction = -direction

        focal = np.asarray(
            self.plotter.camera.focal_point,
            dtype=float,
        )

        current_position = np.asarray(
            self.plotter.camera.position,
            dtype=float,
        )

        distance = float(
            np.linalg.norm(
                current_position - focal
            )
        )

        if distance <= 1.0e-8:
            distance = 10.0

        view_up = np.asarray(
            [0.0, 0.0, 1.0],
            dtype=float,
        )

        if abs(
            float(
                np.dot(
                    direction,
                    view_up,
                )
            )
        ) > 0.95:
            view_up = np.asarray(
                [0.0, 1.0, 0.0],
                dtype=float,
            )

        position = (
            focal
            + distance * direction
        )

        self.plotter.camera_position = [
            position.tolist(),
            focal.tolist(),
            view_up.tolist(),
        ]

        self.plotter.reset_camera()
        self.plotter.render()

    def _save_screenshot(self) -> None:
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save structure screenshot",
            "structure.png",
            (
                "PNG image (*.png);;"
                "JPEG image (*.jpg *.jpeg);;"
                "TIFF image (*.tif *.tiff)"
            ),
        )

        if not path:
            return

        self.plotter.screenshot(
            path,
            transparent_background=False,
        )

    def _show_gpu_information(self) -> None:
        try:
            information = self.plotter.render_window.ReportCapabilities()
        except Exception as exc:
            information = str(
                exc
            )

        dialog = QtWidgets.QMessageBox(
            self
        )

        dialog.setWindowTitle(
            "Renderer and GPU information"
        )

        dialog.setText(
            "VTK/OpenGL renderer information"
        )

        dialog.setDetailedText(
            str(
                information
            )
        )

        dialog.setIcon(
            QtWidgets.QMessageBox.Icon.Information
        )

        dialog.exec()

    def collect_viewer_configuration(
        self,
    ) -> Dict[str, Any]:
        """
        Collect colors, radii, visibility, bond settings, camera, supercell,
        and polyhedron settings.
        """
        try:
            camera_position = [
                [
                    float(component)
                    for component in vector
                ]
                for vector in self.plotter.camera_position
            ]
        except Exception:
            camera_position = None

        try:
            background = [
                float(component)
                for component in self.plotter.renderer.GetBackground()
            ]
        except Exception:
            background = [
                0.9569,
                0.9569,
                0.9569,
            ]

        site_styles = {
            str(index): dict(
                style
            )
            for index, style in self.site_styles.items()
        }

        maximum_distance = float(
            self.spin_bond_maximum.value()
        )

        if maximum_distance <= 0.0:
            maximum_distance = None

        polyhedron_settings = {
            "requested": bool(
                self.polyhedra_requested
            ),
            "visible": bool(
                self.chk_show_polyhedra.isChecked()
            ),
            "central_element": str(
                self.combo_poly_center.currentText()
            ),
            "neighbor_element": str(
                self.combo_poly_neighbor.currentData()
                or "*"
            ),
            "display_mode": str(
                self.combo_poly_mode.currentData()
                or "selected"
            ),
            "face_color": str(
                self.button_poly_face_color.property(
                    "selected_color"
                )
                or "#4A90E2"
            ),
            "edge_color": str(
                self.button_poly_edge_color.property(
                    "selected_color"
                )
                or "#1F4F8A"
            ),
            "opacity": float(
                self.spin_poly_opacity.value()
            ),
            "edge_width": float(
                self.spin_poly_edge_width.value()
            ),
        }

        return {
            "format": "hkl-pdf-structure-view",
            "version": 1,
            "saved_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "structure_path": self.structure_path,
            "input_path": self.input_path,
            "element_styles": copy.deepcopy(
                self.element_styles
            ),
            "site_styles": site_styles,
            "display": {
                "show_atoms": bool(
                    self.chk_show_atoms.isChecked()
                ),
                "show_cell": bool(
                    self.chk_show_cell.isChecked()
                ),
                "show_lattice_vectors": bool(
                    self.chk_show_axes.isChecked()
                ),
                "show_labels": bool(
                    self.chk_show_labels.isChecked()
                ),
                "orthographic": bool(
                    self.chk_orthographic.isChecked()
                ),
                "atom_scale": float(
                    self.atom_scale_slider.value()
                )
                / 100.0,
                "background": background,
            },
            "supercell": {
                "repeat_a": int(
                    self.spin_repeat_a.value()
                ),
                "repeat_b": int(
                    self.spin_repeat_b.value()
                ),
                "repeat_c": int(
                    self.spin_repeat_c.value()
                ),
                "boundary_images": bool(
                    self.chk_boundary_images.isChecked()
                ),
            },
            "bonds": {
                "enabled": bool(
                    self.chk_show_bonds.isChecked()
                ),
                "two_color": bool(
                    self.chk_two_color_bonds.isChecked()
                ),
                "tolerance_scale": float(
                    self.spin_bond_tolerance.value()
                ),
                "minimum_distance": float(
                    self.spin_bond_minimum.value()
                ),
                "maximum_distance": maximum_distance,
                "radius": float(
                    self.spin_bond_radius.value()
                ),
                "opacity": float(
                    self.spin_bond_opacity.value()
                ),
            },
            "polyhedra": polyhedron_settings,
            "camera": {
                "camera_position": camera_position,
                "parallel_projection": bool(
                    self.plotter.camera.GetParallelProjection()
                ),
            },
            "window": {
                "splitter_sizes": [
                    int(value)
                    for value in self.splitter.sizes()
                ],
                "tab_index": int(
                    self.controls.currentIndex()
                ),
                "size": [
                    int(
                        self.width()
                    ),
                    int(
                        self.height()
                    ),
                ],
            },
        }


    def save_viewer_configuration(
        self,
        checked=False,
        *,
        path: Optional[str] = None,
        silent: bool = False,
    ) -> bool:
        if path is None:
            path = self.configuration_path

        path = str(
            path
            or ""
        ).strip()

        if not path:
            if silent:
                return False

            path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save viewer configuration",
                "structure-view.json",
                "JSON files (*.json);;All files (*)",
            )

        if not path:
            return False

        if not path.lower().endswith(
            ".json"
        ):
            path += ".json"

        payload = self.collect_viewer_configuration()

        try:
            with open(
                path,
                "w",
                encoding="utf-8",
            ) as output_file:
                json.dump(
                    payload,
                    output_file,
                    indent=2,
                )

                output_file.write(
                    "\n"
                )

            self.configuration_path = os.path.abspath(
                path
            )

            self.configuration_path_label.setText(
                "Viewer configuration:\n"
                f"{self.configuration_path}"
            )

            if not silent:
                self.statusBar().showMessage(
                    f"Viewer configuration saved: {path}"
                )

            return True

        except Exception as exc:
            if not silent:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Configuration save failed",
                    str(exc),
                )

            return False


    def load_viewer_configuration_dialog(
        self,
    ) -> None:
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load viewer configuration",
            self.configuration_path,
            "JSON files (*.json);;All files (*)",
        )

        if not path:
            return

        self.load_viewer_configuration(
            path,
            silent=False,
        )


    def load_viewer_configuration(
        self,
        path: Optional[str] = None,
        *,
        silent: bool = False,
    ) -> bool:
        path = str(
            path
            or self.configuration_path
            or ""
        ).strip()

        if not path or not os.path.exists(
            path
        ):
            return False

        try:
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as input_file:
                payload = json.load(
                    input_file
                )

            if not isinstance(
                payload,
                dict,
            ):
                raise ValueError(
                    "The selected JSON does not contain a configuration dictionary."
                )

            file_format = payload.get(
                "format"
            )

            if file_format not in (
                None,
                "hkl-pdf-structure-view",
            ):
                raise ValueError(
                    f"Unsupported viewer configuration format: {file_format}"
                )

            self._loading_configuration = True

            element_styles = payload.get(
                "element_styles",
                {}
            )

            if isinstance(
                element_styles,
                dict,
            ):
                for symbol, style in element_styles.items():
                    if isinstance(
                        style,
                        dict,
                    ):
                        self.element_styles[
                            str(symbol)
                        ] = dict(
                            style
                        )

            site_styles = payload.get(
                "site_styles",
                {}
            )

            if isinstance(
                site_styles,
                dict,
            ):
                self.site_styles = {
                    int(index): dict(
                        style
                    )
                    for index, style in site_styles.items()
                    if isinstance(
                        style,
                        dict,
                    )
                }

            display = payload.get(
                "display",
                {}
            ) or {}

            self.chk_show_atoms.setChecked(
                bool(
                    display.get(
                        "show_atoms",
                        True,
                    )
                )
            )

            self.chk_show_cell.setChecked(
                bool(
                    display.get(
                        "show_cell",
                        True,
                    )
                )
            )

            self.chk_show_axes.setChecked(
                bool(
                    display.get(
                        "show_lattice_vectors",
                        True,
                    )
                )
            )

            self.chk_show_labels.setChecked(
                bool(
                    display.get(
                        "show_labels",
                        True,
                    )
                )
            )

            self.chk_orthographic.setChecked(
                bool(
                    display.get(
                        "orthographic",
                        False,
                    )
                )
            )

            atom_scale = float(
                display.get(
                    "atom_scale",
                    1.0,
                )
            )

            self.atom_scale_slider.setValue(
                max(
                    20,
                    min(
                        300,
                        int(
                            round(
                                atom_scale
                                * 100.0
                            )
                        ),
                    ),
                )
            )

            background = display.get(
                "background"
            )

            if (
                isinstance(
                    background,
                    list,
                )
                and len(background) == 3
            ):
                self.plotter.set_background(
                    tuple(
                        float(value)
                        for value in background
                    )
                )

            supercell = payload.get(
                "supercell",
                {}
            ) or {}

            self.spin_repeat_a.setValue(
                max(
                    1,
                    int(
                        supercell.get(
                            "repeat_a",
                            1,
                        )
                    ),
                )
            )

            self.spin_repeat_b.setValue(
                max(
                    1,
                    int(
                        supercell.get(
                            "repeat_b",
                            1,
                        )
                    ),
                )
            )

            self.spin_repeat_c.setValue(
                max(
                    1,
                    int(
                        supercell.get(
                            "repeat_c",
                            1,
                        )
                    ),
                )
            )

            self.chk_boundary_images.setChecked(
                bool(
                    supercell.get(
                        "boundary_images",
                        False,
                    )
                )
            )

            bonds = payload.get(
                "bonds",
                {}
            ) or {}

            self.chk_show_bonds.setChecked(
                bool(
                    bonds.get(
                        "enabled",
                        True,
                    )
                )
            )

            self.chk_two_color_bonds.setChecked(
                bool(
                    bonds.get(
                        "two_color",
                        True,
                    )
                )
            )

            self.spin_bond_tolerance.setValue(
                float(
                    bonds.get(
                        "tolerance_scale",
                        1.15,
                    )
                )
            )

            self.spin_bond_minimum.setValue(
                float(
                    bonds.get(
                        "minimum_distance",
                        0.10,
                    )
                )
            )

            maximum_distance = bonds.get(
                "maximum_distance"
            )

            if maximum_distance is None:
                self.spin_bond_maximum.setValue(
                    0.0
                )
            else:
                self.spin_bond_maximum.setValue(
                    float(
                        maximum_distance
                    )
                )

            self.spin_bond_radius.setValue(
                float(
                    bonds.get(
                        "radius",
                        0.08,
                    )
                )
            )

            self.spin_bond_opacity.setValue(
                float(
                    bonds.get(
                        "opacity",
                        1.0,
                    )
                )
            )

            polyhedra = payload.get(
                "polyhedra",
                {}
            ) or {}

            self.polyhedra_requested = bool(
                polyhedra.get(
                    "requested",
                    False,
                )
            )

            central_element = str(
                polyhedra.get(
                    "central_element",
                    "",
                )
            )

            if central_element:
                self.combo_poly_center.setCurrentText(
                    central_element
                )

            neighbor_element = str(
                polyhedra.get(
                    "neighbor_element",
                    "*",
                )
            )

            neighbor_index = self.combo_poly_neighbor.findData(
                neighbor_element
            )

            if neighbor_index >= 0:
                self.combo_poly_neighbor.setCurrentIndex(
                    neighbor_index
                )

            display_mode = str(
                polyhedra.get(
                    "display_mode",
                    "selected",
                )
            )

            mode_index = self.combo_poly_mode.findData(
                display_mode
            )

            if mode_index >= 0:
                self.combo_poly_mode.setCurrentIndex(
                    mode_index
                )

            face_color = str(
                polyhedra.get(
                    "face_color",
                    "#4A90E2",
                )
            )

            edge_color = str(
                polyhedra.get(
                    "edge_color",
                    "#1F4F8A",
                )
            )

            self.button_poly_face_color.setProperty(
                "selected_color",
                face_color,
            )

            self.button_poly_face_color.setStyleSheet(
                f"background-color: {face_color};"
            )

            self.button_poly_edge_color.setProperty(
                "selected_color",
                edge_color,
            )

            self.button_poly_edge_color.setStyleSheet(
                f"background-color: {edge_color};"
            )

            self.spin_poly_opacity.setValue(
                float(
                    polyhedra.get(
                        "opacity",
                        0.35,
                    )
                )
            )

            self.spin_poly_edge_width.setValue(
                float(
                    polyhedra.get(
                        "edge_width",
                        2.0,
                    )
                )
            )

            self.chk_show_polyhedra.setChecked(
                bool(
                    polyhedra.get(
                        "visible",
                        False,
                    )
                )
            )

            camera = payload.get(
                "camera",
                {}
            ) or {}

            self._pending_camera_state = camera.get(
                "camera_position"
            )

            parallel_projection = bool(
                camera.get(
                    "parallel_projection",
                    False,
                )
            )

            self.chk_orthographic.setChecked(
                parallel_projection
            )

            window_data = payload.get(
                "window",
                {}
            ) or {}

            splitter_sizes = window_data.get(
                "splitter_sizes"
            )

            if (
                isinstance(
                    splitter_sizes,
                    list,
                )
                and len(splitter_sizes) == 2
            ):
                self.splitter.setSizes(
                    [
                        int(
                            splitter_sizes[0]
                        ),
                        int(
                            splitter_sizes[1]
                        ),
                    ]
                )

            try:
                self.controls.setCurrentIndex(
                    int(
                        window_data.get(
                            "tab_index",
                            0,
                        )
                    )
                )
            except Exception:
                pass

            window_size = window_data.get(
                "size"
            )

            if (
                isinstance(
                    window_size,
                    list,
                )
                and len(window_size) == 2
            ):
                self.resize(
                    max(
                        700,
                        int(
                            window_size[0]
                        ),
                    ),
                    max(
                        500,
                        int(
                            window_size[1]
                        ),
                    ),
                )

            self.configuration_path = os.path.abspath(
                path
            )

            self.configuration_path_label.setText(
                "Viewer configuration:\n"
                f"{self.configuration_path}"
            )

            self._populate_style_tree()
            self._rebuild_display_model()

            self._rebuild_scene(
                reset_camera=True
            )

            if self._pending_camera_state is not None:
                try:
                    self.plotter.camera_position = (
                        self._pending_camera_state
                    )
                    self.plotter.render()
                except Exception:
                    pass

            if parallel_projection:
                self.plotter.enable_parallel_projection()
            else:
                self.plotter.disable_parallel_projection()

            self._loading_configuration = False

            if (
                self.polyhedra_requested
                and str(
                    self.combo_poly_mode.currentData()
                    or "selected"
                )
                == "all"
            ):
                QtCore.QTimer.singleShot(
                    0,
                    self._generate_polyhedra,
                )

            if not silent:
                self.statusBar().showMessage(
                    f"Viewer configuration loaded: {path}"
                )

            return True

        except Exception as exc:
            self._loading_configuration = False

            if not silent:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Configuration load failed",
                    str(exc),
                )

            return False

    def export_displayed_cif(
        self,
    ) -> None:
        """
        Export the visible periodic supercell as CIF.

        Redundant boundary-image atoms are intentionally not included because
        they are crystallographically equivalent periodic display copies.
        """
        if self.display_structure is None:
            return

        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export displayed supercell as CIF",
            "displayed_supercell.cif",
            "CIF files (*.cif);;All files (*)",
        )

        if not path:
            return

        if not path.lower().endswith(
            ".cif"
        ):
            path += ".cif"

        try:
            export_structure = visible_periodic_structure(
                display_structure=self.display_structure,
                style_resolver=self._resolved_style,
            )

            export_structure.to(
                filename=path
            )

            self.statusBar().showMessage(
                f"Displayed supercell exported as CIF: {path}"
            )

        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "CIF export failed",
                str(exc),
            )


    def export_displayed_xyz(
        self,
    ) -> None:
        """
        Export exactly the atoms currently displayed.

        Unlike CIF export, this includes optional boundary-image atoms because
        XYZ is non-periodic and represents the visible Cartesian scene.
        """
        records = [
            record
            for record in self.display_atom_records
            if bool(
                record.get(
                    "visible",
                    True,
                )
            )
        ]

        if not records:
            QtWidgets.QMessageBox.information(
                self,
                "No displayed atoms",
                "There are no visible atoms to export.",
            )
            return

        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export displayed atoms as XYZ",
            "displayed_structure.xyz",
            "XYZ files (*.xyz);;All files (*)",
        )

        if not path:
            return

        if not path.lower().endswith(
            ".xyz"
        ):
            path += ".xyz"

        try:
            repeats = self._supercell_repeats()

            with open(
                path,
                "w",
                encoding="utf-8",
            ) as output_file:
                output_file.write(
                    f"{len(records)}\n"
                )

                output_file.write(
                    "HKL-PDF displayed structure; "
                    f"supercell={repeats[0]}x{repeats[1]}x{repeats[2]}; "
                    "boundary_images="
                    f"{self.chk_boundary_images.isChecked()}\n"
                )

                for record in records:
                    x, y, z = [
                        float(value)
                        for value in record[
                            "position"
                        ]
                    ]

                    output_file.write(
                        f"{record['element']:2s} "
                        f"{x: .10f} "
                        f"{y: .10f} "
                        f"{z: .10f}\n"
                    )

            self.statusBar().showMessage(
                f"Displayed atoms exported as XYZ: {path}"
            )

        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "XYZ export failed",
                str(exc),
            )


    def export_selected_coordination_environment(
        self,
    ) -> None:
        """
        Export the selected central atom and its currently bonded neighbors
        as a Cartesian XYZ cluster.
        """
        if self.selected_atom_record is None:
            QtWidgets.QMessageBox.information(
                self,
                "No atom selected",
                "Select a central atom in the 3D viewer first.",
            )
            return

        central_record = self.selected_atom_record
        central_key = central_record.get(
            "key"
        )

        central_position = np.asarray(
            central_record[
                "position"
            ],
            dtype=float,
        )

        neighbor_rows = []

        for bond in self.bond_records:
            if bond.get(
                "center_key"
            ) == central_key:
                neighbor_rows.append(
                    {
                        "element": bond.get(
                            "neighbor_element",
                            "X",
                        ),
                        "label": bond.get(
                            "neighbor_label",
                            "",
                        ),
                        "position": np.asarray(
                            bond["end"],
                            dtype=float,
                        ),
                        "distance": float(
                            bond["distance"]
                        ),
                    }
                )

            elif bond.get(
                "neighbor_key"
            ) == central_key:
                displacement = (
                    np.asarray(
                        bond["start"],
                        dtype=float,
                    )
                    - np.asarray(
                        bond["end"],
                        dtype=float,
                    )
                )

                neighbor_rows.append(
                    {
                        "element": bond.get(
                            "center_element",
                            "X",
                        ),
                        "label": bond.get(
                            "center_label",
                            "",
                        ),
                        "position": (
                            central_position
                            + displacement
                        ),
                        "distance": float(
                            bond["distance"]
                        ),
                    }
                )

        if not neighbor_rows:
            QtWidgets.QMessageBox.information(
                self,
                "No coordination environment",
                "The selected atom has no bonds under the current bond settings.",
            )
            return

        unique_neighbors = []
        seen = set()

        for row in neighbor_rows:
            position = np.asarray(
                row["position"],
                dtype=float,
            )

            key = (
                str(
                    row["element"]
                ),
                round(
                    float(
                        position[0]
                    ),
                    8,
                ),
                round(
                    float(
                        position[1]
                    ),
                    8,
                ),
                round(
                    float(
                        position[2]
                    ),
                    8,
                ),
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            unique_neighbors.append(
                row
            )

        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export selected coordination environment",
            (
                f"{central_record.get('label', 'coordination')}"
                "_coordination.xyz"
            ),
            "XYZ files (*.xyz);;All files (*)",
        )

        if not path:
            return

        if not path.lower().endswith(
            ".xyz"
        ):
            path += ".xyz"

        try:
            with open(
                path,
                "w",
                encoding="utf-8",
            ) as output_file:
                output_file.write(
                    f"{1 + len(unique_neighbors)}\n"
                )

                output_file.write(
                    "Selected coordination environment; "
                    f"center={central_record.get('label', '')}; "
                    f"coordination_number={len(unique_neighbors)}\n"
                )

                output_file.write(
                    f"{central_record.get('element', 'X'):2s} "
                    f"{central_position[0]: .10f} "
                    f"{central_position[1]: .10f} "
                    f"{central_position[2]: .10f}\n"
                )

                for neighbor in unique_neighbors:
                    x, y, z = [
                        float(value)
                        for value in neighbor[
                            "position"
                        ]
                    ]

                    output_file.write(
                        f"{neighbor['element']:2s} "
                        f"{x: .10f} "
                        f"{y: .10f} "
                        f"{z: .10f}\n"
                    )

            csv_path = os.path.splitext(
                path
            )[0] + "_distances.csv"

            with open(
                csv_path,
                "w",
                encoding="utf-8",
            ) as output_file:
                output_file.write(
                    "center_label,center_element,"
                    "neighbor_label,neighbor_element,distance_A\n"
                )

                for neighbor in unique_neighbors:
                    output_file.write(
                        f"{central_record.get('label', '')},"
                        f"{central_record.get('element', 'X')},"
                        f"{neighbor.get('label', '')},"
                        f"{neighbor.get('element', 'X')},"
                        f"{float(neighbor['distance']):.10g}\n"
                    )

            self.statusBar().showMessage(
                "Selected coordination environment exported:\n"
                f"{path}"
            )

        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Coordination export failed",
                str(exc),
            )

                    
    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------
    def show_and_raise(self) -> None:
        self.show()
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(
        self,
        event,
    ) -> None:
        """
        Save the sidecar configuration and hide the viewer.

        The VTK/OpenGL context remains active for fast reopening.
        """
        try:
            if self.configuration_path:
                self.save_viewer_configuration(
                    path=self.configuration_path,
                    silent=True,
                )
        except Exception:
            pass

        event.ignore()
        self.hide()
        self.closed.emit()


    def shutdown(self) -> None:
        """
        Fully close VTK when the main application exits.
        """
        try:
            if self.configuration_path:
                self.save_viewer_configuration(
                    path=self.configuration_path,
                    silent=True,
                )
        except Exception:
            pass

        try:
            self.plotter.disable_picking()
        except Exception:
            pass

        try:
            self.plotter.disable_depth_peeling()
        except Exception:
            pass

        try:
            self.plotter.clear()
        except Exception:
            pass

        try:
            self.plotter.close()
        except Exception:
            pass

        try:
            self.deleteLater()
        except Exception:
            pass