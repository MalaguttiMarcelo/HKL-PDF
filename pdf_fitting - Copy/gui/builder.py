from __future__ import annotations

import os
import re
import tempfile
import itertools
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pdf_fitting.io_handler import read_input_file
from pdf_fitting.models.microstrain_utils import Cell as CFCell, required_terms_for_sg

from .defaults import DEFAULT_PARAMS
from .param_widgets import ParamTableWidget, DynParamTableWidget, LambdaListWidget, SiteParamTableWidget
from .cif_utils import read_cif_asu_sites, clean_el_symbol

from .crystallite_shape_dialog import CrystalliteShapeDialog

def lambda_safe_species_label(name: str) -> str:
    """
    Convert species labels to safe lambda parameter labels.

    Examples
    --------
    Fe0+ -> fe0plus
    Fe2+ -> fe2plus
    O2-  -> o2minus
    """
    s = str(name).strip()
    s = s.replace("+", "plus")
    s = s.replace("-", "minus")
    s = re.sub(r"[^A-Za-z0-9]+", "", s)
    return s.lower()


class InputBuilder(QtWidgets.QWidget):
    """Structured GUI editor to create and edit input files."""

    changed = QtCore.Signal()

    def _on_cf_tab_changed(self, idx: int) -> None:
        """Tab change handler.

        IMPORTANT: do *not* clear values when switching representation.
        Mutual exclusivity is enforced only when exporting to the raw input file.
        """
        self._emit_changed_if_not_blocked()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._block = False
        self._sym_enabled: Optional[set[str]] = None
        self._biso_label_overrides: Dict[str, str] = {}
        self._dyn_biso_keys: set[str] = set()
        self._dyn_use_flags: Dict[str, bool] = {}
        self._last_structure_path: str = ""
        self._site_table_max_rows = int(
            os.environ.get(
                "PDF_FITTING_SITE_TABLE_MAX_ROWS",
                "300",
            )
        )

        self._symmetry_analyzer_max_sites = int(
            os.environ.get(
                "PDF_FITTING_SYMMETRY_ANALYZER_MAX_SITES",
                "500",
            )
        )

        self._element_biso_elements: set[str] = set()

        self._site_records_for_gui: List[Dict[str, Any]] = []
        self._site_biso_keys: set[str] = set()
        self._site_use_flags: Dict[str, bool] = {}
        self._crystallite_shape_spec: Dict[str, Any] = {}

        # (Optional) host if you ever want to dock a plot. Not required for layout.
        self._rwp_widget: Optional[QtWidgets.QWidget] = None
        self._rwp_host = QtWidgets.QWidget()
        _rwp_lay = QtWidgets.QVBoxLayout(self._rwp_host)
        _rwp_lay.setContentsMargins(0, 0, 0, 0)

        # -------------------------
        # Files
        # -------------------------
        self.le_structure = QtWidgets.QLineEdit()
        self.le_grdata = QtWidgets.QLineEdit()
        btn_browse_structure = QtWidgets.QToolButton(text="…")
        btn_browse_gr = QtWidgets.QToolButton(text="…")

        def browse_into(line_edit: QtWidgets.QLineEdit, title: str, filt: str):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, "", filt)
            if path:
                line_edit.setText(path)
                # FIX (7): aggiorna/sincronizza la struttura solo se è cambiato il CIF
                if line_edit is self.le_structure:
                    self._on_structure_path_changed()
                self.changed.emit()

        btn_browse_structure.clicked.connect(
            lambda: browse_into(
                self.le_structure,
                "Select structure file",
                "Structure (*.cif *.vasp *.poscar *.xyz *.pdb);;All files (*)",
            )
        )
        btn_browse_gr.clicked.connect(
            lambda: browse_into(
                self.le_grdata,
                "Select G(r) data file",
                "Data (*.xy *.dat *.txt);;All files (*)",
            )
        )

        files_grid = QtWidgets.QGridLayout()
        files_grid.addWidget(QtWidgets.QLabel("Structure file:"), 0, 0)
        files_grid.addWidget(self.le_structure, 0, 1)
        files_grid.addWidget(btn_browse_structure, 0, 2)
        files_grid.addWidget(QtWidgets.QLabel("G(r) data file:"), 1, 0)
        files_grid.addWidget(self.le_grdata, 1, 1)
        files_grid.addWidget(btn_browse_gr, 1, 2)

        gb_files = QtWidgets.QGroupBox("Input files")
        gb_files.setLayout(files_grid)

        # -------------------------
        # Range
        # -------------------------
        range_validator = QtGui.QDoubleValidator(
            -1.0e12,
            1.0e12,
            12,
            self,
        )
        range_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)

        self.le_rmin = QtWidgets.QLineEdit("1.0")
        self.le_rmax = QtWidgets.QLineEdit("10.0")
        self.le_rext = QtWidgets.QLineEdit("1.2")

        for line_edit in (self.le_rmin, self.le_rmax, self.le_rext):
            line_edit.setValidator(range_validator)
            line_edit.setClearButtonEnabled(False)
            line_edit.setMinimumWidth(120)

        self.le_rmin.setToolTip("Minimum r (Å) used for Rwp.")
        self.le_rmax.setToolTip("Maximum r (Å) used for Rwp.")
        self.le_rext.setToolTip("Extension factor for pair generation / shells (unitless).")

        range_form = QtWidgets.QFormLayout()
        range_form.addRow("r_min (Å):", self.le_rmin)
        range_form.addRow("r_max (Å):", self.le_rmax)
        range_form.addRow("r_extension:", self.le_rext)

        gb_range = QtWidgets.QGroupBox("Refinement range")
        gb_range.setLayout(range_form)

        # -------------------------
        # Collapsible pages (triangles)
        # -------------------------
        self.toolbox = QtWidgets.QToolBox()
        self.toolbox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Refinement info (Input tab)
        self.chk_staged = QtWidgets.QCheckBox("staged_refinement")
        self.chk_staged.setToolTip("If enabled, refine parameters in stages.")
        self.chk_compute_pair_contrib = QtWidgets.QCheckBox("compute_pair_contributions_on_finish")
        self.chk_compute_pair_contrib.setChecked(False)
        self.chk_compute_pair_contrib.setToolTip(
            "If enabled, compute individual pair contribution curves at the end of refinement.\n\n"
            "These are used by the PDF plot 'Show Pairs' tool.\n\n"
            "This can be slow for structures with many pair types or when qmax termination is enabled.\n"
            "Disable this if you only need the final total G(r)."
        )

        self.chk_compute_warren = QtWidgets.QCheckBox("compute_warren_on_finish")
        self.chk_compute_warren.setChecked(False)
        self.chk_compute_warren.setToolTip(
            "If enabled, compute the Warren directional strain plot at the end of refinement.\n\n"
            "This is mainly useful for the Wilkens/dislocation microstrain model.\n\n"
            "The plot is limited to the fitted particle size d when available; otherwise it uses 5 Å.\n"
            "Disable this for faster final output."
        )

        self.chk_compute_local_trends = QtWidgets.QCheckBox("compute_local_trends_on_finish")
        self.chk_compute_local_trends.setChecked(False)
        self.chk_compute_local_trends.setToolTip(
            "If enabled, compute local dynamics trends at the end of refinement.\n\n"
            "This plots lambda coefficients and the delta1/r + delta2/r² continuation for each pair type.\n\n"
            "The plot is limited to particle size d when available; otherwise it uses 5 Å.\n"
            "Disable this for faster final output."
        )

        self.chk_export_finite_coordination = QtWidgets.QCheckBox("export_finite_coordination")
        self.chk_export_finite_coordination.setChecked(False)
        self.chk_export_finite_coordination.setToolTip(
            "If enabled during finite-shape fits, export a CSV table with the finite "
            "coordination per pair distance and crystallographic direction.\n\n"
            "This writes:\n"
            "[refinement]\n"
            "export_finite_coordination = true"
        )


        self.spin_rstep_final = QtWidgets.QDoubleSpinBox()
        self.spin_rstep_final.setDecimals(4)
        self.spin_rstep_final.setRange(1e-6, 1.0)
        self.spin_rstep_final.setValue(0.01)
        self.spin_rstep_final.setToolTip("Final refinement r-grid step (Å).")

        self.spin_max_nfev = QtWidgets.QSpinBox()
        self.spin_max_nfev.setRange(1, 100000)
        self.spin_max_nfev.setValue(60)

        self.spin_ftol = QtWidgets.QDoubleSpinBox()
        self.spin_xtol = QtWidgets.QDoubleSpinBox()
        self.spin_gtol = QtWidgets.QDoubleSpinBox()
        for sp in (self.spin_ftol, self.spin_xtol, self.spin_gtol):
            sp.setDecimals(8)
            sp.setRange(1e-12, 1.0)
            sp.setSingleStep(1e-4)
            sp.setValue(1e-3)

        self.spin_progress = QtWidgets.QDoubleSpinBox()
        self.spin_progress.setDecimals(2)
        self.spin_progress.setRange(0.05, 10.0)
        self.spin_progress.setSingleStep(0.1)
        self.spin_progress.setValue(0.5)

        ref_form = QtWidgets.QFormLayout()
        ref_form.addRow(self.chk_staged)

        ref_form.addRow(self.chk_compute_pair_contrib)
        ref_form.addRow(self.chk_compute_warren)
        ref_form.addRow(self.chk_compute_local_trends)
        ref_form.addRow(self.chk_export_finite_coordination)

        ref_form.addRow("refinement_r_step_final (Å):", self.spin_rstep_final)
        ref_form.addRow("max_nfev_final:", self.spin_max_nfev)
        ref_form.addRow("ftol:", self.spin_ftol)
        ref_form.addRow("xtol:", self.spin_xtol)
        ref_form.addRow("gtol:", self.spin_gtol)
        ref_form.addRow("progress_every_sec:", self.spin_progress)

        self.gb_refineinfo = QtWidgets.QGroupBox("Refinement information")
        self.gb_refineinfo.setLayout(ref_form)

        # -------------------------
        # Instrumental  (2) group toggle
        # -------------------------
        self.gb_instr = QtWidgets.QGroupBox("Instrumental parameters")
        self.gb_instr.setCheckable(True)
        self.gb_instr.setChecked(True)

        self.tbl_instr = ParamTableWidget()
        v_instr = QtWidgets.QVBoxLayout(self.gb_instr)
        v_instr.setContentsMargins(6, 6, 6, 6)
        v_instr.addWidget(QtWidgets.QLabel("Instrumental parameters (e.g. qdamp, qbroad)"))
        v_instr.addWidget(self.tbl_instr, 1)

        self.toolbox.addItem(self.gb_instr, "Instrumental")
        self.gb_instr.toggled.connect(self.tbl_instr.setEnabled)
        self.gb_instr.toggled.connect(lambda _=None: self._emit_changed_if_not_blocked())

        # -------------------------
        # Structural (come prima)
        # -------------------------
        self.tbl_struct = ParamTableWidget()
        w_struct = QtWidgets.QWidget()
        v_struct = QtWidgets.QVBoxLayout(w_struct)
        v_struct.setContentsMargins(6, 6, 6, 6)
        v_struct.addWidget(QtWidgets.QLabel("Scale + symmetry-constrained lattice parameters"))
        v_struct.addWidget(self.tbl_struct, 1)
        self.toolbox.addItem(w_struct, "Structural information")

        # -------------------------
        # Microstructural - FIXED LAYOUT
        # Crystallite size on top, microstrain below (scrolling)
        # -------------------------
        self.combo_micro_model = QtWidgets.QComboBox()
        self.combo_micro_model.addItems([
            "Isotropic microstrain (delta_g)",
            "Wilkens (dislocation)",
            "PAH",
        ])
        self.combo_micro_model.setToolTip("Choose microstructural model. Wilkens enables extra parameters.")
    
        # --- Crystallite Size Section (top) ---
        self.gb_size = QtWidgets.QGroupBox("Crystallite size")
        self.gb_size.setCheckable(True)
        self.gb_size.setChecked(True)
    
        # Size model selector (like microstrain model selector)
        self.combo_size_model = QtWidgets.QComboBox()
        self.combo_size_model.addItems([
            "Spherical model",
            "Cylinder/disk model",
            "Finite crystallite shape",
        ])
        self.combo_size_model.setToolTip("Choose crystallite size model.")
    
        # Spherical model parameters
        self.tbl_size = ParamTableWidget()
        w_spherical = QtWidgets.QWidget()
        v_spherical = QtWidgets.QVBoxLayout(w_spherical)
        v_spherical.setContentsMargins(0, 0, 0, 0)
        v_spherical.addWidget(
            QtWidgets.QLabel("Conventional spherical crystallite size model")
        )
        v_spherical.addWidget(self.tbl_size, 1)
    
        # Cylinder/disk model parameters
        self.tbl_cylinder_size = ParamTableWidget()
        w_cylinder = QtWidgets.QWidget()
        v_cylinder = QtWidgets.QVBoxLayout(w_cylinder)
        v_cylinder.setContentsMargins(0, 0, 0, 0)
        v_cylinder.addWidget(
            QtWidgets.QLabel(
                "Cylinder/disk common-volume model "
                "(HKL-dependent; inactive if diameter or thickness is 0)"
            )
        )
        v_cylinder.addWidget(self.tbl_cylinder_size, 1)
    
        # Finite shape tab
        w_finite_shape = QtWidgets.QWidget()
        v_finite_shape = QtWidgets.QVBoxLayout(w_finite_shape)
        v_finite_shape.setContentsMargins(6, 6, 6, 6)
    
        self.lbl_shape_summary = QtWidgets.QLabel("No finite crystallite shape defined.")
        self.lbl_shape_summary.setWordWrap(True)
    
        self.btn_build_crystallite_shape = QtWidgets.QPushButton("Build crystallite shape…")
        self.btn_build_crystallite_shape.setToolTip(
            "Open a window to build anisotropic finite crystallite shapes "
            "such as disks, cylinders, prisms, and ellipsoids."
        )
    
        v_finite_shape.addWidget(self.lbl_shape_summary)
        v_finite_shape.addWidget(self.btn_build_crystallite_shape)
        v_finite_shape.addStretch(1)
    
        # Stacked widget for size models (like microstrain)
        self.size_model_stack = QtWidgets.QStackedWidget()
        self.size_model_stack.addWidget(w_spherical)    # index 0
        self.size_model_stack.addWidget(w_cylinder)     # index 1
        self.size_model_stack.addWidget(w_finite_shape) # index 2
    
        v_size = QtWidgets.QVBoxLayout(self.gb_size)
        v_size.setContentsMargins(6, 6, 6, 6)
        v_size.setSpacing(4)
        
        # Size model selector row
        row_size_model = QtWidgets.QHBoxLayout()
        row_size_model.addWidget(QtWidgets.QLabel("Size model:"))
        row_size_model.addWidget(self.combo_size_model, 1)
        v_size.addLayout(row_size_model)
        
        v_size.addWidget(self.size_model_stack, 1)
    
        self.btn_build_crystallite_shape.clicked.connect(self.open_crystallite_shape_dialog)
        self.combo_size_model.currentIndexChanged.connect(self._on_size_model_changed)
        self.gb_size.toggled.connect(self.size_model_stack.setEnabled)
        self.gb_size.toggled.connect(lambda _=None: self._emit_changed_if_not_blocked())

        # -------------------------
        # SAXS / missing-low-Q belly term
        # -------------------------
        self.gb_saxs = QtWidgets.QGroupBox("SAXS / missing-low-Q shape belly")
        self.gb_saxs.setCheckable(True)
        self.gb_saxs.setChecked(False)

        self.combo_saxs_model = QtWidgets.QComboBox()
        self.combo_saxs_model.addItems([
            "sphere",
            "cylinder",
            "disk",
            "same as size model",
        ])
        self.combo_saxs_model.setToolTip(
            "Model used only for the smooth negative SAXS/missing-low-Q belly term.\n\n"
            "sphere: use saxs_diameter and saxs_diameter_std.\n"
            "cylinder/disk: use saxs_diameter, saxs_height and their std values.\n"
            "same as size model: use the same average gamma as the main crystallite model."
        )

        self.chk_saxs_apply_qdamp = QtWidgets.QCheckBox("Apply qdamp to SAXS belly")
        self.chk_saxs_apply_qdamp.setChecked(False)
        self.chk_saxs_apply_qdamp.setToolTip(
            "Usually false. qdamp damps structural PDF correlations and may suppress the smooth SAXS belly too strongly."
        )

        self.chk_saxs_apply_qmax = QtWidgets.QCheckBox("Apply qmax termination to SAXS belly")
        self.chk_saxs_apply_qmax.setChecked(True)
        self.chk_saxs_apply_qmax.setToolTip(
            "Usually true. The SAXS belly is part of the Fourier-transformed PDF and should normally see the same finite-Qmax window."
        )

        self.tbl_saxs = ParamTableWidget()

        v_saxs = QtWidgets.QVBoxLayout(self.gb_saxs)
        v_saxs.setContentsMargins(6, 6, 6, 6)

        row_saxs_model = QtWidgets.QHBoxLayout()
        row_saxs_model.addWidget(QtWidgets.QLabel("SAXS belly model:"))
        row_saxs_model.addWidget(self.combo_saxs_model, 1)

        v_saxs.addLayout(row_saxs_model)
        v_saxs.addWidget(self.chk_saxs_apply_qdamp)
        v_saxs.addWidget(self.chk_saxs_apply_qmax)
        v_saxs.addWidget(
            QtWidgets.QLabel(
                "These parameters control only the smooth negative shape/SAXS belly term, "
                "not the PDF peak envelope."
            )
        )
        v_saxs.addWidget(self.tbl_saxs, 1)

        self.gb_saxs.toggled.connect(self.tbl_saxs.setEnabled)
        self.gb_saxs.toggled.connect(lambda _=None: self._emit_changed_if_not_blocked())
        self.combo_saxs_model.currentIndexChanged.connect(lambda _=None: self._emit_changed_if_not_blocked())
        self.chk_saxs_apply_qdamp.toggled.connect(lambda _=None: self._emit_changed_if_not_blocked())
        self.chk_saxs_apply_qmax.toggled.connect(lambda _=None: self._emit_changed_if_not_blocked())

            
        # --- "Build crystallite shape" button just before microstrain ---
        # (This is now inside the finite shape tab, but we also want it visible
        #  when the user scrolls down. The button is already in the finite shape tab.)
    
        # --- Microstrain Section (below size, scrollable) ---
        self.gb_strain = QtWidgets.QGroupBox("Isotropic microstrain (delta_g)")
        self.gb_strain.setCheckable(True)
        self.gb_strain.setChecked(True)
        self.tbl_strain = ParamTableWidget()
        v_iso = QtWidgets.QVBoxLayout(self.gb_strain)
        v_iso.setContentsMargins(0, 0, 0, 0)
        v_iso.addWidget(self.tbl_strain, 1)
        self.gb_strain.toggled.connect(self.tbl_strain.setEnabled)
        self.gb_strain.toggled.connect(lambda _=None: self._emit_changed_if_not_blocked())
    
        self.gb_wilkens = QtWidgets.QGroupBox("Wilkens model parameters")
        self.gb_wilkens.setCheckable(True)
        self.gb_wilkens.setChecked(False)
        self.tbl_wilkens = ParamTableWidget()
        v_wilk = QtWidgets.QVBoxLayout(self.gb_wilkens)
        v_wilk.setContentsMargins(0, 0, 0, 0)
        v_wilk.addWidget(self.tbl_wilkens, 1)
        self.gb_wilkens.toggled.connect(self.tbl_wilkens.setEnabled)
        self.gb_wilkens.toggled.connect(lambda _=None: self._emit_changed_if_not_blocked())
    
        self.gb_pah = QtWidgets.QGroupBox("PAH model parameters")
        self.gb_pah.setCheckable(True)
        self.gb_pah.setChecked(False)
    
        self.tbl_pah = ParamTableWidget()
        
    
        v_pah = QtWidgets.QVBoxLayout(self.gb_pah)
        v_pah.setContentsMargins(0, 0, 0, 0)
        v_pah.addWidget(self.tbl_pah, 1)
    
        self.gb_pah.toggled.connect(self.tbl_pah.setEnabled)
        self.gb_pah.toggled.connect(lambda _=None: self._emit_changed_if_not_blocked())
    
    
        # -------------------------
        # Contrast factor invariants (Ei) for edge/screw
        # -------------------------
        # Separate checkbox controls whether the contrast-factor panel is shown/exported.
        # This avoids wasting space when invariants are not selected.
        self.chk_cf = QtWidgets.QCheckBox("Show/use contrast factors")
        self.chk_cf.setChecked(False)
        self.chk_cf.setToolTip(
            "Enable this only when using Wilkens/dislocation microstrain and contrast factors are required."
        )
    
        self.gb_cf = QtWidgets.QGroupBox("Contrast factor invariants (Ei)")
        self.gb_cf.setCheckable(False)
        self.gb_cf.setVisible(False)
    
        self._cf_terms: List[str] = []
        self.tbl_cf_edge = ParamTableWidget()
        self.tbl_cf_screw = ParamTableWidget()
    
        self.tbl_cf_ab = ParamTableWidget()
    
        # Cubic A/B coefficients are now normal refinable parameters.
        # Therefore Min/Max/Refine columns must remain visible.
    
        self._tabs_cf = QtWidgets.QTabWidget()
        self._cf_tab_ab_index = None
        self._tabs_cf.addTab(self.tbl_cf_edge, "Edge")
        self._tabs_cf.addTab(self.tbl_cf_screw, "Screw")
        self._cf_tab_ab_index = self._tabs_cf.addTab(self.tbl_cf_ab, "Cubic A/B")
    
        # Track representation selection (no auto-mutation of values)
        self._tabs_cf.currentChanged.connect(self._on_cf_tab_changed)
    
        _v_cf = QtWidgets.QVBoxLayout(self.gb_cf)
        _v_cf.setContentsMargins(6, 6, 6, 6)
        _v_cf.addWidget(QtWidgets.QLabel("Invariant coefficients Ei for Chkl expansion (read from CIF symmetry)."))
        _v_cf.addWidget(self._tabs_cf, 1)
    
        self.chk_cf.toggled.connect(self._on_cf_visibility_changed)
    
        # --- Microstructural widget with vertical layout: size on top, strain below ---
        w_micro = QtWidgets.QWidget()
        v_micro = QtWidgets.QVBoxLayout(w_micro)
        v_micro.setContentsMargins(6, 6, 6, 6)
        v_micro.setSpacing(6)

        # --- Microstructural widget with vertical layout: size on top, strain below ---
        w_micro = QtWidgets.QWidget()
        v_micro = QtWidgets.QVBoxLayout(w_micro)
        v_micro.setContentsMargins(6, 6, 6, 6)
        v_micro.setSpacing(6)
    
        # Size section first (with its own model selector inside the group box)
        v_micro.addWidget(self.gb_size)
        v_micro.addWidget(self.gb_saxs)
    
        # Microstrain model selector (below size)
        row_micro = QtWidgets.QHBoxLayout()
        row_micro.addWidget(QtWidgets.QLabel("Microstrain model:"))
        row_micro.addWidget(self.combo_micro_model, 1)
        v_micro.addLayout(row_micro)
    
        # Stacked widget for microstrain models
        self.micro_right_stack = QtWidgets.QStackedWidget()
        self.micro_right_stack.addWidget(self.gb_strain)    # index 0
        self.micro_right_stack.addWidget(self.gb_wilkens)   # index 1
        self.micro_right_stack.addWidget(self.gb_pah)       # index 2
    
        v_micro.addWidget(self.micro_right_stack, 1)
    
        # Contrast factors at the bottom
        v_micro.addWidget(self.chk_cf)
        v_micro.addWidget(self.gb_cf)
    
        # Set minimum heights for the tables
        for _tbl in (
            self.tbl_size,
            self.tbl_cylinder_size,
            self.tbl_strain,
            self.tbl_wilkens,
            self.tbl_pah,
        ):
            _tbl.setMinimumHeight(105)
            _tbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    
        self.toolbox.addItem(w_micro, "Microstructural")        
        # -------------------------
        # Local dynamics / Other
        # Lambda coefficients placed BELOW Debye-Waller/local dynamics table
        # -------------------------
        self.tbl_sites = SiteParamTableWidget()

        # Element-grouped Biso table.
        # Recommended for large/low-symmetry molecular structures.
        self.chk_biso_by_element = QtWidgets.QCheckBox(
            "Use one Biso per element type"
        )
        self.chk_biso_by_element.setChecked(True)
        self.chk_biso_by_element.setToolTip(
            "Recommended for large structures.\n\n"
            "When enabled, Biso is refined as one parameter per element:\n"
            "  biso_C, biso_N, biso_Co, ...\n\n"
            "Site-specific Biso refinement is disabled to avoid refining many redundant "
            "highly-correlated parameters."
        )

        self.tbl_elem_biso = ParamTableWidget()
        self.tbl_elem_biso.setMinimumHeight(115)

        self.tbl_dyn = DynParamTableWidget()
        self.lambda_list = LambdaListWidget()



        w_dyn = QtWidgets.QWidget()
        v_dyn = QtWidgets.QVBoxLayout(w_dyn)
        v_dyn.setContentsMargins(6, 6, 6, 6)
        v_dyn.setSpacing(6)

        split_dyn = QtWidgets.QSplitter(Qt.Vertical)

        # Top: atomic site parameters + pair dynamics table
        w_top_dyn = QtWidgets.QWidget()
        v_top_dyn = QtWidgets.QVBoxLayout(w_top_dyn)
        v_top_dyn.setContentsMargins(0, 0, 0, 0)
        v_top_dyn.setSpacing(4)

        # ---------------------------------------------------------
        # Atomic site panel
        # ---------------------------------------------------------
        w_site_panel = QtWidgets.QWidget()
        v_site_panel = QtWidgets.QVBoxLayout(w_site_panel)
        v_site_panel.setContentsMargins(0, 0, 0, 0)
        v_site_panel.setSpacing(4)

        row_sites_header = QtWidgets.QHBoxLayout()

        lbl_sites = QtWidgets.QLabel("Atomic site parameters (x, y, z, occupancy, Biso)")
        lbl_sites.setStyleSheet("font-weight: bold;")

        self.btn_site_params_window = QtWidgets.QPushButton("Open atomic site window")
        self.btn_site_params_window.setToolTip(
            "Open the atomic-site x/y/z/occupancy/Biso table in a larger window."
        )

        row_sites_header.addWidget(lbl_sites)
        row_sites_header.addStretch(1)
        row_sites_header.addWidget(self.btn_site_params_window)

        v_site_panel.addLayout(row_sites_header)
        v_site_panel.addWidget(self.tbl_sites, 1)

        # Element-grouped Biso controls below site table.
        gb_elem_biso = QtWidgets.QGroupBox("Element-grouped Biso parameters")
        v_elem_biso = QtWidgets.QVBoxLayout(gb_elem_biso)
        v_elem_biso.setContentsMargins(6, 6, 6, 6)
        v_elem_biso.addWidget(self.chk_biso_by_element)
        v_elem_biso.addWidget(self.tbl_elem_biso, 1)

        v_site_panel.addWidget(gb_elem_biso)

        # ---------------------------------------------------------
        # Pair dynamics panel
        # ---------------------------------------------------------
        w_pair_dyn_panel = QtWidgets.QWidget()
        v_pair_dyn_panel = QtWidgets.QVBoxLayout(w_pair_dyn_panel)
        v_pair_dyn_panel.setContentsMargins(0, 0, 0, 0)
        v_pair_dyn_panel.setSpacing(4)

        row_pair_header = QtWidgets.QHBoxLayout()

        lbl_dyn = QtWidgets.QLabel("Pair dynamics parameters (delta1, delta2)")
        lbl_dyn.setStyleSheet("font-weight: bold;")

        self.btn_pair_dynamics_window = QtWidgets.QPushButton("Open pair dynamics window")
        self.btn_pair_dynamics_window.setToolTip(
            "Open pair dynamics and lambda coefficients in a larger window."
        )

        row_pair_header.addWidget(lbl_dyn)
        row_pair_header.addStretch(1)
        row_pair_header.addWidget(self.btn_pair_dynamics_window)

        v_pair_dyn_panel.addLayout(row_pair_header)
        v_pair_dyn_panel.addWidget(self.tbl_dyn, 1)

        # ---------------------------------------------------------
        # Splitter between atomic sites and pair dynamics
        # ---------------------------------------------------------
        self.site_pair_split = QtWidgets.QSplitter(Qt.Vertical)
        self.site_pair_split.addWidget(w_site_panel)
        self.site_pair_split.addWidget(w_pair_dyn_panel)
        self.site_pair_split.setChildrenCollapsible(False)
        self.site_pair_split.setHandleWidth(8)

        try:
            self.site_pair_split.setStretchFactor(0, 3)
            self.site_pair_split.setStretchFactor(1, 2)
            self.site_pair_split.setSizes([360, 230])
        except Exception:
            pass
        
        v_top_dyn.addWidget(self.site_pair_split, 1)

        # Bottom: lambda coefficients
        w_bottom_lambda = QtWidgets.QWidget()
        v_bottom_lambda = QtWidgets.QVBoxLayout(w_bottom_lambda)
        v_bottom_lambda.setContentsMargins(0, 0, 0, 0)
        v_bottom_lambda.setSpacing(4)


        lbl_lambda = QtWidgets.QLabel("Lambda coefficients (add/remove as needed)")
        lbl_lambda.setStyleSheet("font-weight: bold;")
        v_bottom_lambda.addWidget(lbl_lambda)
        v_bottom_lambda.addWidget(self.lambda_list, 1)

        split_dyn.addWidget(w_top_dyn)
        split_dyn.addWidget(w_bottom_lambda)

        split_dyn.setStretchFactor(0, 3)
        split_dyn.setStretchFactor(1, 1)
        split_dyn.setHandleWidth(8)
        split_dyn.setChildrenCollapsible(False)

        self.tbl_sites.setMinimumHeight(220)
        self.tbl_dyn.setMinimumHeight(180)
        self.lambda_list.setMinimumHeight(150)

        try:
            split_dyn.setSizes([560, 180])
        except Exception:
            pass

        v_dyn.addWidget(split_dyn, 1)
        self.toolbox.addItem(w_dyn, "Local dynamics")

        # -------------------------
        # Layout: Input/Refinement vs Parameters (same position; switch tabs)
        # -------------------------
        self.main_tabs = QtWidgets.QTabWidget()
        self.main_tabs.setDocumentMode(True)

        w_input = QtWidgets.QWidget()
        v_input = QtWidgets.QVBoxLayout(w_input)
        v_input.setContentsMargins(0, 0, 0, 0)
        v_input.addWidget(gb_files)
        v_input.addWidget(gb_range)
        v_input.addWidget(self.gb_refineinfo)
        v_input.addStretch(1)

        w_params = QtWidgets.QWidget()
        v_params = QtWidgets.QVBoxLayout(w_params)
        v_params.setContentsMargins(0, 0, 0, 0)
        v_params.addWidget(self.toolbox, 1)


        # -------------------------
        # Constraints ([constraints])
        # -------------------------
        w_constraints = QtWidgets.QWidget()
        v_constraints = QtWidgets.QVBoxLayout(w_constraints)
        v_constraints.setContentsMargins(6, 6, 6, 6)

        gb_constraints = QtWidgets.QGroupBox("Constraints")
        gb_constraints.setToolTip("Define parameter relationships evaluated during refinement.\nExample: Re = d")
        v_gbc = QtWidgets.QVBoxLayout(gb_constraints)

        self.tbl_constraints = QtWidgets.QTableWidget(0, 2)
        self.tbl_constraints.setHorizontalHeaderLabels(["Parameter", "Expression"])
        self.tbl_constraints.horizontalHeader().setStretchLastSection(True)
        self.tbl_constraints.verticalHeader().setVisible(False)
        self.tbl_constraints.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_constraints.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed | QtWidgets.QAbstractItemView.AnyKeyPressed)

        btn_add_c = QtWidgets.QPushButton("Add")
        btn_del_c = QtWidgets.QPushButton("Remove")

        def _add_constraint_row():
            r = self.tbl_constraints.rowCount()
            self.tbl_constraints.insertRow(r)
            self.tbl_constraints.setItem(r, 0, QtWidgets.QTableWidgetItem(""))
            self.tbl_constraints.setItem(r, 1, QtWidgets.QTableWidgetItem(""))
            self._emit_changed_if_not_blocked()

        def _del_constraint_row():
            rows = sorted({i.row() for i in self.tbl_constraints.selectedIndexes()}, reverse=True)
            if not rows:
                return
            for r in rows:
                self.tbl_constraints.removeRow(r)
            self._emit_changed_if_not_blocked()

        btn_add_c.clicked.connect(_add_constraint_row)
        btn_del_c.clicked.connect(_del_constraint_row)
        self.tbl_constraints.itemChanged.connect(lambda _=None: self._emit_changed_if_not_blocked())

        hbtn = QtWidgets.QHBoxLayout()
        hbtn.addWidget(btn_add_c)
        hbtn.addWidget(btn_del_c)
        hbtn.addStretch(1)

        v_gbc.addLayout(hbtn)
        v_gbc.addWidget(self.tbl_constraints, 1)

        v_constraints.addWidget(gb_constraints, 1)

        self.main_tabs.addTab(w_input, "Input & refinement")
        self.main_tabs.addTab(w_params, "Parameters")
        self.main_tabs.addTab(w_constraints, "Constraints")

        top = QtWidgets.QVBoxLayout(self)
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.main_tabs, 1)

        # Change signals
        self.le_structure.textEdited.connect(lambda _=None: self._on_structure_path_changed())
        self.le_structure.textEdited.connect(lambda _=None: self._emit_changed_if_not_blocked())
        self.le_grdata.textEdited.connect(lambda _=None: self._emit_changed_if_not_blocked())
        self.le_rmin.editingFinished.connect(self._emit_changed_if_not_blocked)
        self.le_rmax.editingFinished.connect(self._emit_changed_if_not_blocked)
        self.le_rext.editingFinished.connect(self._emit_changed_if_not_blocked)

        for w in (
            self.chk_staged,
            self.chk_compute_pair_contrib,
            self.chk_compute_warren,
            self.chk_compute_local_trends,
            self.chk_export_finite_coordination,
            self.spin_rstep_final,
            self.spin_max_nfev,
            self.spin_ftol,
            self.spin_xtol,
            self.spin_gtol,
            self.spin_progress,
        ):
            if isinstance(w, QtWidgets.QAbstractButton):
                w.toggled.connect(lambda _=None: self._emit_changed_if_not_blocked())
            else:
                w.valueChanged.connect(lambda _=None: self._emit_changed_if_not_blocked())

        self.tbl_struct.changed.connect(self._emit_changed_if_not_blocked)
        self.tbl_instr.changed.connect(self._emit_changed_if_not_blocked)
        self.tbl_size.changed.connect(self._emit_changed_if_not_blocked)
        self.tbl_cylinder_size.changed.connect(self._emit_changed_if_not_blocked)
        self.tbl_strain.changed.connect(self._emit_changed_if_not_blocked)
        self.tbl_saxs.changed.connect(self._emit_changed_if_not_blocked)
        self.tbl_wilkens.changed.connect(self._emit_changed_if_not_blocked)
        self.tbl_pah.changed.connect(self._emit_changed_if_not_blocked)
        self.tbl_dyn.changed.connect(self._emit_changed_if_not_blocked)
        self.tbl_sites.changed.connect(self._emit_changed_if_not_blocked)
        self.tbl_elem_biso.changed.connect(self._emit_changed_if_not_blocked)
        self.chk_biso_by_element.toggled.connect(self._on_biso_mode_changed)
        self.chk_biso_by_element.toggled.connect(lambda _=None: self._emit_changed_if_not_blocked())
        self.btn_site_params_window.clicked.connect(self.open_site_parameters_dialog)
        self.btn_pair_dynamics_window.clicked.connect(self.open_pair_dynamics_dialog)
        self.lambda_list.changed.connect(self._emit_changed_if_not_blocked)
        self.tbl_cf_edge.changed.connect(self._emit_changed_if_not_blocked)
        self.tbl_cf_screw.changed.connect(self._emit_changed_if_not_blocked)
        self.tbl_cf_ab.changed.connect(self._emit_changed_if_not_blocked)

        self.combo_micro_model.currentIndexChanged.connect(self._on_micro_model_changed)
        self._on_micro_model_changed()

        self._seed_defaults()


    def _on_size_model_changed(self, idx: int) -> None:
        """
        Switch between spherical, cylinder/disk, and finite shape models.
        
        Only the selected model's parameters are visible.
        """
        try:
            self.size_model_stack.setCurrentIndex(idx)
            
            # Update the shape summary label when switching to finite shape
            if idx == 2:
                self._update_shape_summary_label()
            
            self._emit_changed_if_not_blocked()
            
        except Exception:
            pass    

    def _current_lattice_params_for_shape_dialog(self) -> Dict[str, float]:
        """
        Get current lattice parameters from structural table.

        Used only for shape preview.
        """
        out = {
            "a": 1.0,
            "b": 1.0,
            "c": 1.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        }

        try:
            ini, _ref, _bnd = self.tbl_struct.extract()
            for k in out:
                if k in ini:
                    out[k] = float(ini[k])
        except Exception:
            pass

        return out


    def _update_shape_summary_label(self) -> None:
        spec = getattr(self, "_crystallite_shape_spec", {}) or {}

        if not spec:
            self.lbl_shape_summary.setText("No finite crystallite shape defined.")
            return

        search_mode = str(spec.get("search_mode", "grid_search")).strip().lower()

        if search_mode == "predefined":
            txt = (
                f"Mode: pre-defined crystallite dimensions\n"
                f"Shape: {spec.get('shape_type', 'unknown')}\n"
                f"Diameter: {spec.get('diameter_cells', '?')} cells\n"
                f"Height: {spec.get('height_cells', '?')} cells\n"
                f"Axis: [{spec.get('axis_h', 0)} "
                f"{spec.get('axis_k', 0)} "
                f"{spec.get('axis_l', 1)}]"
            )
        else:
            txt = (
                f"Mode: grid search\n"
                f"Shape: {spec.get('shape_type', 'unknown')}\n"
                f"Diameter scan: {spec.get('diameter_min_cells', '?')} "
                f"to {spec.get('diameter_max_cells', '?')} cells\n"
                f"Height scan: {spec.get('height_min_cells', '?')} "
                f"to {spec.get('height_max_cells', '?')} cells\n"
                f"Step: {spec.get('step_cells', '?')} cells\n"
                f"Axis: [{spec.get('axis_h', 0)} "
                f"{spec.get('axis_k', 0)} "
                f"{spec.get('axis_l', 1)}]"
            )

        self.lbl_shape_summary.setText(txt)


    def _current_structure_for_shape_dialog(self):
        """
        Read current structure for crystallite-shape preview.

        This is only for visualization, not refinement.
        """
        path = str(self.le_structure.text() or "").strip()

        if not path or not os.path.exists(path):
            return None

        try:
            if path.lower().endswith(".cif"):
                s, _asu = read_cif_asu_sites(path)
                return s
            return Structure.from_file(path)
        except Exception:
            return None


    def open_crystallite_shape_dialog(self) -> None:
        structure = self._current_structure_for_shape_dialog()

        dlg = CrystalliteShapeDialog(
            self,
            initial_spec=getattr(self, "_crystallite_shape_spec", {}) or {},
            lattice_params=self._current_lattice_params_for_shape_dialog(),
            structure=structure,  # NEW
        )

        def accept_shape(spec: dict):
            spec = dict(spec or {})
            spec["mode"] = "finite_shape"

            self._crystallite_shape_spec = spec

            # Important: force GUI into finite-shape mode.
            try:
                self.gb_size.setChecked(True)
            except Exception:
                pass
            
            try:
                self.combo_size_model.setCurrentIndex(2)
            except Exception:
                pass
            
            self._update_shape_summary_label()
            self._emit_changed_if_not_blocked()
            
        dlg.shapeAccepted.connect(accept_shape)

        QtCore.QTimer.singleShot(0, dlg.showMaximized)
        dlg.exec()

        try:
            mw = self.window()
            if hasattr(mw, "_update_run_button_modes"):
                mw._update_run_button_modes()
        except Exception:
            pass


    # -------------------------
    # Contrast factor Ei helpers
    # -------------------------
    def _cf_set_terms(
        self,
        terms: List[str],
        *,
        edge_seed: Optional[Dict[str, Any]] = None,
        screw_seed: Optional[Dict[str, Any]] = None,
        refinable_seed: Optional[Dict[str, Any]] = None,
        bounds_seed: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Rebuild contrast-factor tables.

        Cubic systems:
            use CEdgeA, CEdgeB, CScrewA, CScrewB in the Cubic A/B tab.

        Non-cubic systems:
            use EdgeE1...EdgeEn and ScrewE1...ScrewEn.
        """
        self._cf_terms = list(terms or [])
        edge_seed = edge_seed or {}
        screw_seed = screw_seed or {}
        refinable_seed = refinable_seed or {}
        bounds_seed = bounds_seed or {}

        is_cubic_terms = (
            len(self._cf_terms) == 2
            and [str(t).upper() for t in self._cf_terms] == ["E1", "E2"]
        )

        # -------------------------
        # Invariant Edge/Screw tables
        # -------------------------
        edge_keys = [f"Edge{t}" for t in self._cf_terms]
        screw_keys = [f"Screw{t}" for t in self._cf_terms]

        self.tbl_cf_edge.set_params(
            edge_keys,
            edge_seed,
            refinable_seed,
            bounds_seed,
            tooltips={k: "Refinable invariant edge contrast-factor coefficient" for k in edge_keys},
        )

        self.tbl_cf_screw.set_params(
            screw_keys,
            screw_seed,
            refinable_seed,
            bounds_seed,
            tooltips={k: "Refinable invariant screw contrast-factor coefficient" for k in screw_keys},
        )

        # -------------------------
        # Cubic A/B table
        # -------------------------
        try:
            ab_keys = ["CEdgeA", "CEdgeB", "CScrewA", "CScrewB"]

            # Preserve existing values if the user already edited them.
            old_ini, old_ref, old_bnd = self.tbl_cf_ab.extract()

            def _old_val(new_key: str, old_key: str, default: float):
                return old_ini.get(new_key, old_ini.get(old_key, default))

            def _old_ref(new_key: str, old_key: str, default: bool = False):
                return bool(old_ref.get(new_key, old_ref.get(old_key, default)))

            def _old_bnd(new_key: str, old_key: str):
                return old_bnd.get(new_key, old_bnd.get(old_key, ("", "")))

            ab_seed = {
                "CEdgeA": _old_val("CEdgeA", "cedgea", 0.265280),
                "CEdgeB": _old_val("CEdgeB", "cedgeb", -0.355950),
                "CScrewA": _old_val("CScrewA", "cscrewa", 0.307288),
                "CScrewB": _old_val("CScrewB", "cscrewb", -0.819979),
            }

            ab_ref = {
                "CEdgeA": _old_ref("CEdgeA", "cedgea"),
                "CEdgeB": _old_ref("CEdgeB", "cedgeb"),
                "CScrewA": _old_ref("CScrewA", "cscrewa"),
                "CScrewB": _old_ref("CScrewB", "cscrewb"),
            }

            ab_bnd = {
                "CEdgeA": _old_bnd("CEdgeA", "cedgea"),
                "CEdgeB": _old_bnd("CEdgeB", "cedgeb"),
                "CScrewA": _old_bnd("CScrewA", "cscrewa"),
                "CScrewB": _old_bnd("CScrewB", "cscrewb"),
            }

            self.tbl_cf_ab.set_params(
                ab_keys,
                ab_seed,
                ab_ref,
                ab_bnd,
                tooltips={
                    "CEdgeA": "Cubic edge contrast-factor A coefficient.",
                    "CEdgeB": "Cubic edge contrast-factor B coefficient.",
                    "CScrewA": "Cubic screw contrast-factor A coefficient.",
                    "CScrewB": "Cubic screw contrast-factor B coefficient.",
                },
            )
        except Exception:
            pass

        # -------------------------
        # Enable/default tab
        # -------------------------
        try:
            if getattr(self, "_cf_tab_ab_index", None) is not None:
                self._tabs_cf.setTabEnabled(int(self._cf_tab_ab_index), bool(is_cubic_terms))

            if is_cubic_terms and getattr(self, "_cf_tab_ab_index", None) is not None:
                self._tabs_cf.setCurrentIndex(int(self._cf_tab_ab_index))
            else:
                self._tabs_cf.setCurrentIndex(0)
        except Exception:
            pass

        if is_cubic_terms:
            self.gb_cf.setTitle("Cubic contrast factors: A/B coefficients")
        elif self._cf_terms:
            self.gb_cf.setTitle(f"Contrast factor invariants (Ei): {len(self._cf_terms)} terms")
        else:
            self.gb_cf.setTitle("Contrast factor invariants (Ei)")


    def _cf_extract(self) -> Dict[str, float]:
        """Extract contrast-factor coefficients as a flat dict.

        UI rule:
        - If the user is on the 'Cubic A/B' tab, export ONLY legacy A/B keys.
        - Otherwise export ONLY the Ei coefficients (Edge*/Screw*).
        """
        out: Dict[str, float] = {}

        # Decide active CF representation from current tab
        try:
            use_ab = (self._tabs_cf.currentIndex() == int(self._cf_tab_ab_index))
        except Exception:
            use_ab = False

        tables = (self.tbl_cf_ab,) if use_ab else (self.tbl_cf_edge, self.tbl_cf_screw)

        for tbl in tables:
            ini, _ref, _bnd = tbl.extract()
            for k, v in (ini or {}).items():
                kk = str(k).strip()
                if not kk:
                    continue
                try:
                    out[kk] = float(v)
                except Exception:
                    continue
        return out

    @staticmethod
    def _cf_split(coeffs: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Accept io_handler's structured coeffs or a flat dict and return (edge_E, screw_E) with keys E1.."""
        edge_E: Dict[str, float] = {}
        screw_E: Dict[str, float] = {}
        if not isinstance(coeffs, dict):
            return edge_E, screw_E

        # Structured: {'edge_E': {'E1':..}, 'screw_E': {...}}
        if isinstance(coeffs.get("edge_E"), dict):
            for k, v in coeffs.get("edge_E", {}).items():
                try:
                    edge_E[str(k).upper()] = float(v)
                except Exception:
                    pass
        if isinstance(coeffs.get("screw_E"), dict):
            for k, v in coeffs.get("screw_E", {}).items():
                try:
                    screw_E[str(k).upper()] = float(v)
                except Exception:
                    pass

        # Flat: EdgeE1 / ScrewE1
        for k, v in coeffs.items():
            ks = str(k).strip()
            m = re.match(r"(?i)^(edge|screw)E(\d+)$", ks)
            if not m:
                continue
            kind = m.group(1).lower()
            en = f"E{int(m.group(2))}"
            try:
                if kind == "edge":
                    edge_E[en] = float(v)
                else:
                    screw_E[en] = float(v)
            except Exception:
                pass
        return edge_E, screw_E

    def set_rwp_widget(self, widget: QtWidgets.QWidget) -> None:
        """Optional: dock a widget into _rwp_host (NOT used in this final layout)."""
        if widget is None:
            return
        if self._rwp_widget is widget:
            return
        try:
            lay = self._rwp_host.layout()
            while lay.count():
                it = lay.takeAt(0)
                w = it.widget()
                if w is not None:
                    w.setParent(None)
        except Exception:
            pass
        try:
            widget.setParent(self._rwp_host)
            self._rwp_host.layout().addWidget(widget)
            self._rwp_widget = widget
        except Exception:
            pass
    
    
    def open_site_parameters_dialog(self) -> None:
        """
        Open atomic-site parameters in a larger dialog.

        The dialog uses a copy of the site table. When Apply is clicked,
        values/refine flags/use flags are copied back to the main Builder table.
        """
        try:
            sites = list(getattr(self, "_site_records_for_gui", []) or [])

            if not sites:
                QtWidgets.QMessageBox.information(
                    self,
                    "No atomic sites",
                    "No atomic-site information is available yet. Load a structure file first.",
                )
                return

            # Capture current table values.
            site_ini, site_ref, site_bnd, site_use = self.tbl_sites.extract_all()

            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Atomic site parameters")
            dlg.setWindowFlags(
                dlg.windowFlags()
                | Qt.Window
                | Qt.WindowMinimizeButtonHint
                | Qt.WindowMaximizeButtonHint
                | Qt.WindowCloseButtonHint
            )

            screen = QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                geo = screen.availableGeometry()
                dlg.resize(
                    int(geo.width() * 0.80),
                    int(geo.height() * 0.70),
                )
                dlg.move(
                    geo.center().x() - dlg.width() // 2,
                    geo.center().y() - dlg.height() // 2,
                )
            else:
                dlg.resize(1000, 650)

            tbl = SiteParamTableWidget()
            tbl.set_sites(
                sites,
                initial=site_ini,
                refinable=site_ref,
                bounds=site_bnd,
                use_flags=site_use,
            )

            buttons = QtWidgets.QHBoxLayout()
            btn_apply = QtWidgets.QPushButton("Apply")
            btn_close = QtWidgets.QPushButton("Close")

            buttons.addStretch(1)
            buttons.addWidget(btn_apply)
            buttons.addWidget(btn_close)

            lay = QtWidgets.QVBoxLayout(dlg)
            lay.addWidget(tbl, 1)
            lay.addLayout(buttons)

            def apply_back():
                try:
                    new_ini, new_ref, new_bnd, new_use = tbl.extract_all()

                    for k, v in (new_use or {}).items():
                        self._site_use_flags[str(k)] = bool(v)

                    self.tbl_sites.set_sites(
                        sites,
                        initial=new_ini,
                        refinable=new_ref,
                        bounds=new_bnd,
                        use_flags=new_use,
                    )

                    self._emit_changed_if_not_blocked()

                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        dlg,
                        "Apply failed",
                        str(e),
                    )

            btn_apply.clicked.connect(apply_back)
            btn_close.clicked.connect(dlg.close)

            dlg.show()
            dlg.raise_()
            dlg.activateWindow()

            # Keep reference so Python does not garbage-collect the dialog.
            self._site_params_dialog = dlg

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Open atomic site window failed",
                str(e),
            )

    def open_pair_dynamics_dialog(self) -> None:
        """
        Open pair dynamics and lambda coefficients in a larger dialog.
    
        The dialog uses copies of the pair-dynamics table and lambda widget.
        Clicking Apply copies values back to the main Builder widgets.
        """
        try:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Pair dynamics and lambda coefficients")
            dlg.setWindowFlags(
                dlg.windowFlags()
                | Qt.Window
                | Qt.WindowMinimizeButtonHint
                | Qt.WindowMaximizeButtonHint
                | Qt.WindowCloseButtonHint
            )
    
            screen = QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                geo = screen.availableGeometry()
                dlg.resize(
                    int(geo.width() * 0.80),
                    int(geo.height() * 0.75),
                )
                dlg.move(
                    geo.center().x() - dlg.width() // 2,
                    geo.center().y() - dlg.height() // 2,
                )
            else:
                dlg.resize(1100, 750)
    
            # ---------------------------------------------------------
            # Copy pair dynamics table
            # ---------------------------------------------------------
            dyn_ini, dyn_ref, dyn_bnd, dyn_use = self.tbl_dyn.extract_all()
    
            dyn_keys: List[str] = []
            for row in range(self.tbl_dyn.rowCount()):
                it = self.tbl_dyn.item(row, 0)
                if it is None:
                    continue
                key = it.data(Qt.UserRole) or it.text().strip()
                key = str(key).strip()
                if key:
                    dyn_keys.append(key)
    
            tbl_dyn_copy = DynParamTableWidget()
            tbl_dyn_copy.set_params(
                dyn_keys,
                dyn_ini,
                dyn_ref,
                dyn_bnd,
                use_flags=dyn_use,
            )
    
            # ---------------------------------------------------------
            # Copy lambda widget
            # ---------------------------------------------------------
            lam_ini, lam_ref, lam_bnd = self.lambda_list.extract_all()
    
            lambda_copy = LambdaListWidget()
    
            try:
                pairs = list(getattr(self.lambda_list, "_pairs", []) or [])
                lambda_copy.set_pairs(pairs, commit_current=False)
            except Exception:
                pass
            
            lambda_copy.set_lambdas(lam_ini, lam_ref, lam_bnd)
    
            # ---------------------------------------------------------
            # Layout
            # ---------------------------------------------------------
            split = QtWidgets.QSplitter(Qt.Vertical)
    
            w_dyn = QtWidgets.QWidget()
            v_dyn = QtWidgets.QVBoxLayout(w_dyn)
            v_dyn.setContentsMargins(0, 0, 0, 0)
    
            lbl_dyn = QtWidgets.QLabel("Pair dynamics parameters (delta1, delta2)")
            lbl_dyn.setStyleSheet("font-weight: bold;")
            v_dyn.addWidget(lbl_dyn)
            v_dyn.addWidget(tbl_dyn_copy, 1)
    
            w_lam = QtWidgets.QWidget()
            v_lam = QtWidgets.QVBoxLayout(w_lam)
            v_lam.setContentsMargins(0, 0, 0, 0)
    
            lbl_lam = QtWidgets.QLabel("Lambda coefficients")
            lbl_lam.setStyleSheet("font-weight: bold;")
            v_lam.addWidget(lbl_lam)
            v_lam.addWidget(lambda_copy, 1)
    
            split.addWidget(w_dyn)
            split.addWidget(w_lam)
            split.setChildrenCollapsible(False)
    
            try:
                split.setStretchFactor(0, 2)
                split.setStretchFactor(1, 2)
                split.setSizes([350, 350])
            except Exception:
                pass
            
            buttons = QtWidgets.QHBoxLayout()
            btn_apply = QtWidgets.QPushButton("Apply")
            btn_close = QtWidgets.QPushButton("Close")
    
            buttons.addStretch(1)
            buttons.addWidget(btn_apply)
            buttons.addWidget(btn_close)
    
            lay = QtWidgets.QVBoxLayout(dlg)
            lay.addWidget(split, 1)
            lay.addLayout(buttons)
    
            def apply_back():
                try:
                    new_dyn_ini, new_dyn_ref, new_dyn_bnd, new_dyn_use = tbl_dyn_copy.extract_all()
    
                    for k, v in (new_dyn_use or {}).items():
                        self._dyn_use_flags[str(k)] = bool(v)
    
                    self.tbl_dyn.set_params(
                        dyn_keys,
                        new_dyn_ini,
                        new_dyn_ref,
                        new_dyn_bnd,
                        use_flags=new_dyn_use,
                    )
    
                    new_lam_ini, new_lam_ref, new_lam_bnd = lambda_copy.extract_all()
    
                    try:
                        pairs = list(getattr(lambda_copy, "_pairs", []) or [])
                        self.lambda_list.set_pairs(pairs, commit_current=False)
                    except Exception:
                        pass
                    
                    self.lambda_list.set_lambdas(
                        new_lam_ini,
                        new_lam_ref,
                        new_lam_bnd,
                    )
    
                    self._emit_changed_if_not_blocked()
    
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        dlg,
                        "Apply failed",
                        str(e),
                    )
    
            btn_apply.clicked.connect(apply_back)
            btn_close.clicked.connect(dlg.close)
    
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
    
            self._pair_dynamics_dialog = dlg
    
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Open pair dynamics window failed",
                str(e),
            )


    def _seed_defaults(self) -> None:
        initial: Dict[str, Any] = {}
        refinable: Dict[str, Any] = {}
        bounds: Dict[str, Any] = {}
        self._apply_params_to_tables(initial, refinable, bounds)
        try:
            self._cf_set_terms([])
        except Exception:
            pass
        # Seed cubic A/B contrast-factor coefficients.
        # These are now normal refinable parameters.
        try:
            ab_keys = ["CEdgeA", "CEdgeB", "CScrewA", "CScrewB"]

            ab_seed = {
                "CEdgeA": 0.265280,
                "CEdgeB": -0.355950,
                "CScrewA": 0.307288,
                "CScrewB": -0.819979,
            }

            ab_ref = {k: False for k in ab_keys}
            ab_bnd = {}

            self.tbl_cf_ab.set_params(
                ab_keys,
                ab_seed,
                ab_ref,
                ab_bnd,
                tooltips={
                    "CEdgeA": "Cubic edge contrast-factor A coefficient.",
                    "CEdgeB": "Cubic edge contrast-factor B coefficient.",
                    "CScrewA": "Cubic screw contrast-factor A coefficient.",
                    "CScrewB": "Cubic screw contrast-factor B coefficient.",
                },
            )
        except Exception:
            pass


    def _element_biso_keys_from_sites(self) -> List[str]:
        """
        Return element-level Biso keys from current structure.

        For large structures the full atomic-site table is intentionally not
        populated, so this function uses the lightweight cached element list
        self._element_biso_elements.
        """
        elems = []

        site_records = getattr(self, "_site_records_for_gui", []) or []

        if site_records:
            for rec in site_records:
                el = clean_el_symbol(rec.get("el", ""))

                if el:
                    elems.append(el)

        else:
            elems = list(
                getattr(
                    self,
                    "_element_biso_elements",
                    set(),
                )
                or []
            )

        elems = sorted(
            set(elems),
            key=lambda x: x.lower(),
        )

        return [
            f"biso_{el}"
            for el in elems
        ]

    def _is_element_biso_key(self, key: str) -> bool:
        """
        True for element-level Biso keys like:
            biso_C, biso_N, biso_Co

        False for site-specific keys like:
            biso_c3, biso_n55
        """
        k = str(key or "").strip()

        if not k.lower().startswith("biso_"):
            return False

        raw = k[len("biso_"):].strip()

        if not raw:
            return False

        try:
            el = clean_el_symbol(raw)
        except Exception:
            return False

        return raw.lower() == el.lower()


    def _on_biso_mode_changed(self, checked: bool) -> None:
        """
        Switch between element-grouped and site-specific Biso mode.

        Element mode ON:
            - element Biso table enabled
            - site Biso controls disabled
            - site Use Biso? and Rbiso checkboxes visually unchecked

        Element mode OFF:
            - element Biso table disabled
            - site Biso controls enabled
        """
        use_element = bool(checked)

        try:
            self.tbl_elem_biso.setEnabled(use_element)
        except Exception:
            pass

        try:
            self.tbl_sites.set_biso_controls_enabled(not use_element)
        except Exception:
            pass

        # Make the site-specific Biso checkboxes visually unambiguous.
        # In element mode, site Biso is not exported anyway, but this prevents
        # confusion in the GUI.
        try:
            if use_element:
                self.tbl_sites.blockSignals(True)

                for row in range(self.tbl_sites.rowCount()):
                    # Column 14 = Use Biso?
                    it_use = self.tbl_sites.item(row, 14)
                    if it_use is not None:
                        it_use.setCheckState(Qt.Unchecked)

                    # Column 15 = Rbiso
                    it_ref = self.tbl_sites.item(row, 15)
                    if it_ref is not None:
                        it_ref.setCheckState(Qt.Unchecked)

                self.tbl_sites.blockSignals(False)

        except Exception:
            try:
                self.tbl_sites.blockSignals(False)
            except Exception:
                pass

    def _emit_changed_if_not_blocked(self):
        if not self._block:
            self.changed.emit()

    # -------------------------
    # (3) push lattice params from CIF
    # -------------------------
    def _set_struct_from_structure(self, s: Structure) -> None:
        """
        Push lattice parameters from structure into the GUI tables, while preserving
        current GUI parameter values/refine flags/bounds.
    
        IMPORTANT:
        This function only rebuilds GUI tables. It must NOT export to raw input.
        Therefore it must NOT use an `out` dictionary.
        """
        init_all: Dict[str, Any] = {}
        ref_all: Dict[str, Any] = {}
        bounds_all: Dict[str, Any] = {}
    
        # ---------------------------------------------------------
        # Preserve current main-table parameters
        # ---------------------------------------------------------
        for tbl in (
            self.tbl_struct,
            self.tbl_instr,
            self.tbl_size,
            self.tbl_cylinder_size,
            self.tbl_saxs,
            self.tbl_strain,
            self.tbl_wilkens,
            self.tbl_pah,
            self.tbl_elem_biso,
        ):
            try:
                ini, ref, bnd = tbl.extract()
    
                init_all.update(ini)
                ref_all.update(ref)
    
                for k, v in (bnd or {}).items():
                    try:
                        if isinstance(v, (tuple, list)) and len(v) >= 2:
                            lo = "" if v[0] in (None, "") else str(v[0]).strip()
                            hi = "" if v[1] in (None, "") else str(v[1]).strip()
                            bounds_all[str(k)] = (lo, hi)
                        else:
                            lo, hi = [x.strip() for x in str(v).split(",", 1)]
                            bounds_all[str(k)] = (lo, hi)
                    except Exception:
                        pass
                    
            except Exception:
                pass
            
        # ---------------------------------------------------------
        # Preserve atomic site parameters
        # ---------------------------------------------------------
        try:
            site_ini, site_ref, site_bnd, site_use = self.tbl_sites.extract_all()
    
            init_all.update(site_ini)
            ref_all.update(site_ref)
    
            for k, v in (site_bnd or {}).items():
                try:
                    if isinstance(v, (tuple, list)) and len(v) >= 2:
                        lo = "" if v[0] in (None, "") else str(v[0]).strip()
                        hi = "" if v[1] in (None, "") else str(v[1]).strip()
                        bounds_all[str(k)] = (lo, hi)
                    else:
                        lo, hi = [x.strip() for x in str(v).split(",", 1)]
                        bounds_all[str(k)] = (lo, hi)
                except Exception:
                    pass
                
            for k, v in (site_use or {}).items():
                self._site_use_flags[str(k)] = bool(v)
    
        except Exception:
            pass
        
        # ---------------------------------------------------------
        # Preserve local dynamics parameters
        # ---------------------------------------------------------
        try:
            dyn_ini, dyn_ref, dyn_bnd, dyn_use = self.tbl_dyn.extract_all()
    
            init_all.update(dyn_ini)
            ref_all.update(dyn_ref)
    
            for k, v in (dyn_bnd or {}).items():
                try:
                    if isinstance(v, (tuple, list)) and len(v) >= 2:
                        lo = "" if v[0] in (None, "") else str(v[0]).strip()
                        hi = "" if v[1] in (None, "") else str(v[1]).strip()
                        bounds_all[str(k)] = (lo, hi)
                    else:
                        lo, hi = [x.strip() for x in str(v).split(",", 1)]
                        bounds_all[str(k)] = (lo, hi)
                except Exception:
                    pass
                
            for k, v in (dyn_use or {}).items():
                self._dyn_use_flags[str(k)] = bool(v)
    
        except Exception:
            pass
        
        # ---------------------------------------------------------
        # Preserve lambda coefficients
        # ---------------------------------------------------------
        try:
            lam_ini, lam_ref, lam_bnd = self.lambda_list.extract_all()
    
            init_all.update(lam_ini)
            ref_all.update(lam_ref)
    
            for k, v in (lam_bnd or {}).items():
                try:
                    if isinstance(v, (tuple, list)) and len(v) >= 2:
                        lo = "" if v[0] in (None, "") else str(v[0]).strip()
                        hi = "" if v[1] in (None, "") else str(v[1]).strip()
                        bounds_all[str(k)] = (lo, hi)
                    else:
                        lo, hi = [x.strip() for x in str(v).split(",", 1)]
                        bounds_all[str(k)] = (lo, hi)
                except Exception:
                    pass
                
        except Exception:
            pass
        
        # ---------------------------------------------------------
        # Seed missing lattice parameters from structure
        # ---------------------------------------------------------
        def put_if_missing(k: str, v: float) -> None:
            if k not in init_all or init_all.get(k, "") in ("", None):
                init_all[k] = v
    
        lat = s.lattice
    
        put_if_missing("a", float(lat.a))
        put_if_missing("b", float(lat.b))
        put_if_missing("c", float(lat.c))
        put_if_missing("alpha", float(lat.alpha))
        put_if_missing("beta", float(lat.beta))
        put_if_missing("gamma", float(lat.gamma))
    
        # ---------------------------------------------------------
        # Rebuild categorized GUI tables
        # ---------------------------------------------------------
        self._apply_params_to_tables(init_all, ref_all, bounds_all)

    def _on_cf_visibility_changed(self, checked: bool) -> None:
        """Show/hide the contrast-factor panel.

        The panel is visible only when:
        1. Wilkens model is selected, and
        2. the user explicitly enables contrast factors.
        """
        use_anisotropic_microstrain = False
        try:
            use_anisotropic_microstrain = self.combo_micro_model.currentIndex() in (1, 2)
        except Exception:
            pass
        
        show = bool(checked) and bool(use_anisotropic_microstrain)

        try:
            self.gb_cf.setVisible(show)
            self.gb_cf.setEnabled(show)
            self._tabs_cf.setEnabled(show)
        except Exception:
            pass

        try:
            if hasattr(self, "micro_vsplit"):
                if show:
                    self.micro_vsplit.setSizes([260, 100])
                else:
                    self.micro_vsplit.setSizes([360, 0])
        except Exception:
            pass

        self._emit_changed_if_not_blocked()


    def _on_micro_model_changed(self) -> None:
        idx = self.combo_micro_model.currentIndex()

        use_iso = idx == 0
        use_wilkens = idx == 1
        use_pah = idx == 2

        try:
            if use_iso:
                self.micro_right_stack.setCurrentIndex(0)
            elif use_wilkens:
                self.micro_right_stack.setCurrentIndex(1)
            elif use_pah:
                self.micro_right_stack.setCurrentIndex(2)
        except Exception:
            pass

        self.gb_strain.setChecked(bool(use_iso))
        self.gb_wilkens.setChecked(bool(use_wilkens))
        self.gb_pah.setChecked(bool(use_pah))

        # Contrast factors are meaningful for Wilkens and PAH.
        try:
            self.chk_cf.setVisible(bool(use_wilkens or use_pah))
            if not (use_wilkens or use_pah):
                self.chk_cf.setChecked(False)
            self._on_cf_visibility_changed(self.chk_cf.isChecked())
        except Exception:
            pass

        self._apply_params_to_tables_from_current()
        self._emit_changed_if_not_blocked()
    def _on_structure_path_changed(self) -> None:
        if self._block:
            return
        path = (self.le_structure.text() or "").strip()
        self._update_symmetry_from_structure(path)
        self._emit_changed_if_not_blocked()

    def _update_symmetry_from_structure(self, path: str) -> None:
        self._sym_enabled = None
        self._biso_label_overrides = {}
        # If the structure file changed, do NOT carry over old per-site Biso values.
        structure_changed = (str(path or "").strip() != str(self._last_structure_path or "").strip())
        if not path or not os.path.exists(path):
            self._apply_params_to_tables_from_current()
            return

        # Read structure; for CIF also extract asymmetric-unit sites for Biso labels.
        asu_sites: List[Dict[str, Any]] = []
        try:
            if str(path).lower().endswith(".cif"):
                s, asu_sites = read_cif_asu_sites(path)
                if s is None:
                    raise RuntimeError("failed to read CIF")
            else:
                s = Structure.from_file(path)
        except Exception:
            self._apply_params_to_tables_from_current()
            return

        try:
            self._element_biso_elements = {
                clean_el_symbol(str(site.specie))
                for site in s.sites
                if clean_el_symbol(str(site.specie))
            }
        except Exception:
            self._element_biso_elements = set()


        # --- Pair options (per-species/oxidation unordered pairs) ---
        try:
            labels = sorted(
                {lambda_safe_species_label(str(site.specie)) for site in s.sites},
                key=lambda x: x.lower(),
            )
            pairs = [f"{a}-{b}" for a, b in itertools.combinations_with_replacement(labels, 2)]
            self.lambda_list.set_pairs(pairs)
            self._pair_labels_for_dynamics = list(pairs)
        except Exception:
            # If anything goes wrong, keep whatever pairs are currently shown
            self._pair_labels_for_dynamics = []
            pass

        enabled = {"scale"}
        sga = None
        cs = ""

        use_spacegroup_analyzer = (
            len(s.sites)
            <= int(
                getattr(
                    self,
                    "_symmetry_analyzer_max_sites",
                    500,
                )
            )
        )

        if use_spacegroup_analyzer:
            try:
                sga = SpacegroupAnalyzer(s, symprec=1e-3)
                cs = (sga.get_crystal_system() or "").lower()
            except Exception:
                sga = None
                cs = ""

        if cs == "cubic":
            enabled |= {"a"}
        elif cs == "tetragonal":
            enabled |= {"a", "c"}
        elif cs == "orthorhombic":
            enabled |= {"a", "b", "c"}
        elif cs in ("hexagonal", "trigonal"):
            enabled |= {"a", "c", "gamma"}
        elif cs == "monoclinic":
            enabled |= {"a", "b", "c", "beta"}
        elif cs == "triclinic":
            enabled |= {"a", "b", "c", "alpha", "beta", "gamma"}
        else:
            enabled |= {"a", "b", "c", "alpha", "beta", "gamma"}

        # --- Contrast factor invariants (Ei) required by symmetry (space group -> Laue class) ---
        terms: List[str] = []
        sg_num = None

        if sga is not None:
            try:
                sg_num = int(sga.get_space_group_number())
            except Exception:
                sg_num = None

        if sg_num is not None:
            try:
                cell0 = CFCell(
                    a=float(s.lattice.a), b=float(s.lattice.b), c=float(s.lattice.c),
                    alpha=float(s.lattice.alpha), beta=float(s.lattice.beta), gamma=float(s.lattice.gamma),
                )
                terms = required_terms_for_sg(sg_num, cell0)
            except Exception:
                terms = []

        # Preserve current GUI values where possible
        try:
            _old_flat = self._cf_extract()
            _old_edge, _old_screw = self._cf_split(_old_flat)
        except Exception:
            _old_edge, _old_screw = ({}, {})

        _edge_seed = {f"Edge{t}": float(_old_edge.get(t, 0.0)) for t in (terms or [])}
        _screw_seed = {f"Screw{t}": float(_old_screw.get(t, 0.0)) for t in (terms or [])}
        self._cf_set_terms(terms, edge_seed=_edge_seed, screw_seed=_screw_seed)

        # --- Local dynamics: build per-ASU-site Biso parameters (CIF atom loop) ---
        self._dyn_biso_keys = set()
        biso_seed: Dict[str, Any] = {}

        # Prefer asymmetric unit sites from CIF, otherwise fall back to unique sites (by frac coord).
        site_list: List[Dict[str, Any]] = []
        if asu_sites:
            site_list = asu_sites
        else:
            # fall back: use unique sites (avoid full symmetry-expanded duplicates)
            seen = set()
            for site in s.sites:
                el = clean_el_symbol(str(site.specie).strip())
                f = tuple(np.round(np.asarray(site.frac_coords, float), 6).tolist())
                key2 = (el, f)
                if key2 in seen:
                    continue
                seen.add(key2)
                site_list.append({"el": el, "frac": f, "occ": getattr(site, "occupancy", None), "biso": site.properties.get("biso", None)})

        # ---------------------------------------------------------
        # Atomic site table records for GUI.
        #
        # For large structures, do not populate the per-site table.
        # QTableWidget becomes very slow with thousands of rows and many
        # columns. Element-grouped Biso remains available through
        # self._element_biso_elements.
        # ---------------------------------------------------------
        self._site_records_for_gui = []
        self._site_biso_keys = set()

        site_table_max_rows = int(
            getattr(
                self,
                "_site_table_max_rows",
                300,
            )
        )

        populate_site_table = (
            site_table_max_rows > 0
            and len(site_list) <= site_table_max_rows
        )

        if populate_site_table:
            _label_counts: Dict[str, int] = {}

            for i, site in enumerate(site_list):
                el = clean_el_symbol(
                    site.get(
                        "el",
                        site.get(
                            "element",
                            "",
                        ),
                    )
                )

                if not el:
                    continue

                label = str(
                    site.get(
                        "label",
                        "",
                    )
                    or ""
                ).strip()

                # Fallback label if CIF has no proper label.
                if not label:
                    _label_counts[el] = _label_counts.get(
                        el,
                        0,
                    ) + 1
                    label = f"{el}{_label_counts[el]}"

                key = SiteParamTableWidget._safe_key(label)

                self._site_biso_keys.add(f"biso_{key}")

                frac = site.get(
                    "frac",
                    (
                        0.0,
                        0.0,
                        0.0,
                    ),
                )

                self._site_records_for_gui.append(
                    {
                        "label": label,
                        "el": el,
                        "frac": frac,
                        "occ": site.get(
                            "occ",
                            1.0,
                        ),
                        "biso": site.get(
                            "biso",
                            "",
                        ),
                    }
                )

        else:
            # Large-structure mode.
            #
            # Keep the per-site GUI table empty. This avoids creating tens of
            # thousands of QTableWidgetItem objects.
            #
            # Element Biso parameters are still created from
            # self._element_biso_elements by _element_biso_keys_from_sites().
            self._site_records_for_gui = []
            self._site_biso_keys = set()

        
        # Site Biso is now handled by tbl_sites, not by the pair-dynamics table.
        self._dyn_biso_keys = set()
        self._biso_label_overrides = {}

        # Reset/update use flags for new structure.
        #
        # Important:
        # Do NOT erase delta1/delta2 or pair-specific delta use flags.
        # Only refresh Biso keys when the structure changes.
        old_dyn_use_flags = dict(getattr(self, "_dyn_use_flags", {}) or {})

        if structure_changed:
            # Keep all non-Biso flags, e.g. delta1, delta2, delta1_ca-o, delta2_ca-o.
            self._dyn_use_flags = {
                k: bool(v)
                for k, v in old_dyn_use_flags.items()
                if not str(k).lower().startswith("biso_")
            }

            # Add Biso flags for the new structure.
            for k in self._dyn_biso_keys:
                self._dyn_use_flags[k] = True

        else:
            # Keep existing flags and ensure Biso keys exist.
            self._dyn_use_flags = dict(old_dyn_use_flags)

            for k in self._dyn_biso_keys:
                self._dyn_use_flags.setdefault(k, True)
        
        
        self._sym_enabled = enabled

        # (3) push lattice -> tables (solo se mancante)
        self._set_struct_from_structure(s)

        # Rebuild tables from current values, but drop stale per-site Biso values when structure changed.
        self._apply_params_to_tables_from_current(
            drop_prefixes=tuple(),
            seed_overrides=biso_seed,
            keep_keys=set(),
        )

        self._last_structure_path = str(path or "").strip()

    def _apply_params_to_tables(
        self,
        initial: Dict[str, Any],
        refinable: Dict[str, Any],
        bounds: Dict[str, Any],
    ) -> None:
        # Reset hidden stores on each full rebuild
        self._hidden_ini = {}
        self._hidden_ref = {}
        self._hidden_bnd = {}

        # IMPORTANT:
        # Display all known parameters, not only refinable=True parameters.
        # The Refine? checkbox already shows whether each parameter is refined.
        all_keys = set(initial.keys()) | set(refinable.keys()) | set(bounds.keys())
        all_keys |= set(DEFAULT_PARAMS.keys())

        deprecated_params = {"qbroad", "eta"}

        all_keys = {
            k for k in all_keys
            if str(k).strip().lower() not in deprecated_params
        }

        # Contrast-factor parameters are handled in the contrast-factor panel,
        # not in the generic parameter tables / hidden store.
        cf_names_lc = {
            "cedgea", "cedgeb", "cscrewa", "cscrewb",
            "burgers_mag", "burgers", "b_mag",
        }

        def _is_cf_key(k) -> bool:
            kl = str(k).strip().lower()
            if kl in cf_names_lc:
                return True
            if re.match(r"^edge[_ ]*e\d+$", kl):
                return True
            if re.match(r"^screw[_ ]*e\d+$", kl):
                return True
            return False

        all_keys = {k for k in all_keys if not _is_cf_key(k)}
        show_keys = set(all_keys)

        # Lambda parameters are handled by LambdaListWidget.
        lambda_keys = {k for k in show_keys if str(k).lower().startswith(("lambda", "lam"))}
        for k in list(lambda_keys):
            show_keys.discard(k)

        structural = [
            k for k in ["scale", "a", "b", "c", "alpha", "beta", "gamma"]
            if k in show_keys
        ]

        instrumental = [
            k for k in [
                "qdamp",
                "delta_broad",
                "qmax",
                "qmax_zeros",
                "qmax_pad",
            ]
            if k in show_keys
        ]

        micro_size = [
            k for k in ["d", "d_std"]
            if k in show_keys
        ]

        cylinder_size = [
            k for k in [
                "cyl_diameter",
                "cyl_thickness",
                "cyl_thickness_std",
                "cyl_axis_h",
                "cyl_axis_k",
                "cyl_axis_l",
            ]
            if k in show_keys
        ]

        saxs_shape = [
            k for k in [
                "saxs_scale",
                "saxs_diameter",
                "saxs_diameter_std",
                "saxs_height",
                "saxs_height_std",
            ]
            if k in show_keys
        ]

        idx_micro = self.combo_micro_model.currentIndex()
        use_wilkens = idx_micro == 1
        use_pah = idx_micro == 2


        
        micro_strain = [] if (use_wilkens or use_pah) else [
            k for k in ["delta_g"]
            if k in show_keys
        ]

        idx_micro = self.combo_micro_model.currentIndex()
        use_wilkens = idx_micro == 1
        use_pah = idx_micro == 2
        wilkens_order = ["rho", "re", "fe"]

        has_wilkens_params = any(
            str(k).lower() in set(wilkens_order)
            for k in show_keys
        )

        wilkens = []

        if use_wilkens or has_wilkens_params:
            for wanted in wilkens_order:
                for k in show_keys:
                    if str(k).lower() == wanted:
                        wilkens.append(k)
                        break

        pah_order = ["pah_a", "pah_b", "fe"]

        has_pah_params = any(
            str(k).lower() in set(pah_order)
            for k in show_keys
        )

        pah = []

        if use_pah or has_pah_params:
            for wanted in pah_order:
                for k in show_keys:
                    if str(k).lower() == wanted:
                        pah.append(k)
                        break

        # Local/pair dynamics keys.
        #
        # If a structure is loaded, Biso parameters are handled either by:
        #   - tbl_sites for small structures, or
        #   - tbl_elem_biso for large structures.
        #
        # Therefore do not show Biso in the local-dynamics table when we know
        # the structure elements.
        has_site_table = bool(
            getattr(
                self,
                "_site_records_for_gui",
                [],
            )
            or []
        )

        has_structure_elements = bool(
            has_site_table
            or (
                getattr(
                    self,
                    "_element_biso_elements",
                    set(),
                )
                or set()
            )
        )

        if has_structure_elements:
            dynamics_set = set()
        else:
            dynamics_set = {
                str(k)
                for k in show_keys
                if str(k).lower().startswith("biso")
            }

        # Always show global delta1/delta2
        dynamics_set.add("delta1")
        dynamics_set.add("delta2")

        # Include pair-specific delta1/delta2 keys based on structure pair list
        for pair in getattr(self, "_pair_labels_for_dynamics", []) or []:
            dynamics_set.add(f"delta1_{pair}")
            dynamics_set.add(f"delta2_{pair}")

        # Preserve any already-existing pair-specific delta1/delta2 from input
        for k in all_keys:
            ks = str(k).strip().lower()
            if ks.startswith("delta1_") or ks.startswith("delta2_"):
                dynamics_set.add(str(k))

        dynamics = sorted(dynamics_set)

        # Site-table parameters are visible in tbl_sites and must not be treated
        # as hidden parameters, otherwise stale hidden flags can overwrite current
        # Biso refinement checkboxes.
        site_param_keys = set()
        
        try:
            for rec in getattr(self, "_site_records_for_gui", []) or []:
                label = str(rec.get("label", "") or "").strip()
                if not label:
                    continue
                
                skey = SiteParamTableWidget._safe_key(label)
        
                site_param_keys.update(
                    {
                        f"x_{skey}",
                        f"y_{skey}",
                        f"z_{skey}",
                        f"occ_{skey}",
                        f"biso_{skey}",
                    }
                )
        except Exception:
            site_param_keys = set()
        
        elem_biso_param_keys = set()

        try:
            elem_biso_param_keys = set(
                self._element_biso_keys_from_sites()
            )
        except Exception:
            elem_biso_param_keys = set()

        used = (
            set(structural)
            | set(instrumental)
            | set(micro_size)
            | set(cylinder_size)
            | set(micro_strain)
            | set(wilkens)
            | set(pah)
            | set(dynamics)
            | set(site_param_keys)
            | set(elem_biso_param_keys)
            | set(saxs_shape)
        )


        hidden = sorted(all_keys - used - set(lambda_keys))

        enabled = self._sym_enabled

        # Apply symmetry enabling only to the structural table.
        # Do not remove non-structural parameters from the GUI.
        self.tbl_struct.set_params(
            [str(k) for k in structural],
            initial,
            refinable,
            bounds,
            enabled_keys=enabled,
        )

        self.tbl_instr.set_params(
            [str(k) for k in instrumental],
            initial,
            refinable,
            bounds,
        )

        self.tbl_size.set_params(
            [str(k) for k in micro_size],
            initial,
            refinable,
            bounds,
        )

        self.tbl_cylinder_size.set_params(
            [str(k) for k in cylinder_size],
            initial,
            refinable,
            bounds,
            tooltips={
                "cyl_diameter": "Cylinder/disk diameter D in Å. Inactive if <= 0.",
                "cyl_thickness": "Mean cylinder/disk thickness t in Å.",
                "cyl_thickness_std": "Real-space standard deviation of lognormal thickness distribution. Use 0 for monodisperse.",
                "cyl_axis_h": "Cylinder axis h index. Default [0 0 1].",
                "cyl_axis_k": "Cylinder axis k index. Default [0 0 1].",
                "cyl_axis_l": "Cylinder axis l index. Default [0 0 1].",
            },
        )

        self.tbl_saxs.set_params(
            [str(k) for k in saxs_shape],
            initial,
            refinable,
            bounds,
            tooltips={
                "saxs_scale": "Independent amplitude of the smooth negative SAXS/missing-low-Q belly term.",
                "saxs_diameter": "Independent effective diameter for the SAXS belly term. If 0, main size model is used.",
                "saxs_diameter_std": "Real-space standard deviation of SAXS belly diameter distribution.",
                "saxs_height": "Independent height/thickness for cylinder/disk SAXS belly model.",
                "saxs_height_std": "Real-space standard deviation of SAXS belly height/thickness distribution.",
            },
        )


        self.tbl_strain.set_params(
            [str(k) for k in micro_strain],
            initial,
            refinable,
            bounds,
        )

        self.tbl_wilkens.set_params(
            [str(k) for k in wilkens],
            initial,
            refinable,
            bounds,
        )

        self.tbl_pah.set_params(
            [str(k) for k in pah],
            initial,
            refinable,
            bounds,
        )


        try:
            self.tbl_sites.set_sites(
                getattr(self, "_site_records_for_gui", []) or [],
                initial,
                refinable,
                bounds=bounds,
                use_flags=getattr(self, "_site_use_flags", {}) or {},
            )
        except Exception:
            pass

        # ---------------------------------------------------------
        # Element-grouped Biso table
        # ---------------------------------------------------------
        try:
            elem_biso_keys = self._element_biso_keys_from_sites()

            elem_initial = dict(initial or {})
            elem_ref = dict(refinable or {})
            elem_bounds = dict(bounds or {})

            # Seed missing element Biso values from available site Biso values
            # or from default 0.3.
            for key in elem_biso_keys:
                if key in elem_initial:
                    continue
                
                el = key[len("biso_"):]
                vals = []

                for rec in getattr(self, "_site_records_for_gui", []) or []:
                    rec_el = clean_el_symbol(rec.get("el", ""))
                    if rec_el != el:
                        continue
                    
                    site_label = str(rec.get("label", "") or "").strip()
                    site_key = SiteParamTableWidget._safe_key(site_label)
                    site_biso_key = f"biso_{site_key}"

                    if site_biso_key in initial:
                        try:
                            vals.append(float(initial[site_biso_key]))
                        except Exception:
                            pass
                    else:
                        try:
                            bv = rec.get("biso", None)
                            if bv not in (None, ""):
                                vals.append(float(bv))
                        except Exception:
                            pass
                        
                elem_initial[key] = float(np.mean(vals)) if vals else 0.3

            # Default safe Biso bounds.
            for key in elem_biso_keys:
                elem_bounds.setdefault(key, ("0", "10"))

            self.tbl_elem_biso.set_params(
                elem_biso_keys,
                elem_initial,
                elem_ref,
                elem_bounds,
                tooltips={
                    k: "One isotropic Biso shared by all atoms of this element type."
                    for k in elem_biso_keys
                },
            )

        except Exception:
            pass
        
        try:
            self._on_biso_mode_changed(self.chk_biso_by_element.isChecked())
        except Exception:
            pass
        
        self.tbl_dyn.set_params(
            [str(k) for k in dynamics],
            initial,
            refinable,
            bounds,
            label_overrides=self._biso_label_overrides,
            use_flags=self._dyn_use_flags,
        )

        # Preserve non-exposed parameters so saving does not delete them.
        for k in hidden:
            kk = str(k)
            if kk in initial:
                self._hidden_ini[kk] = initial.get(kk)
            if kk in refinable:
                self._hidden_ref[kk] = bool(refinable.get(kk, False))
            if kk in bounds:
                self._hidden_bnd[kk] = bounds.get(kk)

        # Lambda list
        lam_init = {k: initial.get(k, 0.0) for k in sorted(lambda_keys)}
        lam_ref = {k: bool(refinable.get(k, False)) for k in sorted(lambda_keys)}
        lam_bnd = {k: bounds.get(k, ("", "")) for k in sorted(lambda_keys)}
        self.lambda_list.set_lambdas(lam_init, lam_ref, lam_bnd)
    
    def _apply_params_to_tables_from_current(
        self,
        drop_prefixes: Tuple[str, ...] = tuple(),
        seed_overrides: Optional[Dict[str, Any]] = None,
        keep_keys: Optional[set[str]] = None,
    ) -> None:
        """
        Collect current values/refine flags/bounds from all parameter tables + lambda list,
        then rebuild the categorized tables consistently (incl. symmetry enable/disable).
        """
        init_all: Dict[str, Any] = {}
        ref_all: Dict[str, Any] = {}
        bounds_all: Dict[str, Any] = {}

        # Collect from all main tables
        for tbl in (
            self.tbl_struct,
            self.tbl_instr,
            self.tbl_size,
            self.tbl_cylinder_size,
            self.tbl_saxs,
            self.tbl_strain,
            self.tbl_wilkens,
            self.tbl_pah,
            self.tbl_elem_biso,
        ):
            ini, ref, bnd = tbl.extract()
            init_all.update(ini)
            ref_all.update(ref)

            # bnd is dict: key -> "lo, hi"
            for k, v in (bnd or {}).items():
                try:
                    lo, hi = [x.strip() for x in str(v).split(",", 1)]
                    bounds_all[str(k)] = (lo, hi)
                except Exception:
                    # ignore malformed bounds
                    pass
        

        # Atomic site parameters
        try:
            site_ini, site_ref, site_bnd, site_use = self.tbl_sites.extract_all()

            init_all.update(site_ini)
            ref_all.update(site_ref)

            for k, v in (site_bnd or {}).items():
                try:
                    if isinstance(v, (tuple, list)) and len(v) >= 2:
                        lo = "" if v[0] in (None, "") else str(v[0]).strip()
                        hi = "" if v[1] in (None, "") else str(v[1]).strip()
                        bounds_all[str(k)] = (lo, hi)
                    else:
                        lo, hi = [x.strip() for x in str(v).split(",", 1)]
                        bounds_all[str(k)] = (lo, hi)
                except Exception:
                    pass
                
            for k, v in (site_use or {}).items():
                self._site_use_flags[str(k)] = bool(v)

        except Exception:
            pass
        

        # Local dynamics (has Use? flags)
        dyn_ini, dyn_ref, dyn_bnd, dyn_use = self.tbl_dyn.extract_all()
        init_all.update(dyn_ini)
        ref_all.update(dyn_ref)
        for k, v in (dyn_bnd or {}).items():
            try:
                lo, hi = [x.strip() for x in str(v).split(",", 1)]
                bounds_all[str(k)] = (lo, hi)
            except Exception:
                pass
        # Store use flags (only for currently visible dyn keys)
        for k, v in (dyn_use or {}).items():
            self._dyn_use_flags[str(k)] = bool(v)

        # Collect lambdas
        lam_ini, lam_ref, lam_bnd = self.lambda_list.extract_all()
        init_all.update(lam_ini)
        ref_all.update(lam_ref)

        for k, v in (lam_bnd or {}).items():
            try:
                if isinstance(v, (tuple, list)) and len(v) >= 2:
                    lo = "" if v[0] in (None, "") else str(v[0]).strip()
                    hi = "" if v[1] in (None, "") else str(v[1]).strip()
                    bounds_all[str(k)] = (lo, hi)
                else:
                    lo, hi = [x.strip() for x in str(v).split(",", 1)]
                    bounds_all[str(k)] = (lo, hi)
            except Exception:
                pass

        # Carry over hidden params (not shown in GUI)
        init_all.update(getattr(self, "_hidden_ini", {}) or {})
        ref_all.update(getattr(self, "_hidden_ref", {}) or {})
        for k, v in (getattr(self, "_hidden_bnd", {}) or {}).items():
            bounds_all[str(k)] = v

        # Drop stale keys (e.g. old dynamic biso_* when structure changes)
        keep_keys = set(keep_keys or set())
        for pref in (drop_prefixes or tuple()):
            for k in list(init_all.keys()):
                if str(k).lower().startswith(str(pref).lower()):
                    if str(k) in keep_keys:
                        continue
                    init_all.pop(k, None)
                    ref_all.pop(k, None)
                    bounds_all.pop(k, None)
                    self._dyn_use_flags.pop(str(k), None)

        # Seed overrides (e.g. read Biso values from CIF)
        if isinstance(seed_overrides, dict):
            for k, v in seed_overrides.items():
                if k not in init_all:
                    init_all[k] = v

        self._apply_params_to_tables(init_all, ref_all, bounds_all)


    def _get_refinable_from_symmetry(self, lattice_system: str) -> set:
        """Get the set of refinable parameters based on lattice symmetry."""
        refinable = set()

        if lattice_system == "cubic":
            refinable = {"a"}
        elif lattice_system == "tetragonal":
            refinable = {"a", "c"}
        elif lattice_system == "orthorhombic":
            refinable = {"a", "b", "c"}
        elif lattice_system in ("hexagonal", "trigonal"):
            refinable = {"a", "c"}
        elif lattice_system == "monoclinic":
            refinable = {"a", "b", "c", "beta"}
        elif lattice_system == "triclinic":
            refinable = {"a", "b", "c", "alpha", "beta", "gamma"}
        else:
            # Default to all lattice parameters
            refinable = {"a", "b", "c", "alpha", "beta", "gamma"}

        return refinable


    def set_from_config(self, cfg: Dict[str, Any]) -> None:
        """
        Populate the builder UI from the dict returned by read_input_file().
        This is intentionally tolerant to multiple schema variants.
        """
        if not isinstance(cfg, dict):
            return

        self._block = True
        try:
            # ---- files ----
            files = cfg.get("files", cfg) if isinstance(cfg.get("files", cfg), dict) else cfg
            structure_path = str(files.get("structure_file", cfg.get("structure_file", "")) or "").strip()
            gr_path = str(files.get("gr_data_file", cfg.get("gr_data_file", "")) or "").strip()

            self.le_structure.setText(structure_path)
            self.le_grdata.setText(gr_path)

            # ---- range ----
            rng = cfg.get("range", cfg) if isinstance(cfg.get("range", cfg), dict) else cfg
            rmin = rng.get("r_min", cfg.get("r_min", None))
            rmax = rng.get("r_max", cfg.get("r_max", None))
            if rmin is not None:
                try:
                    self.le_rmin.setText(str(float(rmin)))
                except Exception:
                    pass
                
            if rmax is not None:
                try:
                    self.le_rmax.setText(str(float(rmax)))
                except Exception:
                    pass

            # ---- pair_generation ----
            pg = cfg.get("pair_generation", cfg) if isinstance(cfg.get("pair_generation", cfg), dict) else cfg
            rext = pg.get("r_extension", cfg.get("r_extension", None))
            if rext is not None:
                try:
                    self.le_rext.setText(str(float(rext)))
                except Exception:
                    pass

            # ---- refinement ----
            refsec = cfg.get("refinement", cfg) if isinstance(cfg.get("refinement", cfg), dict) else cfg
            staged = refsec.get("staged_refinement", cfg.get("staged_refinement", None))
            if staged is not None:
                self.chk_staged.setChecked(bool(staged))
            
            def _cfg_bool(val, default=False):
                if val is None:
                    return bool(default)
                if isinstance(val, bool):
                    return val
                if isinstance(val, (int, float)):
                    return bool(int(val))
                if isinstance(val, str):
                    s = val.strip().lower()
                    if s in ("1", "true", "yes", "y", "on"):
                        return True
                    if s in ("0", "false", "no", "n", "off"):
                        return False
                return bool(default)


            v = refsec.get(
                "compute_pair_contributions_on_finish",
                cfg.get("compute_pair_contributions_on_finish", None),
            )
            if v is not None:
                self.chk_compute_pair_contrib.setChecked(_cfg_bool(v, False))

            v = refsec.get(
                "compute_warren_on_finish",
                cfg.get("compute_warren_on_finish", None),
            )
            if v is not None:
                self.chk_compute_warren.setChecked(_cfg_bool(v, False))

            v = refsec.get(
                "compute_local_trends_on_finish",
                cfg.get("compute_local_trends_on_finish", None),
            )
            if v is not None:
                self.chk_compute_local_trends.setChecked(_cfg_bool(v, False))

            v = refsec.get(
                "export_finite_coordination",
                cfg.get("export_finite_coordination", None),
            )
            if v is not None:
                self.chk_export_finite_coordination.setChecked(_cfg_bool(v, False))


            rstep_final = refsec.get("refinement_r_step_final", cfg.get("refinement_r_step_final", None))
            if rstep_final is not None:
                try:
                    self.spin_rstep_final.setValue(float(rstep_final))
                except Exception:
                    pass

            max_nfev_final = refsec.get("max_nfev_final", cfg.get("max_nfev_final", None))
            if max_nfev_final is not None:
                try:
                    self.spin_max_nfev.setValue(int(max_nfev_final))
                except Exception:
                    pass

            for key, widget in (("ftol", self.spin_ftol), ("xtol", self.spin_xtol), ("gtol", self.spin_gtol)):
                val = refsec.get(key, cfg.get(key, None))
                if val is not None:
                    try:
                        widget.setValue(float(val))
                    except Exception:
                        pass

            prog = refsec.get("progress_every_sec", cfg.get("progress_every_sec", None))
            if prog is not None:
                try:
                    self.spin_progress.setValue(float(prog))
                except Exception:
                    pass

            # ---- SAXS / missing-low-Q belly settings ----
            try:
                saxs_enabled = refsec.get(
                    "saxs_enabled",
                    cfg.get(
                        "saxs_enabled",
                        False,
                    ),
                )

                self.gb_saxs.setChecked(
                    _cfg_bool(
                        saxs_enabled,
                        False,
                    )
                )

                saxs_model = str(
                    refsec.get(
                        "saxs_model",
                        cfg.get(
                            "saxs_model",
                            "sphere",
                        ),
                    )
                    or "sphere"
                ).strip().lower()

                model_map = {
                    "sphere": "sphere",
                    "spherical": "sphere",
                    "cylinder": "cylinder",
                    "cyl": "cylinder",
                    "disk": "disk",
                    "disc": "disk",
                    "same": "same as size model",
                    "same_as_size": "same as size model",
                    "same_as_size_model": "same as size model",
                    "same as size model": "same as size model",
                    "auto": "same as size model",
                }

                shown_model = model_map.get(
                    saxs_model,
                    "sphere",
                )

                idx_saxs = self.combo_saxs_model.findText(
                    shown_model
                )

                if idx_saxs >= 0:
                    self.combo_saxs_model.setCurrentIndex(
                        idx_saxs
                    )

                v = refsec.get(
                    "saxs_apply_qdamp",
                    cfg.get(
                        "saxs_apply_qdamp",
                        None,
                    ),
                )

                if v is not None:
                    self.chk_saxs_apply_qdamp.setChecked(
                        _cfg_bool(
                            v,
                            False,
                        )
                    )

                v = refsec.get(
                    "saxs_apply_qmax",
                    cfg.get(
                        "saxs_apply_qmax",
                        None,
                    ),
                )

                if v is not None:
                    self.chk_saxs_apply_qmax.setChecked(
                        _cfg_bool(
                            v,
                            True,
                        )
                    )

            except Exception:
                pass

            # ---- parameters ----
            initial = cfg.get("initial_values", cfg.get("initial", {}))
            refinable = cfg.get("refinable_parameters", cfg.get("refinable", {}))
            bounds = cfg.get("bounds", {})

            if not isinstance(initial, dict):
                initial = {}
            if not isinstance(refinable, dict):
                refinable = {}
            if not isinstance(bounds, dict):
                bounds = {}

            # ---------------------------------------------------------
            # Rebuild Local dynamics Use? flags from the loaded input.
            #
            # Important:
            # - If delta1/delta2 or pair-specific delta parameters are explicitly
            #   present in the input file, mark them as used.
            # - If they are absent, do not activate them just because DEFAULT_PARAMS has them.
            # ---------------------------------------------------------
            self._dyn_use_flags = {}
            
            _explicit_param_keys = (
                set(str(k) for k in initial.keys())
                | set(str(k) for k in refinable.keys())
                | set(str(k) for k in bounds.keys())
            )
            
            for k in _explicit_param_keys:
                kl = str(k).strip().lower()
            
                if (
                    kl in ("delta1", "delta2")
                    or kl.startswith("delta1_")
                    or kl.startswith("delta2_")
                    or kl.startswith("biso_")
                ):
                    self._dyn_use_flags[str(k)] = True

            # Normalize bounds to (lo, hi)
            bounds_norm: Dict[str, Any] = {}
            for k, v in bounds.items():
                kk = str(k)
                if isinstance(v, (tuple, list)) and len(v) == 2:
                    lo = "" if v[0] in (None, "") else str(v[0])
                    hi = "" if v[1] in (None, "") else str(v[1])
                    if lo != "" or hi != "":
                        bounds_norm[kk] = (lo, hi)
                else:
                    s = str(v)
                    if "," in s:
                        try:
                            lo, hi = [x.strip() for x in s.split(",", 1)]
                            if lo != "" or hi != "":
                                bounds_norm[kk] = (lo, hi)
                        except Exception:
                            pass


            # ---------------------------------------------------------
            # Decide Biso mode from loaded input.
            #
            # If the input explicitly contains site-specific Biso keys and no element
            # Biso keys, reopen in site-specific mode.
            # Otherwise default to element-grouped Biso mode.
            # ---------------------------------------------------------
            try:
                explicit_biso_keys = [
                    str(k)
                    for k in (
                        set(initial.keys())
                        | set(refinable.keys())
                        | set(bounds_norm.keys())
                    )
                    if str(k).strip().lower().startswith("biso_")
                ]

                has_element_biso = any(
                    self._is_element_biso_key(k)
                    for k in explicit_biso_keys
                )

                has_site_biso = any(
                    not self._is_element_biso_key(k)
                    for k in explicit_biso_keys
                )

                # Default: element-grouped mode.
                # Only use site mode if the file has site Biso and no element Biso.
                self.chk_biso_by_element.setChecked(
                    not (has_site_biso and not has_element_biso)
                )

            except Exception:
                self.chk_biso_by_element.setChecked(True)


            # Detect contrast-factor parameters now that [contrast_factors] is deprecated.
            # Cubic A/B:
            #   CEdgeA, CEdgeB, CScrewA, CScrewB
            # Non-cubic invariants:
            #   EdgeE1..., ScrewE1...
            def _detect_cf_params(*dicts):
                keys_lc = set()
                for d in dicts:
                    if isinstance(d, dict):
                        keys_lc |= {str(k).strip().lower() for k in d.keys()}

                has_ab = bool(
                    keys_lc
                    & {
                        "cedgea",
                        "cedgeb",
                        "cscrewa",
                        "cscrewb",
                    }
                )

                has_ei = any(
                    re.match(r"^(edge|screw)[_ ]*e\d+$", k)
                    for k in keys_lc
                )

                return has_ab, has_ei, bool(has_ab or has_ei)

            has_cf_ab_params, has_cf_ei_params, has_any_cf_params = _detect_cf_params(
                initial,
                refinable,
                bounds_norm,
            )

            # Decide micro model early.
            # Wilkens should be selected if rho/re/fe OR contrast-factor parameters exist.
            try:
                keys_lc = {str(k).lower() for k in (set(initial) | set(refinable) | set(bounds_norm))}
                micro_model = str(
                    (cfg.get("refinement") or {}).get("microstrain_model", "")
                ).strip().lower()

                pah_hint = bool(keys_lc & {"pah_a", "pah_b"})
                wilkens_hint = bool(keys_lc & {"rho", "re", "fe"})

                if micro_model in ("pah", "adler-houska", "adler_houska"):
                    self.combo_micro_model.setCurrentIndex(2)
                elif micro_model in ("wilkens", "wilkins", "dislocation"):
                    self.combo_micro_model.setCurrentIndex(1)
                elif pah_hint:
                    self.combo_micro_model.setCurrentIndex(2)
                elif wilkens_hint or bool(has_any_cf_params):
                    self.combo_micro_model.setCurrentIndex(1)
                else:
                    self.combo_micro_model.setCurrentIndex(0)
            except Exception:
                pass
            
            # First apply parameters from the input file into the tables.
            # (Important: do this BEFORE reading the structure; structure-loading must NOT overwrite
            # values explicitly provided in the input.)
            self._apply_params_to_tables(initial, refinable, bounds_norm)

            # Now update symmetry-derived availability and any missing lattice defaults from structure,
            # while preserving the values that were just loaded.
            self._update_symmetry_from_structure(structure_path)


            
            # ---------------------------------------------------------
            # Load crystallite-size model from the input file
            # ---------------------------------------------------------
            try:
                shape_cfg = cfg.get("crystallite_shape", {}) or {}

                if isinstance(shape_cfg, dict):
                    self._crystallite_shape_spec = dict(shape_cfg)
                else:
                    self._crystallite_shape_spec = {}

                mode = str(
                    self._crystallite_shape_spec.get("mode", "conventional")
                ).strip().lower()

                # Finite crystallite shape created by the shape-builder dialog.
                if mode == "finite_shape":
                    self.combo_size_model.setCurrentIndex(2)

                else:
                    # Conventional model:
                    # if cylinder/disk parameters are present and physically active,
                    # reopen the Cylinder/disk model panel.
                    cyl_diameter = float(initial.get("cyl_diameter", 0.0) or 0.0)
                    cyl_thickness = float(initial.get("cyl_thickness", 0.0) or 0.0)

                    if cyl_diameter > 0.0 and cyl_thickness > 0.0:
                        self.combo_size_model.setCurrentIndex(1)
                    else:
                        self.combo_size_model.setCurrentIndex(0)

                self._update_shape_summary_label()

            except Exception:
                self.combo_size_model.setCurrentIndex(0)
                self._crystallite_shape_spec = {}
                self._update_shape_summary_label()


            # After the structure is known, explicitly restore the atomic-site table
            # from the loaded input dictionaries. This is essential because site records
            # only exist after reading the CIF/structure, and otherwise Biso refine flags
            # can be reset to False.
            try:
                site_use_flags = dict(getattr(self, "_site_use_flags", {}) or {})

                # If a site parameter exists in the loaded input/refinable/bounds,
                # consider it used.
                explicit_site_keys = (
                    set(str(k) for k in initial.keys())
                    | set(str(k) for k in refinable.keys())
                    | set(str(k) for k in bounds_norm.keys())
                )

                for k in explicit_site_keys:
                    kl = str(k).strip().lower()
                    if (
                        kl.startswith("x_")
                        or kl.startswith("y_")
                        or kl.startswith("z_")
                        or kl.startswith("occ_")
                        or kl.startswith("biso_")
                    ):
                        site_use_flags[str(k)] = True

                self._site_use_flags = site_use_flags

                self.tbl_sites.set_sites(
                    getattr(self, "_site_records_for_gui", []) or [],
                    initial,
                    refinable,
                    bounds=bounds_norm,
                    use_flags=self._site_use_flags,
                )

            except Exception:
                pass
            
            
            # ---- contrast factors (Ei) ----
            coeffs_in = {}
            coeffs_has_legacy = False
            coeffs_has_ei = False

            try:
                coeffs_in = dict(cfg.get("contrast_coeffs", cfg.get("contrast_factors", {})) or {})

                # Cubic A/B contrast factors are now normal parameters.
                # Do NOT pop them from initial/refinable/bounds.
                # They will be loaded directly into the Cubic A/B table below.

                # Detect legacy cubic A/B coefficients.
                try:
                    if isinstance(coeffs_in.get("legacy", None), dict) and len(coeffs_in["legacy"]) > 0:
                        coeffs_has_legacy = True

                    for _k in ("cedgea", "cedgeb", "cscrewa", "cscrewb"):
                        if _k in coeffs_in:
                            coeffs_has_legacy = True
                except Exception:
                    coeffs_has_legacy = False

                # Seed cubic A/B coefficients table.
                # Preferred source: [initial_values].
                # Backward-compatible source: old [contrast_factors].
                try:
                    ab_keys = ["CEdgeA", "CEdgeB", "CScrewA", "CScrewB"]

                    initial_lc = {str(k).strip().lower(): v for k, v in initial.items()}
                    refinable_lc = {str(k).strip().lower(): v for k, v in refinable.items()}
                    bounds_lc = {str(k).strip().lower(): v for k, v in bounds_norm.items()}

                    legacy_map: Dict[str, Any] = {}
                    leg = coeffs_in.get("legacy", None)

                    if isinstance(leg, dict):
                        for lk, lv in leg.items():
                            ku = str(lk).strip().upper()
                            if ku == "CEDGEA":
                                legacy_map["CEdgeA"] = lv
                            elif ku == "CEDGEB":
                                legacy_map["CEdgeB"] = lv
                            elif ku == "CSCREWA":
                                legacy_map["CScrewA"] = lv
                            elif ku == "CSCREWB":
                                legacy_map["CScrewB"] = lv

                    coeffs_lc = {str(k).strip().lower(): v for k, v in coeffs_in.items()}
                    if "cedgea" in coeffs_lc:
                        legacy_map["CEdgeA"] = coeffs_lc["cedgea"]
                    if "cedgeb" in coeffs_lc:
                        legacy_map["CEdgeB"] = coeffs_lc["cedgeb"]
                    if "cscrewa" in coeffs_lc:
                        legacy_map["CScrewA"] = coeffs_lc["cscrewa"]
                    if "cscrewb" in coeffs_lc:
                        legacy_map["CScrewB"] = coeffs_lc["cscrewb"]

                    defaults_ab = {
                        "CEdgeA": 0.265280,
                        "CEdgeB": -0.355950,
                        "CScrewA": 0.307288,
                        "CScrewB": -0.819979,
                    }

                    ab_seed = {}
                    ab_ref = {}
                    ab_bnd = {}

                    for k in ab_keys:
                        kl = k.lower()

                        if kl in initial_lc:
                            ab_seed[k] = initial_lc[kl]
                        elif k in initial:
                            ab_seed[k] = initial[k]
                        elif k in legacy_map:
                            ab_seed[k] = legacy_map[k]
                        else:
                            ab_seed[k] = defaults_ab[k]

                        if kl in refinable_lc:
                            ab_ref[k] = bool(refinable_lc[kl])
                        elif k in refinable:
                            ab_ref[k] = bool(refinable[k])
                        else:
                            ab_ref[k] = False

                        if kl in bounds_lc:
                            ab_bnd[k] = bounds_lc[kl]
                        elif k in bounds_norm:
                            ab_bnd[k] = bounds_norm[k]

                    self.tbl_cf_ab.set_params(
                        ab_keys,
                        ab_seed,
                        ab_ref,
                        ab_bnd,
                        tooltips={
                            "CEdgeA": "Cubic edge contrast-factor A coefficient.",
                            "CEdgeB": "Cubic edge contrast-factor B coefficient.",
                            "CScrewA": "Cubic screw contrast-factor A coefficient.",
                            "CScrewB": "Cubic screw contrast-factor B coefficient.",
                        },
                    )
                except Exception:
                    pass

                # Seed invariant Edge/Screw coefficients.
                edgeE, screwE = self._cf_split(coeffs_in)
                coeffs_has_ei = bool(edgeE) or bool(screwE)

                terms = list(self._cf_terms) if getattr(self, "_cf_terms", None) else []
                if not terms:
                    allE = sorted(
                        set(edgeE.keys()) | set(screwE.keys()),
                        key=lambda s: int(s[1:]) if s[1:].isdigit() else 999,
                    )
                    terms = allE

                edge_seed = {}
                screw_seed = {}
                cf_ref_seed = {}
                cf_bnd_seed = {}

                for t in terms:
                    # t is like E1, E2, ...
                    edge_key = f"Edge{t}"
                    screw_key = f"Screw{t}"

                    edge_lc = edge_key.lower()
                    screw_lc = screw_key.lower()

                    # Values: initial_values override contrast_factors
                    edge_seed[edge_key] = edgeE.get(
                        t,
                        initial.get(edge_key, initial.get(edge_lc, 0.0)),
                    )
                    screw_seed[screw_key] = screwE.get(
                        t,
                        initial.get(screw_key, initial.get(screw_lc, 0.0)),
                    )

                    # Refine flags
                    if edge_key in refinable:
                        cf_ref_seed[edge_key] = bool(refinable.get(edge_key, False))
                    elif edge_lc in refinable:
                        cf_ref_seed[edge_key] = bool(refinable.get(edge_lc, False))

                    if screw_key in refinable:
                        cf_ref_seed[screw_key] = bool(refinable.get(screw_key, False))
                    elif screw_lc in refinable:
                        cf_ref_seed[screw_key] = bool(refinable.get(screw_lc, False))

                    # Bounds
                    if edge_key in bounds_norm:
                        cf_bnd_seed[edge_key] = bounds_norm.get(edge_key)
                    elif edge_lc in bounds_norm:
                        cf_bnd_seed[edge_key] = bounds_norm.get(edge_lc)

                    if screw_key in bounds_norm:
                        cf_bnd_seed[screw_key] = bounds_norm.get(screw_key)
                    elif screw_lc in bounds_norm:
                        cf_bnd_seed[screw_key] = bounds_norm.get(screw_lc)

                self._cf_set_terms(
                    terms,
                    edge_seed=edge_seed,
                    screw_seed=screw_seed,
                    refinable_seed=cf_ref_seed,
                    bounds_seed=cf_bnd_seed,
                )

                # Select correct contrast-factor representation after loading.
                # Cubic A/B parameters should reopen the Cubic A/B tab.
                # EdgeE*/ScrewE* parameters should reopen the invariant tabs.
                try:
                    if has_cf_ab_params and getattr(self, "_cf_tab_ab_index", None) is not None:
                        self._tabs_cf.setCurrentIndex(int(self._cf_tab_ab_index))
                    elif has_cf_ei_params:
                        self._tabs_cf.setCurrentIndex(0)
                except Exception:
                    pass

                # Select the correct contrast-factor representation.
                # If input uses legacy cubic A/B, keep GUI on Cubic A/B tab,
                # otherwise saving can delete cedgea/cedgeb/cscrewa/cscrewb.
                try:
                    if coeffs_has_legacy and not coeffs_has_ei and getattr(self, "_cf_tab_ab_index", None) is not None:
                        self._tabs_cf.setCurrentIndex(int(self._cf_tab_ab_index))
                    elif coeffs_has_ei:
                        self._tabs_cf.setCurrentIndex(0)
                except Exception:
                    pass

                try:
                    show_cf = bool(coeffs_in) or bool(has_any_cf_params)
                    self.chk_cf.setChecked(show_cf)
                    self._on_cf_visibility_changed(show_cf)
                except Exception:
                    pass

            except Exception:
                coeffs_in = {}
                coeffs_has_legacy = False
                coeffs_has_ei = False                    

            # Decide micro model in UI from presence of Wilkens params OR CF legacy keys
            coeffs_has_legacy = False
            try:
                if isinstance(coeffs_in, dict):
                    # structured legacy
                    if isinstance(coeffs_in.get("legacy", None), dict) and len(coeffs_in["legacy"]) > 0:
                        coeffs_has_legacy = True
                    # flat legacy keys
                    for k in ("cedgea", "cedgeb", "cscrewa", "cscrewb"):
                        if k in coeffs_in:
                            coeffs_has_legacy = True
            except Exception:
                pass

            pah_present = any(
                str(k).lower() in {"pah_a", "pah_b"}
                for k in (set(initial) | set(refinable))
            )

            wilkens_present = (
                coeffs_has_legacy
                or has_any_cf_params
                or any(str(k).lower() in {"rho", "re", "fe"} for k in (set(initial) | set(refinable)))
            )

            micro_model = str(
                (cfg.get("refinement") or {}).get("microstrain_model", "")
            ).strip().lower()

            try:
                if micro_model in ("pah", "adler-houska", "adler_houska"):
                    self.combo_micro_model.setCurrentIndex(2)
                elif micro_model in ("wilkens", "wilkins", "dislocation"):
                    self.combo_micro_model.setCurrentIndex(1)
                elif pah_present:
                    self.combo_micro_model.setCurrentIndex(2)
                elif wilkens_present:
                    self.combo_micro_model.setCurrentIndex(1)
                else:
                    self.combo_micro_model.setCurrentIndex(0)
            except Exception:
                pass

            # ---- constraints ----
            try:
                constraints = cfg.get("constraints", {}) or {}
                if not isinstance(constraints, dict):
                    constraints = {}

                self.tbl_constraints.setRowCount(0)

                for key, expr in constraints.items():
                    r = self.tbl_constraints.rowCount()
                    self.tbl_constraints.insertRow(r)
                    self.tbl_constraints.setItem(r, 0, QtWidgets.QTableWidgetItem(str(key).strip()))
                    self.tbl_constraints.setItem(r, 1, QtWidgets.QTableWidgetItem(str(expr).strip()))
            except Exception:
                pass

        finally:
            self._block = False

        self.changed.emit()

    def to_updates(self) -> Dict[str, Any]:
        # ---------------------------------------------------------
        # Which microstrain model is currently selected?
        # ---------------------------------------------------------
        idx_micro = self.combo_micro_model.currentIndex()
        use_iso = idx_micro == 0
        use_wilkens = idx_micro == 1
        use_pah = idx_micro == 2

        out: Dict[str, Any] = {
            "files": {
                "structure_file": self.le_structure.text().strip(),
                "gr_data_file": self.le_grdata.text().strip(),
            },
            "range": {
                "r_min": float(self.le_rmin.text().strip() or "1.0"),
                "r_max": float(self.le_rmax.text().strip() or "10.0"),
            },
            "pair_generation": {
                "r_extension": float(self.le_rext.text().strip() or "1.2"),
            },
                        "refinement": {
                "staged_refinement": bool(self.chk_staged.isChecked()),
                "microstrain_model": (
                    "isotropic" if use_iso else
                    "wilkens" if use_wilkens else
                    "pah"
                ),
                "saxs_enabled": bool(self.gb_saxs.isChecked()),
                "saxs_model": str(self.combo_saxs_model.currentText()).strip().lower().replace(" ", "_"),
                "saxs_apply_qdamp": bool(self.chk_saxs_apply_qdamp.isChecked()),
                "saxs_apply_qmax": bool(self.chk_saxs_apply_qmax.isChecked()),
                "compute_pair_contributions_on_finish": bool(self.chk_compute_pair_contrib.isChecked()),
                "compute_warren_on_finish": bool(self.chk_compute_warren.isChecked()),
                "compute_local_trends_on_finish": bool(self.chk_compute_local_trends.isChecked()),
                "export_finite_coordination": bool(self.chk_export_finite_coordination.isChecked()),
                "refinement_r_step_final": float(self.spin_rstep_final.value()),
                "max_nfev_final": int(self.spin_max_nfev.value()),
                "ftol": float(self.spin_ftol.value()),
                "xtol": float(self.spin_xtol.value()),
                "gtol": float(self.spin_gtol.value()),
                "progress_every_sec": float(self.spin_progress.value()),
            },
            "initial_values": {},
            "refinable_parameters": {},
            "bounds": {},
            "contrast_factors": {},
            "constraints": {},

            # NEW: anisotropic/finite crystallite shape settings
            "crystallite_shape": {},
        }

        # ---------------------------------------------------------
        # Constraints table -> dict
        # ---------------------------------------------------------
        try:
            cons: Dict[str, str] = {}

            for r in range(self.tbl_constraints.rowCount()):
                it_k = self.tbl_constraints.item(r, 0)
                it_e = self.tbl_constraints.item(r, 1)

                k = "" if it_k is None else str(it_k.text()).strip()
                e = "" if it_e is None else str(it_e.text()).strip()

                if k and e:
                    cons[k] = e

            out["constraints"] = cons

        except Exception:
            out["constraints"] = {}

        # ---------------------------------------------------------
        # Deletion directives used by MainWindow when writing raw file
        # ---------------------------------------------------------
        out["__delete__"] = {}
        out["__delete_sections__"] = []

        def _add_del(section: str, keys: List[str]) -> None:
            if not keys:
                return

            out["__delete__"].setdefault(section, [])

            for k in keys:
                kk = str(k).strip()
                if kk:
                    out["__delete__"][section].append(kk)

        def _add_del_section(section: str) -> None:
            if section and section not in out["__delete_sections__"]:
                out["__delete_sections__"].append(section)

        # ---------------------------------------------------------
        # Export structural parameters
        # ---------------------------------------------------------
        ini, ref, bnd = self.tbl_struct.extract()
        out["initial_values"].update(ini)
        out["refinable_parameters"].update(ref)
        out["bounds"].update(bnd)

        # ---------------------------------------------------------
        # Export instrumental parameters only if group is enabled
        # ---------------------------------------------------------
        if self.gb_instr.isChecked():
            ini, ref, bnd = self.tbl_instr.extract()
            out["initial_values"].update(ini)
            out["refinable_parameters"].update(ref)
            out["bounds"].update(bnd)

        # ---------------------------------------------------------
        # Export size / crystallite-shape model
        # ---------------------------------------------------------
        try:
            idx_size_model = int(self.combo_size_model.currentIndex())
        except Exception:
            idx_size_model = 0

        use_spherical_size = idx_size_model == 0
        use_cylinder_size = idx_size_model == 1
        use_shape_builder = idx_size_model == 2


        if self.gb_size.isChecked():
        
            if use_shape_builder:
                # New finite-shape / anisotropic-shape mode.
                shape_spec = dict(getattr(self, "_crystallite_shape_spec", {}) or {})
                shape_spec["mode"] = "finite_shape"

                out["crystallite_shape"] = shape_spec

                # In finite-shape mode, conventional spherical size parameters
                # must be completely removed from the input file.
                size_keys = [
                    "d",
                    "d_std",
                    "d_mu",
                    "cyl_diameter",
                    "cyl_thickness",
                    "cyl_thickness_std",
                    "cyl_axis_h",
                    "cyl_axis_k",
                    "cyl_axis_l",
                    "cyl_mu",
                    "cyl_sigma",
                ]

                _add_del("initial_values", size_keys)
                _add_del("refinable_parameters", size_keys)
                _add_del("bounds", size_keys)

                for k in size_keys:
                    out["initial_values"].pop(k, None)
                    out["refinable_parameters"].pop(k, None)
                    out["bounds"].pop(k, None)

            else:
                # Existing conventional spherical size model plus optional
                # cylinder/disk HKL-dependent common-volume model.
                out["crystallite_shape"] = {
                    "mode": "conventional",
                }

                sphere_keys = [
                    "d",
                    "d_std",
                    "d_mu",
                ]

                cyl_keys = [
                    "cyl_diameter",
                    "cyl_thickness",
                    "cyl_thickness_std",
                    "cyl_axis_h",
                    "cyl_axis_k",
                    "cyl_axis_l",
                    "cyl_mu",
                    "cyl_sigma",
                ]

                if use_spherical_size:
                    ini, ref, bnd = self.tbl_size.extract()
                    out["initial_values"].update(ini)
                    out["refinable_parameters"].update(ref)
                    out["bounds"].update(bnd)

                    _add_del("initial_values", cyl_keys)
                    _add_del("refinable_parameters", cyl_keys)
                    _add_del("bounds", cyl_keys)

                    for k in cyl_keys:
                        out["initial_values"].pop(k, None)
                        out["refinable_parameters"].pop(k, None)
                        out["bounds"].pop(k, None)

                elif use_cylinder_size:
                    cyl_ini, cyl_ref, cyl_bnd = self.tbl_cylinder_size.extract()
                    out["initial_values"].update(cyl_ini)
                    out["refinable_parameters"].update(cyl_ref)
                    out["bounds"].update(cyl_bnd)

                    _add_del("initial_values", sphere_keys)
                    _add_del("refinable_parameters", sphere_keys)
                    _add_del("bounds", sphere_keys)

                    for k in sphere_keys:
                        out["initial_values"].pop(k, None)
                        out["refinable_parameters"].pop(k, None)
                        out["bounds"].pop(k, None)

                else:
                    ini, ref, bnd = self.tbl_size.extract()
                    out["initial_values"].update(ini)
                    out["refinable_parameters"].update(ref)
                    out["bounds"].update(bnd)
        else:
            # Size group disabled.
            out["crystallite_shape"] = {
                "mode": "disabled",
            }

        
        # ---------------------------------------------------------
        # Export isotropic strain only for isotropic model
        # ---------------------------------------------------------
        if use_iso and self.gb_strain.isChecked():
            ini, ref, bnd = self.tbl_strain.extract()
            out["initial_values"].update(ini)
            out["refinable_parameters"].update(ref)
            out["bounds"].update(bnd)

        # ---------------------------------------------------------
        # Export SAXS / missing-low-Q belly parameters
        #
        # Important:
        # This is intentionally outside the main size-model branch.
        # The SAXS belly can be spherical even when the peak damping model
        # is cylinder/disk, and vice versa.
        # ---------------------------------------------------------
        if self.gb_saxs.isChecked():
            saxs_ini, saxs_ref, saxs_bnd = self.tbl_saxs.extract()

            out["initial_values"].update(saxs_ini)
            out["refinable_parameters"].update(saxs_ref)
            out["bounds"].update(saxs_bnd)

            
        # ---------------------------------------------------------
        # Export Wilkens parameters only for Wilkens model
        # ---------------------------------------------------------
        if use_wilkens and self.gb_wilkens.isChecked():
            ini, ref, bnd = self.tbl_wilkens.extract()
            out["initial_values"].update(ini)
            out["refinable_parameters"].update(ref)
            out["bounds"].update(bnd)

        # ---------------------------------------------------------
        # Export PAH parameters only for PAH model
        # ---------------------------------------------------------
        if use_pah and self.gb_pah.isChecked():
            ini, ref, bnd = self.tbl_pah.extract()
            out["initial_values"].update(ini)
            out["refinable_parameters"].update(ref)
            out["bounds"].update(bnd)

        # ---------------------------------------------------------
        # Biso mode
        # ---------------------------------------------------------
        use_element_biso = bool(self.chk_biso_by_element.isChecked())

        # ---------------------------------------------------------
        # Export atomic site parameters
        #
        # In element-Biso mode:
        #   export x/y/z/occ site parameters,
        #   but skip site-specific biso_* keys.
        #
        # In site-Biso mode:
        #   export site-specific biso_* keys normally.
        # ---------------------------------------------------------
        try:
            site_ini, site_ref, site_bnd, site_use = self.tbl_sites.extract_all()

            for k, v in (site_use or {}).items():
                self._site_use_flags[str(k)] = bool(v)

            for k, v in (site_ini or {}).items():
                kk = str(k)

                if use_element_biso and kk.lower().startswith("biso_"):
                    continue

                if bool(self._site_use_flags.get(kk, True)):
                    out["initial_values"][kk] = v

            for k, v in (site_ref or {}).items():
                kk = str(k)

                if use_element_biso and kk.lower().startswith("biso_"):
                    continue

                if bool(self._site_use_flags.get(kk, True)):
                    out["refinable_parameters"][kk] = bool(v)

            for k, v in (site_bnd or {}).items():
                kk = str(k)

                if use_element_biso and kk.lower().startswith("biso_"):
                    continue

                if bool(self._site_use_flags.get(kk, True)):
                    out["bounds"][kk] = v

        except Exception:
            pass

        # ---------------------------------------------------------
        # Export element-grouped Biso parameters
        #
        # This writes:
        #   biso_C
        #   biso_N
        #   biso_Co
        #   ...
        #
        # only when element-Biso mode is enabled.
        # ---------------------------------------------------------
        if use_element_biso:
            try:
                elem_ini, elem_ref, elem_bnd = self.tbl_elem_biso.extract()

                out["initial_values"].update(elem_ini)
                out["refinable_parameters"].update(elem_ref)
                out["bounds"].update(elem_bnd)

            except Exception:
                pass

        # ---------------------------------------------------------
        # Export local dynamics parameters
        # ---------------------------------------------------------
        dyn_ini, dyn_ref, dyn_bnd, dyn_use = self.tbl_dyn.extract_all()

        for k, v in (dyn_use or {}).items():
            self._dyn_use_flags[str(k)] = bool(v)

        for k, v in (dyn_ini or {}).items():
            if bool(self._dyn_use_flags.get(str(k), True)):
                out["initial_values"][str(k)] = v

        for k, v in (dyn_ref or {}).items():
            if bool(self._dyn_use_flags.get(str(k), True)):
                out["refinable_parameters"][str(k)] = bool(v)

        for k, v in (dyn_bnd or {}).items():
            if bool(self._dyn_use_flags.get(str(k), True)):
                out["bounds"][str(k)] = v

        # ---------------------------------------------------------
        # Preserve hidden parameters
        #
        # IMPORTANT:
        # Do not let stale hidden values overwrite parameters that are currently
        # visible/exported by GUI tables.
        # ---------------------------------------------------------
        visible_keys_now = (
            {str(k) for k in out.get("initial_values", {}).keys()}
            | {str(k) for k in out.get("refinable_parameters", {}).keys()}
            | {str(k) for k in out.get("bounds", {}).keys()}
        )

        hidden_ini = {
            str(k): v
            for k, v in (getattr(self, "_hidden_ini", {}) or {}).items()
            if str(k) not in visible_keys_now
        }

        hidden_ref = {
            str(k): bool(v)
            for k, v in (getattr(self, "_hidden_ref", {}) or {}).items()
            if str(k) not in visible_keys_now
        }

        hidden_bnd = {
            str(k): v
            for k, v in (getattr(self, "_hidden_bnd", {}) or {}).items()
            if str(k) not in visible_keys_now
        }

        out["initial_values"].update(hidden_ini)
        out["refinable_parameters"].update(hidden_ref)
        out["bounds"].update(hidden_bnd)

        # ---------------------------------------------------------
        # Biso exclusivity deletion directives
        #
        # Element-Biso mode:
        #   keep only biso_C, biso_N, biso_Co, ...
        #   delete known site-specific biso_c3, biso_n55, ...
        #
        # Site-Biso mode:
        #   keep only known site-specific Biso keys
        #   delete element-level biso_C, biso_N, ...
        # ---------------------------------------------------------
        if bool(getattr(self, "_site_records_for_gui", []) or []):
            site_biso_keys = sorted(
                {
                    str(k).strip()
                    for k in (getattr(self, "_site_biso_keys", set()) or set())
                    if str(k).strip()
                }
            )

            elem_biso_keys = sorted(
                {
                    str(k).strip()
                    for k in self._element_biso_keys_from_sites()
                    if str(k).strip()
                }
            )

            site_biso_lc = {k.lower() for k in site_biso_keys}
            elem_biso_lc = {k.lower() for k in elem_biso_keys}

            all_biso_keys_in_out = sorted(
                {
                    str(k)
                    for sec in ("initial_values", "refinable_parameters", "bounds")
                    for k in (out.get(sec, {}) or {}).keys()
                    if str(k).strip().lower().startswith("biso_")
                }
            )

            if use_element_biso:
                keep_lc = elem_biso_lc

                # Explicitly delete known site-Biso keys from raw input,
                # even if they are not present in `out`.
                remove_biso_keys = set(site_biso_keys)

            else:
                keep_lc = site_biso_lc

                # Explicitly delete known element-Biso keys from raw input,
                # even if they are not present in `out`.
                remove_biso_keys = set(elem_biso_keys)

            # Also delete any Biso key currently in `out` that is incompatible
            # with the chosen mode.
            for k in all_biso_keys_in_out:
                if str(k).strip().lower() not in keep_lc:
                    remove_biso_keys.add(str(k).strip())

            remove_biso_keys = sorted(k for k in remove_biso_keys if k)

            if remove_biso_keys:
                _add_del("initial_values", remove_biso_keys)
                _add_del("refinable_parameters", remove_biso_keys)
                _add_del("bounds", remove_biso_keys)

                for k in remove_biso_keys:
                    out["initial_values"].pop(k, None)
                    out["refinable_parameters"].pop(k, None)
                    out["bounds"].pop(k, None)

        # ---------------------------------------------------------
        # Export lambda parameters
        # ---------------------------------------------------------
        lam_ini, lam_ref, lam_bnd = self.lambda_list.extract_all()
        out["initial_values"].update(lam_ini)
        out["refinable_parameters"].update(lam_ref)
        out["bounds"].update(lam_bnd)

        # ---------------------------------------------------------
        # Remove deprecated parameters
        # ---------------------------------------------------------
        deprecated_params = ["qbroad", "eta"]

        for sec in ("initial_values", "refinable_parameters", "bounds"):
            _add_del(sec, deprecated_params)

        for p in deprecated_params:
            out["initial_values"].pop(p, None)
            out["refinable_parameters"].pop(p, None)
            out["bounds"].pop(p, None)

        # ---------------------------------------------------------
        # If local dynamics parameter Use? is unchecked, delete it
        # ---------------------------------------------------------
        dyn_disabled = [
            str(k)
            for k, v in (dyn_use or {}).items()
            if not bool(v)
        ]

        if dyn_disabled:
            _add_del("initial_values", dyn_disabled)
            _add_del("refinable_parameters", dyn_disabled)
            _add_del("bounds", dyn_disabled)

        # ---------------------------------------------------------
        # Constraints: constrained params are derived
        # ---------------------------------------------------------
        cons_keys = list((out.get("constraints") or {}).keys())

        if cons_keys:
            for ck in cons_keys:
                out["refinable_parameters"][str(ck)] = False

            _add_del("bounds", cons_keys)

        # ---------------------------------------------------------
        # Group on/off deletion rules
        # ---------------------------------------------------------

        # If Size group is disabled, remove size parameters completely.
        if not self.gb_size.isChecked():
            size_keys = [
                "d",
                "d_std",
                "d_mu",
                "cyl_diameter",
                "cyl_thickness",
                "cyl_thickness_std",
                "cyl_axis_h",
                "cyl_axis_k",
                "cyl_axis_l",
                "cyl_mu",
                "cyl_sigma",
            ]

            _add_del("initial_values", size_keys)
            _add_del("refinable_parameters", size_keys)
            _add_del("bounds", size_keys)

            for k in size_keys:
                out["initial_values"].pop(k, None)
                out["refinable_parameters"].pop(k, None)
                out["bounds"].pop(k, None)
        # If SAXS / missing-low-Q belly group is disabled, remove SAXS belly parameters.
        if not self.gb_saxs.isChecked():
            saxs_keys = [
                "saxs_scale",
                "saxs_diameter",
                "saxs_diameter_std",
                "saxs_height",
                "saxs_height_std",
            ]

            _add_del("initial_values", saxs_keys)
            _add_del("refinable_parameters", saxs_keys)
            _add_del("bounds", saxs_keys)

            for k in saxs_keys:
                out["initial_values"].pop(k, None)
                out["refinable_parameters"].pop(k, None)
                out["bounds"].pop(k, None)


        # If Instrumental group is disabled, remove instrumental parameters.
        if not self.gb_instr.isChecked():
            instr_ini, instr_ref, instr_bnd = self.tbl_instr.extract()
            instr_keys = list((instr_ini or {}).keys())

            _add_del("initial_values", instr_keys)
            _add_del("refinable_parameters", instr_keys)
            _add_del("bounds", instr_keys)

            for k in instr_keys:
                out["initial_values"].pop(k, None)
                out["refinable_parameters"].pop(k, None)
                out["bounds"].pop(k, None)

        # ---------------------------------------------------------
        # Microstrain model exclusivity
        # ---------------------------------------------------------
        wilk_keys = ["rho", "re", "fe"]
        pah_keys = ["pah_a", "pah_b", "fe"]

        wilk_only_keys = ["rho", "re"]
        pah_only_keys = ["pah_a", "pah_b"]

        if use_wilkens:
            remove_keys = ["delta_g"] + pah_only_keys

            _add_del("initial_values", remove_keys)
            _add_del("refinable_parameters", remove_keys)
            _add_del("bounds", remove_keys)

            for k in remove_keys:
                out["initial_values"].pop(k, None)
                out["refinable_parameters"].pop(k, None)
                out["bounds"].pop(k, None)

            if not self.gb_wilkens.isChecked():
                _add_del("initial_values", wilk_keys)
                _add_del("refinable_parameters", wilk_keys)
                _add_del("bounds", wilk_keys)

                for k in wilk_keys:
                    out["initial_values"].pop(k, None)
                    out["refinable_parameters"].pop(k, None)
                    out["bounds"].pop(k, None)

        elif use_pah:
            remove_keys = ["delta_g"] + wilk_only_keys

            _add_del("initial_values", remove_keys)
            _add_del("refinable_parameters", remove_keys)
            _add_del("bounds", remove_keys)

            for k in remove_keys:
                out["initial_values"].pop(k, None)
                out["refinable_parameters"].pop(k, None)
                out["bounds"].pop(k, None)

            if not self.gb_pah.isChecked():
                _add_del("initial_values", pah_keys)
                _add_del("refinable_parameters", pah_keys)
                _add_del("bounds", pah_keys)

                for k in pah_keys:
                    out["initial_values"].pop(k, None)
                    out["refinable_parameters"].pop(k, None)
                    out["bounds"].pop(k, None)

        else:
            remove_keys = list(set(wilk_keys + pah_only_keys))

            _add_del("initial_values", remove_keys)
            _add_del("refinable_parameters", remove_keys)
            _add_del("bounds", remove_keys)

            for k in remove_keys:
                out["initial_values"].pop(k, None)
                out["refinable_parameters"].pop(k, None)
                out["bounds"].pop(k, None)

            if not self.gb_strain.isChecked():
                _add_del("initial_values", ["delta_g"])
                _add_del("refinable_parameters", ["delta_g"])
                _add_del("bounds", ["delta_g"])

                out["initial_values"].pop("delta_g", None)
                out["refinable_parameters"].pop("delta_g", None)
                out["bounds"].pop("delta_g", None)

        # ---------------------------------------------------------
        # Contrast factors
        # ---------------------------------------------------------
        ab_keys = [
            "CEdgeA",
            "CEdgeB",
            "CScrewA",
            "CScrewB",
            "cedgea",
            "cedgeb",
            "cscrewa",
            "cscrewb",
        ]

        ei_keys = []

        if self._cf_terms:
            ei_keys = (
                [f"Edge{t}" for t in self._cf_terms]
                + [f"Screw{t}" for t in self._cf_terms]
            )

        if (
            (use_wilkens or use_pah)
            and getattr(self, "chk_cf", None) is not None
            and self.chk_cf.isChecked()
        ):
            try:
                use_ab = (
                    getattr(self, "_cf_tab_ab_index", None) is not None
                    and self._tabs_cf.currentIndex() == int(self._cf_tab_ab_index)
                )
            except Exception:
                use_ab = False

            if use_ab:
                try:
                    ab_ini, ab_ref, ab_bnd = self.tbl_cf_ab.extract()
                    out["initial_values"].update(ab_ini)
                    out["refinable_parameters"].update(ab_ref)
                    out["bounds"].update(ab_bnd)
                except Exception:
                    pass

                if ei_keys:
                    _add_del("initial_values", ei_keys)
                    _add_del("refinable_parameters", ei_keys)
                    _add_del("bounds", ei_keys)

                    for k in ei_keys:
                        out["initial_values"].pop(k, None)
                        out["refinable_parameters"].pop(k, None)
                        out["bounds"].pop(k, None)

            else:
                try:
                    ei_ini, ei_ref, ei_bnd = self.tbl_cf_edge.extract()
                    out["initial_values"].update(ei_ini)
                    out["refinable_parameters"].update(ei_ref)
                    out["bounds"].update(ei_bnd)

                    si_ini, si_ref, si_bnd = self.tbl_cf_screw.extract()
                    out["initial_values"].update(si_ini)
                    out["refinable_parameters"].update(si_ref)
                    out["bounds"].update(si_bnd)

                except Exception:
                    pass

                _add_del("initial_values", ab_keys)
                _add_del("refinable_parameters", ab_keys)
                _add_del("bounds", ab_keys)

                for k in ab_keys:
                    out["initial_values"].pop(k, None)
                    out["refinable_parameters"].pop(k, None)
                    out["bounds"].pop(k, None)

        else:
            _add_del("initial_values", ab_keys)
            _add_del("refinable_parameters", ab_keys)
            _add_del("bounds", ab_keys)

            for k in ab_keys:
                out["initial_values"].pop(k, None)
                out["refinable_parameters"].pop(k, None)
                out["bounds"].pop(k, None)

            if ei_keys:
                _add_del("initial_values", ei_keys)
                _add_del("refinable_parameters", ei_keys)
                _add_del("bounds", ei_keys)

                for k in ei_keys:
                    out["initial_values"].pop(k, None)
                    out["refinable_parameters"].pop(k, None)
                    out["bounds"].pop(k, None)

        # Always remove deprecated [contrast_factors] section.
        _add_del_section("contrast_factors")
        out["contrast_factors"] = {}

        return out


    def reset_defaults(self) -> None:
        self._block = True
        try:
            self.le_structure.setText("")
            self.le_grdata.setText("")
            self.le_rmin.setText("1.0")
            self.le_rmax.setText("10.0")
            self.le_rext.setText("1.2")
            self.chk_staged.setChecked(False)
            self.chk_compute_pair_contrib.setChecked(False)
            self.chk_compute_warren.setChecked(False)
            self.chk_compute_local_trends.setChecked(False)
            self.chk_export_finite_coordination.setChecked(False)
            self.spin_rstep_final.setValue(0.01)
            self.spin_max_nfev.setValue(60)
            self.spin_ftol.setValue(1e-3)
            self.spin_xtol.setValue(1e-3)
            self.spin_gtol.setValue(1e-3)
            self.spin_progress.setValue(0.5)
            self._sym_enabled = None
            self._biso_label_overrides = {}
            self._dyn_biso_keys = set()
            self._dyn_use_flags = {}
            self._last_structure_path = ""
            self.combo_micro_model.setCurrentIndex(0)
            # ripristina toggles (2)
            self.gb_instr.setChecked(True)
            self.gb_size.setChecked(True)
            self.gb_strain.setChecked(True)
            self.gb_wilkens.setChecked(False)
            self.gb_pah.setChecked(False)
            self.gb_saxs.setChecked(False)
            self.combo_saxs_model.setCurrentText("sphere")
            self.chk_saxs_apply_qdamp.setChecked(False)
            self.chk_saxs_apply_qmax.setChecked(True)


            try:
                self.chk_cf.setChecked(False)
                self.gb_cf.setVisible(False)
            except Exception:
                pass
            for tbl in (
                self.tbl_struct,
                self.tbl_instr,
                self.tbl_size,
                self.tbl_cylinder_size,
                self.tbl_saxs,
                self.tbl_strain,
                self.tbl_wilkens,
                self.tbl_pah,
                self.tbl_elem_biso,
                self.tbl_dyn,
                self.tbl_sites,
            ):
                tbl.clear_params()

            self._site_records_for_gui = []
            self._site_biso_keys = set()
            self._site_use_flags = {}

            self._crystallite_shape_spec = {}

            try:
                self.combo_size_model.setCurrentIndex(0)
                self._update_shape_summary_label()
            except Exception:
                pass
            
            # FIX: LambdaListWidget.set_lambdas richiede 3 dict
            self.lambda_list.set_lambdas({}, {}, {})
            try:
                self._cf_set_terms([])
            except Exception:
                pass
            self._hidden_ini = {}
            self._hidden_ref = {}
            self._hidden_bnd = {}
            self._seed_defaults()
        finally:
            self._block = False
        self.changed.emit()


