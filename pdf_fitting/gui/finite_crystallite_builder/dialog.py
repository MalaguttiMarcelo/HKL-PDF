from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from pymatgen.core.structure import Structure

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
    shape_dimension_lengths,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    SequenceLayer,
    absolute_sequence_layers,
    default_a_sequence,
    default_ab_sequence,
    default_abc_sequence,
    realize_n_layers,
    stack_thickness_metrics,
)

from pdf_fitting.models.stacking_transitions import (
    transitions_from_stack_spec,
)

from pdf_fitting.models.stacked_particle import (
    build_stacked_particle,
)

from pdf_fitting.models.stacking_disorder import (
    generate_correlated_ab_paths,
    state_transition_statistics,
)

from pdf_fitting.models.stochastic_stacked_particle import (
    build_stochastic_stacked_particle,
)

from .generated_layer_table_model import (
    GeneratedLayerTableModel,
)

from .layer_table_model import (
    HighPrecisionFloatDelegate,
    LayerPlacementTableModel,
)

from .renderer import (
    FiniteCrystallitePreview,
)

from .state import (
    FiniteCrystalliteState,
)

from pdf_fitting.models.stacking_search import (
    StackingSearchSpec,
    stacking_candidate_count,
)


from .stacking_refinement_state import (
    STACKING_TREATMENT_CONTINUOUS,
    STACKING_TREATMENT_FIXED,
    STACKING_TREATMENT_GRID,
    StackingRefinementSpec,
)


print(
    "[FINITE DIALOG FILE]",
    __file__,
    flush=True,
)


class FiniteCrystalliteBuilderDialog(
    QtWidgets.QDialog
):
    """
    Finite crystallite-envelope and periodic layer-stacking builder.

    The editable sequence is infinite and periodic. The crystallite shape
    selects a temporary finite realization for preview and export.
    """

    shapeAccepted = QtCore.Signal(
        dict
    )

    def __init__(
        self,
        parent=None,
        *,
        structure: Optional[Structure] = None,
        initial_state: Optional[
            Dict[str, Any]
        ] = None,
    ):
        super().__init__(
            parent
        )

        self.structure = structure

        self.state = (
            FiniteCrystalliteState.from_dict(
                initial_state
            )
        )

        initial_state = dict(
            initial_state
            or {}
        )

        self.stacking_refinement_state = (
            StackingRefinementSpec.from_dict(
                initial_state.get(
                    "stacking_refinement",
                    {},
                )
            )
        )

        stacking_refinement_values = dict(
            initial_state.get(
                "stacking_refinement",
                {},
            )
            or {}
        )

        self.stochastic_vector_refinement = dict(
            stacking_refinement_values.get(
                "stochastic_vectors",
                {},
            )
            or {}
        )

        self.search_spec = StackingSearchSpec.from_dict(
            initial_state.get(
                "stacking_search",
                {},
            )
        )




        if not hasattr(
            self,
            "_stacking_search_spec_from_ui",
        ):
            raise RuntimeError(
                "FiniteCrystalliteBuilderDialog is missing "
                "_stacking_search_spec_from_ui(). "
                "Check dialog.py indentation and restart the application."
            )

        self._has_fitted_scene = False
        self._syncing_dimensions = False

        self._stochastic_preview_uniforms = None
        self._stochastic_preview_uniforms_key = None

        # Load the saved stacking model before building the UI.
        #
        # combo_start_state does not exist here. It is a local widget created
        # later inside _open_generic_markov_editor(). The saved start state
        # must therefore come from LayerStackSpec.
        loaded_stack_spec = LayerStackSpec.from_dict(
            initial_state.get(
                "stacking",
                {},
            )
        )

        self._generic_markov_start_state = int(
            loaded_stack_spec.markov_start_state
        )

        self._generic_markov_states = [
            state.to_dict()
            for state in loaded_stack_spec.markov_states
        ]

        self._generic_markov_transition_matrix = [
            [
                float(
                    probability
                )
                for probability in row
            ]
            for row in loaded_stack_spec.markov_transition_matrix
        ]

        self._generic_markov_probability_controls = [
            [
                dict(
                    control
                )
                for control in row
            ]
            for row in loaded_stack_spec.markov_probability_controls
        ]

        self._preview_timer = QtCore.QTimer(
            self
        )

        self._preview_timer.setSingleShot(
            True
        )

        self._preview_timer.setInterval(
            100
        )

        self._preview_timer.timeout.connect(
            self.update_preview
        )

        self.setWindowTitle(
            "Build finite crystallite"
        )

        self.setWindowFlags(
            self.windowFlags()
            | Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        self.resize(
            1500,
            920,
        )

        self._build_ui()
        self._load_state_to_ui()
        self._connect_signals()

        self._update_shape_controls()
        self._update_search_controls()
        self._update_stacking_controls()
        self._update_candidate_count()

        QtCore.QTimer.singleShot(
            100,
            self.update_preview,
        )

    def _build_ui(
        self,
    ) -> None:
        root = QtWidgets.QVBoxLayout(
            self
        )

        root.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.main_splitter = (
            QtWidgets.QSplitter(
                Qt.Horizontal,
                self,
            )
        )

        self.main_splitter.setChildrenCollapsible(
            False
        )

        root.addWidget(
            self.main_splitter,
            1,
        )

        self.controls_tabs = QtWidgets.QTabWidget()

        self.geometry_tab = QtWidgets.QWidget()
        self.stacking_tab = QtWidgets.QWidget()
        self.visualization_tab = QtWidgets.QWidget()

        self._build_geometry_tab()
        self._build_stacking_tab()
        self._build_visualization_tab()

        self.geometry_scroll = QtWidgets.QScrollArea()
        self.geometry_scroll.setWidgetResizable(
            True
        )
        self.geometry_scroll.setFrameShape(
            QtWidgets.QFrame.NoFrame
        )
        self.geometry_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self.geometry_scroll.setWidget(
            self.geometry_tab
        )

        self.stacking_scroll = QtWidgets.QScrollArea()
        self.stacking_scroll.setWidgetResizable(
            True
        )
        self.stacking_scroll.setFrameShape(
            QtWidgets.QFrame.NoFrame
        )
        self.stacking_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        self.stacking_scroll.setWidget(
            self.stacking_tab
        )

        self.visualization_scroll = QtWidgets.QScrollArea()
        self.visualization_scroll.setWidgetResizable(
            True
        )
        self.visualization_scroll.setFrameShape(
            QtWidgets.QFrame.NoFrame
        )
        self.visualization_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self.visualization_scroll.setWidget(
            self.visualization_tab
        )

        self.controls_tabs.addTab(
            self.geometry_scroll,
            "Crystallite shape",
        )

        self.controls_tabs.addTab(
            self.stacking_scroll,
            "Layer stacking",
        )

        self.controls_tabs.addTab(
            self.visualization_scroll,
            "Visualization and export",
        )

        self.controls_tabs.setMinimumWidth(
            680
        )

        # ---------------------------------------------------------
        # Persistent left-panel footer
        #
        # The buttons are outside the tab widget and outside the scroll area.
        # They therefore remain visible from both the Crystallite shape and
        # Layer stacking tabs.
        # ---------------------------------------------------------
        self.controls_panel = QtWidgets.QWidget()

        controls_panel_layout = QtWidgets.QVBoxLayout(
            self.controls_panel
        )

        controls_panel_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        controls_panel_layout.setSpacing(
            0
        )

        controls_panel_layout.addWidget(
            self.controls_tabs,
            1,
        )

        footer_widget = QtWidgets.QWidget()

        footer_layout = QtWidgets.QHBoxLayout(
            footer_widget
        )

        footer_layout.setContentsMargins(
            8,
            8,
            8,
            8,
        )

        footer_layout.setSpacing(
            8
        )

        self.btn_apply = QtWidgets.QPushButton(
            "Apply to input"
        )

        self.btn_close = QtWidgets.QPushButton(
            "Close"
        )

        self.btn_apply.setMinimumHeight(
            38
        )

        self.btn_close.setMinimumHeight(
            38
        )

        self.btn_apply.setDefault(
            True
        )

        footer_layout.addWidget(
            self.btn_apply,
            1,
        )

        footer_layout.addWidget(
            self.btn_close
        )

        controls_panel_layout.addWidget(
            footer_widget,
            0,
        )

        self.controls_panel.setMinimumWidth(
            680
        )

        self.preview = (
            FiniteCrystallitePreview(
                self
            )
        )

        self.main_splitter.addWidget(
            self.controls_panel
        )

        self.main_splitter.addWidget(
            self.preview
        )

        self.main_splitter.setStretchFactor(
            0,
            4,
        )

        self.main_splitter.setStretchFactor(
            1,
            6,
        )

        self.main_splitter.setSizes(
            [
                720,
                780,
            ]
        )


    def _make_dimension_widgets(
        self,
    ):
        cells = QtWidgets.QDoubleSpinBox()
        angstrom = QtWidgets.QDoubleSpinBox()

        cells.setRange(
            0.001,
            100000.0,
        )

        cells.setDecimals(
            4
        )

        cells.setSingleStep(
            1.0
        )

        cells.setSuffix(
            " cells"
        )

        angstrom.setRange(
            0.001,
            1000000.0,
        )

        angstrom.setDecimals(
            4
        )

        angstrom.setSingleStep(
            1.0
        )

        angstrom.setSuffix(
            " Å"
        )

        return (
            cells,
            angstrom,
        )

    def _build_geometry_tab(
        self,
    ) -> None:
        layout = QtWidgets.QVBoxLayout(
            self.geometry_tab
        )

        layout.setContentsMargins(
            6,
            6,
            6,
            6,
        )

        envelope_group = QtWidgets.QGroupBox(
            "Crystallite clipping envelope"
        )

        envelope_layout = QtWidgets.QGridLayout(
            envelope_group
        )

        self.combo_shape = QtWidgets.QComboBox()

        self.combo_shape.addItems(
            [
                "cylinder",
                "disk",
                "sphere",
                "ellipsoid",
                "prism",
            ]
        )

        envelope_layout.addWidget(
            QtWidgets.QLabel(
                "Shape:"
            ),
            0,
            0,
        )

        envelope_layout.addWidget(
            self.combo_shape,
            0,
            1,
            1,
            2,
        )

        envelope_layout.addWidget(
            QtWidgets.QLabel(
                "Dimension"
            ),
            1,
            0,
        )

        envelope_layout.addWidget(
            QtWidgets.QLabel(
                "Unit-cell lengths"
            ),
            1,
            1,
        )

        envelope_layout.addWidget(
            QtWidgets.QLabel(
                "Angstrom"
            ),
            1,
            2,
        )

        (
            self.spin_axis1,
            self.spin_axis1_angstrom,
        ) = self._make_dimension_widgets()

        (
            self.spin_axis2,
            self.spin_axis2_angstrom,
        ) = self._make_dimension_widgets()

        (
            self.spin_axis3,
            self.spin_axis3_angstrom,
        ) = self._make_dimension_widgets()

        self.label_axis1 = QtWidgets.QLabel(
            "Base axis 1"
        )

        self.label_axis2 = QtWidgets.QLabel(
            "Base axis 2"
        )

        self.label_axis3 = QtWidgets.QLabel(
            "Stack axis"
        )

        for row, (
            label,
            cells,
            angstrom,
        ) in enumerate(
            (
                (
                    self.label_axis1,
                    self.spin_axis1,
                    self.spin_axis1_angstrom,
                ),
                (
                    self.label_axis2,
                    self.spin_axis2,
                    self.spin_axis2_angstrom,
                ),
                (
                    self.label_axis3,
                    self.spin_axis3,
                    self.spin_axis3_angstrom,
                ),
            ),
            start=2,
        ):
            envelope_layout.addWidget(
                label,
                row,
                0,
            )

            envelope_layout.addWidget(
                cells,
                row,
                1,
            )

            envelope_layout.addWidget(
                angstrom,
                row,
                2,
            )

        layout.addWidget(
            envelope_group
        )

        orientation_group = QtWidgets.QGroupBox(
            "Shape orientation"
        )

        orientation_layout = QtWidgets.QGridLayout(
            orientation_group
        )

        orientation_layout.addWidget(
            QtWidgets.QLabel(
                "Direction"
            ),
            0,
            0,
        )

        for column, text in enumerate(
            (
                "h",
                "k",
                "l",
            ),
            start=1,
        ):
            orientation_layout.addWidget(
                QtWidgets.QLabel(
                    text
                ),
                0,
                column,
            )

        self.spin_axis_h = QtWidgets.QSpinBox()
        self.spin_axis_k = QtWidgets.QSpinBox()
        self.spin_axis_l = QtWidgets.QSpinBox()

        self.spin_base1_h = QtWidgets.QSpinBox()
        self.spin_base1_k = QtWidgets.QSpinBox()
        self.spin_base1_l = QtWidgets.QSpinBox()

        self.spin_base2_h = QtWidgets.QSpinBox()
        self.spin_base2_k = QtWidgets.QSpinBox()
        self.spin_base2_l = QtWidgets.QSpinBox()

        self._orientation_widgets = (
            self.spin_axis_h,
            self.spin_axis_k,
            self.spin_axis_l,
            self.spin_base1_h,
            self.spin_base1_k,
            self.spin_base1_l,
            self.spin_base2_h,
            self.spin_base2_k,
            self.spin_base2_l,
        )

        for widget in self._orientation_widgets:
            widget.setRange(
                -20,
                20,
            )

        orientation_rows = (
            (
                "Stack / shape axis",
                self.spin_axis_h,
                self.spin_axis_k,
                self.spin_axis_l,
            ),
            (
                "Base direction 1",
                self.spin_base1_h,
                self.spin_base1_k,
                self.spin_base1_l,
            ),
            (
                "Base direction 2",
                self.spin_base2_h,
                self.spin_base2_k,
                self.spin_base2_l,
            ),
        )

        for row, values in enumerate(
            orientation_rows,
            start=1,
        ):
            orientation_layout.addWidget(
                QtWidgets.QLabel(
                    values[0]
                ),
                row,
                0,
            )

            for column, widget in enumerate(
                values[1:],
                start=1,
            ):
                orientation_layout.addWidget(
                    widget,
                    row,
                    column,
                )

        layout.addWidget(
            orientation_group
        )

        search_group = QtWidgets.QGroupBox(
            "Crystallite-size search"
        )

        search_layout = QtWidgets.QGridLayout(
            search_group
        )

        self.combo_search_mode = QtWidgets.QComboBox()

        self.combo_search_mode.addItem(
            "Pre-defined dimensions",
            "predefined",
        )

        self.combo_search_mode.addItem(
            "Grid search / scan dimensions",
            "grid_search",
        )

        search_layout.addWidget(
            QtWidgets.QLabel(
                "Mode:"
            ),
            0,
            0,
        )

        search_layout.addWidget(
            self.combo_search_mode,
            0,
            1,
            1,
            4,
        )

        headers = (
            "Scan",
            "Minimum",
            "Maximum",
            "Step",
        )

        for column, text in enumerate(
            headers,
            start=1,
        ):
            search_layout.addWidget(
                QtWidgets.QLabel(
                    text
                ),
                1,
                column,
            )

        self.chk_scan_axis1 = QtWidgets.QCheckBox()
        self.chk_scan_axis2 = QtWidgets.QCheckBox()
        self.chk_scan_axis3 = QtWidgets.QCheckBox()

        self.spin_axis1_min = QtWidgets.QDoubleSpinBox()
        self.spin_axis1_max = QtWidgets.QDoubleSpinBox()
        self.spin_axis1_step = QtWidgets.QDoubleSpinBox()

        self.spin_axis2_min = QtWidgets.QDoubleSpinBox()
        self.spin_axis2_max = QtWidgets.QDoubleSpinBox()
        self.spin_axis2_step = QtWidgets.QDoubleSpinBox()

        self.spin_axis3_min = QtWidgets.QDoubleSpinBox()
        self.spin_axis3_max = QtWidgets.QDoubleSpinBox()
        self.spin_axis3_step = QtWidgets.QDoubleSpinBox()

        self._search_spinboxes = (
            self.spin_axis1_min,
            self.spin_axis1_max,
            self.spin_axis1_step,
            self.spin_axis2_min,
            self.spin_axis2_max,
            self.spin_axis2_step,
            self.spin_axis3_min,
            self.spin_axis3_max,
            self.spin_axis3_step,
        )

        for widget in self._search_spinboxes:
            widget.setRange(
                0.001,
                100000.0,
            )

            widget.setDecimals(
                4
            )

            widget.setSingleStep(
                1.0
            )

            widget.setSuffix(
                " cells"
            )

        self.label_search_axis1 = QtWidgets.QLabel(
            "Base axis 1"
        )

        self.label_search_axis2 = QtWidgets.QLabel(
            "Base axis 2"
        )

        self.label_search_axis3 = QtWidgets.QLabel(
            "Stack axis"
        )

        search_rows = (
            (
                self.label_search_axis1,
                self.chk_scan_axis1,
                self.spin_axis1_min,
                self.spin_axis1_max,
                self.spin_axis1_step,
            ),
            (
                self.label_search_axis2,
                self.chk_scan_axis2,
                self.spin_axis2_min,
                self.spin_axis2_max,
                self.spin_axis2_step,
            ),
            (
                self.label_search_axis3,
                self.chk_scan_axis3,
                self.spin_axis3_min,
                self.spin_axis3_max,
                self.spin_axis3_step,
            ),
        )

        for row, values in enumerate(
            search_rows,
            start=2,
        ):
            search_layout.addWidget(
                values[0],
                row,
                0,
            )

            for column, widget in enumerate(
                values[1:],
                start=1,
            ):
                search_layout.addWidget(
                    widget,
                    row,
                    column,
                )

        self.label_candidate_count = QtWidgets.QLabel(
            "Shape candidates: 1"
        )

        search_layout.addWidget(
            self.label_candidate_count,
            5,
            0,
            1,
            5,
        )

        layout.addWidget(
            search_group
        )

        self.label_info = QtWidgets.QLabel()
        self.label_info.setWordWrap(
            True
        )

        layout.addWidget(
            self.label_info
        )

        layout.addStretch(
            1
        )

    def _build_visualization_tab(
        self,
    ) -> None:
        layout = QtWidgets.QVBoxLayout(
            self.visualization_tab
        )

        layout.setContentsMargins(
            6,
            6,
            6,
            6,
        )

        camera_group = QtWidgets.QGroupBox(
            "Camera orientation"
        )

        camera_layout = QtWidgets.QGridLayout(
            camera_group
        )

        self.btn_view_ac = QtWidgets.QPushButton(
            "a horizontal / c vertical"
        )

        self.btn_view_bc = QtWidgets.QPushButton(
            "b horizontal / c vertical"
        )

        self.btn_view_ab = QtWidgets.QPushButton(
            "a horizontal / b vertical"
        )

        self.btn_view_stack_axis = QtWidgets.QPushButton(
            "Stack axis vertical"
        )

        self.btn_view_isometric = QtWidgets.QPushButton(
            "Isometric view"
        )

        self.btn_fit_stacking_view = QtWidgets.QPushButton(
            "Fit scene"
        )

        camera_layout.addWidget(
            QtWidgets.QLabel(
                "Default layered-material view:"
            ),
            0,
            0,
            1,
            2,
        )

        camera_layout.addWidget(
            self.btn_view_ac,
            1,
            0,
            1,
            2,
        )

        camera_layout.addWidget(
            self.btn_view_bc,
            2,
            0,
        )

        camera_layout.addWidget(
            self.btn_view_ab,
            2,
            1,
        )

        camera_layout.addWidget(
            self.btn_view_stack_axis,
            3,
            0,
        )

        camera_layout.addWidget(
            self.btn_view_isometric,
            3,
            1,
        )

        camera_layout.addWidget(
            self.btn_fit_stacking_view,
            4,
            0,
            1,
            2,
        )

        layout.addWidget(
            camera_group
        )

        custom_group = QtWidgets.QGroupBox(
            "Custom crystallographic view"
        )

        custom_layout = QtWidgets.QGridLayout(
            custom_group
        )

        self.combo_custom_view_mode = QtWidgets.QComboBox()

        self.combo_custom_view_mode.addItem(
            "Look along direct direction [u v w]",
            "direction",
        )

        self.combo_custom_view_mode.addItem(
            "Look normal to plane (h k l)",
            "plane_normal",
        )

        self.spin_view_h = QtWidgets.QSpinBox()
        self.spin_view_k = QtWidgets.QSpinBox()
        self.spin_view_l = QtWidgets.QSpinBox()

        self.spin_view_horizontal_h = QtWidgets.QSpinBox()
        self.spin_view_horizontal_k = QtWidgets.QSpinBox()
        self.spin_view_horizontal_l = QtWidgets.QSpinBox()

        for widget in (
            self.spin_view_h,
            self.spin_view_k,
            self.spin_view_l,
            self.spin_view_horizontal_h,
            self.spin_view_horizontal_k,
            self.spin_view_horizontal_l,
        ):
            widget.setRange(
                -20,
                20,
            )

        self.spin_view_h.setValue(
            0
        )

        self.spin_view_k.setValue(
            1
        )

        self.spin_view_l.setValue(
            0
        )

        self.spin_view_horizontal_h.setValue(
            1
        )

        self.spin_view_horizontal_k.setValue(
            0
        )

        self.spin_view_horizontal_l.setValue(
            0
        )

        self.btn_apply_custom_view = QtWidgets.QPushButton(
            "Apply custom view"
        )

        custom_layout.addWidget(
            QtWidgets.QLabel(
                "View mode:"
            ),
            0,
            0,
        )

        custom_layout.addWidget(
            self.combo_custom_view_mode,
            0,
            1,
            1,
            3,
        )

        custom_layout.addWidget(
            QtWidgets.QLabel(
                "View direction:"
            ),
            1,
            0,
        )

        custom_layout.addWidget(
            self.spin_view_h,
            1,
            1,
        )

        custom_layout.addWidget(
            self.spin_view_k,
            1,
            2,
        )

        custom_layout.addWidget(
            self.spin_view_l,
            1,
            3,
        )

        custom_layout.addWidget(
            QtWidgets.QLabel(
                "Horizontal direction:"
            ),
            2,
            0,
        )

        custom_layout.addWidget(
            self.spin_view_horizontal_h,
            2,
            1,
        )

        custom_layout.addWidget(
            self.spin_view_horizontal_k,
            2,
            2,
        )

        custom_layout.addWidget(
            self.spin_view_horizontal_l,
            2,
            3,
        )

        custom_layout.addWidget(
            self.btn_apply_custom_view,
            3,
            0,
            1,
            4,
        )

        layout.addWidget(
            custom_group
        )

        display_group = QtWidgets.QGroupBox(
            "Preview display"
        )

        display_layout = QtWidgets.QFormLayout(
            display_group
        )

        self.chk_show_shape_surface = QtWidgets.QCheckBox(
            "Show crystallite envelope"
        )

        self.chk_show_shape_axes = QtWidgets.QCheckBox(
            "Show shape axes"
        )

        self.chk_show_atoms = QtWidgets.QCheckBox(
            "Show atoms"
        )

        self.chk_show_layer_origins = QtWidgets.QCheckBox(
            "Show layer origins"
        )

        self.chk_show_layer_labels = QtWidgets.QCheckBox(
            "Show layer labels"
        )

        self.chk_show_transition_vectors = QtWidgets.QCheckBox(
            "Show stacking vectors"
        )

        self.chk_color_stacking_layers = QtWidgets.QCheckBox(
            "Color atoms by sequence row"
        )

        self.chk_show_parent_unit_cell = QtWidgets.QCheckBox(
            "Show parent unit cell"
        )

        self.chk_show_lattice_vectors = QtWidgets.QCheckBox(
            "Show parent a, b, c vectors"
        )

        self.chk_stacking_orthographic = QtWidgets.QCheckBox(
            "Use orthographic projection"
        )

        self.spin_shape_opacity = QtWidgets.QDoubleSpinBox()
        self.spin_shape_opacity.setRange(
            0.02,
            0.90,
        )
        self.spin_shape_opacity.setDecimals(
            2
        )

        self.spin_atom_limit = QtWidgets.QSpinBox()
        self.spin_atom_limit.setRange(
            1000,
            1000000,
        )
        self.spin_atom_limit.setSingleStep(
            10000
        )

        self.spin_max_layer_labels = QtWidgets.QSpinBox()
        self.spin_max_layer_labels.setRange(
            1,
            10000,
        )

        self.spin_layer_label_font_size = QtWidgets.QSpinBox()
        self.spin_layer_label_font_size.setRange(
            6,
            48,
        )

        display_layout.addRow(
            self.chk_show_atoms
        )

        display_layout.addRow(
            self.chk_show_shape_surface
        )

        display_layout.addRow(
            self.chk_show_shape_axes
        )

        display_layout.addRow(
            self.chk_show_layer_origins
        )

        display_layout.addRow(
            self.chk_show_layer_labels
        )

        display_layout.addRow(
            self.chk_show_transition_vectors
        )

        display_layout.addRow(
            self.chk_color_stacking_layers
        )

        display_layout.addRow(
            self.chk_show_parent_unit_cell
        )

        display_layout.addRow(
            self.chk_show_lattice_vectors
        )

        display_layout.addRow(
            self.chk_stacking_orthographic
        )

        display_layout.addRow(
            "Envelope opacity:",
            self.spin_shape_opacity,
        )

        display_layout.addRow(
            "Maximum preview atoms:",
            self.spin_atom_limit,
        )

        display_layout.addRow(
            "Maximum layer labels:",
            self.spin_max_layer_labels,
        )

        display_layout.addRow(
            "Layer label font size:",
            self.spin_layer_label_font_size,
        )

        layout.addWidget(
            display_group
        )

        inspector_group = QtWidgets.QGroupBox(
            "Generated layers"
        )

        inspector_layout = QtWidgets.QVBoxLayout(
            inspector_group
        )

        self.chk_show_generated_table = QtWidgets.QCheckBox(
            "Show generated-layer inspector"
        )

        self.generated_layer_model = GeneratedLayerTableModel(
            self
        )

        self.generated_layer_table = QtWidgets.QTableView()

        self.generated_layer_table.setModel(
            self.generated_layer_model
        )

        self.generated_layer_table.setAlternatingRowColors(
            True
        )

        self.generated_layer_table.verticalHeader().setVisible(
            False
        )

        self.generated_layer_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )

        self.generated_layer_table.horizontalHeader().setStretchLastSection(
            True
        )

        self.generated_layer_table.setMinimumHeight(
            220
        )

        inspector_layout.addWidget(
            self.chk_show_generated_table
        )

        inspector_layout.addWidget(
            self.generated_layer_table
        )

        layout.addWidget(
            inspector_group
        )

        export_group = QtWidgets.QGroupBox(
            "Export"
        )

        export_layout = QtWidgets.QGridLayout(
            export_group
        )

        self.btn_export_stacked_xyz = QtWidgets.QPushButton(
            "Export preview particle as XYZ..."
        )

        self.btn_export_prism_periodic_cif = QtWidgets.QPushButton(
            "Export finite prism as periodic CIF..."
        )

        self.btn_export_prism_periodic_cif.setToolTip(
            "Create a periodic CIF supercell matching the current prism "
            "dimensions and crystallographic directions.\n\n"
            "The exported CIF axes are:\n"
            "  a = prism base direction 1\n"
            "  b = prism base direction 2\n"
            "  c = prism axis\n\n"
            "This is intended for later use with the conventional "
            "cylinder/disk common-volume model.\n\n"
            "Only prism shapes with integer cell dimensions are supported."
        )

        self.btn_export_layer_csv = QtWidgets.QPushButton(
            "Export realized layers as CSV..."
        )

        self.btn_export_transition_csv = QtWidgets.QPushButton(
            "Export stacking vectors as CSV..."
        )

        self.btn_export_preview_png = QtWidgets.QPushButton(
            "Save preview screenshot as PNG..."
        )

        export_layout.addWidget(
            self.btn_export_stacked_xyz,
            0,
            0,
            1,
            2,
        )

        export_layout.addWidget(
            self.btn_export_prism_periodic_cif,
            1,
            0,
            1,
            2,
        )

        export_layout.addWidget(
            self.btn_export_layer_csv,
            2,
            0,
        )

        export_layout.addWidget(
            self.btn_export_transition_csv,
            2,
            1,
        )

        export_layout.addWidget(
            self.btn_export_preview_png,
            3,
            0,
            1,
            2,
        )

        export_layout.addWidget(
            self.btn_export_transition_csv,
            1,
            1,
        )

        export_layout.addWidget(
            self.btn_export_preview_png,
            2,
            0,
            1,
            2,
        )

        layout.addWidget(
            export_group
        )

    def _build_stacking_tab(
        self,
    ) -> None:
        layout = QtWidgets.QVBoxLayout(
            self.stacking_tab
        )

        self.stacking_vertical_splitter = QtWidgets.QSplitter(
            Qt.Vertical,
            self.stacking_tab,
        )

        self.stacking_vertical_splitter.setChildrenCollapsible(
            False
        )

        layout.addWidget(
            self.stacking_vertical_splitter,
            1,
        )

        upper_widget = QtWidgets.QWidget()
        upper_layout = QtWidgets.QVBoxLayout(
            upper_widget
        )

        upper_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        lower_widget = QtWidgets.QWidget()
        lower_layout = QtWidgets.QVBoxLayout(
            lower_widget
        )

        lower_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.setContentsMargins(
            6,
            6,
            6,
            6,
        )

        enable_group = QtWidgets.QGroupBox(
            "Infinite periodic stacking definition"
        )

        enable_layout = QtWidgets.QFormLayout(
            enable_group
        )

        self.chk_stacking_enabled = QtWidgets.QCheckBox(
            "Enable periodic layer stacking"
        )

        self.combo_definition_mode = QtWidgets.QComboBox()

        self.combo_definition_mode.addItem(
            "Repeating motif",
            "repeating_motif",
        )

        self.combo_definition_mode.addItem(
            "Repeating explicit sequence",
            "repeating_explicit_sequence",
        )

        explanation = QtWidgets.QLabel(
            "The rows below define one repeating period. The sequence "
            "continues infinitely in both stacking directions. The current "
            "crystallite envelope selects the finite layers visible in the "
            "preview."
        )

        explanation.setWordWrap(
            True
        )

        enable_layout.addRow(
            self.chk_stacking_enabled
        )

        enable_layout.addRow(
            "Definition mode:",
            self.combo_definition_mode,
        )

        self.combo_vector_mode = QtWidgets.QComboBox()

        self.combo_vector_mode.addItem(
            "Incremental vectors from previous layer",
            "incremental",
        )

        self.combo_vector_mode.addItem(
            "Absolute positions from sequence origin",
            "absolute",
        )

        enable_layout.addRow(
            "Vector interpretation:",
            self.combo_vector_mode,
        )

        enable_layout.addRow(
            explanation
        )

        upper_layout.addWidget(
            enable_group
        )

        stochastic_group = QtWidgets.QGroupBox(
            "Stacking model"
        )

        stochastic_layout = QtWidgets.QGridLayout(
            stochastic_group
        )

        self.combo_stacking_model = QtWidgets.QComboBox()

        self.combo_stacking_model.addItem(
            "Deterministic explicit sequence",
            "periodic_sequence",
        )

        self.combo_stacking_model.addItem(
            "Correlated stochastic A/B stacking (TOPAS-like)",
            "correlated_ab_disorder",
        )

        self.combo_stacking_model.addItem(
            "Generic multi-state Markov stacking",
            "generic_markov",
        )

        self.spin_p_alternate = QtWidgets.QDoubleSpinBox()
        self.spin_p_alternate.setRange(
            0.0,
            1.0,
        )
        self.spin_p_alternate.setDecimals(
            6
        )
        self.spin_p_alternate.setSingleStep(
            0.01
        )
        self.spin_p_alternate.setToolTip(
            "TOPAS P_alt.\n\n"
            "1.0: A-B-A-B perfect alternation.\n"
            "0.5: random next-state selection.\n"
            "0.0: same state persists."
        )

        self.label_p_same_state = QtWidgets.QLabel(
            "P(same state) = 0.000000"
        )

        self.chk_p_grid_enabled = QtWidgets.QCheckBox(
            "Grid-search P(alternate)"
        )

        self.spin_p_grid_min = QtWidgets.QDoubleSpinBox()
        self.spin_p_grid_max = QtWidgets.QDoubleSpinBox()
        self.spin_p_grid_step = QtWidgets.QDoubleSpinBox()

        for widget in (
            self.spin_p_grid_min,
            self.spin_p_grid_max,
            self.spin_p_grid_step,
        ):
            widget.setRange(
                0.0,
                1.0,
            )
            widget.setDecimals(
                6
            )
            widget.setSingleStep(
                0.01
            )

        self.spin_p_grid_step.setMinimum(
            0.000001
        )

        self.spin_ensemble_samples = QtWidgets.QSpinBox()
        self.spin_ensemble_samples.setRange(
            1,
            1000,
        )

        self.spin_ensemble_seed = QtWidgets.QSpinBox()
        self.spin_ensemble_seed.setRange(
            0,
            2147483647,
        )

        self.combo_ensemble_start_state = QtWidgets.QComboBox()
        self.combo_ensemble_start_state.addItems(
            [
                "A",
                "B",
            ]
        )

        self.spin_preview_realization = QtWidgets.QSpinBox()
        self.spin_preview_realization.setRange(
            0,
            999,
        )

        self.btn_regenerate_stochastic_stack = QtWidgets.QPushButton(
            "Regenerate random stacking"
        )

        self.btn_refine_stochastic_vectors = QtWidgets.QPushButton(
            "Refine State A/B vectors..."
        )

        self.btn_edit_generic_markov_states = QtWidgets.QPushButton(
            "Edit generic Markov states and probabilities..."
        )


        self.btn_edit_generic_markov_states.setToolTip(
            "Add, remove, or edit arbitrary stacking states and their "
            "transition probability matrix."
        )

        self.btn_refine_stochastic_vectors.setToolTip(
            "Choose which stochastic State A and State B vector components "
            "(Δu, Δv, h) are refined, and define their optimizer bounds."
        )

        self.label_stochastic_summary = QtWidgets.QLabel()
        self.label_stochastic_summary.setWordWrap(
            True
        )

        self.spin_state_a_du = QtWidgets.QDoubleSpinBox()
        self.spin_state_a_dv = QtWidgets.QDoubleSpinBox()
        self.spin_state_a_h = QtWidgets.QDoubleSpinBox()

        self.spin_state_b_du = QtWidgets.QDoubleSpinBox()
        self.spin_state_b_dv = QtWidgets.QDoubleSpinBox()
        self.spin_state_b_h = QtWidgets.QDoubleSpinBox()

        for widget in (
            self.spin_state_a_du,
            self.spin_state_a_dv,
            self.spin_state_b_du,
            self.spin_state_b_dv,
        ):
            widget.setRange(
                -2.0,
                2.0,
            )
            widget.setDecimals(
                10
            )
            widget.setSingleStep(
                0.001
            )

        for widget in (
            self.spin_state_a_h,
            self.spin_state_b_h,
        ):
            widget.setRange(
                0.001,
                100.0,
            )
            widget.setDecimals(
                10
            )
            widget.setSingleStep(
                0.01
            )
            widget.setSuffix(
                " Å"
            )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "Stacking model:"
            ),
            0,
            0,
        )

        stochastic_layout.addWidget(
            self.combo_stacking_model,
            0,
            1,
            1,
            3,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "P(alternate):"
            ),
            1,
            0,
        )

        stochastic_layout.addWidget(
            self.spin_p_alternate,
            1,
            1,
        )

        stochastic_layout.addWidget(
            self.label_p_same_state,
            1,
            2,
            1,
            2,
        )

        stochastic_layout.addWidget(
            self.chk_p_grid_enabled,
            2,
            0,
            1,
            2,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "P min:"
            ),
            2,
            2,
        )

        stochastic_layout.addWidget(
            self.spin_p_grid_min,
            2,
            3,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "P max:"
            ),
            3,
            0,
        )

        stochastic_layout.addWidget(
            self.spin_p_grid_max,
            3,
            1,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "P step:"
            ),
            3,
            2,
        )

        stochastic_layout.addWidget(
            self.spin_p_grid_step,
            3,
            3,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "Ensemble realizations:"
            ),
            4,
            0,
        )

        stochastic_layout.addWidget(
            self.spin_ensemble_samples,
            4,
            1,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "Random seed:"
            ),
            4,
            2,
        )

        stochastic_layout.addWidget(
            self.spin_ensemble_seed,
            4,
            3,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "Start state:"
            ),
            5,
            0,
        )

        stochastic_layout.addWidget(
            self.combo_ensemble_start_state,
            5,
            1,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "Preview realization:"
            ),
            5,
            2,
        )

        stochastic_layout.addWidget(
            self.spin_preview_realization,
            5,
            3,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "State A vector:"
            ),
            6,
            0,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "Δu"
            ),
            6,
            1,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "Δv"
            ),
            6,
            2,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "h / Å"
            ),
            6,
            3,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "A emits:"
            ),
            7,
            0,
        )

        stochastic_layout.addWidget(
            self.spin_state_a_du,
            7,
            1,
        )

        stochastic_layout.addWidget(
            self.spin_state_a_dv,
            7,
            2,
        )

        stochastic_layout.addWidget(
            self.spin_state_a_h,
            7,
            3,
        )

        stochastic_layout.addWidget(
            QtWidgets.QLabel(
                "B emits:"
            ),
            8,
            0,
        )

        stochastic_layout.addWidget(
            self.spin_state_b_du,
            8,
            1,
        )

        stochastic_layout.addWidget(
            self.spin_state_b_dv,
            8,
            2,
        )

        stochastic_layout.addWidget(
            self.spin_state_b_h,
            8,
            3,
        )

        stochastic_layout.addWidget(
            self.btn_regenerate_stochastic_stack,
            9,
            0,
            1,
            4,
        )

        stochastic_layout.addWidget(
            self.btn_refine_stochastic_vectors,
            10,
            0,
            1,
            4,
        )

        stochastic_layout.addWidget(
            self.btn_edit_generic_markov_states,
            11,
            0,
            1,
            4,
        )

        stochastic_layout.addWidget(
            self.label_stochastic_summary,
            12,
            0,
            1,
            4,
        )


        upper_layout.addWidget(
            stochastic_group
        )

        self.stochastic_group = stochastic_group

        search_group = QtWidgets.QGroupBox(
            "Stacking search"
        )

        search_layout = QtWidgets.QGridLayout(
            search_group
        )

        self.combo_spacing_treatment = QtWidgets.QComboBox()
        self.combo_spacing_treatment.addItem(
            "Fixed spacing",
            "fixed",
        )
        self.combo_spacing_treatment.addItem(
            "Continuous refinement",
            "continuous",
        )
        self.combo_spacing_treatment.addItem(
            "Discrete spacing grid",
            "grid",
        )

        self.spin_spacing_search_min = QtWidgets.QDoubleSpinBox()
        self.spin_spacing_search_max = QtWidgets.QDoubleSpinBox()
        self.spin_spacing_search_step = QtWidgets.QDoubleSpinBox()

        for widget in (
            self.spin_spacing_search_min,
            self.spin_spacing_search_max,
            self.spin_spacing_search_step,
        ):
            widget.setRange(
                0.001,
                10000.0,
            )
            widget.setDecimals(
                6
            )
            widget.setSuffix(
                " Å"
            )

        self.combo_lateral_treatment = QtWidgets.QComboBox()
        self.combo_lateral_treatment.addItem(
            "Fixed lateral vector",
            "fixed",
        )
        self.combo_lateral_treatment.addItem(
            "Regular u-v grid",
            "grid",
        )
        self.combo_lateral_treatment.addItem(
            "Random u-v samples",
            "random",
        )

        self.combo_lateral_target = QtWidgets.QComboBox()
        self.combo_lateral_target.addItem(
            "Repeat translation",
            "repeat_translation",
        )
        self.combo_lateral_target.addItem(
            "Selected sequence row",
            "sequence_row",
        )

        self.spin_search_sequence_row = QtWidgets.QSpinBox()
        self.spin_search_sequence_row.setRange(
            0,
            9999,
        )

        self.chk_search_u = QtWidgets.QCheckBox(
            "Search u"
        )
        self.chk_search_v = QtWidgets.QCheckBox(
            "Search v"
        )

        self.spin_u_min = QtWidgets.QDoubleSpinBox()
        self.spin_u_max = QtWidgets.QDoubleSpinBox()
        self.spin_u_step = QtWidgets.QDoubleSpinBox()

        self.spin_v_min = QtWidgets.QDoubleSpinBox()
        self.spin_v_max = QtWidgets.QDoubleSpinBox()
        self.spin_v_step = QtWidgets.QDoubleSpinBox()

        for widget in (
            self.spin_u_min,
            self.spin_u_max,
            self.spin_u_step,
            self.spin_v_min,
            self.spin_v_max,
            self.spin_v_step,
        ):
            widget.setRange(
                -1000.0,
                1000.0,
            )
            widget.setDecimals(
                8
            )

        self.spin_u_step.setMinimum(
            0.000001
        )
        self.spin_v_step.setMinimum(
            0.000001
        )

        self.spin_random_samples = QtWidgets.QSpinBox()
        self.spin_random_samples.setRange(
            1,
            1000000,
        )

        self.spin_random_seed = QtWidgets.QSpinBox()
        self.spin_random_seed.setRange(
            0,
            2147483647,
        )

        self.combo_detail_storage = QtWidgets.QComboBox()
        self.combo_detail_storage.addItem(
            "Do not store candidate curves",
            "none",
        )
        self.combo_detail_storage.addItem(
            "Store best fit for every CS",
            "best_per_cs",
        )
        self.combo_detail_storage.addItem(
            "Store every candidate",
            "all",
        )

        self.edit_detail_directory = QtWidgets.QLineEdit(
            "stacking_search_results"
        )

        self.label_stacking_candidate_count = QtWidgets.QLabel(
            "Stacking candidates: 1"
        )

        search_layout.addWidget(
            QtWidgets.QLabel("Spacing treatment:"),
            0,
            0,
        )
        search_layout.addWidget(
            self.combo_spacing_treatment,
            0,
            1,
            1,
            3,
        )

        search_layout.addWidget(
            QtWidgets.QLabel("Spacing minimum:"),
            1,
            0,
        )
        search_layout.addWidget(
            self.spin_spacing_search_min,
            1,
            1,
        )
        search_layout.addWidget(
            QtWidgets.QLabel("Maximum:"),
            1,
            2,
        )
        search_layout.addWidget(
            self.spin_spacing_search_max,
            1,
            3,
        )

        search_layout.addWidget(
            QtWidgets.QLabel("Spacing step:"),
            2,
            0,
        )
        search_layout.addWidget(
            self.spin_spacing_search_step,
            2,
            1,
        )

        search_layout.addWidget(
            QtWidgets.QLabel("Lateral treatment:"),
            3,
            0,
        )
        search_layout.addWidget(
            self.combo_lateral_treatment,
            3,
            1,
            1,
            3,
        )

        search_layout.addWidget(
            QtWidgets.QLabel("Lateral target:"),
            4,
            0,
        )
        search_layout.addWidget(
            self.combo_lateral_target,
            4,
            1,
        )
        search_layout.addWidget(
            QtWidgets.QLabel("Sequence row:"),
            4,
            2,
        )
        search_layout.addWidget(
            self.spin_search_sequence_row,
            4,
            3,
        )

        search_layout.addWidget(
            self.chk_search_u,
            5,
            0,
        )
        search_layout.addWidget(
            QtWidgets.QLabel("u minimum:"),
            5,
            1,
        )
        search_layout.addWidget(
            self.spin_u_min,
            5,
            2,
        )
        search_layout.addWidget(
            self.spin_u_max,
            5,
            3,
        )

        search_layout.addWidget(
            QtWidgets.QLabel("u step:"),
            6,
            1,
        )
        search_layout.addWidget(
            self.spin_u_step,
            6,
            2,
        )

        search_layout.addWidget(
            self.chk_search_v,
            7,
            0,
        )
        search_layout.addWidget(
            QtWidgets.QLabel("v minimum:"),
            7,
            1,
        )
        search_layout.addWidget(
            self.spin_v_min,
            7,
            2,
        )
        search_layout.addWidget(
            self.spin_v_max,
            7,
            3,
        )

        search_layout.addWidget(
            QtWidgets.QLabel("v step:"),
            8,
            1,
        )
        search_layout.addWidget(
            self.spin_v_step,
            8,
            2,
        )

        search_layout.addWidget(
            QtWidgets.QLabel("Random samples:"),
            9,
            0,
        )
        search_layout.addWidget(
            self.spin_random_samples,
            9,
            1,
        )
        search_layout.addWidget(
            QtWidgets.QLabel("Random seed:"),
            9,
            2,
        )
        search_layout.addWidget(
            self.spin_random_seed,
            9,
            3,
        )

        search_layout.addWidget(
            QtWidgets.QLabel("Store candidate details:"),
            10,
            0,
        )
        search_layout.addWidget(
            self.combo_detail_storage,
            10,
            1,
            1,
            3,
        )

        search_layout.addWidget(
            QtWidgets.QLabel("Result directory:"),
            11,
            0,
        )
        search_layout.addWidget(
            self.edit_detail_directory,
            11,
            1,
            1,
            3,
        )

        search_layout.addWidget(
            self.label_stacking_candidate_count,
            12,
            0,
            1,
            4,
        )

        upper_layout.addWidget(
            search_group
        )

        spacing_group = QtWidgets.QGroupBox(
            "Layer spacing and repeat translation"
        )


        spacing_layout = QtWidgets.QGridLayout(
            spacing_group
        )

        self.combo_spacing_mode = QtWidgets.QComboBox()

        self.combo_spacing_mode.addItem(
            "Independent distance / Å",
            "independent",
        )

        self.combo_spacing_mode.addItem(
            "Fraction of parent c-normal",
            "fraction_of_c_normal",
        )

        spacing_layout.addWidget(
            QtWidgets.QLabel(
                "Spacing mode:"
            ),
            0,
            0,
        )

        spacing_layout.addWidget(
            self.combo_spacing_mode,
            0,
            1,
        )


        self.spin_nominal_spacing = QtWidgets.QDoubleSpinBox()
        self.spin_nominal_spacing.setRange(
            0.001,
            10000.0,
        )
        self.spin_nominal_spacing.setDecimals(
            6
        )
        self.spin_nominal_spacing.setSuffix(
            " Å"
        )

        self.chk_refine_nominal_spacing = QtWidgets.QCheckBox(
            "Refine spacing for every shape candidate"
        )

        self.spin_nominal_spacing_min = QtWidgets.QDoubleSpinBox()
        self.spin_nominal_spacing_max = QtWidgets.QDoubleSpinBox()

        for widget in (
            self.spin_nominal_spacing_min,
            self.spin_nominal_spacing_max,
        ):
            widget.setRange(
                0.001,
                10000.0,
            )
            widget.setDecimals(
                6
            )
            widget.setSuffix(
                " Å"
            )

        self.spin_repeat_period = QtWidgets.QDoubleSpinBox()
        self.spin_repeat_period.setRange(
            0.001,
            100000.0,
        )
        self.spin_repeat_period.setDecimals(
            6
        )

        self.spin_repeat_shift_u = QtWidgets.QDoubleSpinBox()
        self.spin_repeat_shift_v = QtWidgets.QDoubleSpinBox()

        for widget in (
            self.spin_repeat_shift_u,
            self.spin_repeat_shift_v,
        ):
            widget.setRange(
                -1000.0,
                1000.0,
            )
            widget.setDecimals(
                8
            )

        spacing_layout.addWidget(
            QtWidgets.QLabel(
                "Nominal spacing:"
            ),
            1,
            0,
        )

        spacing_layout.addWidget(
            self.spin_nominal_spacing,
            1,
            1,
        )

        spacing_layout.addWidget(
            self.chk_refine_nominal_spacing,
            2,
            0,
            1,
            2,
        )

        spacing_layout.addWidget(
            QtWidgets.QLabel(
                "Distance minimum:"
            ),
            3,
            0,
        )

        spacing_layout.addWidget(
            self.spin_nominal_spacing_min,
            3,
            1,
        )

        spacing_layout.addWidget(
            QtWidgets.QLabel(
                "Distance maximum:"
            ),
            4,
            0,
        )

        spacing_layout.addWidget(
            self.spin_nominal_spacing_max,
            4,
            1,
        )

        self.spin_stacking_fraction = QtWidgets.QDoubleSpinBox()

        self.spin_stacking_fraction.setRange(
            0.001,
            100.0,
        )

        self.spin_stacking_fraction.setDecimals(
            8
        )

        self.chk_refine_stacking_fraction = QtWidgets.QCheckBox(
            "Refine stacking fraction"
        )

        self.spin_stacking_fraction_min = QtWidgets.QDoubleSpinBox()
        self.spin_stacking_fraction_max = QtWidgets.QDoubleSpinBox()

        for widget in (
            self.spin_stacking_fraction_min,
            self.spin_stacking_fraction_max,
        ):
            widget.setRange(
                0.001,
                100.0,
            )

            widget.setDecimals(
                8
            )

        spacing_layout.addWidget(
            QtWidgets.QLabel(
                "Fraction of c-normal:"
            ),
            5,
            0,
        )

        spacing_layout.addWidget(
            self.spin_stacking_fraction,
            5,
            1,
        )

        spacing_layout.addWidget(
            self.chk_refine_stacking_fraction,
            6,
            0,
            1,
            2,
        )

        spacing_layout.addWidget(
            QtWidgets.QLabel(
                "Fraction minimum:"
            ),
            7,
            0,
        )

        spacing_layout.addWidget(
            self.spin_stacking_fraction_min,
            7,
            1,
        )

        spacing_layout.addWidget(
            QtWidgets.QLabel(
                "Fraction maximum:"
            ),
            8,
            0,
        )

        spacing_layout.addWidget(
            self.spin_stacking_fraction_max,
            8,
            1,
        )


        self.label_repeat_period = QtWidgets.QLabel(
            "Closing Δs to next repeat:"
        )

        spacing_layout.addWidget(
            self.label_repeat_period,
            9,
            0,
        )

        spacing_layout.addWidget(
            self.spin_repeat_period,
            9,
            1,
        )

        spacing_layout.addWidget(
            QtWidgets.QLabel(
                "Repeat shift u:"
            ),
            10,
            0,
        )

        spacing_layout.addWidget(
            self.spin_repeat_shift_u,
            10,
            1,
        )

        spacing_layout.addWidget(
            QtWidgets.QLabel(
                "Repeat shift v:"
            ),
            11,
            0,
        )

        spacing_layout.addWidget(
            self.spin_repeat_shift_v,
            11,
            1,
        )


        upper_layout.addWidget(
            spacing_group
        )

        sequence_group = QtWidgets.QGroupBox(
            "Editable repeating sequence"
        )

        sequence_layout = QtWidgets.QVBoxLayout(
            sequence_group
        )

        treatment_row = QtWidgets.QHBoxLayout()

        treatment_row.addWidget(
            QtWidgets.QLabel(
                "Vector treatment:"
            )
        )

        self.combo_stacking_treatment = (
            QtWidgets.QComboBox()
        )

        self.combo_stacking_treatment.addItem(
            "Fixed vectors",
            "fixed",
        )

        self.combo_stacking_treatment.addItem(
            "Continuous refinement",
            "continuous",
        )

        self.combo_stacking_treatment.addItem(
            "Finite grid search",
            "grid",
        )

        treatment_row.addWidget(
            self.combo_stacking_treatment,
            1,
        )

        sequence_layout.addLayout(
            treatment_row
        )

        self.label_stacking_treatment_info = (
            QtWidgets.QLabel()
        )

        self.label_stacking_treatment_info.setWordWrap(
            True
        )

        sequence_layout.addWidget(
            self.label_stacking_treatment_info
        )

        self.layer_table_model = LayerPlacementTableModel(
            self
        )

        self.layer_table = QtWidgets.QTableView()

        self.layer_table.setModel(
            self.layer_table_model
        )

        self.layer_float_delegate = HighPrecisionFloatDelegate(
            self.layer_table,
            minimum=-1000000.0,
            maximum=1000000.0,
            decimals=12,
            single_step=0.01,
        )

        self.layer_float_delegate.closeEditor.connect(
            self._on_layer_editor_closed
        )

        for column in (
            self.layer_table_model.COLUMN_DU,
            self.layer_table_model.COLUMN_DV,
            self.layer_table_model.COLUMN_H,
        ):
            self.layer_table.setItemDelegateForColumn(
                column,
                self.layer_float_delegate,
            )

        self.layer_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )

        self.layer_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )

        self.layer_table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
            | QtWidgets.QAbstractItemView.AnyKeyPressed
        )

        self.layer_table.setAlternatingRowColors(
            True
        )

        self.layer_table.verticalHeader().setVisible(
            False
        )

        self.layer_table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Fixed
        )

        self.layer_table.verticalHeader().setDefaultSectionSize(
            30
        )

        self.layer_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )

        self.layer_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        self.layer_table.setWordWrap(
            False
        )

        header = self.layer_table.horizontalHeader()

        header.setMinimumSectionSize(
            28
        )

        header.setSectionResizeMode(
            self.layer_table_model.COLUMN_ON,
            QtWidgets.QHeaderView.Fixed,
        )

        header.setSectionResizeMode(
            self.layer_table_model.COLUMN_ROW,
            QtWidgets.QHeaderView.Fixed,
        )

        header.setSectionResizeMode(
            self.layer_table_model.COLUMN_LAYER,
            QtWidgets.QHeaderView.Fixed,
        )

        self.layer_table.setColumnWidth(
            self.layer_table_model.COLUMN_ON,
            34,
        )

        self.layer_table.setColumnWidth(
            self.layer_table_model.COLUMN_ROW,
            42,
        )

        self.layer_table.setColumnWidth(
            self.layer_table_model.COLUMN_LAYER,
            55,
        )

        header.setSectionResizeMode(
            self.layer_table_model.COLUMN_TRANSITION,
            QtWidgets.QHeaderView.Stretch,
        )

        for column in (
            self.layer_table_model.COLUMN_DU,
            self.layer_table_model.COLUMN_DV,
            self.layer_table_model.COLUMN_H,
            self.layer_table_model.COLUMN_Z,
        ):
            header.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.Stretch,
            )

        for column in (
            self.layer_table_model.COLUMN_USE_DU,
            self.layer_table_model.COLUMN_USE_DV,
            self.layer_table_model.COLUMN_USE_H,
        ):
            header.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.Fixed,
            )

            self.layer_table.setColumnWidth(
                column,
                42,
            )

        for column in (
            self.layer_table_model.COLUMN_COPIES,
            self.layer_table_model.COLUMN_ATOMS,
        ):
            header.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.Fixed,
            )

            self.layer_table.setColumnWidth(
                column,
                72,
            )

        header.setStretchLastSection(
            False
        )

        sequence_layout.addWidget(
            self.layer_table
        )

        transition_settings_group = QtWidgets.QGroupBox(
            "Selected transition settings"
        )

        transition_settings_layout = QtWidgets.QGridLayout(
            transition_settings_group
        )

        transition_settings_layout.setContentsMargins(
            6,
            4,
            6,
            6,
        )

        transition_settings_layout.setHorizontalSpacing(
            8
        )

        transition_settings_layout.setVerticalSpacing(
            3
        )


        self.label_selected_transition = QtWidgets.QLabel(
            "Select a transition row."
        )

        self.label_selected_transition.setWordWrap(
            True
        )

        transition_settings_layout.addWidget(
            self.label_selected_transition,
            0,
            0,
            1,
            6,
        )

        headers = (
            "Component",
            "Current",
            "Enable",
            "Minimum",
            "Maximum",
            "Step",
        )

        for column, text in enumerate(
            headers
        ):
            transition_settings_layout.addWidget(
                QtWidgets.QLabel(
                    text
                ),
                1,
                column,
            )

        self._transition_editor_syncing = False
        self._transition_editor_widgets = {}

        for row, (
            component,
            label,
        ) in enumerate(
            (
                (
                    "u",
                    "Δu",
                ),
                (
                    "v",
                    "Δv",
                ),
                (
                    "h",
                    "h / Å",
                ),
            ),
            start=2,
        ):
            current = QtWidgets.QDoubleSpinBox()
            current.setDecimals(
                10
            )
            current.setRange(
                -1000000.0,
                1000000.0,
            )
            current.setReadOnly(
                True
            )
            current.setButtonSymbols(
                QtWidgets.QAbstractSpinBox.NoButtons
            )

            enabled_checkbox = QtWidgets.QCheckBox()

            minimum = QtWidgets.QDoubleSpinBox()
            maximum = QtWidgets.QDoubleSpinBox()
            step = QtWidgets.QDoubleSpinBox()

            for widget in (
                minimum,
                maximum,
                step,
            ):
                widget.setDecimals(
                    10
                )
                widget.setRange(
                    -1000000.0,
                    1000000.0,
                )

            step.setMinimum(
                1.0e-10
            )

            transition_settings_layout.addWidget(
                QtWidgets.QLabel(
                    label
                ),
                row,
                0,
            )

            transition_settings_layout.addWidget(
                current,
                row,
                1,
            )

            transition_settings_layout.addWidget(
                enabled_checkbox,
                row,
                2,
            )

            transition_settings_layout.addWidget(
                minimum,
                row,
                3,
            )

            transition_settings_layout.addWidget(
                maximum,
                row,
                4,
            )

            transition_settings_layout.addWidget(
                step,
                row,
                5,
            )

            self._transition_editor_widgets[
                component
            ] = {
                "current": current,
                "enabled": enabled_checkbox,
                "minimum": minimum,
                "maximum": maximum,
                "step": step,
            }

        sequence_layout.addWidget(
            transition_settings_group
        )

        self.transition_settings_group = (
            transition_settings_group
        )

        edit_buttons = QtWidgets.QGridLayout()

        self.btn_add_before = QtWidgets.QPushButton(
            "Add before"
        )
        self.btn_add_after = QtWidgets.QPushButton(
            "Add after"
        )
        self.btn_duplicate = QtWidgets.QPushButton(
            "Duplicate"
        )
        self.btn_remove = QtWidgets.QPushButton(
            "Remove"
        )
        self.btn_move_up = QtWidgets.QPushButton(
            "Move up"
        )
        self.btn_move_down = QtWidgets.QPushButton(
            "Move down"
        )
        self.btn_set_sequence_origin = QtWidgets.QPushButton(
            "Set selected as sequence origin"
        )
        self.btn_normalize_coordinates = QtWidgets.QPushButton(
            "Normalize coordinates"
        )

        edit_button_values = (
            self.btn_add_before,
            self.btn_add_after,
            self.btn_duplicate,
            self.btn_remove,
            self.btn_move_up,
            self.btn_move_down,
            self.btn_set_sequence_origin,
            self.btn_normalize_coordinates,
        )

        for index, button in enumerate(
            edit_button_values
        ):
            edit_buttons.addWidget(
                button,
                index // 2,
                index % 2,
            )

        sequence_layout.addLayout(
            edit_buttons
        )

        generator_buttons = QtWidgets.QHBoxLayout()

        self.btn_generate_a = QtWidgets.QPushButton(
            "Generate A"
        )
        self.btn_generate_ab = QtWidgets.QPushButton(
            "Generate A/B"
        )
        self.btn_generate_abc = QtWidgets.QPushButton(
            "Generate A/B/C"
        )

        generator_buttons.addWidget(
            self.btn_generate_a
        )
        generator_buttons.addWidget(
            self.btn_generate_ab
        )
        generator_buttons.addWidget(
            self.btn_generate_abc
        )

        sequence_layout.addLayout(
            generator_buttons
        )

        lower_layout.addWidget(
            sequence_group
        )

        upper_layout.addStretch(
            1
        )

        lower_layout.addStretch(
            1
        )

        self.stacking_vertical_splitter.addWidget(
            upper_widget
        )

        self.stacking_vertical_splitter.addWidget(
            lower_widget
        )

        self.stacking_vertical_splitter.setStretchFactor(
            0,
            1,
        )

        self.stacking_vertical_splitter.setStretchFactor(
            1,
            1,
        )

        self.stacking_vertical_splitter.setSizes(
            [
                450,
                450,
            ]
        )



    def _load_state_to_ui(
        self,
    ) -> None:
        shape_index = self.combo_shape.findText(
            self.state.shape_type
        )

        if shape_index >= 0:
            self.combo_shape.setCurrentIndex(
                shape_index
            )

        self.spin_axis1.setValue(
            self.state.axis1_cells
        )
        self.spin_axis2.setValue(
            self.state.axis2_cells
        )
        self.spin_axis3.setValue(
            self.state.axis3_cells
        )

        search_index = self.combo_search_mode.findData(
            self.state.search_mode
        )

        if search_index >= 0:
            self.combo_search_mode.setCurrentIndex(
                search_index
            )

        self.chk_scan_axis1.setChecked(
            self.state.scan_axis1
        )
        self.spin_axis1_min.setValue(
            self.state.axis1_min_cells
        )
        self.spin_axis1_max.setValue(
            self.state.axis1_max_cells
        )
        self.spin_axis1_step.setValue(
            self.state.axis1_step_cells
        )

        self.chk_scan_axis2.setChecked(
            self.state.scan_axis2
        )
        self.spin_axis2_min.setValue(
            self.state.axis2_min_cells
        )
        self.spin_axis2_max.setValue(
            self.state.axis2_max_cells
        )
        self.spin_axis2_step.setValue(
            self.state.axis2_step_cells
        )

        self.chk_scan_axis3.setChecked(
            self.state.scan_axis3
        )
        self.spin_axis3_min.setValue(
            self.state.axis3_min_cells
        )
        self.spin_axis3_max.setValue(
            self.state.axis3_max_cells
        )
        self.spin_axis3_step.setValue(
            self.state.axis3_step_cells
        )

        orientation_values = (
            (self.spin_axis_h, self.state.axis_h),
            (self.spin_axis_k, self.state.axis_k),
            (self.spin_axis_l, self.state.axis_l),
            (self.spin_base1_h, self.state.base1_h),
            (self.spin_base1_k, self.state.base1_k),
            (self.spin_base1_l, self.state.base1_l),
            (self.spin_base2_h, self.state.base2_h),
            (self.spin_base2_k, self.state.base2_k),
            (self.spin_base2_l, self.state.base2_l),
        )

        for widget, value in orientation_values:
            widget.setValue(
                value
            )

        self.chk_show_shape_surface.setChecked(
            self.state.show_shape_surface
        )
        self.chk_show_shape_axes.setChecked(
            self.state.show_shape_axes
        )
        self.chk_show_atoms.setChecked(
            self.state.show_atoms
        )
        self.spin_shape_opacity.setValue(
            self.state.shape_opacity
        )
        self.spin_atom_limit.setValue(
            self.state.all_atom_preview_limit
        )

        self.chk_stacking_enabled.setChecked(
            self.state.stacking_enabled
        )

        stacking_model_index = self.combo_stacking_model.findData(
            self.state.stacking_model
        )

        if stacking_model_index >= 0:
            self.combo_stacking_model.setCurrentIndex(
                stacking_model_index
            )

        self.spin_p_alternate.setValue(
            self.state.stacking_p_alternate
        )

        self.chk_p_grid_enabled.setChecked(
            self.state.stacking_p_grid_enabled
        )

        self.spin_p_grid_min.setValue(
            self.state.stacking_p_grid_minimum
        )

        self.spin_p_grid_max.setValue(
            self.state.stacking_p_grid_maximum
        )

        self.spin_p_grid_step.setValue(
            self.state.stacking_p_grid_step
        )

        self.spin_ensemble_samples.setValue(
            self.state.stacking_ensemble_samples
        )

        self.spin_ensemble_seed.setValue(
            self.state.stacking_ensemble_seed
        )

        start_state_index = self.combo_ensemble_start_state.findText(
            str(
                self.state.stacking_ensemble_start_state
            ).upper()
        )

        if start_state_index >= 0:
            self.combo_ensemble_start_state.setCurrentIndex(
                start_state_index
            )

        self.spin_preview_realization.setValue(
            self.state.stacking_preview_realization_index
        )

        self.spin_state_a_du.setValue(
            self.state.stacking_state_a_du
        )

        self.spin_state_a_dv.setValue(
            self.state.stacking_state_a_dv
        )

        self.spin_state_a_h.setValue(
            self.state.stacking_state_a_h
        )

        self.spin_state_b_du.setValue(
            self.state.stacking_state_b_du
        )

        self.spin_state_b_dv.setValue(
            self.state.stacking_state_b_dv
        )

        self.spin_state_b_h.setValue(
            self.state.stacking_state_b_h
        )


        self._update_stochastic_controls()

        mode_index = self.combo_definition_mode.findData(
            self.state.stacking_definition_mode
        )

        if mode_index >= 0:
            self.combo_definition_mode.setCurrentIndex(
                mode_index
            )

        vector_mode_index = self.combo_vector_mode.findData(
            self.state.stacking_vector_mode
        )

        if vector_mode_index >= 0:
            self.combo_vector_mode.setCurrentIndex(
                vector_mode_index
            )

        self.layer_table_model.set_vector_mode(
            self.state.stacking_vector_mode
        )

        self.spin_nominal_spacing.setValue(
            self.state.stacking_nominal_spacing
        )
        self.chk_refine_nominal_spacing.setChecked(
            self.state.refine_stacking_nominal_spacing
        )
        self.spin_nominal_spacing_min.setValue(
            self.state.stacking_nominal_spacing_min
        )
        self.spin_nominal_spacing_max.setValue(
            self.state.stacking_nominal_spacing_max
        )
        spacing_mode_index = self.combo_spacing_mode.findData(
            self.state.stacking_spacing_mode
        )

        if spacing_mode_index >= 0:
            self.combo_spacing_mode.setCurrentIndex(
                spacing_mode_index
            )

        self.spin_stacking_fraction.setValue(
            self.state.stacking_fraction
        )

        self.chk_refine_stacking_fraction.setChecked(
            self.state.refine_stacking_fraction
        )

        self.spin_stacking_fraction_min.setValue(
            self.state.stacking_fraction_min
        )

        self.spin_stacking_fraction_max.setValue(
            self.state.stacking_fraction_max
        )
        self.spin_repeat_period.setValue(
            self.state.stacking_repeat_period
        )
        self.spin_repeat_shift_u.setValue(
            self.state.stacking_repeat_shift_u
        )
        self.spin_repeat_shift_v.setValue(
            self.state.stacking_repeat_shift_v
        )

        self.layer_table_model.set_layers(
            self.state.sequence_layers()
        )

        self._update_layer_table_height()

        self.layer_table_model.set_nominal_spacing(
            float(
                self._stacking_spec_from_ui().nominal_spacing
            )
        )

        self.chk_show_layer_origins.setChecked(
            self.state.show_layer_origins
        )
        self.chk_show_transition_vectors.setChecked(
            self.state.show_transition_vectors
        )
        self.chk_color_stacking_layers.setChecked(
            self.state.color_stacking_layers
        )
        self.chk_show_generated_table.setChecked(
            self.state.show_generated_layers
        )
        self.chk_stacking_orthographic.setChecked(
            self.state.orthographic_projection
        )

        self.chk_show_layer_labels.setChecked(
            self.state.show_layer_labels
        )

        self.chk_show_parent_unit_cell.setChecked(
            self.state.show_parent_unit_cell
        )

        self.chk_show_lattice_vectors.setChecked(
            self.state.show_lattice_vectors
        )

        self.spin_max_layer_labels.setValue(
            self.state.preview_max_layer_labels
        )

        self.spin_layer_label_font_size.setValue(
            self.state.preview_label_font_size
        )

        self.preview.set_orthographic_projection(
            self.state.orthographic_projection
        )

        self.preview.set_parent_unit_cell_visible(
            self.state.show_parent_unit_cell
        )

        self.preview.set_lattice_vectors_visible(
            self.state.show_lattice_vectors
        )

        self.preview.set_view_preset(
            self.state.preview_view_preset
        )

        self._sync_angstrom_from_cells()

        search = self.search_spec

        index = self.combo_spacing_treatment.findData(
            search.spacing_treatment
        )
        if index >= 0:
            self.combo_spacing_treatment.setCurrentIndex(
                index
            )

        self.spin_spacing_search_min.setValue(
            search.spacing_minimum
        )
        self.spin_spacing_search_max.setValue(
            search.spacing_maximum
        )
        self.spin_spacing_search_step.setValue(
            search.spacing_step
        )

        index = self.combo_lateral_treatment.findData(
            search.lateral_treatment
        )
        if index >= 0:
            self.combo_lateral_treatment.setCurrentIndex(
                index
            )

        index = self.combo_lateral_target.findData(
            search.lateral_target
        )
        if index >= 0:
            self.combo_lateral_target.setCurrentIndex(
                index
            )

        self.spin_search_sequence_row.setValue(
            search.sequence_row
        )

        self.chk_search_u.setChecked(
            search.scan_u
        )
        self.spin_u_min.setValue(
            search.u_minimum
        )
        self.spin_u_max.setValue(
            search.u_maximum
        )
        self.spin_u_step.setValue(
            search.u_step
        )

        self.chk_search_v.setChecked(
            search.scan_v
        )
        self.spin_v_min.setValue(
            search.v_minimum
        )
        self.spin_v_max.setValue(
            search.v_maximum
        )
        self.spin_v_step.setValue(
            search.v_step
        )

        self.spin_random_samples.setValue(
            search.random_samples
        )
        self.spin_random_seed.setValue(
            search.random_seed
        )

        index = self.combo_detail_storage.findData(
            search.detail_storage
        )
        if index >= 0:
            self.combo_detail_storage.setCurrentIndex(
                index
            )

        self.edit_detail_directory.setText(
            search.detail_directory
        )

        self._update_stacking_search_controls()

        self._update_spacing_mode_controls()

        self._initialize_stacking_refinement_ui()
        
    def _connect_signals(
        self,
    ) -> None:
        self.combo_shape.currentIndexChanged.connect(
            self._on_shape_changed
        )

        for widget in (
            self.spin_axis1,
            self.spin_axis2,
            self.spin_axis3,
        ):
            widget.valueChanged.connect(
                self._on_cells_dimension_changed
            )

        for widget in (
            self.spin_axis1_angstrom,
            self.spin_axis2_angstrom,
            self.spin_axis3_angstrom,
        ):
            widget.valueChanged.connect(
                self._on_angstrom_dimension_changed
            )

        for widget in self._orientation_widgets:
            widget.valueChanged.connect(
                self._on_orientation_changed
            )

        self.combo_search_mode.currentIndexChanged.connect(
            self._update_search_controls
        )

        self.combo_search_mode.currentIndexChanged.connect(
            self._update_candidate_count
        )

        for widget in (
            self.chk_scan_axis1,
            self.chk_scan_axis2,
            self.chk_scan_axis3,
        ):
            widget.toggled.connect(
                self._update_candidate_count
            )

        for widget in self._search_spinboxes:
            widget.valueChanged.connect(
                self._update_candidate_count
            )

        for widget in (
            self.chk_show_shape_surface,
            self.chk_show_shape_axes,
            self.chk_show_atoms,
        ):
            widget.toggled.connect(
                self.schedule_preview_update
            )

        self.spin_shape_opacity.valueChanged.connect(
            self.schedule_preview_update
        )
        self.spin_atom_limit.valueChanged.connect(
            self.schedule_preview_update
        )

        self.spin_nominal_spacing.valueChanged.connect(
            self._on_spacing_changed
        )

        self.chk_refine_nominal_spacing.toggled.connect(
            self._update_spacing_mode_controls
        )

        self.combo_spacing_mode.currentIndexChanged.connect(
            self._update_spacing_mode_controls
        )

        self.combo_spacing_mode.currentIndexChanged.connect(
            self._on_spacing_changed
        )

        self.spin_stacking_fraction.valueChanged.connect(
            self._on_spacing_changed
        )

        self.chk_refine_stacking_fraction.toggled.connect(
            self._update_spacing_mode_controls
        )

        self.spin_stacking_fraction_min.valueChanged.connect(
            self.schedule_preview_update
        )

        self.spin_stacking_fraction_max.valueChanged.connect(
            self.schedule_preview_update
        )

        self.combo_vector_mode.currentIndexChanged.connect(
            self._on_vector_mode_changed
        )

        for widget in (
            self.combo_definition_mode,
            self.spin_nominal_spacing_min,
            self.spin_nominal_spacing_max,
            self.spin_repeat_period,
            self.spin_repeat_shift_u,
            self.spin_repeat_shift_v,
        ):
            if isinstance(
                widget,
                QtWidgets.QComboBox,
            ):
                widget.currentIndexChanged.connect(
                    self.schedule_preview_update
                )
            else:
                widget.valueChanged.connect(
                    self.schedule_preview_update
                )

        self.layer_table_model.modelReset.connect(
            self._schedule_layer_table_height_update
        )

        self.layer_table_model.sequenceEdited.connect(
            self.schedule_preview_update
        )

        self.layer_table_model.sequenceEdited.connect(
            self._on_layer_sequence_edited
        )

        self.btn_add_before.clicked.connect(
            self._add_before
        )
        self.btn_add_after.clicked.connect(
            self._add_after
        )
        self.btn_duplicate.clicked.connect(
            self._duplicate_selected
        )
        self.btn_remove.clicked.connect(
            self._remove_selected
        )
        self.btn_move_up.clicked.connect(
            self._move_selected_up
        )
        self.btn_move_down.clicked.connect(
            self._move_selected_down
        )
        self.btn_set_sequence_origin.clicked.connect(
            self._set_sequence_origin
        )
        self.btn_normalize_coordinates.clicked.connect(
            self._normalize_coordinates
        )

        self.btn_generate_a.clicked.connect(
            lambda: self._set_generated_sequence(
                default_a_sequence()
            )
        )
        self.btn_generate_ab.clicked.connect(
            lambda: self._set_generated_sequence(
                default_ab_sequence()
            )
        )
        self.btn_generate_abc.clicked.connect(
            lambda: self._set_generated_sequence(
                default_abc_sequence()
            )
        )

        self.chk_show_generated_table.toggled.connect(
            self.generated_layer_table.setVisible
        )
        self.chk_show_generated_table.toggled.connect(
            self.schedule_preview_update
        )

        self.chk_show_layer_origins.toggled.connect(
            self.schedule_preview_update
        )
        self.chk_show_transition_vectors.toggled.connect(
            self.schedule_preview_update
        )
        self.chk_color_stacking_layers.toggled.connect(
            self.schedule_preview_update
        )

        self.chk_stacking_orthographic.toggled.connect(
            self.preview.set_orthographic_projection
        )

        self.btn_fit_stacking_view.clicked.connect(
            self.preview.fit_scene
        )
        self.btn_export_stacked_xyz.clicked.connect(
            self._export_stacked_particle_xyz
        )

        self.btn_export_prism_periodic_cif.clicked.connect(
            self._export_prism_periodic_cif
        )

        self.btn_apply.clicked.connect(
            self.apply_shape
        )
        self.btn_close.clicked.connect(
            self.reject
        )

        for combo in (
            self.combo_spacing_treatment,
            self.combo_lateral_treatment,
            self.combo_lateral_target,
            self.combo_detail_storage,
        ):
            combo.currentIndexChanged.connect(
                self._update_stacking_search_controls
            )

        for checkbox in (
            self.chk_search_u,
            self.chk_search_v,
        ):
            checkbox.toggled.connect(
                self._update_stacking_search_controls
            )

        for widget in (
            self.spin_spacing_search_min,
            self.spin_spacing_search_max,
            self.spin_spacing_search_step,
            self.spin_search_sequence_row,
            self.spin_u_min,
            self.spin_u_max,
            self.spin_u_step,
            self.spin_v_min,
            self.spin_v_max,
            self.spin_v_step,
            self.spin_random_samples,
            self.spin_random_seed,
        ):
            widget.valueChanged.connect(
                self._update_stacking_candidate_count
            )

        self.combo_stacking_treatment.currentIndexChanged.connect(
            self._on_stacking_treatment_changed
        )

        selection_model = (
            self.layer_table.selectionModel()
        )

        if selection_model is not None:
            selection_model.currentRowChanged.connect(
                self._on_transition_row_selected
            )

        self.layer_table_model.transitionControlEdited.connect(
            self._load_selected_transition_editor
        )

        self.layer_table_model.closureEdited.connect(
            self._on_closure_transition_edited
        )

        for component_widgets in (
            self._transition_editor_widgets.values()
        ):
            component_widgets[
                "enabled"
            ].toggled.connect(
                self._save_selected_transition_editor
            )

            component_widgets[
                "minimum"
            ].valueChanged.connect(
                self._save_selected_transition_editor
            )

            component_widgets[
                "maximum"
            ].valueChanged.connect(
                self._save_selected_transition_editor
            )

            component_widgets[
                "step"
            ].valueChanged.connect(
                self._save_selected_transition_editor
            )

        self.btn_view_ac.clicked.connect(
            lambda: self._set_preview_preset(
                "ac_section"
            )
        )

        self.btn_view_bc.clicked.connect(
            lambda: self._set_preview_preset(
                "bc_section"
            )
        )

        self.btn_view_ab.clicked.connect(
            lambda: self._set_preview_preset(
                "ab_plane"
            )
        )

        self.btn_view_stack_axis.clicked.connect(
            lambda: self._set_preview_preset(
                "stack_axis"
            )
        )

        self.btn_view_isometric.clicked.connect(
            lambda: self._set_preview_preset(
                "isometric"
            )
        )

        self.btn_apply_custom_view.clicked.connect(
            self._apply_custom_preview_view
        )

        self.chk_show_parent_unit_cell.toggled.connect(
            self.preview.set_parent_unit_cell_visible
        )

        self.chk_show_lattice_vectors.toggled.connect(
            self.preview.set_lattice_vectors_visible
        )

        for widget in (
            self.chk_show_layer_labels,
            self.spin_max_layer_labels,
            self.spin_layer_label_font_size,
        ):
            if isinstance(
                widget,
                QtWidgets.QAbstractButton,
            ):
                widget.toggled.connect(
                    self.schedule_preview_update
                )
            else:
                widget.valueChanged.connect(
                    self.schedule_preview_update
                )

        self.btn_export_layer_csv.clicked.connect(
            self._export_realized_layers_csv
        )

        self.btn_export_transition_csv.clicked.connect(
            self._export_transition_vectors_csv
        )

        self.btn_export_preview_png.clicked.connect(
            self._export_preview_png
        )
        self.combo_stacking_model.currentIndexChanged.connect(
            self._on_stacking_model_changed
        )

        self.spin_p_alternate.valueChanged.connect(
            self._on_stochastic_value_changed
        )

        self.chk_p_grid_enabled.toggled.connect(
            self._update_stochastic_controls
        )

        for widget in (
            self.spin_p_grid_min,
            self.spin_p_grid_max,
            self.spin_p_grid_step,
            self.spin_ensemble_samples,
            self.spin_ensemble_seed,
            self.spin_preview_realization,
            self.spin_state_a_du,
            self.spin_state_a_dv,
            self.spin_state_a_h,
            self.spin_state_b_du,
            self.spin_state_b_dv,
            self.spin_state_b_h,
        ):
            widget.valueChanged.connect(
                self._on_stochastic_value_changed
            )

        self.combo_ensemble_start_state.currentIndexChanged.connect(
            self._on_stochastic_value_changed
        )

        self.btn_regenerate_stochastic_stack.clicked.connect(
            self._regenerate_stochastic_stacking
        )

        self.btn_refine_stochastic_vectors.clicked.connect(
            self._open_stochastic_vector_refinement_dialog
        )

        self.btn_edit_generic_markov_states.clicked.connect(
            self._open_generic_markov_editor
        )


    def _open_generic_markov_editor(
        self,
    ) -> None:
        """
        Edit generic Markov states, vector refinement, probability grids,
        ensemble settings, and preview randomization.
        """
        if (
            self.combo_stacking_model.currentData()
            != "generic_markov"
        ):
            QtWidgets.QMessageBox.information(
                self,
                "Generic Markov model not selected",
                "Select 'Generic multi-state Markov stacking' first.",
            )
            return

        from pdf_fitting.models.layer_stacking import (
            StackingMarkovState,
            normalize_markov_matrix,
            normalize_markov_probability_controls,
        )

        current_spec = self._stacking_spec_from_ui()

        states = list(
            current_spec.markov_states
            or []
        )

        if not states:
            from pdf_fitting.models.layer_stacking import (
                default_generic_markov_states,
            )

            states = default_generic_markov_states()

        number_of_states = len(
            states
        )

        matrix = normalize_markov_matrix(
            current_spec.markov_transition_matrix,
            number_of_states=number_of_states,
        )

        probability_controls = (
            normalize_markov_probability_controls(
                current_spec.markov_probability_controls,
                number_of_states=number_of_states,
                transition_matrix=matrix,
            )
        )

        dialog = QtWidgets.QDialog(
            self
        )

        dialog.setWindowTitle(
            "Generic Markov stacking model"
        )

        dialog.setWindowFlags(
            dialog.windowFlags()
            | Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        dialog.resize(
            1250,
            800,
        )

        root_layout = QtWidgets.QVBoxLayout(
            dialog
        )

        tabs = QtWidgets.QTabWidget()

        root_layout.addWidget(
            tabs,
            1,
        )

        # =========================================================
        # General / ensemble tab
        # =========================================================
        general_page = QtWidgets.QWidget()

        general_form = QtWidgets.QFormLayout(
            general_page
        )

        spin_realizations = QtWidgets.QSpinBox()
        spin_realizations.setRange(
            1,
            10000,
        )
        spin_realizations.setValue(
            int(
                self.spin_ensemble_samples.value()
            )
        )

        spin_seed = QtWidgets.QSpinBox()
        spin_seed.setRange(
            0,
            2147483647,
        )
        spin_seed.setValue(
            int(
                self.spin_ensemble_seed.value()
            )
        )

        combo_start_state = QtWidgets.QComboBox()

        for state_index, state in enumerate(
            states
        ):
            combo_start_state.addItem(
                str(
                    state.label
                ),
                state_index,
            )

        start_state_index = min(
            max(
                0,
                int(
                    current_spec.markov_start_state
                ),
            ),
            max(
                0,
                number_of_states - 1,
            ),
        )

        combo_start_state.setCurrentIndex(
            start_state_index
        )

        spin_preview_realization = QtWidgets.QSpinBox()
        spin_preview_realization.setRange(
            0,
            max(
                0,
                int(
                    spin_realizations.value()
                )
                - 1,
            ),
        )
        spin_preview_realization.setValue(
            min(
                int(
                    self.spin_preview_realization.value()
                ),
                spin_preview_realization.maximum(),
            )
        )

        button_regenerate = QtWidgets.QPushButton(
            "Regenerate stacking for visualization"
        )

        label_seed_help = QtWidgets.QLabel(
            "Regenerating changes only the random seed and the visualized "
            "frozen realizations. The transition matrix and state vectors "
            "are not changed."
        )
        label_seed_help.setWordWrap(
            True
        )

        general_form.addRow(
            "Number of realizations:",
            spin_realizations,
        )

        general_form.addRow(
            "Random seed:",
            spin_seed,
        )

        general_form.addRow(
            "Initial state:",
            combo_start_state,
        )

        general_form.addRow(
            "Preview realization:",
            spin_preview_realization,
        )

        general_form.addRow(
            button_regenerate
        )

        general_form.addRow(
            label_seed_help
        )

        tabs.addTab(
            general_page,
            "Ensemble and visualization",
        )

        def update_preview_limit():
            spin_preview_realization.setMaximum(
                max(
                    0,
                    int(
                        spin_realizations.value()
                    )
                    - 1,
                )
            )

        spin_realizations.valueChanged.connect(
            update_preview_limit
        )

        def regenerate_preview_seed():
            new_seed = int(
                np.random.default_rng().integers(
                    1,
                    2147483647,
                )
            )

            spin_seed.setValue(
                new_seed
            )

        button_regenerate.clicked.connect(
            regenerate_preview_seed
        )

        # =========================================================
        # State-vector tab
        # =========================================================
        states_page = QtWidgets.QWidget()

        states_layout = QtWidgets.QVBoxLayout(
            states_page
        )

        states_info = QtWidgets.QLabel(
            "Every state emits one stacking vector. Select whether Δu, Δv, "
            "and h are fixed or continuously refined. Minimum and maximum "
            "are optimizer bounds."
        )
        states_info.setWordWrap(
            True
        )
        states_layout.addWidget(
            states_info
        )

        state_table = QtWidgets.QTableWidget(
            0,
            6,
        )

        state_table.setHorizontalHeaderLabels(
            [
                "State",
                "Component",
                "Current value",
                "Refine?",
                "Minimum",
                "Maximum",
            ]
        )

        state_table.verticalHeader().setVisible(
            False
        )

        state_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )

        state_table.horizontalHeader().setStretchLastSection(
            True
        )

        vector_controls = dict(
            self.stochastic_vector_refinement
            or {}
        )

        state_table.setRowCount(
            number_of_states
            * 3
        )

        row_index = 0

        for state_index, state in enumerate(
            states
        ):
            state_uid = str(
                state.uid
            )

            for component, value, default_minimum, default_maximum in (
                (
                    "du",
                    float(
                        state.du
                    ),
                    -0.5,
                    0.5,
                ),
                (
                    "dv",
                    float(
                        state.dv
                    ),
                    -0.5,
                    0.5,
                ),
                (
                    "h",
                    float(
                        state.h
                    ),
                    0.001,
                    20.0,
                ),
            ):
                parameter_name = (
                    f"stack_state_{state_uid}_{component}"
                )

                control = dict(
                    vector_controls.get(
                        parameter_name,
                        {},
                    )
                    or {}
                )

                state_item = QtWidgets.QTableWidgetItem(
                    str(
                        state.label
                    )
                )

                state_item.setData(
                    Qt.UserRole,
                    {
                        "state_index": int(
                            state_index
                        ),
                        "state_uid": state_uid,
                        "parameter_name": parameter_name,
                        "component": component,
                    },
                )

                state_item.setFlags(
                    Qt.ItemIsEnabled
                    | Qt.ItemIsSelectable
                )

                state_table.setItem(
                    row_index,
                    0,
                    state_item,
                )

                component_item = QtWidgets.QTableWidgetItem(
                    {
                        "du": "Δu",
                        "dv": "Δv",
                        "h": "h / Å",
                    }[
                        component
                    ]
                )

                component_item.setFlags(
                    Qt.ItemIsEnabled
                    | Qt.ItemIsSelectable
                )

                state_table.setItem(
                    row_index,
                    1,
                    component_item,
                )

                state_table.setItem(
                    row_index,
                    2,
                    QtWidgets.QTableWidgetItem(
                        f"{value:.12g}"
                    ),
                )

                refine_item = QtWidgets.QTableWidgetItem(
                    ""
                )

                refine_item.setFlags(
                    Qt.ItemIsEnabled
                    | Qt.ItemIsSelectable
                    | Qt.ItemIsUserCheckable
                )

                refine_item.setCheckState(
                    Qt.Checked
                    if bool(
                        control.get(
                            "refine",
                            False,
                        )
                    )
                    else Qt.Unchecked
                )

                state_table.setItem(
                    row_index,
                    3,
                    refine_item,
                )

                state_table.setItem(
                    row_index,
                    4,
                    QtWidgets.QTableWidgetItem(
                        f"{float(control.get('minimum', default_minimum)):.12g}"
                    ),
                )

                state_table.setItem(
                    row_index,
                    5,
                    QtWidgets.QTableWidgetItem(
                        f"{float(control.get('maximum', default_maximum)):.12g}"
                    ),
                )

                row_index += 1

        state_table.resizeColumnsToContents()

        states_layout.addWidget(
            state_table,
            1,
        )

        tabs.addTab(
            states_page,
            "State vectors and refinement",
        )

        # =========================================================
        # Probability-grid tab
        # =========================================================
        probability_page = QtWidgets.QWidget()

        probability_layout = QtWidgets.QVBoxLayout(
            probability_page
        )

        probability_info = QtWidgets.QLabel(
            "Each transition probability can be fixed or grid-searched. "
            "For every source-state row, the selected probability combination "
            "must sum to 1. Invalid combinations are discarded."
        )

        probability_info.setWordWrap(
            True
        )

        probability_layout.addWidget(
            probability_info
        )

        probability_table = QtWidgets.QTableWidget(
            number_of_states
            * number_of_states,
            7,
        )

        probability_table.setHorizontalHeaderLabels(
            [
                "From state",
                "To state",
                "Current probability",
                "Grid?",
                "Minimum",
                "Maximum",
                "Step",
            ]
        )

        probability_table.verticalHeader().setVisible(
            False
        )

        probability_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )

        probability_table.horizontalHeader().setStretchLastSection(
            True
        )

        probability_row = 0

        for source_index, source_state in enumerate(
            states
        ):
            for target_index, target_state in enumerate(
                states
            ):
                control = probability_controls[
                    source_index
                ][
                    target_index
                ]

                source_item = QtWidgets.QTableWidgetItem(
                    str(
                        source_state.label
                    )
                )

                source_item.setData(
                    Qt.UserRole,
                    (
                        source_index,
                        target_index,
                    ),
                )

                source_item.setFlags(
                    Qt.ItemIsEnabled
                    | Qt.ItemIsSelectable
                )

                probability_table.setItem(
                    probability_row,
                    0,
                    source_item,
                )

                target_item = QtWidgets.QTableWidgetItem(
                    str(
                        target_state.label
                    )
                )

                target_item.setFlags(
                    Qt.ItemIsEnabled
                    | Qt.ItemIsSelectable
                )

                probability_table.setItem(
                    probability_row,
                    1,
                    target_item,
                )

                probability_table.setItem(
                    probability_row,
                    2,
                    QtWidgets.QTableWidgetItem(
                        f"{float(matrix[source_index][target_index]):.12g}"
                    ),
                )

                grid_item = QtWidgets.QTableWidgetItem(
                    ""
                )

                grid_item.setFlags(
                    Qt.ItemIsEnabled
                    | Qt.ItemIsSelectable
                    | Qt.ItemIsUserCheckable
                )

                grid_item.setCheckState(
                    Qt.Checked
                    if bool(
                        control.get(
                            "enabled",
                            False,
                        )
                    )
                    else Qt.Unchecked
                )

                probability_table.setItem(
                    probability_row,
                    3,
                    grid_item,
                )

                probability_table.setItem(
                    probability_row,
                    4,
                    QtWidgets.QTableWidgetItem(
                        f"{float(control.get('minimum', matrix[source_index][target_index])):.12g}"
                    ),
                )

                probability_table.setItem(
                    probability_row,
                    5,
                    QtWidgets.QTableWidgetItem(
                        f"{float(control.get('maximum', matrix[source_index][target_index])):.12g}"
                    ),
                )

                probability_table.setItem(
                    probability_row,
                    6,
                    QtWidgets.QTableWidgetItem(
                        f"{float(control.get('step', 0.05)):.12g}"
                    ),
                )

                probability_row += 1

        probability_table.resizeColumnsToContents()

        probability_layout.addWidget(
            probability_table,
            1,
        )

        label_probability_count = QtWidgets.QLabel()

        label_probability_count.setWordWrap(
            True
        )

        probability_layout.addWidget(
            label_probability_count
        )

        def probability_values(
            minimum: float,
            maximum: float,
            step: float,
        ) -> np.ndarray:
            if maximum < minimum:
                minimum, maximum = (
                    maximum,
                    minimum,
                )

            step = max(
                step,
                1.0e-12,
            )

            values = np.arange(
                minimum,
                maximum
                + 0.5
                * step,
                step,
                dtype=float,
            )

            return values[
                values
                <= maximum
                + 1.0e-10
            ]

        def update_probability_count():
            approximate_count = 1

            for row in range(
                probability_table.rowCount()
            ):
                grid_item = probability_table.item(
                    row,
                    3,
                )

                if (
                    grid_item is None
                    or grid_item.checkState()
                    != Qt.Checked
                ):
                    continue

                try:
                    minimum = float(
                        probability_table.item(
                            row,
                            4,
                        ).text()
                    )

                    maximum = float(
                        probability_table.item(
                            row,
                            5,
                        ).text()
                    )

                    step = float(
                        probability_table.item(
                            row,
                            6,
                        ).text()
                    )

                    count = max(
                        1,
                        probability_values(
                            minimum,
                            maximum,
                            step,
                        ).size,
                    )

                    approximate_count *= int(
                        count
                    )

                except Exception:
                    pass

            label_probability_count.setText(
                "Maximum unfiltered probability combinations: "
                f"{approximate_count:,}. "
                "The actual count is smaller because every matrix row "
                "must sum to 1."
            )

        probability_table.itemChanged.connect(
            update_probability_count
        )

        update_probability_count()

        tabs.addTab(
            probability_page,
            "Transition-probability grid",
        )

        # =========================================================
        # State add/remove controls
        # =========================================================
        state_identity_group = QtWidgets.QGroupBox(
            "State labels"
        )

        state_identity_layout = QtWidgets.QVBoxLayout(
            state_identity_group
        )

        state_identity_table = QtWidgets.QTableWidget(
            number_of_states,
            2,
        )

        state_identity_table.setHorizontalHeaderLabels(
            [
                "State label",
                "State UID",
            ]
        )

        state_identity_table.verticalHeader().setVisible(
            False
        )

        state_identity_table.horizontalHeader().setStretchLastSection(
            True
        )

        for state_index, state in enumerate(
            states
        ):
            state_identity_table.setItem(
                state_index,
                0,
                QtWidgets.QTableWidgetItem(
                    str(
                        state.label
                    )
                ),
            )

            uid_item = QtWidgets.QTableWidgetItem(
                str(
                    state.uid
                )
            )

            uid_item.setFlags(
                Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
            )

            state_identity_table.setItem(
                state_index,
                1,
                uid_item,
            )

        state_identity_layout.addWidget(
            state_identity_table
        )

        identity_buttons = QtWidgets.QHBoxLayout()

        button_add_state = QtWidgets.QPushButton(
            "Add state"
        )

        button_remove_state = QtWidgets.QPushButton(
            "Remove selected state"
        )

        identity_buttons.addWidget(
            button_add_state
        )

        identity_buttons.addWidget(
            button_remove_state
        )

        identity_buttons.addStretch(
            1
        )

        state_identity_layout.addLayout(
            identity_buttons
        )

        root_layout.insertWidget(
            0,
            state_identity_group,
        )

        # Adding/removing states changes every table. To avoid silently
        # discarding edits, reopen the editor after applying the new state set.
        def add_state():
            new_state = StackingMarkovState.from_dict(
                {
                    "label": f"S{len(states)}",
                    "du": 0.0,
                    "dv": 0.0,
                    "h": 3.8,
                },
                default_label=f"S{len(states)}",
            )

            states.append(
                new_state
            )

            self._generic_markov_states = [
                state.to_dict()
                for state in states
            ]

            new_size = len(
                states
            )

            expanded_matrix = np.eye(
                new_size,
                dtype=float,
            )

            old_matrix = np.asarray(
                matrix,
                dtype=float,
            )

            expanded_matrix[
                :old_matrix.shape[0],
                :old_matrix.shape[1],
            ] = old_matrix

            self._generic_markov_transition_matrix = (
                expanded_matrix.tolist()
            )

            self._generic_markov_probability_controls = []

            dialog.reject()

            QtCore.QTimer.singleShot(
                0,
                self._open_generic_markov_editor,
            )

        def remove_state():
            selected_rows = {
                index.row()
                for index in state_identity_table.selectedIndexes()
            }

            if not selected_rows:
                return

            remove_index = min(
                selected_rows
            )

            if len(
                states
            ) <= 1:
                QtWidgets.QMessageBox.warning(
                    dialog,
                    "Cannot remove state",
                    "At least one Markov state is required.",
                )
                return

            states.pop(
                remove_index
            )

            reduced_matrix = np.delete(
                np.delete(
                    np.asarray(
                        matrix,
                        dtype=float,
                    ),
                    remove_index,
                    axis=0,
                ),
                remove_index,
                axis=1,
            )

            reduced_matrix = np.asarray(
                normalize_markov_matrix(
                    reduced_matrix,
                    number_of_states=len(
                        states
                    ),
                ),
                dtype=float,
            )

            self._generic_markov_states = [
                state.to_dict()
                for state in states
            ]

            self._generic_markov_transition_matrix = (
                reduced_matrix.tolist()
            )

            self._generic_markov_probability_controls = []

            dialog.reject()

            QtCore.QTimer.singleShot(
                0,
                self._open_generic_markov_editor,
            )

        button_add_state.clicked.connect(
            add_state
        )

        button_remove_state.clicked.connect(
            remove_state
        )

        # =========================================================
        # Footer
        # =========================================================
        footer = QtWidgets.QHBoxLayout()

        footer.addStretch(
            1
        )

        button_apply = QtWidgets.QPushButton(
            "Apply"
        )

        button_cancel = QtWidgets.QPushButton(
            "Cancel"
        )

        footer.addWidget(
            button_apply
        )

        footer.addWidget(
            button_cancel
        )

        root_layout.addLayout(
            footer
        )

        button_apply.clicked.connect(
            dialog.accept
        )

        button_cancel.clicked.connect(
            dialog.reject
        )

        if (
            dialog.exec()
            != QtWidgets.QDialog.Accepted
        ):
            return

        # =========================================================
        # Read state labels
        # =========================================================
        updated_states = []

        for state_index, old_state in enumerate(
            states
        ):
            label_item = state_identity_table.item(
                state_index,
                0,
            )

            label = (
                label_item.text().strip()
                if label_item is not None
                else str(
                    old_state.label
                )
            )

            updated_states.append(
                StackingMarkovState.from_dict(
                    {
                        "uid": str(
                            old_state.uid
                        ),
                        "label": label,
                        "du": float(
                            old_state.du
                        ),
                        "dv": float(
                            old_state.dv
                        ),
                        "h": float(
                            old_state.h
                        ),
                    },
                    default_label=label,
                )
            )

        # =========================================================
        # Read state vector values and refinement settings
        # =========================================================
        updated_vector_controls = {}

        for row in range(
            state_table.rowCount()
        ):
            metadata_item = state_table.item(
                row,
                0,
            )

            if metadata_item is None:
                continue

            metadata = metadata_item.data(
                Qt.UserRole
            )

            if not isinstance(
                metadata,
                dict,
            ):
                continue

            state_index = int(
                metadata[
                    "state_index"
                ]
            )

            parameter_name = str(
                metadata[
                    "parameter_name"
                ]
            )

            component = str(
                metadata[
                    "component"
                ]
            )

            try:
                current_value = float(
                    state_table.item(
                        row,
                        2,
                    ).text()
                )
            except Exception:
                current_value = 0.0

            try:
                minimum = float(
                    state_table.item(
                        row,
                        4,
                    ).text()
                )
            except Exception:
                minimum = (
                    0.001
                    if component == "h"
                    else -0.5
                )

            try:
                maximum = float(
                    state_table.item(
                        row,
                        5,
                    ).text()
                )
            except Exception:
                maximum = (
                    20.0
                    if component == "h"
                    else 0.5
                )

            if maximum < minimum:
                minimum, maximum = (
                    maximum,
                    minimum,
                )

            if component == "h":
                current_value = max(
                    current_value,
                    0.001,
                )

                minimum = max(
                    minimum,
                    0.001,
                )

                maximum = max(
                    maximum,
                    minimum,
                )

            state = updated_states[
                state_index
            ]

            if component == "du":
                state.du = float(
                    current_value
                )

            elif component == "dv":
                state.dv = float(
                    current_value
                )

            else:
                state.h = float(
                    current_value
                )

            refine_item = state_table.item(
                row,
                3,
            )

            updated_vector_controls[
                parameter_name
            ] = {
                "refine": bool(
                    refine_item is not None
                    and refine_item.checkState()
                    == Qt.Checked
                ),
                "minimum": float(
                    minimum
                ),
                "maximum": float(
                    maximum
                ),
            }

        # =========================================================
        # Read matrix values and probability grids
        # =========================================================
        updated_matrix = np.zeros(
            (
                number_of_states,
                number_of_states,
            ),
            dtype=float,
        )

        updated_probability_controls = [
            [
                {}
                for _target_index in range(
                    number_of_states
                )
            ]
            for _source_index in range(
                number_of_states
            )
        ]

        for row in range(
            probability_table.rowCount()
        ):
            source_item = probability_table.item(
                row,
                0,
            )

            if source_item is None:
                continue

            source_index, target_index = source_item.data(
                Qt.UserRole
            )

            current_probability = float(
                probability_table.item(
                    row,
                    2,
                ).text()
            )

            minimum = float(
                probability_table.item(
                    row,
                    4,
                ).text()
            )

            maximum = float(
                probability_table.item(
                    row,
                    5,
                ).text()
            )

            step = float(
                probability_table.item(
                    row,
                    6,
                ).text()
            )

            if maximum < minimum:
                minimum, maximum = (
                    maximum,
                    minimum,
                )

            current_probability = min(
                1.0,
                max(
                    0.0,
                    current_probability,
                ),
            )

            minimum = min(
                1.0,
                max(
                    0.0,
                    minimum,
                ),
            )

            maximum = min(
                1.0,
                max(
                    0.0,
                    maximum,
                ),
            )

            updated_matrix[
                source_index,
                target_index,
            ] = current_probability

            grid_item = probability_table.item(
                row,
                3,
            )

            updated_probability_controls[
                source_index
            ][
                target_index
            ] = {
                "enabled": bool(
                    grid_item is not None
                    and grid_item.checkState()
                    == Qt.Checked
                ),
                "minimum": float(
                    minimum
                ),
                "maximum": float(
                    maximum
                ),
                "step": max(
                    float(
                        step
                    ),
                    1.0e-8,
                ),
            }

        updated_matrix = np.asarray(
            normalize_markov_matrix(
                updated_matrix,
                number_of_states=number_of_states,
            ),
            dtype=float,
        )

        self._generic_markov_states = [
            state.to_dict()
            for state in updated_states
        ]

        self._generic_markov_transition_matrix = (
            updated_matrix.tolist()
        )

        self._generic_markov_probability_controls = (
            updated_probability_controls
        )

        self._generic_markov_start_state = int(
            combo_start_state.currentData()
            if combo_start_state.currentData()
            is not None
            else 0
        )

        self.stochastic_vector_refinement = (
            updated_vector_controls
        )

        self.spin_ensemble_samples.setValue(
            int(
                spin_realizations.value()
            )
        )

        self.spin_ensemble_seed.setValue(
            int(
                spin_seed.value()
            )
        )

        self.spin_preview_realization.setValue(
            int(
                spin_preview_realization.value()
            )
        )

        self._stochastic_preview_uniforms = None
        self._stochastic_preview_uniforms_key = None

        self.schedule_preview_update()


    def _open_stochastic_vector_refinement_dialog(
        self,
    ) -> None:
        """
        Configure continuous refinement of the stochastic State A and State B
        emitted vectors.

        The generated refinement parameters are:

            stack_state_a_du
            stack_state_a_dv
            stack_state_a_h
            stack_state_b_du
            stack_state_b_dv
            stack_state_b_h
        """
        if not self._is_correlated_ab_stacking_model():
            QtWidgets.QMessageBox.information(
                self,
                "Correlated A/B stacking not selected",
                "This dialog is only used for the correlated stochastic "
                "A/B stacking model.\n\n"
                "For generic Markov stacking, use "
                "'Edit generic Markov states and probabilities'.",
            )
            return

        dialog = QtWidgets.QDialog(
            self
        )

        dialog.setWindowTitle(
            "Refine stochastic State A/B vectors"
        )

        dialog.resize(
            760,
            390,
        )

        layout = QtWidgets.QVBoxLayout(
            dialog
        )

        explanation = QtWidgets.QLabel(
            "Select which State A and State B vector components are refined.\n\n"
            "P(alternate) is intentionally not included here. It must remain "
            "an outer grid-search variable because it changes frozen Markov "
            "stacking paths.\n\n"
            "For a nominally zero lateral vector, refine Δu and Δv only with "
            "narrow bounds around zero. Do not set h = 0 because h is the "
            "physical positive layer-to-layer distance."
        )

        explanation.setWordWrap(
            True
        )

        layout.addWidget(
            explanation
        )

        table = QtWidgets.QTableWidget(
            0,
            5,
            dialog,
        )

        table.setHorizontalHeaderLabels(
            [
                "Parameter",
                "Current value",
                "Refine?",
                "Minimum",
                "Maximum",
            ]
        )

        table.verticalHeader().setVisible(
            False
        )

        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )

        table.horizontalHeader().setStretchLastSection(
            True
        )

        rows = [
            (
                "stack_state_a_du",
                "State A Δu",
                float(
                    self.spin_state_a_du.value()
                ),
                -0.5,
                0.5,
            ),
            (
                "stack_state_a_dv",
                "State A Δv",
                float(
                    self.spin_state_a_dv.value()
                ),
                -0.5,
                0.5,
            ),
            (
                "stack_state_a_h",
                "State A h / Å",
                float(
                    self.spin_state_a_h.value()
                ),
                0.001,
                20.0,
            ),
            (
                "stack_state_b_du",
                "State B Δu",
                float(
                    self.spin_state_b_du.value()
                ),
                -0.5,
                0.5,
            ),
            (
                "stack_state_b_dv",
                "State B Δv",
                float(
                    self.spin_state_b_dv.value()
                ),
                -0.5,
                0.5,
            ),
            (
                "stack_state_b_h",
                "State B h / Å",
                float(
                    self.spin_state_b_h.value()
                ),
                0.001,
                20.0,
            ),
        ]

        table.setRowCount(
            len(
                rows
            )
        )

        for row_index, (
            parameter_name,
            label,
            current_value,
            default_minimum,
            default_maximum,
        ) in enumerate(
            rows
        ):
            control = dict(
                self.stochastic_vector_refinement.get(
                    parameter_name,
                    {},
                )
                or {}
            )

            item_name = QtWidgets.QTableWidgetItem(
                label
            )

            item_name.setData(
                Qt.UserRole,
                parameter_name,
            )

            item_name.setFlags(
                Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
            )

            table.setItem(
                row_index,
                0,
                item_name,
            )

            item_current = QtWidgets.QTableWidgetItem(
                f"{current_value:.12g}"
            )

            item_current.setFlags(
                Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
            )

            table.setItem(
                row_index,
                1,
                item_current,
            )

            item_refine = QtWidgets.QTableWidgetItem(
                ""
            )

            item_refine.setFlags(
                Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
                | Qt.ItemIsUserCheckable
            )

            item_refine.setCheckState(
                Qt.Checked
                if bool(
                    control.get(
                        "refine",
                        False,
                    )
                )
                else Qt.Unchecked
            )

            table.setItem(
                row_index,
                2,
                item_refine,
            )

            minimum_value = float(
                control.get(
                    "minimum",
                    default_minimum,
                )
            )

            maximum_value = float(
                control.get(
                    "maximum",
                    default_maximum,
                )
            )

            item_minimum = QtWidgets.QTableWidgetItem(
                f"{minimum_value:.12g}"
            )

            item_maximum = QtWidgets.QTableWidgetItem(
                f"{maximum_value:.12g}"
            )

            table.setItem(
                row_index,
                3,
                item_minimum,
            )

            table.setItem(
                row_index,
                4,
                item_maximum,
            )

        table.resizeColumnsToContents()

        layout.addWidget(
            table,
            1,
        )

        button_layout = QtWidgets.QHBoxLayout()

        button_layout.addStretch(
            1
        )

        button_apply = QtWidgets.QPushButton(
            "Apply"
        )

        button_cancel = QtWidgets.QPushButton(
            "Cancel"
        )

        button_layout.addWidget(
            button_apply
        )

        button_layout.addWidget(
            button_cancel
        )

        layout.addLayout(
            button_layout
        )

        button_apply.clicked.connect(
            dialog.accept
        )

        button_cancel.clicked.connect(
            dialog.reject
        )

        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return

        updated_controls = {}

        for row_index in range(
            table.rowCount()
        ):
            item_name = table.item(
                row_index,
                0,
            )

            if item_name is None:
                continue

            parameter_name = str(
                item_name.data(
                    Qt.UserRole
                )
                or ""
            ).strip()

            if not parameter_name:
                continue

            item_refine = table.item(
                row_index,
                2,
            )

            item_minimum = table.item(
                row_index,
                3,
            )

            item_maximum = table.item(
                row_index,
                4,
            )

            try:
                minimum = float(
                    item_minimum.text()
                )
            except Exception:
                minimum = -0.5

            try:
                maximum = float(
                    item_maximum.text()
                )
            except Exception:
                maximum = 0.5

            if parameter_name.endswith(
                "_h"
            ):
                minimum = max(
                    minimum,
                    0.001,
                )

                maximum = max(
                    maximum,
                    minimum,
                )

            elif maximum < minimum:
                minimum, maximum = (
                    maximum,
                    minimum,
                )

            updated_controls[
                parameter_name
            ] = {
                "refine": bool(
                    item_refine is not None
                    and item_refine.checkState()
                    == Qt.Checked
                ),
                "minimum": float(
                    minimum
                ),
                "maximum": float(
                    maximum
                ),
            }

        self.stochastic_vector_refinement = (
            updated_controls
        )

        self.schedule_preview_update()

    def _is_correlated_ab_stacking_model(
        self,
    ) -> bool:
        return bool(
            self.combo_stacking_model.currentData()
            == "correlated_ab_disorder"
        )

    def _is_generic_markov_stacking_model(
        self,
    ) -> bool:
        return bool(
            self.combo_stacking_model.currentData()
            == "generic_markov"
        )

    def _is_stochastic_stacking_model(
        self,
    ) -> bool:
        return bool(
            self.combo_stacking_model.currentData()
            in {
                "correlated_ab_disorder",
                "generic_markov",
            }
        )


    def _on_stacking_model_changed(
        self,
    ) -> None:
        self._stochastic_preview_uniforms = None
        self._stochastic_preview_uniforms_key = None

        self._update_stochastic_controls()
        self.schedule_preview_update()

    def _on_stochastic_value_changed(
        self,
    ) -> None:
        self._update_stochastic_controls()
        self.schedule_preview_update()

    def _regenerate_stochastic_stacking(
        self,
    ) -> None:
        random_seed = int(
            np.random.default_rng().integers(
                1,
                2147483647,
            )
        )

        self.spin_ensemble_seed.setValue(
            random_seed
        )

        self._stochastic_preview_uniforms = None
        self._stochastic_preview_uniforms_key = None

        self.schedule_preview_update()

    def _update_stochastic_controls(
        self,
    ) -> None:
        correlated_ab = bool(
            self.combo_stacking_model.currentData()
            == "correlated_ab_disorder"
        )

        generic_markov = bool(
            self.combo_stacking_model.currentData()
            == "generic_markov"
        )

        any_stochastic = bool(
            correlated_ab
            or generic_markov
        )

        stacking_enabled = bool(
            self.chk_stacking_enabled.isChecked()
        )

        stochastic_enabled = bool(
            stacking_enabled
            and any_stochastic
        )

        # Ensemble settings apply to both stochastic models.
        for widget in (
            self.spin_ensemble_samples,
            self.spin_ensemble_seed,
            self.spin_preview_realization,
            self.btn_regenerate_stochastic_stack,
        ):
            widget.setEnabled(
                stochastic_enabled
            )

        # A/B controls apply only to the correlated two-state model.
        for widget in (
            self.spin_p_alternate,
            self.chk_p_grid_enabled,
            self.combo_ensemble_start_state,
            self.spin_state_a_du,
            self.spin_state_a_dv,
            self.spin_state_a_h,
            self.spin_state_b_du,
            self.spin_state_b_dv,
            self.spin_state_b_h,
            self.btn_refine_stochastic_vectors,
        ):
            widget.setEnabled(
                bool(
                    stacking_enabled
                    and correlated_ab
                )
            )

        p_grid_enabled = bool(
            stacking_enabled
            and correlated_ab
            and self.chk_p_grid_enabled.isChecked()
        )

        for widget in (
            self.spin_p_grid_min,
            self.spin_p_grid_max,
            self.spin_p_grid_step,
        ):
            widget.setEnabled(
                p_grid_enabled
            )

        self.btn_edit_generic_markov_states.setEnabled(
            bool(
                stacking_enabled
                and generic_markov
            )
        )

        maximum_preview = max(
            0,
            int(
                self.spin_ensemble_samples.value()
            )
            - 1,
        )

        self.spin_preview_realization.setMaximum(
            maximum_preview
        )

        if (
            self.spin_preview_realization.value()
            > maximum_preview
        ):
            self.spin_preview_realization.setValue(
                maximum_preview
            )

        if generic_markov:
            number_of_states = max(
                1,
                len(
                    getattr(
                        self,
                        "_generic_markov_states",
                        [],
                    )
                    or []
                ),
            )

            self.label_stochastic_summary.setText(
                "Generic multi-state Markov stacking\n"
                f"States = {number_of_states}\n"
                f"Realizations = {int(self.spin_ensemble_samples.value())}\n"
                f"Seed = {int(self.spin_ensemble_seed.value())}\n"
                "Use 'Edit generic Markov states and probabilities' "
                "to configure vector refinement and probability grids."
            )

            return

        if correlated_ab:
            p_alternate = float(
                self.spin_p_alternate.value()
            )

            self.label_p_same_state.setText(
                "P(same state) = "
                f"{1.0 - p_alternate:.6f}"
            )

            self.label_stochastic_summary.setText(
                "TOPAS-compatible correlated A/B stacking\n"
                f"P(alternate) = {p_alternate:.6f}\n"
                f"P(same state) = {1.0 - p_alternate:.6f}\n"
                f"Realizations = {int(self.spin_ensemble_samples.value())}\n"
                f"Seed = {int(self.spin_ensemble_seed.value())}"
            )

            return

        self.label_stochastic_summary.setText(
            "Deterministic periodic sequence is active."
        )



    def _stochastic_preview_paths(
        self,
        stack_spec: LayerStackSpec,
    ):
        number_of_layers = max(
            1,
            int(
                round(
                    float(
                        self.spin_axis3.value()
                    )
                )
            ),
        )

        ensemble_samples = max(
            1,
            int(
                stack_spec.ensemble_samples
            ),
        )

        if stack_spec.model == "generic_markov":
            from pdf_fitting.models.stacking_disorder import (
                generate_generic_markov_paths,
            )

            return generate_generic_markov_paths(
                number_of_layers=number_of_layers,
                ensemble_samples=ensemble_samples,
                transition_matrix=np.asarray(
                    stack_spec.markov_transition_matrix,
                    dtype=float,
                ),
                seed=int(
                    stack_spec.ensemble_seed
                ),
                start_state=int(
                    stack_spec.markov_start_state
                ),
            )

        uniform_key = (
            int(
                number_of_layers
            ),
            int(
                ensemble_samples
            ),
            int(
                stack_spec.ensemble_seed
            ),
        )

        if (
            self._stochastic_preview_uniforms is None
            or self._stochastic_preview_uniforms_key
            != uniform_key
        ):
            path_set = generate_correlated_ab_paths(
                number_of_layers=number_of_layers,
                ensemble_samples=ensemble_samples,
                p_alternate=float(
                    stack_spec.p_alternate
                ),
                seed=int(
                    stack_spec.ensemble_seed
                ),
                start_state=str(
                    stack_spec.ensemble_start_state
                ),
            )

            self._stochastic_preview_uniforms = (
                path_set.uniform_randoms.copy()
            )

            self._stochastic_preview_uniforms_key = (
                uniform_key
            )

        return generate_correlated_ab_paths(
            number_of_layers=number_of_layers,
            ensemble_samples=ensemble_samples,
            p_alternate=float(
                stack_spec.p_alternate
            ),
            seed=int(
                stack_spec.ensemble_seed
            ),
            start_state=str(
                stack_spec.ensemble_start_state
            ),
            uniform_randoms=self._stochastic_preview_uniforms,
        )

    
    def _current_stacking_treatment(
        self,
    ) -> str:
        treatment = str(
            self.combo_stacking_treatment.currentData()
            or STACKING_TREATMENT_FIXED
        ).strip().lower()

        if treatment not in {
            STACKING_TREATMENT_FIXED,
            STACKING_TREATMENT_CONTINUOUS,
            STACKING_TREATMENT_GRID,
        }:
            treatment = STACKING_TREATMENT_FIXED

        return treatment


    def _initialize_stacking_refinement_ui(
        self,
    ) -> None:
        """
        Restore treatment, transition controls and closure values.
        """
        treatment_index = (
            self.combo_stacking_treatment.findData(
                self.stacking_refinement_state.treatment
            )
        )

        self.combo_stacking_treatment.blockSignals(
            True
        )

        try:
            if treatment_index >= 0:
                self.combo_stacking_treatment.setCurrentIndex(
                    treatment_index
                )
            else:
                fixed_index = (
                    self.combo_stacking_treatment.findData(
                        STACKING_TREATMENT_FIXED
                    )
                )

                if fixed_index >= 0:
                    self.combo_stacking_treatment.setCurrentIndex(
                        fixed_index
                    )

        finally:
            self.combo_stacking_treatment.blockSignals(
                False
            )

        transition_controls = {}

        for (
            transition_id,
            transition_control,
        ) in (
            self.stacking_refinement_state
            .transition_map()
            .items()
        ):
            transition_controls[
                transition_id
            ] = transition_control.to_dict()

        self.layer_table_model.set_transition_controls(
            transition_controls
        )

        self._sync_closure_from_current_stack_spec()

        self.layer_table_model.set_treatment(
            self._current_stacking_treatment()
        )

        self._update_stacking_treatment_controls()

        if self.layer_table_model.rowCount() > 1:
            self.layer_table.selectRow(
                1
            )
        elif self.layer_table_model.rowCount() > 0:
            self.layer_table.selectRow(
                0
            )

        self._load_selected_transition_editor()

        self._schedule_layer_table_height_update()


    def _sync_closure_from_current_stack_spec(
        self,
    ) -> None:
        """
        Derive the displayed closing transition from the current stack spec.
        """
        try:
            stack_spec = self._stacking_spec_from_ui()

            transitions = transitions_from_stack_spec(
                stack_spec
            )

            if not transitions:
                return

            closure = transitions[
                -1
            ]

            self.layer_table_model.set_closure_values(
                du=float(
                    closure.du
                ),
                dv=float(
                    closure.dv
                ),
                h=float(
                    closure.h
                ),
            )

        except Exception:
            pass


    def _selected_transition_row(
        self,
    ) -> Optional[int]:
        current_index = self.layer_table.currentIndex()

        if not current_index.isValid():
            return None

        row = int(
            current_index.row()
        )

        if (
            row <= 0
            or row >= self.layer_table_model.rowCount()
        ):
            return None

        return row


    def _on_transition_row_selected(
        self,
        current,
        previous,
    ) -> None:
        _ = current
        _ = previous

        self._load_selected_transition_editor()


    def _load_selected_transition_editor(
        self,
    ) -> None:
        row = self._selected_transition_row()

        self._transition_editor_syncing = True

        try:
            if row is None:
                self.label_selected_transition.setText(
                    "Select a transition row. The reference row cannot "
                    "be refined or scanned."
                )

                self.transition_settings_group.setEnabled(
                    False
                )

                return

            transition_id = (
                self.layer_table_model.transition_id_at_row(
                    row
                )
            )

            transition_label = (
                self.layer_table_model.transition_label_at_row(
                    row
                )
            )

            transition_values = (
                self.layer_table_model.transition_values_at_row(
                    row
                )
            )

            transition_control = (
                self.layer_table_model.transition_control_at_row(
                    row
                )
            )

            if (
                transition_id is None
                or transition_values is None
                or transition_control is None
            ):
                self.transition_settings_group.setEnabled(
                    False
                )

                return

            self.label_selected_transition.setText(
                f"{transition_label}  ({transition_id})"
            )

            self.transition_settings_group.setEnabled(
                True
            )

            for component in (
                "u",
                "v",
                "h",
            ):
                widgets = self._transition_editor_widgets[
                    component
                ]

                control = transition_control[
                    component
                ]

                widgets[
                    "current"
                ].setValue(
                    float(
                        transition_values[
                            component
                        ]
                    )
                )

                widgets[
                    "enabled"
                ].setChecked(
                    bool(
                        control.get(
                            "enabled",
                            False,
                        )
                    )
                )

                widgets[
                    "minimum"
                ].setValue(
                    float(
                        control.get(
                            "minimum",
                            (
                                0.001
                                if component == "h"
                                else -0.5
                            ),
                        )
                    )
                )

                widgets[
                    "maximum"
                ].setValue(
                    float(
                        control.get(
                            "maximum",
                            (
                                20.0
                                if component == "h"
                                else 0.5
                            ),
                        )
                    )
                )

                widgets[
                    "step"
                ].setValue(
                    float(
                        control.get(
                            "step",
                            0.05,
                        )
                    )
                )

        finally:
            self._transition_editor_syncing = False

        self._refresh_transition_editor_enabled_state()


    def _refresh_transition_editor_enabled_state(
        self,
    ) -> None:
        treatment = self._current_stacking_treatment()

        stacking_enabled = bool(
            self.chk_stacking_enabled.isChecked()
        )

        incremental_mode = bool(
            self.combo_vector_mode.currentData()
            == "incremental"
        )

        valid_transition = bool(
            self._selected_transition_row()
            is not None
        )

        treatment_active = bool(
            stacking_enabled
            and incremental_mode
            and valid_transition
            and treatment
            != STACKING_TREATMENT_FIXED
        )

        self.transition_settings_group.setEnabled(
            valid_transition
        )

        for widgets in (
            self._transition_editor_widgets.values()
        ):
            widgets[
                "current"
            ].setEnabled(
                valid_transition
            )

            widgets[
                "enabled"
            ].setEnabled(
                treatment_active
            )

            component_enabled = bool(
                treatment_active
                and widgets[
                    "enabled"
                ].isChecked()
            )

            widgets[
                "minimum"
            ].setEnabled(
                component_enabled
            )

            widgets[
                "maximum"
            ].setEnabled(
                component_enabled
            )

            widgets[
                "step"
            ].setEnabled(
                component_enabled
                and treatment
                == STACKING_TREATMENT_GRID
            )


    def _save_selected_transition_editor(
        self,
        *args,
    ) -> None:
        _ = args

        if self._transition_editor_syncing:
            return

        row = self._selected_transition_row()

        if row is None:
            return

        transition_control = (
            self.layer_table_model.transition_control_at_row(
                row
            )
        )

        if transition_control is None:
            return

        for component in (
            "u",
            "v",
            "h",
        ):
            widgets = self._transition_editor_widgets[
                component
            ]

            minimum = float(
                widgets[
                    "minimum"
                ].value()
            )

            maximum = float(
                widgets[
                    "maximum"
                ].value()
            )

            if maximum < minimum:
                minimum, maximum = (
                    maximum,
                    minimum,
                )

            transition_control[
                component
            ][
                "enabled"
            ] = bool(
                widgets[
                    "enabled"
                ].isChecked()
            )

            transition_control[
                component
            ][
                "minimum"
            ] = minimum

            transition_control[
                component
            ][
                "maximum"
            ] = maximum

            transition_control[
                component
            ][
                "step"
            ] = max(
                float(
                    widgets[
                        "step"
                    ].value()
                ),
                1.0e-10,
            )

        controls = (
            self.layer_table_model.transition_controls()
        )

        self.layer_table_model.set_transition_controls(
            controls
        )

        self._refresh_transition_editor_enabled_state()


    def _on_stacking_treatment_changed(
        self,
    ) -> None:
        treatment = self._current_stacking_treatment()

        self.layer_table_model.set_treatment(
            treatment
        )

        self._update_stacking_treatment_controls()
        self._load_selected_transition_editor()


    def _update_stacking_treatment_controls(
        self,
    ) -> None:
        stacking_enabled = bool(
            self.chk_stacking_enabled.isChecked()
        )

        stochastic = self._is_stochastic_stacking_model()

        if stochastic:
            self.combo_stacking_treatment.setEnabled(
                False
            )

            self.transition_settings_group.setEnabled(
                False
            )

            if self._is_generic_markov_stacking_model():
                self.label_stacking_treatment_info.setText(
                    "Generic Markov stacking is active. Each Markov state "
                    "emits its own Δu, Δv, and h vector. State-vector "
                    "components may be continuously refined in the generic "
                    "Markov editor. Transition probabilities are handled as "
                    "outer grid-search variables."
                )

            else:
                self.label_stacking_treatment_info.setText(
                    "TOPAS-like stochastic A/B stacking is active. "
                    "State-A and State-B vectors are the continuous parameters. "
                    "P(alternate) is an outer grid-search parameter and is not "
                    "refined by the continuous optimizer."
                )

            return

        incremental_mode = bool(
            self.combo_vector_mode.currentData()
            == "incremental"
        )

        self.combo_stacking_treatment.setEnabled(
            stacking_enabled
            and incremental_mode
        )

        treatment = self._current_stacking_treatment()

        effective_treatment = (
            treatment
            if (
                stacking_enabled
                and incremental_mode
            )
            else STACKING_TREATMENT_FIXED
        )

        self.layer_table_model.set_treatment(
            effective_treatment
        )

        if not stacking_enabled:
            information = (
                "Enable periodic layer stacking to configure vector "
                "refinement or grid search."
            )

        elif not incremental_mode:
            information = (
                "Transition refinement requires incremental vector mode."
            )

        elif treatment == STACKING_TREATMENT_CONTINUOUS:
            information = (
                "Continuous refinement: R/S checkboxes select refined "
                "Δu, Δv and h components. Minimum and maximum are used "
                "as optimizer bounds."
            )

        elif treatment == STACKING_TREATMENT_GRID:
            information = (
                "Finite grid search: R/S checkboxes select scanned "
                "components. Minimum, maximum and step define each grid."
            )

        else:
            information = (
                "Fixed vectors: transition values are used as entered "
                "and are not refined or scanned."
            )

        self.label_stacking_treatment_info.setText(
            information
        )

        self._refresh_transition_editor_enabled_state()


    def _on_layer_sequence_edited(
        self,
    ) -> None:
        """
        Keep transition controls and closure values synchronized after edits.
        """
        self._sync_closure_from_current_stack_spec()
        self._update_stacking_controls()
        self._load_selected_transition_editor()


    def _on_closure_transition_edited(
        self,
        closure_du: float,
        closure_dv: float,
        closure_h: float,
    ) -> None:
        """
        Convert the displayed closing transition back to the existing
        LayerStackSpec repeat representation.
        """
        try:
            layers = self.layer_table_model.layers()

            if not layers:
                return

            temporary_spec = LayerStackSpec(
                enabled=bool(
                    self.chk_stacking_enabled.isChecked()
                ),
                definition_mode=str(
                    self.combo_definition_mode.currentData()
                    or "repeating_motif"
                ),
                vector_mode="incremental",
                nominal_spacing=float(
                    self.spin_nominal_spacing.value()
                ),
                repeat_period=float(
                    self.spin_repeat_period.value()
                ),
                repeat_shift_u=float(
                    self.spin_repeat_shift_u.value()
                ),
                repeat_shift_v=float(
                    self.spin_repeat_shift_v.value()
                ),
                sequence=layers,
            ).normalized()

            absolute_layers = absolute_sequence_layers(
                temporary_spec
            )

            final_layer = absolute_layers[
                -1
            ]

            total_repeat_u = (
                float(
                    final_layer.shift_u
                )
                + float(
                    closure_du
                )
            )

            total_repeat_v = (
                float(
                    final_layer.shift_v
                )
                + float(
                    closure_dv
                )
            )

            spacing = max(
                float(
                    self.spin_nominal_spacing.value()
                ),
                1.0e-12,
            )

            closing_delta_s = (
                float(
                    closure_h
                )
                + float(
                    final_layer.normal_delta
                )
            ) / spacing

            self.spin_repeat_shift_u.blockSignals(
                True
            )

            self.spin_repeat_shift_v.blockSignals(
                True
            )

            self.spin_repeat_period.blockSignals(
                True
            )

            try:
                self.spin_repeat_shift_u.setValue(
                    total_repeat_u
                )

                self.spin_repeat_shift_v.setValue(
                    total_repeat_v
                )

                self.spin_repeat_period.setValue(
                    max(
                        closing_delta_s,
                        1.0e-12,
                    )
                )

            finally:
                self.spin_repeat_shift_u.blockSignals(
                    False
                )

                self.spin_repeat_shift_v.blockSignals(
                    False
                )

                self.spin_repeat_period.blockSignals(
                    False
                )

            self.schedule_preview_update()

        except Exception:
            pass


    def _shape_scales(
        self,
    ) -> tuple[float, float, float]:
        if self.structure is None:
            return (
                1.0,
                1.0,
                1.0,
            )

        try:
            return shape_dimension_lengths(
                np.asarray(
                    self.structure.lattice.matrix,
                    dtype=float,
                ),
                axis_hkl=(
                    self.spin_axis_h.value(),
                    self.spin_axis_k.value(),
                    self.spin_axis_l.value(),
                ),
                base1_hkl=(
                    self.spin_base1_h.value(),
                    self.spin_base1_k.value(),
                    self.spin_base1_l.value(),
                ),
                base2_hkl=(
                    self.spin_base2_h.value(),
                    self.spin_base2_k.value(),
                    self.spin_base2_l.value(),
                ),
            )
        except Exception:
            return (
                1.0,
                1.0,
                1.0,
            )

    def _current_stack_thickness_metrics(
        self,
    ) -> Dict[str, float]:
        """
        Return thickness information for exactly the current Axis-3 layer
        count using the current periodic sequence.
        """
        number_of_layers = max(
            1,
            int(
                round(
                    float(
                        self.spin_axis3.value()
                    )
                )
            ),
        )

        try:
            stack_spec = self._stacking_spec_from_ui()

            realized_layers = realize_n_layers(
                stack_spec,
                number_of_layers=number_of_layers,
                center_stack=True,
            )

            return stack_thickness_metrics(
                realized_layers,
                fallback_spacing=(
                    stack_spec.nominal_spacing
                ),
            )

        except Exception:
            spacing = max(
                float(
                    self.spin_nominal_spacing.value()
                ),
                1.0e-12,
            )

            return {
                "center_span": float(
                    max(
                        0,
                        number_of_layers - 1,
                    )
                    * spacing
                ),
                "boundary_thickness": float(
                    number_of_layers
                    * spacing
                ),
                "lower_boundary": float(
                    -0.5
                    * number_of_layers
                    * spacing
                ),
                "upper_boundary": float(
                    0.5
                    * number_of_layers
                    * spacing
                ),
                "minimum_center": 0.0,
                "maximum_center": 0.0,
            }
        
    def _sync_angstrom_from_cells(
        self,
    ) -> None:
        if self._syncing_dimensions:
            return

        self._syncing_dimensions = True

        try:
            (
                scale_1,
                scale_2,
                scale_3,
            ) = self._shape_scales()

            axis1_angstrom = (
                self.spin_axis1.value()
                * scale_1
            )

            axis2_angstrom = (
                self.spin_axis2.value()
                * scale_2
            )

            if self.chk_stacking_enabled.isChecked():
                metrics = (
                    self._current_stack_thickness_metrics()
                )

                axis3_angstrom = float(
                    metrics[
                        "boundary_thickness"
                    ]
                )
            else:
                axis3_angstrom = (
                    self.spin_axis3.value()
                    * scale_3
                )

            values = (
                (
                    self.spin_axis1_angstrom,
                    axis1_angstrom,
                ),
                (
                    self.spin_axis2_angstrom,
                    axis2_angstrom,
                ),
                (
                    self.spin_axis3_angstrom,
                    axis3_angstrom,
                ),
            )

            for widget, value in values:
                widget.blockSignals(
                    True
                )

                widget.setValue(
                    value
                )

                widget.blockSignals(
                    False
                )

        finally:
            self._syncing_dimensions = False


    def _sync_cells_from_angstrom(
        self,
    ) -> None:
        if self._syncing_dimensions:
            return

        self._syncing_dimensions = True

        try:
            (
                scale_1,
                scale_2,
                scale_3,
            ) = self._shape_scales()

            self.spin_axis1.blockSignals(
                True
            )
            self.spin_axis2.blockSignals(
                True
            )

            try:
                self.spin_axis1.setValue(
                    self.spin_axis1_angstrom.value()
                    / max(
                        scale_1,
                        1.0e-12,
                    )
                )

                self.spin_axis2.setValue(
                    self.spin_axis2_angstrom.value()
                    / max(
                        scale_2,
                        1.0e-12,
                    )
                )

            finally:
                self.spin_axis1.blockSignals(
                    False
                )
                self.spin_axis2.blockSignals(
                    False
                )

            # Axis 3 is derived and read-only in stacking mode.
            if not self.chk_stacking_enabled.isChecked():
                self.spin_axis3.blockSignals(
                    True
                )

                try:
                    self.spin_axis3.setValue(
                        self.spin_axis3_angstrom.value()
                        / max(
                            scale_3,
                            1.0e-12,
                        )
                    )

                finally:
                    self.spin_axis3.blockSignals(
                        False
                    )

        finally:
            self._syncing_dimensions = False


    def _on_shape_changed(
        self,
    ) -> None:
        self._update_shape_controls()
        self._sync_angstrom_from_cells()
        self._update_search_controls()
        self._update_candidate_count()
        self.schedule_preview_update()

    def _on_cells_dimension_changed(
        self,
    ) -> None:
        self._sync_angstrom_from_cells()
        self.schedule_preview_update()

    def _on_angstrom_dimension_changed(
        self,
    ) -> None:
        self._sync_cells_from_angstrom()
        self.schedule_preview_update()

    def _on_orientation_changed(
        self,
    ) -> None:
        self._sync_angstrom_from_cells()
        self.schedule_preview_update()


    def _on_stacking_mode_changed(
        self,
    ) -> None:
        """
        Switch Axis 3 between a continuous geometric dimension and an
        integer number of complete layer positions.
        """
        self._update_stacking_controls()
        self._update_shape_controls()
        self._update_search_controls()
        self._sync_angstrom_from_cells()
        self._update_candidate_count()
        self.schedule_preview_update()


    def _on_vector_mode_changed(
        self,
    ) -> None:
        vector_mode = str(
            self.combo_vector_mode.currentData()
            or "incremental"
        )

        self.layer_table_model.set_vector_mode(
            vector_mode
        )

        if vector_mode == "incremental":
            self.label_repeat_period.setText(
                "Closing Δs to next repeat:"
            )
        else:
            self.label_repeat_period.setText(
                "Absolute repeat period:"
            )

        self._update_stacking_controls()
        self._sync_angstrom_from_cells()
        self.schedule_preview_update()

    def _on_spacing_changed(
        self,
    ) -> None:
        try:
            stack_spec = self._stacking_spec_from_ui()

            self.layer_table_model.set_nominal_spacing(
                float(
                    stack_spec.nominal_spacing
                )
            )

        except Exception:
            self.layer_table_model.set_nominal_spacing(
                float(
                    self.spin_nominal_spacing.value()
                )
            )

        self._sync_angstrom_from_cells()
        self.schedule_preview_update()

    def _update_shape_controls(
        self,
    ) -> None:
        shape_type = (
            self.combo_shape.currentText()
            .strip()
            .lower()
        )

        sphere = (
            shape_type
            == "sphere"
        )

        circular = shape_type in {
            "cylinder",
            "disk",
        }

        stacking_enabled = (
            self.chk_stacking_enabled.isChecked()
        )

        if sphere:
            self.label_axis1.setText(
                "Sphere AB diameter"
                if stacking_enabled
                else "Sphere diameter"
            )

            self.label_axis2.setText(
                "Unused"
            )

        elif circular:
            self.label_axis1.setText(
                "AB-base diameter"
            )

            self.label_axis2.setText(
                "Circular base"
            )

        else:
            self.label_axis1.setText(
                "Base axis 1"
            )

            self.label_axis2.setText(
                "Base axis 2"
            )

        if stacking_enabled:
            self.label_axis3.setText(
                "Number of complete layers"
            )

            self.spin_axis3.setRange(
                1.0,
                100000.0,
            )

            self.spin_axis3.setDecimals(
                0
            )

            self.spin_axis3.setSingleStep(
                1.0
            )

            self.spin_axis3.setSuffix(
                " layers"
            )

            self.spin_axis3_angstrom.setReadOnly(
                True
            )

            self.spin_axis3_angstrom.setButtonSymbols(
                QtWidgets.QAbstractSpinBox.NoButtons
            )

            self.spin_axis3_angstrom.setToolTip(
                "Derived nominal stack thickness. "
                "The particle always contains an integer number "
                "of complete layer positions."
            )

            self.label_search_axis3.setText(
                "Number of layers"
            )

            for widget in (
                self.spin_axis3_min,
                self.spin_axis3_max,
                self.spin_axis3_step,
            ):
                widget.setDecimals(
                    0
                )

                widget.setSingleStep(
                    1.0
                )

                widget.setSuffix(
                    " layers"
                )

                widget.setMinimum(
                    1.0
                )

        else:
            self.label_axis3.setText(
                "Stack axis"
            )

            self.spin_axis3.setRange(
                0.001,
                100000.0,
            )

            self.spin_axis3.setDecimals(
                4
            )

            self.spin_axis3.setSingleStep(
                1.0
            )

            self.spin_axis3.setSuffix(
                " cells"
            )

            self.spin_axis3_angstrom.setReadOnly(
                False
            )

            self.spin_axis3_angstrom.setButtonSymbols(
                QtWidgets.QAbstractSpinBox.UpDownArrows
            )

            self.spin_axis3_angstrom.setToolTip(
                ""
            )

            self.label_search_axis3.setText(
                "Stack axis"
            )

            for widget in (
                self.spin_axis3_min,
                self.spin_axis3_max,
                self.spin_axis3_step,
            ):
                widget.setDecimals(
                    4
                )

                widget.setSingleStep(
                    1.0
                )

                widget.setSuffix(
                    " cells"
                )

                widget.setMinimum(
                    0.001
                )

        axis2_enabled = not (
            sphere
            or circular
        )

        # Axis 3 must remain available in stacking mode even for a sphere:
        # it controls the number of complete layer slices.
        axis3_enabled = (
            stacking_enabled
            or not sphere
        )

        self.spin_axis2.setEnabled(
            axis2_enabled
        )

        self.spin_axis2_angstrom.setEnabled(
            axis2_enabled
        )

        self.spin_axis3.setEnabled(
            axis3_enabled
        )

        self.spin_axis3_angstrom.setEnabled(
            axis3_enabled
        )


    def _update_search_controls(
        self,
    ) -> None:
        grid_mode = (
            self.combo_search_mode.currentData()
            == "grid_search"
        )

        shape_type = (
            self.combo_shape.currentText()
            .strip()
            .lower()
        )

        stacking_enabled = (
            self.chk_stacking_enabled.isChecked()
        )

        sphere = (
            shape_type
            == "sphere"
        )

        circular = shape_type in {
            "cylinder",
            "disk",
        }

        axis2_available = not (
            sphere
            or circular
        )

        # In stacking mode Axis 3 always means layer count.
        axis3_available = (
            stacking_enabled
            or not sphere
        )

        self.chk_scan_axis1.setEnabled(
            grid_mode
        )

        self.chk_scan_axis2.setEnabled(
            grid_mode
            and axis2_available
        )

        self.chk_scan_axis3.setEnabled(
            grid_mode
            and axis3_available
        )

        groups = (
            (
                self.chk_scan_axis1,
                True,
                self.spin_axis1_min,
                self.spin_axis1_max,
                self.spin_axis1_step,
            ),
            (
                self.chk_scan_axis2,
                axis2_available,
                self.spin_axis2_min,
                self.spin_axis2_max,
                self.spin_axis2_step,
            ),
            (
                self.chk_scan_axis3,
                axis3_available,
                self.spin_axis3_min,
                self.spin_axis3_max,
                self.spin_axis3_step,
            ),
        )

        for (
            checkbox,
            available,
            minimum,
            maximum,
            step,
        ) in groups:
            enabled = (
                grid_mode
                and available
                and checkbox.isChecked()
            )

            minimum.setEnabled(
                enabled
            )

            maximum.setEnabled(
                enabled
            )

            step.setEnabled(
                enabled
            )


    @staticmethod
    def _scan_count(
        minimum: float,
        maximum: float,
        step: float,
    ) -> int:
        if maximum < minimum:
            minimum, maximum = (
                maximum,
                minimum,
            )

        step = max(
            float(step),
            1.0e-12,
        )

        return max(
            1,
            int(
                np.floor(
                    (maximum - minimum)
                    / step
                    + 0.5
                )
            )
            + 1,
        )

    def _update_candidate_count(
        self,
    ) -> None:
        self._update_search_controls()

        if (
            self.combo_search_mode.currentData()
            != "grid_search"
        ):
            self.label_candidate_count.setText(
                "Shape candidates: 1"
            )
            return

        shape_type = self.combo_shape.currentText().strip().lower()

        count_1 = (
            self._scan_count(
                self.spin_axis1_min.value(),
                self.spin_axis1_max.value(),
                self.spin_axis1_step.value(),
            )
            if self.chk_scan_axis1.isChecked()
            else 1
        )

        if shape_type in {
            "sphere",
            "cylinder",
            "disk",
        }:
            count_2 = 1
        else:
            count_2 = (
                self._scan_count(
                    self.spin_axis2_min.value(),
                    self.spin_axis2_max.value(),
                    self.spin_axis2_step.value(),
                )
                if self.chk_scan_axis2.isChecked()
                else 1
            )

        stacking_enabled = (
            self.chk_stacking_enabled.isChecked()
        )

        if (
            shape_type == "sphere"
            and not stacking_enabled
        ):
            count_3 = 1

        else:
            count_3 = (
                self._scan_count(
                    self.spin_axis3_min.value(),
                    self.spin_axis3_max.value(),
                    self.spin_axis3_step.value(),
                )
                if self.chk_scan_axis3.isChecked()
                else 1
            )

        total = (
            count_1
            * count_2
            * count_3
        )

        self.label_candidate_count.setText(
            f"Shape candidates: {total:,}"
        )

    def _update_spacing_mode_controls(
        self,
    ) -> None:
        stacking_enabled = bool(
            self.chk_stacking_enabled.isChecked()
        )

        fraction_mode = bool(
            stacking_enabled
            and self.combo_spacing_mode.currentData()
            == "fraction_of_c_normal"
        )

        independent_mode = bool(
            stacking_enabled
            and not fraction_mode
        )

        self.combo_spacing_mode.setEnabled(
            stacking_enabled
        )

        self.spin_nominal_spacing.setEnabled(
            independent_mode
        )

        self.chk_refine_nominal_spacing.setEnabled(
            independent_mode
        )

        refine_distance = bool(
            independent_mode
            and self.chk_refine_nominal_spacing.isChecked()
        )

        self.spin_nominal_spacing_min.setEnabled(
            refine_distance
        )

        self.spin_nominal_spacing_max.setEnabled(
            refine_distance
        )

        self.spin_stacking_fraction.setEnabled(
            fraction_mode
        )

        self.chk_refine_stacking_fraction.setEnabled(
            fraction_mode
        )

        refine_fraction = bool(
            fraction_mode
            and self.chk_refine_stacking_fraction.isChecked()
        )

        self.spin_stacking_fraction_min.setEnabled(
            refine_fraction
        )

        self.spin_stacking_fraction_max.setEnabled(
            refine_fraction
        )

    def _update_stacking_controls(
        self,
    ) -> None:
        enabled = bool(
            self.chk_stacking_enabled.isChecked()
        )

        stochastic = self._is_stochastic_stacking_model()

        widgets = (
            self.combo_definition_mode,
            self.combo_vector_mode,
            self.spin_repeat_period,
            self.spin_repeat_shift_u,
            self.spin_repeat_shift_v,
            self.layer_table,
            self.btn_add_before,
            self.btn_add_after,
            self.btn_duplicate,
            self.btn_remove,
            self.btn_move_up,
            self.btn_move_down,
            self.btn_set_sequence_origin,
            self.btn_normalize_coordinates,
            self.btn_generate_a,
            self.btn_generate_ab,
            self.btn_generate_abc,
            self.chk_show_generated_table,
            self.generated_layer_table,
            self.chk_show_layer_origins,
            self.chk_show_transition_vectors,
            self.chk_color_stacking_layers,
            self.btn_export_stacked_xyz,
        )

        for widget in widgets:
            widget.setEnabled(
                enabled
            )

        deterministic_enabled = bool(
            enabled
            and not stochastic
        )

        for widget in (
            self.combo_definition_mode,
            self.combo_vector_mode,
            self.spin_repeat_period,
            self.spin_repeat_shift_u,
            self.spin_repeat_shift_v,
            self.layer_table,
            self.btn_add_before,
            self.btn_add_after,
            self.btn_duplicate,
            self.btn_remove,
            self.btn_move_up,
            self.btn_move_down,
            self.btn_set_sequence_origin,
            self.btn_normalize_coordinates,
            self.btn_generate_a,
            self.btn_generate_ab,
            self.btn_generate_abc,
            self.transition_settings_group,
        ):
            widget.setEnabled(
                deterministic_enabled
            )

        self._update_stochastic_controls()

        self._update_spacing_mode_controls()

        self._update_stacking_search_controls()

        self._update_stacking_treatment_controls()


    def _update_layer_table_height(
        self,
    ) -> None:
        """
        Size the editable sequence table so all ordinary motif rows and the
        periodic closing-transition row are visible.

        Very long motifs are limited to 12 visible rows, after which the
        table uses its vertical scrollbar.
        """
        if not hasattr(
            self,
            "layer_table",
        ):
            return

        if not hasattr(
            self,
            "layer_table_model",
        ):
            return

        row_count = int(
            self.layer_table_model.rowCount()
        )

        visible_row_count = max(
            1,
            min(
                row_count,
                12,
            ),
        )

        row_height = max(
            int(
                self.layer_table.verticalHeader()
                .defaultSectionSize()
            ),
            30,
        )

        header_height = max(
            int(
                self.layer_table.horizontalHeader()
                .sizeHint()
                .height()
            ),
            30,
        )

        frame_height = (
            2
            * int(
                self.layer_table.frameWidth()
            )
        )

        scrollbar_margin = 8

        table_height = (
            header_height
            + visible_row_count
            * row_height
            + frame_height
            + scrollbar_margin
        )

        self.layer_table.setMinimumHeight(
            table_height
        )

        self.layer_table.setMaximumHeight(
            table_height
        )


    def _schedule_layer_table_height_update(
        self,
    ) -> None:
        """
        Recalculate the table height after Qt finishes resetting the model.
        """
        QtCore.QTimer.singleShot(
            0,
            self._update_layer_table_height,
        )


    def _selected_row(
        self,
    ) -> Optional[int]:
        selection = self.layer_table.selectionModel()

        if selection is None:
            return None

        rows = selection.selectedRows()

        if not rows:
            return None

        return rows[0].row()

    def _set_layers_and_select(
        self,
        layers,
        row: int,
    ) -> None:
        self.layer_table_model.set_layers(
            layers
        )

        self._update_layer_table_height()

        if layers:
            row = max(
                0,
                min(
                    row,
                    len(layers) - 1,
                ),
            )
            self.layer_table.selectRow(
                row
            )

        self.schedule_preview_update()

    def _new_layer_near(
        self,
        row: int,
        *,
        after: bool,
    ) -> tuple[SequenceLayer, int]:
        layers = self.layer_table_model.layers()

        if not layers:
            return (
                SequenceLayer(),
                0,
            )

        row = max(
            0,
            min(
                row,
                len(layers) - 1,
            ),
        )

        selected = layers[row]

        if after:
            insert_row = row + 1
            coordinate = (
                selected.stack_coordinate
                + 1.0
            )
        else:
            insert_row = row
            coordinate = (
                selected.stack_coordinate
                - 1.0
            )

        layer = SequenceLayer(
            enabled=True,
            label="A",
            shift_u=selected.shift_u,
            shift_v=selected.shift_v,
            stack_coordinate=coordinate,
            normal_delta=0.0,
        )

        return (
            layer,
            insert_row,
        )

    def _add_before(
        self,
    ) -> None:
        layers = self.layer_table_model.layers()
        row = self._selected_row()

        if row is None:
            row = 0

        layer, insert_row = self._new_layer_near(
            row,
            after=False,
        )

        layers.insert(
            insert_row,
            layer,
        )

        self._set_layers_and_select(
            layers,
            insert_row,
        )

    def _add_after(
        self,
    ) -> None:
        layers = self.layer_table_model.layers()
        row = self._selected_row()

        if row is None:
            row = max(
                0,
                len(layers) - 1,
            )

        layer, insert_row = self._new_layer_near(
            row,
            after=True,
        )

        layers.insert(
            insert_row,
            layer,
        )

        self._set_layers_and_select(
            layers,
            insert_row,
        )

    def _duplicate_selected(
        self,
    ) -> None:
        row = self._selected_row()

        if row is None:
            return

        layers = self.layer_table_model.layers()
        source = layers[row]

        duplicate = SequenceLayer(
            enabled=source.enabled,
            label=source.label,
            shift_u=source.shift_u,
            shift_v=source.shift_v,
            stack_coordinate=(
                source.stack_coordinate
                + 1.0
            ),
            normal_delta=source.normal_delta,
        )

        insert_row = row + 1

        layers.insert(
            insert_row,
            duplicate,
        )

        self._set_layers_and_select(
            layers,
            insert_row,
        )

    def _remove_selected(
        self,
    ) -> None:
        row = self._selected_row()

        if row is None:
            return

        layers = self.layer_table_model.layers()

        if len(layers) <= 1:
            QtWidgets.QMessageBox.information(
                self,
                "Sequence requires one row",
                "A periodic sequence must contain at least one row.",
            )
            return

        layers.pop(
            row
        )

        self._set_layers_and_select(
            layers,
            row,
        )

    def _move_selected_up(
        self,
    ) -> None:
        row = self._selected_row()

        if row is None or row <= 0:
            return

        layers = self.layer_table_model.layers()

        layers[row - 1], layers[row] = (
            layers[row],
            layers[row - 1],
        )

        self._set_layers_and_select(
            layers,
            row - 1,
        )

    def _move_selected_down(
        self,
    ) -> None:
        row = self._selected_row()
        layers = self.layer_table_model.layers()

        if (
            row is None
            or row >= len(layers) - 1
        ):
            return

        layers[row + 1], layers[row] = (
            layers[row],
            layers[row + 1],
        )

        self._set_layers_and_select(
            layers,
            row + 1,
        )

    def _set_sequence_origin(
        self,
    ) -> None:
        row = self._selected_row()

        if row is None:
            return

        layers = self.layer_table_model.layers()
        reference = layers[row]

        for layer in layers:
            layer.shift_u -= reference.shift_u
            layer.shift_v -= reference.shift_v
            layer.stack_coordinate -= reference.stack_coordinate
            layer.normal_delta -= reference.normal_delta

        self._set_layers_and_select(
            layers,
            row,
        )

    def _normalize_coordinates(
        self,
    ) -> None:
        layers = self.layer_table_model.layers()

        vector_mode = str(
            self.combo_vector_mode.currentData()
            or "incremental"
        )

        for row, layer in enumerate(
            layers
        ):
            if vector_mode == "incremental":
                layer.stack_coordinate = (
                    0.0
                    if row == 0
                    else 1.0
                )
            else:
                layer.stack_coordinate = float(
                    row
                )

            layer.normal_delta = 0.0

        if vector_mode == "incremental":
            self.spin_repeat_period.setValue(
                1.0
            )
        else:
            self.spin_repeat_period.setValue(
                float(
                    max(
                        1,
                        len(layers),
                    )
                )
            )


        self._set_layers_and_select(
            layers,
            0,
        )

    def _set_generated_sequence(
        self,
        layers,
    ) -> None:
        self.layer_table_model.set_layers(
            layers
        )

        self._update_layer_table_height()

        self.spin_repeat_period.setValue(
            float(
                max(
                    1,
                    len(layers),
                )
            )
        )

        self.spin_repeat_shift_u.setValue(
            0.0
        )
        self.spin_repeat_shift_v.setValue(
            0.0
        )

        self.layer_table.selectRow(
            0
        )

        self._update_stacking_controls()
        self.schedule_preview_update()

    def _shape_spec_from_ui(
        self,
    ) -> CrystalliteShapeSpec:
        axis3_value = float(
            self.spin_axis3.value()
        )

        if self.chk_stacking_enabled.isChecked():
            axis3_value = float(
                max(
                    1,
                    int(
                        round(
                            axis3_value
                        )
                    ),
                )
            )

        return CrystalliteShapeSpec(
            shape_type=(
                self.combo_shape.currentText()
                .strip()
            ),
            search_mode=str(
                self.combo_search_mode.currentData()
                or "predefined"
            ),
            diameter_cells=float(
                self.spin_axis1.value()
            ),
            height_cells=axis3_value,
            axis1_cells=float(
                self.spin_axis1.value()
            ),
            axis2_cells=float(
                self.spin_axis2.value()
            ),
            axis3_cells=axis3_value,
            axis_h=int(
                self.spin_axis_h.value()
            ),
            axis_k=int(
                self.spin_axis_k.value()
            ),
            axis_l=int(
                self.spin_axis_l.value()
            ),
            base1_h=int(
                self.spin_base1_h.value()
            ),
            base1_k=int(
                self.spin_base1_k.value()
            ),
            base1_l=int(
                self.spin_base1_l.value()
            ),
            base2_h=int(
                self.spin_base2_h.value()
            ),
            base2_k=int(
                self.spin_base2_k.value()
            ),
            base2_l=int(
                self.spin_base2_l.value()
            ),
        )

    
    def _stacking_spec_from_ui(
        self,
    ) -> LayerStackSpec:
        spec = LayerStackSpec(
            enabled=bool(
                self.chk_stacking_enabled.isChecked()
            ),
            model=str(
                self.combo_stacking_model.currentData()
                or "periodic_sequence"
            ),
            p_alternate=float(
                self.spin_p_alternate.value()
            ),
            p_grid_enabled=bool(
                self.chk_p_grid_enabled.isChecked()
            ),
            p_grid_minimum=float(
                self.spin_p_grid_min.value()
            ),
            p_grid_maximum=float(
                self.spin_p_grid_max.value()
            ),
            p_grid_step=float(
                self.spin_p_grid_step.value()
            ),
            ensemble_samples=int(
                self.spin_ensemble_samples.value()
            ),
            ensemble_seed=int(
                self.spin_ensemble_seed.value()
            ),
            ensemble_start_state=str(
                self.combo_ensemble_start_state.currentText()
                or "A"
            ),
            preview_realization_index=int(
                self.spin_preview_realization.value()
            ),
            state_a_du=float(
                self.spin_state_a_du.value()
            ),
            state_a_dv=float(
                self.spin_state_a_dv.value()
            ),
            state_a_h=float(
                self.spin_state_a_h.value()
            ),
            state_b_du=float(
                self.spin_state_b_du.value()
            ),
            state_b_dv=float(
                self.spin_state_b_dv.value()
            ),
            state_b_h=float(
                self.spin_state_b_h.value()
            ),
            markov_states=[
                dict(
                    state
                )
                for state in getattr(
                    self,
                    "_generic_markov_states",
                    [],
                )
            ],
            markov_transition_matrix=[
                list(
                    row
                )
                for row in getattr(
                    self,
                    "_generic_markov_transition_matrix",
                    [],
                )
            ],
            markov_probability_controls=[
                [
                    dict(
                        control
                    )
                    for control in row
                ]
                for row in getattr(
                    self,
                    "_generic_markov_probability_controls",
                    [],
                )
            ],
            markov_start_state=int(
                getattr(
                    self,
                    "_generic_markov_start_state",
                    0,
                )
            ),
            definition_mode=str(
                self.combo_definition_mode.currentData()
                or "repeating_motif"
            ),
            vector_mode=str(
                self.combo_vector_mode.currentData()
                or "incremental"
            ),
            spacing_mode=str(
                self.combo_spacing_mode.currentData()
                or "independent"
            ),
            nominal_spacing=float(
                self.spin_nominal_spacing.value()
            ),
            refine_nominal_spacing=bool(
                self.chk_refine_nominal_spacing.isChecked()
            ),
            nominal_spacing_min=float(
                self.spin_nominal_spacing_min.value()
            ),
            nominal_spacing_max=float(
                self.spin_nominal_spacing_max.value()
            ),
            stacking_fraction=float(
                self.spin_stacking_fraction.value()
            ),
            refine_stacking_fraction=bool(
                self.chk_refine_stacking_fraction.isChecked()
            ),
            stacking_fraction_min=float(
                self.spin_stacking_fraction_min.value()
            ),
            stacking_fraction_max=float(
                self.spin_stacking_fraction_max.value()
            ),
            repeat_period=float(
                self.spin_repeat_period.value()
            ),
            repeat_shift_u=float(
                self.spin_repeat_shift_u.value()
            ),
            repeat_shift_v=float(
                self.spin_repeat_shift_v.value()
            ),
            sequence=self.layer_table_model.layers(),
            show_transition_vectors=bool(
                self.chk_show_transition_vectors.isChecked()
            ),
            color_layers=bool(
                self.chk_color_stacking_layers.isChecked()
            ),
        )


        spec = spec.normalized()

        if (
            spec.spacing_mode
            == "fraction_of_c_normal"
            and self.structure is not None
        ):
            from pdf_fitting.models.layer_stacking import (
                evaluate_stacking_spacing,
            )

            spec.nominal_spacing = float(
                evaluate_stacking_spacing(
                    spec,
                    np.asarray(
                        self.structure.lattice.matrix,
                        dtype=float,
                    ),
                    parameters={
                        "stacking_fraction": (
                            spec.stacking_fraction
                        ),
                    },
                )
            )

        return spec

    
    def _state_from_ui(
        self,
    ) -> FiniteCrystalliteState:
        stack = self._stacking_spec_from_ui()

        axis3_value = float(
            self.spin_axis3.value()
        )

        if stack.enabled:
            axis3_value = float(
                max(
                    1,
                    int(
                        round(
                            axis3_value
                        )
                    ),
                )
            )

        return FiniteCrystalliteState(
            shape_type=self.combo_shape.currentText().strip(),
            axis1_cells=self.spin_axis1.value(),
            axis2_cells=self.spin_axis2.value(),
            axis3_cells=axis3_value,
            search_mode=str(
                self.combo_search_mode.currentData()
                or "predefined"
            ),
            scan_axis1=self.chk_scan_axis1.isChecked(),
            axis1_min_cells=self.spin_axis1_min.value(),
            axis1_max_cells=self.spin_axis1_max.value(),
            axis1_step_cells=self.spin_axis1_step.value(),
            scan_axis2=self.chk_scan_axis2.isChecked(),
            axis2_min_cells=self.spin_axis2_min.value(),
            axis2_max_cells=self.spin_axis2_max.value(),
            axis2_step_cells=self.spin_axis2_step.value(),
            scan_axis3=self.chk_scan_axis3.isChecked(),
            axis3_min_cells=self.spin_axis3_min.value(),
            axis3_max_cells=self.spin_axis3_max.value(),
            axis3_step_cells=self.spin_axis3_step.value(),
            axis_h=self.spin_axis_h.value(),
            axis_k=self.spin_axis_k.value(),
            axis_l=self.spin_axis_l.value(),
            base1_h=self.spin_base1_h.value(),
            base1_k=self.spin_base1_k.value(),
            base1_l=self.spin_base1_l.value(),
            base2_h=self.spin_base2_h.value(),
            base2_k=self.spin_base2_k.value(),
            base2_l=self.spin_base2_l.value(),
            show_shape_surface=self.chk_show_shape_surface.isChecked(),
            show_shape_axes=self.chk_show_shape_axes.isChecked(),
            shape_opacity=self.spin_shape_opacity.value(),
            show_atoms=self.chk_show_atoms.isChecked(),
            show_layer_origins=self.chk_show_layer_origins.isChecked(),
            all_atom_preview_limit=self.spin_atom_limit.value(),
            orthographic_projection=self.chk_stacking_orthographic.isChecked(),
            show_layer_labels=self.chk_show_layer_labels.isChecked(),
            show_parent_unit_cell=self.chk_show_parent_unit_cell.isChecked(),
            show_lattice_vectors=self.chk_show_lattice_vectors.isChecked(),
            preview_view_preset="ac_section",
            preview_view_mode="3d",
            preview_max_layer_labels=self.spin_max_layer_labels.value(),
            preview_label_font_size=self.spin_layer_label_font_size.value(),
            stacking_model=stack.model,
            stacking_p_alternate=stack.p_alternate,
            stacking_p_grid_enabled=stack.p_grid_enabled,
            stacking_p_grid_minimum=stack.p_grid_minimum,
            stacking_p_grid_maximum=stack.p_grid_maximum,
            stacking_p_grid_step=stack.p_grid_step,
            stacking_ensemble_samples=stack.ensemble_samples,
            stacking_ensemble_seed=stack.ensemble_seed,
            stacking_ensemble_start_state=stack.ensemble_start_state,
            stacking_preview_realization_index=(
                stack.preview_realization_index
            ),
            stacking_state_a_du=stack.state_a_du,
            stacking_state_a_dv=stack.state_a_dv,
            stacking_state_a_h=stack.state_a_h,
            stacking_state_b_du=stack.state_b_du,
            stacking_state_b_dv=stack.state_b_dv,
            stacking_state_b_h=stack.state_b_h,
            stacking_enabled=stack.enabled,
            stacking_definition_mode=stack.definition_mode,
            stacking_vector_mode=stack.vector_mode,
            stacking_nominal_spacing=stack.nominal_spacing,
            refine_stacking_nominal_spacing=stack.refine_nominal_spacing,
            stacking_nominal_spacing_min=stack.nominal_spacing_min,
            stacking_nominal_spacing_max=stack.nominal_spacing_max,
            stacking_spacing_mode=stack.spacing_mode,
            stacking_fraction=stack.stacking_fraction,
            refine_stacking_fraction=stack.refine_stacking_fraction,
            stacking_fraction_min=stack.stacking_fraction_min,
            stacking_fraction_max=stack.stacking_fraction_max,
            stacking_repeat_period=stack.repeat_period,
            stacking_repeat_shift_u=stack.repeat_shift_u,
            stacking_repeat_shift_v=stack.repeat_shift_v,
            stacking_sequence=[
                layer.to_dict()
                for layer in stack.sequence
            ],
            show_transition_vectors=stack.show_transition_vectors,
            color_stacking_layers=stack.color_layers,
            show_generated_layers=self.chk_show_generated_table.isChecked(),
        )

    @staticmethod
    def _sequence_colors(
        sequence_ids: np.ndarray,
        *,
        enabled: bool,
    ):
        if not enabled:
            return None

        sequence_ids = np.asarray(
            sequence_ids,
            dtype=np.int32,
        )

        if sequence_ids.size == 0:
            return None

        palette = np.asarray(
            [
                (0.90, 0.15, 0.15, 1.0),
                (0.10, 0.45, 0.90, 1.0),
                (0.15, 0.70, 0.25, 1.0),
                (0.82, 0.35, 0.85, 1.0),
                (0.95, 0.55, 0.10, 1.0),
                (0.10, 0.72, 0.72, 1.0),
                (0.55, 0.30, 0.10, 1.0),
                (0.45, 0.45, 0.45, 1.0),
            ],
            dtype=np.float32,
        )

        return palette[
            sequence_ids
            % palette.shape[0]
        ]

    def _update_sequence_statistics(
        self,
        realized_layers,
    ) -> None:
        stats = {}

        for layer in realized_layers:
            copies, atoms = stats.get(
                layer.sequence_index,
                (0, 0),
            )

            if layer.atom_count > 0:
                copies += 1

            atoms += int(
                layer.atom_count
            )

            stats[layer.sequence_index] = (
                copies,
                atoms,
            )

        self.layer_table_model.set_visible_stats(
            stats
        )

        # Background colors depend on the statistics, but repainting the
        # viewport does not reset active cell editors.
        if hasattr(
            self,
            "layer_table",
        ):
            self.layer_table.viewport().update()

    def _on_layer_editor_closed(
        self,
        editor,
        hint=None,
    ) -> None:
        """
        Repaint the editable table after Qt removes the cell editor.

        The model already contains the committed value at this point.
        """
        QtCore.QTimer.singleShot(
            0,
            self._refresh_layer_table_after_edit,
        )

    def _refresh_layer_table_after_edit(
        self,
    ) -> None:
        if not hasattr(
            self,
            "layer_table",
        ):
            return

        self.layer_table.clearFocus()
        self.layer_table.viewport().update()

        self.schedule_preview_update()


    def schedule_preview_update(
        self,
    ) -> None:
        self._preview_timer.start()

    def _set_preview_preset(
        self,
        preset: str,
    ) -> None:
        self.preview.set_view_preset(
            preset
        )

    def _apply_custom_preview_view(
        self,
    ) -> None:
        if self.structure is None:
            return

        lattice = np.asarray(
            self.structure.lattice.matrix,
            dtype=float,
        )

        view_indices = np.asarray(
            [
                self.spin_view_h.value(),
                self.spin_view_k.value(),
                self.spin_view_l.value(),
            ],
            dtype=float,
        )

        horizontal_indices = np.asarray(
            [
                self.spin_view_horizontal_h.value(),
                self.spin_view_horizontal_k.value(),
                self.spin_view_horizontal_l.value(),
            ],
            dtype=float,
        )

        if np.linalg.norm(
            view_indices
        ) <= 1.0e-12:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid viewing direction",
                "The viewing direction cannot be [0 0 0].",
            )
            return

        if np.linalg.norm(
            horizontal_indices
        ) <= 1.0e-12:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid horizontal direction",
                "The horizontal direction cannot be [0 0 0].",
            )
            return

        mode = str(
            self.combo_custom_view_mode.currentData()
            or "direction"
        )

        if mode == "plane_normal":
            reciprocal = np.asarray(
                self.structure.lattice.reciprocal_lattice.matrix,
                dtype=float,
            )

            view_cartesian = (
                view_indices
                @ reciprocal
            )

        else:
            view_cartesian = (
                view_indices
                @ lattice
            )

        horizontal_cartesian = (
            horizontal_indices
            @ lattice
        )

        self.preview.set_custom_view_direction(
            view_cartesian,
            horizontal_direction=horizontal_cartesian,
        )

    def _export_realized_layers_csv(
        self,
    ) -> None:
        if self.structure is None:
            return

        try:
            particle = build_stacked_particle(
                structure=self.structure,
                shape_spec=self._shape_spec_from_ui(),
                stack_spec=self._stacking_spec_from_ui(),
                max_atoms=None,
            )

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export realized layers as CSV",
                "realized_layers.csv",
                "CSV files (*.csv);;All files (*)",
            )

            if not path:
                return

            if not path.lower().endswith(
                ".csv"
            ):
                path += ".csv"

            with open(
                path,
                "w",
                encoding="utf-8",
            ) as file:
                file.write(
                    "realized_layer_index,"
                    "sequence_row,"
                    "repeat_index,"
                    "layer_label,"
                    "shift_u,"
                    "shift_v,"
                    "stack_coordinate,"
                    "normal_offset_angstrom,"
                    "origin_x_angstrom,"
                    "origin_y_angstrom,"
                    "origin_z_angstrom,"
                    "atom_count\n"
                )

                for layer, origin in zip(
                    particle.realized_layers,
                    particle.layer_origins,
                ):
                    file.write(
                        f"{int(layer.realized_index)},"
                        f"{int(layer.sequence_index)},"
                        f"{int(layer.repeat_index)},"
                        f"{str(layer.label)},"
                        f"{float(layer.shift_u):.12g},"
                        f"{float(layer.shift_v):.12g},"
                        f"{float(layer.stack_coordinate):.12g},"
                        f"{float(layer.normal_offset):.12g},"
                        f"{float(origin[0]):.12g},"
                        f"{float(origin[1]):.12g},"
                        f"{float(origin[2]):.12g},"
                        f"{int(layer.atom_count)}\n"
                    )

            QtWidgets.QMessageBox.information(
                self,
                "Layer CSV exported",
                f"Saved realized layers to:\n{path}",
            )

        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Layer CSV export failed",
                str(exc),
            )

    def _export_transition_vectors_csv(
        self,
    ) -> None:
        if self.structure is None:
            return

        try:
            particle = build_stacked_particle(
                structure=self.structure,
                shape_spec=self._shape_spec_from_ui(),
                stack_spec=self._stacking_spec_from_ui(),
                max_atoms=None,
            )

            starts = np.asarray(
                particle.transition_starts,
                dtype=float,
            )

            ends = np.asarray(
                particle.transition_ends,
                dtype=float,
            )

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export stacking vectors as CSV",
                "stacking_vectors.csv",
                "CSV files (*.csv);;All files (*)",
            )

            if not path:
                return

            if not path.lower().endswith(
                ".csv"
            ):
                path += ".csv"

            with open(
                path,
                "w",
                encoding="utf-8",
            ) as file:
                file.write(
                    "transition_index,"
                    "from_x_angstrom,"
                    "from_y_angstrom,"
                    "from_z_angstrom,"
                    "to_x_angstrom,"
                    "to_y_angstrom,"
                    "to_z_angstrom,"
                    "dx_angstrom,"
                    "dy_angstrom,"
                    "dz_angstrom,"
                    "length_angstrom\n"
                )

                for index, (
                    start,
                    end,
                ) in enumerate(
                    zip(
                        starts,
                        ends,
                    )
                ):
                    delta = end - start
                    length = float(
                        np.linalg.norm(
                            delta
                        )
                    )

                    file.write(
                        f"{index},"
                        f"{float(start[0]):.12g},"
                        f"{float(start[1]):.12g},"
                        f"{float(start[2]):.12g},"
                        f"{float(end[0]):.12g},"
                        f"{float(end[1]):.12g},"
                        f"{float(end[2]):.12g},"
                        f"{float(delta[0]):.12g},"
                        f"{float(delta[1]):.12g},"
                        f"{float(delta[2]):.12g},"
                        f"{length:.12g}\n"
                    )

            QtWidgets.QMessageBox.information(
                self,
                "Stacking-vector CSV exported",
                f"Saved transition vectors to:\n{path}",
            )

        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Stacking-vector export failed",
                str(exc),
            )

    def _export_preview_png(
        self,
    ) -> None:
        try:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save preview screenshot",
                "finite_crystallite_preview.png",
                "PNG files (*.png);;All files (*)",
            )

            if not path:
                return

            if not path.lower().endswith(
                ".png"
            ):
                path += ".png"

            self.preview.save_screenshot(
                path
            )

            QtWidgets.QMessageBox.information(
                self,
                "Preview screenshot saved",
                f"Saved preview image to:\n{path}",
            )

        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Preview screenshot export failed",
                str(exc),
            )

    def update_preview(
        self,
    ) -> None:
        """
        Rebuild the finite-crystallite preview.

        Supported stacking models:

            periodic_sequence
            correlated_ab_disorder
            generic_markov
        """
        if (
            hasattr(
                self,
                "layer_table",
            )
            and self.layer_table.state()
            == QtWidgets.QAbstractItemView.EditingState
        ):
            self._preview_timer.start(
                150
            )
            return

        if self.structure is None:
            self.label_info.setText(
                "No crystal structure is loaded."
            )

            self.preview.clear_scene()
            return

        try:
            state = self._state_from_ui()
            self.state = state

            lattice = np.asarray(
                self.structure.lattice.matrix,
                dtype=np.float32,
            )

            shape_spec = self._shape_spec_from_ui()

            if state.stacking_enabled:
                stack_spec = self._stacking_spec_from_ui()

                stochastic_path_set = None
                stochastic_statistics = None
                realization_index = 0

                # -------------------------------------------------
                # Correlated two-state A/B stochastic model
                # -------------------------------------------------
                if (
                    stack_spec.model
                    == "correlated_ab_disorder"
                ):
                    stochastic_path_set = (
                        self._stochastic_preview_paths(
                            stack_spec
                        )
                    )

                    realization_index = min(
                        max(
                            0,
                            int(
                                self.spin_preview_realization.value()
                            ),
                        ),
                        max(
                            0,
                            int(
                                stochastic_path_set.ensemble_samples
                            )
                            - 1,
                        ),
                    )

                    preview_states = (
                        stochastic_path_set.states[
                            realization_index
                        ]
                    )

                    particle = build_stochastic_stacked_particle(
                        structure=self.structure,
                        shape_spec=shape_spec,
                        stack_spec=stack_spec,
                        states=preview_states,
                        parameters={},
                        max_atoms=state.all_atom_preview_limit,
                    )

                    stochastic_statistics = (
                        state_transition_statistics(
                            stochastic_path_set.states
                        )
                    )

                # -------------------------------------------------
                # Generic N-state Markov model
                # -------------------------------------------------
                elif (
                    stack_spec.model
                    == "generic_markov"
                ):
                    stochastic_path_set = (
                        self._stochastic_preview_paths(
                            stack_spec
                        )
                    )

                    realization_index = min(
                        max(
                            0,
                            int(
                                self.spin_preview_realization.value()
                            ),
                        ),
                        max(
                            0,
                            int(
                                stochastic_path_set.ensemble_samples
                            )
                            - 1,
                        ),
                    )

                    preview_states = (
                        stochastic_path_set.states[
                            realization_index
                        ]
                    )

                    particle = build_stochastic_stacked_particle(
                        structure=self.structure,
                        shape_spec=shape_spec,
                        stack_spec=stack_spec,
                        states=preview_states,
                        parameters={},
                        max_atoms=state.all_atom_preview_limit,
                    )

                # -------------------------------------------------
                # Deterministic periodic sequence
                # -------------------------------------------------
                else:
                    particle = build_stacked_particle(
                        structure=self.structure,
                        shape_spec=shape_spec,
                        stack_spec=stack_spec,
                        max_atoms=state.all_atom_preview_limit,
                    )

                # -------------------------------------------------
                # Update generated-layer tables and preview
                # -------------------------------------------------
                self.generated_layer_model.set_layers(
                    particle.realized_layers
                )

                self._update_sequence_statistics(
                    particle.realized_layers
                )

                colors = self._sequence_colors(
                    particle.atom_sequence_ids,
                    enabled=stack_spec.color_layers,
                )

                layer_labels = [
                    (
                        f"{layer.label} "
                        f"#{int(layer.realized_index)}"
                    )
                    for layer in particle.realized_layers
                ]

                self.preview.set_scene(
                    lattice_matrix=lattice,
                    atom_cart=particle.atom_cart,
                    atom_elements=particle.atom_elements,
                    atom_colors=colors,
                    layer_origins=particle.layer_origins,
                    show_atoms=state.show_atoms,
                    show_layer_origins=state.show_layer_origins,
                    preview_center=particle.stack_center,
                    shape_spec=shape_spec,
                    show_shape_surface=state.show_shape_surface,
                    shape_opacity=state.shape_opacity,
                    show_shape_axes=state.show_shape_axes,
                    transition_starts=particle.transition_starts,
                    transition_ends=particle.transition_ends,
                    show_transition_vectors=(
                        state.show_transition_vectors
                    ),
                    layer_labels=layer_labels,
                    show_layer_labels=bool(
                        self.chk_show_layer_labels.isChecked()
                    ),
                    layer_label_font_size=int(
                        self.spin_layer_label_font_size.value()
                    ),
                    maximum_layer_labels=int(
                        self.spin_max_layer_labels.value()
                    ),
                    auto_fit=not self._has_fitted_scene,
                )

                # -------------------------------------------------
                # Information panel
                # -------------------------------------------------
                if (
                    stack_spec.model
                    == "correlated_ab_disorder"
                    and stochastic_path_set is not None
                    and stochastic_statistics is not None
                ):
                    state_path_text = " ".join(
                        layer.label
                        for layer in particle.realized_layers
                    )

                    self.label_info.setText(
                        "TOPAS-like correlated A/B stacking preview\n"
                        f"Preview realization: "
                        f"{realization_index + 1}/"
                        f"{stochastic_path_set.ensemble_samples}\n"
                        f"State path: {state_path_text}\n"
                        f"P(alternate): "
                        f"{stack_spec.p_alternate:.6f}\n"
                        f"P(same state): "
                        f"{1.0 - stack_spec.p_alternate:.6f}\n"
                        f"Observed A→A: "
                        f"{stochastic_statistics['A_to_A']}\n"
                        f"Observed A→B: "
                        f"{stochastic_statistics['A_to_B']}\n"
                        f"Observed B→A: "
                        f"{stochastic_statistics['B_to_A']}\n"
                        f"Observed B→B: "
                        f"{stochastic_statistics['B_to_B']}\n"
                        f"Atoms drawn: "
                        f"{particle.atom_cart.shape[0]:,}\n"
                        f"Stack center span: "
                        f"{particle.stack_center_span:.6g} Å\n"
                        f"Nominal boundary thickness: "
                        f"{particle.stack_boundary_thickness:.6g} Å"
                    )

                elif (
                    stack_spec.model
                    == "generic_markov"
                    and stochastic_path_set is not None
                ):
                    state_path_text = " ".join(
                        layer.label
                        for layer in particle.realized_layers
                    )

                    state_index = int(
                        stack_spec.markov_start_state
                    )

                    if (
                        0
                        <= state_index
                        < len(
                            stack_spec.markov_states
                        )
                    ):
                        start_state_label = str(
                            stack_spec.markov_states[
                                state_index
                            ].label
                        )
                    else:
                        start_state_label = f"S{state_index}"

                    self.label_info.setText(
                        "Generic multi-state Markov stacking preview\n"
                        f"Preview realization: "
                        f"{realization_index + 1}/"
                        f"{stochastic_path_set.ensemble_samples}\n"
                        f"State path: {state_path_text}\n"
                        f"Number of states: "
                        f"{len(stack_spec.markov_states)}\n"
                        f"Start state: "
                        f"{start_state_label} "
                        f"(index {state_index})\n"
                        f"Random seed: "
                        f"{int(stack_spec.ensemble_seed)}\n"
                        f"Atoms drawn: "
                        f"{particle.atom_cart.shape[0]:,}\n"
                        f"Stack center span: "
                        f"{particle.stack_center_span:.6g} Å\n"
                        f"Nominal boundary thickness: "
                        f"{particle.stack_boundary_thickness:.6g} Å"
                    )

                else:
                    self.label_info.setText(
                        "Deterministic complete-layer stacking preview\n"
                        f"Sequence rows: "
                        f"{len(stack_spec.sequence):,}\n"
                        f"Requested complete layer positions: "
                        f"{particle.number_of_layer_positions:,}\n"
                        f"Enabled structural layers: "
                        f"{particle.number_of_enabled_layers:,}\n"
                        f"Outer layer-center span: "
                        f"{particle.stack_center_span:.6g} Å\n"
                        f"Nominal boundary thickness: "
                        f"{particle.stack_boundary_thickness:.6g} Å\n"
                        f"Atoms inside lateral envelope before sampling: "
                        f"{particle.n_clipped_atoms:,}\n"
                        f"Atoms drawn: "
                        f"{particle.atom_cart.shape[0]:,}\n"
                        f"Nominal stacking distance: "
                        f"{stack_spec.nominal_spacing:.6g} Å"
                    )

            # -----------------------------------------------------
            # Stacking disabled
            # -----------------------------------------------------
            else:
                atom_cart = np.asarray(
                    self.structure.cart_coords,
                    dtype=np.float32,
                )

                if atom_cart.shape[0]:
                    center = 0.5 * (
                        np.min(
                            atom_cart,
                            axis=0,
                        )
                        + np.max(
                            atom_cart,
                            axis=0,
                        )
                    )

                    atom_cart -= center[
                        None,
                        :
                    ]

                elements = [
                    str(
                        site.species
                    )
                    for site in self.structure.sites
                ]

                self.generated_layer_model.set_layers(
                    []
                )

                self.layer_table_model.set_visible_stats(
                    {}
                )

                self.preview.set_scene(
                    lattice_matrix=lattice,
                    atom_cart=atom_cart,
                    atom_elements=elements,
                    atom_colors=None,
                    layer_origins=np.zeros(
                        (
                            0,
                            3,
                        ),
                        dtype=np.float32,
                    ),
                    show_atoms=state.show_atoms,
                    show_layer_origins=False,
                    preview_center=np.zeros(
                        3,
                        dtype=np.float32,
                    ),
                    shape_spec=shape_spec,
                    show_shape_surface=state.show_shape_surface,
                    shape_opacity=state.shape_opacity,
                    show_shape_axes=state.show_shape_axes,
                    transition_starts=None,
                    transition_ends=None,
                    show_transition_vectors=False,
                    layer_labels=[],
                    show_layer_labels=False,
                    layer_label_font_size=int(
                        self.spin_layer_label_font_size.value()
                    ),
                    maximum_layer_labels=int(
                        self.spin_max_layer_labels.value()
                    ),
                    auto_fit=not self._has_fitted_scene,
                )

                self.label_info.setText(
                    "Periodic stacking is disabled. "
                    "The input structure is shown without layer repetition."
                )

            self._has_fitted_scene = True

        except Exception as exc:
            import traceback

            traceback.print_exc()

            self.label_info.setText(
                "Preview failed:\n"
                f"{exc}"
            )

    def _export_prism_periodic_cif(
        self,
    ) -> None:
        """
        Export the currently displayed prism realization as one periodic CIF.

        The exported CIF contains the actual current preview realization:

            - deterministic displayed stack; or
            - selected stochastic realization; or
            - non-stacked prism.

        The exported cell repeats that displayed realization periodically.
        """
        if self.structure is None:
            QtWidgets.QMessageBox.information(
                self,
                "No structure",
                "Load a structure before exporting a CIF.",
            )
            return

        shape_spec = self._shape_spec_from_ui()

        shape_type = str(
            shape_spec.shape_type
            or ""
        ).strip().lower()

        if shape_type not in {
            "prism",
            "rectangular_prism",
        }:
            QtWidgets.QMessageBox.information(
                self,
                "Prism export only",
                "The periodic realization CIF exporter is available only "
                "for prism shapes.",
            )
            return

        try:
            from .prism_realization_cif import (
                _deterministic_period_vector,
                _nonstacked_period_vector,
                _stochastic_period_vector,
                export_displayed_prism_realization_cif,
            )

            stack_spec = self._stacking_spec_from_ui()

            source_lattice_matrix = np.asarray(
                self.structure.lattice.matrix,
                dtype=float,
            )

            export_kind = "non-stacked prism"

            if stack_spec.enabled:
                if stack_spec.model in {
                    "correlated_ab_disorder",
                    "generic_markov",
                }:
                    stochastic_paths = self._stochastic_preview_paths(
                        stack_spec
                    )

                    realization_index = min(
                        max(
                            0,
                            int(
                                self.spin_preview_realization.value()
                            ),
                        ),
                        max(
                            0,
                            int(
                                stochastic_paths.ensemble_samples
                            )
                            - 1,
                        ),
                    )

                    state_path = stochastic_paths.states[
                        realization_index
                    ]

                    from pdf_fitting.models.stochastic_stacked_particle import (
                        build_stochastic_stacked_particle,
                    )

                    particle = build_stochastic_stacked_particle(
                        structure=self.structure,
                        shape_spec=shape_spec,
                        stack_spec=stack_spec,
                        states=state_path,
                        parameters={},
                        max_atoms=None,
                    )

                    period_vector = _stochastic_period_vector(
                        source_lattice_matrix=(
                            source_lattice_matrix
                        ),
                        states=state_path,
                        stack_spec=stack_spec,
                        parameters={},
                    )

                    export_kind = (
                        "stochastic realization "
                        f"{realization_index + 1}/"
                        f"{stochastic_paths.ensemble_samples}"
                    )

                else:
                    particle = build_stacked_particle(
                        structure=self.structure,
                        shape_spec=shape_spec,
                        stack_spec=stack_spec,
                        max_atoms=None,
                    )

                    period_vector = _deterministic_period_vector(
                        source_lattice_matrix=(
                            source_lattice_matrix
                        ),
                        shape_spec=shape_spec,
                        stack_spec=stack_spec,
                    )

                    export_kind = (
                        "deterministic displayed stack"
                    )

            else:
                from pdf_fitting.models.stacked_particle import (
                    build_stacked_particle,
                )

                nonstacked_spec = LayerStackSpec(
                    enabled=True,
                    model="periodic_sequence",
                    sequence=[
                        SequenceLayer(
                            label="A",
                            shift_u=0.0,
                            shift_v=0.0,
                            stack_coordinate=0.0,
                            normal_delta=0.0,
                        )
                    ],
                    repeat_period=1.0,
                    repeat_shift_u=0.0,
                    repeat_shift_v=0.0,
                    nominal_spacing=float(
                        np.linalg.norm(
                            source_lattice_matrix[
                                2
                            ]
                        )
                    ),
                ).normalized()

                particle = build_stacked_particle(
                    structure=self.structure,
                    shape_spec=shape_spec,
                    stack_spec=nonstacked_spec,
                    max_atoms=None,
                )

                period_vector = _nonstacked_period_vector(
                    source_lattice_matrix=(
                        source_lattice_matrix
                    ),
                    shape_spec=shape_spec,
                )

            if particle.atom_cart.shape[0] <= 0:
                raise RuntimeError(
                    "The displayed prism realization contains no atoms."
                )

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export displayed prism realization as periodic CIF",
                "prism_realization_periodic.cif",
                "CIF files (*.cif);;All files (*)",
            )

            if not path:
                return

            if not path.lower().endswith(
                ".cif"
            ):
                path += ".cif"

            information = export_displayed_prism_realization_cif(
                atom_cart=particle.atom_cart,
                atom_elements=particle.atom_elements,
                source_lattice_matrix=source_lattice_matrix,
                shape_spec=shape_spec,
                period_vector=period_vector,
                path=path,
            )

            QtWidgets.QMessageBox.information(
                self,
                "Periodic realization CIF exported",
                "The currently displayed finite prism realization was "
                "written as one periodic CIF unit cell.\n\n"
                f"Type: {export_kind}\n"
                f"Atoms: {information['number_of_atoms']:,}\n\n"
                f"File:\n{information['path']}\n\n"
                "New lattice:\n"
                f"a = {information['lattice_a']:.8g} Å\n"
                f"b = {information['lattice_b']:.8g} Å\n"
                f"c = {information['lattice_c']:.8g} Å\n"
                f"α = {information['lattice_alpha']:.8g}°\n"
                f"β = {information['lattice_beta']:.8g}°\n"
                f"γ = {information['lattice_gamma']:.8g}°\n\n"
                "The displayed realization is now periodic. "
                "When using the conventional cylinder/disk model on this "
                "new CIF, use the c axis as the cylinder axis:\n\n"
                "cyl_axis_h = 0\n"
                "cyl_axis_k = 0\n"
                "cyl_axis_l = 1",
            )

        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Periodic realization CIF export failed",
                str(
                    exc
                ),
            )
                          

    def _export_stacked_particle_xyz(
        self,
    ) -> None:
        if self.structure is None:
            return

        if not self.chk_stacking_enabled.isChecked():
            QtWidgets.QMessageBox.information(
                self,
                "Stacking disabled",
                "Enable periodic stacking before exporting.",
            )
            return

        try:
            particle = build_stacked_particle(
                structure=self.structure,
                shape_spec=self._shape_spec_from_ui(),
                stack_spec=self._stacking_spec_from_ui(),
                max_atoms=None,
            )

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export periodic stacked particle",
                "periodic_stacked_particle.xyz",
                "XYZ files (*.xyz);;All files (*)",
            )

            if not path:
                return

            if not path.lower().endswith(
                ".xyz"
            ):
                path += ".xyz"

            with open(
                path,
                "w",
                encoding="utf-8",
            ) as file:
                file.write(
                    f"{particle.atom_cart.shape[0]}\n"
                )

                file.write(
                    "Finite AB envelope with complete discrete layers; "
                    f"layer_positions={particle.number_of_layer_positions}; "
                    f"enabled_layers={particle.number_of_enabled_layers}; "
                    f"center_span_A={particle.stack_center_span:.10g}; "
                    f"boundary_thickness_A="
                    f"{particle.stack_boundary_thickness:.10g}\n"
                )

                for (
                    element,
                    position,
                    layer_index,
                    sequence_index,
                ) in zip(
                    particle.atom_elements,
                    particle.atom_cart,
                    particle.atom_layer_indices,
                    particle.atom_sequence_ids,
                ):
                    file.write(
                        f"{str(element):2s} "
                        f"{float(position[0]): .10f} "
                        f"{float(position[1]): .10f} "
                        f"{float(position[2]): .10f} "
                        f"# layer={int(layer_index)} "
                        f"sequence_row={int(sequence_index)}\n"
                    )

            QtWidgets.QMessageBox.information(
                self,
                "XYZ exported",
                f"Exported {particle.atom_cart.shape[0]:,} atoms.\n\n{path}",
            )

        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "XYZ export failed",
                str(exc),
            )

    def _stacking_search_spec_from_ui(
        self,
    ) -> StackingSearchSpec:
        """
        Read all stacking-search controls from the dialog.
        """
        return StackingSearchSpec(
            spacing_treatment=str(
                self.combo_spacing_treatment.currentData()
                or "fixed"
            ),
            spacing_minimum=float(
                self.spin_spacing_search_min.value()
            ),
            spacing_maximum=float(
                self.spin_spacing_search_max.value()
            ),
            spacing_step=float(
                self.spin_spacing_search_step.value()
            ),
            lateral_treatment=str(
                self.combo_lateral_treatment.currentData()
                or "fixed"
            ),
            lateral_target=str(
                self.combo_lateral_target.currentData()
                or "repeat_translation"
            ),
            sequence_row=int(
                self.spin_search_sequence_row.value()
            ),
            scan_u=bool(
                self.chk_search_u.isChecked()
            ),
            u_minimum=float(
                self.spin_u_min.value()
            ),
            u_maximum=float(
                self.spin_u_max.value()
            ),
            u_step=float(
                self.spin_u_step.value()
            ),
            scan_v=bool(
                self.chk_search_v.isChecked()
            ),
            v_minimum=float(
                self.spin_v_min.value()
            ),
            v_maximum=float(
                self.spin_v_max.value()
            ),
            v_step=float(
                self.spin_v_step.value()
            ),
            random_samples=int(
                self.spin_random_samples.value()
            ),
            random_seed=int(
                self.spin_random_seed.value()
            ),
            detail_storage=str(
                self.combo_detail_storage.currentData()
                or "none"
            ),
            detail_directory=str(
                self.edit_detail_directory.text()
                or "stacking_search_results"
            ).strip()
            or "stacking_search_results",
        )


    def _stacking_refinement_state_from_ui(
        self,
    ) -> Dict[str, Any]:
        controls = (
            self.layer_table_model.transition_controls()
        )

        transitions = [
            controls[
                transition_id
            ]
            for transition_id in sorted(
                controls.keys()
            )
        ]

        state = StackingRefinementSpec.from_dict(
            {
                "treatment": (
                    self._current_stacking_treatment()
                ),
                "transitions": transitions,
            }
        )

        self.stacking_refinement_state = state

        return state.to_dict()
    
    def apply_shape(
        self,
    ) -> None:
        """
        Apply the complete finite-crystallite configuration to the main
        input Builder.

        This includes:
            - shape dimensions;
            - crystallite-size search;
            - periodic layer stacking;
            - stacking spacing/u-v search.
        """
        # Commit any currently active cell/spinbox editor.
        focused_widget = QtWidgets.QApplication.focusWidget()

        if focused_widget is not None:
            focused_widget.clearFocus()

        # Commit a table editor explicitly if one is still active.
        if self.layer_table.state() == (
            QtWidgets.QAbstractItemView.EditingState
        ):
            self.layer_table.closePersistentEditor(
                self.layer_table.currentIndex()
            )

            self.layer_table.clearFocus()

        state = self._state_from_ui()

        output = state.to_dict()

        # Explicitly collect all stacking data before emitting the signal.
        # Do not rely only on FiniteCrystalliteState.to_dict(), because the
        # refinement controls are maintained separately by the table model.
        stack_spec = self._stacking_spec_from_ui()

        search_spec = (
            self._stacking_search_spec_from_ui()
        )

        stacking_refinement = (
            self._stacking_refinement_state_from_ui()
        )

        stacking_refinement[
            "stochastic_vectors"
        ] = dict(
            self.stochastic_vector_refinement
        )

        output["mode"] = "finite_shape"
        output["schema_version"] = 3

        output["stacking"] = (
            stack_spec.to_dict()
        )

        output["stacking_search"] = (
            search_spec.to_dict()
        )

        output["stacking_refinement"] = dict(
            stacking_refinement
        )

        # ---------------------------------------------------------
        # Keep legacy aliases synchronized.
        # ---------------------------------------------------------
        output["diameter_cells"] = float(
            state.axis1_cells
        )

        output["height_cells"] = float(
            state.axis3_cells
        )

        output["scan_diameter"] = bool(
            state.scan_axis1
        )

        output["diameter_min_cells"] = float(
            state.axis1_min_cells
        )

        output["diameter_max_cells"] = float(
            state.axis1_max_cells
        )

        output["scan_height"] = bool(
            state.scan_axis3
        )

        output["height_min_cells"] = float(
            state.axis3_min_cells
        )

        output["height_max_cells"] = float(
            state.axis3_max_cells
        )

        output["step_cells"] = float(
            state.axis1_step_cells
        )

        # ---------------------------------------------------------
        # Read the stacking-search controls BEFORE emitting.
        # ---------------------------------------------------------
        search_spec = (
            self._stacking_search_spec_from_ui()
        )

        output["stacking_search"] = (
            search_spec.to_dict()
        )

        print(
            "[FINITE DIALOG APPLY] "
            f"keys={sorted(output.keys())}, "
            f"search_mode={output.get('search_mode')}, "
            f"sequence_rows="
            f"{len(output.get('stacking', {}).get('sequence', []))}, "
            f"stacking_refinement="
            f"{output.get('stacking_refinement')}",
            flush=True,
        )

        # Emit only after every output dictionary is complete.
        self.shapeAccepted.emit(
            output
        )

        self.accept()

    def _update_stacking_search_controls(
        self,
    ) -> None:
        enabled = bool(
            self.chk_stacking_enabled.isChecked()
        )

        stochastic = self._is_stochastic_stacking_model()

        spacing_treatment = str(
            self.combo_spacing_treatment.currentData()
            or "fixed"
        )

        spacing_grid = bool(
            enabled
            and spacing_treatment == "grid"
        )

        spacing_continuous = bool(
            enabled
            and spacing_treatment == "continuous"
        )

        self.spin_spacing_search_min.setEnabled(
            spacing_grid
        )
        self.spin_spacing_search_max.setEnabled(
            spacing_grid
        )
        self.spin_spacing_search_step.setEnabled(
            spacing_grid
        )

        self.chk_refine_nominal_spacing.setChecked(
            spacing_continuous
        )
        self.chk_refine_nominal_spacing.setEnabled(
            False
        )

        lateral_treatment = str(
            self.combo_lateral_treatment.currentData()
            or "fixed"
        )

        lateral_search = bool(
            enabled
            and lateral_treatment in {
                "grid",
                "random",
            }
        )

        regular_grid = bool(
            lateral_search
            and lateral_treatment == "grid"
        )

        random_search = bool(
            lateral_search
            and lateral_treatment == "random"
        )

        for widget in (
            self.combo_lateral_target,
            self.spin_search_sequence_row,
            self.chk_search_u,
            self.chk_search_v,
            self.spin_u_min,
            self.spin_u_max,
            self.spin_v_min,
            self.spin_v_max,
        ):
            widget.setEnabled(
                lateral_search
            )

        self.spin_u_step.setEnabled(
            regular_grid
            and self.chk_search_u.isChecked()
        )
        self.spin_v_step.setEnabled(
            regular_grid
            and self.chk_search_v.isChecked()
        )

        self.spin_random_samples.setEnabled(
            random_search
        )
        self.spin_random_seed.setEnabled(
            random_search
        )

        self.spin_search_sequence_row.setEnabled(
            lateral_search
            and self.combo_lateral_target.currentData()
            == "sequence_row"
        )

        self._update_stacking_candidate_count()


    def _update_stacking_candidate_count(
        self,
    ) -> None:
        try:
            search = self._stacking_search_spec_from_ui()

            counts = stacking_candidate_count(
                search,
                fixed_spacing=float(
                    self.spin_nominal_spacing.value()
                ),
                fixed_u=float(
                    self.spin_repeat_shift_u.value()
                ),
                fixed_v=float(
                    self.spin_repeat_shift_v.value()
                ),
            )

            self.label_stacking_candidate_count.setText(
                "Stacking candidates: "
                f"{counts['total']:,} "
                f"({counts['spacing']:,} spacing × "
                f"{counts['lateral']:,} lateral)"
            )

        except Exception as exc:
            self.label_stacking_candidate_count.setText(
                f"Invalid stacking search: {exc}"
            )