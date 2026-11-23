from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from pdf_fitting.models.layer_stacking import (
    SequenceLayer,
)


_PALETTE = (
    (230, 75, 75),
    (65, 125, 225),
    (65, 180, 95),
    (195, 90, 210),
    (240, 150, 50),
    (45, 180, 180),
    (145, 95, 45),
    (125, 125, 125),
)

class HighPrecisionFloatDelegate(
    QtWidgets.QStyledItemDelegate
):
    """
    High-precision text editor for floating-point stacking values.

    A QLineEdit is used instead of QDoubleSpinBox so narrow table columns do
    not lose most of their width to spin buttons.
    """

    def __init__(
        self,
        parent=None,
        *,
        minimum: float = -1000000.0,
        maximum: float = 1000000.0,
        decimals: int = 12,
        single_step: float = 0.01,
    ):
        super().__init__(
            parent
        )

        self.minimum = float(
            minimum
        )

        self.maximum = float(
            maximum
        )

        self.decimals = int(
            decimals
        )

        # Retained for compatibility with the dialog constructor.
        self.single_step = float(
            single_step
        )

    def createEditor(
        self,
        parent,
        option,
        index,
    ):
        editor = QtWidgets.QLineEdit(
            parent
        )

        validator = QtGui.QDoubleValidator(
            self.minimum,
            self.maximum,
            self.decimals,
            editor,
        )

        validator.setNotation(
            QtGui.QDoubleValidator.ScientificNotation
        )

        validator.setLocale(
            QtCore.QLocale.c()
        )

        editor.setValidator(
            validator
        )

        editor.setAlignment(
            Qt.AlignRight
            | Qt.AlignVCenter
        )

        editor.setFrame(
            False
        )

        editor.setClearButtonEnabled(
            False
        )

        return editor

    def setEditorData(
        self,
        editor,
        index,
    ) -> None:
        value = index.model().data(
            index,
            Qt.EditRole,
        )

        try:
            text = format(
                float(value),
                ".12g",
            )
        except Exception:
            text = "0"

        editor.setText(
            text
        )

        editor.selectAll()

    def setModelData(
        self,
        editor,
        model,
        index,
    ) -> None:
        text = str(
            editor.text()
            or ""
        ).strip()

        if not text:
            return

        locale = QtCore.QLocale.c()

        value, valid = locale.toDouble(
            text
        )

        if not valid:
            try:
                value = float(
                    text
                )
            except ValueError:
                return

        value = max(
            self.minimum,
            min(
                self.maximum,
                float(value),
            ),
        )

        model.setData(
            index,
            value,
            Qt.EditRole,
        )
            
    def updateEditorGeometry(
        self,
        editor,
        option,
        index,
    ) -> None:
        editor.setGeometry(
            option.rect
        )

class LayerPlacementTableModel(
    QtCore.QAbstractTableModel
):
    """
    Editable periodic stacking-transition table.

    Normal rows represent sequence layers. Row zero is the fixed sequence
    origin. Rows 1...N-1 represent transitions from the preceding layer to
    the current layer.

    The final virtual row represents the periodic closing transition from the
    final sequence layer to row zero of the next repeat.
    """

    sequenceEdited = QtCore.Signal()
    transitionControlEdited = QtCore.Signal()
    closureEdited = QtCore.Signal(
        float,
        float,
        float,
    )

    COLUMN_ON = 0
    COLUMN_ROW = 1
    COLUMN_LAYER = 2
    COLUMN_TRANSITION = 3
    COLUMN_DU = 4
    COLUMN_USE_DU = 5
    COLUMN_DV = 6
    COLUMN_USE_DV = 7
    COLUMN_H = 8
    COLUMN_USE_H = 9
    COLUMN_Z = 10
    COLUMN_COPIES = 11
    COLUMN_ATOMS = 12

    HEADERS = (
        "On",
        "Row",
        "Layer",
        "Transition",
        "Δu",
        "R/S",
        "Δv",
        "R/S",
        "h / Å",
        "R/S",
        "z / Å",
        "Copies",
        "Atoms",
    )

    def __init__(
        self,
        parent=None,
        layers: Optional[List[SequenceLayer]] = None,
    ):
        super().__init__(
            parent
        )

        self._layers = list(
            layers
            or []
        )

        self._nominal_spacing = 1.0
        self._vector_mode = "incremental"
        self._treatment = "fixed"

        self._visible_stats: Dict[
            int,
            Tuple[int, int],
        ] = {}

        self._transition_controls: Dict[
            str,
            Dict[str, Dict[str, Any]],
        ] = {}

        self._closure_du = 0.0
        self._closure_dv = 0.0
        self._closure_h = 1.0

    def rowCount(
        self,
        parent=QtCore.QModelIndex(),
    ):
        if parent.isValid():
            return 0

        if not self._layers:
            return 0

        # Sequence rows plus one virtual closing-transition row.
        return len(
            self._layers
        ) + 1

    def columnCount(
        self,
        parent=QtCore.QModelIndex(),
    ):
        if parent.isValid():
            return 0

        return len(
            self.HEADERS
        )

    def headerData(
        self,
        section,
        orientation,
        role=Qt.DisplayRole,
    ):
        if (
            orientation == Qt.Horizontal
            and role == Qt.DisplayRole
            and 0 <= section < len(self.HEADERS)
        ):
            return self.HEADERS[
                section
            ]

        if (
            orientation == Qt.Horizontal
            and role == Qt.ToolTipRole
        ):
            tooltips = (
                "Enable this sequence layer.",
                "Sequence-row index.",
                "Layer label.",
                "Transition represented by this row.",
                "Fractional translation along parent a.",
                "Refine in continuous mode or scan in grid mode.",
                "Fractional translation along parent b.",
                "Refine in continuous mode or scan in grid mode.",
                "Physical normal transition height in Å.",
                "Refine in continuous mode or scan in grid mode.",
                "Cumulative normal position in the sequence period.",
                "Number of visible generated copies.",
                "Number of visible atoms.",
            )

            if 0 <= section < len(tooltips):
                return tooltips[
                    section
                ]

        if (
            role == Qt.DisplayRole
            and orientation == Qt.Vertical
        ):
            return str(
                section + 1
            )

        return None

    def is_closure_row(
        self,
        row: int,
    ) -> bool:
        return bool(
            self._layers
            and int(row) == len(
                self._layers
            )
        )

    def transition_id_at_row(
        self,
        row: int,
    ) -> Optional[str]:
        row = int(
            row
        )

        if self.is_closure_row(
            row
        ):
            return "closure"

        if (
            row <= 0
            or row >= len(
                self._layers
            )
        ):
            return None

        return (
            "layer:"
            + str(
                self._layers[
                    row
                ].uid
            )
        )

    def transition_label_at_row(
        self,
        row: int,
    ) -> str:
        row = int(
            row
        )

        if not self._layers:
            return ""

        if row == 0:
            return "Reference"

        if self.is_closure_row(
            row
        ):
            return (
                f"{self._layers[-1].label} → "
                f"next {self._layers[0].label}"
            )

        if 0 < row < len(
            self._layers
        ):
            return (
                f"{self._layers[row - 1].label} → "
                f"{self._layers[row].label}"
            )

        return ""

    def _default_component_control(
        self,
        component: str,
    ) -> Dict[str, Any]:
        if component == "h":
            return {
                "enabled": False,
                "minimum": 0.001,
                "maximum": 20.0,
                "step": 0.05,
            }

        return {
            "enabled": False,
            "minimum": -0.5,
            "maximum": 0.5,
            "step": 0.05,
        }

    def transition_control_at_row(
        self,
        row: int,
    ) -> Optional[Dict[str, Any]]:
        transition_id = self.transition_id_at_row(
            row
        )

        if transition_id is None:
            return None

        if transition_id not in self._transition_controls:
            self._transition_controls[
                transition_id
            ] = {
                "transition_id": transition_id,
                "label": self.transition_label_at_row(
                    row
                ),
                "u": self._default_component_control(
                    "u"
                ),
                "v": self._default_component_control(
                    "v"
                ),
                "h": self._default_component_control(
                    "h"
                ),
            }

        return self._transition_controls[
            transition_id
        ]

    def transition_values_at_row(
        self,
        row: int,
    ) -> Optional[Dict[str, float]]:
        row = int(
            row
        )

        if self.is_closure_row(
            row
        ):
            return {
                "u": float(
                    self._closure_du
                ),
                "v": float(
                    self._closure_dv
                ),
                "h": float(
                    self._closure_h
                ),
            }

        if (
            row <= 0
            or row >= len(
                self._layers
            )
        ):
            return None

        layer = self._layers[
            row
        ]

        transition_height = (
            float(
                layer.stack_coordinate
            )
            * float(
                self._nominal_spacing
            )
            + float(
                layer.normal_delta
            )
        )

        return {
            "u": float(
                layer.shift_u
            ),
            "v": float(
                layer.shift_v
            ),
            "h": float(
                transition_height
            ),
        }

    def _cumulative_height(
        self,
        row: int,
    ) -> float:
        row = int(
            row
        )

        if row <= 0:
            return 0.0

        maximum_normal_row = min(
            row,
            len(
                self._layers
            )
            - 1,
        )

        cumulative = 0.0

        for layer_row in range(
            1,
            maximum_normal_row + 1,
        ):
            values = self.transition_values_at_row(
                layer_row
            )

            if values is not None:
                cumulative += float(
                    values["h"]
                )

        if self.is_closure_row(
            row
        ):
            cumulative += float(
                self._closure_h
            )

        return float(
            cumulative
        )

    def flags(
        self,
        index,
    ):
        if not index.isValid():
            return Qt.NoItemFlags

        row = int(
            index.row()
        )

        column = int(
            index.column()
        )

        closure_row = self.is_closure_row(
            row
        )

        flags = (
            Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
        )

        if (
            column == self.COLUMN_ON
            and not closure_row
        ):
            flags |= Qt.ItemIsUserCheckable

        if (
            row > 0
            and column in {
                self.COLUMN_DU,
                self.COLUMN_DV,
                self.COLUMN_H,
            }
        ):
            flags |= Qt.ItemIsEditable

        if (
            row > 0
            and self._treatment != "fixed"
            and column in {
                self.COLUMN_USE_DU,
                self.COLUMN_USE_DV,
                self.COLUMN_USE_H,
            }
        ):
            flags |= Qt.ItemIsUserCheckable

        return flags

    def data(
        self,
        index,
        role=Qt.DisplayRole,
    ):
        if not index.isValid():
            return None

        row = int(
            index.row()
        )

        column = int(
            index.column()
        )

        if not (
            0 <= row < self.rowCount()
        ):
            return None

        closure_row = self.is_closure_row(
            row
        )

        if closure_row:
            layer = None
            visible_copies = 0
            visible_atoms = 0
        else:
            layer = self._layers[
                row
            ]

            visible_copies, visible_atoms = (
                self._visible_stats.get(
                    row,
                    (
                        0,
                        0,
                    ),
                )
            )

        if (
            role == Qt.CheckStateRole
            and column == self.COLUMN_ON
            and layer is not None
        ):
            return (
                Qt.Checked
                if layer.enabled
                else Qt.Unchecked
            )

        if (
            role == Qt.CheckStateRole
            and column in {
                self.COLUMN_USE_DU,
                self.COLUMN_USE_DV,
                self.COLUMN_USE_H,
            }
            and row > 0
        ):
            control = self.transition_control_at_row(
                row
            )

            if control is None:
                return Qt.Unchecked

            component = {
                self.COLUMN_USE_DU: "u",
                self.COLUMN_USE_DV: "v",
                self.COLUMN_USE_H: "h",
            }[
                column
            ]

            return (
                Qt.Checked
                if bool(
                    control[
                        component
                    ].get(
                        "enabled",
                        False,
                    )
                )
                else Qt.Unchecked
            )

        if role in {
            Qt.DisplayRole,
            Qt.EditRole,
        }:
            values = self.transition_values_at_row(
                row
            )

            if column == self.COLUMN_ON:
                return None

            if column == self.COLUMN_ROW:
                return (
                    "—"
                    if closure_row
                    else row
                )

            if column == self.COLUMN_LAYER:
                if closure_row:
                    return (
                        "Next "
                        + str(
                            self._layers[
                                0
                            ].label
                        )
                    )

                return str(
                    layer.label
                )

            if column == self.COLUMN_TRANSITION:
                return self.transition_label_at_row(
                    row
                )

            if column == self.COLUMN_DU:
                if row == 0:
                    value = 0.0
                else:
                    value = (
                        values["u"]
                        if values is not None
                        else 0.0
                    )

                return (
                    float(
                        value
                    )
                    if role == Qt.EditRole
                    else f"{float(value):.12g}"
                )

            if column == self.COLUMN_DV:
                if row == 0:
                    value = 0.0
                else:
                    value = (
                        values["v"]
                        if values is not None
                        else 0.0
                    )

                return (
                    float(
                        value
                    )
                    if role == Qt.EditRole
                    else f"{float(value):.12g}"
                )

            if column == self.COLUMN_H:
                if row == 0:
                    value = 0.0
                else:
                    value = (
                        values["h"]
                        if values is not None
                        else 0.0
                    )

                return (
                    float(
                        value
                    )
                    if role == Qt.EditRole
                    else f"{float(value):.12g}"
                )

            if column in {
                self.COLUMN_USE_DU,
                self.COLUMN_USE_DV,
                self.COLUMN_USE_H,
            }:
                return None

            if column == self.COLUMN_Z:
                return f"{self._cumulative_height(row):.12g}"

            if column == self.COLUMN_COPIES:
                return int(
                    visible_copies
                )

            if column == self.COLUMN_ATOMS:
                return int(
                    visible_atoms
                )

        if role == Qt.TextAlignmentRole:
            if column in {
                self.COLUMN_LAYER,
                self.COLUMN_TRANSITION,
            }:
                return int(
                    Qt.AlignLeft
                    | Qt.AlignVCenter
                )

            return int(
                Qt.AlignRight
                | Qt.AlignVCenter
            )

        if role == Qt.BackgroundRole:
            if closure_row:
                return QtGui.QBrush(
                    QtGui.QColor(
                        220,
                        220,
                        220,
                    )
                )

            red, green, blue = _PALETTE[
                row % len(
                    _PALETTE
                )
            ]

            if not layer.enabled:
                return QtGui.QBrush(
                    QtGui.QColor(
                        220,
                        220,
                        220,
                    )
                )

            alpha = (
                105
                if visible_atoms > 0
                else 35
            )

            return QtGui.QBrush(
                QtGui.QColor(
                    red,
                    green,
                    blue,
                    alpha,
                )
            )

        if (
            role == Qt.FontRole
            and (
                closure_row
                or visible_atoms > 0
            )
        ):
            font = QtGui.QFont()
            font.setBold(
                True
            )

            if closure_row:
                font.setItalic(
                    True
                )

            return font

        return None

    def setData(
        self,
        index,
        value,
        role=Qt.EditRole,
    ):
        if not index.isValid():
            return False

        row = int(
            index.row()
        )

        column = int(
            index.column()
        )

        closure_row = self.is_closure_row(
            row
        )

        if (
            role == Qt.CheckStateRole
            and column == self.COLUMN_ON
            and not closure_row
        ):
            layer = self._layers[
                row
            ]

            checked = bool(
                value == Qt.Checked
                or int(
                    value
                )
                == int(
                    Qt.Checked
                )
            )

            layer.enabled = checked

            self.dataChanged.emit(
                index,
                index,
                [
                    Qt.CheckStateRole,
                    Qt.BackgroundRole,
                ],
            )

            self.sequenceEdited.emit()

            return True

        if (
            role == Qt.CheckStateRole
            and column in {
                self.COLUMN_USE_DU,
                self.COLUMN_USE_DV,
                self.COLUMN_USE_H,
            }
            and row > 0
        ):
            control = self.transition_control_at_row(
                row
            )

            if control is None:
                return False

            component = {
                self.COLUMN_USE_DU: "u",
                self.COLUMN_USE_DV: "v",
                self.COLUMN_USE_H: "h",
            }[
                column
            ]

            control[
                component
            ][
                "enabled"
            ] = bool(
                value == Qt.Checked
                or int(
                    value
                )
                == int(
                    Qt.Checked
                )
            )

            self.dataChanged.emit(
                index,
                index,
                [
                    Qt.CheckStateRole,
                ],
            )

            self.transitionControlEdited.emit()

            return True

        if role != Qt.EditRole:
            return False

        try:
            number = float(
                value
            )
        except Exception:
            return False

        if row <= 0:
            return False

        if closure_row:
            if column == self.COLUMN_DU:
                self._closure_du = number

            elif column == self.COLUMN_DV:
                self._closure_dv = number

            elif column == self.COLUMN_H:
                if number <= 0.0:
                    return False

                self._closure_h = number

            else:
                return False

            self.dataChanged.emit(
                self.index(
                    row,
                    self.COLUMN_DU,
                ),
                self.index(
                    row,
                    self.COLUMN_Z,
                ),
                [
                    Qt.DisplayRole,
                    Qt.EditRole,
                ],
            )

            self.closureEdited.emit(
                float(
                    self._closure_du
                ),
                float(
                    self._closure_dv
                ),
                float(
                    self._closure_h
                ),
            )

            self.sequenceEdited.emit()

            return True

        layer = self._layers[
            row
        ]

        if column == self.COLUMN_DU:
            layer.shift_u = number

        elif column == self.COLUMN_DV:
            layer.shift_v = number

        elif column == self.COLUMN_H:
            if number <= 0.0:
                return False

            layer.normal_delta = (
                number
                - float(
                    layer.stack_coordinate
                )
                * float(
                    self._nominal_spacing
                )
            )

        else:
            return False

        self.dataChanged.emit(
            self.index(
                row,
                self.COLUMN_DU,
            ),
            self.index(
                self.rowCount() - 1,
                self.COLUMN_Z,
            ),
            [
                Qt.DisplayRole,
                Qt.EditRole,
            ],
        )

        self.sequenceEdited.emit()

        return True

    def set_treatment(
        self,
        treatment: str,
    ) -> None:
        treatment = str(
            treatment
            or "fixed"
        ).strip().lower()

        if treatment not in {
            "fixed",
            "continuous",
            "grid",
        }:
            treatment = "fixed"

        if treatment == self._treatment:
            return

        self._treatment = treatment

        if self.rowCount() > 0:
            self.dataChanged.emit(
                self.index(
                    0,
                    self.COLUMN_USE_DU,
                ),
                self.index(
                    self.rowCount() - 1,
                    self.COLUMN_USE_H,
                ),
                [
                    Qt.CheckStateRole,
                ],
            )

        # Refinement/scan columns change their item flags when the treatment
        # changes, so the view must also refresh the model layout.
        self.layoutChanged.emit()

    def treatment(
        self,
    ) -> str:
        return str(
            self._treatment
        )

    def set_transition_controls(
        self,
        controls: Dict[str, Any],
    ) -> None:
        self._transition_controls = {
            str(
                key
            ): dict(
                value
            )
            for key, value in (
                controls
                or {}
            ).items()
            if isinstance(
                value,
                dict,
            )
        }

        if self.rowCount() > 0:
            self.dataChanged.emit(
                self.index(
                    0,
                    self.COLUMN_USE_DU,
                ),
                self.index(
                    self.rowCount() - 1,
                    self.COLUMN_USE_H,
                ),
                [
                    Qt.CheckStateRole,
                ],
            )

    def transition_controls(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        # Ensure controls exist for all current transition rows.
        for row in range(
            1,
            self.rowCount()
        ):
            self.transition_control_at_row(
                row
            )

        return {
            key: {
                "transition_id": str(
                    value.get(
                        "transition_id",
                        key,
                    )
                ),
                "label": str(
                    value.get(
                        "label",
                        "",
                    )
                ),
                "u": dict(
                    value.get(
                        "u",
                        {},
                    )
                ),
                "v": dict(
                    value.get(
                        "v",
                        {},
                    )
                ),
                "h": dict(
                    value.get(
                        "h",
                        {},
                    )
                ),
            }
            for key, value in self._transition_controls.items()
        }

    def set_closure_values(
        self,
        *,
        du: float,
        dv: float,
        h: float,
    ) -> None:
        self._closure_du = float(
            du
        )

        self._closure_dv = float(
            dv
        )

        self._closure_h = max(
            float(
                h
            ),
            1.0e-12,
        )

        if self._layers:
            row = len(
                self._layers
            )

            self.dataChanged.emit(
                self.index(
                    row,
                    self.COLUMN_DU,
                ),
                self.index(
                    row,
                    self.COLUMN_Z,
                ),
                [
                    Qt.DisplayRole,
                    Qt.EditRole,
                ],
            )

    def set_vector_mode(
        self,
        vector_mode: str,
    ) -> None:
        value = str(
            vector_mode
            or "incremental"
        ).strip().lower()

        if value not in {
            "incremental",
            "absolute",
        }:
            value = "incremental"

        self._vector_mode = value

    def set_nominal_spacing(
        self,
        value: float,
    ) -> None:
        self._nominal_spacing = max(
            float(
                value
            ),
            1.0e-12,
        )

        if self.rowCount() > 0:
            self.dataChanged.emit(
                self.index(
                    0,
                    self.COLUMN_H,
                ),
                self.index(
                    self.rowCount() - 1,
                    self.COLUMN_Z,
                ),
                [
                    Qt.DisplayRole,
                ],
            )

    def set_visible_stats(
        self,
        stats: Dict[
            int,
            Tuple[int, int],
        ],
    ) -> None:
        new_stats = dict(
            stats
            or {}
        )

        if new_stats == self._visible_stats:
            return

        self._visible_stats = new_stats

        if self._layers:
            self.dataChanged.emit(
                self.index(
                    0,
                    self.COLUMN_COPIES,
                ),
                self.index(
                    len(
                        self._layers
                    )
                    - 1,
                    self.COLUMN_ATOMS,
                ),
                [
                    Qt.DisplayRole,
                ],
            )

    def set_layers(
        self,
        layers: List[SequenceLayer],
    ) -> None:
        self.beginResetModel()

        self._layers = [
            SequenceLayer.from_dict(
                layer.to_dict()
                if isinstance(
                    layer,
                    SequenceLayer,
                )
                else layer,
                default_coordinate=float(
                    row
                ),
                nominal_spacing=self._nominal_spacing,
            )
            for row, layer in enumerate(
                layers
                or []
            )
        ]

        self._visible_stats = {}

        valid_transition_ids = {
            "closure"
        }

        valid_transition_ids.update(
            "layer:"
            + str(
                layer.uid
            )
            for layer in self._layers[
                1:
            ]
        )

        self._transition_controls = {
            key: value
            for key, value in self._transition_controls.items()
            if key in valid_transition_ids
        }

        self.endResetModel()

    def layers(
        self,
    ) -> List[SequenceLayer]:
        return [
            SequenceLayer.from_dict(
                layer.to_dict(),
                default_coordinate=float(
                    row
                ),
                nominal_spacing=self._nominal_spacing,
            )
            for row, layer in enumerate(
                self._layers
            )
        ]

    def layer_at(
        self,
        row: int,
    ) -> Optional[SequenceLayer]:
        if not (
            0 <= row < len(
                self._layers
            )
        ):
            return None

        return self._layers[
            row
        ]